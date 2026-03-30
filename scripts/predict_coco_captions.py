"""
predict_coco_captions.py
========================
Generate COCO Captions test predictions for EvalAI submission.

Output: JSON array  →  [{"image_id": int, "caption": str}, ...]

Usage
-----
python scripts/predict_coco_captions.py \
    --model_id   lamao-ab/paligemma-blind-assist-jetson-ready \
    --output     outputs/coco_caps_predictions.json \
    --batch_size 64
"""

import argparse
import json
import os
import zipfile
from pathlib import Path

import requests
import torch
from PIL import Image as PILImage
from tqdm import tqdm
from transformers import (
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)

COCO_TEST_IMAGES   = "http://images.cocodataset.org/zips/test2014.zip"
COCO_TEST_INFO     = "http://images.cocodataset.org/annotations/image_info_test2014.zip"
DEFAULT_MODEL      = "lamao-ab/paligemma-blind-assist-jetson-ready"
PROMPT             = "<image>caption en"


def download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    print(f"Downloading {url} …")
    r = requests.get(url, stream=True, timeout=180)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    return dest


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name} …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def load_model(model_id: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, quantization_config=bnb, attn_implementation="sdpa", device_map="auto"
    )
    model.eval()
    return model, processor


@torch.inference_mode()
def batch_generate(model, processor, images, max_new_tokens=64):
    prompts = [PROMPT] * len(images)
    inputs = processor(
        images=images, text=prompts, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return [
        processor.decode(o[input_len:], skip_special_tokens=True).strip()
        for o in outputs
    ]


def main():
    parser = argparse.ArgumentParser(description="COCO Captions test predictions")
    parser.add_argument("--model_id",       default=DEFAULT_MODEL)
    parser.add_argument("--output",         default="outputs/coco_caps_predictions.json")
    parser.add_argument("--batch_size",     type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--workdir",        default="data/coco_test")
    args = parser.parse_args()

    workdir = Path(args.workdir)

    img_zip  = download_file(COCO_TEST_IMAGES, workdir / "test2014.zip")
    info_zip = download_file(COCO_TEST_INFO,   workdir / "image_info_test2014.zip")
    img_dir  = workdir / "test2014"
    info_dir = workdir / "info"
    extract_zip(img_zip,  img_dir)
    extract_zip(info_zip, info_dir)

    info_path = next(info_dir.rglob("*.json"))
    with open(info_path) as f:
        info_data = json.load(f)
    image_records = info_data["images"]

    img_index = {p.name: p for p in img_dir.rglob("*.jpg")}

    print(f"Loading model: {args.model_id}")
    model, processor = load_model(args.model_id)

    predictions = []
    tmp_path = args.output + ".tmp.jsonl"
    done_ids: set[int] = set()
    if os.path.exists(tmp_path):
        with open(tmp_path) as f:
            for line in f:
                row = json.loads(line)
                predictions.append(row)
                done_ids.add(row["image_id"])
        print(f"Resuming from {len(predictions)} existing predictions …")

    tmp_file = open(tmp_path, "a")
    batch_imgs, batch_ids = [], []

    def flush():
        captions = batch_generate(model, processor, batch_imgs, args.max_new_tokens)
        for img_id, caption in zip(batch_ids, captions):
            row = {"image_id": img_id, "caption": caption}
            predictions.append(row)
            tmp_file.write(json.dumps(row) + "\n")
            tmp_file.flush()
        batch_imgs.clear(); batch_ids.clear()

    for record in tqdm(image_records, desc="Generating COCO captions"):
        img_id   = record["id"]
        img_name = record["file_name"]
        if img_id in done_ids:
            continue
        img_path = img_index.get(img_name)
        if img_path is None:
            continue
        batch_imgs.append(PILImage.open(img_path).convert("RGB"))
        batch_ids.append(img_id)
        if len(batch_imgs) == args.batch_size:
            flush()
    if batch_imgs:
        flush()

    tmp_file.close()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)
    os.remove(tmp_path)

    print(f"\n✅ {len(predictions)} captions saved to {args.output}")


if __name__ == "__main__":
    main()
