"""
predict_vizwiz_vqa.py
=====================
Generate VizWiz VQA test predictions in the EvalAI submission format.

Output: JSON array  →  [{"image": filename, "answer": str}, ...]

Usage
-----
python scripts/predict_vizwiz_vqa.py \
    --model_id   lamao-ab/paligemma-blind-assist-jetson-ready \
    --output     outputs/vizwiz_vqa_predictions.json \
    --batch_size 32
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

VIZWIZ_TEST_IMAGES = "https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip"
VIZWIZ_TEST_ANN    = "https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_ann/test.json"
DEFAULT_MODEL      = "lamao-ab/paligemma-blind-assist-jetson-ready"


def download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    print(f"Downloading {url} …")
    r = requests.get(url, stream=True, timeout=120)
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
def batch_generate(model, processor, images, prompts, max_new_tokens=20):
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
    parser = argparse.ArgumentParser(description="VizWiz VQA test predictions")
    parser.add_argument("--model_id",   default=DEFAULT_MODEL)
    parser.add_argument("--output",     default="outputs/vizwiz_vqa_predictions.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workdir",    default="data/vizwiz_test")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    args = parser.parse_args()

    workdir = Path(args.workdir)
    img_zip  = download_file(VIZWIZ_TEST_IMAGES, workdir / "test.zip")
    ann_path = download_file(VIZWIZ_TEST_ANN,    workdir / "test.json")
    img_dir  = workdir / "test"
    extract_zip(img_zip, img_dir)

    with open(ann_path) as f:
        annotations = json.load(f)

    img_index = {p.name: p for p in img_dir.rglob("*.jpg")}

    print(f"Loading model: {args.model_id}")
    model, processor = load_model(args.model_id)

    predictions = []
    batch_imgs, batch_prompts, batch_names = [], [], []

    def flush():
        answers = batch_generate(
            model, processor, batch_imgs, batch_prompts, args.max_new_tokens
        )
        for name, answer in zip(batch_names, answers):
            predictions.append({"image": name, "answer": answer})
        batch_imgs.clear(); batch_prompts.clear(); batch_names.clear()

    for ann in tqdm(annotations, desc="Generating VQA predictions"):
        img_name = ann["image"]
        img_path = img_index.get(img_name)
        if img_path is None:
            continue
        batch_imgs.append(PILImage.open(img_path).convert("RGB"))
        batch_prompts.append(f"<image>Assist a blind person: {ann['question']}")
        batch_names.append(img_name)
        if len(batch_imgs) == args.batch_size:
            flush()
    if batch_imgs:
        flush()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"\n✅ {len(predictions)} predictions saved to {args.output}")


if __name__ == "__main__":
    main()
