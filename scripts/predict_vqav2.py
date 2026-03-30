"""
predict_vqav2.py
================
Generate VQA v2 test predictions for EvalAI submission.

Output: JSON array  →  [{"question_id": int, "answer": str}, ...]

Usage
-----
python scripts/predict_vqav2.py \
    --model_id   lamao-ab/paligemma-blind-assist-jetson-ready \
    --output     outputs/vqav2_predictions.json \
    --batch_size 128
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

VQAV2_QUESTIONS_URL = (
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/"
    "v2_Questions_Test_mscoco.zip"
)
VQAV2_IMAGES_URL = "http://images.cocodataset.org/zips/test2015.zip"
DEFAULT_MODEL    = "lamao-ab/paligemma-blind-assist-jetson-ready"


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
def batch_generate(model, processor, images, prompts, max_new_tokens=10):
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
    parser = argparse.ArgumentParser(description="VQA v2 test predictions")
    parser.add_argument("--model_id",       default=DEFAULT_MODEL)
    parser.add_argument("--output",         default="outputs/vqav2_predictions.json")
    parser.add_argument("--batch_size",     type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--workdir",        default="data/vqav2_test")
    args = parser.parse_args()

    workdir = Path(args.workdir)

    q_zip  = download_file(VQAV2_QUESTIONS_URL, workdir / "questions.zip")
    i_zip  = download_file(VQAV2_IMAGES_URL,    workdir / "test2015.zip")
    q_dir  = workdir / "questions"
    i_dir  = workdir / "test2015"
    extract_zip(q_zip, q_dir)
    extract_zip(i_zip, i_dir)

    q_file = next(q_dir.rglob("*.json"))
    with open(q_file) as f:
        questions_data = json.load(f)
    questions = questions_data["questions"]

    img_index = {p.name: p for p in i_dir.rglob("*.jpg")}

    print(f"Loading model: {args.model_id}")
    model, processor = load_model(args.model_id)

    predictions = []
    batch_imgs, batch_prompts, batch_qids = [], [], []

    def flush():
        answers = batch_generate(
            model, processor, batch_imgs, batch_prompts, args.max_new_tokens
        )
        for qid, answer in zip(batch_qids, answers):
            predictions.append({"question_id": qid, "answer": answer})
        batch_imgs.clear(); batch_prompts.clear(); batch_qids.clear()

    for q in tqdm(questions, desc="Generating VQA v2 predictions"):
        # COCO test2015 filename: COCO_test2015_000000XXXXXX.jpg
        img_name = f"COCO_test2015_{q['image_id']:012d}.jpg"
        img_path = img_index.get(img_name)
        if img_path is None:
            continue
        batch_imgs.append(PILImage.open(img_path).convert("RGB"))
        batch_prompts.append(f"<image>Answer concisely: {q['question']}")
        batch_qids.append(q["question_id"])
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
