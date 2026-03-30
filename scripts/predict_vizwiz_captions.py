"""
predict_vizwiz_captions.py
==========================
Generate VizWiz Captions test predictions for EvalAI submission.

Output: JSON array  →  [{"image_id": int, "caption": str}, ...]

Usage
-----
python scripts/predict_vizwiz_captions.py \
    --model_id   lamao-ab/paligemma-blind-assist-jetson-ready \
    --output     outputs/vizwiz_caps_predictions.json \
    --batch_size 32
"""

import argparse
import io
import json
import os

import torch
from datasets import load_dataset
from PIL import Image as PILImage
from tqdm import tqdm
from transformers import (
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)

DEFAULT_MODEL = "lamao-ab/paligemma-blind-assist-jetson-ready"
PROMPT        = "<image>Describe this scene for a blind person."


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


def open_hf_image(img_field) -> PILImage.Image:
    if isinstance(img_field, PILImage.Image):
        return img_field.convert("RGB")
    if isinstance(img_field, dict) and img_field.get("bytes"):
        return PILImage.open(io.BytesIO(img_field["bytes"])).convert("RGB")
    raise ValueError(f"Cannot open image: {type(img_field)}")


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
    parser = argparse.ArgumentParser(description="VizWiz Captions test predictions")
    parser.add_argument("--model_id",       default=DEFAULT_MODEL)
    parser.add_argument("--output",         default="outputs/vizwiz_caps_predictions.json")
    parser.add_argument("--batch_size",     type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--split",          default="test",
                        help="HF dataset split to use (test / validation)")
    args = parser.parse_args()

    print("Loading VizWiz-Caps dataset …")
    ds = load_dataset("lmms-lab/VizWiz-Caps", split=args.split)

    print(f"Loading model: {args.model_id}")
    model, processor = load_model(args.model_id)

    predictions = []
    # Resume support: temp file
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

    for idx, row in enumerate(tqdm(ds, desc="Generating captions")):
        if idx in done_ids:
            continue
        img = open_hf_image(row["image"])
        batch_imgs.append(img)
        batch_ids.append(idx)
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
