"""
inference_demo.py
=================
Quick single-image or batch-folder inference demo.

Usage
-----
# Single image with a question
python scripts/inference_demo.py \
    --image    photo.jpg \
    --question "What is in front of me?"

# Single image — both VQA and captioning
python scripts/inference_demo.py --image photo.jpg

# Batch folder inference
python scripts/inference_demo.py \
    --image_dir  path/to/images/ \
    --question   "What do you see?"

# Use a PEFT adapter instead of a merged model
python scripts/inference_demo.py \
    --model_id     outputs/run1/final_adapter \
    --base_id      google/paligemma-3b-mix-224 \
    --image        photo.jpg
"""

import argparse
import os
import time
from pathlib import Path

import torch
from PIL import Image as PILImage
from transformers import (
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)

DEFAULT_MODEL = "lamao-ab/paligemma-blind-assist-jetson-ready"
IMAGE_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(model_id: str, base_id: str | None):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"

    try:
        from peft import PeftModel
        base = PaliGemmaForConditionalGeneration.from_pretrained(
            base_id or "google/paligemma-3b-mix-224",
            quantization_config=bnb,
            attn_implementation="sdpa",
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, model_id)
        print(f"Loaded PEFT adapter from {model_id}")
    except Exception:
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb,
            attn_implementation="sdpa",
            device_map="auto",
        )
        print(f"Loaded merged model from {model_id}")

    model.eval()
    return model, processor


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.inference_mode()
def run_inference(
    model,
    processor,
    image: PILImage.Image,
    prompt: str,
    max_new_tokens: int = 64,
) -> tuple[str, float]:
    start = time.perf_counter()
    inputs = processor(
        images=[image],
        text=[prompt],
        return_tensors="pt",
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    latency = (time.perf_counter() - start) * 1000  # ms

    answer = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    return answer, latency


def vram_stats() -> str:
    if not torch.cuda.is_available():
        return "N/A (no CUDA)"
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved  = torch.cuda.memory_reserved()  / 1e9
    return f"{allocated:.2f} GB allocated / {reserved:.2f} GB reserved"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PaliGemma blind-assist inference demo")
    parser.add_argument("--model_id",       default=DEFAULT_MODEL)
    parser.add_argument("--base_id",        default=None)
    parser.add_argument("--image",          default=None,
                        help="Path to a single image file")
    parser.add_argument("--image_dir",      default=None,
                        help="Directory of images for batch demo")
    parser.add_argument("--question",       default=None,
                        help="Question for VQA (omit to run both VQA + caption)")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    if args.image is None and args.image_dir is None:
        parser.error("Provide --image or --image_dir")

    print(f"\nLoading model …")
    model, processor = load_model(args.model_id, args.base_id)
    print(f"VRAM after load: {vram_stats()}\n")

    # Collect images
    if args.image_dir:
        image_paths = [
            p for p in Path(args.image_dir).iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        ]
    else:
        image_paths = [Path(args.image)]

    for img_path in image_paths:
        image = PILImage.open(img_path).convert("RGB")
        print(f"── {img_path.name} {'─' * (50 - len(img_path.name))}")

        if args.question:
            prompts = [(f"<image>Assist a blind person: {args.question}", "VQA")]
        else:
            prompts = [
                ("<image>Assist a blind person: What do you see?",      "VQA"),
                ("<image>Describe this scene for a blind person.",       "Caption"),
            ]

        for prompt, label in prompts:
            answer, ms = run_inference(
                model, processor, image, prompt, args.max_new_tokens
            )
            print(f"  [{label}] {answer}  ({ms:.0f} ms)")

    print(f"\nVRAM after inference: {vram_stats()}")


if __name__ == "__main__":
    main()
