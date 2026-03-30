"""
predict.py
============
Run VQA and/or captioning inference on a local folder of images.

Accepts EITHER a local LoRA adapter OR a HuggingFace Hub model ID —
auto-detected from the path (same logic as evaluate_vizwiz.py).

Usage
-----
# Hub model, both tasks, default image folder
python src/predict.py \
    --model_id   lamao-ab/paligemma-blind-assist-jetson-ready \
    --image_dir  /path/to/images

# Local adapter, VQA only, custom output
python src/predict.py \
    --model_id   outputs/run/final_adapter \
    --task       vqa \
    --image_dir  /path/to/images \
    --output     results/my_results.json \
    --batch_size 4 \
    --max_tokens 64

# Hub model, captioning only
python src/inference.py \
    --model_id   lamao-ab/paligemma-blind-assist-jetson-ready \
    --task       caps \
    --image_dir  /path/to/images
"""

import argparse
import gc
import glob
import json
import os
import time

import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)
from peft import PeftModel


# ── Prompts ───────────────────────────────────────────────────────────────────

VQA_PROMPT = "Assist a blind person: List all the objects you see in this image."
CAP_PROMPT = "Describe this scene for a blind person."


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_memory_stats():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved  = torch.cuda.memory_reserved()  / 1024**3
    return allocated, reserved


def is_local_adapter(model_id: str) -> bool:
    return os.path.isdir(model_id)


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(model_id: str, base_model_id: str):
    """
    Load model from either:
      - a local LoRA adapter directory  → base_model + PeftModel
      - a HuggingFace Hub model ID      → full model directly
    """
    print("\n📦 Loading model...")

    q_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_quant_storage=torch.uint8,
    )

    start = time.time()

    if is_local_adapter(model_id):
        print(f"🔌 Detected LOCAL adapter : {model_id}")
        print(f"   Base model             : {base_model_id}")

        processor = PaliGemmaProcessor.from_pretrained(base_model_id)
        processor.tokenizer.padding_side = "left"

        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            base_model_id,
            quantization_config=q_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(base_model, model_id)

    else:
        print(f"🤗 Detected HUB model: {model_id}")

        processor = PaliGemmaProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=q_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

    model.eval()

    elapsed = time.time() - start
    alloc, res = get_memory_stats()
    print(f"✅ Model loaded in {elapsed:.1f}s")
    print(f"   Padding side : {processor.tokenizer.padding_side}")
    print(f"   VRAM         : {alloc:.2f} GB allocated / {res:.2f} GB reserved")

    return model, processor


# ── Image discovery ───────────────────────────────────────────────────────────

def find_images(image_dir: str) -> list:
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    return sorted(set(paths))


# ── Batch inference ───────────────────────────────────────────────────────────

def run_batch(model, processor, images: list, prompt: str, max_tokens: int) -> list:
    """Run a single prompt over a batch of PIL images, return decoded strings."""
    prompts = [f"<image>{prompt}" for _ in images]

    inputs = processor(
        text=prompts,
        images=images,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        decoded = processor.batch_decode(
            outputs[:, input_len:], skip_special_tokens=True
        )

    del inputs, outputs
    return [d.strip() for d in decoded]


# ── Main inference loop ───────────────────────────────────────────────────────

def run_inference(args):
    # ── Discover images ───────────────────────────────────────────────────────
    image_paths = find_images(args.image_dir)
    if not image_paths:
        print(f"⚠️  No images found in {args.image_dir}. Check your path.")
        return

    print(f"\n📁 Found {len(image_paths)} images in {args.image_dir}")

    # ── Load model ────────────────────────────────────────────────────────────
    model, processor = load_model(args.model_id, args.base_model_id)

    # ── Warmup ────────────────────────────────────────────────────────────────
    print("\n🔥 Running warmup pass...")
    dummy = [Image.new("RGB", (224, 224))]
    run_batch(model, processor, dummy, "warmup", max_tokens=4)
    torch.cuda.empty_cache()
    print("   ✅ Warmup done")

    # ── Inference ─────────────────────────────────────────────────────────────
    do_vqa  = args.task in ("vqa",  "both")
    do_caps = args.task in ("caps", "both")

    print("\n" + "=" * 60)
    print(f"🎯 STARTING INFERENCE")
    print(f"   Task        : {args.task}")
    print(f"   Images      : {len(image_paths)}")
    print(f"   Batch size  : {args.batch_size}")
    print(f"   Max tokens  : {args.max_tokens}")
    print("=" * 60)

    all_results = []
    total_time  = 0.0
    n_batches   = (len(image_paths) + args.batch_size - 1) // args.batch_size

    for i in tqdm(range(0, len(image_paths), args.batch_size),
                  total=n_batches, desc="Batches"):

        batch_paths = image_paths[i : i + args.batch_size]

        # Load images
        loaded, filenames = [], []
        for p in batch_paths:
            try:
                loaded.append(Image.open(p).convert("RGB"))
                filenames.append(os.path.basename(p))
            except Exception as e:
                print(f"\n⚠️  Could not load {p}: {e}")

        if not loaded:
            continue

        batch_result = {fn: {"image": fn} for fn in filenames}

        # VQA pass
        if do_vqa:
            t0 = time.time()
            vqa_preds = run_batch(model, processor, loaded, VQA_PROMPT, args.max_tokens)
            total_time += time.time() - t0
            for fn, pred in zip(filenames, vqa_preds):
                batch_result[fn]["vqa_answer"] = pred

        # Caption pass
        if do_caps:
            t0 = time.time()
            cap_preds = run_batch(model, processor, loaded, CAP_PROMPT, args.max_tokens)
            total_time += time.time() - t0
            for fn, pred in zip(filenames, cap_preds):
                batch_result[fn]["caption"] = pred

        # Print batch results
        for fn, res in batch_result.items():
            print(f"\n  [{fn}]")
            if "vqa_answer" in res:
                print(f"    VQA : {res['vqa_answer']}")
            if "caption" in res:
                print(f"    CAP : {res['caption']}")

        all_results.extend(batch_result.values())

        torch.cuda.empty_cache()
        gc.collect()

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(all_results)
    alloc, res = get_memory_stats()

    print("\n" + "=" * 60)
    print("✅ INFERENCE COMPLETE")
    print(f"   Images processed         : {n}")
    print(f"   Total inference time     : {total_time:.2f}s")
    print(f"   Avg time per image       : {total_time / max(n, 1):.2f}s")
    print(f"   Final VRAM allocated     : {alloc:.2f} GB")
    print(f"   Final VRAM reserved      : {res:.2f} GB")
    print("=" * 60)

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n💾 Results saved to: {args.output}")
    print(f"   Format: [{{'image': 'file.jpg', 'vqa_answer': '...', 'caption': '...'}}]")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Blind-assist inference: VQA and/or captioning on a folder of images"
    )
    parser.add_argument(
        "--model_id", required=True,
        help=(
            "Local adapter path  (e.g. outputs/run/final_adapter)  OR  "
            "HuggingFace Hub ID  (e.g. lamao-ab/paligemma-blind-assist-jetson-ready)"
        ),
    )
    parser.add_argument(
        "--task", choices=["vqa", "caps", "both"], default="both",
        help="Which task(s) to run (default: both)",
    )
    parser.add_argument(
        "--image_dir", default="/content/images",
        help="Directory containing images (.jpg / .jpeg / .png)",
    )
    parser.add_argument(
        "--output", default="outputs/inference_results.json",
        help="Path for the output JSON file",
    )
    parser.add_argument(
        "--base_model_id", default="google/paligemma-3b-mix-224",
        help="Base model ID — only used when --model_id is a local adapter",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=64)

    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
