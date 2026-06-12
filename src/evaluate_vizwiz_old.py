"""
evaluate_vizwiz.py
==================
VizWiz VQA test predictions  +  VizWiz Captions test predictions.

The --model_id argument accepts EITHER:
  • A local adapter path  (e.g. outputs/run/final_adapter)
  • A HuggingFace Hub ID  (e.g. lamao-ab/paligemma-blind-assist-jetson-ready)

The script auto-detects which one you passed and loads accordingly.

Usage
-----
# Option A: freshly trained local adapter
python src/evaluate_vizwiz.py \
    --model_id   outputs/run/final_adapter \
    --task       both \
    --output_dir outputs/predictions

# Option B: pre-trained model from the Hub
python src/evaluate_vizwiz.py \
    --model_id   lamao-ab/paligemma-blind-assist-jetson-ready \
    --task       both \
    --output_dir outputs/predictions
"""

import argparse
import gc
import io
import json
import os
import re
import time
import zipfile

import requests
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)
from peft import PeftModel


# ── Model loader (shared) ─────────────────────────────────────────────────────

def is_local_adapter(model_id: str) -> bool:
    """Return True if model_id points to a local adapter directory."""
    return os.path.isdir(model_id)


def load_model(model_id: str, base_model_id: str):
    """
    Load model from either:
      - a local LoRA adapter directory  → base_model + PeftModel
      - a HuggingFace Hub model ID      → full model directly
    """
    print("\n📦 Loading Model...")
    start = time.time()

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_quant_storage=torch.uint8,
    )

    if is_local_adapter(model_id):
        # ── Local adapter: load base model then attach adapter ────────────────
        print(f"🔌 Detected LOCAL adapter: {model_id}")
        print(f"   Base model            : {base_model_id}")

        processor = PaliGemmaProcessor.from_pretrained(base_model_id)
        processor.tokenizer.padding_side = "left"

        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            base_model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(base_model, model_id)

    else:
        # ── HuggingFace Hub: load processor + full model directly ─────────────
        print(f"🤗 Detected HUB model: {model_id}")

        processor = PaliGemmaProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

    model.eval()

    elapsed = time.time() - start
    print(f"✅ Tokenizer padding side set to: {processor.tokenizer.padding_side}")
    print(f"✅ Model loaded in {elapsed:.1f}s")
    print(f"💾 Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    return model, processor


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 1 — VizWiz VQA
# ══════════════════════════════════════════════════════════════════════════════

VQA_IMAGES_URL      = "https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip"
VQA_ANNOTATIONS_URL = "https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip"


def download_and_extract(url: str, dest_folder: str) -> None:
    filename  = url.split("/")[-1]
    file_path = os.path.join(dest_folder, filename)

    if not os.path.exists(file_path):
        print(f"⬇️  Downloading {filename}...")
        response = requests.get(url, stream=True, timeout=120)
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"✅ {filename} already exists.")

    print(f"📦 Extracting {filename}...")
    with zipfile.ZipFile(file_path, "r") as zf:
        zf.extractall(dest_folder)


def find_json(folder: str, name: str) -> str:
    """Recursively find *name*.json under *folder*."""
    for root, _, files in os.walk(folder):
        for fname in files:
            if fname == f"{name}.json":
                return os.path.join(root, fname)
    all_json = [f for r, _, fs in os.walk(folder) for f in fs if f.endswith(".json")]
    raise FileNotFoundError(
        f"Could not find '{name}.json' under {folder}. "
        f"Available: {all_json}"
    )


def run_vqa(model, processor, args):
    print("\n" + "=" * 70)
    print("🚀 VQA PREDICTION — VizWiz Test Set")
    print("=" * 70)
    print(f"📂 Model     : {args.model_id}")
    print(f"📊 Batch size: {args.batch_size}")
    print(f"📏 Max tokens: {args.max_tokens}")

    # ── Data setup ────────────────────────────────────────────────────────────
    data_root = os.path.join(args.output_dir, "vizwiz_vqa_data")
    os.makedirs(data_root, exist_ok=True)

    print("\n🏗️  Setting up Data...")
    download_and_extract(VQA_IMAGES_URL,      data_root)
    download_and_extract(VQA_ANNOTATIONS_URL, data_root)

    images_dir     = os.path.join(data_root, "test")
    test_json_path = find_json(data_root, "test")

    print(f"\n📋 Loading official test list from {test_json_path}...")
    with open(test_json_path) as f:
        test_data = json.load(f)

    if isinstance(test_data, dict):
        test_data = test_data.get("data", test_data.get("annotations", []))

    print(f"   ✅ Loaded {len(test_data):,} test items.")
    if test_data:
        print(f"   Sample Item: {test_data[0]}")

    # ── Inference ─────────────────────────────────────────────────────────────
    output_file = os.path.join(args.output_dir, "vizwiz_vqa_test_predictions.json")
    print(f"\n🚀 Starting Predictions on {len(test_data):,} images...")

    results       = []
    batch_images  = []
    batch_prompts = []
    batch_ids     = []

    for idx, item in enumerate(tqdm(test_data)):
        image_filename = item["image"]
        question       = item["question"]
        image_path     = os.path.join(images_dir, image_filename)

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"⚠️  Missing image: {image_filename}")
            image = Image.new("RGB", (224, 224), color="black")

        prompt = f"<image>Assist a blind person: {question}"
        batch_images.append(image)
        batch_prompts.append(prompt)
        batch_ids.append(image_filename)

        if len(batch_images) == args.batch_size or idx == len(test_data) - 1:
            inputs = processor(
                text=batch_prompts,
                images=batch_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(model.device)

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    do_sample=False,
                )
                generated = processor.batch_decode(
                    outputs[:, input_len:], skip_special_tokens=True
                )

            for img_id, ans in zip(batch_ids, generated):
                results.append({"image": img_id, "answer": ans.strip()})

            batch_images  = []
            batch_prompts = []
            batch_ids     = []

    # ── Save ──────────────────────────────────────────────────────────────────
    results.sort(key=lambda x: x["image"])

    print(f"\n💾 Saving {len(results):,} predictions to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("✅ DONE! VQA file is ready for EvalAI submission.")
    print(f"   File: {output_file}")
    return output_file


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 2 — VizWiz Captions
# ══════════════════════════════════════════════════════════════════════════════

def load_image(img_data) -> Image.Image:
    if isinstance(img_data, dict):
        if "bytes" in img_data and img_data["bytes"]:
            return Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
        if "path" in img_data and img_data["path"]:
            return Image.open(img_data["path"]).convert("RGB")
    return img_data.convert("RGB")


def extract_image_id(item: dict, index: int) -> int:
    """Extract image ID as integer — mirrors notebook logic exactly."""
    img_id = item.get("image_id")
    if img_id is not None:
        try:
            return int(img_id)
        except (ValueError, TypeError):
            pass

    img_id = item.get("id")
    if img_id is not None:
        try:
            return int(img_id)
        except (ValueError, TypeError):
            pass

    filename = item.get("image")
    if filename and isinstance(filename, str):
        match = re.search(r"(\d+)", str(filename))
        if match:
            return int(match.group(1))

    return index


def batch_predict(model, processor, images, prompts, max_tokens):
    try:
        inputs = processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
            clean_ids   = outputs[:, input_len:]
            predictions = processor.batch_decode(clean_ids, skip_special_tokens=True)

        del inputs, outputs, clean_ids
        return [p.strip() for p in predictions]

    except Exception as e:
        print(f"\n⚠️  Batch error: {e}")
        return ["error" for _ in prompts]


def run_caps(model, processor, args):
    from datasets import load_dataset

    print("\n" + "=" * 70)
    print("🚀 CAPTION TEST PREDICTIONS — VizWiz Caps")
    print("=" * 70)
    print(f"📂 Model     : {args.model_id}")
    print(f"📊 Batch size: {args.batch_size}")
    print(f"📏 Max tokens: {args.max_tokens}")

    temp_output_file  = os.path.join(args.output_dir, "vizwiz_caption_temp_results.jsonl")
    final_output_file = os.path.join(args.output_dir, "vizwiz_caption_test_predictions.json")

    # ── Resume support ────────────────────────────────────────────────────────
    processed_image_ids: set = set()
    if os.path.exists(temp_output_file):
        print(f"\n🔄 Found existing progress file")
        with open(temp_output_file) as f:
            for line in f:
                try:
                    processed_image_ids.add(json.loads(line)["image_id"])
                except Exception:
                    continue
        print(f"   ⏩ Already processed: {len(processed_image_ids):,} images")
    else:
        print(f"\n📝 Starting fresh")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("\n📥 Loading VizWiz-Caps test set...")
    caption_test = load_dataset("lmms-lab/VizWiz-Caps", split="test")
    print(f"   ✅ Loaded {len(caption_test):,} samples")

    indices_to_process = [
        i for i, item in enumerate(caption_test)
        if extract_image_id(item, i) not in processed_image_ids
    ]

    if not indices_to_process:
        print("\n🎉 All images already processed!")
    else:
        print(f"   🔮 Remaining: {len(indices_to_process):,} images")
        caption_subset = caption_test.select(indices_to_process)

        print(f"\n🚀 Generating captions...")
        print(f"   Expected time: ~{len(indices_to_process) * 2 / 60:.0f} minutes")

        batch_images    = []
        batch_prompts   = []
        batch_image_ids = []
        predictions_count = 0
        errors_count      = 0
        start_time        = time.time()

        with open(temp_output_file, "a") as f_out:
            for idx, item in enumerate(tqdm(caption_subset, desc="Caption Test")):
                try:
                    image  = load_image(item["image"])
                    prompt = "<image>Describe this scene for a blind person."
                    original_index = indices_to_process[idx]
                    image_id = extract_image_id(item, original_index)

                    batch_images.append(image)
                    batch_prompts.append(prompt)
                    batch_image_ids.append(image_id)

                    if len(batch_images) == args.batch_size or idx == len(caption_subset) - 1:
                        preds = batch_predict(
                            model, processor,
                            batch_images, batch_prompts, args.max_tokens,
                        )

                        for img_id, pred in zip(batch_image_ids, preds):
                            if pred != "error":
                                record = {"image_id": img_id, "caption": pred}
                                f_out.write(json.dumps(record) + "\n")
                                predictions_count += 1
                            else:
                                errors_count += 1

                        f_out.flush()
                        batch_images    = []
                        batch_prompts   = []
                        batch_image_ids = []

                        if predictions_count % 100 == 0:
                            gc.collect()
                            torch.cuda.empty_cache()

                except Exception as e:
                    print(f"\n⚠️  Error at index {idx}: {e}")
                    errors_count += 1
                    continue

        elapsed = time.time() - start_time
        print(f"\n✅ Generation complete!")
        print(f"   Predictions : {predictions_count:,}")
        print(f"   Errors      : {errors_count}")
        print(f"   Time        : {elapsed/60:.1f} minutes")

    # ── Convert temp JSONL → final JSON ───────────────────────────────────────
    print("\n💾 Creating final JSON for EvalAI...")
    final_data: list = []
    if os.path.exists(temp_output_file):
        with open(temp_output_file) as f:
            for line in f:
                try:
                    final_data.append(json.loads(line))
                except Exception:
                    pass

        seen: dict = {}
        for item in final_data:
            seen[item["image_id"]] = item
        final_data = list(seen.values())

        with open(final_output_file, "w") as f:
            json.dump(final_data, f, indent=2)

        print(f"   ✅ Saved: {final_output_file}")
        print(f"   Total captions: {len(final_data):,}")

        print(f"\n📝 Sample captions:")
        for i, sample in enumerate(final_data[:3], 1):
            print(f"      {i}. Image ID : {sample['image_id']}")
            print(f"         Caption  : {sample['caption'][:80]}...")

        lengths = [len(item["caption"].split()) for item in final_data]
        print(f"\n📊 Caption statistics:")
        print(f"   Avg length : {sum(lengths)/len(lengths):.1f} words")
        print(f"   Min        : {min(lengths)} words")
        print(f"   Max        : {max(lengths)} words")

    print("\n" + "=" * 70)
    print("✅ CAPTION TEST PREDICTIONS COMPLETE!")
    print("=" * 70)
    print(f"\n🚀 Ready for EvalAI submission:")
    print(f"   File   : {final_output_file}")
    print(f"   Format : [{{'image_id': 23431, 'caption': 'text'}}]")
    print("=" * 70)
    return final_output_file


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VizWiz VQA + Caption test predictions"
    )
    parser.add_argument(
        "--model_id", required=True,
        help=(
            "Local adapter path (e.g. outputs/run/final_adapter) "
            "OR HuggingFace Hub model ID (e.g. lamao-ab/paligemma-blind-assist-jetson-ready)"
        ),
    )
    parser.add_argument("--task",          choices=["vqa", "caps", "both"], default="both")
    parser.add_argument("--base_model_id", default="google/paligemma-3b-mix-224",
                        help="Base model ID — only used when --model_id is a local adapter")
    parser.add_argument("--output_dir",    default="outputs/predictions")
    parser.add_argument("--batch_size",    type=int, default=32)
    parser.add_argument("--max_tokens",    type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model once, reuse for both tasks
    model, processor = load_model(args.model_id, args.base_model_id)

    if args.task in ("vqa", "both"):
        run_vqa(model, processor, args)

    if args.task in ("caps", "both"):
        run_caps(model, processor, args)

    print("\n" + "=" * 70)
    print("✅ ALL TASKS COMPLETE")
    print("=" * 70)
    print(f"\n📂 All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
