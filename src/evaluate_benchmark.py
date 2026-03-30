"""
evaluate_benchmark.py
=====================
VQA v2 test predictions  +  COCO Captions test predictions.
Converted from the original Colab notebooks — output format, prompts,
batch logic, resume support, and file formats are identical.

--model_id accepts EITHER:
  - A local adapter path  : outputs/run1/final_adapter
  - A HuggingFace Hub id  : lamao-ab/paligemma-blind-assist-jetson-ready

Usage
-----
# VQA v2 only
python src/evaluate_benchmark.py \
    --model_id   lamao-ab/paligemma-blind-assist-jetson-ready \
    --task       vqa \
    --output_dir outputs/predictions

# COCO Captions only
python src/evaluate_benchmark.py \
    --model_id   lamao-ab/paligemma-blind-assist-jetson-ready \
    --task       caps \
    --output_dir outputs/predictions

# Both tasks back-to-back (model loaded only once)
python src/evaluate_benchmark.py \
    --model_id   lamao-ab/paligemma-blind-assist-jetson-ready \
    --task       both \
    --output_dir outputs/predictions

# Use a local adapter instead of a Hub model
python src/evaluate_benchmark.py \
    --model_id   outputs/run1/final_adapter \
    --task       both \
    --output_dir outputs/predictions
"""

import argparse
import gc
import json
import os
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


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(model_id: str, base_model_id: str):
    """
    Load model from either:
      - A local adapter path  (contains adapter_config.json)
      - A HuggingFace Hub id  (merged or adapter repo)
    """
    print("\n📦 Loading Model...")
    start_time = time.time()

    # 1. Load Processor
    # For a local adapter the processor was saved alongside the adapter;
    # for a Hub model the processor is in the same repo.
    processor_source = model_id
    processor = PaliGemmaProcessor.from_pretrained(processor_source)
    # FIX THE PADDING SIDE - THIS IS CRITICAL!
    processor.tokenizer.padding_side = "left"
    print(f"✅ Tokenizer padding side set to: {processor.tokenizer.padding_side}")

    # 2. Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 3. Detect whether model_id is a local PEFT adapter or a full/Hub model
    is_local_adapter = (
        os.path.isdir(model_id)
        and os.path.exists(os.path.join(model_id, "adapter_config.json"))
    )

    if is_local_adapter:
        from peft import PeftModel
        print(f"   Detected local PEFT adapter → loading base + adapter")
        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            base_model_id,
            device_map="auto",
            quantization_config=bnb_config,
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(base_model, model_id)
        print(f"   ✅ Adapter loaded from {model_id}")
    else:
        # Hub model (merged) or Hub adapter repo
        try:
            from peft import PeftModel
            # Try loading as a Hub PEFT adapter on top of base_model_id
            base_model = PaliGemmaForConditionalGeneration.from_pretrained(
                base_model_id,
                device_map="auto",
                quantization_config=bnb_config,
                attn_implementation="sdpa",
            )
            model = PeftModel.from_pretrained(base_model, model_id)
            print(f"   ✅ Hub PEFT adapter loaded from {model_id}")
        except Exception:
            # Fall back: load as a fully merged Hub model
            model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=bnb_config,
                attn_implementation="sdpa",
            )
            print(f"   ✅ Merged Hub model loaded from {model_id}")

    model.eval()

    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.1f}s")
    print(f"   ✅ Model loaded on {next(model.parameters()).device}")
    print(f"   💾 Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    return model, processor


# ── Shared download helper ────────────────────────────────────────────────────

def download_and_extract(url: str, dest_folder: str) -> None:
    filename  = url.split("/")[-1]
    file_path = os.path.join(dest_folder, filename)

    if not os.path.exists(file_path):
        print(f"⬇️  Downloading {filename}...")
        response   = requests.get(url, stream=True, timeout=180)
        total_size = int(response.headers.get("content-length", 0))
        with open(file_path, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc=filename
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    else:
        print(f"✅ {filename} already exists.")

    print(f"📦 Extracting {filename}...")
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(dest_folder)


# ── Shared batch predict ──────────────────────────────────────────────────────

def batch_predict(model, processor, images, prompts, max_tokens):
    """Batch prediction"""
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
        return [pred.strip() for pred in predictions]

    except Exception as e:
        print(f"\n⚠️  Batch error: {e}")
        return ["error" for _ in prompts]


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 1 — VQA v2
# ══════════════════════════════════════════════════════════════════════════════

VQA_IMAGES_URL    = "http://images.cocodataset.org/zips/test2015.zip"
VQA_QUESTIONS_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip"


def run_vqa(model, processor, args):
    print("\n" + "=" * 70)
    print("🚀 VQA v2 PREDICTION — QLORA (4-BIT QUANTIZATION)")
    print("=" * 70)
    print(f"📂 Model     : {args.model_id}")
    print(f"📊 Batch size: {args.vqa_batch_size}")
    print(f"📏 Max tokens: {args.max_tokens}")

    # ── Data setup ────────────────────────────────────────────────────────────
    data_root = os.path.join(args.output_dir, "vqav2_data")
    os.makedirs(data_root, exist_ok=True)

    print("\n🏗️  Setting up VQA v2 Data...")
    download_and_extract(VQA_IMAGES_URL,    data_root)
    download_and_extract(VQA_QUESTIONS_URL, data_root)

    IMAGES_DIR          = os.path.join(data_root, "test2015")
    TEST_QUESTIONS_PATH = os.path.join(
        data_root, "v2_OpenEnded_mscoco_test2015_questions.json"
    )

    if not os.path.exists(IMAGES_DIR):
        print(f"⚠️  Images directory not found at {IMAGES_DIR}")
    if not os.path.exists(TEST_QUESTIONS_PATH):
        print(f"⚠️  Questions file not found at {TEST_QUESTIONS_PATH}")

    # ── Load questions ────────────────────────────────────────────────────────
    print(f"\n📋 Loading official test questions from {TEST_QUESTIONS_PATH}...")
    with open(TEST_QUESTIONS_PATH) as f:
        test_questions = json.load(f)

    questions_list = test_questions["questions"]
    print(f"   ✅ Loaded {len(questions_list):,} test questions.")
    print(f"   Dataset Info: {test_questions.get('info', 'N/A')}")
    print(f"   Sample Question: {questions_list[0]}")

    # ── Inference ─────────────────────────────────────────────────────────────
    output_file = os.path.join(args.output_dir, "vqav2_test_predictions.json")
    print(f"\n🚀 Starting Predictions on {len(questions_list):,} questions...")

    results             = []
    batch_images        = []
    batch_prompts       = []
    batch_question_ids  = []

    for idx, item in enumerate(tqdm(questions_list)):
        question_id    = item["question_id"]
        image_id       = item["image_id"]
        question_text  = item["question"]

        # VQA v2 uses COCO format: COCO_test2015_000000xxxxxx.jpg
        image_filename = f"COCO_test2015_{image_id:012d}.jpg"
        image_path     = os.path.join(IMAGES_DIR, image_filename)

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"⚠️  Missing image: {image_filename} (question_id: {question_id})")
            image = Image.new("RGB", (224, 224), color="black")

        prompt = f"<image>{question_text}"

        batch_images.append(image)
        batch_prompts.append(prompt)
        batch_question_ids.append(question_id)

        if len(batch_images) == args.vqa_batch_size or idx == len(questions_list) - 1:
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

            # Save Results in VQA v2 format
            for q_id, ans in zip(batch_question_ids, generated):
                results.append({"question_id": q_id, "answer": ans.strip()})

            batch_images       = []
            batch_prompts      = []
            batch_question_ids = []

    # ── Save ──────────────────────────────────────────────────────────────────
    # Sort by question_id for consistency
    results.sort(key=lambda x: x["question_id"])

    print(f"\n💾 Saving {len(results):,} predictions to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("✅ DONE! File is ready for VQA v2 EvalAI submission.")
    print(f"\n📊 Submission Format:")
    print(f"   - Total predictions: {len(results):,}")
    print(f"   - Format: [{{'question_id': int, 'answer': str}}, ...]")
    print(f"\n🔗 Submit at: https://eval.ai/web/challenges/challenge-page/830/overview")
    return output_file


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 2 — COCO Captions
# ══════════════════════════════════════════════════════════════════════════════

COCO_IMAGES_URL = "http://images.cocodataset.org/zips/test2014.zip"
COCO_INFO_URL   = "http://images.cocodataset.org/annotations/image_info_test2014.zip"


def run_caps(model, processor, args):
    print("\n" + "=" * 70)
    print("🚀 COCO CAPTIONS TEST PREDICTIONS — QLORA (4-BIT QUANTIZATION)")
    print("=" * 70)
    print(f"📂 Model     : {args.model_id}")
    print(f"📊 Batch size: {args.caps_batch_size}")
    print(f"📏 Max tokens: {args.max_tokens}")

    # ── Data setup ────────────────────────────────────────────────────────────
    data_root = os.path.join(args.output_dir, "coco_data")
    os.makedirs(data_root, exist_ok=True)

    print("\n🏗️  Setting up COCO Captions Data...")
    download_and_extract(COCO_IMAGES_URL, data_root)
    download_and_extract(COCO_INFO_URL,   data_root)

    IMAGES_DIR     = os.path.join(data_root, "test2014")
    TEST_INFO_PATH = os.path.join(data_root, "annotations", "image_info_test2014.json")

    # ── Load image info ───────────────────────────────────────────────────────
    print(f"\n📋 Loading test image info from {TEST_INFO_PATH}...")
    with open(TEST_INFO_PATH) as f:
        test_info = json.load(f)

    test_images = test_info["images"]
    print(f"   ✅ Loaded {len(test_images):,} test images")
    print(f"   Sample: {test_images[0]}")

    # ── Resume support ────────────────────────────────────────────────────────
    temp_output_file  = os.path.join(args.output_dir, "coco_caption_temp_results.jsonl")
    final_output_file = os.path.join(args.output_dir, "coco_caption_test_predictions.json")

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

    # ── Filter unprocessed ────────────────────────────────────────────────────
    images_to_process = [
        img for img in test_images
        if img["id"] not in processed_image_ids
    ]

    if not images_to_process:
        print("\n🎉 All images already processed!")
    else:
        print(f"   🔮 Remaining: {len(images_to_process):,} images")
        print(f"\n🚀 Generating captions...")
        print(f"   Expected time: ~{len(images_to_process) * 2 / 60:.0f} minutes")

        batch_images    = []
        batch_prompts   = []
        batch_image_ids = []
        predictions_count = 0
        errors_count      = 0
        start_time        = time.time()

        with open(temp_output_file, "a") as f_out:
            for item in tqdm(images_to_process, desc="COCO Caption Test"):
                try:
                    image_id  = item["id"]
                    filename  = item["file_name"]
                    image_path = os.path.join(IMAGES_DIR, filename)

                    try:
                        image = Image.open(image_path).convert("RGB")
                    except FileNotFoundError:
                        print(f"⚠️  Missing image: {filename}")
                        errors_count += 1
                        continue

                    prompt = "<image>caption en"

                    batch_images.append(image)
                    batch_prompts.append(prompt)
                    batch_image_ids.append(image_id)

                    if (len(batch_images) == args.caps_batch_size
                            or item == images_to_process[-1]):

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
                    print(f"\n⚠️  Error processing image {item.get('id', 'unknown')}: {e}")
                    errors_count += 1
                    continue

        elapsed_time = time.time() - start_time
        print(f"\n✅ Generation complete!")
        print(f"   Predictions : {predictions_count:,}")
        print(f"   Errors      : {errors_count}")
        print(f"   Time        : {elapsed_time/60:.1f} minutes")

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

        # Remove duplicates (keep last occurrence)
        seen: dict = {}
        for item in final_data:
            seen[item["image_id"]] = item
        final_data = list(seen.values())

        # Sort by image_id for consistency
        final_data.sort(key=lambda x: x["image_id"])

        with open(final_output_file, "w") as f:
            json.dump(final_data, f, indent=2)

        print(f"   ✅ Saved: {final_output_file}")
        print(f"   Total captions: {len(final_data):,}")

        print(f"\n📝 Sample captions:")
        for i, sample in enumerate(final_data[:3], 1):
            print(f"      {i}. Image ID : {sample['image_id']}")
            print(f"         Caption  : {sample['caption'][:80]}...")

        if final_data:
            lengths = [len(item["caption"].split()) for item in final_data]
            print(f"\n📊 Caption statistics:")
            print(f"   Avg length : {sum(lengths)/len(lengths):.1f} words")
            print(f"   Min        : {min(lengths)} words")
            print(f"   Max        : {max(lengths)} words")
        else:
            print("\n⚠️  No captions generated yet.")

    print("\n" + "=" * 70)
    print("✅ COCO CAPTIONS TEST PREDICTIONS COMPLETE!")
    print("=" * 70)
    print(f"\n🚀 Ready for EvalAI submission:")
    print(f"   File   : {final_output_file}")
    print(f"   Format : [{{'image_id': int, 'caption': 'text'}}]")
    print(f"\n🔗 Submit at: https://competitions.codalab.org/competitions/3221")
    print("=" * 70)
    return final_output_file


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VQA v2 + COCO Captions test predictions"
    )
    parser.add_argument(
        "--model_id", required=True,
        help=(
            "Either a local adapter path (outputs/run1/final_adapter) "
            "or a HuggingFace Hub model id (lamao-ab/paligemma-blind-assist-jetson-ready)"
        ),
    )
    parser.add_argument("--task",           choices=["vqa", "caps", "both"], default="both")
    parser.add_argument("--base_model_id",  default="google/paligemma-3b-mix-224",
                        help="Base model id (only used when model_id is a local adapter)")
    parser.add_argument("--output_dir",     default="outputs/predictions")
    parser.add_argument("--max_tokens",     type=int, default=64)
    parser.add_argument("--vqa_batch_size", type=int, default=128,
                        help="Batch size for VQA v2 inference")
    parser.add_argument("--caps_batch_size", type=int, default=128,
                        help="Batch size for COCO captions inference")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model once — shared across both tasks
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
