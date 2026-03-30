"""
package_for_deployment.py
=========================
Merge a QLoRA adapter into the base model weights, re-quantize to 4-bit NF4,
and push the packaged model to the Hugging Face Hub.

Steps
-----
1. Load base model in BF16 + PEFT adapter → merge_and_unload → save temp BF16
2. Free GPU memory
3. Reload merged weights with 4-bit NF4 quantization
4. Save final package (model + processor) locally
5. Upload to Hub

Usage
-----
python src/package_for_deployment.py \
    --adapter_path outputs/run1/final_adapter \
    --hub_repo_id  your-username/paligemma-blind-assist
"""

import argparse
import gc
import os
import shutil
import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import (
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


# ── Steps ─────────────────────────────────────────────────────────────────────

def step1_merge_to_bf16(
    adapter_path: str,
    base_id: str,
    tmp_bf16: str,
) -> None:
    """Load base (BF16) + adapter, merge, save to *tmp_bf16*."""
    print("[1/4] Loading base model in BF16 …")
    base = PaliGemmaForConditionalGeneration.from_pretrained(
        base_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("[1/4] Applying PEFT adapter …")
    model = PeftModel.from_pretrained(base, adapter_path)

    print("[1/4] Merging and unloading adapter weights …")
    model = model.merge_and_unload()
    model.save_pretrained(tmp_bf16)

    processor = PaliGemmaProcessor.from_pretrained(adapter_path)
    processor.save_pretrained(tmp_bf16)

    del model, base
    free_memory()
    print(f"  ✓ BF16 merged model saved to {tmp_bf16}")


def step2_quantize_to_nf4(tmp_bf16: str, package_dir: str) -> None:
    """Reload merged BF16 weights with NF4 quantization, save package."""
    print("[2/4] Reloading merged model with 4-bit NF4 quantization …")
    bnb = build_bnb_config()
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        tmp_bf16,
        quantization_config=bnb,
        device_map="auto",
    )
    processor = PaliGemmaProcessor.from_pretrained(tmp_bf16)

    print(f"[3/4] Saving quantized package to {package_dir} …")
    os.makedirs(package_dir, exist_ok=True)
    model.save_pretrained(package_dir)
    processor.save_pretrained(package_dir)

    del model
    free_memory()
    print("  ✓ Quantized package saved.")


def step3_write_model_card(package_dir: str, hub_repo_id: str, base_id: str) -> None:
    card = f"""---
license: apache-2.0
base_model: {base_id}
tags:
  - paligemma
  - qlora
  - blind-assistance
  - vizwiz
  - vqa
  - image-captioning
---

# PaliGemma Blind Assist — Jetson Ready

Fine-tuned **{base_id}** with 4-bit QLoRA on [VizWiz](https://vizwiz.org/) for
blind assistance tasks (VQA + image captioning).

## Results

| Benchmark | Metric | Score |
|---|---|---|
| VizWiz VQA | Accuracy | 76.23 % |
| VizWiz Captions | CIDEr | 0.7373 |
| COCO Captions | CIDEr | 0.9477 |
| COCO Captions | BLEU-4 | 0.3100 |

## Quick Start

```python
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor, BitsAndBytesConfig
import torch
from PIL import Image

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=torch.bfloat16)

processor = PaliGemmaProcessor.from_pretrained("{hub_repo_id}")
model = PaliGemmaForConditionalGeneration.from_pretrained(
    "{hub_repo_id}", quantization_config=bnb, device_map="auto"
)

image = Image.open("photo.jpg").convert("RGB")
inputs = processor(images=[image], text=["<image>What is in front of me?"],
                   return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=64)
print(processor.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True))
```
"""
    with open(os.path.join(package_dir, "README.md"), "w") as f:
        f.write(card)


def step4_push_to_hub(package_dir: str, hub_repo_id: str) -> None:
    print(f"[4/4] Uploading to Hub: {hub_repo_id} …")
    api = HfApi()
    api.create_repo(repo_id=hub_repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=package_dir,
        repo_id=hub_repo_id,
        repo_type="model",
        commit_message="Upload 4-bit NF4 packaged model",
    )
    print(f"  ✓ Model pushed to https://huggingface.co/{hub_repo_id}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Merge adapter & push to HF Hub")
    parser.add_argument("--adapter_path", required=True,
                        help="Path to saved PEFT adapter (final_adapter/)")
    parser.add_argument("--hub_repo_id",  required=True,
                        help="HF Hub repository id, e.g. username/model-name")
    parser.add_argument("--base_model_id", default="google/paligemma-3b-mix-224")
    parser.add_argument("--tmp_merged",    default=None,
                        help="Temp dir for BF16 merged model (auto if not set)")
    parser.add_argument("--tmp_package",   default=None,
                        help="Dir for final NF4 package (auto if not set)")
    parser.add_argument("--no_push",       action="store_true",
                        help="Skip Hub upload (useful for local testing)")
    args = parser.parse_args()

    use_tmp_merged  = args.tmp_merged  is None
    use_tmp_package = args.tmp_package is None

    tmp_merged_dir  = args.tmp_merged  or tempfile.mkdtemp(prefix="paligemma_bf16_")
    tmp_package_dir = args.tmp_package or tempfile.mkdtemp(prefix="paligemma_nf4_")

    try:
        step1_merge_to_bf16(args.adapter_path, args.base_model_id, tmp_merged_dir)
        step2_quantize_to_nf4(tmp_merged_dir, tmp_package_dir)
        step3_write_model_card(tmp_package_dir, args.hub_repo_id, args.base_model_id)

        if not args.no_push:
            step4_push_to_hub(tmp_package_dir, args.hub_repo_id)
        else:
            print(f"[skip] Hub upload skipped. Package at: {tmp_package_dir}")

    finally:
        if use_tmp_merged and Path(tmp_merged_dir).exists():
            shutil.rmtree(tmp_merged_dir, ignore_errors=True)
        if use_tmp_package and args.no_push and Path(tmp_package_dir).exists():
            pass  # keep for inspection when --no_push is set

    print("\n✅ Packaging complete.")


if __name__ == "__main__":
    main()
