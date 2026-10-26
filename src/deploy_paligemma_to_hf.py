# =============================================================================
# Deploy PaliGemma Blind-Assist models to Hugging Face
#   • LoRA  -> standalone bf16 merged checkpoint
#   • QLoRA -> genuine 4-bit NF4 checkpoint (for Jetson Orin Nano edge inference)
#
# HOW TO RUN (Google Colab):
#   1. Runtime -> Change runtime type -> GPU   (4-bit quantization needs CUDA)
#   2. Run "CELL 1" (install + mount Drive)
#   3. Paste your HF WRITE token into login() in "CELL 2", then run CELL 2
#
# Make sure the HF account you log in with has accepted the Gemma license at
# https://huggingface.co/google/paligemma-3b-mix-224 (it's a gated repo).
# =============================================================================


# ============================== CELL 1: setup ================================
# !pip install -q -U transformers peft huggingface_hub accelerate safetensors bitsandbytes
#
# from google.colab import drive
# drive.mount('/content/drive')


# ========================= CELL 2: deploy both models =======================
import os, shutil, gc, torch
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login, whoami, hf_hub_download, HfApi

# ---------- AUTH (paste a WRITE token from the account with PaliGemma access) ----------
login(token="hf_xxxxxxxxxxxxxxxxxxxxx")
print("Logged in as:", whoami()["name"])
hf_hub_download("google/paligemma-3b-mix-224", "config.json")  # fails fast if gating not accepted
print("✅ Base model access OK")

# ---------- SHARED CONFIG ----------
BASE = {"base_model_id": "google/paligemma-3b-mix-224"}
TEMP_ROOT = "/tmp/merged_upload"
PRIVATE = False

LORA_TRAINING = """\
- **Base:** `google/paligemma-3b-mix-224` (bfloat16)
- **Adapter:** rank 8, alpha 16, dropout 0.05 · 11.3 M trainable params (0.385 %)
- **Target modules:** `q/k/v/o_proj`, `gate/up/down_proj`
- **Data:** 137,678 train / 27,569 val · seed 123
- **Schedule:** 3 epochs, 3,225 steps · effective batch 128 (16 × 8 grad-accum) · max seq 512
- **Optimiser:** `adamw_torch_fused` · LR 2e-4, cosine (warmup 0.03)
- **Best val loss:** 1.6655 (checkpoint-2152) · **Peak GPU memory:** 53.1 GB
"""
QLORA_TRAINING = """\
- **Base:** `google/paligemma-3b-mix-224` (4-bit NF4 quantised)
- **Adapter:** rank 8, alpha 16, dropout 0.05 · 11.3 M trainable params (0.385 %)
- **Target modules:** `q/k/v/o_proj`, `gate/up/down_proj`
- **Data:** 137,678 train / 27,569 val · seed 123
- **Schedule:** 3 epochs, 3,225 steps · effective batch 128 (16 × 8 grad-accum) · max seq 512
- **Optimiser:** `paged_adamw_8bit` · LR 2e-4, cosine (warmup 0.03)
- **Best val loss:** 1.6723 (checkpoint-2152) · **Peak GPU memory:** 64.4 GB
"""

LORA_METRICS = """\
**Blind-domain (target task) — VizWiz**

*VizWiz-VQA*
| Model | Overall | Yes/No | Number | Other | Unans. |
|-------|:-------:|:------:|:------:|:-----:|:------:|
| Base  | 73.95 | 88.71 | 66.10 | 66.10 | 91.67 |
| **LoRA** | **75.80 ±0.21** | 86.55 | 67.56 | 66.54 | **97.73** |

*VizWiz-Caps*
| Model | CIDEr-D | BLEU-4 | METEOR | ROUGE-L | Len. |
|-------|:-------:|:------:|:------:|:-------:|:----:|
| Base  | 55.31 | 12.07 | 14.08 | 28.76 | 5.03 |
| **LoRA** | **98.08 ±1.51** | **30.77 ±0.36** | **23.45 ±0.31** | **49.54 ±0.47** | 10.30 |

**General-domain (control) — VQAv2 / COCO-Caps**

*VQAv2*
| Model | Overall | Yes/No | Number | Other |
|-------|:-------:|:------:|:------:|:-----:|
| Base  | 81.65 | 94.74 | 67.53 | 73.49 |
| LoRA  | 81.15 ±0.04 | 94.57 | 66.38 | 72.85 |

*COCO-Caps*
| Model | CIDEr-D | BLEU-4 | METEOR | ROUGE-L | Len. |
|-------|:-------:|:------:|:------:|:-------:|:----:|
| Base  | 131.21 | 31.96 | 30.62 | 59.17 | 12.40 |
| LoRA  | 124.54 ±1.36 | 34.58 ±0.39 | 30.63 ±0.06 | 58.62 ±0.19 | 11.20 |

<sub>LoRA/QLoRA scores are mean ± std over 3 seeds (42, 123, 7). This checkpoint is the **seed-123** run.</sub>
"""
QLORA_METRICS = """\
**Blind-domain (target task) — VizWiz**

*VizWiz-VQA*
| Model | Overall | Yes/No | Number | Other | Unans. |
|-------|:-------:|:------:|:------:|:-----:|:------:|
| Base   | 73.95 | 88.71 | 66.10 | 66.10 | 91.67 |
| **QLoRA** | **75.71 ±0.14** | 86.54 | 63.88 | 66.51 | **97.68** |

*VizWiz-Caps*
| Model | CIDEr-D | BLEU-4 | METEOR | ROUGE-L | Len. |
|-------|:-------:|:------:|:------:|:-------:|:----:|
| Base   | 55.31 | 12.07 | 14.08 | 28.76 | 5.03 |
| **QLoRA** | **97.44 ±1.45** | **30.63 ±0.29** | **23.30 ±0.35** | **49.33 ±0.38** | 10.24 |

**General-domain (control) — VQAv2 / COCO-Caps**

*VQAv2*
| Model | Overall | Yes/No | Number | Other |
|-------|:-------:|:------:|:------:|:-----:|
| Base   | 81.65 | 94.74 | 67.53 | 73.49 |
| QLoRA  | 80.72 ±0.04 | 94.38 | 65.20 | 72.39 |

*COCO-Caps*
| Model | CIDEr-D | BLEU-4 | METEOR | ROUGE-L | Len. |
|-------|:-------:|:------:|:------:|:-------:|:----:|
| Base   | 131.21 | 31.96 | 30.62 | 59.17 | 12.40 |
| QLoRA  | 123.34 ±1.75 | 34.26 ±0.60 | 30.44 ±0.09 | 58.39 ±0.28 | 11.22 |

<sub>LoRA/QLoRA scores are mean ± std over 3 seeds (42, 123, 7). This checkpoint is the **seed-123** run.</sub>
"""

LORA_SUMMARY = (
    "Adapts PaliGemma to assist blind / low-vision users. On in-domain VizWiz captioning, "
    "CIDEr-D rises **55.3 → 98.1** and BLEU-4 **12.1 → 30.8**, while general-domain VQAv2 stays "
    "within ~0.5 pts of base — strong target gains with minimal catastrophic forgetting. "
    "Full-precision bf16 merge, no adapter required at inference."
)
QLORA_SUMMARY = (
    "Edge-optimised 4-bit (NF4) variant for on-device deployment (e.g. NVIDIA Jetson Orin Nano). "
    "Matches full-precision LoRA almost exactly (VizWiz-VQA 75.71 vs 75.80; CIDEr-D 97.4 vs 98.1; "
    "val loss 1.6723 vs 1.6655) at roughly a third of the memory footprint."
)

MODELS = [
    {
        "tag": "LoRA", "method": "LoRA",
        "adapter_path": "/content/drive/MyDrive/VLMs/outputs/lora_seed_123_rank_8/final_adapter",
        "hub_repo_id": "lamao-ab/paligemma-blind-assist-lora-merged-v1",
        "quantize_4bit": False, "precision_label": "bf16", "peak_vram": "53.1 GB",
        "card": {"summary": LORA_SUMMARY, "training_details": LORA_TRAINING,
                 "metrics_table": LORA_METRICS, "notes": ""},
    },
    {
        "tag": "QLoRA", "method": "QLoRA (4-bit NF4)",
        "adapter_path": "/content/drive/MyDrive/VLMs/outputs/qlora_seed_123_rank_8/final_adapter",
        "hub_repo_id": "lamao-ab/paligemma-blind-assist-qlora-merged-v1",
        "quantize_4bit": True, "precision_label": "4-bit NF4 (double-quant)", "peak_vram": "64.4 GB",
        "card": {"summary": QLORA_SUMMARY, "training_details": QLORA_TRAINING,
                 "metrics_table": QLORA_METRICS,
                 "notes": (
                     "> **Deployment note:** genuine 4-bit (NF4, double-quantised) checkpoint for edge "
                     "inference. Loading requires `bitsandbytes` + a CUDA GPU. On Jetson (L4T/JetPack), "
                     "pip wheels do not work — build bitsandbytes from source with the device's compute "
                     "capability (`-DCOMPUTE_CAPABILITY=87` for Orin)."
                 )},
    },
]


# ---------- MODEL CARD ----------
def build_model_card(cfg, footprint_str):
    c = cfg["card"]
    method_tag = cfg["method"].split()[0].lower()
    if cfg["quantize_4bit"]:
        usage = f"""```python
# 4-bit checkpoint — quantization config travels with the model, no BitsAndBytesConfig needed.
# Requires bitsandbytes + CUDA. On Jetson Orin, build bitsandbytes from source (sm_87).
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from PIL import Image
import torch

repo = "{cfg['hub_repo_id']}"
model = PaliGemmaForConditionalGeneration.from_pretrained(repo, device_map="cuda")
processor = PaliGemmaProcessor.from_pretrained(repo)

image = Image.open("example.jpg").convert("RGB")
prompt = "describe the scene for a visually impaired person"  # match your training prompt
inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=128)
print(processor.decode(out[0], skip_special_tokens=True))
```"""
    else:
        usage = f"""```python
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from PIL import Image
import torch

repo = "{cfg['hub_repo_id']}"
model = PaliGemmaForConditionalGeneration.from_pretrained(
    repo, torch_dtype=torch.bfloat16, device_map="auto"
)
processor = PaliGemmaProcessor.from_pretrained(repo)

image = Image.open("example.jpg").convert("RGB")
prompt = "describe the scene for a visually impaired person"  # match your training prompt
inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=128)
print(processor.decode(out[0], skip_special_tokens=True))
```"""
    return f"""---
license: gemma
library_name: transformers
pipeline_tag: image-text-to-text
base_model: {BASE['base_model_id']}
tags:
- paligemma
- vision-language
- {method_tag}
- blind-assistance
- accessibility
- merged
---

# PaliGemma Blind-Assist ({cfg['method']})

Fine-tuned [`{BASE['base_model_id']}`](https://huggingface.co/{BASE['base_model_id']}) with **{cfg['method']}**.

{c['summary']}

## Model footprint
{footprint_str}

## Training
{c['training_details']}

## Results
{c['metrics_table']}

## Usage
{usage}

{c['notes']}

## License
Inherits the [Gemma license](https://ai.google.dev/gemma/terms) from the base model.
"""


# ---------- WORKER ----------
footprint_summary = []


def merge_and_deploy(cfg):
    print("\n" + "=" * 70)
    print(f"▶️  PROCESSING: {cfg['tag']}  ->  {cfg['hub_repo_id']}")
    print("=" * 70)

    temp_dir = os.path.join(TEMP_ROOT, cfg["tag"].lower())
    scratch_dir = temp_dir + "_bf16"
    for d in (temp_dir, scratch_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
    os.makedirs(temp_dir, exist_ok=True)

    print(f"🧠 Loading base ({BASE['base_model_id']}) in bf16 on CPU...")
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        BASE["base_model_id"], torch_dtype=torch.bfloat16, device_map="cpu")
    print(f"🔗 Attaching {cfg['method']} adapter from {cfg['adapter_path']}...")
    model = PeftModel.from_pretrained(base_model, cfg["adapter_path"])
    print("🔄 Merging adapter weights...")
    merged_model = model.merge_and_unload()

    if not cfg["quantize_4bit"]:
        print(f"💾 Saving merged bf16 model to {temp_dir}...")
        merged_model.save_pretrained(temp_dir, safe_serialization=True)
        report_model = merged_model
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit quantization needs a CUDA GPU. Switch the Colab runtime to GPU.")
        print(f"💾 Staging merged bf16 to {scratch_dir}...")
        merged_model.save_pretrained(scratch_dir, safe_serialization=True)
        merged_model = model = base_model = None
        gc.collect()
        torch.cuda.empty_cache()
        print("🧊 Re-loading in 4-bit NF4 (double-quant) on GPU...")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        report_model = PaliGemmaForConditionalGeneration.from_pretrained(
            scratch_dir, quantization_config=bnb, device_map="auto")
        print(f"💾 Saving 4-bit checkpoint to {temp_dir}...")
        report_model.save_pretrained(temp_dir, safe_serialization=True)
        shutil.rmtree(scratch_dir, ignore_errors=True)

    n_params = sum(p.numel() for p in report_model.parameters())
    footprint_bytes = report_model.get_memory_footprint()
    footprint_gb = footprint_bytes / 1024 ** 3
    precision = cfg["precision_label"]
    print(f"\n🧮 FOOTPRINT: {n_params:,} params ({n_params/1e9:.3f} B) | {precision} | "
          f"{footprint_gb:.3f} GB in memory | train peak {cfg['peak_vram']}")
    footprint_str = (
        f"- **Parameters:** {n_params:,} (~{n_params/1e9:.2f} B)\n"
        f"- **Precision:** {precision} (merged, standalone)\n"
        f"- **In-memory size:** {footprint_gb:.2f} GB (`model.get_memory_footprint()`)\n"
        f"- **Training peak VRAM:** {cfg['peak_vram']}")

    processor = PaliGemmaProcessor.from_pretrained(BASE["base_model_id"])
    processor.save_pretrained(temp_dir)
    with open(os.path.join(temp_dir, "README.md"), "w") as f:
        f.write(build_model_card(cfg, footprint_str))

    print("📂 On-disk size:")
    total = 0
    for root, _, files in os.walk(temp_dir):
        for fn in files:
            sz = os.path.getsize(os.path.join(root, fn))
            total += sz
            mark = "⭐" if sz > 100 * 1024 * 1024 else "•"
            unit = f"{sz/1024**3:.2f} GB" if sz > 100 * 1024 * 1024 else f"{sz/1024**2:.2f} MB"
            print(f"  {mark} {fn:30} | {unit}")
    disk_gb = total / 1024 ** 3
    print(f"📦 TOTAL ON DISK: {disk_gb:.2f} GB")

    footprint_summary.append({"tag": cfg["tag"], "params": n_params, "ram_gb": footprint_gb,
                              "disk_gb": disk_gb, "prec": precision, "peak_vram": cfg["peak_vram"]})

    print(f"☁️  Uploading to {cfg['hub_repo_id']}...")
    api = HfApi()
    api.create_repo(repo_id=cfg["hub_repo_id"], exist_ok=True, private=PRIVATE)
    api.upload_folder(folder_path=temp_dir, repo_id=cfg["hub_repo_id"],
                      commit_message=f"Upload standalone {precision} {cfg['method']} model + processor")
    print(f"✅ DONE: https://huggingface.co/{cfg['hub_repo_id']}")

    merged_model = report_model = processor = base_model = model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(temp_dir, ignore_errors=True)


# ---------- RUN ----------
for cfg in MODELS:
    try:
        merge_and_deploy(cfg)
    except Exception as e:
        print(f"\n❌ FAILED for {cfg['tag']}: {e}")

if footprint_summary:
    print("\n" + "=" * 84)
    print("📊 FOOTPRINT COMPARISON")
    print("=" * 84)
    print(f"{'Model':<8} | {'Params':>14} | {'Precision':>22} | {'RAM (GB)':>9} | "
          f"{'Disk (GB)':>9} | {'Train VRAM':>10}")
    print("-" * 84)
    for r in footprint_summary:
        print(f"{r['tag']:<8} | {r['params']:>14,} | {r['prec']:>22} | "
              f"{r['ram_gb']:>9.2f} | {r['disk_gb']:>9.2f} | {r['peak_vram']:>10}")

print("\n🎉 All done.")
