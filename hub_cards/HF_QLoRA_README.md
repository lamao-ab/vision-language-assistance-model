---
license: gemma
library_name: transformers
pipeline_tag: image-text-to-text
base_model: google/paligemma-3b-mix-224
tags:
- paligemma
- vision-language
- qlora
- blind-assistance
- accessibility
- merged
---

# 👁️ PaliGemma-BlindAssist: A parameter-efficient vision-language model adapter for blind assistance VQA & Captioning using QLoRA

<div align="center">

[![Model](https://img.shields.io/badge/Base-PaliGemma--3B-blue)](https://huggingface.co/google/paligemma-3b-mix-224)
[![Adapter](https://img.shields.io/badge/Method-QLoRA%20(4--bit)-green)](https://arxiv.org/abs/2305.14314)
[![Dataset](https://img.shields.io/badge/Dataset-VizWiz-orange)](https://vizwiz.org/)
[![License](https://img.shields.io/badge/License-Gemma-yellow)](https://ai.google.dev/gemma/terms)

</div>

---


## 🎯 Model Overview

A 4-bit (NF4) quantized adaptation of [`google/paligemma-3b-mix-224`](https://huggingface.co/google/paligemma-3b-mix-224)
for accessibility vision-language tasks, trained with QLoRA on the VizWiz
datasets (seed 123). The adapter is merged into the bfloat16 weights and the
result is requantized to 4 bits as a standalone checkpoint. It performs visual
question answering and scene captioning for blind and low-vision users, and is
intended for on-device inference on memory-constrained hardware such as the
NVIDIA Jetson Orin Nano 8 GB (model-level characterization below).

QLoRA and the [LoRA variant](https://huggingface.co/lamao-ab/paligemma-blind-assist-lora-merged-v1)
are two practical training recipes; they differ in more than weight precision
(optimizer, k-bit preparation and loading path), so this is not a single-variable
ablation of quantization. Three-seed means of the unmerged adapters: VizWiz-VQA
75.71 ± 0.14 (QLoRA) and 75.80 ± 0.21 (LoRA); VizWiz-Captions test CIDEr-D
103.02 ± 1.18 and 104.37 ± 1.36 (QLoRA lower in every seed). This checkpoint
(adapter merged and requantized to 4 bits) scores 75.45 on VizWiz-VQA and 102.24
CIDEr-D on the VizWiz-Captions test images, against 76.05 and 105.37 for the merged
bfloat16 LoRA checkpoint, while reducing the static memory footprint from 5.45 GB
to 2.14 GB. The bfloat16 model does not fit on the Jetson Orin Nano.


## 🔧 Specifications
- **Base:** `google/paligemma-3b-mix-224` (4-bit NF4 quantized during training)
- **Adapter:** rank 8, alpha 16, dropout 0.05 · 11,298,816 trainable params (0.385 %)
- **Target modules:** applied to the language model (`q/k/v/o_proj`, `gate/up/down_proj`; 126 modules, 9,805,824 parameters) and to the vision encoder (SigLIP `q/k/v` projections; 81 modules, 1,492,992 parameters)
- **Data:** 137,678 train / 27,569 val · seed 123
- **Schedule:** 3 epochs (3,228 steps) · effective batch 128 (16 × 8 grad-accum) · max seq 512 · `checkpoint-2152` (end of epoch 2) selected by validation loss
- **Optimiser:** `paged_adamw_8bit` · LR 2e-4, cosine (warmup 0.03)
- **Parameters:** 2.92 B (adapter merged into the bfloat16 weights, then requantized)
- **Precision:** 4-bit NF4, double-quantized, bfloat16 compute
- **Size:** 2.29 GB on disk · 2.14 GB static footprint after loading (`torch.cuda.memory_allocated()`; `model.get_memory_footprint()` reports 2.10 GB because it does not count the 36.5 MB of quantization constants)
- **Training peak VRAM:** 64.4 GB


## 📊 Measured Performance

VizWiz-Captions results on the **test** images are the main captioning results. The
VizWiz-Captions validation images contributed to checkpoint, rank and mixing-ratio
selection through the validation loss; the test images were not used for selection. LoRA/QLoRA rows are mean ± std over 3 adapter
seeds (42, 123, 7); this checkpoint is the **seed-123** run (see "This checkpoint").
Adapted models are evaluated with the accessibility prompt; the base model is shown
under its generic prompt and, for VizWiz, also under the accessibility prompt.

### Blind-domain (target task) — VizWiz

*VizWiz-VQA (test)*
| Model | Overall | Yes/No | Number | Other | Unans. |
|-------|:-------:|:------:|:------:|:-----:|:------:|
| Base (generic prompt) | 73.95 | 88.71 | 66.10 | 66.10 | 91.67 |
| Base (accessibility prompt) | 74.83 | 87.16 | 64.63 | 66.40 | 94.52 |
| **QLoRA** | **75.71 ±0.14** | 86.54 | 63.88 | 66.51 | **97.68** |

*VizWiz-Captions (test)*
| Model | CIDEr-D | BLEU-4 | METEOR | ROUGE-L | Len. |
|-------|:-------:|:------:|:------:|:-------:|:----:|
| Base (generic prompt) | 60.17 | 12.53 | 14.22 | 29.40 | 5.00 |
| Base (accessibility prompt) | 25.01 | 0.48 | 7.00 | 15.67 | 1.83 |
| **QLoRA** | **103.02 ±1.18** | **31.39 ±0.28** | **23.64 ±0.25** | **50.40 ±0.32** | 10.18 |

*VizWiz-Captions (validation)*
| Model | CIDEr-D | BLEU-4 | METEOR | ROUGE-L | Len. |
|-------|:-------:|:------:|:------:|:-------:|:----:|
| Base (generic prompt) | 55.31 | 12.07 | 14.08 | 28.76 | 5.03 |
| QLoRA | 97.44 ±1.45 | 30.63 ±0.29 | 23.30 ±0.35 | 49.33 ±0.38 | 10.24 |

### This checkpoint (seed 123, merged and requantized to 4 bits)

| | VizWiz-VQA test | Captions test CIDEr-D | Captions val CIDEr-D |
|---|:---:|:---:|:---:|
| This checkpoint | 75.45 | 102.24 | 97.19 |
| Unmerged seed-123 adapter | 75.86 | 103.53 | 98.95 |

On this checkpoint: VizWiz-VQA yes/no 85.62, number 65.45, other 66.10, unanswerable 97.81;
VizWiz-Captions test BLEU-4 31.03, METEOR 23.56, ROUGE-L 49.83, 10.32 words per caption.
Conversion to the deployed representation lowers the seed-123 scores by 0.40 points on
VizWiz-VQA (95% paired interval −0.75 to −0.04) and by 1.28 CIDEr-D on the test captions
(interval −2.19 to −0.37).

### General-domain (control) — VQAv2 / COCO-Caps (generic prompt)

*VQAv2 (test-standard)*
| Model | Overall | Yes/No | Number | Other |
|-------|:-------:|:------:|:------:|:-----:|
| Base   | 81.65 | 94.74 | 67.53 | 73.49 |
| QLoRA  | 80.72 ±0.04 | 94.38 | 65.20 | 72.39 |

*COCO-Caps (val2014; a retention measure, not a held-out test: 75.3% of val2014 are restval images listed in the base model's COCO training data)*
| Model | CIDEr-D | BLEU-4 | METEOR | ROUGE-L | Len. |
|-------|:-------:|:------:|:------:|:-------:|:----:|
| Base   | 131.21 | 31.96 | 30.62 | 59.17 | 12.40 |
| QLoRA  | 123.34 ±1.75 | 34.26 ±0.60 | 30.44 ±0.09 | 58.39 ±0.28 | 11.22 |

### Confidence (VizWiz-VQA test, this checkpoint)

The mean token log-probability of an answer, exponentiated, is overconfident: mean confidence 0.846 against accuracy 0.755 (expected calibration error 0.093). 4,386 of the 8,000 predictions (54.8%) have confidence of at least 0.9; 146 of them are wrong (no annotator agrees) and 2,868 are `unanswerable` outputs. High confidence on `unanswerable` is an abstention, not a reliable answer.

### On-device characterization (Jetson Orin Nano 8 GB, 15 W)

| | |
|---|---|
| Inference latency | 0.93 s (VQA, 30 tokens) / 1.97 s (captioning, 64 tokens) |
| End-to-end response | 1.36–2.41 s (capture, inference, speech synthesis) |
| Power | 9.60 W active, 4.14 W idle |
| Energy per query | 16.62 ± 0.16 J over the generation window (9.02 ± 0.15 J above idle); one fixed frame, n = 30, 64-token cap |


## 🧪 Evaluation protocol
- **Prompts (accessibility):** VQA `<image>Assist a blind person: {question}`; captioning `<image>Describe this scene for a blind person.`
- **Decoding:** greedy, at most 64 new tokens, batch size 128.
- **VizWiz-VQA metric:** the official consensus score, i.e. the average over the ten leave-one-out annotator subsets of min(1, matches/3). It is not a binary indicator.
- **VizWiz-Captions test:** the officially released evaluation captions, matched to the test images by the authors (the file provides 7,983 caption sets for the 8,000 test images, 9 of which match no image, so 7,974 images are evaluated).
- **Reproducibility:** predictions, scorers, per-run score files and an evaluation manifest are in the
  [code repository](https://github.com/lamao-ab/vision-language-assistance-model). Every reported score is recomputed exactly from the released predictions. Greedy generation can differ slightly between hardware and batch compositions, so differences below about 0.15 points should not be interpreted.
- **Revision:** the paper cites revision `0a66cea` of this repository. Editing this card does not change the weights; use the full commit hash from the commit history to pin that revision.


## ⚠️ Scope and limitations
These figures characterize model-level execution on a development board and benchmark behaviour. The model has not been evaluated with blind or visually impaired users, and is not a validated assistive product. It can give confident wrong answers and should not be relied on for safety-critical decisions such as medication, expiry dates or street crossing.


## 🚀 Usage
```python
# 4-bit checkpoint — quantization config travels with the model, no BitsAndBytesConfig needed.
# Requires bitsandbytes + CUDA. On Jetson Orin, build bitsandbytes from source (sm_87).
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from PIL import Image
import torch

repo = "lamao-ab/paligemma-blind-assist-qlora-merged-v1"
model = PaliGemmaForConditionalGeneration.from_pretrained(repo, device_map="cuda")
processor = PaliGemmaProcessor.from_pretrained(repo)

image = Image.open("example.jpg").convert("RGB")
prompt = "<image>Describe this scene for a blind person."   # VQA: "<image>Assist a blind person: <question>"
inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

> **Deployment note:** genuine 4-bit (NF4, double-quantized) checkpoint for edge inference. Loading requires `bitsandbytes` + a CUDA GPU. On Jetson (L4T/JetPack), pip wheels do not work — build bitsandbytes from source with the device's compute capability (`-DCOMPUTE_CAPABILITY=87` for Orin).

## 📄 License
Inherits the [Gemma license](https://ai.google.dev/gemma/terms) from the base model.