---
license: gemma
library_name: transformers
pipeline_tag: image-text-to-text
base_model: google/paligemma-3b-mix-224
tags:
- paligemma
- vision-language
- lora
- blind-assistance
- accessibility
- merged
---


# 👁️ PaliGemma-BlindAssist: A parameter-efficient vision-language model adapter for blind assistance VQA & Captioning using LoRA

<div align="center">

[![Model](https://img.shields.io/badge/Base-PaliGemma--3B-blue)](https://huggingface.co/google/paligemma-3b-mix-224)
[![Adapter](https://img.shields.io/badge/Method-LoRA-green)](https://arxiv.org/abs/2106.09685)
[![Dataset](https://img.shields.io/badge/Dataset-VizWiz-orange)](https://vizwiz.org/)
[![License](https://img.shields.io/badge/License-Gemma-yellow)](https://ai.google.dev/gemma/terms)

</div>

---


## 🎯 Model Overview

A full-precision (bfloat16) adaptation of
[`google/paligemma-3b-mix-224`](https://huggingface.co/google/paligemma-3b-mix-224)
for accessibility vision-language tasks, trained with LoRA on the VizWiz
datasets and merged into a standalone checkpoint (seed 123). It performs visual
question answering and scene captioning for blind and low-vision users.

It is the full-precision counterpart of the
[4-bit QLoRA variant](https://huggingface.co/lamao-ab/paligemma-blind-assist-qlora-merged-v1),
which is the variant intended for on-device deployment: this bfloat16 model does
not fit on the 8 GB Jetson Orin Nano (out-of-memory error at load). LoRA and
QLoRA are two practical training recipes that differ in more than weight
precision (optimizer, k-bit preparation and loading path). Compared with the
QLoRA recipe, the three-seed means differ by 0.09 points on VizWiz-VQA
(95% paired interval −0.38 to +0.19) and by 1.35 CIDEr-D on the VizWiz-Captions
test images (QLoRA lower in every seed).

Adaptation improves the target task. On the VizWiz-Captions test images,
CIDEr-D rises from 60.17 (base model, generic prompt) to 104.37 (three-seed
mean; 105.37 for this checkpoint) and the mean caption length from 5.00 to
10.19 words, matching the dataset's more descriptive reference style.
VizWiz-VQA accuracy rises from 73.95 (generic prompt) or 74.83 (accessibility
prompt) to 75.80, a gain of +0.97 points over the better base setting.
General-domain performance declines modestly, by 0.50 points on VQAv2 and by
5% relative on COCO-Caps; COCO is a retention measure and not a held-out test
(75.3% of COCO val2014 are restval images that the base model's card lists as
COCO training data).



## 🔧 Specifications
- **Base:** `google/paligemma-3b-mix-224` (bfloat16)
- **Adapter:** LoRA, rank 8, alpha 16, dropout 0.05 · 11,298,816 trainable params (0.385 %)
- **Target modules:** applied to the language model (`q/k/v/o_proj`, `gate/up/down_proj`; 126 modules, 9,805,824 parameters) and to the vision encoder (SigLIP `q/k/v` projections; 81 modules, 1,492,992 parameters)
- **Data:** 137,678 train / 27,569 val · seed 123
- **Schedule:** 3 epochs (3,228 steps) · effective batch 128 (16 × 8 grad-accum) · max seq 512 · `checkpoint-2152` (end of epoch 2) selected by validation loss
- **Optimiser:** `adamw_torch_fused` · LR 2e-4, cosine (warmup 0.03)
- **Parameters:** 2.92 B (adapter merged into the base weights)
- **Precision:** bf16 (merged, standalone)
- **Size:** 5.85 GB on disk · 5.45 GB in memory after loading
- **Training peak VRAM:** 53.1 GB


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
| **LoRA** | **75.80 ±0.21** | 86.55 | 67.56 | 66.54 | **97.73** |

*VizWiz-Captions (test)*
| Model | CIDEr-D | BLEU-4 | METEOR | ROUGE-L | Len. |
|-------|:-------:|:------:|:------:|:-------:|:----:|
| Base (generic prompt) | 60.17 | 12.53 | 14.22 | 29.40 | 5.00 |
| Base (accessibility prompt) | 25.01 | 0.48 | 7.00 | 15.67 | 1.83 |
| **LoRA** | **104.37 ±1.36** | **31.70 ±0.20** | **23.82 ±0.31** | **50.63 ±0.41** | 10.19 |

*VizWiz-Captions (validation)*
| Model | CIDEr-D | BLEU-4 | METEOR | ROUGE-L | Len. |
|-------|:-------:|:------:|:------:|:-------:|:----:|
| Base (generic prompt) | 55.31 | 12.07 | 14.08 | 28.76 | 5.03 |
| LoRA | 98.08 ±1.51 | 30.77 ±0.36 | 23.45 ±0.31 | 49.54 ±0.47 | 10.30 |

### This checkpoint (seed 123, merged bfloat16)

| | VizWiz-VQA test | Captions test CIDEr-D | Captions val CIDEr-D |
|---|:---:|:---:|:---:|
| This checkpoint | 76.05 | 105.37 | 99.16 |
| Unmerged seed-123 adapter | 75.91 | 105.65 | 99.05 |

On this checkpoint: VizWiz-VQA yes/no 85.44, number 65.77, other 66.89, unanswerable 98.05;
VizWiz-Captions test BLEU-4 31.80, METEOR 24.09, ROUGE-L 50.93, 10.31 words per caption.

### General-domain (control) — VQAv2 / COCO-Caps (generic prompt)

*VQAv2 (test-standard)*
| Model | Overall | Yes/No | Number | Other |
|-------|:-------:|:------:|:------:|:-----:|
| Base  | 81.65 | 94.74 | 67.53 | 73.49 |
| LoRA  | 81.15 ±0.04 | 94.57 | 66.38 | 72.85 |

*COCO-Caps (val2014; a retention measure, not a held-out test)*
| Model | CIDEr-D | BLEU-4 | METEOR | ROUGE-L | Len. |
|-------|:-------:|:------:|:------:|:-------:|:----:|
| Base  | 131.21 | 31.96 | 30.62 | 59.17 | 12.40 |
| LoRA  | 124.54 ±1.36 | 34.58 ±0.39 | 30.63 ±0.06 | 58.62 ±0.19 | 11.20 |


## 🧪 Evaluation protocol
- **Prompts (accessibility):** VQA `<image>Assist a blind person: {question}`; captioning `<image>Describe this scene for a blind person.`
- **Decoding:** greedy, at most 64 new tokens, batch size 128.
- **VizWiz-VQA metric:** the official consensus score, i.e. the average over the ten leave-one-out annotator subsets of min(1, matches/3). It is not a binary indicator.
- **VizWiz-Captions test:** the officially released evaluation captions, matched to the test images by the authors (the file provides 7,983 caption sets for the 8,000 test images, 9 of which match no image, so 7,974 images are evaluated).
- **Reproducibility:** predictions, scorers, per-run score files and an evaluation manifest are in the
  [code repository](https://github.com/lamao-ab/vision-language-assistance-model). Every reported score is recomputed exactly from the released predictions. Greedy generation can differ slightly between hardware and batch compositions, so differences below about 0.15 points should not be interpreted.
- **Revision:** the paper cites revision `0545d03` of this repository. Editing this card does not change the weights; use the full commit hash from the commit history to pin that revision.


## ⚠️ Scope and limitations
These figures characterize model-level behaviour on benchmarks. The model has not been evaluated with blind or visually impaired users and is not a validated assistive product. It can give confident wrong answers and should not be relied on for safety-critical decisions such as medication, expiry dates or street crossing.


## 🚀 Usage
```python
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from PIL import Image
import torch

repo = "lamao-ab/paligemma-blind-assist-lora-merged-v1"
model = PaliGemmaForConditionalGeneration.from_pretrained(
    repo, torch_dtype=torch.bfloat16, device_map="auto"
)
processor = PaliGemmaProcessor.from_pretrained(repo)

image = Image.open("example.jpg").convert("RGB")
prompt = "<image>Describe this scene for a blind person."   # VQA: "<image>Assist a blind person: <question>"
inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```



## 📄 License
Inherits the [Gemma license](https://ai.google.dev/gemma/terms) from the base model.
