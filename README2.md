# PaliGemma + LoRA/QLoRA Blind Assistance: Edge-Deployable Vision-Language Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-red.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/lamao-ab)

**Parameter-efficient fine-tuning of PaliGemma-3B for blind assistance using LoRA/QLoRA on VizWiz dataset (VQA & Captionning), deployable on NVIDIA Jetson Orin Nano 8GB.**

 ![jetson-nano](https://github.com/user-attachments/assets/5c949189-4f9c-44c8-aab8-c5df9e49b691)

📄 **Paper**: [Link to ArXiv/IEEE](YOUR_PAPER_LINK)  
🤗 **Models**: [HuggingFace Hub](https://huggingface.co/lamao-ab)  

---

## 🎯 Overview
This repository provides a complete pipeline for training and deploying vision-language models on edge devices for real-time blind assistance. Our deployed 4-bit (QLoRA) model achieves:
- **75.71% VQA accuracy** on VizWiz-VQA (10-choose-9 metric), within 0.09 points of the full-precision LoRA model (75.80%)
- **97.44 CIDEr-D** on VizWiz-Captions, within 0.6 points of full-precision LoRA (98.08)
- **2.10 GB static memory footprint** (2.25 GB peak during inference), enabling deployment on an 8 GB Jetson Orin Nano
- **0.93–1.97 s inference latency** (device-side compute) for interactive VQA and scene-description queries
- **~24 hours realistic battery life** (10 queries/hour, 100 Wh pack); ~10.4 hours under continuous inference

### Key Features
✅ **Parameter-Efficient Fine-Tuning**: Train only 11.3M parameters (0.385% of total) using LoRA \
✅ **4-bit Quantization**: QLoRA reduces the deployed footprint from 5.45 GB (bf16) to 2.13 GB — the bf16 model does not fit in the Jetson's available memory, making quantization a deployment requirement, not just an optimization \
✅ **Multi-Task Learning**: Joint training on VQA and image captioning, with both tasks improving over the base checkpoint \
✅ **Edge Deployment**: Runs on an NVIDIA Jetson Orin Nano 8GB (4-bit model only — see Hardware Requirements) \
✅ **Privacy-Aware**: All processing on-device, reducing (not eliminating) exposure of sensitive data to external services \
✅ **Open Source**: All code, configs, and trained weights available 

---

## 🎥 Demo 


https://github.com/user-attachments/assets/5e2f2d8d-28ae-4121-8e7a-79bf5d7b91ee

https://github.com/user-attachments/assets/1c8ed7b6-f0dd-49cd-87ef-601997330a3f


  # Real images captured by usb camera:
   **VQA (number of person?):** 6 \
   **CAP:** A group of people sitting at a table in a restaurant
  ![A group of people sitting at a table in a restaurant](https://github.com/user-attachments/assets/b641e527-86c4-490e-a53d-df98fe0bac6b)

   **VQA (traffic light color?):** unanswerable \
   **CAP:** Quality issues are too severe to recognize visual content.
   ![last_view_22](https://github.com/user-attachments/assets/4d189a91-b759-4e8b-a785-5aea7a52a4ba)


  **VQA (screen color?):** green \
  **CAP:** A television screen with a soccer game on it. \
  ![A television screen with a soccer game on it](https://github.com/user-attachments/assets/2d4d3794-109b-42ba-aa9e-22a9d669c363)

  
  **VQA(identify objetcs?:)** Air purifier \
  **CAP:** A white air conditioner with a blue light on top. 
  ![A white air conditioner with a blue light on top](https://github.com/user-attachments/assets/c83630bc-33e8-40f4-997e-3318cca42bcd)  


## 🚀 Quick Start

### Installation

```bash
# Install torch
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
# Clone repository
git clone https://github.com/lamao-ab/vision-language-assistance-model.git

# Create conda environment
conda create -n paligemma python=3.10
conda activate paligemma

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

```bash
# Download and prepare VizWiz dataset
python data/prepare_dataset.py \
    --workdir            data/vizwiz \
    --train_output       data/train_dataset \
    --val_output         data/val_dataset
```

### Training
**HuggingFace Authentication**
```bash
import os
from huggingface_hub import login
if not os.path.exists(os.path.expanduser("~/.cache/huggingface/token")):
    print("Login required...")
    login()
else:
    print("✅ Already logged in")
```

**LoRA adapter training:**
```bash
python src/train_lora.py \
    --train_dataset_path data/train_dataset \
    --val_dataset_path   data/val_dataset \
    --base_output_dir    outputs/lora \
    --seed               123 \
    --learning_rate      2e-4 \
    --lora_rank          8 \
    --num_epochs         3 \
    --batch_size         16 \
    --grad_accum         8 
```

**QLoRA adapter training (4-bit Quantized) :**
```bash
python src/train_qlora.py \
    --train_dataset_path data/train_dataset/ \
    --val_dataset_path   data/val_dataset \
    --output_dir         outputs/qlora \
    --seed              123 \
    --learning_rate      2e-4 \
    --lora_rank          8 \
    --num_epochs         3 \
    --batch_size         16 \
    --grad_accum         8 
```

### Evaluation and Scoring on VizWiz & Benchmark Datasets
```bash
# Option A — local adapter
python src/evaluate_vizwiz.py \
    --model_id   outputs/run/final_adapter \    
    --task  both \                              # vqa, caps or both
    --prompt_style custom \                     # or generic
    --precision    4bit \                       # or bf16
    --workdir      eval_data \
    --output_dir outputs/predictions \
    --batch_size 32 \
    --max_tokens 64 \
    --tag  qlora_seed123_custom

python src/score_vizwiz_vqa.py \
    --gt   eval_data/vizwiz_vqa_data/VQA_test.json  \
    --pred outputs/predictions/vizwiz_vqa_test_predictions_base_custom.json \
    --out outputs/predictions/score_vizwiz_vqa_test_predictions_base_custom.json

python src/score_captions.py --dataset vizwiz \
    --gt   eval_data/vizwiz_caps_data/annotations/val.json \
    --pred outputs/predictions/vizwiz_caption_val_predictions_base_custom.json \
    --out  outputs/predictions/score_vizwiz_caption_val_predictions_base_custom.json
```

```bash
# Option B — Hub model 
python src/evaluate_benchmark.py \
    --model_id   lamao-ab/paligemma-blind-assist-qlora-merged-v1 \
    --task  both \
    --prompt_style generic \
    --precision    bf16 \
    --workdir      eval_data \
    --batch_size  32 \
    --output_dir outputs/predictions \
    --batch_size 32 \
    --max_tokens 64 \
    --tag base_generic

python src/score_captions.py --dataset coco \ 
    --gt    eval_data/coco_data/annotations/captions_val2014.json \
    --pred  outputs/predictions/coco_caption_val_predictions_base_generic.json \
    --out   outputs/predictions/score_coco_caption_val_predictions_base_generic.json
```

### Inference 
```bash
# Option A — local adapter
python src/predict.py \
    --model_id   outputs/run/final_adapter \
    --task       vqa \
    --image_dir  /content/paligemma-blind-assist/data/images \
    --output     results/vqa_only.json
    --batch_size 32 \
    --max_tokens 64

# Option B — Hub model 
python src/predict.py \
    --model_id   lamao-ab/paligemma-blind-assist-qlora-merged-v1 \
    --task       caps \
    --image_dir  /content/paligemma-blind-assist/data/images \
    --output     results/caps_results.json \
    --batch_size 32 \
    --max_tokens 64
```

### Deployment on Jetson Nano Orin 8GB

```bash
# Interactive demo on Jetson Nano Orin 8GB
cd vision-language-assistance-model/deployment
pip install -r requirements.txt
python blind-assistance-system.py 
```


## 📊 Results

### VQA Performance (VizWiz-VQA test set, VQAv2 test-standard server)
Mean ± std over 3 seeds (42, 123, 7) for LoRA/QLoRA. Base is the off-the-shelf PaliGemma `mix` checkpoint.

| Model                    | VizWiz-VQA        | VQAv2             | Memory Footprint (GB) |
|--------------------------|--------------------|--------------------|------------------------|
| PaliGemma-3B (`mix`)     | 73.95%             | 81.65%             | 5.45\*                 |
| PaliGemma-3B (`mix`), accessibility prompt | 74.83%       | —                  | 5.45                   |
| PaliGemma-3B + LoRA      | 75.80% ± 0.21      | 81.15% ± 0.04      | 5.45                   |
| **PaliGemma-3B + QLoRA** | **75.71% ± 0.14**  | **80.72% ± 0.04**  | **2.13**               |

\*Base and LoRA share the same bf16 backbone footprint (LoRA freezes the base weights); only the adapter differs.

Gain of the adapted models over the base **at its better prompt** (accessibility): LoRA +0.97 points (95% CI 0.46 to 1.46), QLoRA +0.88 (0.35 to 1.39). Against the generic-prompt base the gains are +1.85 and +1.76, of which +0.88 comes from the prompt alone. Excluding the 328 test images whose photograph also occurs in the training data under another file name, LoRA scores 75.56 and QLoRA 75.47.

The QLoRA rows are the training-time adapter over the 4-bit backbone. The exported merged model (Hub revision `0a66cea`, adapter merged into BF16 weights and requantized) scores 75.53 under the same evaluation settings.

### Captioning Performance (VizWiz-Caps validation set, development-set results)
| Model                    | CIDEr-D            | BLEU-4             | METEOR             | ROUGE-L            | Avg. Length |
|--------------------------|---------------------|--------------------|--------------------|---------------------|-------------|
| PaliGemma-3B (`mix`)     | 55.31               | 12.07              | 14.08              | 28.76               | 5.03        |
| PaliGemma-3B (`mix`), accessibility prompt | 22.30      | 0.39               | 6.68               | 15.06               | 1.76        |
| PaliGemma-3B + LoRA      | 98.08 ± 1.51        | 30.77 ± 0.36       | 23.45 ± 0.31       | 49.54 ± 0.47        | 10.30       |
| **PaliGemma-3B + QLoRA** | **97.44 ± 1.45**    | **30.63 ± 0.29**   | **23.30 ± 0.35**   | **49.33 ± 0.38**    | **10.24**   |
| QLoRA, exported merged model (seed 123) | 96.99            | 30.46              | 23.29              | 48.87               | 10.37       |
The validation images also selected the checkpoints (epoch 2, `checkpoint-2152`, in all six runs), so these are development-set results. On the 3,207 eligible validation images that are neither VQA photographs nor duplicates of training photographs, LoRA scores 100.45 ± 1.33 CIDEr-D against 56.65 for the base (CIDEr-D document frequencies are recomputed on each subset).


### Captioning Performance (COCO-Caps validation set — general-domain control)
| Model                    | CIDEr-D             | BLEU-4             | METEOR             | ROUGE-L             | Avg. Length |
|--------------------------|----------------------|--------------------|--------------------|----------------------|-------------|
| PaliGemma-3B (`mix`)     | 131.21               | 31.96              | 30.62              | 59.17                | 12.4        |
| PaliGemma-3B + LoRA      | 124.54 ± 1.36        | 34.58 ± 0.39       | 30.63 ± 0.06       | 58.62 ± 0.19         | 11.20       |
| **PaliGemma-3B + QLoRA** | **123.34 ± 1.75**    | **34.26 ± 0.60**   | **30.44 ± 0.09**   | **58.39 ± 0.28**     | **11.22**   |

General-domain performance declines only modestly (≤6% CIDEr-D) relative to the large VizWiz-domain gains, while BLEU-4 actually *improves* over the base model on COCO-Caps. COCO val2014 contains the 30,504 Karpathy restval images (75.3% of 40,504) that the PaliGemma model card lists as COCO training data, so COCO is a retention measure and not a held-out test.

### Deployment Metrics (NVIDIA Jetson Orin Nano 8GB, 15W mode, clocks locked)
Only the 4-bit QLoRA model (the exported merged model) is deployed on-device; the bf16 model (5.45 GB) does not reliably fit in the device's available memory and is not used for edge inference. Raw logs, per-run latencies and checksums are in [`deployment/records/`](deployment/records/).

- **Static memory footprint**: 2.14 GB (2.25 GB peak CUDA memory reserved during inference)
- **System memory**: 4.10 GB used of 7.44 GB total, 3.09 GB available
- **Inference latency** (device-side, run 3 of 3 runs): 0.93 ± 0.11 s (VQA, 30-token cap) and 1.97 ± 0.26 s (captioning, 64-token cap); pooled medians 0.90 s and 1.90 s. System latency including camera capture, excluding speech playback: 1.36 s and 2.41 s; user-perceived latency including speech synthesis: 2.37 s and 5.88 s
- **Power** (VDD_IN, tegrastats, 100 ms): 9.60 W active, 4.14 W idle with the model loaded
- **Energy per caption query**: 16.62 ± 0.16 J over the generation window, 9.02 ± 0.15 J above idle (single capture, n = 30, fixed frame, 64-token cap)
- **Battery life**: illustrative calculation from the measured power and an assumed duty cycle, not a measurement: 24.0 h at 10 queries per hour and 10.4 h of continuous inference on a 100 Wh pack
- **A100 reference** (A100-SXM4-40GB, merged 4-bit checkpoint, batch size 1): 0.27 s (VQA) and 0.57 s (captioning) per query at 65 to 77 W card power. This compares power draw, not energy, because the measurement boundaries differ

---

## 🛠️ Repository Structure

```
├── archive/           # superseded files kept for provenance (stale training config)
├── data/              # dataset download and preprocessing
├── src/               # training, evaluation, scoring and analysis scripts
├── deployment/        # Jetson application, benchmarks, A100 measurement; records/ holds raw logs and checksums
├── results/           # outputs behind the reported tables (see below)
├── notebooks/         # all-in-one Colab notebooks with their outputs
├── requirements.txt   # environment that trained the published models
└── CHANGELOG.md       # changes since the audited commit 4f4d378 (tag v1.0)
```

<!-- See [TRAINING.md](docs/TRAINING.md) for detailed training instructions and [DEPLOYMENT.md](docs/DEPLOYMENT.md) for edge deployment guide.
 -->

---

## 🔎 Evaluation Representations

| Result | Representation evaluated |
|---|---|
| LoRA tables | LoRA adapter over the bf16 backbone, bf16 precision |
| QLoRA tables (three-seed means, Table 20) | training-time adapter, unmerged, over the 4-bit NF4 backbone (`checkpoint-2152` of each seed) |
| Deployment (memory, latency, energy, A100) and the confidence analysis | exported merged model: adapter merged into BF16 weights, requantized to 4 bits |

Hub revisions: LoRA `0545d03`, QLoRA merged `0a66cea` ([huggingface.co/lamao-ab](https://huggingface.co/lamao-ab)). On VizWiz-VQA, seed 123: unmerged adapter 75.86, exported model 75.53 under identical settings (same script, accessibility prompt, batch size 128, greedy decoding, 64 new tokens, 4-bit), 75.59 in the confidence analysis (batch size 32).

---

## 🔁 Reproducing the Analyses

| Analysis | Script | Output |
|---|---|---|
| Trainable-parameter inventory and adapter audit | `src/inventory_trainable.py`, `src/audit_adapters.py` | `results/adapters/` |
| Completeness of prediction files | `src/check_completeness.py` | `results/completeness_report_vqa.json` |
| Image overlap between partitions and tasks | `src/make_overlap_spec.py`, `src/audit_overlap.py`, `src/overlap_exposure_analysis.py` | `results/overlap/` |
| Selective prediction (retained-set composition, cross-fitted thresholds) | `src/selective_prediction_analysis.py` | `results/selective/` |
| Paired VQA comparisons (gains, prompt and adaptation decomposition, adapter against exported model) | `src/paired_comparisons.py` | `results/paired_comparisons.json` |
| Blinded caption review (single rater) | `src/build_caption_review.py`, `src/analyze_caption_review.py`, `src/caption_review_stats.py` | `results/caption_review/` |
| Export to the Hub (merge in bf16, 4-bit NF4 requantization) | `src/deploy_paligemma_to_hf.py` | Hub revisions `0545d03`, `0a66cea` |
| Baseline checkpoints (native resolution) | `src/evaluate_baselines.py`, `src/score_captions_baseline.py` | `results/` |
| Deployment measurements | `deployment/*.py` | `deployment/records/` |

The scorers `src/score_vizwiz_vqa.py` and `src/score_captions.py` are unchanged since the audited commit `4f4d378`.

---

## 🎓 Citation

If you use this work in your research, please cite:

```bibtex
@article{,
  title={},
  author={},
  journal={IEEE Access},
  year={2026},
  volume={XX},
  pages={XXX-XXX},
  doi={XX.XXXX/ACCESS.XXXX.XXXXXXX}
}
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **VizWiz Dataset**: [Gurari et al., 2018](https://vizwiz.org/)
- **PaliGemma Model**: [Google DeepMind, 2024](https://huggingface.co/google/paligemma-3b-mix-224)
- **QLoRA**: [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)
- **PEFT Library**: [Hugging Face](https://github.com/huggingface/peft)

---

## 📧 Contact

For questions or collaboration:
- **Email**: a.boussihmed@ump.ac.ma
- **GitHub Issues**: [Open an issue](https://github.com/lamao-ab/paligemma-qlora-blind-assist/issues)
- **HugFace**: [@lamao-ab](https://huggingface.co/lamao-ab)

---
## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=BOUSSIHMED_AHMED/paligemma-qlora-blind-assistance&type=Date)](https://star-history.com/#BOUSSIHMED_AHMED/paligemma-qlora-blind-assistance&Date)

