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
- **2.13 GB static memory footprint** (2.25 GB peak during inference), enabling deployment on an 8 GB Jetson Orin Nano
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
# Clone repository
git clone https://github.com/lamao-ab/vision-language-assistance-model.git

# Create conda environment
conda create -n paligemma python=3.10
conda activate paligemma

# Install dependencies
cd vision-language-assistance-model
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
    --lora_rank          8 \
    --num_epochs         3 \
    --batch_size         16 \
    --grad_accum         8 \
    --dataloader_workers 8
```

**QLoRA adapter training (4-bit Quantized) :**
```bash
python src/train_qlora.py \
    --train_dataset_path data/train_dataset/ \
    --val_dataset_path   data/val_dataset \
    --output_dir         outputs/qlora \
    --num_epochs         3 \
    --batch_size         16 \
    --grad_accum         8 \
    --dataloader_workers 8
```
### Evaluation on VizWiz & Benchmark Datasets
```bash
# Option A — local adapter
python src/evaluate_vizwiz.py \
    --model_id   outputs/run/final_adapter \
    --task  vqa \            # vqa, caps or both
    --output_dir outputs/predictions
    --batch_size 32 \
    --max_tokens 64

# Option B — Hub model 
python src/evaluate_vizwiz.py \
    --model_id   lamao-ab/paligemma-blind-assist-qlora-merged-v1 \
    --task  caps \  
    --output_dir outputs/predictions \
    --batch_size 32 \
    --max_tokens 64

# Option A — local adapter
python src/evaluate_benchmark.py \
    --model_id   outputs/run/final_adapter \ 
    --task  both \
    --output_dir outputs/predictions \
    --batch_size 32 \
    --max_tokens 64

# Option B — Hub model 
python src/evaluate_benchmark.py \
    --model_id   lamao-ab/paligemma-blind-assist-qlora-merged-v1 \
    --task       both \
    --output_dir outputs/predictions \
    --batch_size 32 \
    --max_tokens 64
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
cd paligemma-qlora-blind-assistance/deployment
pip install -r requirements.txt
python blind-assistance-system.py 
```


## 📊 Results

### VQA Performance (VizWiz-VQA test set, VQAv2 test-standard server)
Mean ± std over 3 seeds (42, 123, 7) for LoRA/QLoRA. Base is the off-the-shelf PaliGemma `mix` checkpoint.

| Model                    | VizWiz-VQA        | VQAv2             | Memory Footprint (GB) |
|--------------------------|--------------------|--------------------|------------------------|
| PaliGemma-3B (`mix`)     | 73.95%             | 81.65%             | 5.45\*                 |
| PaliGemma-3B + LoRA      | 75.80% ± 0.21      | 81.15% ± 0.04      | 5.45                   |
| **PaliGemma-3B + QLoRA** | **75.71% ± 0.14**  | **80.72% ± 0.04**  | **2.13**               |

\*Base and LoRA share the same bf16 backbone footprint (LoRA freezes the base weights); only the adapter differs.

### Captioning Performance (VizWiz-Caps validation set)
| Model                    | CIDEr-D            | BLEU-4             | METEOR             | ROUGE-L            | Avg. Length |
|--------------------------|---------------------|--------------------|--------------------|---------------------|-------------|
| PaliGemma-3B (`mix`)     | 55.31               | 12.07              | 14.08              | 28.76               | 5.03        |
| PaliGemma-3B + LoRA      | 98.08 ± 1.51        | 30.77 ± 0.36       | 23.45 ± 0.31       | 49.54 ± 0.47        | 10.30       |
| **PaliGemma-3B + QLoRA** | **97.44 ± 1.45**    | **30.63 ± 0.29**   | **23.30 ± 0.35**   | **49.33 ± 0.38**    | **10.24**   |

### Captioning Performance (COCO-Caps validation set — general-domain control)
| Model                    | CIDEr-D             | BLEU-4             | METEOR             | ROUGE-L             | Avg. Length |
|--------------------------|----------------------|--------------------|--------------------|----------------------|-------------|
| PaliGemma-3B (`mix`)     | 131.21               | 31.96              | 30.62              | 59.17                | 12.4        |
| PaliGemma-3B + LoRA      | 124.54 ± 1.36        | 34.58 ± 0.39       | 30.63 ± 0.06       | 58.62 ± 0.19         | 11.20       |
| **PaliGemma-3B + QLoRA** | **123.34 ± 1.75**    | **34.26 ± 0.60**   | **30.44 ± 0.09**   | **58.39 ± 0.28**     | **11.22**   |

General-domain performance declines only modestly (≤6% CIDEr-D) relative to the large VizWiz-domain gains, while BLEU-4 actually *improves* over the base model on COCO-Caps.

### Deployment Metrics (NVIDIA Jetson Orin Nano 8GB, 15W mode, clocks locked)
Only the 4-bit QLoRA model is deployed on-device; the bf16 model (5.45 GB) does not reliably fit in the device's available memory and is not used for edge inference.

- **Static Memory Footprint**: 2.13 GB (2.25 GB peak during inference)
- **System Memory Utilization**: 4.15 GB of 7.44 GB total, leaving ~3.0 GB headroom
- **Inference Latency**: 0.93 s (VQA, 30 tokens) / 1.97 s (captioning, 64 tokens) — device-side compute; 1.36–2.41 s including camera capture and TTS synthesis
- **Power Consumption**: 9.60W (active inference), 4.14W (idle, model loaded)
- **Energy per Query**: 15.8 J (64-token caption query)
- **Battery Life**: ~24 hours under realistic use (10 queries/hour, 100Wh pack); 10.4 hours under continuous inference

---

## 🖥️ Hardware Requirements

### Training
- **GPU**: NVIDIA A100 (40GB+) or V100 (32GB+)
- **RAM**: 32 GB+
- **Storage**: 200 GB+ (dataset + checkpoints)

### Deployment
- **Device**: NVIDIA Jetson Orin Nano 8GB (4-bit QLoRA model only)
- **Memory**: 8 GB unified (2.13 GB static footprint; ~4.15 GB total system utilization during inference)
- **Power**: 15W mode recommended, clocks locked (`jetson_clocks`)
- **Accessories**: USB camera, microphone, speaker
---

<!-- ## 📖 Documentation

- **[Training Guide](docs/TRAINING.md)**: Detailed training instructions and hyperparameter tuning
- **[Deployment Guide](docs/DEPLOYMENT.md)**: Step-by-step Jetson setup and deployment
- **[Troubleshooting](docs/TROUBLESHOOTING.md)**: Common issues and solutions
- **[API Reference](docs/API.md)**: Code documentation and examples

---  -->

## 🛠️ Repository Structure

```
├── configs/           # Training and deployment configurations
├── data/              # Dataset download and preprocessing
├── src/               # Core source code (model, trainer, metrics)
├── scripts/           # Training and evaluation scripts
├── deployment/        # Jetson deployment code
├── notebooks/         # Jupyter notebooks for analysis
├── docs/              # Detailed documentation
└── examples/          # Sample inputs and outputs
```

<!-- See [TRAINING.md](docs/TRAINING.md) for detailed training instructions and [DEPLOYMENT.md](docs/DEPLOYMENT.md) for edge deployment guide.
 -->

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

