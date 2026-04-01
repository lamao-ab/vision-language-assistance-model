# PaliGemma + LoRA/QLoRA Blind Assist
Optimized for Jetson Orin Nano.

# PaliGemma + LoRA/QLoRA Blind Assistance: Edge-Deployable Vision-Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-red.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/lamao-ab)

**Parameter-efficient fine-tuning of PaliGemma-3B for blind assistance using LoRA/QLoRA on VizWiz dataset, deployable on NVIDIA Jetson Orin Nano 8GB.**
![demo](https://github.com/user-attachments/assets/5cc68256-1692-4c09-9d19-89f99bf1757c)

📄 **Paper**: [Link to ArXiv/IEEE](YOUR_PAPER_LINK)  
🤗 **Models**: [HuggingFace Hub](https://huggingface.co/lamao-ab)  
---
## 🎥 Demo 
### Video Demo

https://github.com/user-attachments/assets/ad495a73-14f1-4582-98b1-c1281f91f399

### Demo with images

    vqa_prompt = "Assist a blind person: List all the objects you see in this image."
    cap_prompt = "Describe this scene for a blind person."
    
  VQA: Building, House, Tree.
  CAP: A woman sitting on a bench in front of a building.
  ![demo1](https://github.com/user-attachments/assets/68443941-abf0-4278-93bd-c3d695eab8a7)

  VQA: Bench, Furniture, Plant, Tree.
  CAP: Two park benches are in a park with a fire hydrant in the background.
  ![demo](https://github.com/user-attachments/assets/b7fec2f4-59f9-42aa-b7b6-f9c5af697c62)

  VQA: Car, Sky, Stop sign, Vehicle.
  CAP: A stop sign with graffiti on it in a foreign language.
  ![demo3](https://github.com/user-attachments/assets/0d31a0f4-5994-4dfb-9ca9-f72b1ff75993)

    vqa_prompt = "Assist a blind person: How many people in this image?"
    cap_prompt = "Describe this scene for a blind person."

  VQA: 7
  CAP: A group of people are standing at a bus stop.
 ![demo4](https://github.com/user-attachments/assets/1aa621b8-a0ac-40ed-8dc1-29c686bf013c)

  VQA: 1.
  CAP: A bus is parked on the side of the street.
 ![fe54cc512921db99_jpg rf 30136ce3400f26a48da06f249a94878d](https://github.com/user-attachments/assets/f5f3989e-b6bd-4f3e-b7e6-f5466b3194a2)

---

## 🎯 Overview

This repository provides a complete pipeline for training and deploying vision-language models on edge devices for real-time blind assistance. Our approach achieves:

- **75.65% VQA accuracy** on VizWiz-VQA (10-choose-9 metric)
- **0.737 CIDEr-D score** on VizWiz-Captions
- **2.78 GB memory footprint** (QLoRA) enabling Jetson deployment
- **1.3-2.2 seconds inference latency** for interactive assistance
- **12.4+ hours battery life (Continuous)** with 100Wh battery pack

### Key Features

✅ **Parameter-Efficient Fine-Tuning**: Train only 11.3M parameters (0.385% of total) using LoRA  
✅ **4-bit Quantization**: QLoRA reduces memory from ~12 GB to ~3.0 GB  
✅ **Multi-Task Learning**: Joint training on VQA and image captioning  
✅ **Edge Deployment**: Run on affordable hardware (Jetson Orin Nano 8GB)  
✅ **Privacy-Preserving**: All processing on-device, no cloud required  
✅ **Open Source**: All code, configs, and trained weights available  

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/lamao-ab/paligemma-qlora-blind-assist.git
cd paligemma-qlora-blind-assistance

# Create conda environment
conda create -n paligemma python=3.10
conda activate paligemma
```

### Download Dataset

```bash
# Download and prepare VizWiz dataset
python data/download_vizwiz.py --output_dir ./datasets/vizwiz
python data/prepare_dataset.py --data_dir ./datasets/vizwiz
```

### Training

**LoRA (Full Precision):**
```bash
python scripts/train_lora.py \
    --config configs/training_config.yaml \
    --output_dir ./checkpoints/lora \
    --epochs 2
```

**QLoRA (4-bit Quantized):**
```bash
python scripts/train_qlora.py \
    --config configs/training_config.yaml \
    --output_dir ./checkpoints/qlora \
    --epochs 3
```

### Inference 

```bash
# Single image inference
python scripts/inference.py \
    --model_path ./checkpoints/qlora/final \
    --image_path ./examples/images/sample.jpg \
    --question "What color is this object?"
```

### Deployment on Jetson Nano Orin 8GB

```bash
# Interactive demo on Jetson Nano Orin 8GB
cd paligemma-qlora-blind-assistance/deployment
pip install -r requirements.txt
python deployment/python3 blind-assistance-system.py 
```


## 📊 Results

### VQA Performance (VizWiz-VQA Test-Standard Server)

| Model | Method | VQA Accuracy | Memory (GB) | Latency (s) |
|-------|--------|--------------|-------------|-------------|
| PaliGemma-3B | Zero-shot | 73.89% | 5.65 | 1.0 |
| PaliGemma-3B | LoRA | 75.89% | 5.65 | 1.2 |
| **PaliGemma-3B** | **QLoRA** | **76.45%** | **2.78** | **1.5** |

### Captioning Performance (VizWiz-Captions Validation Set)

| Model | CIDEr-D | BLEU-4 | METEOR | ROUGE-L | Avg. Length |
|-------|---------|--------|--------|---------|-------------|
| Zero-shot | 0.405 | 0.028 | 0.145 | 0.216 | 5.3 |
| LoRA | 0.758 | 0.090 | 0.275 | 0.274 | 10.3 |
| **QLoRA** | **0.737** | **0.072** | **0.274** | **0.294** | **10.3** |

### Deployment Metrics (Jetson Orin Nano 8GB)

- **Peak Memory**: ~3.6 GB (QLoRA) vs ~6.0 GB (LoRA)
- **Inference Latency**: 1.3-2.3 seconds end-to-end
- **Power Consumption**: 8.05W (active), 0.83W (idle)
- **Battery Life**: 25.6+ hours (100Wh pack, 10 queries/hour)

---

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

See [TRAINING.md](docs/TRAINING.md) for detailed training instructions and [DEPLOYMENT.md](docs/DEPLOYMENT.md) for edge deployment guide.

---

## 🖥️ Hardware Requirements

### Training

- **GPU**: NVIDIA A100 (40GB+) or V100 (32GB+)
- **RAM**: 32 GB+
- **Storage**: 50 GB (dataset + checkpoints)

### Deployment

- **Device**: NVIDIA Jetson Orin Nano 8GB
- **Memory**: 8 GB unified (2.78 GB for model)
- **Power**: 15W mode recommended
- **Accessories**: USB camera, microphone, speaker

---

## 📖 Documentation

- **[Training Guide](docs/TRAINING.md)**: Detailed training instructions and hyperparameter tuning
- **[Deployment Guide](docs/DEPLOYMENT.md)**: Step-by-step Jetson setup and deployment
- **[Troubleshooting](docs/TROUBLESHOOTING.md)**: Common issues and solutions
- **[API Reference](docs/API.md)**: Code documentation and examples

---

## 🎓 Citation

If you use this work in your research, please cite:

```bibtex
@article{yourname2024paligemma,
  title={Parameter-Efficient Adaptation of Vision-Language Models for Edge-Based Blind Assistance},
  author={Your Name and Co-authors},
  journal={IEEE Access},
  year={2024},
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

