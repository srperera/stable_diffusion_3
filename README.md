# Stable Diffusion 3 (SD3) – From Scratch Implementation

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-green.svg)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-red.svg)

## Overview
This repository provides a **from-scratch implementation of Stable Diffusion 3 (SD3)**, the next-generation diffusion-based generative model for **text-to-image synthesis**. Unlike previous versions, SD3 introduces **Multi-Modal Diffusion Transformers (MMDiT)** for enhanced scalability and performance.

The goal of this project is to **fully re-implement the SD3 architecture**, training pipeline, and inference workflow for **researchers, enthusiasts, and developers** who want to understand the internals of modern diffusion models.

---

## Key Features
- **Full SD3 Architecture** implemented from scratch  
- **Multi-Modal Diffusion Transformer (MMDiT)** backbone  
- **Text & Image Conditioning** via cross-attention  
- **DDPM/DDIM Sampling** for efficient inference  
- **Configurable Training Pipeline** with mixed precision support  
- **Extensible Design** for fine-tuning and future optimizations  

---

## Architecture Highlights
- **Encoder**: Variational Autoencoder (VAE) for latent representation  
- **Conditioning**: Text embeddings from a transformer-based text encoder  
- **Diffusion Backbone**: Multi-Modal DiT blocks with attention  
- **Scheduler**: Discrete noise schedule for forward and reverse processes

## Roadmap
Add LoRA-based fine-tuning for SD3
Provide pre-trained weights for quick experimentation (hopefully get better compute)

---

## 🚀 Quick Start

### ✅ Prerequisites
- Python **3.10+**
- [Jupyter Notebook](https://jupyter.org/)
- Dependencies (auto managed via provided `pyproject.toml` and `uv.lock` files.)

### 📥 Installation
```bash
# Clone the repository
git clone https://github.com/srperera/ImarisParser.git
cd ImarisParser

# Install dependencies
# Option 1: Install UV
wget -qO- https://astral.sh/uv/install.sh | sh

# Option 2: Using pip
pip install uv

# Install dependencices with (after you have installed uv)
uv sync 

# Activate provided virtual environment
source .venv/bin/activate
```

### 📥 Project Structure
```bash
sd3_/
│
├── configs/          # Training & inference configs
├── models/           # Model components (VAE, MMDiT, U-Net)
├── data/             # Data loading & preprocessing scripts
├── scripts/          # Utility scripts (sampling, evaluation)
├── train.py          # Training entry point
├── generate.py       # Inference script
└── README.md         # Project documentation
```

## Citation
``` bibtex
@misc{sd3_from_scratch,
  author = {Shehan Perera},
  title = {Stable Diffusion 3 - From Scratch Implementation},
  year = {2025},
  url = {https://github.com/srperera/sd3_}
}
```
