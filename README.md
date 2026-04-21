# VtT: Reclaiming Lost Text Layers for Source-Free Cross-Domain Few-Shot Learning

Official implementation for **CVPR 2026** poster paper: *"Reclaiming Lost Text Layers for Source-Free Cross-Domain Few-Shot Learning"*.

## 📋 Overview

This repository presents VtT (teach Vision To learn from Text), a novel method for Source-Free Cross-Domain Few-Shot Learning (SF-CDFSL). 

### Key Features
- **Source-Free**: No access to source domain training data required
- **Cross-Domain**: Handles significant domain shifts between source and target
- **Few-Shot**: Learns from limited labeled examples (1-shot or 5-shot)
- **Vision-to-Text Bridge**: Novel Mamba architecture for cross-modal alignment

## 🛠️ Environment Setup


**Note**: Creating a new environment requires matching CUDA versions. The code was developed with CUDA 11.8.

#### 1. Create Conda Environment

```bash
conda create -n VtT python=3.10 -y
conda activate VtT
```

#### 2. Install PyTorch with CUDA Support

```bash
# For CUDA 11.8
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Install Mamba Dependencies

```bash
pip install mamba-ssm causal-conv1d
```

#### 4. Install Other Dependencies

```bash
pip install -r requirements.txt
```

#### 5. Install CLIP (OpenAI)

```bash
pip install git+https://github.com/openai/CLIP.git
```

## 📦 Datasets

- **EuroSAT**: Satellite imagery land cover classification
- **CropDisease**: Plant disease recognition
- **ISIC**: Skin lesion classification (dermatology)
- **ChestX**: Chest X-ray disease classification

Please prepare your datasets in the appropriate directory structure before running experiments.

## 🚀 Usage

### Training Commands

**For EuroSAT and CropDisease:**

```bash
# 1-shot setting
python main.py --encoder vision --r 16 --alpha 8 --epochs 250 --shot 1 --episodes 800 --dataset EuroSAT

python main.py --encoder vision --r 16 --alpha 8 --epochs 250 --shot 1 --episodes 800 --dataset CropDisease

# 5-shot setting
python main.py --encoder vision --r 16 --alpha 8 --epochs 250 --shot 5 --episodes 400 --dataset EuroSAT

python main.py --encoder vision --r 16 --alpha 8 --epochs 250 --shot 5 --episodes 400 --dataset CropDisease
```

**For ISIC and ChestX:**

```bash
# 1-shot setting
python main.py --encoder vision --r 16 --alpha 8 --epochs 250 --shot 1 --episodes 800 --dataset ISIC --lr 4e-4 --mamba_lr 1e-3

python main.py --encoder vision --r 16 --alpha 8 --epochs 250 --shot 1 --episodes 800 --dataset ChestX --lr 4e-4 --mamba_lr 1e-3

# 5-shot setting
python main.py --encoder vision --r 16 --alpha 8 --epochs 250 --shot 5 --episodes 400 --dataset ISIC --lr 4e-4 --mamba_lr 1e-3

python main.py --encoder vision --r 16 --alpha 8 --epochs 250 --shot 5 --episodes 400 --dataset ChestX --lr 4e-4 --mamba_lr 1e-3
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset name (EuroSAT/CropDisease/ISIC/ChestX) | ISIC |
| `--shot` | Number of support examples per class (1 or 5) | 5 |
| `--way` | Number of classes per episode | 5 |
| `--episodes` | Number of test episodes | 400 |
| `--epochs` | Training epochs per episode | 250 |
| `--encoder` | Which encoder to fine-tune (vision/text/both) | vision |
| `--r` | LoRA rank | 16 |
| `--alpha` | LoRA scaling factor | 8 |
| `--lr` | Learning rate for LoRA parameters | 2e-4 |
| `--mamba_lr` | Learning rate for Mamba parameters | 5e-4 |
| `--beta` | Gradient projection coefficient | 7 |
| `--grad_steps` | Gradient steps for beta update | 50 |

## 🔧 Implementation Details

**Important**: We modified the CLIP text encoder implementation to support an **absorb token** mechanism that replaces the class name token. This is critical for the VtT method to work correctly.

See `/home/zzy/VLM_FSL/VtT/clip/model.py` line 265 onwards for the implementation details.

### Architecture Overview

The model consists of:
1. **CLIP backbone** (ViT-B/16) for feature extraction
2. **LoRA adapters** for efficient fine-tuning
3. **Mamba-based cross-modal bridge** for vision-to-text alignment
4. **Absorb token mechanism** in the text encoder

## 📝 Citation

If you find this work useful for your research, please cite:

```bibtex
@article{zhang2026reclaiming,
  title={Reclaiming Lost Text Layers for Source-Free Cross-Domain Few-Shot Learning},
  author={Zhang, Zhenyu and Chen, Guangyao and Zou, Yixiong and Li, Yuhua and Li, Ruixuan},
  journal={arXiv preprint arXiv:2603.05235},
  year={2026}
}
```

## 🙏 Acknowledgments

This repository is developed based on:
- [CLIP-LoRA](https://github.com/MaxZanella/CLIP-LoRA)
- [Mamba](https://github.com/state-spaces/mamba)
- [StepSPT](https://github.com/xuhuali-mxj/StepSPT)

We thank the authors for their excellent codebases.

## 📄 License

This project is released under the MIT License.

## 📧 Contact

For questions or issues, please open an issue on this repository or contact the authors.