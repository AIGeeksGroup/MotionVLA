# MotionVLA: End-to-End Vision-Language-Action Motion Generation

This is the official repository for the paper:
> **MotionVLA: End-to-End Vision-Language-Action Motion Generation with Dual-Stream FAST Tokenization**
>
> AIGeeksGroup
>
> \*Equal contribution. †Project lead. <sup>#</sup>Corresponding author.
>
> ### [Paper]() | [Website](https://aigeeksgroup.github.io/MotionVLA/) | [HuggingFace](https://huggingface.co/AIGeeksGroup/MotionVLA)

> [!NOTE]
> 💪 MotionVLA generates expressive human motion end-to-end from visual and language inputs by combining **Qwen3.5-VL** perception with a **Dual-Stream FAST Tokenizer** (DS-FAST) and **LoRA fine-tuning**, achieving strong performance on motion generation benchmarks.

## ✏️ Citation
If you find our code or paper helpful, please consider starring ⭐ us and citing:
```bibtex
@article{motionvla2026,
  title={MotionVLA: End-to-End Vision-Language-Action Motion Generation with Dual-Stream FAST Tokenization},
  author={AIGeeksGroup},
  year={2026}
}
```
---

## 🤸 Introduction to MotionVLA

Vision-Language Models (VLMs) have made impressive progress on multimodal understanding, yet directly mapping visual and textual context into **continuous human motion** remains challenging: motion is high-dimensional, temporally structured, and spans both semantic intent (what to do) and physical dynamics (how it moves). Existing motion generators typically tokenize the full pose stream as one sequence, conflating these two modes and limiting controllability.

To address this, we propose **MotionVLA**, an end-to-end Vision-Language-Action framework that pairs a frozen **Qwen3.5-VL** backbone with a **Dual-Stream FAST Tokenizer (DS-FAST)** and a lightweight **Mixture-of-Experts (MoE)** residual layer. DS-FAST decomposes 276-dim motion into a **Base stream** (joint rotations, positions, root orientation — 201-dim, semantic intent) and a **Phys stream** (joint/root velocities — 75-dim, physical dynamics) via DCT, allowing each stream to be quantized with its own vocabulary and predicted by its own head. LoRA fine-tuning over Qwen3.5-VL keeps training efficient while a T5 decoder path is also supported for ablation.

The result is a single model that can be conditioned on **images + text instructions** and produce coherent, controllable motion sequences, with clean separation between *intent* and *dynamics*.

### Key Features

- **End-to-End VLA Pipeline**: Image + text → Qwen3.5-VL → motion tokens → motion, all in one model
- **Dual-Stream FAST Tokenizer**: Frequency-domain decomposition of motion into Base (semantic) and Phys (dynamic) streams
- **MoE Residual Layer**: 8-expert MoE on top of the decoder for capacity without full fine-tuning cost
- **Two Training Paths**: Qwen LoRA (CUDA) and Qwen3.5-VL + T5 decoder (MPS/Mac), plus an `ms-swift` recipe
- **Two-Phase Training**: Phase 1 embed warmup, Phase 2 LoRA SFT — stable and reproducible

![architecture](./figs/overview.png)

## 📰 News

<b>2026/06/01:</b> 🔔 Project website is live at [aigeeksgroup.github.io/MotionVLA](https://aigeeksgroup.github.io/MotionVLA/).

<b>2026/06/01:</b> 📌 Code, models, and dataset are now available on [HuggingFace](https://huggingface.co/AIGeeksGroup/MotionVLA).

## 📋 TODO List

> [!IMPORTANT]
> We are actively developing and improving MotionVLA. Stay tuned for updates!

- [x] Release MotionVLA training and inference code
- [x] Release DS-FAST tokenizer and training scripts
- [x] Release `ms-swift` training recipe (Phase 1 + Phase 2)
- [ ] Upload paper to arXiv and finalize project page
- [ ] Release pre-trained MotionVLA checkpoints on HuggingFace
- [ ] Release MotionVLA dataset on HuggingFace
- [ ] Add interactive demo on HuggingFace Spaces
- [ ] Release motion visualization tools

## 🏗️ Architecture

```
Image + Text → Qwen3.5-VL (frozen)
                   ↓ hidden_states[-2]
          VisualFeatureResampler (Conv1d + interpolate)
                   ↓ encoder_hidden_states (B, 256, 768)
          T5 Decoder (full fine-tuning) / Qwen LoRA
          MotionMoELayer (MoE residual, 8 experts)
                   ↓
     Base Head (4096 vocab) | Phys Head (2048 vocab)
```

The DS-FAST tokenizer decomposes 276-dim motion into two streams via DCT:
- **Base (201-dim)**: joint rotations + positions + root orientation → semantic intent
- **Phys (75-dim)**: joint/root velocities → physical dynamics

## 📁 Repository Structure

```
MotionVLA/
├── model/
│   ├── motion_qwen.py        # Qwen3.5-VL + LoRA + MoE (main training path)
│   └── motion_vla.py         # Qwen3.5-VL + T5 decoder + MoE
├── dataset/
│   ├── motion_qwen_dataset.py  # Dataset for Qwen LoRA training
│   └── motion_vla_dataset.py   # Dataset for T5 training
├── tokenizer/
│   ├── ds_fast_tokenizer.py    # DS-FAST dual-stream tokenizer core
│   ├── train_tokenizer.py      # Train tokenizer from raw 276-dim data
│   ├── tokenize_dataset.py     # Batch tokenize a dataset
│   └── 276to263/               # 276-dim ↔ 263-dim conversion tools
├── training/
│   ├── train_qwen.py           # Qwen LoRA trainer (CUDA, recommended)
│   ├── train_t5.py             # T5 decoder trainer (MPS/Mac)
│   ├── warmup_embed.py         # Phase 1: embed warmup
│   ├── prepare_swift_data.py   # Convert dataset to ms-swift JSONL format
│   ├── train_swift_phase1.sh   # ms-swift Phase 1: embed warmup
│   ├── train_swift_phase2.sh   # ms-swift Phase 2: LoRA SFT
│   └── train_swift_h100.sh     # Combined H100 training script
├── analysis/                   # Frequency analysis scripts
├── theory/                     # Theoretical analysis (DCT, MoE, tokenizer)
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATA_FORMAT.md
│   └── TRAINING.md
└── README.md                   # This file
```

## ⚡ Quick Start

### Environment Setup

Our code is tested with CUDA 11.8 and Python 3.10. To run the codes, first install the required packages:

```bash
# Create conda environment
conda create -n motionvla python=3.10
conda activate motionvla

# Install PyTorch
pip install torch>=2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

Key dependencies (see [requirements.txt](./requirements.txt)):
- `torch>=2.1.0`
- `transformers>=4.45.0`
- `peft>=0.12.0`
- `ms-swift>=2.0.0`
- `qwen-vl-utils`

### Data Preparation

#### Download MotionVLA Models & Data

Pretrained models, the DS-FAST tokenizer, and the dataset are hosted on [HuggingFace](https://huggingface.co/AIGeeksGroup/MotionVLA):

```bash
# MotionVLA model & tokenizer
huggingface-cli download AIGeeksGroup/MotionVLA --local-dir checkpoints/MotionVLA

# Qwen3.5-VL backbone
huggingface-cli download Qwen/Qwen3.5-VL-2B-Instruct --local-dir checkpoints/Qwen3.5-VL-8B

# T5 motion decoder (for the T5 training path)
huggingface-cli download wbz0505/t2m-ft-from-GSPretrained-base \
    --local-dir checkpoints/t2m-ft-from-GSPretrained-base
```

For the dataset JSON format and motion file layout, see [`docs/DATA_FORMAT.md`](./docs/DATA_FORMAT.md).

## 🔧 DS-FAST Tokenizer Training

To train the DS-FAST tokenizer from raw 276-dim motion data:

```bash
python tokenizer/train_tokenizer.py \
    --motiondata_root data/motions \
    --output_dir      tokenizer/checkpoints \
    --K_base 5 --K_phys 15 \
    --base_vocab 4096 --phys_vocab 2048
```

This produces dual-stream codebooks (`base_vocab=4096`, `phys_vocab=2048`) used by both training paths.

## 💻 Training

### Recommended: ms-swift Pipeline

The recommended training pipeline uses [ms-swift](https://github.com/modelscope/ms-swift) and runs in two phases.

**Step 1: Prepare data**
```bash
python training/prepare_swift_data.py \
    --json  data/dataset.json \
    --root  . \
    --out   data/swift
```

**Step 2: Embed warmup (Phase 1, ~500 steps)**
```bash
bash training/train_swift_phase1.sh
```

**Step 3: LoRA SFT (Phase 2)**
```bash
bash training/train_swift_phase2.sh
```

**Combined H100 script:**
```bash
bash training/train_swift_h100.sh
```

### Custom PyTorch Training

```bash
# Qwen LoRA (CUDA)
python training/train_qwen.py \
    --model_path checkpoints/Qwen3.5-VL-8B \
    --json_path  data/dataset.json \
    --epochs 30

# T5 decoder (MPS / Mac)
python training/train_t5.py \
    --qwen_model_path checkpoints/Qwen3.5-VL-8B \
    --t5_model_path   checkpoints/t2m-ft-from-GSPretrained-base \
    --json_path       data/dataset.json \
    --epochs 50
```

For training hyperparameters and detailed configuration, see [`docs/TRAINING.md`](./docs/TRAINING.md).

## 📊 Evaluation

Evaluation scripts and metrics for motion generation (FID, R-Precision, MM-Dist, Diversity) will be released alongside the pre-trained checkpoints. See the project [website](https://aigeeksgroup.github.io/MotionVLA/) for the latest benchmark numbers.

## 🎯 Use Cases

MotionVLA can be applied to a wide range of motion-generation tasks:

### 1. Animation & Content Creation
- Text-to-motion for previsualization
- Image-conditioned motion synthesis for game / film
- Rapid prototyping of character behaviors

### 2. Robotics & Embodied AI
- Vision-language conditioned motion priors for humanoid policies
- Motion retargeting from instruction
- Behavior libraries for simulation

### 3. AR / VR & Interactive Media
- Avatar animation from natural-language prompts
- Real-time motion-from-instruction for virtual characters
- Motion-driven storytelling

### 4. Research
- Studying disentanglement of semantic intent vs. physical dynamics
- Frequency-domain motion tokenization
- VLM-conditioned action generation

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=AIGeeksGroup/MotionVLA&type=Date)](https://www.star-history.com/#AIGeeksGroup/MotionVLA&Date)

## 🤝 Contributing

We welcome contributions to MotionVLA! Please feel free to:
- Report bugs and issues
- Submit pull requests
- Suggest new features
- Share your results and applications

## 📄 License

This project is released under the MIT License. See [LICENSE](./LICENSE) for details.

## 😘 Acknowledgement

We thank the authors of the following projects for their open-source contributions:
- [Qwen](https://github.com/QwenLM/Qwen) for the Qwen3.5-VL backbone
- [MS-SWIFT](https://github.com/modelscope/ms-swift) for the training framework
- [PEFT](https://github.com/huggingface/peft) for LoRA implementations
- [T5](https://github.com/google-research/text-to-text-transfer-transformer) for the decoder backbone
- The motion-generation research community for datasets and prior work on motion tokenization

## 📧 Contact

For questions and discussions, please:
- Open an issue on GitHub
- Visit our [project website](https://aigeeksgroup.github.io/MotionVLA/)
- Browse our models and datasets on [HuggingFace](https://huggingface.co/AIGeeksGroup/MotionVLA)
