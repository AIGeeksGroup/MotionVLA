# MotionVLA: Vision-Language-Action Model for Humanoid Motion

This is the official repository for the paper:
> **MotionVLA: Vision-Language-Action Model for Humanoid Motion**
>
> [Nonghai Zhang](https://github.com/sleepyDogseasea)\*, [Siyu Zhai](https://github.com/almightyfish)\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*, and [Hao Tang](https://ha0tang.github.io/)<sup>#</sup>
>
> \*Equal contribution. †Project lead. <sup>#</sup>Corresponding author.
>
> ### [Paper]() | [Website](https://aigeeksgroup.github.io/MotionVLA/) | [HuggingFace](https://huggingface.co/AIGeeksGroup/MotionVLA)

> [!NOTE]
> 💪 MotionVLA generates humanoid motion from a scene image and a text instruction by combining a **Qwen3.5** autoregressive backbone with **DSFT (Dual-Stream Frequency-domain Tokenizer)**, which decouples low-frequency pose semantics from high-frequency physical dynamics.

## ✏️ Citation
If you find our code or paper helpful, please consider starring ⭐ us and citing:
```bibtex
@article{motionvla2026,
  title={MotionVLA: Vision-Language-Action Model for Humanoid Motion},
  author={Zhang, Nonghai and Zhai, Siyu and Zhang, Zeyu and Tang, Hao},
  year={2026}
}
```
---

## 🤸 Introduction to MotionVLA

Generating realistic humanoid motion from scene images and text involves both **low-frequency pose semantics** and **high-frequency physical dynamics**. Many existing methods tokenize motion with a single shared codebook, forcing heterogeneous motion signals into the same quantization space. Our frequency-domain analysis of human motion shows a clear mismatch: five DCT coefficients capture **93%** of joint-position energy but only **37%** of joint-velocity energy, biasing single-codebook quantization toward pose statistics and under-representing high-frequency velocity components.

To address this, we propose **DSFT (Dual-Stream Frequency-domain Tokenizer)**, which separates motion into a **Base** stream (joint rotations + positions + root orientation/coordinates) and a **Phys** stream (joint velocities + root velocities), and compresses them independently with **DCT truncation + BPE**. We then present **MotionVLA**, a **Qwen3.5**-based autoregressive model that arranges the two streams in a unified sequence

```
[ M_BOS, b_1, …, b_N, M_SEP, p_1, …, p_M, M_EOS ]
```

so that Phys tokens are predicted **after** all Base tokens through causal attention — a hierarchical *semantic-to-physical* generation order.

Despite using a lightweight 2B backbone, MotionVLA reduces the Diversity gap to real data by **over 50%** on HumanML3D and improves Motion-Condition Consistency by **3.8%** on MBench, supporting frequency-aware dual-stream decoupling as an effective formulation for autoregressive motion generation.

### Key Features

- **DSFT Tokenizer**: Frequency-domain decomposition of motion into Base (semantic) and Phys (dynamic) streams with independent DCT truncation (default `K_base=5`, `K_phys=25`) and BPE compression.
- **Qwen3.5 VLA Backbone**: A standard autoregressive transformer extended with motion tokens, supporting scene-image + text conditioning.
- **Unified Sequence**: Base-then-Phys layout enables phase-aware causal generation; a logit mask enforces the BASE → SEP → PHYS → EOS order at inference.
- **Two-Phase ms-swift Training**: Phase 1 warms up new motion-token embeddings, Phase 2 runs LoRA SFT on the full backbone.
- **Lightweight Default**: 2B backbone is sufficient for SOTA-competitive results; ablations cover 0.8B / 2B / 4B / 9B.

## 📰 News

<b>2026/06/01:</b> 🔔 Project website is live at [aigeeksgroup.github.io/MotionVLA](https://aigeeksgroup.github.io/MotionVLA/).

<b>2026/06/01:</b> 📌 Code is released; models and tokenizer checkpoints will be uploaded to [HuggingFace](https://huggingface.co/AIGeeksGroup/MotionVLA) shortly.

## 📋 TODO List

> [!IMPORTANT]
> We are actively developing and improving MotionVLA. Stay tuned for updates!

- [x] Release MotionVLA training code (ms-swift two-phase pipeline)
- [x] Release DSFT tokenizer training & inference code
- [x] Release dual-stream frequency analysis scripts
- [ ] Upload paper to arXiv and finalize project page
- [ ] Release pre-trained MotionVLA checkpoints (2B / 4B / 9B) on HuggingFace
- [ ] Release the ViMoGen-derived training JSONL on HuggingFace
- [ ] Release motion visualization & MuJoCo simulation toolkit
- [ ] Add interactive demo on HuggingFace Spaces

## 🏗️ Architecture

```
   Scene Image  +  Text Instruction
                 │
                 ▼
        Qwen3.5 (default 2B)              ┌── new motion vocabulary ──┐
   Vocabulary V = V_LM + V_motion + 3     │  Base : 4096 tokens       │
                                          │  Phys : 4096 tokens       │
                                          │  M_BOS / M_SEP / M_EOS    │
                                          └───────────────────────────┘
                 │
                 ▼   masked next-token prediction
   [ M_BOS, b_1, …, b_N, M_SEP, p_1, …, p_M, M_EOS ]
                 │
                 ▼   phase-aware logit mask at inference
              BPE⁻¹ + IDCT  (per-stream)
                 │
                 ▼
         Reconstructed motion ∈ ℝ^{T × D}
```

DSFT decomposes motion into two streams via DCT (HumanML3D 263-dim → Base 190 / Phys 73; ViMoGen 276-dim → Base 201 / Phys 75):

- **Base stream**: joint rotations + positions + root orientation/coordinates → low-frequency pose semantics (~86–93% of energy in the first 5 DCT coefficients).
- **Phys stream**: joint velocities + root velocities → high-frequency physical dynamics (only ~37% energy in the first 5 coefficients).

Each stream is DCT-truncated (`K_base=5`, `K_phys=25`) and BPE-encoded independently, yielding two compact discrete vocabularies.

## 📁 Repository Structure

```
MotionVLA/
├── tokenizer/
│   ├── ds_fast_tokenizer.py    # DSFT dual-stream tokenizer (DSFT class)
│   ├── train_tokenizer.py      # Train DSFT from raw motion data
│   ├── tokenize_dataset.py     # Batch tokenize a motion dataset
│   └── 276to263/               # 276-dim ↔ 263-dim conversion utilities
├── training/
│   ├── prepare_swift_data.py   # Convert tokenized dataset → ms-swift JSONL
│   ├── train_swift_phase1.sh   # Phase 1: motion-token embed warmup
│   ├── train_swift_phase2.sh   # Phase 2: LoRA SFT
│   └── train_swift_h100.sh     # Combined Phase 1 + Phase 2 (H100)
├── analysis/                   # Frequency analysis & DSFT reconstruction quality
│   ├── freq_analysis_combined.py
│   └── dsfast/                 # Per-dim low-freq ratio, energy coverage, rRMSE
├── theory/                     # Supporting theoretical analyses
│   ├── theory1_dualstream.py
│   ├── theory3_tokenizer.py
│   └── theory4_dualstream_vs_single.py
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATA_FORMAT.md
│   └── TRAINING.md
├── requirements.txt
└── README.md                   # This file
```

## ⚡ Quick Start

### Environment Setup

Tested with CUDA 11.8 / 12.x and Python 3.10:

```bash
conda create -n motionvla python=3.10
conda activate motionvla

# PyTorch (pick the build matching your CUDA)
pip install torch>=2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118

# Other dependencies
pip install -r requirements.txt
```

Key dependencies (see [requirements.txt](./requirements.txt)):

- `torch>=2.1.0`
- `transformers>=4.45.0`
- `peft>=0.12.0`
- `ms-swift>=2.0.0`
- `qwen-vl-utils`
- `tokenizers`, `scipy`

### Download Models

```bash
# Qwen3.5 backbone (replace size with the one you want to use)
huggingface-cli download <qwen3.5-checkpoint-name> --local-dir checkpoints/Qwen3.5-VL-8B

# (Once released) DSFT tokenizer & MotionVLA checkpoints
huggingface-cli download AIGeeksGroup/MotionVLA --local-dir checkpoints/MotionVLA
```

### Data Preparation

The dataset JSON format and motion `.pt` layout are described in [`docs/DATA_FORMAT.md`](./docs/DATA_FORMAT.md). At a high level:

1. Each entry has `id`, `text`, `motion_path`, optional `image_path`.
2. Motion `.pt` files contain a tokenized sequence `seq = [BOS, base…, SEP, phys…, EOS]` produced by DSFT.

## 🔧 DSFT Tokenizer

Train DSFT on raw motion data:

```bash
python tokenizer/train_tokenizer.py \
    --motiondata_root data/motions \
    --output_dir      tokenizer/checkpoints \
    --K_base 5 --K_phys 25 \
    --base_vocab 4096 --phys_vocab 2048
```

Then tokenize a dataset:

```bash
python tokenizer/tokenize_dataset.py \
    --json       data/dataset.json \
    --motiondata data/motions \
    --tok_dir    tokenizer/checkpoints \
    --out_dir    data/motions_tokenized \
    --out_json   data/dataset_tokenized.json \
    --workers    4
```

## 💻 Training (ms-swift, two-phase)

The official training pipeline uses [ms-swift](https://github.com/modelscope/ms-swift) and runs in two phases.

**Step 1 — Prepare ms-swift JSONL**

```bash
python training/prepare_swift_data.py \
    --json   data/dataset_tokenized.json \
    --root   . \
    --out    data/swift \
    --split  0.9
```

This writes `train.jsonl`, `val.jsonl`, and `motion_tokens.txt` (the new motion-token vocabulary) into `data/swift/`.

**Step 2 — Phase 1: Embed warmup**

Freeze all transformer layers; train only `embed_tokens` and `lm_head` rows for the new motion tokens (LR `1e-3`, ~500 steps):

```bash
bash training/train_swift_phase1.sh
```

Outputs land under `checkpoints/phase1_embed/`.

**Step 3 — Phase 2: LoRA SFT**

Load the Phase 1 checkpoint and run LoRA SFT (`rank=32`, `alpha=64`, `target_modules=all-linear`, `LR=2e-4`, 3 epochs):

```bash
bash training/train_swift_phase2.sh
```

Outputs land under `checkpoints/swift_lora/`.

**Combined H100 recipe** (Phase 1 + Phase 2 in one script):

```bash
bash training/train_swift_h100.sh
```

For configuration details and hyper-parameters see [`docs/TRAINING.md`](./docs/TRAINING.md).

## 📊 Benchmark Results

### MBench (scene-conditioned, ViMoGen-228K)

| Method | M-C Cons. ↑ | M-Gen. ↑ | Jitter ↓ | Dynamic ↑ | F-Float ↓ | F-Slide ↓ | B-Pen ↓ | P-Qual ↓ |
|---|---|---|---|---|---|---|---|---|
| MDM (ICLR'23) | 0.42 | 0.51 | 0.0136 | 0.0376 | 0.156 | 0.0136 | 1.68 | 2.67 |
| T2M-GPT (CVPR'23) | 0.39 | 0.38 | 0.0156 | 0.0349 | 0.209 | 0.0156 | 1.33 | 2.43 |
| MoMask (CVPR'24) | 0.38 | 0.44 | 0.0147 | 0.0396 | 0.178 | 0.0147 | 1.48 | 2.67 |
| MotionDiffuse (TPAMI'24) | 0.44 | 0.42 | 0.0111 | 0.0289 | 0.126 | 0.0063 | 1.35 | 2.21 |
| MotionLCM (ECCV'24) | 0.48 | 0.55 | 0.0218 | 0.0439 | 0.193 | 0.0202 | 1.73 | 2.40 |
| FineMoGen (NeurIPS'24) | 0.37 | 0.42 | 0.0118 | 0.0386 | 0.281 | 0.0091 | 1.18 | 2.28 |
| MotionCraft (CVPR'25) | 0.42 | 0.45 | 0.0132 | 0.0420 | 0.402 | 0.0090 | 1.15 | 2.12 |
| ViMoGen (ICLR'26) | 0.53 | **0.68** | 0.0108 | 0.0251 | 0.204 | 0.0064 | 1.78 | 2.38 |
| ViMoGen-light (ICLR'26) | 0.47 | 0.55 | 0.0129 | 0.0294 | 0.155 | 0.0051 | 1.43 | **2.10** |
| **MotionVLA (Ours)** † | **0.55** | 0.66 | **0.0110** | 0.0419 | **0.149** | **0.0049** | **1.34** | 2.14 |

†: uses additional visual (scene) input. Best in **bold**.

### HumanML3D (text-to-motion)

| Method | R-Top1 ↑ | R-Top2 ↑ | R-Top3 ↑ | FID ↓ | MM-Dist ↓ | Diversity → | MModality ↑ |
|---|---|---|---|---|---|---|---|
| Real | 0.511 | 0.703 | 0.797 | 0.002 | 2.974 | 9.503 | – |
| TM2T (ECCV'22) | 0.424 | 0.618 | 0.729 | 1.501 | 3.467 | 8.589 | 2.424 |
| MDM (ICLR'23) | – | – | 0.611 | 0.544 | 5.566 | – | 2.799 |
| MotionDiffuse (TPAMI'24) | 0.491 | 0.681 | 0.782 | 0.630 | 3.113 | 9.410 | 1.553 |
| T2M-GPT (CVPR'23) | 0.492 | 0.679 | 0.775 | 0.141 | 3.121 | 9.722 | 1.831 |
| FineMoGen (NeurIPS'24) | 0.504 | 0.690 | 0.784 | 0.151 | 2.998 | 9.263 | 2.696 |
| MoMask (CVPR'24) | 0.521 | 0.713 | 0.807 | 0.045 | 2.958 | – | 1.241 |
| DisCoRD (ICCV'25) | 0.524 | 0.715 | 0.809 | 0.032 | 2.938 | – | 1.288 |
| GenM3 (ICCV'25) | 0.511 | 0.705 | 0.804 | 0.046 | 2.852 | 9.675 | – |
| MG-MotionLLM (CVPR'25) | 0.516 | 0.706 | 0.802 | 0.303 | 2.952 | 9.960 | 2.125 |
| **MotionVLA (Ours)** | 0.507 | 0.699 | 0.798 | 0.071 | 2.906 | **9.548** | **2.821** |

MotionVLA achieves the Diversity score closest to real data and the highest MModality among generated methods, despite using only a 2B backbone.

### Backbone Scale Ablation (MBench)

| Backbone | Params | M-C Cons. ↑ | M-Gen. ↑ | Jitter ↓ | F-Slide ↓ |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | 0.8B | 0.51 | 0.60 | 0.0122 | 0.0058 |
| **Qwen3.5-2B (default)** | 2B | 0.55 | 0.66 | 0.0110 | 0.0049 |
| Qwen3.5-4B | 4B | 0.55 | 0.66 | 0.0109 | 0.0049 |
| Qwen3.5-9B | 9B | 0.56 | 0.68 | 0.0107 | 0.0047 |

## 🎯 Use Cases

- **Animation & content creation**: Text- and image-conditioned humanoid motion for previs, games, and film.
- **Robotics & embodied AI**: Vision-language motion priors for humanoid policies; deployment on platforms such as Unitree G1.
- **AR / VR & interactive media**: Avatar animation from natural-language prompts and scene context.
- **Research**: Studying disentanglement of semantic intent vs. physical dynamics, and frequency-aware motion tokenization.

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=AIGeeksGroup/MotionVLA&type=Date)](https://www.star-history.com/#AIGeeksGroup/MotionVLA&Date)

## 🤝 Contributing

We welcome contributions. Please feel free to:
- Report bugs and issues
- Submit pull requests
- Suggest new features
- Share your results and applications

## 📄 License

This project is released under the MIT License. See [LICENSE](./LICENSE) for details.

## 😘 Acknowledgement

We thank the authors of the following projects for their open-source contributions:
- [Qwen](https://github.com/QwenLM/Qwen) for the Qwen3.5 backbone
- [ms-swift](https://github.com/modelscope/ms-swift) for the training framework
- [PEFT](https://github.com/huggingface/peft) for LoRA implementations
- [HumanML3D](https://github.com/EricGuo5513/HumanML3D) and ViMoGen for datasets and benchmarks
- The motion-generation research community for prior work on motion tokenization

## 📧 Contact

For questions and discussions, please:
- Open an issue on GitHub
- Visit our [project website](https://aigeeksgroup.github.io/MotionVLA/)
- Browse our models on [HuggingFace](https://huggingface.co/AIGeeksGroup/MotionVLA)
