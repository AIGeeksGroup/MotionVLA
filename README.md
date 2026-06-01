# MotionVLA

End-to-end motion generation from vision and language, combining **Qwen3.5-VL** (visual-language perception) with a **Dual-Stream FAST Tokenizer** (DS-FAST) and **LoRA fine-tuning**.

## Architecture

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

## Models & Data

Pretrained models and datasets are hosted on HuggingFace:
- **Model**: `[your-hf-username]/MotionVLA`
- **Dataset**: `[your-hf-username]/MotionVLA-Dataset`
- **Tokenizer**: `[your-hf-username]/MotionVLA-Tokenizer`

## Setup

```bash
pip install -r requirements.txt
```

Download models from HuggingFace into `checkpoints/`:
```bash
# Qwen3.5-VL backbone
huggingface-cli download Qwen/Qwen3.5-VL-2B-Instruct --local-dir checkpoints/Qwen3.5-VL-8B

# T5 motion decoder (for T5 training path)
huggingface-cli download wbz0505/t2m-ft-from-GSPretrained-base --local-dir checkpoints/t2m-ft-from-GSPretrained-base

# DS-FAST tokenizer checkpoints
huggingface-cli download [your-hf-username]/MotionVLA-Tokenizer --local-dir tokenizer/checkpoints
```

## Project Structure

```
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
└── docs/
    ├── ARCHITECTURE.md
    ├── DATA_FORMAT.md
    └── TRAINING.md
```

## Training (ms-swift, recommended)

The recommended training pipeline uses [ms-swift](https://github.com/modelscope/ms-swift):

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

For H100 (combined):
```bash
bash training/train_swift_h100.sh
```

## Training (custom PyTorch)

```bash
# Qwen LoRA (CUDA)
python training/train_qwen.py \
    --model_path checkpoints/Qwen3.5-VL-8B \
    --json_path  data/dataset.json \
    --epochs 30

# T5 decoder (MPS/Mac)
python training/train_t5.py \
    --qwen_model_path checkpoints/Qwen3.5-VL-8B \
    --t5_model_path   checkpoints/t2m-ft-from-GSPretrained-base \
    --json_path       data/dataset.json \
    --epochs 50
```

## Tokenizer Training

To train the DS-FAST tokenizer from raw 276-dim motion data:

```bash
python tokenizer/train_tokenizer.py \
    --motiondata_root data/motions \
    --output_dir      tokenizer/checkpoints \
    --K_base 5 --K_phys 15 \
    --base_vocab 4096 --phys_vocab 2048
```

## Data Format

See `docs/DATA_FORMAT.md` for the dataset JSON format and motion file structure.

## License

[LICENSE](LICENSE)
