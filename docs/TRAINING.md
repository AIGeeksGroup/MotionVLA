# MotionVLA Training Guide

This guide walks through the end-to-end training pipeline as implemented in this repository: DSFT tokenizer training → dataset tokenization → ms-swift two-phase fine-tuning of a Qwen3.5 backbone.

There is no T5 path, no MoE branch, and no custom PyTorch trainer — the production training entry points are the three shell scripts under `training/`.

## Contents

1. [Environment](#1-environment)
2. [Data preparation](#2-data-preparation)
3. [DSFT tokenizer](#3-dsft-tokenizer)
4. [Convert to ms-swift JSONL](#4-convert-to-ms-swift-jsonl)
5. [Phase 1 — Embed warmup](#5-phase-1--embed-warmup)
6. [Phase 2 — LoRA SFT](#6-phase-2--lora-sft)
7. [Combined H100 recipe](#7-combined-h100-recipe)
8. [Inference](#8-inference)
9. [FAQ](#9-faq)

---

## 1. Environment

```bash
# Run from the MotionVLA repository root
conda create -n motionvla python=3.10
conda activate motionvla

pip install torch>=2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Key packages: `torch>=2.1.0`, `transformers>=4.45.0`, `peft>=0.12.0`, `ms-swift>=2.0.0`, `qwen-vl-utils`, `tokenizers`, `scipy`.

### Pretrained backbone

| Model | HuggingFace | Local path used by the scripts |
|---|---|---|
| Qwen3.5 backbone (default 2B) | (your chosen Qwen3.5 / Qwen3.5-VL checkpoint) | `checkpoints/Qwen3.5-VL-8B` |

The shell scripts default to `MODEL="checkpoints/Qwen3.5-VL-8B"`. If you use a different size or path, edit the variable at the top of each script.

## 2. Data preparation

### Source dataset JSON

```json
[
  {
    "id": "171542",
    "text": "The person sits on the floor, leans back, and then falls over.",
    "motion_path": "data/motions/171542.pt",
    "image_path":  "data/images/171542.jpg"
  }
]
```

| Field | Description |
|---|---|
| `id` | Sample ID |
| `text` | Natural-language motion description |
| `motion_path` | Path to the raw motion `.pt` (276-dim tensor or `{"motion": tensor}`) |
| `image_path` | Optional scene image used as additional conditioning |

### 276-dim motion layout (ViMoGen)

```
0:126   body_pose_6d      (21 × 6)
126:192 joints_xyz        (22 × 3)
192:258 joints_vel        (22 × 3)
258:264 root_orient_6d
264:270 root_vel_6d
270:273 root_trans
273:276 root_trans_vel
```

DSFT splits this into:

```
Base (201) = [0:126] + [126:192] + [258:264] + [270:273]
Phys (75)  = [192:258] + [264:270] + [273:276]
```

For HumanML3D the 263-dim layout produces `(D_b, D_p) = (190, 73)`.

## 3. DSFT tokenizer

Train DSFT once per dataset (the tokenizer is dataset-specific):

```bash
python tokenizer/train_tokenizer.py \
    --motiondata_root data/motions \
    --output_dir      tokenizer/checkpoints \
    --K_base 5 \
    --K_phys 25 \
    --base_vocab 4096 \
    --phys_vocab 2048 \
    --scale 10.0
```

Produces `tokenizer/checkpoints/{base,phys}/` with the BPE artifacts and per-stream `fast_config.json`.

Then convert the source dataset into per-sample tokenized `.pt` files:

```bash
python tokenizer/tokenize_dataset.py \
    --json       data/dataset.json \
    --motiondata data/motions \
    --tok_dir    tokenizer/checkpoints \
    --out_dir    data/motions_tokenized \
    --out_json   data/dataset_tokenized.json \
    --workers    4
```

Each output `.pt` contains:

```python
{
  "T":        T,                       # original number of frames
  "seq":      LongTensor [BOS, base…, SEP, phys…, EOS],
  "base_len": <int>,
  "phys_len": <int>,
}
```

The `seq` IDs are written in the tokenizer's intermediate namespace (see `tokenize_dataset.py`); `prepare_swift_data.py` remaps them to the Qwen vocabulary in the next step.

## 4. Convert to ms-swift JSONL

```bash
python training/prepare_swift_data.py \
    --json   data/dataset_tokenized.json \
    --root   . \
    --out    data/swift \
    --split  0.9
```

Outputs:

| File | Purpose |
|---|---|
| `data/swift/train.jsonl` | Training rows (`messages` chat format) |
| `data/swift/val.jsonl`   | Validation rows |
| `data/swift/motion_tokens.txt` | New special tokens added to Qwen via `--new_special_tokens` |

`motion_tokens.txt` lists `<mot_bos>`, `<mot_sep>`, `<mot_eos>`, plus 4096 `<mot_b_XXXX>` and 4096 `<mot_p_XXXX>` entries.

## 5. Phase 1 — Embed warmup

Goal: warm up `embed_tokens` and `lm_head` rows for the new motion tokens while everything else stays frozen.

```bash
bash training/train_swift_phase1.sh
```

Configuration (default in the script):

| Setting | Value |
|---|---|
| Tuner | `full` (with `--freeze_parameters_regex "model\.layers|model\.norm"`) |
| Trainable | `embed_tokens`, `lm_head` |
| Learning rate | `1e-3` |
| Steps | `500` |
| Optimizer | `adafactor` |
| Batch / Grad-accum | `4 / 4` |
| Max length | `512` |
| Output | `checkpoints/phase1_embed/` |

## 6. Phase 2 — LoRA SFT

Goal: full LoRA fine-tuning over `all-linear` modules, starting from the Phase 1 checkpoint.

Edit the `MODEL` line in `training/train_swift_phase2.sh` to point at the actual Phase 1 checkpoint directory (e.g. `checkpoints/phase1_embed/v1-<timestamp>/checkpoint-500`), then:

```bash
bash training/train_swift_phase2.sh
```

Configuration:

| Setting | Value |
|---|---|
| Tuner | `lora`, `rank=32`, `alpha=64`, `target_modules=all-linear` |
| `modules_to_save` | `embed_tokens`, `lm_head` |
| Learning rate | `2e-4` cosine, `warmup_ratio=0.03` |
| Epochs | `3` |
| Batch / Grad-accum | `4 / 4` |
| Max length | `2048` |
| Optimizer | `adamw_torch` (bfloat16, gradient checkpointing on) |
| Output | `checkpoints/swift_lora/` |

## 7. Combined H100 recipe

`training/train_swift_h100.sh` runs both phases sequentially and auto-detects the latest Phase 1 checkpoint:

```bash
bash training/train_swift_h100.sh
```

Differences from the standalone scripts:

- `per_device_train_batch_size=32`, `gradient_accumulation_steps=1` in Phase 2 (H100 80 GB).
- `max_length=4096` and `dataset_num_proc=8` in Phase 2.
- Saves at most 3 checkpoints (`save_total_limit=3`).

## 8. Inference

After Phase 2 finishes, generate motion via ms-swift inference (or any HuggingFace pipeline that loads the merged LoRA weights):

```python
# pseudo-code
from swift.llm import get_model_tokenizer, inference

model, tok = get_model_tokenizer("checkpoints/swift_lora/<run>/checkpoint-best",
                                  model_type="qwen3_5")
prompt = [
    {"type": "image", "image": "scene.jpg"},
    {"type": "text",  "text": "Generate motion for: A person walks forward and waves their hand."},
]
ids = inference(model, tok, prompt)        # token IDs in Qwen vocabulary
# Phase-aware decoding mask is applied during generation:
#   while last_token != M_EOS:
#     if not seen M_SEP: mask = base_tokens ∪ {M_SEP}
#     else:              mask = phys_tokens ∪ {M_EOS}
```

The generated `[M_BOS, b…, M_SEP, p…, M_EOS]` token stream is then decoded through DSFT's `BPE⁻¹` + `IDCT` per stream and recombined into the original `[T, D]` motion tensor.

## 9. FAQ

### Mismatch between the Phase 1 / Phase 2 dataset directory

`prepare_swift_data.py` writes `train.jsonl` and `val.jsonl`. Make sure your Phase 2 script points at the same `DATA_DIR`.

### Out-of-memory in Phase 2

- Reduce `per_device_train_batch_size`.
- Lower `max_length` (default 2048 in `phase2.sh`, 4096 in `h100.sh`).
- Keep `gradient_checkpointing true`.

### `motion_tokens.txt` size mismatch

The tokens file ships with 4096 Base + 4096 Phys placeholders. If you train DSFT with `phys_vocab < 4096`, only the first `phys_vocab` Phys tokens are actually populated; the unused entries remain in the vocabulary as inert special tokens.

### How does this differ from older drafts of the codebase?

Older internal drafts mentioned a T5 decoder, a Mixture-of-Experts residual layer, and a custom PyTorch training loop. **None of those are part of the published MotionVLA.** The training entry points are the three `train_swift_*.sh` scripts only.
