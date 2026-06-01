# MotionVLA Data Format

This document describes the formats consumed and produced at each stage of the MotionVLA pipeline:

```
raw motion (.pt, [T, D])
   │
   │  tokenizer/tokenize_dataset.py  (DSFT encode → BPE)
   ▼
tokenized motion (.pt, {"seq": [BOS, base…, SEP, phys…, EOS], …})
   │
   │  training/prepare_swift_data.py (remap to Qwen vocab + chat template)
   ▼
ms-swift JSONL (data/swift/{train,val}.jsonl, motion_tokens.txt)
   │
   │  bash training/train_swift_phase{1,2}.sh
   ▼
ms-swift checkpoints
```

---

## 1. Source dataset JSON

Each entry:

```json
{
  "id": "171542",
  "text": "The person sits on the floor, leans back, and then falls over.",
  "motion_path": "data/motions/171542.pt",
  "image_path":  "data/images/171542.jpg"
}
```

| Field | Type | Notes |
|---|---|---|
| `id` | str | Sample identifier |
| `text` | str | Natural-language description |
| `motion_path` | str | Path (relative to `--root`) to a raw motion `.pt` file |
| `image_path` | str | Optional scene image — when present, used as additional conditioning |

## 2. Raw motion `.pt`

A 2-D float tensor in either of these shapes:

- HumanML3D: `[T, 263]`
- ViMoGen: `[T, 276]`

May be saved as a bare tensor or as `{"motion": tensor}`.

### 276-dim ViMoGen layout

```
0:126   body_pose_6d      (21 joints × 6D rotation)
126:192 joints_xyz        (22 joints × 3 coords)
192:258 joints_vel        (22 joints × 3 velocities)
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

## 3. Tokenized motion `.pt` (DSFT output)

Produced by `tokenizer/tokenize_dataset.py`:

```python
pt = torch.load("171542.pt")
# {
#   "T":        T,                 # original number of frames
#   "seq":      LongTensor[L],     # [BOS, base…, SEP, phys…, EOS]
#   "base_len": int,               # number of base tokens
#   "phys_len": int,               # number of phys tokens
# }
```

`seq` is a single concatenated sequence in the **tokenizer's intermediate namespace**:

```
seq = [ BOS=0, base_1+32100, …, base_N+32100, SEP=32099,
        phys_1+36196, …, phys_M+36196, EOS=1 ]
```

Offsets:

| Marker | Intermediate ID |
|---|---|
| `BOS` | `0` |
| `EOS` | `1` |
| `SEP` | `32099` |
| Base BPE id `b` | `32100 + b`  (`b ∈ [0, base_vocab)`) |
| Phys BPE id `p` | `36196 + p`  (`p ∈ [0, phys_vocab)`) |

> The `32100`/`36196` numbers are an internal, BPE-friendly numbering inherited from the FAST-style tokenizer; they are **not** related to any T5 model. They are remapped to Qwen IDs in the next step.

## 4. ms-swift JSONL (after `prepare_swift_data.py`)

Each row uses the chat-format expected by ms-swift:

```json
{
  "messages": [
    {"role": "user", "content": [
      {"type": "image", "image": "data/images/171542.jpg"},
      {"type": "text",  "text": "Generate motion for: <description>"}
    ]},
    {"role": "assistant", "content": "<mot_bos><mot_b_1639><mot_b_1420>…<mot_sep><mot_p_0261>…<mot_eos>"}
  ]
}
```

If `image_path` is absent the user content is a plain string `"Generate motion for: <description>"`.

### Special-token vocabulary (`motion_tokens.txt`)

`prepare_swift_data.py` writes a `motion_tokens.txt` containing 8195 entries that ms-swift adds via `--new_special_tokens`:

```
<mot_bos>
<mot_sep>
<mot_eos>
<mot_b_0000>
<mot_b_0001>
…
<mot_b_4095>
<mot_p_0000>
<mot_p_0001>
…
<mot_p_4095>
```

### Qwen vocabulary layout (after extension)

```
[ 0,        V_LM       )  ← original Qwen vocabulary
[ 248320,   252416     )  ← Base motion tokens   (4096)
[ 252416,   256512     )  ← Phys motion tokens   (4096; ≤2048 actually used by default DSFT)
  256512                  ← M_BOS
  256513                  ← M_SEP
  256514                  ← M_EOS
```

(The exact placement is fixed in `training/prepare_swift_data.py` — `BASE_OFFSET=248320`, `PHYS_OFFSET=252416`, `MOTION_BOS_ID=256512`, etc.)

## 5. Training-time tensor shapes (during ms-swift)

ms-swift assembles the tokenized chat into:

| Tensor | Shape | Notes |
|---|---|---|
| `input_ids` | `(B, L)` | `<system>` + `<user>` + `<assistant>` + EOS, padded to `max_length` |
| `attention_mask` | `(B, L)` | Padding mask |
| `labels` | `(B, L)` | `-100` outside the assistant span; motion tokens elsewhere |
| `pixel_values` | `(N, P)` | Optional, when `image` is provided |
| `image_grid_thw` | `(B', 3)` | Optional, image grid metadata |

Loss is the standard masked next-token cross-entropy over the assistant span (motion-token positions and structural markers only).

## 6. Inference output

`generate(...)` produces a Qwen-vocabulary stream of the form:

```
[ <prompt tokens> , M_BOS , b_1 , … , b_N , M_SEP , p_1 , … , p_M , M_EOS ]
```

A phase-aware logit mask is applied during decoding:

- before `M_SEP`: only Base tokens (`248320`–`252415`) or `M_SEP` are allowed;
- after  `M_SEP`: only Phys tokens (`252416`–`256511`) or `M_EOS` are allowed.

To reconstruct the motion:

```python
base_bpe = [t - 248320 for t in stream if 248320 <= t < 252416]
phys_bpe = [t - 252416 for t in stream if 252416 <= t < 256512]

base_recon = dsft.base_tok.decode(base_bpe, T)   # [T, 201]
phys_recon = dsft.phys_tok.decode(phys_bpe, T)   # [T, 75]

motion = recombine(base_recon, phys_recon)        # [T, 276]
```

## 7. Quick reference

| Constant | Value | Defined in |
|---|---|---|
| `K_base` (default) | 5 | `tokenizer/train_tokenizer.py` |
| `K_phys` (default) | 25 (paper) / 15 (script default — override at training time) | `tokenizer/train_tokenizer.py` |
| `base_vocab` | 4096 | DSFT |
| `phys_vocab` | 2048 | DSFT |
| `BASE_OFFSET` | 248320 | `training/prepare_swift_data.py` |
| `PHYS_OFFSET` | 252416 | `training/prepare_swift_data.py` |
| `MOTION_BOS_ID` | 256512 | `training/prepare_swift_data.py` |
| `MOTION_SEP_ID` | 256513 | `training/prepare_swift_data.py` |
| `MOTION_EOS_ID` | 256514 | `training/prepare_swift_data.py` |
