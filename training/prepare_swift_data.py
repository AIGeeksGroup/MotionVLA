"""
Convert a tokenized motion dataset to ms-swift JSONL format.

Tokenized .pt files (produced by tokenizer/tokenize_dataset.py) carry
token IDs in the tokenizer's intermediate namespace. This script remaps
them to the Qwen vocabulary used during ms-swift training:

  Qwen base token (248320 + i) → <mot_b_{i:04d}>
  Qwen phys token (252416 + i) → <mot_p_{i:04d}>
  MOTION_BOS (256512)          → <mot_bos>
  MOTION_SEP (256513)          → <mot_sep>
  MOTION_EOS (256514)          → <mot_eos>

Usage:
  python prepare_swift_data.py \
      --json   data/full/dataset.json \
      --root   . \
      --out    data/swift \
      --split  0.9 \
      --image_output_dir data/swift/images
"""

import os, json, random, argparse
import torch

# Qwen-vocabulary motion token offsets (used by ms-swift training).
BASE_OFFSET   = 248320
PHYS_OFFSET   = 252416
MOTION_BOS_ID = 256512
MOTION_SEP_ID = 256513
MOTION_EOS_ID = 256514

# Intermediate-namespace constants used by tokenizer/tokenize_dataset.py.
# (Inherited from the FAST-style numbering; not related to any T5 model.)
NS_BASE_OFFSET = 32100
NS_PHYS_OFFSET = 36196
NS_SEP_ID      = 32099
NS_BOS_ID      = 0
NS_EOS_ID      = 1


def remap_to_qwen(seq: torch.Tensor) -> torch.Tensor:
    """Remap a tokenized sequence from the intermediate namespace to the
    Qwen vocabulary used during ms-swift training."""
    out = seq.clone()
    phys_mask = seq >= NS_PHYS_OFFSET
    out[phys_mask] = PHYS_OFFSET + (seq[phys_mask] - NS_PHYS_OFFSET)
    base_mask = (seq >= NS_BASE_OFFSET) & ~phys_mask
    out[base_mask] = BASE_OFFSET + (seq[base_mask] - NS_BASE_OFFSET)
    out[seq == NS_BOS_ID] = MOTION_BOS_ID
    out[seq == NS_EOS_ID] = MOTION_EOS_ID
    out[seq == NS_SEP_ID] = MOTION_SEP_ID
    return out


def id_to_str(token_id: int) -> str:
    if token_id == MOTION_BOS_ID: return "<mot_bos>"
    if token_id == MOTION_SEP_ID: return "<mot_sep>"
    if token_id == MOTION_EOS_ID: return "<mot_eos>"
    if BASE_OFFSET <= token_id < PHYS_OFFSET:
        return f"<mot_b_{token_id - BASE_OFFSET:04d}>"
    if PHYS_OFFSET <= token_id < MOTION_BOS_ID:
        return f"<mot_p_{token_id - PHYS_OFFSET:04d}>"
    return f"<unk_{token_id}>"


def load_motion_seq(motion_path: str) -> torch.Tensor | None:
    if not motion_path or not os.path.exists(motion_path):
        return None
    try:
        pt  = torch.load(motion_path, map_location="cpu", weights_only=True)
        seq = pt["seq"]
        if len(seq) > 0 and seq[0].item() < BASE_OFFSET:
            seq = remap_to_qwen(seq)
        return seq
    except Exception as e:
        print(f"[warn] {motion_path}: {e}")
        return None


def sample_to_swift(
    item: dict,
    data_root: str,
    image_output_dir: str = "",
) -> dict | None:
    def abs_path(p):
        if not p:
            return ""
        return os.path.join(data_root, p) if data_root and not os.path.isabs(p) else p

    text        = item.get("text", "")
    raw_image_path = item.get("image_path", "") or ""
    image_path  = abs_path(raw_image_path)
    motion_path = abs_path(item.get("motion_path", ""))

    seq = load_motion_seq(motion_path)
    if seq is None or len(seq) == 0:
        return None

    response = "".join(id_to_str(t.item()) for t in seq)

    if raw_image_path:
        image_out = image_path
        if image_output_dir:
            image_out = os.path.join(image_output_dir, os.path.basename(raw_image_path))
        user_content = [
            {"type": "image", "image": image_out},
            {"type": "text",  "text": f"Generate motion for: {text}"},
        ]
    elif image_path and os.path.isfile(image_path):  # 保留旧逻辑兼容
        user_content = [
            {"type": "image", "image": image_path},
            {"type": "text",  "text": f"Generate motion for: {text}"},
        ]
    else:
        user_content = f"Generate motion for: {text}"

    return {
        "messages": [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": response},
        ]
    }


def write_tokens_file(path: str):
    tokens = ["<mot_bos>", "<mot_sep>", "<mot_eos>"]
    tokens += [f"<mot_b_{i:04d}>" for i in range(4096)]
    tokens += [f"<mot_p_{i:04d}>" for i in range(4096)]
    with open(path, "w") as f:
        f.write("\n".join(tokens))
    print(f"Wrote {len(tokens)} special tokens → {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json",  required=True, help="dataset JSON path")
    ap.add_argument("--root",  default="",    help="data root prefix")
    ap.add_argument("--out",   default="data/swift", help="output directory")
    ap.add_argument("--split", type=float, default=0.9, help="train ratio")
    ap.add_argument("--seed",  type=int,   default=42)
    ap.add_argument(
        "--image_output_dir",
        default="",
        help="rewrite image path to this directory by basename, useful for server-side image roots",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    write_tokens_file(os.path.join(args.out, "motion_tokens.txt"))

    with open(args.json, encoding="utf-8") as f:
        data = json.load(f)

    random.seed(args.seed)
    random.shuffle(data)

    records, skip = [], 0
    for i, item in enumerate(data):
        rec = sample_to_swift(item, args.root, args.image_output_dir)
        if rec is None:
            skip += 1
        else:
            records.append(rec)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(data)}  ok={len(records)}  skip={skip}")

    cut   = int(len(records) * args.split)
    train = records[:cut]
    val   = records[cut:]

    for name, rows in [("train.jsonl", train), ("val.jsonl", val)]:
        out_path = os.path.join(args.out, name)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} rows → {out_path}")


if __name__ == "__main__":
    main()
