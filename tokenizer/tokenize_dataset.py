"""
Tokenize a motion dataset using a trained DSFT (Dual-Stream Frequency-domain
Tokenizer).

Output format — a single token sequence in the tokenizer's intermediate
namespace:

  [BOS=0, base_1+BASE_OFFSET, ..., SEP, phys_1+PHYS_OFFSET, ..., EOS=1]

The intermediate IDs (32100/36196/32099) are an internal BPE-friendly
numbering inherited from the FAST-style tokenizer; `prepare_swift_data.py`
remaps them into the Qwen vocabulary before ms-swift training.

Usage:
  python tokenizer/tokenize_dataset.py \
      --json          data/dataset.json \
      --motiondata    data/motions \
      --tok_dir       tokenizer/checkpoints \
      --out_dir       data/motions_tokenized \
      --out_json      data/dataset_tokenized.json \
      --workers       4
"""

import os, sys, json, argparse, torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Intermediate-namespace offsets (remapped to Qwen IDs in prepare_swift_data.py).
NAMESPACE_BASE  = 32100
BASE_VOCAB_SIZE = 4096
PHYS_VOCAB_SIZE = 2048
BASE_OFFSET     = NAMESPACE_BASE                     # 32100
PHYS_OFFSET     = NAMESPACE_BASE + BASE_VOCAB_SIZE   # 36196
BOS_ID          = 0
EOS_ID          = 1
SEP_ID          = 32099


def process_one(args):
    item, motiondata_root, out_motion_dir, tok_dir = args
    sid        = str(item["id"])
    rel_mpath  = item.get("motion_path", "")
    mpath      = os.path.join(motiondata_root, rel_mpath) if motiondata_root and not os.path.isabs(rel_mpath) else rel_mpath
    out        = os.path.join(out_motion_dir, f"{sid}.pt")

    if os.path.exists(out):
        return {"status": "skip", "id": sid,
                "text": item.get("text", ""),
                "image_path": item.get("image_path", ""),
                "motion_path": out}

    if not os.path.exists(mpath):
        return {"status": "no_motion", "id": sid}

    if not hasattr(process_one, "_tok"):
        from ds_fast_tokenizer import DSFTTokenizer
        process_one._tok = DSFTTokenizer.load(tok_dir)

    tok = process_one._tok
    try:
        raw    = torch.load(mpath, map_location="cpu")
        motion = raw["motion"].numpy() if isinstance(raw, dict) else raw.numpy()
        if motion.shape[-1] != 276 or motion.shape[0] < 5:
            return {"status": "bad_dim", "id": sid}

        result   = tok.encode(motion)
        T        = result["T"]
        base_bpe = np.array(result["base_tokens"], dtype=np.int64)
        phys_bpe = np.array(result["phys_tokens"], dtype=np.int64)

        base_ids = base_bpe + BASE_OFFSET
        phys_ids = phys_bpe + PHYS_OFFSET

        seq = np.concatenate([
            [BOS_ID], base_ids, [SEP_ID], phys_ids, [EOS_ID]
        ]).astype(np.int64)

        torch.save({
            "T":        T,
            "seq":      torch.tensor(seq, dtype=torch.long),
            "base_len": len(base_ids),
            "phys_len": len(phys_ids),
        }, out)

        return {"status": "ok", "id": sid,
                "text": item.get("text", ""),
                "image_path": item.get("image_path", ""),
                "motion_path": out}
    except Exception as e:
        return {"status": "error", "id": sid, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",        required=True, help="Dataset JSON with motion_path entries")
    parser.add_argument("--motiondata",  default="",    help="Root prefix for relative motion_paths in JSON")
    parser.add_argument("--tok_dir",     default="tokenizer/checkpoints")
    parser.add_argument("--out_dir",     default="data/motions_tokenized")
    parser.add_argument("--out_json",    default="data/dataset_tokenized.json")
    parser.add_argument("--workers",     type=int, default=4)
    parser.add_argument("--limit",       type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.json) as f:
        all_items = json.load(f)

    if args.limit:
        all_items = all_items[:args.limit]
    print(f"Total items: {len(all_items)}")

    task_args = [(item, args.motiondata, args.out_dir, args.tok_dir) for item in all_items]

    results = []
    if args.workers == 1:
        for a in tqdm(task_args):
            results.append(process_one(a))
    else:
        with Pool(processes=args.workers) as pool:
            for r in tqdm(pool.imap_unordered(process_one, task_args, chunksize=20),
                          total=len(task_args)):
                results.append(r)

    ok   = [r for r in results if r.get("status") in ("ok", "skip")]
    errs = [r for r in results if r.get("status") == "error"]
    print(f"\nok/skip={len(ok)}, errors={len(errs)}")

    dataset = []
    for r in ok:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        entry = {"id": r["id"], "text": text, "motion_path": r["motion_path"]}
        if r.get("image_path"):
            entry["image_path"] = r["image_path"]
        dataset.append(entry)

    with open(args.out_json, "w") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(dataset)} entries → {args.out_json}")
    print(f"Token ID ranges: BOS={BOS_ID}, SEP={SEP_ID}, EOS={EOS_ID}")
    print(f"  base: [{BASE_OFFSET}, {BASE_OFFSET+BASE_VOCAB_SIZE-1}]")
    print(f"  phys: [{PHYS_OFFSET}, {PHYS_OFFSET+PHYS_VOCAB_SIZE-1}]")


if __name__ == "__main__":
    main()
