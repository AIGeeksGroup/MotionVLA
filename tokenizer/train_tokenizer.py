"""
DSFT (Dual-Stream Frequency-domain Tokenizer) training script.
Trains the dual-stream tokenizer on raw 276-dim ViMoGen motion data.

Usage:
  python tokenizer/train_tokenizer.py \
      --motiondata_root data/motions \
      --output_dir      tokenizer/checkpoints \
      --max_samples     20000 \
      --K_base 5 --K_phys 25 \
      --base_vocab 4096 --phys_vocab 2048
"""

import os, sys, glob, argparse, random
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ds_fast_tokenizer import DSFTTokenizer


def load_motions(motiondata_root: str, max_samples: int = None) -> list:
    subdirs = ["in_the_wild_video", "synthetic_video"]
    files = []
    for sd in subdirs:
        d = os.path.join(motiondata_root, sd)
        if os.path.isdir(d):
            files.extend(glob.glob(os.path.join(d, "*.pt")))
    # fallback: search root directly
    if not files:
        files = glob.glob(os.path.join(motiondata_root, "*.pt"))

    random.seed(42)
    random.shuffle(files)
    if max_samples:
        files = files[:max_samples]

    print(f"Loading {len(files)} motion files ...")
    motions, errors = [], 0
    for f in tqdm(files):
        try:
            d = torch.load(f, map_location="cpu")
            m = d["motion"] if isinstance(d, dict) else d
            if m.shape[-1] != 276 or m.shape[0] < 5:
                continue
            motions.append(m.numpy() if hasattr(m, "numpy") else m)
        except Exception:
            errors += 1

    print(f"Loaded: {len(motions)}, failed: {errors}")
    return motions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motiondata_root", default="data/motions",
                        help="Root directory containing motion .pt files")
    parser.add_argument("--output_dir",      default="tokenizer/checkpoints")
    parser.add_argument("--max_samples",     type=int,   default=20000)
    parser.add_argument("--K_base",          type=int,   default=5)
    parser.add_argument("--K_phys",          type=int,   default=25)
    parser.add_argument("--scale",           type=float, default=10.0)
    parser.add_argument("--base_vocab",      type=int,   default=4096)
    parser.add_argument("--phys_vocab",      type=int,   default=2048)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    motions = load_motions(args.motiondata_root, args.max_samples)
    if not motions:
        raise RuntimeError("No motion data found!")

    T_list = [m.shape[0] for m in motions]
    print(f"\nStats: {len(motions)} samples, T mean={np.mean(T_list):.0f}, "
          f"min={min(T_list)}, max={max(T_list)}")

    print(f"\nTraining: K_base={args.K_base}, K_phys={args.K_phys}, "
          f"scale={args.scale}, base_vocab={args.base_vocab}, phys_vocab={args.phys_vocab}")

    tokenizer = DSFTTokenizer.fit(
        motions,
        K_base=args.K_base,
        K_phys=args.K_phys,
        scale=args.scale,
        base_vocab=args.base_vocab,
        phys_vocab=args.phys_vocab,
    )

    tokenizer.save(args.output_dir)

    print("\n=== Validation (first 5 samples) ===")
    total_base, total_phys, total_frames = 0, 0, 0
    for m in motions[:5]:
        result = tokenizer.encode(m)
        nb = len(result["base_tokens"])
        np_ = len(result["phys_tokens"])
        T = result["T"]
        total_base += nb
        total_phys += np_
        total_frames += T
        print(f"  T={T}: base={nb} ({nb/T:.1f}/frame), phys={np_} ({np_/T:.1f}/frame)")

    print(f"\nAverage: base={total_base/total_frames:.2f} tok/frame, "
          f"phys={total_phys/total_frames:.2f} tok/frame")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
