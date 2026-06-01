"""
276-dim → 263-dim + 18-dim phys residual conversion pipeline.

Steps:
  1. Convert 276-dim to 263-dim base + 18-dim phys residual
  2. Render 263-dim (optional)
  3. Reconstruct 276-dim from 263 + 18
  4. Render reconstructed 276-dim (optional)

Usage:
  python tokenizer/276to263/run_pipeline.py --input data/motions/sample.pt
"""

import os, subprocess, argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   required=True, help="Input 276-dim .pt file")
    parser.add_argument("--out_dir", default="output/pipeline", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    input_file = os.path.abspath(args.input)

    print(f"Pipeline: {input_file} → {args.out_dir}")

    # Step 1: Convert to 263 + 18
    base_path = os.path.join(args.out_dir, "base_263.pt")
    phys_path = os.path.join(args.out_dir, "phys_18.pt")
    print("\n--- Step 1: Convert 276 → 263 + 18 ---")
    subprocess.run([
        "python3", os.path.join(BASE_DIR, "convert_276_to_263.py"),
        "--input",       input_file,
        "--output_base", base_path,
        "--output_phys", phys_path,
    ], check=True)

    # Step 2: Reconstruct 276
    rec_path = os.path.join(args.out_dir, "rec_276.pt")
    print("\n--- Step 2: Reconstruct 276 from 263 + 18 ---")
    subprocess.run([
        "python3", os.path.join(BASE_DIR, "reconstruct_276.py"),
        "--base_in", base_path,
        "--phys_in", phys_path,
        "--output",  rec_path,
    ], check=True)

    print(f"\nDone!")
    print(f"  Base 263: {base_path}")
    print(f"  Phys  18: {phys_path}")
    print(f"  Rec  276: {rec_path}")


if __name__ == "__main__":
    main()
