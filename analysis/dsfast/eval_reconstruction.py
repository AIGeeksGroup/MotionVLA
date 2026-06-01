"""
DSFT Tokenizer Reconstruction Quality Evaluation
================================================
Evaluates DCT-based compression on the HumanML3D 263D test set.

Split strategy: data-driven from freq_analysis_hml3d.py (threshold=0.6)
  Base dims (low-freq dominant, ~190D): root_height + ric_data + rot_6d
  Phys dims (high-freq dominant, ~73D):  root_rot_vel + root_lin_vel + vel_data + foot_contact

Default K values match DSFT (paper):
  K_base = 5
  K_phys = 25

Output: results/reconstruction_metrics.json, results/per_sample.csv
"""

import os, sys, json, csv
import numpy as np
from scipy.fft import dct, idct
from scipy.linalg import sqrtm
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR  = os.path.join(os.path.dirname(__file__),
                         "../../data/HumanML3D/HumanML3D/new_joint_vecs")
STAT_DIR  = os.path.join(os.path.dirname(__file__),
                         "../../data/HumanML3D/HumanML3D")
TEST_FILE = os.path.join(STAT_DIR, "test.txt")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── DSFT hyperparameters ───────────────────────────────────────────────
K_BASE = 5
K_PHYS = 25

# ── HumanML3D 263D semantic split ──────────────────────────────────────
# Data-driven from freq_analysis_hml3d.py (threshold=0.6 on low-freq energy)
# Confirmed via semantics: low-freq → pose/rotation, high-freq → velocity
HML3D_SEMANTICS = {
    "root_rot_vel":  list(range(0, 4)),    # high-freq → Phys
    "root_lin_vel":  list(range(4, 7)),    # high-freq → Phys
    "root_height":   [7],                  # low-freq  → Base
    "ric_data":      list(range(8, 71)),   # low-freq  → Base (21 joints × 3D)
    "rot_6d":        list(range(71, 197)), # low-freq  → Base (21 joints × 6D)
    "vel_data":      list(range(197, 260)),# high-freq → Phys
    "foot_contact":  list(range(260, 263)),# high-freq → Phys
}

BASE_DIMS = sorted(
    HML3D_SEMANTICS["root_height"] +
    HML3D_SEMANTICS["ric_data"] +
    HML3D_SEMANTICS["rot_6d"]
)  # 190D

PHYS_DIMS = sorted(
    HML3D_SEMANTICS["root_rot_vel"] +
    HML3D_SEMANTICS["root_lin_vel"] +
    HML3D_SEMANTICS["vel_data"] +
    HML3D_SEMANTICS["foot_contact"]
)  # 73D

# Joint positions in HumanML3D (ric_data: 21 joints × 3D, normalized)
JOINT_POS_DIMS = list(range(8, 71))  # 63D = 21 joints × 3D

BASE_DIMS_SET = set(BASE_DIMS)
PHYS_DIMS_SET = set(PHYS_DIMS)

print(f"Split: Base={len(BASE_DIMS)}D (K={K_BASE}), Phys={len(PHYS_DIMS)}D (K={K_PHYS})")
print(f"Total dims accounted: {len(BASE_DIMS) + len(PHYS_DIMS)} / 263")


# ── DCT reconstruction (pure compression, no BPE quantization) ─────────
def dct_compress(motion: np.ndarray, K: int) -> np.ndarray:
    """[T, D] → DCT → keep K rows → IDCT → [T, D]"""
    T = motion.shape[0]
    K_eff = min(K, T)
    freq = dct(motion.astype(np.float64), axis=0, norm="ortho")
    freq[K_eff:] = 0.0
    return idct(freq, axis=0, norm="ortho").astype(np.float32)


def reconstruct_263(motion: np.ndarray) -> np.ndarray:
    """Apply DSFT-style DCT compression on 263D HumanML3D motion."""
    recon = motion.copy()
    base_arr = motion[:, BASE_DIMS]
    phys_arr = motion[:, PHYS_DIMS]
    recon[:, BASE_DIMS] = dct_compress(base_arr, K_BASE)
    recon[:, PHYS_DIMS] = dct_compress(phys_arr, K_PHYS)
    return recon


# ── Load stats for denormalization ─────────────────────────────────────
Mean = np.load(os.path.join(STAT_DIR, "Mean.npy"))  # (263,)
Std  = np.load(os.path.join(STAT_DIR, "Std.npy"))   # (263,)
Std  = np.where(Std < 1e-8, 1.0, Std)               # avoid div-by-zero


def denorm(x: np.ndarray) -> np.ndarray:
    return x * Std + Mean


# ── Load test set ───────────────────────────────────────────────────────
with open(TEST_FILE) as f:
    test_ids = [line.strip() for line in f if line.strip()]
print(f"Test set: {len(test_ids)} sequences")

# ── Evaluate ────────────────────────────────────────────────────────────
per_sample_rows = []

rmse_full_norm  = []  # normalized RMSE (full 263D)
rmse_base_norm  = []  # normalized RMSE (Base dims only)
rmse_phys_norm  = []  # normalized RMSE (Phys dims only)
rmse_full_mm    = []  # denormalized RMSE (full 263D)
mpjpe_mm        = []  # MPJPE on joint positions (mm)

energy_coverage_base = []  # actual DCT energy coverage at K_BASE
energy_coverage_phys = []  # actual DCT energy coverage at K_PHYS

skipped = 0

for sid in tqdm(test_ids, desc="Evaluating"):
    path = os.path.join(DATA_DIR, f"{sid}.npy")
    if not os.path.exists(path):
        skipped += 1
        continue

    gt = np.load(path)  # (T, 263) normalized
    if gt.shape[0] < max(K_BASE, K_PHYS) or gt.ndim != 2 or gt.shape[1] != 263:
        skipped += 1
        continue

    T = gt.shape[0]
    rc = reconstruct_263(gt)

    # ── Normalized RMSE ─────────────────────────────────────
    err_full = gt - rc
    r_full = float(np.sqrt((err_full ** 2).mean()))
    r_base = float(np.sqrt((err_full[:, BASE_DIMS] ** 2).mean()))
    r_phys = float(np.sqrt((err_full[:, PHYS_DIMS] ** 2).mean()))

    rmse_full_norm.append(r_full)
    rmse_base_norm.append(r_base)
    rmse_phys_norm.append(r_phys)

    # ── Denormalized RMSE ───────────────────────────────────
    gt_dn = denorm(gt)
    rc_dn = denorm(rc)
    err_dn = gt_dn - rc_dn
    r_dn = float(np.sqrt((err_dn ** 2).mean()))
    rmse_full_mm.append(r_dn)

    # ── MPJPE on joint positions (ric_data, in units of Mean/Std) ──
    # ric_data is already in normalized local joint coordinates
    # Denormalize dims 8:71 only
    gt_jp = gt_dn[:, JOINT_POS_DIMS].reshape(T, 21, 3)  # (T, 21, 3)
    rc_jp = rc_dn[:, JOINT_POS_DIMS].reshape(T, 21, 3)
    mpjpe_val = float(np.sqrt(((gt_jp - rc_jp) ** 2).sum(axis=-1)).mean())
    mpjpe_mm.append(mpjpe_val)

    # ── DCT energy coverage ─────────────────────────────────
    def energy_coverage(motion_slice, K):
        T_ = motion_slice.shape[0]
        K_eff = min(K, T_)
        freq = dct(motion_slice.astype(np.float64), axis=0, norm="ortho")
        e_total = (freq ** 2).sum()
        e_k     = (freq[:K_eff] ** 2).sum()
        return float(e_k / e_total) if e_total > 1e-12 else 0.0

    cov_b = energy_coverage(gt[:, BASE_DIMS], K_BASE)
    cov_p = energy_coverage(gt[:, PHYS_DIMS], K_PHYS)
    energy_coverage_base.append(cov_b)
    energy_coverage_phys.append(cov_p)

    per_sample_rows.append({
        "id": sid, "T": T,
        "rmse_norm": r_full, "rmse_base_norm": r_base, "rmse_phys_norm": r_phys,
        "rmse_denorm": r_dn, "mpjpe": mpjpe_val,
        "energy_coverage_base": cov_b, "energy_coverage_phys": cov_p,
    })

print(f"\nEvaluated {len(per_sample_rows)} sequences, skipped {skipped}")

# ── Summary statistics ──────────────────────────────────────────────────
def stats(arr):
    a = np.array(arr)
    return {"mean": float(a.mean()), "std": float(a.std()),
            "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}

summary = {
    "n_sequences": len(per_sample_rows),
    "K_base": K_BASE, "K_phys": K_PHYS,
    "base_dims": len(BASE_DIMS), "phys_dims": len(PHYS_DIMS),
    "rmse_full_normalized":  stats(rmse_full_norm),
    "rmse_base_normalized":  stats(rmse_base_norm),
    "rmse_phys_normalized":  stats(rmse_phys_norm),
    "rmse_full_denormalized": stats(rmse_full_mm),
    "mpjpe_normalized_units": stats(mpjpe_mm),
    "energy_coverage_base_K5":  stats(energy_coverage_base),
    "energy_coverage_phys_K25": stats(energy_coverage_phys),
}

# ── Save results ────────────────────────────────────────────────────────
json_path = os.path.join(OUT_DIR, "reconstruction_metrics.json")
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {json_path}")

csv_path = os.path.join(OUT_DIR, "per_sample.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=per_sample_rows[0].keys())
    writer.writeheader()
    writer.writerows(per_sample_rows)
print(f"Saved: {csv_path}")

# ── Print paper-ready table ─────────────────────────────────────────────
print("\n" + "="*60)
print("DSFT Reconstruction Quality on HumanML3D Test Set")
print("="*60)
print(f"Sequences evaluated:      {len(per_sample_rows)}")
print(f"Base stream:              {len(BASE_DIMS)}D, K={K_BASE}")
print(f"Phys stream:              {len(PHYS_DIMS)}D, K={K_PHYS}")
print()
print(f"DCT energy coverage (Base, K={K_BASE}): {np.mean(energy_coverage_base)*100:.1f}%")
print(f"DCT energy coverage (Phys, K={K_PHYS}): {np.mean(energy_coverage_phys)*100:.1f}%")
print()
print(f"RMSE (normalized space):")
print(f"  Full 263D:  {np.mean(rmse_full_norm):.5f} ± {np.std(rmse_full_norm):.5f}")
print(f"  Base dims:  {np.mean(rmse_base_norm):.5f} ± {np.std(rmse_base_norm):.5f}")
print(f"  Phys dims:  {np.mean(rmse_phys_norm):.5f} ± {np.std(rmse_phys_norm):.5f}")
print()
print(f"RMSE (denormalized):  {np.mean(rmse_full_mm):.5f} ± {np.std(rmse_full_mm):.5f}")
print(f"MPJPE (joint pos, denormalized): {np.mean(mpjpe_mm):.5f} ± {np.std(mpjpe_mm):.5f}")
print()
print("NOTE: MPJPE is computed from ric_data (21 joints × 3D local positions).")
print("Units match HumanML3D normalization (not yet converted to meters).")
print("To get mm: multiply MPJPE by Std[8:71].mean() and scale to data units.")
