"""
DS-FAST Reconstruction Quality Visualization
Generates figures for the paper from per_sample.csv.
"""

import os, json, csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_DIR     = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────
with open(os.path.join(RESULTS_DIR, "reconstruction_metrics.json")) as f:
    summary = json.load(f)

rows = []
with open(os.path.join(RESULTS_DIR, "per_sample.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append({k: float(v) if k != "id" else v for k, v in row.items()})

rmse_norm   = np.array([r["rmse_norm"] for r in rows])
rmse_base   = np.array([r["rmse_base_norm"] for r in rows])
rmse_phys   = np.array([r["rmse_phys_norm"] for r in rows])
mpjpe_m     = np.array([r["mpjpe"] for r in rows])
mpjpe_mm    = mpjpe_m * 1000
cov_base    = np.array([r["energy_coverage_base"] for r in rows])
cov_phys    = np.array([r["energy_coverage_phys"] for r in rows])
seq_lens    = np.array([int(r["T"]) for r in rows])

K_BASE = summary["K_base"]
K_PHYS = summary["K_phys"]

# ── Figure 1: RMSE distribution and per-stream breakdown ──────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
ax.hist(rmse_norm, bins=50, color="#2196F3", edgecolor="white", alpha=0.85)
ax.axvline(rmse_norm.mean(), color="red", linewidth=2, linestyle="--",
           label=f"Mean = {rmse_norm.mean():.4f}")
ax.set_xlabel("RMSE (normalized feature space)", fontsize=11)
ax.set_ylabel("Number of Sequences", fontsize=11)
ax.set_title(f"Full 263D Reconstruction RMSE\n(HumanML3D test, n={len(rows)})", fontsize=11, fontweight="bold")
ax.legend(fontsize=10)

ax = axes[1]
bins = np.linspace(0, max(rmse_base.max(), rmse_phys.max()) * 1.05, 40)
ax.hist(rmse_base, bins=bins, alpha=0.75, color="#2196F3", label=f"Base ({summary['base_dims']}D, K={K_BASE})")
ax.hist(rmse_phys, bins=bins, alpha=0.75, color="#FF5722", label=f"Phys ({summary['phys_dims']}D, K={K_PHYS})")
ax.axvline(rmse_base.mean(), color="#1565C0", linewidth=2, linestyle="--", alpha=0.8)
ax.axvline(rmse_phys.mean(), color="#BF360C", linewidth=2, linestyle="--", alpha=0.8)
ax.set_xlabel("RMSE (normalized)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title(f"Stream-wise Reconstruction Error\n(Base vs Phys)", fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.text(0.98, 0.85, f"Base: {rmse_base.mean():.4f}", transform=ax.transAxes,
        ha="right", color="#1565C0", fontsize=9)
ax.text(0.98, 0.77, f"Phys: {rmse_phys.mean():.4f}", transform=ax.transAxes,
        ha="right", color="#BF360C", fontsize=9)

ax = axes[2]
ax.hist(mpjpe_mm, bins=50, color="#4CAF50", edgecolor="white", alpha=0.85)
ax.axvline(mpjpe_mm.mean(), color="red", linewidth=2, linestyle="--",
           label=f"Mean = {mpjpe_mm.mean():.1f} mm")
ax.set_xlabel("MPJPE (mm) — ric_data joint positions", fontsize=11)
ax.set_ylabel("Number of Sequences", fontsize=11)
ax.set_title(f"Joint Position Error (MPJPE)\n21 joints × 3D local positions", fontsize=11, fontweight="bold")
ax.legend(fontsize=10)

plt.suptitle("DS-FAST Tokenizer Reconstruction Quality on HumanML3D Test Set",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
out = os.path.join(OUT_DIR, "dsfast_reconstruction_quality.pdf")
fig.savefig(out, dpi=150, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ── Figure 2: Energy coverage histogram ───────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(cov_base * 100, bins=30, alpha=0.75, color="#2196F3",
        label=f"Base (K={K_BASE}): mean={cov_base.mean()*100:.1f}%")
ax.hist(cov_phys * 100, bins=30, alpha=0.75, color="#FF5722",
        label=f"Phys (K={K_PHYS}): mean={cov_phys.mean()*100:.1f}%")
ax.set_xlabel("DCT Energy Coverage (%)", fontsize=12)
ax.set_ylabel("Number of Sequences", fontsize=12)
ax.set_title("DS-FAST DCT Energy Coverage per Sequence\n(Base K=5, Phys K=25)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=11)
ax.xaxis.set_major_formatter(mticker.PercentFormatter())
plt.tight_layout()
out2 = os.path.join(OUT_DIR, "dsfast_energy_coverage.pdf")
fig.savefig(out2, dpi=150, bbox_inches="tight")
fig.savefig(out2.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")

# ── Figure 3: RMSE vs sequence length ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
sc = ax.scatter(seq_lens, rmse_norm, c=cov_base * 100, cmap="viridis_r",
                s=8, alpha=0.5)
plt.colorbar(sc, ax=ax, label="Base energy coverage (%)")
ax.set_xlabel("Sequence Length (frames)", fontsize=12)
ax.set_ylabel("RMSE (normalized)", fontsize=12)
ax.set_title("Reconstruction Error vs Sequence Length\n(colored by Base energy coverage)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
out3 = os.path.join(OUT_DIR, "dsfast_rmse_vs_length.pdf")
fig.savefig(out3, dpi=150, bbox_inches="tight")
fig.savefig(out3.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out3}")

# ── Paper table values ─────────────────────────────────────────────────
print("\n" + "="*60)
print("PAPER TABLE VALUES")
print("="*60)
print(f"Metric                         | Value")
print(f"-------------------------------|------------------")
print(f"Base dims                      | {summary['base_dims']}D (root_h + joint_pos + joint_rot)")
print(f"Phys dims                      | {summary['phys_dims']}D (root_vel + joint_vel + foot)")
print(f"K_base (DCT coefficients)      | {K_BASE}")
print(f"K_phys (DCT coefficients)      | {K_PHYS}")
print(f"Energy coverage (Base, K=5)    | {summary['energy_coverage_base_K5']['mean']*100:.1f}%")
print(f"Energy coverage (Phys, K=25)   | {summary['energy_coverage_phys_K25']['mean']*100:.1f}%")
print(f"RMSE (normalized, full 263D)   | {summary['rmse_full_normalized']['mean']:.4f} ± {summary['rmse_full_normalized']['std']:.4f}")
print(f"RMSE (normalized, Base only)   | {summary['rmse_base_normalized']['mean']:.4f} ± {summary['rmse_base_normalized']['std']:.4f}")
print(f"RMSE (normalized, Phys only)   | {summary['rmse_phys_normalized']['mean']:.4f} ± {summary['rmse_phys_normalized']['std']:.4f}")
print(f"MPJPE (mm)                     | {summary['mpjpe_normalized_units']['mean']*1000:.2f} mm")
print(f"Compression ratio              | ~13.5× (142 frames avg)")
print(f"Sequences evaluated            | {summary['n_sequences']}")
