"""
合并图：HumanML3D + ViMoGen 频域分析
四个子图横排：
  (a) HumanML3D 逐维度低频占比散点
  (b) HumanML3D 累积能量曲线
  (c) ViMoGen   逐维度低频占比散点
  (d) ViMoGen   累积能量曲线
"""

import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from scipy.fft import dct
from tqdm import tqdm

OUT_DIR = "analysis/figures"
os.makedirs(OUT_DIR, exist_ok=True)

K_THRESHOLD = 5
K_TOTAL     = 50
SPLIT_THR   = 0.6
k_values    = np.arange(1, K_TOTAL + 1)

# ── 通用 DCT 能量计算 ────────────────────────────────────────
def compute_energy_map(motions, n_dims):
    energy_map = np.zeros((n_dims, K_TOTAL))
    count = 0
    for m in motions:
        T = m.shape[0]
        if T < K_TOTAL:
            continue
        coeffs = dct(m, axis=0, norm='ortho')[:K_TOTAL, :]
        energy = coeffs ** 2
        dim_total = energy.sum(axis=0, keepdims=True) + 1e-8
        energy_map += (energy / dim_total).T
        count += 1
    return energy_map / count

def find_k(cumulative, target):
    for k, c in enumerate(cumulative):
        if c >= target: return k + 1
    return K_TOTAL

# ════════════════════════════════════════════════════════════
# 加载 HumanML3D
# ════════════════════════════════════════════════════════════
print("加载 HumanML3D...")
HML_DIR = "data/HumanML3D/new_joint_vecs"
files = sorted(os.listdir(HML_DIR))
np.random.seed(42)
sampled = np.random.choice(files, min(2000, len(files)), replace=False)
hml_motions = []
for f in tqdm(sampled):
    m = np.load(os.path.join(HML_DIR, f))
    if m.shape[0] >= 20 and m.shape[1] == 263:
        hml_motions.append(m)
print(f"  {len(hml_motions)} 条")

hml_energy   = compute_energy_map(hml_motions, 263)
hml_lfr      = hml_energy[:, :K_THRESHOLD].sum(axis=1)
hml_base     = np.where(hml_lfr >= SPLIT_THR)[0]
hml_phys     = np.where(hml_lfr <  SPLIT_THR)[0]
hml_cum_base = hml_energy[hml_base, :].mean(axis=0).cumsum()
hml_cum_phys = hml_energy[hml_phys, :].mean(axis=0).cumsum()

# HumanML3D 语义区段（用于背景色）
HML_REGIONS = [
    (0,   7,   '#FF7043', 'Root vel'),
    (7,   71,  '#42A5F5', 'Joint pos'),
    (71,  197, '#1565C0', 'Joint rot'),
    (197, 260, '#EF5350', 'Joint vel'),
    (260, 263, '#FF8A65', 'Foot'),
]

# ════════════════════════════════════════════════════════════
# 加载 ViMoGen
# ════════════════════════════════════════════════════════════
print("加载 ViMoGen...")
VIM_DIR = "data/motions/in_the_wild_video"
files = sorted(os.listdir(VIM_DIR))
np.random.seed(42)
sampled = np.random.choice(files, min(2000, len(files)), replace=False)
vim_motions = []
for f in tqdm(sampled):
    try:
        d = torch.load(os.path.join(VIM_DIR, f), map_location='cpu', weights_only=False)
        m = d['motion'].numpy()
        if m.shape[0] >= 20 and m.shape[1] == 276:
            vim_motions.append(m)
    except:
        continue
print(f"  {len(vim_motions)} 条")

vim_energy   = compute_energy_map(vim_motions, 276)
vim_lfr      = vim_energy[:, :K_THRESHOLD].sum(axis=1)
vim_base     = np.where(vim_lfr >= SPLIT_THR)[0]
vim_phys     = np.where(vim_lfr <  SPLIT_THR)[0]
vim_cum_base = vim_energy[vim_base, :].mean(axis=0).cumsum()
vim_cum_phys = vim_energy[vim_phys, :].mean(axis=0).cumsum()

VIM_REGIONS = [
    (0,   126, '#42A5F5', 'Joint rot/pos'),
    (126, 192, '#1565C0', 'Root orient'),
    (192, 258, '#EF5350', 'Joint vel'),
    (258, 270, '#90CAF9', 'Root coord'),
    (270, 276, '#FF7043', 'Root vel/extra'),
]

# 打印关键数字
for name, lfr, base, phys, cb, cp in [
    ("HumanML3D", hml_lfr, hml_base, hml_phys, hml_cum_base, hml_cum_phys),
    ("ViMoGen",   vim_lfr, vim_base, vim_phys, vim_cum_base, vim_cum_phys),
]:
    k5b, k5p = cb[4], cp[4]
    print(f"\n{name}: Base={len(base)}D  Phys={len(phys)}D")
    print(f"  Base K=5: {k5b*100:.1f}%   Phys K=5: {k5p*100:.1f}%")
    print(f"  Base 覆盖86% K={find_k(cb,0.86)}   Phys 覆盖86% K={find_k(cp,0.86)}")

# ════════════════════════════════════════════════════════════
# 绘图：1行4列
# ════════════════════════════════════════════════════════════
COLOR_BASE = '#2196F3'
COLOR_PHYS = '#FF5722'
ALPHA_BG   = 0.10

fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
fig.subplots_adjust(wspace=0.35)

def draw_scatter(ax, lfr, regions, n_dims, title, dataset_label):
    """逐维度低频能量占比散点图"""
    colors = [COLOR_BASE if v >= SPLIT_THR else COLOR_PHYS for v in lfr]
    ax.scatter(range(n_dims), lfr, c=colors, s=10, alpha=0.75, linewidths=0)
    ax.axhline(y=SPLIT_THR, color='gray', linewidth=1.2, linestyle='--', alpha=0.7)

    for start, end, col, label in regions:
        ax.axvspan(start, end, alpha=ALPHA_BG, color=col)
        mid = (start + end) / 2
        rot = 90 if (end - start) < 50 else 0
        ax.text(mid, 0.03, label, ha='center', va='bottom',
                fontsize=6.5, color=col, rotation=rot, fontweight='bold')

    ax.set_xlim(-2, n_dims + 2)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Dimension Index", fontsize=9)
    ax.set_ylabel(f"Low-freq Energy Ratio (K={K_THRESHOLD})", fontsize=9)
    ax.set_title(f"{dataset_label}\nPer-Dimension Frequency Ratio", fontsize=9.5, fontweight='bold')

    patch_b = mpatches.Patch(color=COLOR_BASE, label=f'Low-freq dominant (Base)')
    patch_p = mpatches.Patch(color=COLOR_PHYS, label=f'High-freq dominant (Phys)')
    ax.legend(handles=[patch_b, patch_p], fontsize=7, loc='lower right')

def draw_curves(ax, cum_base, cum_phys, n_base, n_phys, title, dataset_label):
    """累积能量曲线"""
    k5b, k5p = cum_base[4], cum_phys[4]
    ax.plot(k_values, cum_base * 100, color=COLOR_BASE, linewidth=2.2,
            label=f'Base ({n_base}D): {k5b*100:.0f}% @ K=5')
    ax.plot(k_values, cum_phys * 100, color=COLOR_PHYS, linewidth=2.2,
            label=f'Phys ({n_phys}D): {k5p*100:.0f}% @ K=5')

    ax.axvline(x=5, color='gray', linewidth=1, linestyle=':', alpha=0.7)
    ax.axhline(y=86, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.text(K_TOTAL * 0.97, 87, '86%', color='gray', fontsize=8, ha='right')

    ax.annotate(f'K=5', xy=(5, k5p * 100),
                xytext=(9, max(k5p * 100 - 15, 10)),
                fontsize=8, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.9))

    ax.set_xlim(1, K_TOTAL)
    ax.set_ylim(0, 100)
    ax.set_xlabel("DCT Coefficients Retained (K)", fontsize=9)
    ax.set_ylabel("Cumulative Energy (%)", fontsize=9)
    ax.set_title(f"{dataset_label}\nCumulative Energy Coverage", fontsize=9.5, fontweight='bold')
    ax.legend(fontsize=7.5, loc='lower right')
    ax.grid(True, alpha=0.25)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())

# (a) HumanML3D scatter
draw_scatter(axes[0], hml_lfr, HML_REGIONS, 263, "", "(a) HumanML3D")

# (b) HumanML3D curves
draw_curves(axes[1], hml_cum_base, hml_cum_phys,
            len(hml_base), len(hml_phys), "", "(b) HumanML3D")

# (c) ViMoGen scatter
draw_scatter(axes[2], vim_lfr, VIM_REGIONS, 276, "", "(c) ViMoGen")

# (d) ViMoGen curves
draw_curves(axes[3], vim_cum_base, vim_cum_phys,
            len(vim_base), len(vim_phys), "", "(d) ViMoGen")

fig.suptitle(
    "Motion dimensions naturally cluster into low-frequency (Base) and high-frequency (Phys) groups "
    "— consistent across datasets",
    fontsize=10.5, fontweight='bold', y=1.01
)

out = os.path.join(OUT_DIR, "freq_analysis_combined.pdf")
fig.savefig(out, dpi=180, bbox_inches='tight')
fig.savefig(out.replace('.pdf', '.png'), dpi=180, bbox_inches='tight')
print(f"\n合并图保存至: {out}")
plt.close()
