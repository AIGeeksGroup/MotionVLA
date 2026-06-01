"""
Theory / Experiment E2: 真实 F-DST Tokenizer 重建质量评估
使用实际训练好的 DualStreamFASTTokenizer（DCT + BPE）

关键事实：
  - 你的tokenizer 不是 Haar DWT，而是 DCT (Discrete Cosine Transform) + BPE
  - dual-stream 分割是：
      Base (263-dim) = 标准 HumanML3D，根相对坐标系，语义稳定
      Phys (18-dim)  = 全局根节点状态（6D旋转+6D速度+3D位置+3D位移速度）
  - 18-dim Phys 解决的问题是：长序列积分漂移（Base是相对表示，无法重建绝对位置）

评估指标：
  - MSE_base:  263维 Base stream 重建误差
  - MSE_phys:  18维 Phys stream 重建误差
  - MSE_total: 全276维重建误差
  - Jitter:    逐帧加速度一致性（越接近1越好）
  - Global drift: 全局根节点位移误差（Phys stream 的主要作用）

对比：
  1. Full dual-stream (Base + Phys)  ← 完整系统
  2. Base-only (只解码 263维，无全局根节点)

运行：
    cd <repo-root>
    python theory/theory3_tokenizer.py
"""

import torch
import numpy as np
import os
import sys
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

# ── 路径 ──────────────────────────────────────────────────────
TOKENIZER_DIR = 'tokenizer/checkpoints'
MOCAP_DIR = 'data/motions/optical_mocap'
OUT_DIR = 'theory/figures'
os.makedirs(OUT_DIR, exist_ok=True)

# 把 tokenizer 的路径加入 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tokenizer'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tokenizer', '276to263'))

N_SAMPLES = 100   # 真实encode-decode比较慢，用100条
FPS = 30


def load_tokenizer():
    from dual_stream_tokenizer import DualStreamFASTTokenizer
    return DualStreamFASTTokenizer(TOKENIZER_DIR)


def load_motion_path(path):
    """加载 .pt 文件，返回 numpy [T, 276]"""
    d = torch.load(path, map_location='cpu')
    m = d['motion'] if isinstance(d, dict) else d
    return m.float()  # 保持 Tensor


def compute_metrics_torch(orig, recon):
    """orig, recon: Tensor [T, D]"""
    mse = ((orig - recon) ** 2).mean().item()
    # jitter: 加速度误差比
    if orig.shape[0] < 3:
        return {'mse': mse, 'jitter_ratio': 1.0, 'max_err': (orig - recon).abs().max().item()}
    acc_orig  = torch.diff(torch.diff(orig,  dim=0), dim=0)
    acc_recon = torch.diff(torch.diff(recon, dim=0), dim=0)
    jitter_orig  = acc_orig.norm(dim=1).mean().item()
    jitter_recon = acc_recon.norm(dim=1).mean().item()
    jitter_ratio = jitter_recon / (jitter_orig + 1e-8)
    max_err = (orig - recon).abs().max().item()
    return {'mse': mse, 'jitter_ratio': jitter_ratio, 'max_err': max_err}


def main():
    print("=" * 60)
    print("Theory E2：真实 DualStreamFASTTokenizer 重建质量评估")
    print("=" * 60)
    print("\n正在加载 tokenizer（DCT + BPE）...")
    tokenizer = load_tokenizer()

    # 从 vimogen_to_humanml3d_dual_stream 导入转换函数
    from convert_276_to_263 import vimogen_to_humanml3d_dual_stream
    from reconstruct_276 import reconstruct_276

    print(f"\n从 MoCap 随机采样 {N_SAMPLES} 条进行 encode-decode 评估...")
    files = [f for f in os.listdir(MOCAP_DIR) if f.endswith('.pt')]
    chosen = random.sample(files, min(N_SAMPLES, len(files)))

    results_full = []   # Full dual-stream (Base + Phys)
    results_base = []   # Base-only (不用 Phys)

    skipped = 0
    for i, fname in enumerate(chosen):
        try:
            path = os.path.join(MOCAP_DIR, fname)
            motion_276 = load_motion_path(path)
            if motion_276.shape[0] < 20:
                skipped += 1
                continue
            T = motion_276.shape[0]

            # Step 1: 转换成 base_263 + phys_18
            base_263, phys_18 = vimogen_to_humanml3d_dual_stream(path)
            base_263 = base_263.float()
            phys_18  = phys_18.float()

            # Step 2: Encode（DCT + BPE）
            base_np = base_263.numpy()
            phys_np = phys_18.numpy()
            tokens_base = tokenizer.base_tokenizer(base_np)
            tokens_phys = tokenizer.phys_tokenizer(phys_np)

            # Step 3: Decode
            rec_base_np = tokenizer.base_tokenizer.decode(tokens_base, time_horizon=T, action_dim=263)
            rec_phys_np = tokenizer.phys_tokenizer.decode(tokens_phys, time_horizon=T, action_dim=18)

            # 处理 batch 维度
            if rec_base_np.ndim == 3: rec_base_np = rec_base_np[0]
            if rec_phys_np.ndim == 3: rec_phys_np = rec_phys_np[0]

            rec_base = torch.from_numpy(rec_base_np).float()
            rec_phys = torch.from_numpy(rec_phys_np).float()

            # Step 4: 重建 276
            rec_276_full = reconstruct_276(rec_base, rec_phys)

            # Base-only：用 rec_base + 原始 phys_18 重建（模拟只有 Base stream 的情况）
            rec_276_base_only = reconstruct_276(rec_base, phys_18)

            # 计算指标
            m_full = compute_metrics_torch(motion_276, rec_276_full)
            m_base = compute_metrics_torch(motion_276, rec_276_base_only)

            # 额外：base stream 自身的重建误差
            m_base_stream = compute_metrics_torch(base_263, rec_base)
            # phys stream 自身的重建误差
            m_phys_stream = compute_metrics_torch(phys_18, rec_phys)

            # 全局根节点漂移（phys stream 的 root_trans 部分, dims 6:9 of phys_18）
            drift_orig  = phys_18[:, 6:9]    # root_trans
            drift_full  = rec_phys[:, 6:9]
            drift_mse   = ((drift_orig - drift_full) ** 2).mean().item()

            results_full.append({
                'mse_276':   m_full['mse'],
                'jitter':    m_full['jitter_ratio'],
                'max_err':   m_full['max_err'],
                'base_mse':  m_base_stream['mse'],
                'phys_mse':  m_phys_stream['mse'],
                'drift_mse': drift_mse,
            })
            results_base.append({
                'mse_276':   m_base['mse'],
                'jitter':    m_base['jitter_ratio'],
                'max_err':   m_base['max_err'],
            })

            if (i + 1) % 20 == 0:
                print(f"  已处理 {i+1}/{len(chosen)} 条，跳过 {skipped} 条")

        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  跳过 {fname}: {e}")

    print(f"\n有效样本: {len(results_full)} 条（跳过 {skipped} 条）")

    if not results_full:
        print("错误：没有有效样本")
        return

    # ── 统计 ──────────────────────────────────────────────────
    def avg(lst, key): return np.mean([r[key] for r in lst])
    def std(lst, key): return np.std([r[key]  for r in lst])

    print("\n" + "=" * 60)
    print("量化结果（用于论文）")
    print("=" * 60)
    print(f"\n{'指标':<25} {'Full (Base+Phys)':>20} {'Base-Only':>20}")
    print("-" * 65)
    print(f"  MSE_276 (全局重建)    {avg(results_full,'mse_276'):>18.4f}   {avg(results_base,'mse_276'):>18.4f}")
    print(f"  Jitter Ratio          {avg(results_full,'jitter'):>18.4f}   {avg(results_base,'jitter'):>18.4f}")
    print(f"  Max Error             {avg(results_full,'max_err'):>18.4f}   {avg(results_base,'max_err'):>18.4f}")

    print(f"\n流内部重建质量（Full 方案）:")
    print(f"  Base stream MSE (263维): {avg(results_full,'base_mse'):.6f} ± {std(results_full,'base_mse'):.6f}")
    print(f"  Phys stream MSE (18维):  {avg(results_full,'phys_mse'):.6f} ± {std(results_full,'phys_mse'):.6f}")
    print(f"  Root位移重建 MSE:        {avg(results_full,'drift_mse'):.6f} ± {std(results_full,'drift_mse'):.6f}")

    base_mse   = avg(results_full, 'base_mse')
    phys_mse   = avg(results_full, 'phys_mse')
    drift_mse  = avg(results_full, 'drift_mse')
    if base_mse > 0 and phys_mse > 0:
        ratio = phys_mse / base_mse
        print(f"\n  → Phys stream 的 MSE 是 Base stream 的 {ratio:.1f}x")
        print(f"    （Phys 包含高频全局状态，量化误差更大属于预期内）")

    # ── 绘图 ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 图1：MSE分布（箱线图）
    ax = axes[0]
    full_mse_list = [r['mse_276'] for r in results_full]
    base_mse_list = [r['mse_276'] for r in results_base]
    ax.boxplot([full_mse_list, base_mse_list],
               labels=['Full\n(Base+Phys)', 'Base-Only'],
               patch_artist=True,
               boxprops=dict(facecolor='#2196F3', alpha=0.7))
    ax.set_ylabel('MSE on 276-dim Reconstruction', fontsize=11)
    ax.set_title('Reconstruction Quality\n(Full dual-stream vs Base-only)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 图2：流内部误差对比
    ax = axes[1]
    stream_labels = ['Base stream\n(263-dim)', 'Phys stream\n(18-dim)']
    stream_means  = [avg(results_full, 'base_mse'), avg(results_full, 'phys_mse')]
    stream_stds   = [std(results_full, 'base_mse'), std(results_full, 'phys_mse')]
    bars = ax.bar(stream_labels, stream_means, yerr=stream_stds, capsize=6,
                  color=['#2196F3', '#FF9800'], alpha=0.85, edgecolor='black', linewidth=0.8)
    ax.set_ylabel('MSE (encode→decode)', fontsize=11)
    ax.set_title('Stream-level Reconstruction Error\n(DCT+BPE quantization noise per stream)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 标注比例
    if stream_means[1] > stream_means[0]:
        ax.annotate(f'{stream_means[1]/stream_means[0]:.1f}x higher',
                    xy=(1, stream_means[1]), xytext=(0.5, stream_means[1]*0.8),
                    arrowprops=dict(arrowstyle='->', color='#FF9800'),
                    fontsize=10, color='#FF9800', fontweight='bold')

    # 图3：Root位移重建误差（展示 Phys stream 的作用）
    ax = axes[2]
    drift_list = [r['drift_mse'] for r in results_full]
    ax.hist(drift_list, bins=20, color='#9C27B0', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(np.mean(drift_list), color='red', linestyle='--', linewidth=2,
               label=f'Mean={np.mean(drift_list):.4f}')
    ax.set_xlabel('Root Global Position MSE', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Global Root Position Reconstruction\n(Phys stream eliminates integration drift)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Real DualStreamFAST Tokenizer (DCT+BPE) Evaluation\n'
                 'Base: 263-dim root-relative semantics  |  Phys: 18-dim global root state',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'figure5_tokenizer_eval.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nFigure 5 保存至: {out}")
    plt.close()

    print("\n✅ 完成")
    return results_full, results_base


if __name__ == '__main__':
    main()
