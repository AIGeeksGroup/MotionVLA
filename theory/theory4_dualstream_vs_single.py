"""
Theory Experiment E4: 双流(263+18) vs 单流(276维) Tokenizer 对比

核心问题：双流设计是否真的比直接预测276维更好？

实验设计（完全公平对比）：
  - 单流276维：在相同 MoCap 数据上训练 FAST tokenizer，vocab_size=8192
  - 双流263+18：使用已训练好的 DualStreamFASTTokenizer
  - 在两个域上测试：
      * optical_mocap（干净 MoCap，in-domain）
      * in_the_wild_video（真实视频，out-of-domain）

关键指标：
  1. 重建 MSE（精度）
  2. Token 序列长度（压缩效率）
  3. 域外泛化差距 = wild_MSE / mocap_MSE（越接近1越好）

运行：
    cd <repo-root>
    python theory/theory4_dualstream_vs_single.py
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tokenizer'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tokenizer', '276to263'))


MOCAP_DIR = 'data/motions/optical_mocap'
WILD_DIR = 'data/motions/in_the_wild_video'
TOKENIZER_DIR = 'tokenizer/checkpoints'
OUT_DIR = 'theory/figures'
os.makedirs(OUT_DIR, exist_ok=True)

N_TRAIN  = 500   # 训练单流tokenizer用的样本数
N_TEST   = 100   # 每个域测试样本数
FPS      = 30


# ──────────────────────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────────────────────
def load_motion_276(path):
    d = torch.load(path, map_location='cpu')
    m = d['motion'] if isinstance(d, dict) else d
    return m.float().numpy()   # [T, 276]


def sample_files(directory, n, min_T=30):
    files = [f for f in os.listdir(directory) if f.endswith('.pt')]
    random.shuffle(files)
    chosen, skipped = [], 0
    for f in files:
        if len(chosen) >= n:
            break
        try:
            m = load_motion_276(os.path.join(directory, f))
            if m.shape[0] >= min_T and m.shape[1] >= 276:
                chosen.append(os.path.join(directory, f))
        except Exception:
            skipped += 1
    print(f"  采样 {len(chosen)} 条（跳过 {skipped} 条）from {os.path.basename(directory)}")
    return chosen


# ──────────────────────────────────────────────────────────────
# 步骤1：训练单流276维 tokenizer（公平对比）
# ──────────────────────────────────────────────────────────────
def train_single_stream_tokenizer(train_files):
    from processing_action_tokenizer import UniversalActionProcessor

    print(f"\n训练单流276维 tokenizer（{len(train_files)} 条 MoCap 数据）...")
    action_data = []
    for path in train_files:
        m = load_motion_276(path)
        T = min(m.shape[0], 128)
        action_data.append(m[:T])   # [T, 276]

    tokenizer = UniversalActionProcessor.fit(
        action_data,
        scale=100,
        vocab_size=8192,
    )
    print("  单流276维 tokenizer 训练完成")
    return tokenizer


# ──────────────────────────────────────────────────────────────
# 步骤2：加载双流 tokenizer
# ──────────────────────────────────────────────────────────────
def load_dual_stream_tokenizer():
    from dual_stream_tokenizer import DualStreamFASTTokenizer
    print("\n加载双流 263+18 tokenizer...")
    tok = DualStreamFASTTokenizer(TOKENIZER_DIR)
    print("  双流 tokenizer 加载完成")
    return tok


# ──────────────────────────────────────────────────────────────
# 步骤3：评估函数
# ──────────────────────────────────────────────────────────────
def eval_single_stream(tokenizer, paths):
    """评估单流276维 tokenizer 在给定路径上的重建质量"""
    from convert_276_to_263 import vimogen_to_humanml3d_dual_stream

    results = []
    for path in paths:
        try:
            m = load_motion_276(path)
            T = min(m.shape[0], 128)
            chunk = m[:T]   # [T, 276]

            tokens = tokenizer(chunk)
            n_tokens = len(tokens[0])

            rec = tokenizer.decode(tokens, time_horizon=T, action_dim=276)
            if rec.ndim == 3:
                rec = rec[0]

            mse = float(np.mean((chunk - rec) ** 2))
            results.append({'mse': mse, 'n_tokens': n_tokens, 'T': T})
        except Exception as e:
            pass
    return results


def eval_dual_stream(tokenizer, paths):
    """评估双流 263+18 tokenizer 在给定路径上的重建质量"""
    from convert_276_to_263 import vimogen_to_humanml3d_dual_stream
    from reconstruct_276 import reconstruct_276

    results = []
    for path in paths:
        try:
            m_orig = load_motion_276(path)
            T = min(m_orig.shape[0], 128)

            base_263, phys_18 = vimogen_to_humanml3d_dual_stream(path)
            base_np = base_263.numpy().astype(np.float32)[:T]
            phys_np = phys_18.numpy().astype(np.float32)[:T]

            tokens_base = tokenizer.base_tokenizer(base_np)
            tokens_phys = tokenizer.phys_tokenizer(phys_np)
            n_tokens = len(tokens_base[0]) + len(tokens_phys[0])

            rec_base = tokenizer.base_tokenizer.decode(tokens_base, time_horizon=T, action_dim=263)
            rec_phys = tokenizer.phys_tokenizer.decode(tokens_phys, time_horizon=T, action_dim=18)
            if rec_base.ndim == 3: rec_base = rec_base[0]
            if rec_phys.ndim == 3: rec_phys = rec_phys[0]

            rec_276 = reconstruct_276(
                torch.from_numpy(rec_base).float(),
                torch.from_numpy(rec_phys).float()
            ).numpy()

            mse = float(np.mean((m_orig[:T] - rec_276) ** 2))
            results.append({'mse': mse, 'n_tokens': n_tokens, 'T': T})
        except Exception as e:
            pass
    return results


# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Theory E4：双流(263+18) vs 单流(276维) 公平对比")
    print("=" * 60)

    # 采样文件
    print("\n【数据采样】")
    train_files  = sample_files(MOCAP_DIR,  N_TRAIN, min_T=30)
    mocap_test   = sample_files(MOCAP_DIR,  N_TEST,  min_T=30)
    wild_test    = sample_files(WILD_DIR,   N_TEST,  min_T=30)

    # 训练单流276维 tokenizer
    single_tok = train_single_stream_tokenizer(train_files)

    # 加载双流 tokenizer
    dual_tok = load_dual_stream_tokenizer()

    # 评估：MoCap（in-domain）
    print("\n【评估 MoCap（in-domain）】")
    print("  单流276维...")
    single_mocap = eval_single_stream(single_tok, mocap_test)
    print(f"  有效: {len(single_mocap)} 条")
    print("  双流263+18...")
    dual_mocap   = eval_dual_stream(dual_tok,   mocap_test)
    print(f"  有效: {len(dual_mocap)} 条")

    # 评估：Wild Video（out-of-domain）
    print("\n【评估 Wild Video（out-of-domain）】")
    print("  单流276维...")
    single_wild = eval_single_stream(single_tok, wild_test)
    print(f"  有效: {len(single_wild)} 条")
    print("  双流263+18...")
    dual_wild   = eval_dual_stream(dual_tok,   wild_test)
    print(f"  有效: {len(dual_wild)} 条")

    # ── 统计 ────────────────────────────────────────────────
    def stats(lst, key):
        vals = [r[key] for r in lst]
        return np.mean(vals), np.std(vals)

    sm_mse,  sm_std  = stats(single_mocap, 'mse')
    dm_mse,  dm_std  = stats(dual_mocap,   'mse')
    sw_mse,  sw_std  = stats(single_wild,  'mse')
    dw_mse,  dw_std  = stats(dual_wild,    'mse')

    sm_tok = stats(single_mocap, 'n_tokens')[0]
    dm_tok = stats(dual_mocap,   'n_tokens')[0]
    sw_tok = stats(single_wild,  'n_tokens')[0]
    dw_tok = stats(dual_wild,    'n_tokens')[0]

    # 泛化差距：域外MSE / 域内MSE（越接近1，泛化越好）
    single_gap = sw_mse / (sm_mse + 1e-10)
    dual_gap   = dw_mse / (dm_mse + 1e-10)

    print("\n" + "=" * 60)
    print("定量结果（用于论文）")
    print("=" * 60)
    print(f"\n{'指标':<30} {'单流276维':>15} {'双流263+18':>15}")
    print("-" * 60)
    print(f"  MoCap MSE (in-domain)    {sm_mse:>13.5f}   {dm_mse:>13.5f}")
    print(f"  Wild  MSE (out-of-domain){sw_mse:>13.5f}   {dw_mse:>13.5f}")
    print(f"  泛化差距 (wild/mocap)    {single_gap:>13.2f}x  {dual_gap:>13.2f}x")
    print(f"  Token数 (MoCap)          {sm_tok:>13.1f}   {dm_tok:>13.1f}")
    print(f"  Token数 (Wild)           {sw_tok:>13.1f}   {dw_tok:>13.1f}")

    print(f"\n核心发现：")
    if dual_gap < single_gap:
        print(f"  ✅ 双流泛化差距 ({dual_gap:.2f}x) < 单流泛化差距 ({single_gap:.2f}x)")
        print(f"     → 双流在 Sim-to-Real 迁移中表现更稳定")
    else:
        print(f"  ⚠️  双流泛化差距 ({dual_gap:.2f}x) >= 单流泛化差距 ({single_gap:.2f}x)")

    if dm_mse < sm_mse:
        print(f"  ✅ 双流域内 MSE ({dm_mse:.5f}) < 单流域内 MSE ({sm_mse:.5f})")
    else:
        print(f"  ⚠️  双流域内 MSE ({dm_mse:.5f}) >= 单流域内 MSE ({sm_mse:.5f})")

    if dm_tok < sm_tok:
        print(f"  ✅ 双流 Token 数 ({dm_tok:.0f}) < 单流 Token 数 ({sm_tok:.0f}) → 压缩效率更高")
    else:
        print(f"  ⚠️  双流 Token 数 ({dm_tok:.0f}) >= 单流 Token 数 ({sm_tok:.0f})")

    # ── 可视化 ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {'single': '#FF5722', 'dual': '#2196F3'}

    # 图1：MSE 对比（in-domain vs out-of-domain）
    ax = axes[0]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w/2, [sm_mse, sw_mse], w,
           yerr=[sm_std, sw_std], capsize=5,
           color=colors['single'], alpha=0.85, label='Single-stream 276-dim',
           edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, [dm_mse, dw_mse], w,
           yerr=[dm_std, dw_std], capsize=5,
           color=colors['dual'], alpha=0.85, label='Dual-stream 263+18',
           edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['MoCap\n(in-domain)', 'Wild Video\n(out-of-domain)'], fontsize=11)
    ax.set_ylabel('Reconstruction MSE', fontsize=11)
    ax.set_title('Reconstruction Quality\nIn-domain vs Out-of-domain', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # 图2：泛化差距对比
    ax = axes[1]
    gaps = [single_gap, dual_gap]
    bar_colors = [colors['single'], colors['dual']]
    bars = ax.bar(['Single\n276-dim', 'Dual\n263+18'], gaps,
                  color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.8)
    ax.axhline(1.0, color='green', linestyle='--', linewidth=1.5,
               label='Perfect generalization (gap=1.0)')
    ax.set_ylabel('Generalization Gap\n(Out-of-domain MSE / In-domain MSE)', fontsize=10)
    ax.set_title('Sim-to-Real Generalization\n(Lower = Better Transfer)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f'{val:.2f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 图3：Token 数量对比（压缩效率）
    ax = axes[2]
    x = np.arange(2)
    ax.bar(x - w/2, [sm_tok, sw_tok], w,
           color=colors['single'], alpha=0.85, label='Single-stream 276-dim',
           edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, [dm_tok, dw_tok], w,
           color=colors['dual'], alpha=0.85, label='Dual-stream 263+18',
           edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['MoCap\n(in-domain)', 'Wild Video\n(out-of-domain)'], fontsize=11)
    ax.set_ylabel('Number of Tokens', fontsize=11)
    ax.set_title('Compression Efficiency\n(Fewer tokens = Better compression)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Dual-stream (263+18) vs Single-stream (276-dim) Tokenizer Comparison\n'
                 'Evaluated on MoCap (in-domain) and Wild Video (out-of-domain)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'figure6_dual_vs_single.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nFigure 6 保存至: {out}")
    plt.close()

    print("\n✅ 完成")
    return {
        'single_mocap_mse': sm_mse, 'dual_mocap_mse': dm_mse,
        'single_wild_mse':  sw_mse, 'dual_wild_mse':  dw_mse,
        'single_gap': single_gap,   'dual_gap': dual_gap,
        'single_tokens': sm_tok,    'dual_tokens': dm_tok,
    }


if __name__ == '__main__':
    main()
