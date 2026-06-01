# MotionVLA 架构设计文档

> 本文档反映当前代码的真实实现状态，作为后续开发和论文写作的依据。

---

## 一、DS-FAST Tokenizer

### 1.1 核心思路

将 276 维 motion 数据按语义拆分为两个独立流，分别做 DCT 截断 + BPE 编码：

```
原始 motion: (T 帧, 276 维)
          ↓ 按维度拆分
Base (201 维): 关节旋转 + 关节位置 + 根节点朝向 + 根节点坐标  ← 低频语义
Phys (75 维):  关节速度 + 根节点速度                         ← 高频物理动态
```

维度切片定义：
```python
BASE_SLICES = [(0,126), (126,192), (258,264), (270,273)]  # 201 维
PHYS_SLICES = [(192,258), (264,270), (273,276)]            # 75 维
```

### 1.2 编码流程

```
Base: (T, 201) → DCT axis=0 → 取前 K_base=5 行 → flatten → BPE → ~477 tokens
Phys: (T, 75)  → DCT axis=0 → 取前 K_phys=15 行 → flatten → BPE → ~40 tokens
```

**为什么 K 不同？**

Base 信号低频主导，前 5 行 DCT 系数覆盖 86% 的 AC 能量，K=5 足够。  
Phys 信号高频主导，前 5 行只覆盖 37%，需 K=15 才能覆盖 56%。

### 1.3 Tokenizer 训练结果

| 流 | K | vocab | 词表使用率 | 序列长度 (median/p95) |
|---|---|---|---|---|
| Base | 5 | 4096 | 67.1% | 477 / 614 |
| Phys | 15 | 2048 | 75.4% | 40 / 117 |

保存路径：`tokenizer/DS-FAST/checkpoints/`

### 1.4 双流的贡献

Base 捕捉"做什么动作"（姿态轨迹），Phys 捕捉"怎么做"（速度节奏、加速减速曲线）。Phys 的加入使帧间过渡有惯性感，动作更自然，而非机械跳跃。

---

## 二、词表设计

### 2.1 T5 词表扩容

参考 FAST 官方实现：动作 token 追加到 LLM 原始词表后，使用 `resize_token_embeddings`，而非独立 Embedding 层。

```
T5 原始词表（32100 个）：
  ID 0       = <pad>        ← BOS / PAD（decoder start token）
  ID 1       = </s>         ← EOS
  ID 32099   = <extra_id_0> ← SEP，base 与 phys 的分界线

追加的动作 token：
  ID 32100 ~ 36195  ← Base BPE token（4096 个，BPE_id + 32100）
  ID 36196 ~ 38243  ← Phys BPE token（2048 个，BPE_id + 36196）

扩容后总词表：32100 + 4096 + 2048 = 38244
```

**为什么用 extra_id_0 作 SEP？**  
T5 词表中已预留 100 个 sentinel token（extra_id_0~99），无需新增，且已有预训练 embedding，语义上表示"段落分界"，与 SEP 角色吻合。

**为什么不用独立 Embedding？**  
独立 Embedding 层会导致 action token 与文本 token 处于不同空间，T5 decoder 的 attention 无法跨空间对齐。统一词表让 decoder 用同一套 embed_tokens 处理所有 token，注意力计算天然统一。

### 2.2 初始化细节

`resize_token_embeddings` 默认用多元正态初始化新 token（std ≈ 4.6），经过 `lm_head_scale`（≈ 0.036）后 logit std 仍过大，softmax 极度集中，导致初始 loss 虚高（实测 ~71）。

**修复**：将新增 action token 的 lm_head 行和 embed_tokens 行重新初始化为 std=0.02，使初始 logit 接近均匀分布，初始 loss 降至 8.85（理论值 8.72）。

---

## 三、训练数据

### 3.1 数据来源

仅使用 **In-the-Wild Video** 真实视频数据（排除 Synthetic 合成视频和 Optical MoCap 纯动捕数据）：

| 数据集 | 条数 | 说明 |
|---|---|---|
| In-the-Wild Video（原始）| 41,971 | 真实视频 + 文本注释 |
| tokenize 成功 | 41,965 | 6 条编码错误 |
| **有图片 + 有 motion（训练用）** | **41,961** | 最终训练集 |

图片为视频第一帧，预提取存于 `data/vimogen_full/images/{id}.jpg`。

### 3.2 序列格式

每条样本是一个单一 token 序列：

```
[BOS=0, base_1+32100, base_2+32100, ..., base_N+32100, SEP=32099, phys_1+36196, ..., phys_M+36196, EOS=1]
```

典型示例（T=65 帧的动作）：
```
总长度 = 1(BOS) + 500(base) + 1(SEP) + 20(phys) + 1(EOS) = 523 个 token
```

### 3.3 Teacher Forcing

```
input:  seq[:-1]  = [BOS, base_1, ..., base_N, SEP, phys_1, ..., phys_{M-1}]
target: seq[1:]   = [base_1, ..., base_N, SEP, phys_1, ..., phys_M, EOS]
```

单一 CrossEntropy loss，ignore_index=PAD_ID（=0）。

---

## 四、模型架构

### 4.1 整体数据流

```
图片（视频第一帧）+ 文本描述
           ↓
    Qwen3.5-VL（完全冻结）
           ↓ hidden_states[-2]，shape: (B, L_q, H_q)
    VisualFeatureResampler
      Linear → Conv1d×2（stride=2）→ interpolate → LayerNorm
           ↓ encoder_hidden_states，shape: (B, 256, 768)，bfloat16
    T5 Decoder（Full Fine-Tuning）
      cross-attend Qwen context
      自回归生成：[BOS → base tokens → SEP → phys tokens → EOS]
           ↓ last_hidden_state，shape: (B, L-1, 768)
    MotionMoELoRALayer（残差叠加）
           ↓
    lm_head（Linear 768→38244，float32）
      × lm_head_scale（= d_model^{-0.5} ≈ 0.036）
           ↓
    logits，shape: (B, L-1, 38244)
      训练时加 train_logit_mask（屏蔽 32098 个无效 T5 token）
           ↓
    CrossEntropy Loss
```

### 4.2 关键组件

#### Qwen3.5-VL（视觉-语言编码器）
- 完全冻结，仅提取特征
- 取倒数第二层 hidden states（最后一层过于任务特化）
- 输入图片限制 `max_pixels=3136`（约 4 个 patch），防止 patch 过多爆显存

#### VisualFeatureResampler（上下文投影器）
- Linear(H_q → 768) → Conv1d(stride=2) → GELU → Conv1d(stride=2) → GELU
- F.interpolate 对齐到固定长度 256（MPS 兼容，替代 AdaptiveAvgPool1d）
- LayerNorm 稳定输出
- 运行精度：bfloat16

#### T5 Decoder（动作生成器）
- 加载预训练权重，Full Fine-Tuning
- 词表扩容至 38244，embed_tokens 统一管理所有 token 的 embedding
- 开启 **gradient checkpointing**：不存储中间激活，backward 时重算，节省显存
- lm_head 权重与 embed_tokens 解绑（独立训练）
- 应用 T5 原生缩放：`h *= d_model^{-0.5}` 再送入 lm_head

#### MotionMoELoRALayer（混合专家残差层）
- 8 个 Expert，每个为 Bottleneck LoRA（rank=32）：Linear(768→32) → SiLU → Linear(32→768)
- Router：Linear(768→8) + Softmax，soft routing（加权混合所有 expert）
- 以残差形式叠加在 T5 decoder hidden states 上
- 运行精度：bfloat16
- 附加 load balancing loss（expert 负载方差，权重 0.01），防止 expert 坍塌

**MoE 的作用时机**：  
生成 base token 阶段，MoE 学习不同动作类型的表示风格；  
生成 phys token 阶段（核心），hidden state 已包含完整 base 上下文，MoE 据此路由到对应的速度动态 expert（缓慢流畅 / 急促有力 / 节奏韵律等）。

### 4.3 可训练参数

| 模块 | 参数量 | 学习率 |
|---|---|---|
| T5 Decoder | ~137M | 2e-5 |
| lm_head | ~29M | 2e-4 |
| VisualFeatureResampler | ~8M | 2e-4 |
| MotionMoELoRALayer | ~3M | 2e-4 |
| **合计可训练** | **~176M** | — |
| Qwen3.5-VL（冻结）| ~853M | 0 |
| **模型总参数** | **~1030M** | — |

### 4.4 Loss 设计

```python
# 训练时 logit mask：屏蔽 32098 个永远不会是 target 的原始 T5 token
# 有效 token 仅 6146 个：EOS(1) + SEP(32099) + base(4096) + phys(2048)
masked_logits = logits + train_logit_mask   # 无效位置 → -inf
loss = CrossEntropy(masked_logits, targets, ignore_index=PAD_ID)
       + 0.01 × MoE_load_balance_loss
```

**为什么需要训练时 logit mask？**  
不加 mask 时，38244 个 token 参与 softmax 竞争，但 target 只来自 6146 个有效 token。原始 T5 词表的 32098 个 token 抢走大量概率，初始 loss 虚高至 ~71（理论值 8.72）。加 mask 后初始 loss = 8.85，收敛更快。

---

## 五、推理流程

```
1. 输入：视频第一帧 + 文本描述
2. Qwen-VL 提取 context（冻结，无梯度）
3. T5 Decoder 自回归生成序列，从 BOS=0 开始：
   - 生成 base token（logit mask 强制只在 [32100, 36195] 采样，允许 SEP）
   - 遇到 SEP=32099，切换到 phys 模式
   - 生成 phys token（logit mask 强制只在 [36196, 38243] 采样，允许 EOS）
   - 遇到 EOS=1，停止
4. 解码：
   - base token IDs − 32100 → BPE decode → reshape(K_base=5, 201) → IDCT → (T, 201)
   - phys token IDs − 36196 → BPE decode → reshape(K_phys=15, 75) → IDCT → (T, 75)
   - 按维度切片合并 → (T, 276) 完整 motion
```

---

## 六、训练配置

| 超参数 | 值 | 说明 |
|---|---|---|
| BATCH_SIZE | 1 | MPS 显存限制 |
| GRAD_ACCUM | 8 | 等效 batch=8 |
| MAX_SEQ_LEN | 800 | 截断极少数超长序列 |
| LR_T5 | 2e-5 | T5 decoder 小学习率 |
| LR_NEW | 2e-4 | 新增模块大学习率 |
| Warmup | 500 步（固定）| 避免占用数个 epoch |
| Epochs | 50 | |
| MoE aux weight | 0.01 | load balancing |

### 显存优化

- **Gradient checkpointing**：T5 decoder 不存中间激活，节省约 40% 显存
- **del ctx**：Qwen 隐藏层投影后立即释放
- **torch.mps.empty_cache()**：每步 backward 后清空 MPS 碎片

---

## 七、数据与模型路径

| 资源 | 路径 |
|---|---|
| Qwen3.5-VL | `checkpoints/Qwen3.5-08B` |
| T5 / MG-MotionLLM | `checkpoints/t2m-ft-from-GSPretrained-base` |
| DS-FAST Tokenizer | `tokenizer/DS-FAST/checkpoints/` |
| 训练数据 JSON | `data/vimogen_full/dataset_wild_v2.json` |
| Motion pt 文件 | `data/vimogen_full/motions_dsfast_v2/{id}.pt` |
| 训练日志 | `train_v4.log` |
| 模型代码 | `src/model/motion_vla.py` |
| Trainer | `src/trainer/trainer.py` |
| Dataset | `dataset/motion_vla_dataset.py` |
