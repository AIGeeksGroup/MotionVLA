# MotionQwen 数据格式文档

## 概览

MotionQwen 的数据流分为三层：原始磁盘文件 → 数据集读取 → Collate 拼接为模型输入。

---

## 1. 磁盘原始文件

### 1.1 JSON 索引文件（`data/vimogen_full/dataset_wild_v2.json`）

每条样本为一个 JSON 对象：

```json
{
  "id": "171542",
  "text": "The person sits on the floor, leans back, and then falls over.",
  "motion_path": "data/motions_dsfast_v2/171542.pt",
  "image_path":  "data/images/171542.jpg"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | str | 样本唯一 ID |
| `text` | str | 动作自然语言描述 |
| `motion_path` | str | DS-FAST tokenizer 预处理后的 `.pt` 文件 |
| `image_path` | str | 对应场景图片（可选，若不存在则纯文本输入） |

### 1.2 Motion `.pt` 文件（DS-FAST 预处理产物）

```python
pt = torch.load("171542.pt")
# keys: ['T', 'seq', 'base_len', 'phys_len']
```

| Key | Shape | 说明 |
|-----|-------|------|
| `seq` | `(523,)` int64 | T5 空间 token ID 序列，格式见下 |
| `T` | scalar | 原始动作帧数 |
| `base_len` | scalar | Base token 数量（本例 500） |
| `phys_len` | scalar | Phys token 数量（本例 20） |

**T5 空间序列格式：**
```
seq = [BOS=0, base_1, base_2, ..., base_N, SEP=32099, phys_1, ..., phys_M, EOS=1]
```

实际值示例（本例 id=171542）：
```
seq[:5] = [0, 33739, 33520, 33641, 33967]  ← BOS + 前4个 base token
seq[501]= 32099  ← SEP
seq[522]= 1      ← EOS
```

---

## 2. 数据集读取层（`MotionQwenDataset`）

读取后将 T5 空间 token ID **重映射**到 Qwen 扩容词表空间：

| T5 Token | T5 ID | Qwen ID | 说明 |
|----------|--------|---------|------|
| BOS | 0 | **254464** | MOTION_BOS |
| SEP | 32099 | **254465** | MOTION_SEP |
| EOS | 1 | **254466** | MOTION_EOS |
| Base token x | `32100 + x` | `248320 + x` | x ∈ [0, 4095] |
| Phys token x | `36196 + x` | `252416 + x` | x ∈ [0, 2047] |

重映射后的 `motion_ids`（Qwen 空间）：
```
[254464, 249959, 249740, ...(500个base)..., 254465, 252677, ...(20个phys)..., 254466]
  BOS     base_1  base_2                    SEP     phys_1                    EOS
```

Base BPE ID 实际分布范围：约 **1496–2045**（跳绳、行走等常见动作），phys 约 **0–2000**。

---

## 3. 训练数据（Collate 后的 Batch）

以样本 id=171542 为例，`collate_fn_qwen` 输出：

### 3.1 `input_ids`（长度 628）

```
[  0 ..  104 ]   Qwen prompt tokens（105 个）
                  = Qwen chat template("Generate motion for: <text>") + 图像 patch tokens
[ 105        ]   MOTION_BOS = 254464
[ 106 .. 605 ]   Base tokens（500 个），Qwen ID 范围 249816~250365（BPE 1496~2045）
[ 606        ]   MOTION_SEP = 254465
[ 607 .. 626 ]   Phys tokens（20 个），Qwen ID 范围 252416~254463
[ 627        ]   MOTION_EOS = 254466
```

### 3.2 `labels`（长度 628，**左移一位，修复 off-by-one**）

```
[   0 .. 104 ]   -100（prompt 全部 ignore，不参与 loss）
[ 105        ]   249959 = base_1（BPE ID 1639）← BOS 位置预测第一个 base token
[ 106        ]   249740 = base_2（BPE ID 1420）
   ...
[ 605        ]   MOTION_SEP = 254465            ← 最后一个 base 位置预测 SEP
[ 606        ]   phys_1 = 252677（BPE 261）
   ...
[ 626        ]   MOTION_EOS = 254466            ← 最后一个 phys 位置预测 EOS
[ 627        ]   -100（EOS 位置无 next token）
```

> **关键**：`labels[i]` 是位置 `i+1` 的 token，因为 `logits[i]` 预测的是下一个位置。

### 3.3 其他字段

| 字段 | Shape | 说明 |
|------|-------|------|
| `attention_mask` | `(628,)` | 全 1（有效位置） |
| `pixel_values` | `(288, 1536)` | 图像 patch 特征（由 Qwen processor 生成） |
| `image_grid_thw` | `(1, 3)` = `[1, 12, 24]` | 图像 grid：1帧 × 12行 × 24列 patch |

**有效 loss 位置数**：522（= 500 base + 1 SEP + 20 phys + 1 EOS，不含 prompt 和 BOS）

---

## 4. 推理数据格式

推理时**不需要 motion 数据**，只需 text + 可选 image。

### 4.1 输入构建

```python
messages = [{"role": "user", "content": [
    {"type": "image", "image": "path/to/image.jpg", "max_pixels": 3136},
    {"type": "text",  "text": "Generate motion for: A person walks forward"},
]}]

# processor 处理后：
enc = processor(text=[prompt], images=img_inputs, ...)
# input_ids shape: (1, 101)   ← 仅 prompt，比训练时短（无 motion 序列）
# pixel_values shape: (N_patches, 1536)
```

### 4.2 推理序列生成过程

```
输入:  [prompt tokens (101)] + [MOTION_BOS=254464]
          ↓ generate_motion()
生成:  base_1, base_2, ..., base_N, MOTION_SEP, phys_1, ..., phys_M, MOTION_EOS
```

生成时使用 **phase-aware logit mask**：
- Base 阶段：只允许 token ∈ [248320, 252416) ∪ {MOTION_SEP=254465}
- Phys 阶段（SEP 后）：只允许 token ∈ [252416, 254464) ∪ {MOTION_EOS=254466}

### 4.3 输出解析

```python
# 生成序列示例（正确推理时）：
gen = [254464, 249959, 249740, ..., 254465, 252677, ..., 254466]
#     BOS     base_1  base_2        SEP     phys_1        EOS

# 解码回 BPE ID：
base_bpe = [t - 248320 for t in gen if 248320 <= t < 252416]  # e.g. [1639, 1420, ...]
phys_bpe = [t - 252416 for t in gen if 252416 <= t < 254464]  # e.g. [261, 360, ...]

# 再经 DS-FAST detokenizer 还原为 276 维动作序列
```

---

## 5. 词表设计总览

```
Qwen 原始词表:   [0,        248320)  ← 语言 token（冻结）
Base motion:     [248320,   252416)  ← 4096 个，编码身体语义（低频 DCT）
Phys motion:     [252416,   254464)  ← 2048 个，编码根节点物理（全局锚点）
特殊 token:      254464 = MOTION_BOS
                 254465 = MOTION_SEP
                 254466 = MOTION_EOS
总词表大小:       254467
```

---

## 6. 关键数字速查

| 参数 | 值 |
|------|----|
| 训练集大小（全量） | 41,961 条 |
| 当前训练子集 | 1,000 条（800 train / 200 val） |
| 典型序列总长 | 600–700 tokens（含 prompt） |
| 典型 base token 数 | 400–550 |
| 典型 phys token 数 | 15–60 |
| 最大序列截断长 | 1024 tokens |
| 图像 patch 数 | 288（≈ 3136 pixels / 28×28 × 3 channels） |
