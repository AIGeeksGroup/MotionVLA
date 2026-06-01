# MotionVLA 训练指南

## 目录
1. [环境配置](#1-环境配置)
2. [数据准备](#2-数据准备)
3. [Tokenizer 训练](#3-tokenizer-训练)
4. [模型训练](#4-模型训练)
5. [多阶段训练策略](#5-多阶段训练策略)
6. [监控与调试](#6-监控与调试)
7. [常见问题](#7-常见问题)

---

## 1. 环境配置

### 依赖安装

```bash
cd /path/to/motionvla
pip install torch torchvision transformers peft
pip install scipy scikit-learn tqdm
pip install qwen-vl-utils  # Qwen-VL 图像处理工具
```

### 预训练模型下载

| 模型 | HuggingFace | 本地路径 |
|------|-------------|---------|
| Qwen3-VL | `Qwen/Qwen3-VL-2B-Instruct` | `checkpoints/Qwen3.5-08B` |
| T5/MotionLLM | `wbz0505/t2m-ft-from-GSPretrained-base` | `checkpoints/t2m-ft-from-GSPretrained-base` |

```bash
# 使用 huggingface_hub 下载
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-VL-2B-Instruct', local_dir='checkpoints/Qwen3.5-08B')
snapshot_download('wbz0505/t2m-ft-from-GSPretrained-base', local_dir='checkpoints/t2m-ft-from-GSPretrained-base')
"
```

---

## 2. 数据准备

### 数据格式

训练数据为 JSON 文件，每条样本包含：

```json
[
  {
    "id": "sample_001",
    "text": "A person walks forward and waves their hand",
    "image_path": "/absolute/path/to/scene.jpg",
    "motion_path": "/absolute/path/to/motion.pt"
  }
]
```

**运动数据格式**：`.pt` 文件，加载后为 `[T, 276]` 的 Tensor（DART 276维格式）

```
276维 DART 格式：
  0:126   body_pose_6d  (21×6，身体关节6D旋转)
  126:192 joints        (22×3，关节绝对坐标)
  192:258 joints_vel    (22×3，关节速度)
  258:264 root_orient_6d (根节点朝向)
  264:270 root_vel_6d   (根节点旋转速度)
  270:273 root_trans    (根节点绝对位置)
  273:276 root_trans_vel (根节点线速度)
```

### 数据目录结构

```
data/
├── ViMogen/
│   └── dataset_vlm.json       # 训练数据索引
├── testdata/
│   └── dataset_motionvla.json # 测试数据索引

motiondata/motions/            # 运动数据（外部路径）
├── optical_mocap/             # 干净 MoCap 数据（Sim）
└── in_the_wild_video/         # 真实视频提取数据（Real）
```

### 生成测试数据

```bash
# 生成 dummy 数据验证训练流程
python prepare_dataset.py
```

---

## 3. Tokenizer 训练

MotionVLA 使用两套 Tokenizer，按需选择：

### 方案A：DCT-VQ Tokenizer（推荐，新设计）

基于时间轴DCT频域切分的 VQ 量化 Tokenizer。

```
263维运动  →  DCT(时间轴)  →  前17行（低频）→ Base tokens（K=17个/样本）
                              后T-17行（高频）↘
18维全局根节点 → DCT(时间轴) → 后T-17行      → Phys tokens（T-17个/样本）
```

**训练命令：**

```bash
cd /path/to/motionvla
python tokenizer/dct_vq/train_tokenizer.py
```

**配置参数**（修改 `train_tokenizer.py` 顶部）：

```python
MOCAP_DIR  = '/path/to/motiondata/motions/optical_mocap'
CKPT_DIR   = 'tokenizer/dct_vq/checkpoints'
N_TRAIN    = 5000   # 训练样本数（越多越好，5000约需5分钟）
K          = 17     # 低频系数个数（90%能量点，建议不改）
BASE_VOCAB = 8192   # Base codebook大小
PHYS_VOCAB = 8192   # Phys codebook大小
```

**输出：**
```
tokenizer/dct_vq/checkpoints/
├── config.pkl          # K, vocab_size 等配置
├── base_codebook.npy   # Base codebook [8192, 263]
└── phys_codebook.npy   # Phys codebook [8192, 281]
```

**使用示例：**

```python
from tokenizer.dct_vq.dct_vq_tokenizer import DCTVQTokenizer

tok = DCTVQTokenizer.load('tokenizer/dct_vq/checkpoints')

# encode：276维运动 → 双流 tokens
base_263, phys_18 = convert_276_to_dual_stream(motion_276)  # 自行转换
base_ids, phys_ids = tok.encode(base_263, phys_18)
# base_ids: [K=17]，phys_ids: [T-K]

# decode：双流 tokens → 恢复运动
base_rec, phys_rec = tok.decode(base_ids, phys_ids, T=motion_T)
```

### 方案B：FAST Tokenizer（当前默认）

使用 DCT+BPE 的 FAST 风格 Tokenizer（已训练好）。

```
tokenizer/fast/checkpoints/
├── base/   # Base stream tokenizer（263维，vocab=8192）
└── phys/   # Phys stream tokenizer（18维，vocab=8192）
```

无需重训，直接使用：

```python
from tokenizer.fast.dual_stream_tokenizer import DualStreamFASTTokenizer
tok = DualStreamFASTTokenizer('tokenizer/fast/checkpoints')
result = tok.encode(motion_276_path)
```

---

## 4. 模型训练

### 架构概览

```
图像 + 文本
    ↓
Qwen3-VL（冻结）→ hidden_states[-2]
    ↓
VisualFeatureResampler（Conv1d × 2 + interpolate → 256帧）
    ↓
T5 Decoder（全参微调）← Base+Phys embedding 之和
    ↓
MotionMoELoRALayer（8个LoRA专家，残差叠加）
    ↓
Base Head（Linear → 8192）    Phys Head（Linear(2H→H) → 512）
                                         ↑
                               concat(hidden, base_embed)
```

### 快速启动

```bash
# 方式1：直接运行训练脚本
python src/trainer/trainer.py

# 方式2：通过 Shell 脚本（包含环境变量）
bash run_train.sh
```

### 训练配置（修改 `src/trainer/trainer.py`）

```python
# ── 路径 ──
qwen_path      = "checkpoints/Qwen3.5-08B"
t5_path        = "checkpoints/t2m-ft-from-GSPretrained-base"
json_path      = "data/ViMogen/dataset_vlm.json"
tokenizer_dir  = "tokenizer/fast/checkpoints"

# ── 超参数 ──
batch_size     = 2       # GPU显存8G建议2，16G可用4
epochs         = 10
learning_rate  = 2e-4    # Projector/Head/MoE学习率
t5_lr_scale    = 0.1     # T5 Decoder学习率 = lr × 0.1 = 2e-5
max_seq_len    = 100     # 最大token序列长度

# ── 模型 ──
base_vocab_size = 8192
phys_vocab_size = 512
num_experts     = 8
```

### 学习率分组策略

| 模块 | 学习率 | 说明 |
|------|--------|------|
| Qwen3-VL | 0（冻结） | 仅作特征提取 |
| VisualFeatureResampler | `2e-4` | 跨模态对齐 |
| T5 Decoder | `2e-5` | 全参微调，×0.1防止遗忘 |
| MoE Layer | `2e-4` | 新增模块，正常学习率 |
| Base/Phys Head | `2e-4` | 新增模块，正常学习率 |

### 损失函数

```
Loss = CrossEntropy(Base) + 0.5 × CrossEntropy(Phys)
```

Base stream 权重更高（语义意图更重要），Phys 作为辅助监督。

---

## 5. 多阶段训练策略

### Stage 1：Sim 预训练（干净MoCap数据）

```bash
# 使用 optical_mocap 数据，专注学习身体语义
json_path = "data/ViMogen/dataset_vlm.json"  # Sim数据
epochs = 10
```

目标：在干净实验室数据上建立基础运动生成能力。

### Stage 2：Sim+Real 混合微调（Sim-to-Real迁移）

```bash
# 混合 Sim + Wild Video 数据
json_path = "data/mixed/dataset_sim_real.json"
learning_rate = 5e-5  # 降低学习率，微调不大改
epochs = 5
```

目标：通过 MoE 专家的域特化，实现从实验室到真实场景的迁移。

### Checkpoint 保存

训练脚本每个 epoch 自动保存：

```
checkpoints/
└── stage1_sim/
    ├── step_100.pt
    ├── step_200.pt
    └── best.pt
```

---

## 6. 监控与调试

### 实时 Loss 监控

```bash
# 终端1：运行训练
bash run_train.sh

# 终端2：实时监控 Loss 曲线（生成图表）
python watch_loss.py

# 终端3：监控系统资源（CPU/GPU/内存）
bash monitor.sh
```

### 预期 Loss 曲线

```
Epoch 1：Base Loss ~8.5, Phys Loss ~5.8  （随机初始化水平）
Epoch 3：Base Loss ~6.0, Phys Loss ~4.0  （开始收敛）
Epoch 10：Base Loss ~3.0, Phys Loss ~2.0  （正常收敛目标）
```

> 若 Loss 不下降，检查：tokenizer路径是否正确、数据格式是否匹配

### 验证重建质量

```bash
# 验证 DCT-VQ tokenizer 重建质量
python tokenizer/dct_vq/train_tokenizer.py  # 末尾有验证输出

# 运行理论实验验证
python theory/theory1_dualstream.py  # 积分漂移验证
python theory/theory4_dualstream_vs_single.py  # 双流vs单流对比
```

---

## 7. 常见问题

### MPS 相关（Apple Silicon）

```
RuntimeError: "adaptive_avg_pool1d" not implemented for MPS
```
→ 已修复，使用 `F.interpolate` 替代。确保使用最新版 `motion_vla.py`。

```
RuntimeError: "mm" not supported for bfloat16 on MPS
```
→ 已修复，Head 层强制转 float32。

### 显存不足

```python
# 减小 batch_size
batch_size = 1

# 限制图像分辨率（减少 Qwen patch 数量）
max_pixels = 3136  # 约4个patch，在 dataset 的 process_vision_info 中设置
```

### Tokenizer 路径错误

```
FileNotFoundError: Cannot find 'base' or 'phys' checkpoints
```
→ 检查 `tokenizer_dir` 是否指向包含 `base/` 和 `phys/` 子目录的路径：
```
tokenizer/fast/checkpoints/
├── base/processor_config.json  ← 必须存在
└── phys/processor_config.json  ← 必须存在
```

### 数据 JSON 格式错误

```
KeyError: 'motion_path'
```
→ 确认 JSON 格式包含 `id`, `text`, `image_path`, `motion_path` 四个字段。

---

## 附录：数据流示意图

```
训练时数据流：

dataset_vlm.json
    ↓ MotionVLADataset.__getitem__()
{text, image, motion_276_path}
    ↓ Qwen Processor
qwen_input_ids, pixel_values
    ↓ DualStreamFASTTokenizer
base_token_ids [T], phys_token_ids [T]
    ↓ collate_fn（pad到batch内最大长度，最多100）
batch → MotionVLA.forward()
    ↓
Loss = CE(base) + 0.5 × CE(phys)
    ↓ backward + optimizer.step()
```
