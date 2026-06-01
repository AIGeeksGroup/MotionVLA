# Motion Representation Converter: 276D <-> 263D + 18D

本文档详细介绍了 MotionVLA 中用于在 **Raw Global Motion (276 dim)** 与 **Root-Relative HumanML3D Motion (263 dim) + Global Phys Residual (18 dim)** 之间进行无损转换和重建的核心 Pipeline。

## 1. 核心目标

为了在 Sim-to-Real 迁移中实现更稳健的控制，我们需要将机器人的动作拆解为：
1.  **Semantic Base (263 dim)**: 描述动作的意图（如“向前走”），采用 Root-Relative 坐标系，去除了绝对位置和朝向。
2.  **Phys Residual (18 dim)**: 描述全局根节点状态（位置、朝向、速度），作为 Sim-to-Real 的 **Global Anchor (全局锚点)**，消除积分漂移。

本目录下的脚本实现了这两个空间之间的双向转换。

## 2. 脚本概览

所有脚本位于 `motionvla/tokenizer/276to263/`。

### 2.1 转换脚本 (Conversion)
**脚本**: `convert_276_to_263.py`

**功能**:
将 ViMoGen 原始数据 (276维) 拆分为 Base (263维) 和 Phys (18维)。

**数学逻辑**:
1.  **计算 Root Heading (Y-axis rotation)**: 从 `root_orient_6d` 中提取 Forward 向量，计算 Heading 角度。
2.  **Root-Relative 变换**:
    *   将所有关节位置 (`joints`) 和速度 (`joints_vel`) 减去根节点位置。
    *   将所有向量绕 Y 轴旋转 `-heading`，转换到根节点局部坐标系。
3.  **提取特征**:
    *   `r_velocity`: 根节点 Y 轴角速度。
    *   `l_velocity`: 根节点局部 XZ 线速度。
    *   `root_y`: 根节点高度。
    *   `ric_data`: 局部关节位置 (63 dim)。
    *   `rot_data`: 局部关节旋转 (126 dim, body_pose)。
    *   `vel_data`: 局部关节速度 (66 dim)。
    *   `foot_contact`: 足部接触状态 (4 dim)。
4.  **提取 Global Anchor (Phys 18 dim)**:
    *   直接保留 `root_orient_6d` (6), `root_trans` (3), `root_vel_6d` (6), `root_trans_vel` (3)。
    *   这保证了后续能无损重建全局状态。

**用法**:
```bash
python3 convert_276_to_263.py --input raw_276.pt --base_out base_263.pt --phys_out phys_18.pt
```

### 2.2 重建脚本 (Reconstruction)
**脚本**: `reconstruct_276.py`

**功能**:
将 Base (263维) 和 Phys (18维) 重新组合为原始数据 (276维)。

**数学逻辑 (逆变换)**:
1.  **恢复 Root State**: 直接从 Phys Token 中读取全局 Root Orient, Pos, Vel。
2.  **计算 Root Heading**: 从恢复的 Root Orient 中提取 Heading。
3.  **Global 变换**:
    *   将 `ric_data` 和 `vel_data` (局部) 绕 Y 轴旋转 `+heading`。
    *   加上 `root_trans` (全局位置)。
4.  **拼装**: 将恢复的关节数据与 Root 数据拼接，还原 276 维 Tensor。

**优势**:
*   **无积分漂移**: 每一帧的全局位置直接来自 Phys Token，而不是通过速度积分累积得到。这对于长序列生成至关重要。

**用法**:
```bash
python3 reconstruct_276.py --base_in base_263.pt --phys_in phys_18.pt --output rec_276.pt
```

### 2.3 渲染脚本 (Rendering)

#### `render_276.py`
*   **输入**: 276维原始数据 (Global)。
*   **功能**: 使用 SMPL/SMPLX 模型渲染全身动作。
*   **特点**: 
    *   复用了 `ViMoGen` 的渲染逻辑。
    *   修正了 `smplx_root.pt` 路径问题。
    *   修正了 `Mock` 对象以支持 PyTorch 计算。
    *   视角: Z-up (ViMoGen 默认)。

#### `render_263.py`
*   **输入**: 263维 Base 数据 (Root-Relative)。
*   **功能**: 渲染 Base 流动作。
*   **特点**:
    *   **积分逻辑**: 由于 Base 数据没有绝对位置，脚本通过积分 `l_velocity` 和 `r_velocity` 来恢复轨迹。
    *   **视角**: Y-up (标准 HumanML3D/SMPL)。
    *   **用途**: 验证 Base Token 是否正确捕获了动作语义（去除了绝对位置干扰）。

## 3. Pipeline 集成

这些脚本被上层 Pipeline (`motionvla/tokenizer/fast/pipeline/`) 调用：
1.  `batch_convert.py` -> 调用 `convert_276_to_263.py` 和渲染脚本。
2.  `run_fast_pipeline.py` -> 调用 `reconstruct_276.py` 进行最终验证。

## 4. 依赖

*   PyTorch
*   NumPy
*   SciPy (Rotation)
*   SMPL/SMPLX (用于渲染)
*   PyRender (用于渲染)
*   OpenCV (用于视频生成)
