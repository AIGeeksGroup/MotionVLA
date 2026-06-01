import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

# T5 词表常量（与 tokenize_dataset.py 保持一致）
BOS_ID  = 0      # <pad>
EOS_ID  = 1      # </s>
SEP_ID  = 32099  # <extra_id_0>
PAD_ID  = 0      # padding 用 <pad>


class MotionVLADataset(Dataset):
    """
    MotionVLA 数据集 v2。
    motion_path 指向 .pt 文件，包含单一 token 序列：
      seq: [BOS, base_1+32100, ..., base_N+32100, SEP=32099, phys_1+36196, ..., phys_M+36196, EOS=1]
    """

    def __init__(self, json_path):
        super().__init__()
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Dataset JSON not found: {json_path}")
        with open(json_path, encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # ── 文本 ─────────────────────────────────────────
        text = item.get("text", "")

        # ── 图像（可选）──────────────────────────────────
        image_path = item.get("image_path", "") or ""
        image = None
        if image_path and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert("RGB")
                image.thumbnail((224, 224))
            except Exception as e:
                print(f"[Warning] 图像加载失败 {image_path}: {e}")

        # ── 单一 token 序列 ───────────────────────────────
        motion_path = item.get("motion_path", "")
        seq = torch.empty(0, dtype=torch.long)
        T   = 0

        if motion_path and os.path.exists(motion_path):
            try:
                pt  = torch.load(motion_path, map_location="cpu", weights_only=True)
                seq = pt["seq"]   # [BOS, base..., SEP, phys..., EOS]
                T   = pt.get("T", 0)
            except Exception as e:
                print(f"[Warning] Token 加载失败 {motion_path}: {e}")

        return {
            "id":         item.get("id", str(idx)),
            "text":       text,
            "image":      image,
            "image_path": image_path,
            "has_image":  image is not None,
            "seq":        seq,    # 完整序列 tensor
            "T":          T,
        }


def collate_fn(batch, max_seq_len=800, pad_id=PAD_ID):
    """
    单序列 collate。
    将变长序列 pad 到 batch 内最大长度（上限 max_seq_len）。

    input  = seq[:-1]   (B, L-1)  teacher forcing 输入
    target = seq[1:]    (B, L-1)  预测目标
    mask   = 1 表示真实 token，0 表示 padding（用于 loss 计算）
    """
    B = len(batch)

    texts, image_paths, images, has_images = [], [], [], []
    seqs = []

    for item in batch:
        texts.append(item["text"])
        image_paths.append(item["image_path"])
        images.append(item["image"])
        has_images.append(item["has_image"])
        # 截断到最大长度
        seqs.append(item["seq"][:max_seq_len])

    # 对齐到 batch 内最大长度
    max_len = min(max(s.shape[0] for s in seqs), max_seq_len)

    padded = torch.full((B, max_len), pad_id, dtype=torch.long)
    mask   = torch.zeros(B, max_len, dtype=torch.long)

    for i, s in enumerate(seqs):
        l = s.shape[0]
        padded[i, :l] = s
        mask[i, :l]   = 1

    # teacher forcing 切分
    input_ids  = padded[:, :-1]   # (B, L-1)
    target_ids = padded[:, 1:]    # (B, L-1)
    attn_mask  = mask[:, :-1]     # (B, L-1)  1=有效 0=pad

    return {
        "input_ids":   input_ids,    # decoder 输入
        "target_ids":  target_ids,   # 预测目标（含 PAD，loss 时 ignore）
        "attn_mask":   attn_mask,    # decoder attention mask
        "texts":       texts,
        "images":      images,
        "image_paths": image_paths,
        "has_images":  has_images,
    }
