"""
MotionQwen Dataset — Server version
Reads DS-FAST preprocessed .pt files and remaps T5 token IDs to Qwen vocab space.

Token ID remapping:
  T5 BOS  (0)         → MOTION_BOS (254464)
  T5 EOS  (1)         → MOTION_EOS (254466)
  T5 SEP  (32099)     → MOTION_SEP (254465)
  T5 base (32100 + x) → Qwen base  (248320 + x)
  T5 phys (36196 + x) → Qwen phys  (252416 + x)
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

T5_BASE_OFFSET = 32100
T5_PHYS_OFFSET = 36196
T5_SEP_ID      = 32099
T5_BOS_ID      = 0
T5_EOS_ID      = 1

QWEN_ORIG_VOCAB = 248320
BASE_OFFSET     = QWEN_ORIG_VOCAB
PHYS_OFFSET     = QWEN_ORIG_VOCAB + 4096   # 252416
MOTION_BOS_ID   = QWEN_ORIG_VOCAB + 4096 + 4096   # 256512 (v2)
MOTION_SEP_ID   = MOTION_BOS_ID + 1               # 256513
MOTION_EOS_ID   = MOTION_BOS_ID + 2               # 256514


def remap_t5_to_qwen(seq: torch.Tensor) -> torch.Tensor:
    """Vectorized T5 → Qwen token ID remapping."""
    out = seq.clone()
    # phys tokens (check before base to avoid overlap)
    phys_mask = seq >= T5_PHYS_OFFSET
    out[phys_mask] = PHYS_OFFSET + (seq[phys_mask] - T5_PHYS_OFFSET)
    # base tokens
    base_mask = (seq >= T5_BASE_OFFSET) & ~phys_mask
    out[base_mask] = BASE_OFFSET + (seq[base_mask] - T5_BASE_OFFSET)
    # special tokens
    out[seq == T5_BOS_ID] = MOTION_BOS_ID
    out[seq == T5_EOS_ID] = MOTION_EOS_ID
    out[seq == T5_SEP_ID] = MOTION_SEP_ID
    return out


class MotionQwenDataset(Dataset):
    def __init__(self, json_path: str, data_root: str = ""):
        """
        Args:
            json_path: path to dataset JSON
            data_root: optional prefix to prepend to relative paths in JSON
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Dataset not found: {json_path}")
        with open(json_path, encoding="utf-8") as f:
            self.data = json.load(f)
        self.data_root = data_root

    def _abs(self, path: str) -> str:
        if self.data_root and not os.path.isabs(path):
            return os.path.join(self.data_root, path)
        return path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item       = self.data[idx]
        text       = item.get("text", "")
        image_path = self._abs(item.get("image_path", "") or "")
        motion_path = self._abs(item.get("motion_path", ""))

        image = None
        if image_path and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"[Warning] image load failed {image_path}: {e}")

        motion_ids = torch.empty(0, dtype=torch.long)
        if motion_path and os.path.exists(motion_path):
            try:
                pt  = torch.load(motion_path, map_location="cpu", weights_only=True)
                seq = pt["seq"]
                # v3: already in Qwen space (first token >= 248320)
                if len(seq) > 0 and seq[0].item() >= QWEN_ORIG_VOCAB:
                    motion_ids = seq
                else:
                    motion_ids = remap_t5_to_qwen(seq)
            except Exception as e:
                print(f"[Warning] motion load failed {motion_path}: {e}")

        return {
            "id":         item.get("id", str(idx)),
            "text":       text,
            "image":      image,
            "image_path": image_path,
            "has_image":  image is not None,
            "motion_ids": motion_ids,
        }


def collate_fn_qwen(batch, qwen_processor, max_seq_len=1024,
                    device="cpu", max_pixels=200704):
    """
    Build Qwen inputs from a batch of samples.

    max_pixels: controls image resolution fed to Qwen visual encoder.
      - 3136   (~2 patches):  original broken value — DO NOT USE
      - 50176  (~64 patches): safe minimum, captures rough pose
      - 200704 (~256 patches): recommended for GPU (A100/H100)
      - 802816 (~1024 patches): high quality, needs large VRAM

    Labels are left-shifted by 1:
      labels[i] = token at position i+1
      prompt positions → -100 (ignored)
    """
    from qwen_vl_utils import process_vision_info

    all_input_ids, all_labels, all_attn_masks, all_mm_type_ids = [], [], [], []
    pv_list, gt_list = [], []

    for item in batch:
        text       = item["text"]
        image_path = item["image_path"]
        has_image  = item["has_image"]
        motion_ids = item["motion_ids"]

        if has_image and image_path and os.path.exists(image_path):
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image_path, "max_pixels": max_pixels},
                {"type": "text",  "text": f"Generate motion for: {text}"},
            ]}]
        else:
            messages = [{"role": "user", "content": [
                {"type": "text", "text": f"Generate motion for: {text}"},
            ]}]

        prompt = qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        img_inputs, vid_inputs = process_vision_info(messages)
        enc = qwen_processor(
            text=[prompt], images=img_inputs, videos=vid_inputs,
            padding=False, return_tensors="pt")

        prompt_ids = enc["input_ids"][0]
        prompt_len = prompt_ids.shape[0]

        if "pixel_values"   in enc: pv_list.append(enc["pixel_values"])
        if "image_grid_thw" in enc: gt_list.append(enc["image_grid_thw"])

        # mm_token_type_ids: 0=text, 1=image token; pad motion positions with 0
        if "mm_token_type_ids" in enc:
            prompt_type_ids = enc["mm_token_type_ids"][0]
        else:
            prompt_type_ids = torch.zeros(prompt_len, dtype=torch.long)

        max_motion = max_seq_len - prompt_len
        m_ids      = motion_ids[:max_motion]
        motion_len = m_ids.shape[0]

        full_ids = torch.cat([prompt_ids, m_ids], dim=0)
        full_labels = torch.cat([
            torch.full((prompt_len,), -100, dtype=torch.long),
            m_ids[1:],
            torch.full((1,), -100, dtype=torch.long),
        ], dim=0)
        full_type_ids = torch.cat([
            prompt_type_ids,
            torch.zeros(motion_len, dtype=torch.long),
        ], dim=0)

        all_input_ids.append(full_ids)
        all_labels.append(full_labels)
        all_attn_masks.append(torch.ones(len(full_ids), dtype=torch.long))
        all_mm_type_ids.append(full_type_ids)

    max_len = max(x.shape[0] for x in all_input_ids)
    B = len(batch)
    pad_input    = torch.zeros(B, max_len, dtype=torch.long)
    pad_labels   = torch.full((B, max_len), -100, dtype=torch.long)
    pad_mask     = torch.zeros(B, max_len, dtype=torch.long)
    pad_type_ids = torch.zeros(B, max_len, dtype=torch.long)

    for i, (ids, lbs, msk, tids) in enumerate(zip(all_input_ids, all_labels, all_attn_masks, all_mm_type_ids)):
        L = ids.shape[0]
        pad_input[i, :L]    = ids
        pad_labels[i, :L]   = lbs
        pad_mask[i, :L]     = msk
        pad_type_ids[i, :L] = tids

    pixel_values   = torch.cat(pv_list, dim=0) if pv_list else None
    image_grid_thw = torch.cat(gt_list, dim=0) if gt_list else None

    return {
        "input_ids":         pad_input,
        "labels":            pad_labels,
        "attention_mask":    pad_mask,
        "pixel_values":      pixel_values,
        "image_grid_thw":    image_grid_thw,
        "mm_token_type_ids": pad_type_ids,
    }
