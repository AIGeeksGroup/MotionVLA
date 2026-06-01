"""
Phase 1: Warm up motion token embeddings.

Only trains embed_tokens and lm_head for the NEW motion token rows (via gradient hook),
all other parameters are frozen. LR = 1e-3, ~500 steps.

Output: ./checkpoints/embed_warmed/  (full model dir, ready for ms-swift Phase 2)

Usage:
    python warmup_embed.py
"""

import os, sys
from functools import partial
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.motion_qwen_dataset import MotionQwenDataset, collate_fn_qwen

# ── Config ────────────────────────────────────────────────────────────────
MODEL_PATH  = "checkpoints/Qwen3.5-VL-8B"
TOKEN_FILE  = "data/swift/motion_tokens.txt"
JSON_PATH   = "data/dataset.json"
DATA_ROOT   = "."
OUTPUT_DIR  = "checkpoints/embed_warmed"
MAX_STEPS   = 500
LR          = 1e-3
BATCH_SIZE  = 4
MAX_SEQ_LEN = 512      # 预热阶段用短序列加速
MAX_PIXELS  = 50176
ORIG_VOCAB  = 248320   # Qwen3.5 原始词表大小
# ─────────────────────────────────────────────────────────────────────────


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. 加载模型 + tokenizer
    print("[1/4] Loading model ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True)

    # 2. 扩词表（与 train_swift.sh 保持一致）
    print("[2/4] Expanding vocabulary ...")
    with open(TOKEN_FILE) as f:
        new_tokens = [t for t in f.read().split() if t]
    added = tokenizer.add_tokens(new_tokens, special_tokens=True)
    processor.tokenizer = tokenizer          # 同步给 processor
    print(f"  Added {added} tokens → total {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

    # 3. 冻结全部参数，只解冻 embed + lm_head；
    #    再用 grad hook 把原有行的梯度清零，只让 NEW 行更新
    for p in model.parameters():
        p.requires_grad = False

    embed_w   = model.get_input_embeddings().weight
    lm_head_w = model.lm_head.weight
    embed_w.requires_grad   = True
    lm_head_w.requires_grad = True

    def _zero_orig_rows(grad, orig):
        g = grad.clone()
        g[:orig] = 0          # 原始 token 行梯度归零
        return g

    embed_w.register_hook(partial(_zero_orig_rows, orig=ORIG_VOCAB))
    lm_head_w.register_hook(partial(_zero_orig_rows, orig=ORIG_VOCAB))

    new_row_params = (len(tokenizer) - ORIG_VOCAB) * model.config.hidden_size * 2
    print(f"  Effectively training {new_row_params/1e6:.1f}M params (new token rows only)")

    # 4. 数据集
    print("[3/4] Loading dataset ...")
    dataset = MotionQwenDataset(JSON_PATH, data_root=DATA_ROOT)
    collate = partial(collate_fn_qwen, qwen_processor=processor,
                      max_seq_len=MAX_SEQ_LEN, device="cpu",
                      max_pixels=MAX_PIXELS)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         collate_fn=collate, num_workers=0, drop_last=True)

    # 5. 训练
    print(f"[4/4] Warming up {MAX_STEPS} steps (LR={LR}) ...")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR)

    model.train()
    step, pbar = 0, tqdm(total=MAX_STEPS)

    while step < MAX_STEPS:
        for batch in loader:
            if step >= MAX_STEPS:
                break

            def to(t):
                return t.to(device) if t is not None else None

            out = model(
                input_ids         = to(batch["input_ids"]),
                attention_mask    = to(batch["attention_mask"]),
                labels            = to(batch["labels"]),
                pixel_values      = to(batch["pixel_values"]),
                image_grid_thw    = to(batch["image_grid_thw"]),
                mm_token_type_ids = to(batch["mm_token_type_ids"]),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            pbar.set_postfix(loss=f"{out.loss.item():.3f}")
            pbar.update(1)

    pbar.close()

    # 6. 保存完整模型（ms-swift Phase 2 直接从这里加载）
    print(f"\nSaving warmed model → {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("Done. Run Phase 2: bash train_swift.sh")


if __name__ == "__main__":
    main()
