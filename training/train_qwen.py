"""
MotionQwen Trainer — CUDA optimized (main training path).

Key design:
  - Qwen3.5-VL + LoRA + optional MoE
  - max_pixels=200704 (256 patches, real visual signal)
  - All paths configurable via CLI args
  - CUDA bf16 mixed precision

Usage:
  # Quick validation (1000 samples):
  python training/train_qwen.py \
      --model_path checkpoints/Qwen3.5-VL-8B \
      --json_path  data/dataset.json \
      --data_root  . \
      --epochs 10 --subset 1000

  # Full training:
  python training/train_qwen.py \
      --model_path checkpoints/Qwen3.5-VL-8B \
      --json_path  data/dataset.json \
      --data_root  . \
      --epochs 30 --subset 0
"""

import os, sys, argparse, random, torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.motion_qwen import MotionQwen
from dataset.motion_qwen_dataset import MotionQwenDataset, collate_fn_qwen


DEFAULTS = dict(
    model_path  = "checkpoints/Qwen3.5-VL-8B",
    json_path   = "data/dataset.json",
    data_root   = ".",
    log_file    = "logs/train.log",
    ckpt_dir    = "checkpoints/saved",
    epochs      = 10,
    subset      = 0,
    batch_size  = 2,
    grad_accum  = 4,
    max_seq_len = 1024,
    max_pixels  = 200704,
    lr_lora     = 2e-4,
    lr_embed    = 1e-3,
    lr_moe      = 2e-4,
    lora_rank   = 32,
    lora_alpha  = 64,
    use_moe     = True,
    num_experts = 8,
    grad_clip   = 1.0,
    seed        = 42,
    grad_ckpt   = False,
    save_every  = 1,
)


def parse_args():
    p = argparse.ArgumentParser(description="MotionQwen CUDA Trainer")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", action="store_true", default=v)
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    return p.parse_args()


def compute_moe_aux_loss(model):
    if not model.use_moe or not hasattr(model, "moe_layer"):
        return torch.tensor(0.0)
    moe = model.moe_layer
    if not hasattr(moe, "_router_logits"):
        return torch.tensor(0.0)
    probs = torch.softmax(moe._router_logits, dim=-1)
    return probs.mean(dim=(0, 1)).var()


def train():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(os.path.dirname(args.log_file) or "logs", exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA: {torch.cuda.device_count()} GPU(s) — {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("WARNING: no GPU found, training on CPU will be very slow")

    print("\n[1/4] Initializing MotionQwen ...")
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    model = MotionQwen(
        qwen_model_path=args.model_path,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        use_moe=args.use_moe,
        num_experts=args.num_experts,
        device_map=device_map,
    )
    if args.grad_ckpt:
        model.qwen.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    if args.use_moe and hasattr(model, "moe_layer"):
        model.moe_layer.to(device)
    model.train_logit_mask = model.train_logit_mask.to(device)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {n_train/1e6:.1f}M / {n_total/1e6:.1f}M")

    print("\n[2/4] Loading dataset ...")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    dataset   = MotionQwenDataset(args.json_path, data_root=args.data_root)
    indices   = list(range(len(dataset)))
    random.shuffle(indices)

    if args.subset > 0:
        indices = indices[:args.subset]
        print(f"Subset: {args.subset} samples")

    split    = int(0.8 * len(indices))
    train_ds = torch.utils.data.Subset(dataset, indices[:split])
    val_ds   = torch.utils.data.Subset(dataset, indices[split:])
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    _collate = partial(collate_fn_qwen, qwen_processor=processor,
                       max_seq_len=args.max_seq_len, device=device,
                       max_pixels=args.max_pixels)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=_collate, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          collate_fn=_collate, num_workers=4, pin_memory=True)

    print("\n[3/4] Setting up optimizer ...")
    from transformers import get_linear_schedule_with_warmup
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if "lora" in n and p.requires_grad], "lr": args.lr_lora},
        {"params": [p for n, p in model.named_parameters()
                    if n in ("motion_embed", "motion_lm_head_w") and p is not None],
         "lr": args.lr_embed, "weight_decay": 0.0},
    ]
    if args.use_moe and hasattr(model, "moe_layer"):
        param_groups.append({"params": list(model.moe_layer.parameters()), "lr": args.lr_moe})

    optimizer    = optim.AdamW(param_groups, weight_decay=0.01)
    total_steps  = (len(train_dl) // args.grad_accum) * args.epochs
    warmup_steps = min(200, total_steps // 10)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"Total steps: {total_steps}, warmup: {warmup_steps}")

    print(f"\n[4/4] Training ... (log → {args.log_file})")
    log_f    = open(args.log_file, "w", buffering=1)
    best_val = float("inf")

    def log(msg):
        print(msg); log_f.write(msg + "\n")

    log(f"Config: {vars(args)}")

    for epoch in range(args.epochs):
        model.train()
        total_loss, grad_norm = 0.0, 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")

        for step, batch in enumerate(pbar):
            to = lambda t: t.to(device) if t is not None else None
            out      = model(input_ids=to(batch["input_ids"]),
                             attention_mask=to(batch["attention_mask"]),
                             labels=to(batch["labels"]),
                             pixel_values=to(batch["pixel_values"]),
                             image_grid_thw=to(batch["image_grid_thw"]),
                             mm_token_type_ids=batch.get("mm_token_type_ids"))
            ce_loss  = out["loss"]
            aux_loss = compute_moe_aux_loss(model)
            loss     = ce_loss + 0.01 * aux_loss
            (loss / args.grad_accum).backward()
            ce_val, aux_val = ce_loss.item(), aux_loss.item()
            del out, ce_loss, aux_loss, loss

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_dl):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.grad_clip).item()
                optimizer.step(); scheduler.step(); optimizer.zero_grad()

            total_loss += ce_val
            pbar.set_postfix(CE=f"{ce_val:.4f}", aux=f"{aux_val:.4f}",
                             g=f"{grad_norm:.3f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")

        avg_train = total_loss / len(train_dl)

        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                to = lambda t: t.to(device) if t is not None else None
                out = model(input_ids=to(batch["input_ids"]),
                            attention_mask=to(batch["attention_mask"]),
                            labels=to(batch["labels"]),
                            pixel_values=to(batch["pixel_values"]),
                            image_grid_thw=to(batch["image_grid_thw"]),
                            mm_token_type_ids=to(batch["mm_token_type_ids"]))
                total_val += out["loss"].item()
        avg_val = total_val / len(val_dl)

        log(f"Epoch {epoch+1} | Train {avg_train:.4f} | Val {avg_val:.4f}")

        if avg_val < best_val and (epoch + 1) % args.save_every == 0:
            best_val = avg_val
            ckpt = os.path.join(args.ckpt_dir, f"best_ep{epoch+1}_val{avg_val:.4f}")
            os.makedirs(ckpt, exist_ok=True)
            model.qwen.save_pretrained(ckpt)
            if args.use_moe and hasattr(model, "moe_layer"):
                torch.save(model.moe_layer.state_dict(), f"{ckpt}/moe_layer.pt")
            torch.save(model.motion_embed.data.cpu(), f"{ckpt}/motion_embed.pt")
            if model.motion_lm_head_w is not None:
                torch.save(model.motion_lm_head_w.data.cpu(), f"{ckpt}/motion_lm_head.pt")
            log(f"  → Saved: {ckpt}")

    final = os.path.join(args.ckpt_dir, "final")
    os.makedirs(final, exist_ok=True)
    model.qwen.save_pretrained(final)
    if args.use_moe and hasattr(model, "moe_layer"):
        torch.save(model.moe_layer.state_dict(), f"{final}/moe_layer.pt")
    torch.save(model.motion_embed.data.cpu(), f"{final}/motion_embed.pt")
    if model.motion_lm_head_w is not None:
        torch.save(model.motion_lm_head_w.data.cpu(), f"{final}/motion_lm_head.pt")
    log(f"Final checkpoint: {final}  |  Best val: {best_val:.4f}")
    log_f.close()


if __name__ == "__main__":
    train()
