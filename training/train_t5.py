"""
MotionVLA T5 Trainer (Mac/MPS local training)
---------------------------------------------
Sequence format: [BOS=0, base_1+32100, ..., SEP=32099, phys_1+36196, ..., EOS=1]
Vocab size: 38244

Usage:
  python training/train_t5.py \
      --qwen_model_path checkpoints/Qwen3.5-VL-8B \
      --t5_model_path   checkpoints/t2m-ft-from-GSPretrained-base \
      --json_path       data/dataset.json \
      --epochs 50
"""

import os, sys, argparse, torch, random
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.motion_vla import MotionVLA
from dataset.motion_vla_dataset import MotionVLADataset, collate_fn


DEFAULTS = dict(
    qwen_model_path = "checkpoints/Qwen3.5-VL-8B",
    t5_model_path   = "checkpoints/t2m-ft-from-GSPretrained-base",
    json_path       = "data/dataset.json",
    ckpt_dir        = "checkpoints/saved",
    max_seq_len     = 800,
    batch_size      = 1,
    grad_accum      = 8,
    epochs          = 50,
    lr_t5           = 2e-5,
    lr_new          = 2e-4,
    moe_aux_weight  = 0.01,
    subset          = 0,
    seed            = 42,
)


def parse_args():
    p = argparse.ArgumentParser(description="MotionVLA T5 Trainer")
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    return p.parse_args()


def prepare_qwen_input(texts, images, image_paths, has_images, qwen_processor, device):
    from qwen_vl_utils import process_vision_info
    input_ids_list, attn_mask_list, pv_list, gt_list = [], [], [], []

    for text, _, img_path, has_img in zip(texts, images, image_paths, has_images):
        if has_img and img_path and os.path.exists(img_path):
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img_path, "max_pixels": 3136},
                {"type": "text",  "text": text},
            ]}]
        else:
            messages = [{"role": "user", "content": [
                {"type": "text", "text": text},
            ]}]

        prompt = qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        img_inputs, vid_inputs = process_vision_info(messages)
        out = qwen_processor(text=[prompt], images=img_inputs, videos=vid_inputs,
                             padding=True, return_tensors="pt")

        input_ids_list.append(out["input_ids"][0])
        attn_mask_list.append(out["attention_mask"][0])
        if "pixel_values"   in out: pv_list.append(out["pixel_values"])
        if "image_grid_thw" in out: gt_list.append(out["image_grid_thw"])

    pad_tok = getattr(qwen_processor.tokenizer, "pad_token_id", 151643) or 151643
    max_len  = max(ids.shape[0] for ids in input_ids_list)
    pad_ids  = torch.full((len(input_ids_list), max_len), pad_tok, dtype=torch.long)
    pad_mask = torch.zeros(len(input_ids_list), max_len, dtype=torch.long)
    for i, (ids, mask) in enumerate(zip(input_ids_list, attn_mask_list)):
        pad_ids[i, -len(ids):]   = ids
        pad_mask[i, -len(mask):] = mask

    pv = torch.cat(pv_list, dim=0).to(device) if pv_list else None
    gt = torch.cat(gt_list, dim=0).to(device) if gt_list else None
    return pad_ids.to(device), pad_mask.to(device), pv, gt


def compute_moe_aux_loss(model):
    if not hasattr(model.moe_layer, "_router_logits"):
        return torch.tensor(0.0)
    probs = torch.softmax(model.moe_layer._router_logits, dim=-1)
    return probs.mean(dim=(0, 1)).var()


def train():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    print("\n[1/4] Initializing model ...")
    model = MotionVLA(
        qwen_model_path=args.qwen_model_path,
        t5_model_path=args.t5_model_path,
        num_experts=8,
        context_target_len=256,
    )
    for p in model.action_decoder.parameters():    p.requires_grad = True
    for p in model.lm_head.parameters():           p.requires_grad = True
    for p in model.context_projector.parameters(): p.requires_grad = True
    for p in model.moe_layer.parameters():         p.requires_grad = True
    model.action_decoder.gradient_checkpointing_enable()
    model.to(device)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {n_train/1e6:.1f}M / {n_total/1e6:.1f}M")

    print("\n[2/4] Loading dataset ...")
    from transformers import AutoProcessor
    qwen_processor = AutoProcessor.from_pretrained(
        args.qwen_model_path, trust_remote_code=True)
    dataset = MotionVLADataset(args.json_path)

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    if args.subset > 0:
        indices = indices[:args.subset]
        print(f"Subset: {args.subset} samples")

    split    = int(0.8 * len(indices))
    train_ds = torch.utils.data.Subset(dataset, indices[:split])
    val_ds   = torch.utils.data.Subset(dataset, indices[split:])
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    _collate = partial(collate_fn, max_seq_len=args.max_seq_len, pad_id=0)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=_collate, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          collate_fn=_collate, num_workers=0)

    print("\n[3/4] Configuring optimizer ...")
    from transformers import get_linear_schedule_with_warmup
    param_groups = [
        {"params": list(model.action_decoder.parameters()), "lr": args.lr_t5},
        {"params": list(model.lm_head.parameters()),        "lr": args.lr_new},
        {"params": list(model.context_projector.parameters()), "lr": args.lr_new},
        {"params": list(model.moe_layer.parameters()),         "lr": args.lr_new},
    ]
    optimizer    = optim.AdamW(param_groups, weight_decay=0.01)
    total_steps  = (len(train_dl) // args.grad_accum) * args.epochs
    scheduler    = get_linear_schedule_with_warmup(optimizer, 500, total_steps)

    print("\n[4/4] Training ...")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss, t5_norm = 0.0, 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")

        for step, batch in enumerate(pbar):
            input_ids  = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            attn_mask  = batch["attn_mask"].to(device)
            qids, qmask, pv, gt = prepare_qwen_input(
                batch["texts"], batch["images"], batch["image_paths"],
                batch["has_images"], qwen_processor, device)

            out      = model(qwen_input_ids=qids, qwen_attention_mask=qmask,
                             pixel_values=pv, image_grid_thw=gt,
                             input_ids=input_ids, target_ids=target_ids, attn_mask=attn_mask)
            ce_loss  = out["loss"]
            aux_loss = compute_moe_aux_loss(model)
            loss     = ce_loss + args.moe_aux_weight * aux_loss
            (loss / args.grad_accum).backward()

            ce_val, aux_val = ce_loss.item(), aux_loss.item()
            del out, ce_loss, aux_loss, loss
            if device.type == "mps":
                torch.mps.empty_cache()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_dl):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                t5_norm = (sum(p.grad.norm().item() ** 2
                               for p in model.action_decoder.parameters()
                               if p.grad is not None) ** 0.5)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()

            total_loss += ce_val
            pbar.set_postfix(CE=f"{ce_val:.4f}", Aux=f"{aux_val:.4f}",
                             T5g=f"{t5_norm:.3f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")

        avg_train = total_loss / len(train_dl)

        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                qids, qmask, pv, gt = prepare_qwen_input(
                    batch["texts"], batch["images"], batch["image_paths"],
                    batch["has_images"], qwen_processor, device)
                out = model(qwen_input_ids=qids, qwen_attention_mask=qmask,
                            pixel_values=pv, image_grid_thw=gt,
                            input_ids=batch["input_ids"].to(device),
                            target_ids=batch["target_ids"].to(device),
                            attn_mask=batch["attn_mask"].to(device))
                total_val += out["loss"].item()

        avg_val = total_val / len(val_dl)
        print(f"Epoch {epoch+1} | Train {avg_train:.4f} | Val {avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            ckpt = os.path.join(args.ckpt_dir, f"best_ep{epoch+1}_val{avg_val:.4f}.pt")
            torch.save({"epoch": epoch+1, "model_state_dict": model.state_dict(),
                        "val_loss": avg_val, "train_loss": avg_train}, ckpt)
            print(f"  → Saved: {ckpt}")


if __name__ == "__main__":
    train()
