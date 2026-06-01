#!/bin/bash
# MotionVLA ms-swift 训练脚本（H100 80GB）
# 两阶段：Phase 1 embed 预热 → Phase 2 LoRA SFT
#
# 使用方法：
#   bash train_swift_h100.sh
#
# 依赖：
#   1. data/swift/train.jsonl / val.jsonl / motion_tokens.txt 已准备好
#      python prepare_swift_data.py --json data/full/dataset.json --root . --out data/swift
#   2. ms-swift 已安装
#   3. 模型已下载到 MODEL_PATH

set -e

# ── 配置区（按实际情况修改）────────────────────────────────────────────────
MODEL_PATH="checkpoints/Qwen3.5-VL-8B"   # Set to your downloaded model path
DATA_DIR="data/swift"
PHASE1_OUTPUT="checkpoints/phase1_embed"
PHASE2_OUTPUT="checkpoints/phase2_lora"
# ────────────────────────────────────────────────────────────────────────────

echo "=========================================="
echo " Phase 1: Embed Warmup (500 steps, ~40min)"
echo "=========================================="

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model                       "$MODEL_PATH" \
    --model_type                  qwen3_5 \
    --tuner_type                  full \
    --freeze_parameters_regex     "model\.layers|model\.norm" \
    --trainable_parameters        embed_tokens lm_head \
    --new_special_tokens          "$DATA_DIR/motion_tokens.txt" \
    --dataset                     "$DATA_DIR/train.jsonl" \
    --torch_dtype                 bfloat16 \
    --learning_rate               1e-3 \
    --lr_scheduler_type           cosine \
    --max_steps                   500 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing      true \
    --max_length                  1024 \
    --optim                       adafactor \
    --save_steps                  500 \
    --logging_steps               10 \
    --output_dir                  "$PHASE1_OUTPUT"

# 自动找 Phase 1 最新 checkpoint
PHASE1_CKPT=$(ls -dt "$PHASE1_OUTPUT"/v*/checkpoint-* 2>/dev/null | head -1)
if [ -z "$PHASE1_CKPT" ]; then
    echo "[ERROR] Phase 1 checkpoint not found in $PHASE1_OUTPUT"
    exit 1
fi
echo ""
echo "Phase 1 checkpoint: $PHASE1_CKPT"

echo ""
echo "=========================================="
echo " Phase 2: LoRA SFT"
echo "=========================================="

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model                       "$PHASE1_CKPT" \
    --model_type                  qwen3_5 \
    --tuner_type                  lora \
    --lora_rank                   32 \
    --lora_alpha                  64 \
    --target_modules              all-linear \
    --modules_to_save             embed_tokens lm_head \
    --dataset                     "$DATA_DIR/train.jsonl" \
    --val_dataset                 "$DATA_DIR/val.jsonl" \
    --torch_dtype                 bfloat16 \
    --num_train_epochs            3 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate               2e-4 \
    --lr_scheduler_type           cosine \
    --warmup_ratio                0.05 \
    --max_length                  4096 \
    --gradient_checkpointing      true \
    --optim                       adamw_torch \
    --save_steps                  200 \
    --eval_steps                  200 \
    --logging_steps               10 \
    --save_total_limit            3 \
    --dataset_num_proc            8 \
    --dataloader_num_workers      4 \
    --output_dir                  "$PHASE2_OUTPUT"

echo ""
echo "=========================================="
echo " 训练完成！模型保存在 $PHASE2_OUTPUT"
echo "=========================================="
