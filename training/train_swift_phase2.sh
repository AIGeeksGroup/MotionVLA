#!/bin/bash
# Phase 2: LoRA SFT，从 Phase 1 预热后的 checkpoint 加载
#
# 执行顺序：
#   bash train_swift_phase1.sh          # Phase 1，约 30 分钟
#   bash train_swift.sh                 # Phase 2，正式训练
#
# Phase 1 结束后，将下面 MODEL 路径更新为实际 checkpoint 目录，例如：
#   checkpoints/phase1_embed/v1-20260504-143025/checkpoint-500

set -e

DATA_DIR="data/swift"
MODEL="checkpoints/phase1_embed/v1-20260504-000000/checkpoint-500"   # ← 按实际修改
OUTPUT="checkpoints/swift_lora"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=256 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model              "$MODEL" \
    --model_type         qwen3_5 \
    --tuner_type         lora \
    --modules_to_save    embed_tokens lm_head \
    --target_modules     all-linear \
    --lora_rank          32 \
    --lora_alpha         64 \
    --dataset            "$DATA_DIR/train.jsonl" \
    --val_dataset        "$DATA_DIR/val.jsonl" \
    --torch_dtype        bfloat16 \
    --num_train_epochs   3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate      2e-4 \
    --max_length         2048 \
    --gradient_checkpointing true \
    --group_by_length    true \
    --save_steps         500 \
    --eval_steps         500 \
    --logging_steps      10 \
    --save_total_limit   3 \
    --output_dir         "$OUTPUT" \
    --warmup_ratio       0.03 \
    --dataset_num_proc   4 \
    --dataloader_num_workers 4
