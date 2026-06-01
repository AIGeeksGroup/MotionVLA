#!/bin/bash
# Phase 1: 只训练 embed_tokens + lm_head 新增 motion token 行
# 冻结所有 transformer 层，LR=1e-3，跑 500 步预热

set -e

DATA_DIR="data/swift"
MODEL="checkpoints/Qwen3.5-VL-8B"
OUTPUT="checkpoints/phase1_embed"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model              "$MODEL" \
    --model_type         qwen3_5 \
    --tuner_type         full \
    --freeze_parameters_regex "model\.layers|model\.norm" \
    --trainable_parameters embed_tokens lm_head \
    --new_special_tokens "$DATA_DIR/motion_tokens.txt" \
    --dataset            "$DATA_DIR/train.jsonl" \
    --torch_dtype        bfloat16 \
    --learning_rate      1e-3 \
    --max_steps          500 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --max_length         512 \
    --optim              adafactor \
    --save_steps         500 \
    --logging_steps      10 \
    --output_dir         "$OUTPUT" \
    --dataloader_num_workers 4
