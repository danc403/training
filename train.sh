#!/bin/bash

# iDragonfly Training Launcher: Optimized

DATA_DIR="./data"
CKPT_BASE="./checkpoints"
extra_mask="0.235" #percent extra random mask (decays to 0.0 by start of last epoch)

# Inputs from Command Line
MODEL_NAME=${1:-"sprite"}      # Defaults to sprite if $1 is empty
TRAINING_TYPE=${2:-"base"}    # Defaults to base if $2 is empty
EPOCHS=${3:-4}                # Defaults to 4 if $3 is empty
VRAM_GB=${4:-24}              # Defaults to 24 (your 3090)

# Define Log File Name early
LOG_FILE="${MODEL_NAME}_${TRAINING_TYPE}.log"

# --- Use exec for global redirection instead of braces ---
exec > >(tee -a "$LOG_FILE") 2>&1

# --- Dynamic Token Calculation ---
# Calculate total tokens from all data shards (2 bytes per token)
TOTAL_BYTES=$(stat -c%s ${DATA_DIR}/*_data.bin | awk '{s+=$1} END {print s}')
TOTAL_TOKENS=$((TOTAL_BYTES / 2))

# --- 1. Set Global Batch Size based on Type ---
if [ "$TRAINING_TYPE" == "tune" ]; then
    GLOBAL_BATCH_SIZE=65536
else
    GLOBAL_BATCH_SIZE=131072
fi

# --- 2. Set Micro-Batch Size based on Model & VRAM ---
case $MODEL_NAME in
    "sprite")
        if [ "$VRAM_GB" -le 12 ]; then GLOBAL_BATCH_SIZE=65536; else GLOBAL_BATCH_SIZE=131072; fi
        if [ "$VRAM_GB" -le 12 ]; then MICRO_BATCH_SIZE=8; else MICRO_BATCH_SIZE=16; fi
        LEARNING_RATE=0.0014
        LEARNING_RATE_TUNE=0.0003
        ;;
    "nymph")
        if [ "$VRAM_GB" -le 12 ]; then GLOBAL_BATCH_SIZE=65536; else GLOBAL_BATCH_SIZE=131072; fi
        if [ "$VRAM_GB" -le 12 ]; then MICRO_BATCH_SIZE=4; else MICRO_BATCH_SIZE=8; fi
        LEARNING_RATE=0.0012
        LEARNING_RATE_TUNE=0.0003
        ;;
    "dragonfly")
        if [ "$VRAM_GB" -le 12 ]; then GLOBAL_BATCH_SIZE=65536; else GLOBAL_BATCH_SIZE=131072; fi
        if [ "$VRAM_GB" -le 12 ]; then MICRO_BATCH_SIZE=4; else MICRO_BATCH_SIZE=8; fi
        LEARNING_RATE=0.0010
        LEARNING_RATE_TUNE=0.0003
        ;;
    "wyrm")
        if [ "$VRAM_GB" -le 12 ]; then GLOBAL_BATCH_SIZE=65536; else GLOBAL_BATCH_SIZE=131072; fi
        if [ "$VRAM_GB" -le 12 ]; then MICRO_BATCH_SIZE=4; else MICRO_BATCH_SIZE=8; fi
        LEARNING_RATE=0.0008
        LEARNING_RATE_TUNE=0.0003
        ;;
esac

STEPS_PER_EPOCH=$((TOTAL_TOKENS / GLOBAL_BATCH_SIZE))
MAX_STEPS=$((STEPS_PER_EPOCH * EPOCHS))

# --- 3. Set Active Learning Rate and Warmup Logic ---
if [ "$TRAINING_TYPE" == "tune" ]; then
    ACTIVE_LR=$LEARNING_RATE_TUNE
    WARMUP_STEPS=25
else
    ACTIVE_LR=$LEARNING_RATE
    WARMUP_STEPS=$((MAX_STEPS / 3))
fi

WEIGHT_DECAY=0.1
LOG_FREQ=15
SAVE_FREQ=$((STEPS_PER_EPOCH / 2))
DEVICE="cuda"
GPU_NAME="rtx3090"
COMPILE_FLAG=""

RESUME_PATH=${5:-""} 
RESUME_ARG=""
if [ -f "$RESUME_PATH" ]; then
    RESUME_ARG="--resume $RESUME_PATH"
fi

# Pre-flight Clean
rm -rf /tmp/torchinductor_root/*
rm -rf ~/.triton/cache/*
pkill -9 python 2>/dev/null

echo "--- STARTING $MODEL_NAME ON $GPU_NAME ---"
echo "Detected Tokens: $TOTAL_TOKENS"
echo "TRAINING TYPE: $TRAINING_TYPE"
echo "GLOBAL_BATCH_SIZE: $GLOBAL_BATCH_SIZE"
echo "MICRO_BATCH_SIZE: $MICRO_BATCH_SIZE"
echo "ACTIVE_LR: $ACTIVE_LR"
echo "WARMUP_STEPS: $WARMUP_STEPS"
echo "Steps Per Epoch: $STEPS_PER_EPOCH"
echo "Total Target Steps: $MAX_STEPS over $EPOCHS Epochs"
echo "Save Interval: Every $SAVE_FREQ steps"
echo "----------------------------------------"

export PYTHONPATH=$PYTHONPATH:.
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

python3 -m trainer.train \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_DIR" \
    --ckpt_dir "$CKPT_BASE" \
    --lr "$ACTIVE_LR" \
    --batch_size "$MICRO_BATCH_SIZE" \
    --total_batch_size "$GLOBAL_BATCH_SIZE" \
    --max_steps "$MAX_STEPS" \
    --warmup_steps "$WARMUP_STEPS" \
    --weight_decay "$WEIGHT_DECAY" \
    --log_interval "$LOG_FREQ" \
    --save_interval "$SAVE_FREQ" \
    --device "$DEVICE" \
    $RESUME_ARG \
    --max_extra_mask  $extra_mask \
    --use_loss_controller \
    $COMPILE_FLAG
