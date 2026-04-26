#!/bin/bash

# iDragonfly Training Launcher: Optimized with Automated Step Detection
DEVICE="cuda"

DATA_DIR="./instruct"
CKPT_BASE="./checkpoints"

# Inputs from Command Line
MODEL_NAME=${1:-"sprite"}      # Defaults to sprite if $1 is empty
TRAINING_TYPE=${2:-"base"}    # Defaults to base if $2 is empty
EPOCHS=${3:-4}                # Defaults to 4 if $3 is empty (interpreted as additional epochs if resuming)
# --- Auto-detect GPU and VRAM ---
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n 1)
    # Get total memory in MiB and convert to GB (integer)
    VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    VRAM_GB=$((VRAM_MIB / 1024))
else
    GPU_NAME="Unknown"
    VRAM_GB=${4:-24} # Fallback to manual input or 24
fi

RESUME_PATH=${5:-""} 
freeze_arg=${6:-""}

LOG_FILE="${MODEL_NAME}_${TRAINING_TYPE}.log"

# --- Use exec for global redirection instead of braces ---
exec > >(tee -a "$LOG_FILE") 2>&1

# --- Dynamic Token Calculation ---
# Calculate total tokens from all data shards (2 bytes per token)
TOTAL_BYTES=$(stat -c%s ${DATA_DIR}/*_data.bin | awk '{s+=$1} END {print s}')
TOTAL_TOKENS=$((TOTAL_BYTES / 2))
no_opt="--no_opt"

# --- 1. Set Global Batch Size based on Type ---
# SWAPPED logic: Using 65k for tune to increase update frequency for instructions.
if [ "$TRAINING_TYPE" == "tune" ]; then
    GLOBAL_BATCH_SIZE=65536
else
    GLOBAL_BATCH_SIZE=32768  #65536 #131072  #262144
fi

# --- 2. Set Micro-Batch Size based on Model & VRAM ---
case $MODEL_NAME in
    "sprite")
        if [ "$VRAM_GB" -le 12 ]; then GLOBAL_BATCH_SIZE=32768; else GLOBAL_BATCH_SIZE=65536; fi
        if [ "$VRAM_GB" -le 12 ]; then MICRO_BATCH_SIZE=8; else MICRO_BATCH_SIZE=16; fi
        LEARNING_RATE=0.0003
        ;;
    "nymph")
        if [ "$VRAM_GB" -le 12 ]; then GLOBAL_BATCH_SIZE=32768; else GLOBAL_BATCH_SIZE=65536; fi
        if [ "$VRAM_GB" -le 12 ]; then MICRO_BATCH_SIZE=4; else MICRO_BATCH_SIZE=8; fi
        LEARNING_RATE=0.0003
        ;;
    "dragonfly")
        if [ "$VRAM_GB" -le 12 ]; then GLOBAL_BATCH_SIZE=32768; else GLOBAL_BATCH_SIZE=65536; fi
        if [ "$VRAM_GB" -le 12 ]; then MICRO_BATCH_SIZE=4; else MICRO_BATCH_SIZE=8; fi
        LEARNING_RATE=0.0003
        ;;
    "wyrm")
        if [ "$VRAM_GB" -le 12 ]; then GLOBAL_BATCH_SIZE=32768; else GLOBAL_BATCH_SIZE=65536; fi
        if [ "$VRAM_GB" -le 12 ]; then MICRO_BATCH_SIZE=4; else MICRO_BATCH_SIZE=8; fi
        LEARNING_RATE=0.0003
        ;;
esac

# --- 3. Step & Resume Calculation ---
STEPS_PER_EPOCH=$((TOTAL_TOKENS / GLOBAL_BATCH_SIZE))

RESUME_ARG=""
COMPLETED_STEPS=0
if [ -f "$RESUME_PATH" ]; then
    RESUME_ARG="--resume $RESUME_PATH"
    # Extract global_step or step from the checkpoint dictionary
    COMPLETED_STEPS=$(python3 -c "import torch; ckpt=torch.load('$RESUME_PATH', map_location='cpu', weights_only=True); print(ckpt.get('global_step', ckpt.get('step', 0)))")
    echo "Checkpoint detected. Resuming from step: $COMPLETED_STEPS"
fi

# Calculate Target: Argument $3 is "How many more epochs to run"
ADDITIONAL_STEPS=$((STEPS_PER_EPOCH * EPOCHS))
MAX_STEPS=$((COMPLETED_STEPS + ADDITIONAL_STEPS))

# Adjust Warmup based on the new steps being added
WARMUP_STEPS=200 #$((ADDITIONAL_STEPS / 10))
WEIGHT_DECAY=0.05

# 4. Frequency Control
LOG_FREQ=15
SAVE_FREQ=$((STEPS_PER_EPOCH / 2))

# 5. Hardware & Optimization Flags
COMPILE_FLAG=""

# Pre-flight Clean
rm -rf /tmp/torchinductor_root/*
rm -rf ~/.triton/cache/*
pkill -9 python 2>/dev/null

echo "--- STARTING $MODEL_NAME ON $GPU_NAME ---"
echo "Detected Tokens: $TOTAL_TOKENS"
echo "GLOBAL_BATCH_SIZE: $GLOBAL_BATCH_SIZE"
echo "MICRO_BATCH_SIZE: $MICRO_BATCH_SIZE"
echo "Steps Per Epoch: $STEPS_PER_EPOCH"
echo "Completed Progress: $COMPLETED_STEPS steps"
echo "Additional Steps to run: $ADDITIONAL_STEPS"
echo "Final Stop Target (MAX_STEPS): $MAX_STEPS"
echo "Save Interval: Every $SAVE_FREQ steps"
echo "----------------------------------------"

export PYTHONPATH=$PYTHONPATH:.
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

python3 -m trainer.tune \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_DIR" \
    --ckpt_dir "$CKPT_BASE" \
    --lr "$LEARNING_RATE" \
    --batch_size "$MICRO_BATCH_SIZE" \
    --total_batch_size "$GLOBAL_BATCH_SIZE" \
    --max_steps "$MAX_STEPS" \
    --warmup_steps "$WARMUP_STEPS" \
    --weight_decay "$WEIGHT_DECAY" \
    --log_interval "$LOG_FREQ" \
    --save_interval "$SAVE_FREQ" \
    --device "$DEVICE" \
    $RESUME_ARG \
    $freeze_arg \
    $no_opt \
    $COMPILE_FLAG
