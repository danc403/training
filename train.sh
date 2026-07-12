#!/bin/bash

# iDragonfly Training Launcher: Optimized

DATA_DIR="./base"
CKPT_BASE="./checkpoints"
extra_mask="0.235" #percent extra random mask (decays to 0.0 by start of last epoch)

# --- Data Loader Configuration ---
# Set FORCE_CPU_LOADER to "true" to pin the dataset in System RAM, bypassing VRAM limits.
# Set to "false" to allow standard GPU-resident data loading.
FORCE_CPU_LOADER="false"

# Inputs from Command Line
MODEL_NAME=${1:-"sprite"}      # Defaults to sprite if $1 is empty
TRAINING_TYPE=${2:-"base"}    # Defaults to base if $2 is empty
EPOCHS=${3:-4}                # Defaults to 4 if $3 is empty

# --- Auto-detect GPU and VRAM ---
if command -v nvidia-smi &> /dev/null; then
    DEVICE="cuda"
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n 1)
    VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    VRAM_GB=$((VRAM_MIB / 1024))
elif command -v amd-smi &> /dev/null; then
    DEVICE="rocm"
    # Extract GPU Name
    GPU_NAME=$(amd-smi static -g 0 --json | jq -r '.gpu_data[0].asic.market_name')

    # Extract VRAM size in MB
    VRAM_MB=$(amd-smi static -g 0 --json | jq -r '.gpu_data[0].vram.size.value')

    # Convert to GB (Integer division)
    VRAM_GB=$(( (VRAM_MB + 1023) / 1024 ))
else
    DEVICE="cpu" 
    GPU_NAME="None/CPU"
    VRAM_GB=0
fi

# After calculating VRAM_GB
if [ -z "$VRAM_GB" ] || [ "$VRAM_GB" -eq 0 ]; then
    echo "Warning: Failed to detect VRAM via amd-smi. Defaulting to 8GB safety mode."
    VRAM_GB=8
fi

# Define Log File Name early
LOG_FILE="${MODEL_NAME}_${TRAINING_TYPE}.log"

# --- Use exec for global redirection instead of braces ---
exec > >(tee -a "$LOG_FILE") 2>&1

# --- Dynamic Token Calculation ---
# Calculate total tokens from all data shards (2 bytes per token)
TOTAL_BYTES=$(stat -c%s ${DATA_DIR}/*_data.bin 2>/dev/null | awk '{s+=$1} END {print s}')
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
        GLOBAL_BATCH_SIZE=65536;
        MICRO_BATCH_SIZE=10;
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

echo "--- STARTING $MODEL_NAME ON $GPU_NAME ($DEVICE) ---"
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

# --- Optimized Environment Setup ---
export PYTHONPATH=$PYTHONPATH:.
export PYTHONUNBUFFERED=1

if [ "$DEVICE" == "cuda" ]; then
    # NVIDIA Specifics
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
    export CUDA_DEVICE_ORDER="PCI_BUS_ID"
elif [ "$DEVICE" == "rocm" ]; then
    # AMD/ROCm Specifics
    export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:128"
    # Essential for many RDNA3 cards (like your 7600 XT) to avoid "gfx" mismatches
    if [ -z "$HSA_OVERRIDE_GFX_VERSION" ]; then
        export HSA_OVERRIDE_GFX_VERSION=11.0.2
    fi
    # Helps stability of the ROCm caching allocator
    export HIP_FORCE_DEV_KERNELS=1
    # Force the loader to prefer the ROCm-specific libraries first
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libamdhip64.so.6
fi

# # Commented out hacked force cuda
# DEVICE="cuda"

rm -rf ~/.cache/torch/kernels
rm -rf ~/.cache/triton
rm -rf ~/.triton/cache
rm -rf /tmp/torchinductor_*

# Prepare CPU loader flag if enabled
LOADER_ARG=""
if [ "$FORCE_CPU_LOADER" == "true" ]; then
    LOADER_ARG="--force_cpu_loader"
fi

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
    $LOADER_ARG \
    $RESUME_ARG \
    --max_extra_mask  $extra_mask \
    --use_loss_controller \
    $COMPILE_FLAG
