# \# iDragonfly Training Pipeline

# 

# Version: 1.0.0 (April 2026)

# 

# \## Overview

# 

# The iDragonfly pipeline is a high-velocity, native PyTorch training system designed for the Nymph family of models. It is optimized for consumer-grade hardware (NVIDIA 40-series) and distributed headless environments. By utilizing a binary-native data ingestion strategy, the pipeline achieves world-class throughput and high Model Flops Utilization (MFU) without the overhead of on-the-fly tokenization or complex compilation stacks.

# 

# \## Core Architecture

# 

# The Nymph model family utilizes a modern Transformer-based architecture optimized for local inference:

# 

# \* \*\*Models:\*\* Sprite (22.9m), Nymph (50m), Dragonfly (65m), and Wyrm (135m).

# \* \*\*Attention:\*\* Grouped Query Attention (GQA) for efficient memory scaling.

# \* \*\*Stability:\*\* QK-Norm and RMSNorm integration for high-learning-rate stability.

# \* \*\*Activation:\*\* SwiGLU MLP.

# \* \*\*Efficiency:\*\* Designed for Zero-KV-Cache inference potential and rapid pre-training.

# 

# \## Performance Benchmarks

# 

# Based on an NVIDIA RTX 4060 8GB GPU:

# 

# \* \*\*Sprite:\*\* 60,000 to 80,000 tokens per second (85 percent MFU).

# \* \*\*Nymph/Dragonfly:\*\* \~38,000 tokens per second.

# \* \*\*Wyrm:\*\* \~28,000 tokens per second.

# 

# A full 3-epoch base pre-training run and 2-epoch instruct fine-tuning for the Sprite model can be completed in under one hour, producing coherent creative capabilities.

# 

# \## Data Strategy

# 

# The pipeline operates on a "Binary First" principle. Data must be pre-processed into binary shards aligned with the model context window before training begins. This preparatory phase eliminates CPU bottlenecks.

# 

# \* \*\*Dataset Repository:\*\* Official shards and raw data available at https://github.com/danc403/datasets

# \* \*\*Context Alignment:\*\* Binary datasets are created specifically for the target model's context window.

# 

# \## Installation and Requirements

# 

# \* \*\*Environment:\*\* Python 3.10+, PyTorch 2.1+.

# \* \*\*Hardware:\*\* NVIDIA GPU with bfloat16 support (Ampere, Ada, or Blackwell).

# \* \*\*Dependencies:\*\* Standard Torch stack (no external compilers required).

# 

# \## Usage

# 

# Training is managed via a hardware-aware shell script that automates VRAM calculations and accumulation steps.

# 

# \### Command Structure:

# `./launch \[model\_name] \[training\_type] \[epochs] \[vram\_gb]`

# 

# \### Examples:

# \* \*\*Sprite Base Training:\*\* `./launch sprite base 3 8`

# \* \*\*Nymph Instruct Tuning:\*\* `./launch nymph instruct 2 12`

# \* \*\*Wyrm Large-Scale Run:\*\* `./launch wyrm base 5 24`

# 

# \## Features

# 

# 1\.  \*\*Dual-Stack Optimizer:\*\* Uses Muon for 2D parameters (weights/filters) and AdamW for 1D parameters (embeddings/biases).

# 2\.  \*\*Loss Controller:\*\* Optional "Shock-and-Recovery" management for handling high-LR instabilities.

# 3\.  \*\*Automated Checkpointing:\*\* Saves state every half-epoch to prevent data loss.

# 4\.  \*\*Reporting:\*\* Detailed metrics at every 15-step interval, including Loss, Learning Rate, Pressure (masking), TPS, and MFU.

# 5\.  \*\*Native AMP:\*\* Leverages torch.amp.autocast for high-speed mixed precision.

# 

# \## Configuration

# 

# Model-specific hyperparameters (layers, hidden dimensions, head counts) are managed in:

# `trainer/config.py`

# 

# Advanced training logic and optimizer settings can be adjusted in:

# `trainer/trainer.config.py`

# 

# \## License

# 

# Private Development - iDragonfly Project.



