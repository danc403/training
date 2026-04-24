#!/bin/bash
touch /root/.no_auto_tmux

# 1. Update system and install base tools
apt-get update
apt-get install -y build-essential dos2unix python3-venv python3-pip python-is-python3 nano

# 2. Create and enter the virtual environment
python3 -m venv train_venv
source train_venv/bin/activate

# 3. Foundation Phase: Install NumPy first to lock C-API
pip install --upgrade pip
pip install "numpy<2.1.0"

# 4. Core Phase: Install the cu128 Torch build
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# 5. Requirements Phase: Install everything else
pip install -r requirements.txt

echo "Finished setting up system for training."
