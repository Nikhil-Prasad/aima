#!/bin/bash
# Setup script for GPU instances
# This can be called from SkyPilot or run manually

set -e

echo "Setting up Crown GPU environment..."

# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    build-essential \
    git \
    curl \
    htop \
    nvtop \
    tmux

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Setup Python environment
cd ~/crown
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
uv pip install -e ".[dev,rl]"

# Install additional GPU monitoring tools
pip install nvitop gpustat

# Create necessary directories
mkdir -p models
mkdir -p logs
mkdir -p checkpoints
mkdir -p data

# Setup Weights & Biases (if API key provided)
if [ ! -z "$WANDB_API_KEY" ]; then
    wandb login --relogin $WANDB_API_KEY
fi

# Configure Jupyter (optional)
# jupyter lab --generate-config
# echo "c.ServerApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_lab_config.py
# echo "c.ServerApp.allow_remote_access = True" >> ~/.jupyter/jupyter_lab_config.py

echo "Setup complete!"
echo ""
echo "To activate the environment:"
echo "  cd ~/crown && source venv/bin/activate"
echo ""
echo "GPU Status:"
nvidia-smi