#!/bin/bash

# Setup script for Qwen2.5 Quantization Pipeline
# This script installs all required dependencies

set -e

# Allow overriding python executable (e.g., PYTHON_BIN=python).
PYTHON_BIN=${PYTHON_BIN:-python3}

echo "=========================================="
echo "Qwen2.5 Quantization Pipeline Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$($PYTHON_BIN --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment (optional)
read -p "Do you want to create a virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Creating virtual environment..."
    $PYTHON_BIN -m venv venv
    source venv/bin/activate
    echo "Virtual environment activated"
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install CUDA-enabled PyTorch first (recommended on GPU servers)
echo ""
echo "Installing PyTorch (CUDA) ..."
if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
    echo "PyTorch not found; installing CUDA wheels (cu124)."
    echo "If your server uses CUDA 12.1, replace cu124 with cu121."
    pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
fi
"$PYTHON_BIN" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'is_available', torch.cuda.is_available())"

# Install core dependencies
echo ""
echo "Installing core dependencies..."
pip install -r requirements.txt

# Optional external libraries
echo ""
read -p "Install optional external PiSSA/SpinQuant/GaLore tooling (y/n)? " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Installing official PiSSA (GraphPKU/PiSSA)..."
    pip install git+https://github.com/GraphPKU/PiSSA.git

    echo "Installing GaLore optimizer package (galore-torch)..."
    pip install galore-torch

    echo "Installing official SpinQuant (ModelTC/SpinQuant)..."
    pip install git+https://github.com/ModelTC/SpinQuant.git
else
    echo "Skipping optional external tooling. Using in-repo implementations where applicable."
fi

# Create default configuration
echo ""
echo "Creating default configuration..."
$PYTHON_BIN pipeline.py --create_config


# Create output directories
echo ""
echo "Creating output directories..."
mkdir -p outputs/adapters
mkdir -p outputs/quantized_models
mkdir -p outputs/checkpoints
mkdir -p logs

echo ""
echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review and modify configs/pipeline_config.yaml"
echo "2. Run the pipeline: python pipeline.py"
echo ""
