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

# Install core dependencies
echo ""
echo "Installing core dependencies..."
pip install -r requirements.txt

# Install PiSSA
echo ""
echo "Installing PiSSA..."
pip install git+https://github.com/hiyouga/PiSSA.git

# Install GaLore
echo ""
echo "Installing GaLore..."
pip install galore-torch

# Install SpinQuant
echo ""
echo "Installing SpinQuant..."
pip install git+https://github.com/ModelTC/SpinQuant.git

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
