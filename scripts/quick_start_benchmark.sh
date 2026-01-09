#!/bin/bash

# Quick Start Script for Qwen2.5 Benchmarking
# This script helps you get started with benchmarking quickly

set -e

# Allow overriding python executable (e.g., PYTHON_BIN=python).
PYTHON_BIN=${PYTHON_BIN:-python3}

echo "=========================================="
echo "Qwen2.5 Benchmarking Quick Start"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v "$PYTHON_BIN" &> /dev/null; then
    echo "Error: $PYTHON_BIN is not installed or not in PATH"
    exit 1
fi

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✓ CUDA detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠ Warning: CUDA not detected. Benchmarking will run on CPU (very slow)"
fi

echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p datasets
mkdir -p benchmark_results
mkdir -p benchmark_reports
mkdir -p logs/benchmark
mkdir -p logs/reports
echo "✓ Directories created"
echo ""

# Download sample datasets (optional)
echo "Do you want to download sample datasets? (y/n)"
read -r download_datasets

if [ "$download_datasets" = "y" ] || [ "$download_datasets" = "Y" ]; then
    echo "Downloading sample datasets..."
     "$PYTHON_BIN" -c "
 from benchmarks.aime_benchmark import download_aime_dataset
 from benchmarks.math_benchmark import download_math_dataset
 from benchmarks.gpqa_benchmark import download_gpqa_dataset
 download_aime_dataset('datasets/aime')
 download_math_dataset('datasets/math')
 download_gpqa_dataset('datasets/gpqa')
 print('✓ Sample datasets downloaded')
 "
else
    echo "Skipping dataset download. Benchmarks will use built-in sample data."
fi

echo ""

# Ask for benchmark configuration
echo "Select benchmark configuration:"
echo "1) Quick test (small models, few samples)"
echo "2) Standard evaluation (medium models, standard samples)"
echo "3) Full evaluation (all models, all samples)"
echo "4) Custom configuration"
read -r config_choice

case $config_choice in
    1)
        echo "Running quick test..."
        "$PYTHON_BIN" scripts/run_benchmarks.py \
            --model_sizes 0.5B \
            --benchmarks aime math \
            --output_dir benchmark_results/quick_test
        ;;
    2)
        echo "Running standard evaluation..."
        "$PYTHON_BIN" scripts/run_benchmarks.py \
            --model_sizes 0.5B 7B \
            --benchmarks aime math gpqa \
            --output_dir benchmark_results/standard
        ;;
    3)
        echo "Running full evaluation..."
        "$PYTHON_BIN" scripts/run_benchmarks.py \
            --output_dir benchmark_results/full
        ;;
    4)
        echo "Custom configuration selected."
        echo "Please edit configs/benchmark_config.yaml and run:"
        echo "$PYTHON_BIN scripts/run_benchmarks.py"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Benchmarking completed!"
echo "=========================================="
echo ""

# Generate reports
echo "Generating reports..."
"$PYTHON_BIN" scripts/aggregate_results.py --all --results_dir benchmark_results
echo "✓ Reports generated in benchmark_reports/"
echo ""

echo "Next steps:"
echo "1. View results: cat benchmark_reports/summary_report.txt"
echo "2. View plots: ls benchmark_reports/plots/"
echo "3. For more information, see docs/BENCHMARKING_GUIDE.md"
echo ""
echo "Happy benchmarking!"
