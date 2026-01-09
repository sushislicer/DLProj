#!/bin/bash

# Fast Benchmarking Script for Qwen2.5 Models
# Optimized for 4/8 GPU clusters with 4-bit quantization
# Completes within 2-4 hours

set -e

# Allow overriding python executable (e.g., PYTHON_BIN=python).
PYTHON_BIN=${PYTHON_BIN:-python3}

echo "=========================================="
echo "Qwen2.5 Fast Benchmarking (4-bit Quantized)"
echo "=========================================="
echo ""

# Default values
NUM_GPUS=${NUM_GPUS:-4}
TIME_BUDGET=${TIME_BUDGET:-4}  # hours
QUANTIZE=${QUANTIZE:-true}
DEBUG=${DEBUG:-false}
RUN_BASELINES=${RUN_BASELINES:-false}
AUTO_INSTALL_FLASH_ATTN=${AUTO_INSTALL_FLASH_ATTN:-true}

echo "Configuration:"
echo "  Number of GPUs: $NUM_GPUS"
echo "  Time Budget: ${TIME_BUDGET} hours"
echo "  Quantization: $QUANTIZE"
echo "  Debug: $DEBUG"
echo "  Run baselines (4-bit / 4-bit QLoRA): $RUN_BASELINES"
echo "  Auto-install FlashAttention2: $AUTO_INSTALL_FLASH_ATTN"
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
    echo ""
else
    echo "⚠ Warning: CUDA not detected. Benchmarking will run on CPU (very slow)"
    echo ""
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p datasets
mkdir -p benchmark_results
mkdir -p benchmark_reports
mkdir -p logs/benchmark
mkdir -p logs/reports
mkdir -p cache
echo "✓ Directories created"
echo ""

# Run benchmarks with 4-bit quantization
echo "Starting fast benchmarking with 4-bit quantization..."
echo "This will run 4 models (0.5B, 7B, 14B, 72B) on 5 benchmarks"
echo "Expected completion time: 2-4 hours"
echo ""

CMD=("$PYTHON_BIN" scripts/run_benchmarks.py \
    --config configs/benchmark_config.yaml \
    --num_gpus $NUM_GPUS \
    --time_budget $TIME_BUDGET \
    --quantize)

if [ "$AUTO_INSTALL_FLASH_ATTN" = "true" ]; then
  export AUTO_INSTALL_FLASH_ATTN=1
fi

if [ "$DEBUG" = "true" ]; then
  CMD+=(--debug)
fi

if [ "$RUN_BASELINES" = "true" ]; then
  CMD+=(--run_baselines)
fi

"${CMD[@]}"

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

echo "Results summary:"
echo "  - Results: benchmark_results/"
echo "  - Reports: benchmark_reports/"
echo "  - Logs: logs/benchmark/"
echo ""
echo "Next steps:"
echo "1. View summary: cat benchmark_reports/summary_report.txt"
echo "2. View plots: ls benchmark_reports/plots/"
echo "3. View CSV: cat benchmark_reports/results.csv"
echo ""
echo "For more information, see docs/BENCHMARKING_GUIDE.md"
