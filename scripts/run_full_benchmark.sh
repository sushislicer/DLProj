#!/bin/bash

# Full Benchmarking Script for Qwen2.5 (0.5B & 7B)
# ------------------------------------------------
# This script runs the complete pipeline and benchmarking suite.
#
# Pipeline Stages:
# 1. PiSSA Adapter Extraction
# 2. SpinQuant Quantization (Rotations disabled for compatibility)
# 3. GaLore Training (on Alpaca + Math + Code datasets)
#
# Benchmarks:
# - AIME, MATH, GPQA, LiveCodeBench
# - Baselines: 4-bit Quantization, 4-bit LoRA
#
# Configuration:
# - Benchmark Config: configs/benchmark_config_4x5090.yaml
# - Pipeline Config: configs/pipeline_config.yaml

# Set environment variables for stability
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_LAUNCH_BLOCKING=1  # Uncomment for debugging CUDA errors

echo "Starting Full Benchmark Run..."
echo "Models: 0.5B, 7B"
echo "GPUs: 4"
echo "Time Budget: 4 hours"

python3 scripts/run_benchmarks.py \
    --config configs/benchmark_config_4x5090.yaml \
    --model_sizes 0.5B 7B \
    --num_gpus 4 \
    --time_budget 4 \
    --quantize \
    --run_pipeline \
    --pipeline_config configs/pipeline_config.yaml \
    --pipeline_output_root outputs/pipeline \
    --run_baselines

echo "Benchmarking completed."
