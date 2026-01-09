#!/bin/bash

# Fast 14B Benchmark Runner
# - Optimized for ~2-4h wall-clock depending on hardware and enabled benchmarks
# - Optional baseline comparison: native 4-bit (no adapters) and 4-bit QLoRA

set -e

PYTHON_BIN=${PYTHON_BIN:-python3}

NUM_GPUS=${NUM_GPUS:-4}
TIME_BUDGET=${TIME_BUDGET:-4}
DEBUG=${DEBUG:-false}
RUN_BASELINES=${RUN_BASELINES:-false}

echo "=========================================="
echo "Qwen2.5 14B Fast Benchmark (4-bit Quantized)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Number of GPUs: $NUM_GPUS"
echo "  Time Budget: ${TIME_BUDGET} hours"
echo "  Debug: $DEBUG"
echo "  Run baselines (4-bit / 4-bit QLoRA): $RUN_BASELINES"
echo ""

mkdir -p datasets benchmark_results benchmark_reports logs/benchmark logs/reports cache

CMD=("$PYTHON_BIN" scripts/run_benchmarks.py \
  --config configs/benchmark_config.yaml \
  --model_sizes 14B \
  --num_gpus "$NUM_GPUS" \
  --time_budget "$TIME_BUDGET" \
  --quantize)

if [ "$DEBUG" = "true" ]; then
  CMD+=(--debug)
fi

if [ "$RUN_BASELINES" = "true" ]; then
  CMD+=(--run_baselines)
fi

"${CMD[@]}"

"$PYTHON_BIN" scripts/aggregate_results.py --all --results_dir benchmark_results

echo "=========================================="
echo "14B Benchmarking completed!"
echo "Results: benchmark_results/"
echo "Reports: benchmark_reports/"
echo "Logs: logs/benchmark/"
echo "=========================================="

