# Qwen2.5 Quantization + Benchmarking Toolkit

This repository provides a unified pipeline for **quantization-aware adaptation** (PiSSA + SpinQuant + GaLore) and a comprehensive **benchmarking suite** for Qwen2.5 models on reasoning tasks (AIME, MATH, GPQA, LiveCodeBench).

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
# Optional: Install flash-attn if supported
pip install flash-attn --no-build-isolation
```

### 2. Benchmarking
Run the lightweight evaluation script (0.5B/7B models, reduced samples):
```bash
python scripts/run_lightweight_eval.py
```

Run the full benchmark suite (all models, full datasets):
```bash
python scripts/run_benchmarks.py --model_sizes 7B 14B --quantize --run_baselines
```

See [`docs/BENCHMARKING_GUIDE.md`](docs/BENCHMARKING_GUIDE.md) for detailed configuration.

### 3. Pipeline (Training)
Run the full quantization and adaptation pipeline:
```bash
python scripts/pipeline.py --config configs/pipeline_config.yaml
```

## Architecture

The pipeline integrates three techniques to optimize for **quantized deployment**:
1.  **PiSSA**: Initializes adapters using principal singular components for faster convergence.
2.  **SpinQuant**: Applies rotation matrices (e.g., Hadamard) to the base model before quantization to reduce outliers.
3.  **GaLore**: Projects gradients onto a low-rank subspace during training to save memory.

## Results & Visualization

Aggregate results and generate plots:
```bash
python scripts/aggregate_results.py --plot
```

## Citation

If you use this codebase, please cite the original Qwen2.5, PiSSA, SpinQuant, and GaLore papers.
