# Qwen2.5 Benchmarking Guide

This guide provides comprehensive instructions for benchmarking Qwen2.5 models on difficult benchmarks including AIME, MATH, LiveCodeBench, SWE-Bench, and GPQA.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running Benchmarks](#running-benchmarks)
5. [Supported Benchmarks](#supported-benchmarks)
6. [Results and Reporting](#results-and-reporting)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

## Overview

The benchmarking framework supports:

- **Multiple Model Sizes**: configurable via [`configs/benchmark_config.yaml`](../configs/benchmark_config.yaml:1)
  (the default config includes 0.5B, 7B, 14B, 72B).
- **Multi-GPU Support**: Automatic distribution across 1-8 GPUs (via Hugging Face device maps)
- **Five Difficult Benchmarks**:
  - AIME (American Invitational Mathematics Examination)
  - MATH (Mathematical problem solving)
  - LiveCodeBench (Code generation and execution)
  - SWE-Bench (Software engineering tasks)
  - GPQA (Graduate-level multiple choice questions)

## Installation

These instructions assume you're running on a **remote SSH GPU server** and the
repo is cloned under your home directory (e.g. `~/DLProj`).

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 1-8 GPUs with sufficient VRAM (see model requirements below)

### Install Dependencies

```bash
cd ~/DLProj

# 1) Install a CUDA-enabled PyTorch wheel first.
# Pick ONE index URL that matches your CUDA driver/toolkit.
# - CUDA 12.4 (recommended on newer 50-series boxes):
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

# - If your server is on CUDA 12.1 instead, use:
# pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# Verify CUDA is visible to torch:
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'is_available', torch.cuda.is_available())"

# 2) Install the rest of repo dependencies.
pip install -r requirements.txt

# Optional (only if you need GPTQ tooling):
# pip install -r requirements_gptq.txt
```

### Model VRAM Requirements

These numbers vary significantly with quantization, attention kernels, sequence
length, and batch size. The default runner is typically used with 4-bit
BitsAndBytes quantization enabled (base model quantized; adapters in fp16).

As a starting point (4-bit base model, rough estimates):

| Model Size | 1 GPU | 2 GPUs | 4 GPUs | 8 GPUs |
|------------|-------|--------|--------|--------|
| 0.5B       | ~1 GB | -      | -      | -      |
| 7B         | ~4 GB | ~2 GB  | -      | -      |
| 14B        | ~8 GB | ~4 GB  | ~2 GB  | -      |
| 72B        | ~40 GB | ~20 GB | ~10 GB | ~5 GB |

If you hit OOM, the first levers to adjust are:
- reduce `batch_size` (per-model),
- reduce `max_length` / `max_new_tokens`,
- increase `gpu.num_gpus`,
- keep `gpu.quantization.enabled: true`.

### Why VRAM may only drop ~2× when “quantizing weights”

Weight quantization reduces the *weights* footprint, but total VRAM during
generation includes:

- **KV cache** (often the dominant term): scales with sequence length, batch
  size, hidden size, number of layers.
- **Activations** and temporary buffers.
- **Adapter weights** (usually small) and framework overhead.

So end-to-end VRAM reduction is often smaller than the raw 4× weight reduction.
To reduce VRAM further, prefer cutting **sequence length / max_new_tokens** over
micro-optimizing adapter ranks.

## Configuration

The benchmarking configuration is stored in [`configs/benchmark_config.yaml`](../configs/benchmark_config.yaml).

### Key Configuration Sections

#### Model Configuration

```yaml
models:
  - name: "Qwen/Qwen2.5-7B-Instruct"
    size: "7B"
    max_length: 4096
    batch_size: 2
```

#### GPU Configuration

```yaml
gpu:
  num_gpus: 2  # Number of GPUs to use
  device_map: "auto"  # or "balanced", "sequential"
  torch_dtype: "float16"  # or "bfloat16", "float32"
```

#### Benchmark Configuration

Each benchmark has its own configuration:

```yaml
benchmarks:
  aime:
    enabled: true
    num_samples: 250
    max_length: 2048
    temperature: 0.0
    metrics:
      - "accuracy"
      - "pass_at_1"
```

## Running Benchmarks

### Basic Usage

Run all benchmarks on all configured models:

```bash
cd ~/DLProj
python src/evaluation/runner.py
```

### Baseline comparison (native 4-bit quant + 4-bit LoRA / QLoRA-style adapter)

The repo can optionally run additional **baseline variants** for each selected
model size:

- `baseline_4bit`: base model in 4-bit (no adapters)
- `baseline_4bit_lora`: base model in 4-bit + a PEFT adapter (`adapter_path`)

This is **opt-in** because it increases runtime.

```bash
cd ~/DLProj
python src/evaluation/runner.py --run_baselines
```

For the LoRA/QLoRA baseline, ensure [`configs/benchmark_config.yaml`](../configs/benchmark_config.yaml:1)
sets `baselines.quantization_4bit_lora.adapter_path` to a valid adapter folder (local path) or a HuggingFace Hub repo id.

### Run Specific Model Sizes

```bash
cd ~/DLProj
python src/evaluation/runner.py --model_sizes 0.5B 7B 32B
```

### Run Specific Benchmarks

```bash
cd ~/DLProj
python src/evaluation/runner.py --benchmarks aime math gpqa
```

### Custom Configuration

```bash
cd ~/DLProj
python src/evaluation/runner.py --config configs/benchmark_config.yaml
```

### Custom Output Directory

```bash
cd ~/DLProj
python src/evaluation/runner.py --output_dir my_benchmark_results
```

### Complete Example

```bash
# Run AIME and MATH benchmarks on 7B and 32B models
python src/evaluation/runner.py \
  --model_sizes 7B 32B \
  --benchmarks aime math \
  --output_dir results/qwen_benchmarks
```

## Supported Benchmarks

### AIME (American Invitational Mathematics Examination)

- **Type**: Mathematical problem solving
- **Dataset Size**: 250 problems
- **Evaluation**: Exact match
- **Difficulty**: High school competition level
- **Configuration**: [`configs/benchmark_config.yaml`](../configs/benchmark_config.yaml) (aime section)

**Example**:
```bash
python src/evaluation/runner.py --benchmarks aime
```

### MATH

- **Type**: Mathematical problem solving
- **Dataset Size**: 5,000 problems
- **Evaluation**: Exact match
- **Difficulty**: High school to undergraduate
- **Subjects**: Algebra, Calculus, Geometry, Number Theory, etc.

**Example**:
```bash
python src/evaluation/runner.py --benchmarks math
```

### LiveCodeBench

- **Type**: Code generation and execution
- **Dataset Size**: 500 problems (configurable)
- **Evaluation**: Code execution against test cases
- **Difficulty**: Easy to Hard
- **Languages**: Python (primary)

**Example**:
```bash
python src/evaluation/runner.py --benchmarks livecodebench
```

### SWE-Bench (Software Engineering Benchmark)

- **Type**: Software engineering tasks
- **Dataset Size**: 300 problems (configurable)
- **Evaluation**: Patch application and test execution
- **Difficulty**: Medium to Hard
- **Focus**: Real-world GitHub issues

**Example**:
```bash
python src/evaluation/runner.py --benchmarks swe_bench
```

### GPQA (Graduate-Level Google-Proof Q&A)

- **Type**: Multiple choice questions
- **Dataset Size**: 548 questions
- **Evaluation**: Multiple choice accuracy
- **Difficulty**: Graduate-level
- **Subjects**: Biology, Chemistry, Physics, Computer Science, etc.

**Example**:
```bash
python src/evaluation/runner.py --benchmarks gpqa
```

## Results and Reporting

### Result Structure

Results are saved in the configured output directory:

```
benchmark_results/
├── benchmark_20240109_120000/
│   ├── final_results.json
│   ├── final_results.csv
│   ├── intermediate_7B_aime.json
│   ├── intermediate_7B_math.json
│   └── ...
```

### Generate Reports

After running benchmarks, generate comprehensive reports:

```bash
# Generate all reports
python scripts/aggregate_results.py --all --results_dir benchmark_results

# Generate specific reports
python scripts/aggregate_results.py \
  --generate_summary \
  --generate_csv \
  --generate_plots \
  --results_dir benchmark_results
```

### Report Types

1. **Summary Report** (`summary_report.txt`): Text-based summary of all results
2. **CSV Report** (`results.csv`): Machine-readable format for further analysis
3. **LaTeX Report** (`report.tex`): Academic paper format
4. **Plots** (`plots/`): Visualizations including:
   - Accuracy bar charts
   - Heatmaps
   - Model comparison plots

### Example Report Output

```
================================================================================
BENCHMARK RESULTS SUMMARY
================================================================================
Generated: 2024-01-09 12:00:00
Results directory: benchmark_results
================================================================================

================================================================================
PERFORMANCE SUMMARY
================================================================================

Model Size  Benchmark    Accuracy
----------- ------------ ----------
0.5B        AIME         12.50%
0.5B        MATH         15.20%
7B          AIME         35.80%
7B          MATH         42.30%
32B         AIME         52.40%
32B         MATH         58.70%
```

## Troubleshooting

### Out of Memory Errors

**Problem**: CUDA out of memory error

**Solutions**:
1. Reduce batch size in configuration
2. Use smaller model size
3. Enable gradient checkpointing
4. Use 2 GPUs instead of 1

```yaml
models:
  - name: "Qwen/Qwen2.5-7B-Instruct"
    batch_size: 1  # Reduce from 2
```

### Slow Performance

**Problem**: Benchmarking is taking too long

**Solutions**:
1. Reduce number of samples
2. Increase batch size (if memory allows)
3. Use smaller model for testing
4. Enable flash attention

```yaml
benchmarks:
  aime:
    num_samples: 50  # Reduce from 250
```

### Dataset Not Found

**Problem**: Dataset not found error

**Solutions**:
1. Download dataset using provided functions
2. Place dataset in correct location
3. Update dataset path in configuration

**Important (paper runs)**:
Some benchmark configs (e.g. [`configs/benchmark_config_4x5090.yaml`](configs/benchmark_config_4x5090.yaml:1)) set `allow_sample_data: false` to prevent silently falling back to tiny synthetic “sample datasets”.
If a dataset is missing or can’t be downloaded, the run will now fail fast so you don’t accidentally report numbers computed on 5–10 toy items.

```python
from benchmarks.aime_benchmark import download_aime_dataset
download_aime_dataset('datasets/aime')
```

Or download all supported benchmark datasets at once:

```bash
python3 src/data/downloader.py
```

Select specific datasets / output dir / token:

```bash
python3 src/data/downloader.py --datasets aime math gpqa livecodebench --output_dir datasets
python3 src/data/downloader.py --hf_token "$HF_TOKEN"
```

If your machine cannot reach `huggingface.co` directly (common behind corporate firewalls or in some regions), use a Hub mirror endpoint:

```bash
python3 src/data/downloader.py --hf_endpoint https://hf-mirror.com
```

### Notes on dataset availability

- Some dataset repo ids can change over time.
- Some datasets (notably **GPQA**) may be **gated**, requiring a HuggingFace token.
- The download helpers support overriding dataset ids via environment variables:
  - `BENCH_MATH_HF_DATASET` (default: `qwedsacf/competition_math`)
  - `BENCH_AIME_HF_DATASET` (default: `Maxwell-Jia/AIME_2024`)
  - `BENCH_GPQA_HF_DATASET` (default: `idavidrein/gpqa-extended`)
  - `BENCH_LIVECODEBENCH_HF_DATASET` (default: `livecodebench/code_generation_lite`)

If a download returns 0 samples, rerun with `--hf_token` (for gated datasets) or set the appropriate `BENCH_*_HF_DATASET` variable to a valid mirror.

### Import Errors

**Problem**: Module not found errors

**Solutions**:
1. Install all dependencies: `pip install -r requirements.txt`
2. Ensure you're in the project root directory
3. Add project to PYTHONPATH: `export PYTHONPATH=$PYTHONPATH:/path/to/project`

## Advanced Usage

### Custom Benchmarks

Create a custom benchmark by extending the [`BaseBenchmark`](../benchmarks/base_benchmark.py) class:

```python
from benchmarks.base_benchmark import BaseBenchmark

class MyCustomBenchmark(BaseBenchmark):
    def _load_dataset(self):
        # Load your dataset
        return [...]
    
    def _format_prompt(self, sample):
        # Format prompt for your task
        return f"Question: {sample['question']}"
    
    def _extract_answer(self, response, sample):
        # Extract answer from model response
        return response.strip()
    
    def _evaluate_sample(self, prediction, ground_truth):
        # Evaluate prediction
        return prediction == ground_truth
```

### Multi-GPU Optimization

For optimal multi-GPU performance:

```yaml
gpu:
  num_gpus: 2
  device_map: "balanced"  # Better for 2 GPUs
  torch_dtype: "bfloat16"  # Better for newer GPUs
```

### Batch Processing

Process multiple benchmarks in parallel:

```bash
# Run benchmarks in background
nohup python src/evaluation/runner.py --model_sizes 7B > benchmark_7B.log 2>&1 &

# Monitor progress
tail -f benchmark_7B.log
```

### Resume from Checkpoint

The framework automatically saves intermediate results. To resume:

```bash
# Results are saved incrementally
# Check intermediate files in output directory
ls benchmark_results/benchmark_*/intermediate_*
```

### Performance Profiling

Enable detailed performance tracking:

```yaml
performance:
  track_memory: true
  track_latency: true
  track_throughput: true
  log_interval: 5  # Log every 5 samples
```

## Best Practices

1. **Start Small**: Test with small models and few samples first
2. **Monitor Resources**: Use `nvidia-smi` to monitor GPU usage
3. **Save Results**: Always keep backup of result files
4. **Document Configurations**: Keep track of configuration changes
5. **Use Version Control**: Track benchmark code and configurations
6. **Validate Results**: Cross-check with known baselines

## Citation

If you use this benchmarking framework, please cite:

```bibtex
@software{qwen_benchmarking,
  title = {Qwen2.5 Benchmarking Framework},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/qwen-benchmarking}
}
```

## Support

For issues and questions:
- Check the [Troubleshooting](#troubleshooting) section
- Review configuration files in [`configs/`](../configs/)
- Examine logs in `logs/benchmark/`
- Open an issue on GitHub

## License

This benchmarking framework is provided under the same license as the Qwen2.5 models.
