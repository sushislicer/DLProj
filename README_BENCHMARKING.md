# Qwen2.5 Benchmarking Framework

Comprehensive benchmarking framework for evaluating Qwen2.5 models on difficult benchmarks including AIME, MATH, LiveCodeBench, SWE-Bench, and GPQA.

## Features

- **Multi-Model Support**: Benchmark 4 representative Qwen2.5 model sizes (0.5B, 7B, 14B, 72B)
- **Multi-GPU Support**: Automatic distribution across 1-8 GPUs
- **4-Bit Quantization**: Base model quantization with adapters in full precision
- **Time Budget Management**: Complete benchmarks within 2-4 hours
- **Five Difficult Benchmarks**:
  - [AIME](#aime) - American Invitational Mathematics Examination
  - [MATH](#math) - Mathematical problem solving
  - [LiveCodeBench](#livecodebench) - Code generation and execution
  - [SWE-Bench](#swe-bench) - Software engineering tasks
  - [GPQA](#gpqa) - Graduate-level multiple choice questions
- **Comprehensive Reporting**: Generate text, CSV, LaTeX reports and visualizations
- **Flexible Configuration**: Easy customization via YAML configuration files

## Quick Start

These commands are intended to be run on your **remote SSH server** after you
clone the repo under your home directory (e.g. `~/DLProj`).

### Installation

```bash
# SSH to your server, then:
cd ~/DLProj

# Install CUDA-enabled PyTorch first (recommended on GPU servers)
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

# If your server uses CUDA 12.1 instead, use:
# pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# Verify CUDA is visible to torch
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'is_available', torch.cuda.is_available())"

# Install remaining dependencies
pip install -r requirements.txt

# Optional (only if you need GPTQ tooling)
# pip install -r requirements_gptq.txt
```

### Fast Benchmarking (Recommended)

Run optimized benchmarks with 4-bit quantization on 4 GPUs:

```bash
# From the repo root on the remote server:
cd ~/DLProj

# Make the script executable (Linux/Mac)
chmod +x scripts/run_fast_benchmark.sh

# Run fast benchmarking (4 GPUs, 4-bit quantization, 4-hour time budget)
./scripts/run_fast_benchmark.sh

 # Or customize:
 NUM_GPUS=8 TIME_BUDGET=2 ./scripts/run_fast_benchmark.sh

 # Optional: also run baseline variants (native 4-bit quant + 4-bit QLoRA).
 # Note: this increases runtime because it runs additional model variants.
 RUN_BASELINES=true ./scripts/run_fast_benchmark.sh
```

### Manual Quick Start

```bash
# From the repo root on the remote server:
cd ~/DLProj

# Run with 4-bit quantization on 4 GPUs
 python scripts/run_benchmarks.py \
   --num_gpus 4 \
   --quantize \
   --time_budget 4

 # Optional: include baselines (native 4-bit quant + 4-bit QLoRA)
 python scripts/run_benchmarks.py \
   --num_gpus 4 \
   --quantize \
   --time_budget 4 \
   --run_baselines

# Run specific models and benchmarks
  python scripts/run_benchmarks.py \
  --model_sizes 0.5B 7B \
  --benchmarks aime math gpqa \
  --num_gpus 4 \
  --quantize
```

## Project Structure

```
.
├── configs/
│   └── benchmark_config.yaml      # Main benchmarking configuration
├── benchmarks/
│   ├── __init__.py
│   ├── base_benchmark.py          # Base benchmark class
│   ├── aime_benchmark.py          # AIME benchmark implementation
│   ├── math_benchmark.py          # MATH benchmark implementation
│   ├── livecodebench_benchmark.py # LiveCodeBench implementation
│   ├── swe_bench_benchmark.py     # SWE-Bench implementation
│   └── gpqa_benchmark.py          # GPQA implementation
├── scripts/
│   ├── run_benchmarks.py          # Main benchmarking script
│   ├── run_fast_benchmark.sh      # Fast benchmarking script (recommended)
│   ├── aggregate_results.py       # Results aggregation and reporting
│   └── quick_start_benchmark.sh   # Quick start script
├── docs/
│   └── BENCHMARKING_GUIDE.md      # Comprehensive guide
├── benchmark_results/             # Output directory for results
└── benchmark_reports/             # Output directory for reports
```

## Usage

### Fast Benchmarking (Recommended)

```bash
# Run optimized benchmarks with 4-bit quantization
./scripts/run_fast_benchmark.sh

# Customize GPU count and time budget
NUM_GPUS=8 TIME_BUDGET=2 ./scripts/run_fast_benchmark.sh
```

### Advanced Usage

```bash
# Run all benchmarks on all configured models
python scripts/run_benchmarks.py

# Run specific model sizes
python scripts/run_benchmarks.py --model_sizes 0.5B 7B 14B

# Run specific benchmarks
python scripts/run_benchmarks.py --benchmarks aime math gpqa

# Enable/disable quantization
python scripts/run_benchmarks.py --quantize
python scripts/run_benchmarks.py --no_quantize

# Set number of GPUs
python scripts/run_benchmarks.py --num_gpus 8

# Set time budget (in hours)
python scripts/run_benchmarks.py --time_budget 2

# Custom configuration
python scripts/run_benchmarks.py --config configs/benchmark_config.yaml

# Complete example
python scripts/run_benchmarks.py \
  --model_sizes 7B 14B \
  --benchmarks aime math \
  --num_gpus 4 \
  --quantize \
  --time_budget 3 \
  --output_dir results/qwen_benchmarks
```

## Supported Benchmarks

### AIME

**American Invitational Mathematics Examination**

- **Type**: Mathematical problem solving
- **Dataset Size**: 50 problems (optimized for speed)
- **Evaluation**: Exact match
- **Difficulty**: High school competition level

```bash
python scripts/run_benchmarks.py --benchmarks aime --quantize
```

### MATH

**Mathematical problem solving**

- **Type**: Mathematical problem solving
- **Dataset Size**: 100 problems (optimized for speed)
- **Evaluation**: Exact match
- **Difficulty**: High school to undergraduate
- **Subjects**: Algebra, Calculus, Geometry, Number Theory, etc.

```bash
python scripts/run_benchmarks.py --benchmarks math --quantize
```

### LiveCodeBench

**Code generation and execution**

- **Type**: Code generation and execution
- **Dataset Size**: 50 problems (optimized for speed)
- **Evaluation**: Code execution against test cases
- **Difficulty**: Easy to Hard
- **Languages**: Python (primary)

```bash
python scripts/run_benchmarks.py --benchmarks livecodebench --quantize
```

### SWE-Bench

**Software Engineering Benchmark**

- **Type**: Software engineering tasks
- **Dataset Size**: 30 problems (optimized for speed)
- **Evaluation**: Patch application and test execution
- **Difficulty**: Medium to Hard
- **Focus**: Real-world GitHub issues

```bash
python scripts/run_benchmarks.py --benchmarks swe_bench --quantize
```

### GPQA

**Graduate-Level Google-Proof Q&A**

- **Type**: Multiple choice questions
- **Dataset Size**: 100 questions (optimized for speed)
- **Evaluation**: Multiple choice accuracy
- **Difficulty**: Graduate-level
- **Subjects**: Biology, Chemistry, Physics, Computer Science, etc.

```bash
python scripts/run_benchmarks.py --benchmarks gpqa --quantize
```

## Configuration

The main configuration file is [`configs/benchmark_config.yaml`](configs/benchmark_config.yaml).

### Key Configuration Options

#### Model Configuration

```yaml
models:
  - name: "Qwen/Qwen2.5-7B-Instruct"
    size: "7B"
    max_length: 4096
    batch_size: 4
    quantization: "4bit"  # 4-bit quantization
```

#### GPU Configuration

```yaml
gpu:
  num_gpus: 4  # Number of GPUs to use (1-8)
  device_map: "auto"  # or "balanced", "sequential"
  torch_dtype: "float16"
  quantization:
    enabled: true
    bits: 4  # 4-bit quantization
    quant_type: "nf4"
    double_quant: true
```

#### Execution Configuration

```yaml
execution:
  max_execution_time: 4  # 4 hours total
  enable_time_budget: true
  model_priority:
    - "3B"
    - "7B"
    - "14B"
    - "32B"
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
│   └── ...
```

### Generate Reports

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

1. **Summary Report** (`summary_report.txt`): Text-based summary
2. **CSV Report** (`results.csv`): Machine-readable format
3. **LaTeX Report** (`report.tex`): Academic paper format
4. **Plots** (`plots/`): Visualizations

## Performance Optimization

### 4-Bit Quantization

The framework uses 4-bit quantization (NF4) by default for fast execution.

Important nuance: 4-bit quantization mainly reduces **model weight** memory.
Total VRAM during generation is often dominated by **KV cache** (sequence length
× batch size × layers) and **activations**, which are *not* reduced by weight
quantization. In practice you may see total VRAM reduction closer to ~1.5–2.5×
depending on context length and batch size.

- **Weight memory reduction**: up to ~4× (vs fp16 weights)
- **End-to-end VRAM reduction**: often smaller due to KV cache/activations
- **Speed**: can improve, but is hardware/kernel dependent

### Multi-GPU Scaling

Automatic distribution across 1-8 GPUs:

| GPUs | 0.5B Model | 7B Model | 14B Model | 72B Model |
|------|----------|----------|-----------|-----------|
| 1    | ✓          | ✓        | ✗         | ✗         |
| 2    | ✓          | ✓        | ✓         | ✗         |
| 4    | ✓          | ✓        | ✓         | ✓         |
| 8    | ✓          | ✓        | ✓         | ✓         |

### Time Budget

The framework includes time budget management:

- Default: 4 hours
- Configurable via `--time_budget` flag
- Automatic prioritization of faster benchmarks
- Early stopping if time exceeded

## Model VRAM Requirements (4-bit Quantized)

These are rough *practical* starting points for **total VRAM** when running the
benchmark runner with 4-bit base weights. If you increase `max_length`/
`max_new_tokens` or batch size, KV cache grows and VRAM can jump significantly.

| Model Size | Single GPU | 2 GPUs | 4 GPUs | 8 GPUs |
|------------|------------|--------|--------|--------|
| 0.5B       | ~1 GB      | -      | -      | -      |
| 7B         | ~4 GB      | ~2 GB  | -      | -      |
| 14B        | ~8 GB      | ~4 GB  | ~2 GB  | -      |
| 72B        | ~40 GB     | ~20 GB | ~10 GB | ~5 GB  |

## Expected Execution Times

With 4-bit quantization on 4 GPUs:

| Model | AIME | MATH | LiveCodeBench | SWE-Bench | GPQA | Total |
|-------|------|------|---------------|-----------|------|-------|
| 0.5B  | 2m   | 5m   | 8m            | 10m       | 2m   | 27m   |
| 7B    | 10m  | 20m  | 30m           | 40m       | 10m  | 110m  |
| 14B   | 20m  | 40m  | 60m           | 80m       | 20m  | 220m  |
| 72B   | 32m  | 48m  | 60m           | **SKIPPED** | 24m  | 164m  |

**Total**: depends on enabled benchmarks/sample sizes and hardware.
**With time budget (4h)**: prioritizes smaller models first (see `execution.model_priority`).

## Troubleshooting

### Out of Memory Errors

Reduce batch size or use more GPUs:

```yaml
models:
  - name: "Qwen/Qwen2.5-7B-Instruct"
    batch_size: 2  # Reduce from 4
```

Or use more GPUs:

```bash
python scripts/run_benchmarks.py --num_gpus 8
```

### Slow Performance

Enable quantization and use more GPUs:

```bash
python scripts/run_benchmarks.py --quantize --num_gpus 8
```

### Dataset Not Found

Download dataset using provided functions:

```python
from benchmarks.aime_benchmark import download_aime_dataset
download_aime_dataset('datasets/aime')
```

## Documentation

For comprehensive documentation, see [`docs/BENCHMARKING_GUIDE.md`](docs/BENCHMARKING_GUIDE.md).

## Requirements

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 1-8 GPUs with sufficient VRAM
- See [`requirements.txt`](requirements.txt) for full dependencies

## License

This benchmarking framework is provided under the same license as the Qwen2.5 models.

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
- Check the [BENCHMARKING_GUIDE.md](docs/BENCHMARKING_GUIDE.md)
- Review configuration files in [`configs/`](configs/)
- Examine logs in `logs/benchmark/`
- Open an issue on GitHub
