# Qwen2.5 Quantization + Benchmarking Toolkit

This repository contains two related but distinct toolchains:

1. **Quantization / finetuning pipeline** (PiSSA → SpinQuant → GaLore) for producing a *quantized base model + trained adapters*.
2. **Benchmarking framework** for evaluating Qwen2.5 models on difficult benchmarks (AIME, MATH, LiveCodeBench, SWE-Bench, GPQA), with optional 4-bit BitsAndBytes quantization.

If you only want to benchmark models, start at [`README_BENCHMARKING.md`](README_BENCHMARKING.md).

## Overview

This pipeline implements a three-stage quantization workflow:

1. **PiSSA (Principal Singular values and Singular Vectors Adaptation)**: Extracts low-rank adapters and residual base model
2. **SpinQuant**: Quantizes and freezes the residual base model
3. **GaLore (Gradient Low-Rank Projection)**: Trains the non-quantized adapters with memory-efficient optimization

## Benchmarking (separate entrypoint)

The benchmarking framework lives under [`scripts/run_benchmarks.py`](scripts/run_benchmarks.py:1) and [`benchmarks/`](benchmarks/__init__.py:1).

- Quick start: [`README_BENCHMARKING.md`](README_BENCHMARKING.md)
- Full guide: [`docs/BENCHMARKING_GUIDE.md`](docs/BENCHMARKING_GUIDE.md:1)

Important distinction:
- The benchmarking runner's `--quantize` flag uses **BitsAndBytes 4-bit quantization**.
- The pipeline's SpinQuant stage is a *separate workflow* implemented in [`scripts/spinquant.py`](scripts/spinquant.py:1).
  This repo ships a **SpinQuant-lite** backend (blockwise Givens rotations +
  weight-domain fake-quant objective). It also supports an **activation-weighted**
  calibration objective and optional **bitsandbytes 4/8-bit reload**, plus a
  **post-quant distillation** step during GaLore training.

## External dependencies (PiSSA / SpinQuant / GaLore)

This repo can be used in two modes:

1) **Self-contained (default)**: the pipeline + benchmarking run using the code
   inside this repo.
   - "PiSSA" stage: implemented via PEFT/LoRA in [`scripts/pissa_extraction.py`](scripts/pissa_extraction.py:1)
   - "SpinQuant" stage: implemented in-repo in [`scripts/spinquant.py`](scripts/spinquant.py:1)
   - "GaLore" stage: training loop is in-repo; optimizer can optionally come
     from `galore_torch` if installed (fallback to AdamW).

2) **Paper-faithful / external tooling (optional)**: install official upstream
   repos to reproduce their exact algorithms/kernels.
   This is *not required* to run benchmarks.

## Pipeline Architecture

```
┌─────────────────┐
│  Qwen2.5 Model  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   PiSSA Stage   │
│  - Extract      │
│    adapters     │
│  - Extract      │
│    residuals    │
└────────┬────────┘
         │
         ├──────────────┐
         ▼              ▼
┌──────────────┐  ┌──────────────┐
│   Adapters   │  │   Residual   │
│  (Full       │  │   Base Model │
│   Precision) │  └──────┬───────┘
└──────────────┘         │
                         ▼
                ┌─────────────────┐
                │  SpinQuant      │
                │  - Quantize     │
                │  - Freeze       │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Quantized      │
                │  Base Model     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  GaLore Stage   │
                │  - Train        │
                │    adapters     │
                │  - Low-rank     │
                │    gradients    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Final Model    │
                │  - Quantized    │
                │    base         │
                │  - Trained      │
                │    adapters     │
                └─────────────────┘
```

## Project Structure

```
.
├── pipeline.py                 # Main pipeline orchestration script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── configs/                    # Configuration files
│   ├── pipeline_config.yaml    # Main pipeline configuration
│   ├── pissa_config.yaml       # PiSSA configuration
│   ├── spinquant_config.yaml   # SpinQuant configuration
│   └── galore_config.yaml      # GaLore configuration
├── scripts/                    # Individual stage scripts
│   ├── pissa_extraction.py     # PiSSA adapter extraction
│   ├── spinquant.py            # SpinQuant quantization
│   └── galore_training.py      # GaLore adapter training
├── utils/                      # Utility functions
│   ├── helpers.py              # Helper functions
│   └── memory_tracker.py       # Memory tracking utility
├── outputs/                    # Output directories (auto-created)
│   ├── adapters/               # PiSSA adapters
│   ├── quantized_models/       # Quantized base models
│   └── checkpoints/            # Training checkpoints
└── logs/                       # Log files (auto-created)
    ├── pipeline.log            # Pipeline execution logs
    ├── memory_usage.log        # Memory usage logs
    └── memory_history.json    # Detailed memory history
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 40GB+ GPU memory recommended for 7B model

### Setup

1. Clone the repository:
```bash
# On your SSH server, clone under your home directory (example):
cd ~
git clone <YOUR_REPO_URL> DLProj
cd ~/DLProj
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install additional libraries (optional):

These are only needed if you want the *official* implementations.

**PiSSA (official):**
```bash
pip install git+https://github.com/GraphPKU/PiSSA.git
```

**GaLore optimizer package (optional acceleration):**
```bash
pip install galore-torch
```

**SpinQuant (official):**
```bash
pip install git+https://github.com/ModelTC/SpinQuant.git
```

## Usage

### Quick Start

Run the complete pipeline with default settings:

```bash
python pipeline.py --create_config
python pipeline.py
```

### Custom Configuration

1. Create or modify configuration file:
```bash
python pipeline.py --create_config
# Edit configs/pipeline_config.yaml
```

2. Run with custom configuration:
```bash
python pipeline.py --config configs/pipeline_config.yaml
```

### Run Individual Stages

#### Stage 1: PiSSA Adapter Extraction
```bash
python scripts/pissa_extraction.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --output_dir outputs/adapters \
    --rank 64 \
    --alpha 128
```

#### Stage 2: SpinQuant Quantization
```bash
python scripts/spinquant.py \
    --residual_model_path outputs/residual_model \
    --output_dir outputs/quantized_models \
    --bits 8 \
    --double_quant
```

#### Stage 3: GaLore Training
```bash
python scripts/galore_training.py \
    --quantized_model_path outputs/quantized_model \
    --adapter_path outputs/adapters \
    --output_dir outputs/checkpoints \
    --rank 128 \
    --learning_rate 1e-4 \
    --num_epochs 3
```

### Resume from Specific Stage

If the pipeline is interrupted, you can resume from a specific stage:

```bash
# Resume from SpinQuant stage
python pipeline.py --start_from spinquant

# Resume from GaLore stage
python pipeline.py --start_from galore
```

## Configuration

### Pipeline Configuration

Edit [`configs/pipeline_config.yaml`](configs/pipeline_config.yaml) to customize:

- **Model**: Qwen2.5 model variant
- **PiSSA**: Rank, alpha, dropout, target modules
- **SpinQuant**: Bits (4/8), quantization type
- **GaLore**: Rank, learning rate, training parameters
- **Training**: Epochs, batch size, dataset

### Key Parameters

#### PiSSA
- `rank`: LoRA rank (default: 64)
- `alpha`: LoRA alpha (default: 128)
- `dropout`: Dropout rate (default: 0.05)
- `target_modules`: Modules to apply LoRA

#### SpinQuant
- `bits`: Quantization bits (4 or 8)
- `double_quant`: Enable double quantization
- `quant_type`: "nf4" or "fp4"

#### GaLore
- `rank`: Gradient projection rank (default: 128)
- `learning_rate`: Learning rate (default: 1e-4)
- `update_proj_gap`: Projection update frequency

## Model Variants

Supported Qwen2.5 models:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-14B-Instruct`
- `Qwen/Qwen2.5-32B-Instruct`
- `Qwen/Qwen2.5-72B-Instruct`

## Memory Requirements

| Model Size | GPU Memory (Full) | GPU Memory (Quantized) |
|------------|-------------------|------------------------|
| 0.5B       | 4 GB              | 2 GB                   |
| 1.5B       | 8 GB              | 4 GB                   |
| 3B         | 16 GB             | 8 GB                   |
| 7B         | 28 GB             | 14 GB                  |
| 14B        | 56 GB             | 28 GB                  |
| 32B        | 128 GB            | 64 GB                  |
| 72B        | 288 GB            | 144 GB                 |

## Implementation Notes

### GaLore Training on Adapters Only

**Important**: The GaLore stage trains ONLY the PiSSA-extracted adapters, not the frozen quantized base model. This is explicitly enforced:

```python
# Freeze base model (quantized model) - only train adapters
for param in self.base_model.parameters():
    param.requires_grad = False
```

This ensures:
- The quantized base model remains frozen (no gradients)
- Only adapter parameters are updated during training
- Memory-efficient training with GaLore optimizer
- Preserves quantization benefits while fine-tuning adapters

### SpinQuant Rotation Matrix

**Important**: The current SpinQuant implementation uses a simplified bitsandbytes quantization approach as a placeholder.

**Actual SpinQuant** uses learned rotation matrices to minimize quantization error:

1. **Rotation Matrix Learning**: An orthogonal rotation matrix R is learned on calibration data
2. **Quantization**: Quantizes `W @ R` instead of `W` directly
3. **Inference**: Applies inverse rotation `R^T` during forward pass
4. **Objective**: Minimizes `||W - dequantize(quantize(W @ R))||_F`

**For production use**, integrate with the official SpinQuant library:

```bash
pip install git+https://github.com/ModelTC/SpinQuant.git
```

The pipeline includes warnings when using the placeholder implementation. Replace the quantization code in [`scripts/spinquant.py`](scripts/spinquant.py) with the official SpinQuant library for production use.

## Memory Tracking

The pipeline includes comprehensive memory tracking to monitor GPU and system memory usage throughout all stages:

### Features

- **Real-time Monitoring**: Tracks GPU and system memory at key points
- **Peak Memory Detection**: Records peak memory usage for each stage
- **Detailed Logging**: Saves memory usage to log files
- **Per-Stage Statistics**: Provides memory statistics for each pipeline stage
- **JSON Export**: Exports complete memory history to JSON

### Memory Logs

- `logs/memory_usage.log` - Main memory usage log
- `logs/pissa_memory.log` - PiSSA stage memory log
- `logs/spinquant_memory.log` - SpinQuant stage memory log
- `logs/galore_memory.log` - GaLore stage memory log
- `logs/memory_history.json` - Complete memory history with timestamps

### Using Memory Tracker

The memory tracker is automatically enabled when running the pipeline. To manually check memory:

```python
from utils.memory_tracker import print_memory_info

# Print current memory status
print_memory_info()
```

### Memory Report Format

```
================================================================================
MEMORY USAGE SUMMARY
================================================================================

GPU Memory:
  Peak:     24567.89 MB
  Average:  18234.56 MB
  Average %: 65.2%

System Memory:
  Peak:     32145.67 MB
  Average:  28901.23 MB
  Average %: 78.5%

================================================================================
```

## Output

The pipeline generates:

1. **Adapters**: `outputs/adapters/` - PiSSA-extracted adapters
2. **Residual Model**: `outputs/residual_model/` - Base model without adapters
3. **Quantized Model**: `outputs/quantized_model/` - Quantized base model
4. **Trained Adapters**: `outputs/trained_adapters/` - GaLore-trained adapters
5. **Logs**: `logs/pipeline.log` - Execution logs
6. **Memory Logs**: `logs/memory_usage.log`, `logs/memory_history.json` - Memory tracking logs
7. **Checkpoints**: `outputs/checkpoints/` - Training checkpoints

## Inference

Load the final quantized model with trained adapters:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load quantized base model
base_model = AutoModelForCausalLM.from_pretrained(
    "outputs/quantized_model",
    device_map="auto",
    torch_dtype=torch.float16
)

# Load trained adapters
model = PeftModel.from_pretrained(
    base_model,
    "outputs/trained_adapters"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("outputs/quantized_model")

# Generate
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs, max_length=512)
print(tokenizer.decode(outputs[0]))
```

## Troubleshooting

### Out of Memory

- Reduce `batch_size` in configuration
- Enable `gradient_checkpointing`
- Use smaller model variant
- Reduce PiSSA/GaLore rank
- Check memory logs to identify peak usage stages
- Consider using 4-bit quantization instead of 8-bit

### CUDA Errors

- Ensure CUDA version matches PyTorch
- Check GPU memory availability
- Verify driver compatibility

### Import Errors

- Install missing dependencies from requirements.txt
- Ensure PiSSA, GaLore, and SpinQuant are properly installed
- For SpinQuant: The current implementation uses bitsandbytes as placeholder. Install official SpinQuant library for production use

### Implementation Notes

- **GaLore**: Explicitly freezes quantized base model, trains only adapters
- **SpinQuant**: Current implementation is a placeholder using bitsandbytes. For production, integrate with official SpinQuant library that uses learned rotation matrices

## Performance Tips

1. **Use Mixed Precision**: Enable FP16 for faster training
2. **Gradient Accumulation**: Increase for larger effective batch sizes
3. **Gradient Checkpointing**: Trade compute for memory
4. **Optimal Ranks**: 
   - PiSSA rank: 32-128
   - GaLore rank: 64-256
5. **Quantization**: 4-bit for maximum compression, 8-bit for better quality

## Citation

If you use this pipeline, please cite the original papers:

```bibtex
@article{pissa2024,
  title={PiSSA: Principal Singular values and Singular Vectors Adaptation},
  author={...},
  year={2024}
}

@article{spinquant2024,
  title={SpinQuant: LLM Quantization with Learned Rotations},
  author={...},
  year={2024}
}

@article{galore2024,
  title={GaLore: Gradient Low-Rank Projection for Memory-Efficient LLM Training},
  author={...},
  year={2024}
}
```

## License

This project is provided as-is for research purposes. Please refer to the original Qwen2.5, PiSSA, SpinQuant, and GaLore licenses.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

For questions or issues, please open an issue on the repository.
