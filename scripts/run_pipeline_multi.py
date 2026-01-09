"""Run the PiSSA->SpinQuant->GaLore pipeline across multiple Qwen2.5 model sizes.

This is a convenience wrapper so you can produce per-size pipeline adapters
(`trained_adapters`) for the benchmark triad:
  - pipeline_4bit (ours)
  - baseline_4bit (native 4-bit)
  - baseline_4bit_lora (4-bit + LoRA baseline)

By default it reads model names from `configs/benchmark_config.yaml`.

Outputs:
  outputs/pipeline/<SIZE>/trained_adapters
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Dict, List

import yaml


def _load_benchmark_models(path: str) -> List[Dict]:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return list(cfg.get('models', []))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run pipeline for multiple model sizes")
    ap.add_argument('--pipeline_config', type=str, default='configs/pipeline_config.yaml')
    ap.add_argument('--benchmark_config', type=str, default='configs/benchmark_config.yaml')
    ap.add_argument('--sizes', type=str, nargs='*', default=None, help='Optional subset of sizes (e.g., 0.5B 7B 14B 72B)')
    ap.add_argument('--output_root', type=str, default='outputs/pipeline', help='Root output directory for per-size outputs')
    args = ap.parse_args()

    models = _load_benchmark_models(args.benchmark_config)
    if args.sizes:
        want = set(args.sizes)
        models = [m for m in models if str(m.get('size')) in want]

    if not models:
        raise SystemExit('No models found to run')

    for m in models:
        name = str(m['name'])
        size = str(m['size'])
        out_dir = os.path.join(args.output_root, size)
        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            sys.executable,
            'pipeline.py',
            '--config', args.pipeline_config,
            '--model_name', name,
            '--output_dir', out_dir,
        ]
        print(f"\n[run_pipeline_multi] Running pipeline for {size}: {name}")
        print(' '.join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()

