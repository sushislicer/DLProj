"""Download benchmark datasets into a local directory.

This is used to ensure paper runs evaluate on real datasets (not synthetic
fallback samples).

Examples:
  python3 src/data/downloader.py
  python3 src/data/downloader.py --datasets aime math gpqa
  python3 src/data/downloader.py --output_dir datasets
  python3 src/data/downloader.py --hf_token "$HF_TOKEN"
"""

import argparse
import os
import sys
import logging

# Add project root to path (so `import benchmarks...` works when running as a script).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.aime_benchmark import download_aime_dataset
from benchmarks.math_benchmark import download_math_dataset
from benchmarks.gpqa_benchmark import download_gpqa_dataset
from benchmarks.livecodebench_benchmark import download_livecodebench_dataset
from benchmarks.swe_bench_benchmark import download_swe_bench_dataset

logging.basicConfig(level=logging.INFO)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download benchmark datasets into a local directory")
    ap.add_argument(
        '--datasets',
        nargs='*',
        default=['aime', 'math', 'gpqa', 'livecodebench'],
        help='Datasets to download: aime math gpqa livecodebench swe_bench (default: all except swe_bench)',
    )
    ap.add_argument(
        '--output_dir',
        type=str,
        default='datasets',
        help='Base output directory (default: datasets)',
    )
    ap.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='Optional HuggingFace token (sets HF_TOKEN + HUGGINGFACE_HUB_TOKEN for this process)',
    )
    ap.add_argument(
        '--hf_endpoint',
        type=str,
        default=None,
        help=(
            'Optional HuggingFace Hub endpoint (sets HF_ENDPOINT). '
            'Useful in restricted networks / mirrors (e.g., https://hf-mirror.com).'
        ),
    )
    args = ap.parse_args()

    if args.hf_endpoint:
        os.environ.setdefault('HF_ENDPOINT', str(args.hf_endpoint))
        # Also set HF_HUB_ENDPOINT which some older libraries/versions might use
        os.environ.setdefault('HF_HUB_ENDPOINT', str(args.hf_endpoint))

    if args.hf_token:
        os.environ.setdefault('HF_TOKEN', str(args.hf_token))
        os.environ.setdefault('HUGGINGFACE_HUB_TOKEN', str(args.hf_token))

    want = [str(x).strip().lower() for x in (args.datasets or [])]
    out_base = str(args.output_dir)
    os.makedirs(out_base, exist_ok=True)

    def _run(name: str, fn):
        out_path = os.path.join(out_base, name)
        print(f"\n== Downloading {name} -> {out_path} ==")
        data = fn(out_path)
        n = len(data) if hasattr(data, '__len__') else 0
        print(f"{name}: {n} samples")

    # Keep order stable.
    if 'aime' in want:
        _run('aime', download_aime_dataset)
    if 'math' in want:
        _run('math', download_math_dataset)
    if 'gpqa' in want:
        _run('gpqa', download_gpqa_dataset)
    if 'livecodebench' in want:
        _run('livecodebench', download_livecodebench_dataset)
    if 'swe_bench' in want or 'swe-bench' in want:
        _run('swe_bench', download_swe_bench_dataset)


if __name__ == "__main__":
    main()
