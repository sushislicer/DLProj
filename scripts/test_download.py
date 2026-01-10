
import logging
from benchmarks.aime_benchmark import download_aime_dataset
from benchmarks.math_benchmark import download_math_dataset
from benchmarks.gpqa_benchmark import download_gpqa_dataset
from benchmarks.livecodebench_benchmark import download_livecodebench_dataset

logging.basicConfig(level=logging.INFO)

def test_download():
    print("Testing AIME download...")
    data = download_aime_dataset("datasets/aime")
    print(f"AIME: {len(data)} samples")

    print("\nTesting MATH download...")
    data = download_math_dataset("datasets/math")
    print(f"MATH: {len(data)} samples")

    print("\nTesting GPQA download...")
    data = download_gpqa_dataset("datasets/gpqa")
    print(f"GPQA: {len(data)} samples")

    print("\nTesting LiveCodeBench download...")
    data = download_livecodebench_dataset("datasets/livecodebench")
    print(f"LiveCodeBench: {len(data)} samples")

if __name__ == "__main__":
    test_download()
