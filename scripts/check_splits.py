
import os
from datasets import get_dataset_split_names

datasets_to_check = [
    ("qwedsacf/competition_math", "BENCH_MATH_HF_SPLIT"),
    ("Maxwell-Jia/AIME_2024", "BENCH_AIME_HF_SPLIT"),
    ("idavidrein/gpqa-extended", "BENCH_GPQA_HF_SPLIT"),
    ("livecodebench/code_generation_lite", "BENCH_LIVECODEBENCH_HF_SPLIT")
]

print("Checking dataset splits...")
for dataset_id, env_var in datasets_to_check:
    try:
        splits = get_dataset_split_names(dataset_id)
        print(f"Dataset: {dataset_id}")
        print(f"  Available splits: {splits}")
    except Exception as e:
        print(f"Error checking {dataset_id}: {e}")
