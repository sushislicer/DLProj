
"""
Aggregate benchmark results from JSON files into a summary CSV and plots.
"""

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

def main():
    parser = argparse.ArgumentParser(description="Aggregate benchmark results")
    parser.add_argument('--results_dir', type=str, default='benchmark_results', help='Directory containing benchmark results')
    parser.add_argument('--output_file', type=str, default='benchmark_summary.csv', help='Output CSV file')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    args = parser.parse_args()

    results = []
    
    # Find all final_results.json files
    pattern = os.path.join(args.results_dir, '**', 'final_results.json')
    files = glob(pattern, recursive=True)
    
    print(f"Found {len(files)} result files.")
    
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            timestamp = data.get('system_info', {}).get('timestamp', 'unknown')
            
            benchmarks = data.get('benchmarks', {})
            for model_key, model_data in benchmarks.items():
                if 'error' in model_data:
                    continue
                    
                model_name = model_data.get('model_name', 'unknown')
                model_size = model_data.get('model_size', 'unknown')
                variant = model_data.get('variant', 'main')
                
                model_benchmarks = model_data.get('benchmarks', {})
                for bench_name, bench_data in model_benchmarks.items():
                    if 'error' in bench_data:
                        continue
                        
                    metrics = bench_data.get('metrics', {})
                    
                    # Extract key metric
                    score = metrics.get('accuracy') or metrics.get('pass_at_1') or metrics.get('pass_rate') or 0.0
                    
                    results.append({
                        'Timestamp': timestamp,
                        'Model': model_name,
                        'Size': model_size,
                        'Variant': variant,
                        'Benchmark': bench_name,
                        'Score': score,
                        'Throughput': metrics.get('throughput', 0.0)
                    })
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not results:
        print("No results found.")
        return

    df = pd.DataFrame(results)
    df.to_csv(args.output_file, index=False)
    print(f"Summary saved to {args.output_file}")
    
    if args.plot:
        try:
            plt.figure(figsize=(12, 6))
            sns.barplot(data=df, x='Benchmark', y='Score', hue='Variant')
            plt.title('Benchmark Scores by Model Variant')
            plt.tight_layout()
            plt.savefig('benchmark_scores.png')
            print("Plot saved to benchmark_scores.png")
        except Exception as e:
            print(f"Error plotting: {e}")

if __name__ == "__main__":
    main()
