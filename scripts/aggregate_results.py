"""
Results aggregation and reporting script for benchmark results.
Generates comprehensive reports and visualizations.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import setup_logging, ensure_dir


class ResultsAggregator:
    """Aggregate and analyze benchmark results."""
    
    def __init__(self, results_dir: str, logger: logging.Logger):
        """
        Initialize results aggregator.
        
        Args:
            results_dir: Directory containing benchmark results
            logger: Logger instance
        """
        self.results_dir = results_dir
        self.logger = logger
        self.results = self._load_all_results()
    
    def _load_all_results(self) -> Dict:
        """Load all benchmark results from directory."""
        results = {}
        
        if not os.path.exists(self.results_dir):
            self.logger.warning(f"Results directory not found: {self.results_dir}")
            return results
        
        # Find all JSON files
        json_files = list(Path(self.results_dir).glob("**/*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Use filename as key
                key = json_file.stem
                results[key] = {
                    'data': data,
                    'path': str(json_file)
                }
                
                self.logger.info(f"Loaded results from: {json_file}")
            
            except Exception as e:
                self.logger.error(f"Error loading {json_file}: {e}")
        
        return results
    
    def generate_summary_report(self, output_path: str = None) -> str:
        """
        Generate summary report of all benchmark results.
        
        Args:
            output_path: Path to save report (optional)
        
        Returns:
            Summary report as string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("BENCHMARK RESULTS SUMMARY")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Results directory: {self.results_dir}")
        report_lines.append("=" * 80)
        
        # Aggregate metrics by model and benchmark
        model_benchmark_metrics = defaultdict(lambda: defaultdict(list))
        
        for result_key, result_data in self.results.items():
            data = result_data['data']
            
            # Extract model size and benchmark info
            if 'benchmarks' in data:
                for model_size, model_data in data['benchmarks'].items():
                    if 'benchmarks' in model_data:
                        for benchmark_name, benchmark_data in model_data['benchmarks'].items():
                            if 'metrics' in benchmark_data:
                                for metric_name, metric_value in benchmark_data['metrics'].items():
                                    if isinstance(metric_value, (int, float)):
                                        model_benchmark_metrics[model_size][benchmark_name].append({
                                            'metric': metric_name,
                                            'value': metric_value
                                        })
        
        # Generate summary table
        report_lines.append("\n" + "=" * 80)
        report_lines.append("PERFORMANCE SUMMARY")
        report_lines.append("=" * 80)
        
        # Create DataFrame for summary
        summary_data = []
        for model_size in sorted(model_benchmark_metrics.keys()):
            for benchmark_name in sorted(model_benchmark_metrics[model_size].keys()):
                metrics = model_benchmark_metrics[model_size][benchmark_name]
                
                # Extract accuracy if available
                accuracy = None
                for m in metrics:
                    if m['metric'] == 'accuracy':
                        accuracy = m['value']
                        break
                
                summary_data.append({
                    'Model Size': model_size,
                    'Benchmark': benchmark_name,
                    'Accuracy': f"{accuracy:.2%}" if accuracy is not None else "N/A"
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            report_lines.append("\n" + df.to_string(index=False))
        else:
            report_lines.append("\nNo benchmark results found.")
        
        # Generate detailed metrics table
        report_lines.append("\n" + "=" * 80)
        report_lines.append("DETAILED METRICS")
        report_lines.append("=" * 80)
        
        for model_size in sorted(model_benchmark_metrics.keys()):
            report_lines.append(f"\n{model_size}:")
            report_lines.append("-" * 80)
            
            for benchmark_name in sorted(model_benchmark_metrics[model_size].keys()):
                report_lines.append(f"\n  {benchmark_name}:")
                metrics = model_benchmark_metrics[model_size][benchmark_name]
                
                for m in metrics:
                    metric_name = m['metric']
                    metric_value = m['value']
                    
                    # Format value based on type
                    if 'accuracy' in metric_name.lower() or 'pass' in metric_name.lower():
                        formatted_value = f"{metric_value:.2%}"
                    elif 'latency' in metric_name.lower() or 'time' in metric_name.lower():
                        formatted_value = f"{metric_value:.2f}s"
                    elif 'throughput' in metric_name.lower():
                        formatted_value = f"{metric_value:.2f} samples/s"
                    else:
                        formatted_value = str(metric_value)
                    
                    report_lines.append(f"    {metric_name}: {formatted_value}")
        
        # Generate comparison table
        report_lines.append("\n" + "=" * 80)
        report_lines.append("MODEL COMPARISON")
        report_lines.append("=" * 80)
        
        comparison_data = []
        for benchmark_name in set().union(*[model_benchmark_metrics[m].keys() for m in model_benchmark_metrics]):
            row = {'Benchmark': benchmark_name}
            
            for model_size in sorted(model_benchmark_metrics.keys()):
                if benchmark_name in model_benchmark_metrics[model_size]:
                    metrics = model_benchmark_metrics[model_size][benchmark_name]
                    accuracy = None
                    for m in metrics:
                        if m['metric'] == 'accuracy':
                            accuracy = m['value']
                            break
                    row[model_size] = f"{accuracy:.2%}" if accuracy is not None else "N/A"
                else:
                    row[model_size] = "N/A"
            
            comparison_data.append(row)
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            report_lines.append("\n" + df_comparison.to_string(index=False))
        
        # Generate system info
        report_lines.append("\n" + "=" * 80)
        report_lines.append("SYSTEM INFORMATION")
        report_lines.append("=" * 80)
        
        for result_key, result_data in self.results.items():
            data = result_data['data']
            if 'system_info' in data:
                sys_info = data['system_info']
                report_lines.append(f"\n{result_key}:")
                report_lines.append(f"  Python: {sys_info.get('python_version', 'N/A')}")
                report_lines.append(f"  PyTorch: {sys_info.get('torch_version', 'N/A')}")
                report_lines.append(f"  CUDA: {sys_info.get('cuda_version', 'N/A')}")
                report_lines.append(f"  GPUs: {sys_info.get('num_gpus', 0)}")
                
                for gpu_info in sys_info.get('gpu_info', []):
                    report_lines.append(f"    GPU {gpu_info['device_id']}: {gpu_info['name']} ({gpu_info['total_memory']})")
        
        report_lines.append("\n" + "=" * 80)
        
        report = "\n".join(report_lines)
        
        # Save report if output path provided
        if output_path:
            ensure_dir(os.path.dirname(output_path))
            with open(output_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Summary report saved to: {output_path}")
        
        return report
    
    def generate_csv_report(self, output_path: str):
        """
        Generate CSV report of benchmark results.
        
        Args:
            output_path: Path to save CSV report
        """
        # Aggregate data
        csv_data = []
        
        for result_key, result_data in self.results.items():
            data = result_data['data']
            
            if 'benchmarks' in data:
                for model_size, model_data in data['benchmarks'].items():
                    if 'benchmarks' in model_data:
                        for benchmark_name, benchmark_data in model_data['benchmarks'].items():
                            if 'metrics' in benchmark_data:
                                row = {
                                    'Result Key': result_key,
                                    'Model Size': model_size,
                                    'Benchmark': benchmark_name
                                }
                                
                                for metric_name, metric_value in benchmark_data['metrics'].items():
                                    row[metric_name] = metric_value
                                
                                csv_data.append(row)
        
        # Create DataFrame and save
        if csv_data:
            df = pd.DataFrame(csv_data)
            ensure_dir(os.path.dirname(output_path))
            df.to_csv(output_path, index=False)
            self.logger.info(f"CSV report saved to: {output_path}")
        else:
            self.logger.warning("No data to write to CSV report")
    
    def generate_latex_report(self, output_path: str):
        """
        Generate LaTeX report for academic papers.
        
        Args:
            output_path: Path to save LaTeX report
        """
        latex_lines = []
        
        latex_lines.append("\\documentclass{article}")
        latex_lines.append("\\usepackage{booktabs}")
        latex_lines.append("\\usepackage{longtable}")
        latex_lines.append("\\begin{document}")
        
        latex_lines.append("\\section{Benchmark Results}")
        latex_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d')}")
        
        # Aggregate metrics
        model_benchmark_metrics = defaultdict(lambda: defaultdict(dict))
        
        for result_key, result_data in self.results.items():
            data = result_data['data']
            
            if 'benchmarks' in data:
                for model_size, model_data in data['benchmarks'].items():
                    if 'benchmarks' in model_data:
                        for benchmark_name, benchmark_data in model_data['benchmarks'].items():
                            if 'metrics' in benchmark_data:
                                for metric_name, metric_value in benchmark_data['metrics'].items():
                                    model_benchmark_metrics[model_size][benchmark_name][metric_name] = metric_value
        
        # Generate tables for each benchmark
        for benchmark_name in set().union(*[model_benchmark_metrics[m].keys() for m in model_benchmark_metrics]):
            latex_lines.append(f"\\subsection{{{benchmark_name}}}")
            latex_lines.append("\\begin{table}[h]")
            latex_lines.append("\\centering")
            latex_lines.append("\\begin{tabular}{lcc}")
            latex_lines.append("\\toprule")
            latex_lines.append("Model Size & Accuracy & Avg Latency (s) \\\\")
            latex_lines.append("\\midrule")
            
            for model_size in sorted(model_benchmark_metrics.keys()):
                if benchmark_name in model_benchmark_metrics[model_size]:
                    metrics = model_benchmark_metrics[model_size][benchmark_name]
                    accuracy = metrics.get('accuracy', 0)
                    latency = metrics.get('avg_latency', 0)
                    latex_lines.append(f"{model_size} & {accuracy:.2%} & {latency:.2f} \\\\")
            
            latex_lines.append("\\bottomrule")
            latex_lines.append("\\end{tabular}")
            latex_lines.append("\\caption{Performance on " + benchmark_name + "}")
            latex_lines.append("\\end{table}")
        
        latex_lines.append("\\end{document}")
        
        # Save LaTeX report
        ensure_dir(os.path.dirname(output_path))
        with open(output_path, 'w') as f:
            f.write('\n'.join(latex_lines))
        
        self.logger.info(f"LaTeX report saved to: {output_path}")
    
    def plot_results(self, output_path: str):
        """
        Generate plots of benchmark results.
        
        Args:
            output_path: Path to save plots
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Aggregate data
            plot_data = []
            
            for result_key, result_data in self.results.items():
                data = result_data['data']
                
                if 'benchmarks' in data:
                    for model_size, model_data in data['benchmarks'].items():
                        if 'benchmarks' in model_data:
                            for benchmark_name, benchmark_data in model_data['benchmarks'].items():
                                if 'metrics' in benchmark_data:
                                    accuracy = benchmark_data['metrics'].get('accuracy', 0)
                                    plot_data.append({
                                        'Model Size': model_size,
                                        'Benchmark': benchmark_name,
                                        'Accuracy': accuracy
                                    })
            
            if not plot_data:
                self.logger.warning("No data to plot")
                return
            
            df = pd.DataFrame(plot_data)
            
            # Create output directory
            ensure_dir(output_path)
            
            # Plot 1: Bar chart of accuracy by model and benchmark
            plt.figure(figsize=(12, 6))
            sns.barplot(data=df, x='Benchmark', y='Accuracy', hue='Model Size')
            plt.title('Benchmark Accuracy by Model Size')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'accuracy_by_model.png'))
            plt.close()
            
            # Plot 2: Heatmap
            pivot_df = df.pivot(index='Model Size', columns='Benchmark', values='Accuracy')
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot_df, annot=True, fmt='.2%', cmap='YlOrRd')
            plt.title('Benchmark Accuracy Heatmap')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'accuracy_heatmap.png'))
            plt.close()
            
            self.logger.info(f"Plots saved to: {output_path}")
        
        except ImportError:
            self.logger.warning("Matplotlib or seaborn not available, skipping plots")
        except Exception as e:
            self.logger.error(f"Error generating plots: {e}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Aggregate and report benchmark results"
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='benchmark_results',
        help='Directory containing benchmark results'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='benchmark_reports',
        help='Directory to save reports'
    )
    parser.add_argument(
        '--generate_summary',
        action='store_true',
        help='Generate summary report'
    )
    parser.add_argument(
        '--generate_csv',
        action='store_true',
        help='Generate CSV report'
    )
    parser.add_argument(
        '--generate_latex',
        action='store_true',
        help='Generate LaTeX report'
    )
    parser.add_argument(
        '--generate_plots',
        action='store_true',
        help='Generate plots'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate all reports and plots'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(log_dir='logs/reports', log_file='reports.log', logger_name='reports')
    
    # Create aggregator
    aggregator = ResultsAggregator(args.results_dir, logger)
    
    # Generate reports
    if args.all or args.generate_summary:
        summary_path = os.path.join(args.output_dir, 'summary_report.txt')
        aggregator.generate_summary_report(summary_path)
    
    if args.all or args.generate_csv:
        csv_path = os.path.join(args.output_dir, 'results.csv')
        aggregator.generate_csv_report(csv_path)
    
    if args.all or args.generate_latex:
        latex_path = os.path.join(args.output_dir, 'report.tex')
        aggregator.generate_latex_report(latex_path)
    
    if args.all or args.generate_plots:
        plots_path = os.path.join(args.output_dir, 'plots')
        aggregator.plot_results(plots_path)
    
    logger.info("Results aggregation completed")


if __name__ == '__main__':
    main()
