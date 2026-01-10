
"""
Lightweight evaluation script for Qwen2.5 0.5B and 7B models.
Runs benchmarks with reduced sample sizes for quick assessment.
"""

import os
import sys
import argparse
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_benchmarks import BenchmarkRunner
from utils.helpers import setup_logging

def main():
    parser = argparse.ArgumentParser(description="Lightweight evaluation for 0.5B and 7B models")
    parser.add_argument('--num_samples', type=int, default=20, help="Number of samples per benchmark")
    parser.add_argument('--output_dir', type=str, default="benchmark_results/lightweight", help="Output directory")
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(log_dir='logs/benchmark', log_file='lightweight_eval.log', logger_name='lightweight')
    
    config_path = 'configs/benchmark_config.yaml'
    
    # Initialize runner
    runner = BenchmarkRunner(config_path, logger)
    
    # Override config for lightweight run
    runner.config['execution']['max_execution_time'] = 1.0  # 1 hour budget
    runner.output_dir = args.output_dir
    if not os.path.exists(runner.output_dir):
        os.makedirs(runner.output_dir)
        
    # Filter models: keep only 0.5B and 7B
    runner.config['models'] = [
        m for m in runner.config['models'] 
        if m['size'] in ['0.5B', '7B']
    ]
    
    # Enable baselines
    runner.run_baselines = True
    
    # Override sample counts
    runner.override_num_samples = args.num_samples
    
    # Ensure quantization is enabled (as per user request for 4-bit baseline)
    runner.config['gpu']['quantization']['enabled'] = True

    # Disable FlashAttention auto-install to prevent hanging on compilation
    runner.config.setdefault('advanced', {})
    runner.config['advanced']['auto_install_flash_attention'] = False
    runner.config['advanced']['use_flash_attention'] = False # Fallback to SDPA/eager
    
    logger.info(f"Starting lightweight evaluation for 0.5B and 7B models with {args.num_samples} samples...")
    
    # Run benchmarks
    runner.run_all_models()
    
    logger.info("Lightweight evaluation completed.")

if __name__ == "__main__":
    main()
