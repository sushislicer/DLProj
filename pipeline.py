"""
Main pipeline orchestration script for Qwen2.5 quantization pipeline.
Orchestrates PiSSA -> SpinQuant -> GaLore workflow.
"""

import os
import sys
import argparse
import subprocess
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.helpers import (
    setup_logging, load_config, save_config,
    ensure_dir, format_time, ProgressTracker
)
from utils.memory_tracker import MemoryTracker


class QuantizationPipeline:
    """Orchestrate the complete quantization pipeline."""
    
    def __init__(self, config: dict, logger):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.stages = ['pissa', 'spinquant', 'galore']
        self.progress_tracker = ProgressTracker(
            stages=self.stages,
            log_dir=config.get('log_dir', 'logs')
        )
        
        # Initialize memory tracker
        self.memory_tracker = MemoryTracker(
            log_dir=config.get('log_dir', 'logs'),
            log_file='memory_usage.log'
        )
        
        # Ensure output directories exist
        ensure_dir(config['output_dir'])
        ensure_dir(config.get('log_dir', 'logs'))
    
    def run_pissa_extraction(self):
        """
        Stage 1: Extract adapters using PiSSA.
        
        Returns:
            Dictionary with paths to adapters and residual model
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 1: PiSSA Adapter Extraction")
        self.logger.info("=" * 60)
        
        self.memory_tracker.log_memory("PiSSA", "Stage started")
        start_time = time.time()
        
        # Build command
        cmd = [
            sys.executable, 'scripts/pissa_extraction.py',
            '--model_name', self.config['model_name'],
            '--output_dir', self.config['output_dir'],
            '--rank', str(self.config['pissa']['rank']),
            '--alpha', str(self.config['pissa']['alpha'])
        ]
        
        self.logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run command
        result = subprocess.run(cmd, check=True)
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"PiSSA extraction completed in {format_time(elapsed_time)}")
        self.memory_tracker.log_memory("PiSSA", f"Stage completed in {format_time(elapsed_time)}")
        
        # Return paths
        return {
            'adapter_path': os.path.join(self.config['output_dir'], 'adapters'),
            'residual_model_path': os.path.join(self.config['output_dir'], 'residual_model')
        }
    
    def run_spinquant(self, residual_model_path: str):
        """
        Stage 2: Quantize residual model using SpinQuant.
        
        Args:
            residual_model_path: Path to residual model from PiSSA
        
        Returns:
            Path to quantized model
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 2: SpinQuant Quantization")
        self.logger.info("=" * 60)
        
        self.memory_tracker.log_memory("SpinQuant", "Stage started")
        start_time = time.time()
        
        # Build command
        cmd = [
            sys.executable, 'scripts/spinquant.py',
            '--residual_model_path', residual_model_path,
            '--output_dir', self.config['output_dir'],
            '--bits', str(self.config['spinquant']['bits'])
        ]

        # SpinQuant-lite backend selection.
        # IMPORTANT: `scripts/spinquant.py` defaults to blockwise_givens, which can
        # be extremely slow. If you want the fast fixed rotation path, you must
        # pass `--backend hadamard` explicitly.
        backend = str(self.config.get('spinquant', {}).get('backend', 'blockwise_givens'))
        cmd.extend(['--backend', backend])

        # Pass through key SpinQuant-lite knobs so pipeline_config.yaml is honored.
        if 'block_size' in self.config.get('spinquant', {}):
            cmd.extend(['--block_size', str(self.config['spinquant']['block_size'])])
        if 'num_steps' in self.config.get('spinquant', {}):
            cmd.extend(['--num_steps', str(self.config['spinquant']['num_steps'])])
        if 'lr' in self.config.get('spinquant', {}):
            cmd.extend(['--lr', str(self.config['spinquant']['lr'])])
        if 'num_sweeps' in self.config.get('spinquant', {}):
            cmd.extend(['--num_sweeps', str(self.config['spinquant']['num_sweeps'])])
        if 'max_layers' in self.config.get('spinquant', {}):
            cmd.extend(['--max_layers', str(self.config['spinquant']['max_layers'])])

        # Optional fast path: skip rotation learning.
        if bool(self.config['spinquant'].get('skip_rotations', False)):
            cmd.append('--skip_rotations')
        
        if self.config['spinquant'].get('double_quant', False):
            cmd.append('--double_quant')
        
        cmd.extend(['--quant_type', self.config['spinquant'].get('quant_type', 'nf4')])

        # SpinQuant-lite knobs
        if self.config['spinquant'].get('use_bnb_quantization', False):
            cmd.append('--use_bnb_quantization')
        if not self.config['spinquant'].get('use_activation_objective', True):
            cmd.append('--no_activation_objective')
        if 'calibration_vectors_per_layer' in self.config['spinquant']:
            cmd.extend(['--calibration_vectors_per_layer', str(self.config['spinquant']['calibration_vectors_per_layer'])])
        if 'keep_fp16_modules' in self.config['spinquant']:
            keep = ','.join(self.config['spinquant']['keep_fp16_modules'])
            cmd.extend(['--keep_fp16_modules', keep])
        
        self.logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run command
        result = subprocess.run(cmd, check=True)
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"SpinQuant quantization completed in {format_time(elapsed_time)}")
        self.memory_tracker.log_memory("SpinQuant", f"Stage completed in {format_time(elapsed_time)}")
        
        # Return path
        return {
            'quantized_model_path': os.path.join(self.config['output_dir'], 'quantized_model')
        }
    
    def run_galore_training(self, quantized_model_path: str, adapter_path: str):
        """
        Stage 3: Train adapters using GaLore.
        
        Args:
            quantized_model_path: Path to quantized base model
            adapter_path: Path to PiSSA adapters
        
        Returns:
            Path to trained adapters
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 3: GaLore Adapter Training")
        self.logger.info("=" * 60)
        
        self.memory_tracker.log_memory("GaLore", "Stage started")
        start_time = time.time()
        
        # Build command
        cmd = [
            sys.executable, 'scripts/galore_training.py',
            '--quantized_model_path', quantized_model_path,
            '--adapter_path', adapter_path,
            '--output_dir', self.config['output_dir'],
            '--rank', str(self.config['galore']['rank']),
            '--learning_rate', str(self.config['galore']['learning_rate']),
            '--num_epochs', str(self.config.get('num_epochs', 3)),
            '--batch_size', str(self.config.get('batch_size', 4))
        ]

        # Speed/compute caps
        if 'max_samples' in self.config:
            cmd.extend(['--max_samples', str(self.config.get('max_samples'))])
        if 'max_length' in self.config:
            cmd.extend(['--max_length', str(self.config.get('max_length'))])
        if 'gradient_accumulation_steps' in self.config:
            cmd.extend(['--gradient_accumulation_steps', str(self.config.get('gradient_accumulation_steps'))])
        if 'max_steps' in self.config:
            cmd.extend(['--max_steps', str(self.config.get('max_steps'))])
        if self.config.get('gradient_checkpointing', None) is False:
            cmd.append('--no_gradient_checkpointing')

        # Optional: post-quant distillation.
        distill_cfg = self.config.get('distill', {})
        if distill_cfg.get('enabled', False):
            # Teacher: fp16 residual model produced by PiSSA.
            teacher_path = os.path.join(self.config['output_dir'], 'residual_model')
            cmd.extend(['--teacher_model_path', teacher_path])
            cmd.extend(['--distill_alpha', str(distill_cfg.get('alpha', 0.5))])
            cmd.extend(['--distill_temperature', str(distill_cfg.get('temperature', 2.0))])

        # Projection-space schedule (GaLore-like)
        proj_cfg = self.config.get('projection', {})
        if proj_cfg.get('enabled', False):
            cmd.append('--enable_proj')
            cmd.extend(['--proj_rank', str(proj_cfg.get('rank', 64))])
            cmd.extend(['--proj_update_gap', str(proj_cfg.get('update_gap', 200))])
            cmd.extend(['--proj_drift_threshold', str(proj_cfg.get('drift_threshold', 0.35))])
        
        if 'dataset' in self.config:
            cmd.extend(['--dataset', self.config['dataset']])
        
        self.logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run command
        result = subprocess.run(cmd, check=True)
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"GaLore training completed in {format_time(elapsed_time)}")
        self.memory_tracker.log_memory("GaLore", f"Stage completed in {format_time(elapsed_time)}")
        
        # Return path
        return {
            'trained_adapter_path': os.path.join(self.config['output_dir'], 'trained_adapters')
        }
    
    def run(self, start_from: str = None):
        """
        Run the complete pipeline.
        
        Args:
            start_from: Stage to start from ('pissa', 'spinquant', 'galore')
        
        Returns:
            Dictionary with all output paths
        """
        total_start_time = time.time()
        
        # Start memory tracking
        self.memory_tracker.start_tracking()
        
        self.logger.info("=" * 60)
        self.logger.info("QWEN2.5 QUANTIZATION PIPELINE")
        self.logger.info("=" * 60)
        self.logger.info(f"Model: {self.config['model_name']}")
        self.logger.info(f"Output directory: {self.config['output_dir']}")
        self.logger.info("=" * 60)
        
        results = {}
        
        # Determine starting stage
        if start_from:
            if start_from in self.stages:
                self.progress_tracker.current_stage = self.stages.index(start_from)
                self.logger.info(f"Resuming from stage: {start_from}")
            else:
                self.logger.warning(f"Invalid start stage: {start_from}, starting from beginning")

        # If we are resuming mid-pipeline, infer expected artifact paths.
        # This avoids KeyError when starting from later stages.
        output_dir = self.config['output_dir']
        inferred = {
            'adapter_path': os.path.join(output_dir, 'adapters'),
            'residual_model_path': os.path.join(output_dir, 'residual_model'),
            'quantized_model_path': os.path.join(output_dir, 'quantized_model'),
            'trained_adapter_path': os.path.join(output_dir, 'trained_adapters'),
        }

        current_stage = self.progress_tracker.get_current_stage()
        if current_stage in ('spinquant', 'galore'):
            # PiSSA outputs must exist.
            results.setdefault('adapter_path', inferred['adapter_path'])
            results.setdefault('residual_model_path', inferred['residual_model_path'])
        if current_stage == 'galore':
            # SpinQuant output must exist.
            results.setdefault('quantized_model_path', inferred['quantized_model_path'])
        
        # Stage 1: PiSSA
        if self.progress_tracker.get_current_stage() == 'pissa':
            results.update(self.run_pissa_extraction())
            self.progress_tracker.advance_stage()
        
        # Stage 2: SpinQuant
        if self.progress_tracker.get_current_stage() == 'spinquant':
            results.update(self.run_spinquant(results['residual_model_path']))
            self.progress_tracker.advance_stage()
        
        # Stage 3: GaLore
        if self.progress_tracker.get_current_stage() == 'galore':
            results.update(self.run_galore_training(
                results['quantized_model_path'],
                results['adapter_path']
            ))
            self.progress_tracker.advance_stage()
        
        total_elapsed_time = time.time() - total_start_time
        
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total time: {format_time(total_elapsed_time)}")
        self.logger.info("\nOutput paths:")
        for key, path in results.items():
            self.logger.info(f"  {key}: {path}")
        self.logger.info("=" * 60)
        
        # Stop memory tracking and print summary
        self.memory_tracker.stop_tracking()
        self.memory_tracker.print_summary()
        
        return results


def create_default_config():
    """Create default configuration."""
    return {
        'model_name': 'Qwen/Qwen2.5-7B-Instruct',
        'output_dir': 'outputs',
        'log_dir': 'logs',
        'seed': 42,
        # Keep default training time bounded.
        'num_epochs': 1,
        'batch_size': 4,
        'gradient_accumulation_steps': 2,
        'max_steps': 800,
        # NOTE: This dataset is for *adapter training* in the pipeline.
        # Benchmark datasets (AIME/MATH/GPQA/...) are configured separately.
        'dataset': 'c4',
        'max_samples': 2000,
        'max_length': 256,
        'pissa': {
            'rank': 64,
            'alpha': 128,
            'dropout': 0.05,
            'target_modules': [
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj'
            ],
            'bias': 'none'
        },
        'spinquant': {
            'bits': 8,
            'double_quant': True,
            'quant_type': 'nf4',
            'group_size': 128,
            'symmetric': False,
            # SpinQuant-lite backend (blockwise Givens rotations)
            'backend': 'blockwise_givens',
            'block_size': 64,
            'num_steps': 20,
            'lr': 0.05,
            'num_sweeps': 2,
            'max_layers': 8,
            # Rotation learning is expensive; default to skipping for bounded runtime.
            'skip_rotations': True,
            'use_bnb_quantization': True,
            'use_activation_objective': True,
            'calibration_vectors_per_layer': 256,
        },
        'galore': {
            'rank': 128,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'update_proj_gap': 200,
            'scale': 0.25,
            'proj_type': 'std'
        },
        # Optional: post-quant adapter distillation (teacher = fp16 residual model)
        # Disabled by default to keep the pipeline primarily a compression/fine-tune
        # replacement (LoRA/QLoRA-style). Enable only if needed.
        'distill': {
            'enabled': False,
            'alpha': 0.5,
            'temperature': 2.0,
        },

        # GaLore-like low-rank gradient projection schedule.
        'projection': {
            'enabled': True,
            'rank': 64,
            'update_gap': 200,
            'drift_threshold': 0.35,
            # Optional schedules (step-based):
            # 'rank_schedule': [{'step': 0, 'value': 32}, {'step': 200, 'value': 64}],
            # 'update_gap_schedule': [{'step': 0, 'value': 400}, {'step': 1000, 'value': 200}],
        },
    }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Qwen2.5 Quantization Pipeline: PiSSA -> SpinQuant -> GaLore"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/pipeline_config.yaml',
        help='Path to pipeline configuration file'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='Qwen/Qwen2.5-7B-Instruct',
        help='Model name or path'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='Output directory'
    )
    parser.add_argument(
        '--start_from',
        type=str,
        choices=['pissa', 'spinquant', 'galore'],
        help='Start from a specific stage'
    )
    parser.add_argument(
        '--create_config',
        action='store_true',
        help='Create default configuration file and exit'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(log_dir='logs', log_file='pipeline.log', logger_name='pipeline')
    
    # Create default config if requested
    if args.create_config:
        config = create_default_config()
        config_path = args.config
        ensure_dir(os.path.dirname(config_path))
        save_config(config, config_path)
        logger.info(f"Default configuration created at: {config_path}")
        return
    
    # Load or create configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
        logger.info(f"Configuration loaded from: {args.config}")
    else:
        logger.warning(f"Configuration file not found: {args.config}")
        logger.info("Using default configuration")
        config = create_default_config()
    
    # Override with command line arguments
    if args.model_name:
        config['model_name'] = args.model_name
    if args.output_dir:
        config['output_dir'] = args.output_dir

    # Important: keep pipeline stage progress isolated per output directory.
    # ProgressTracker persists state to <log_dir>/progress.txt.
    # If log_dir is shared across multiple runs (e.g., different model sizes),
    # a previous run can cause a new run to incorrectly resume at a later stage,
    # leading to missing artifacts like `residual_model/`.
    config['log_dir'] = os.path.join(config['output_dir'], 'logs')
    
    # Run pipeline
    pipeline = QuantizationPipeline(config, logger)
    results = pipeline.run(start_from=args.start_from)
    
    logger.info("Pipeline execution completed successfully")


if __name__ == '__main__':
    main()
