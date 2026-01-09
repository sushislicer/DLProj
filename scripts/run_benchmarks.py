"""
Main benchmarking script for Qwen2.5 models on difficult benchmarks.
Supports multiple model sizes and multi-GPU execution (1-8 GPUs).
Includes 4-bit quantization support for faster execution.
Features:
- Debug mode with small sample datasets for quick validation
- Baseline comparison (4-bit quantization, 4-bit LoRA)
- Optimized execution times for larger models (14B, 32B, 72B)
"""

import os
import sys
import argparse
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import yaml
import torch
import numpy as np
from tqdm import tqdm

# NOTE: Do not import `transformers` at module import time.
# If the user's environment has an incompatible torch/transformers combination
# (common on fresh GPU servers), importing transformers can fail before we can
# emit a helpful error message. We import lazily inside helper functions.

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import setup_logging, ensure_dir, format_time
from utils.memory_tracker import MemoryTracker
from utils.flash_attention import pick_attn_implementation
from utils.hf_download import resolve_path_or_hf_repo, is_probably_hf_repo_id


def _import_transformers(logger: logging.Logger):
    """Lazy import transformers.

    Raises an ImportError with actionable remediation hints when a common
    torch/transformers mismatch occurs.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        return AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except Exception as e:
        # Common mismatch: newer transformers expects newer torch pytree API.
        msg = str(e)
        if "torch.utils._pytree" in msg and "register_pytree_node" in msg:
            raise ImportError(
                "Failed to import transformers due to a torch/transformers version mismatch.\n"
                "Your torch is too old for the installed transformers.\n\n"
                "Fix:\n"
                "- Install/upgrade a GPU-compatible torch wheel using `python3 scripts/fix_torch.py --reinstall --channel nightly` (RTX 50xx)\n"
                "  or reinstall a stable CUDA torch for your driver, then re-run `pip install -r requirements.txt`.\n"
                f"Original error: {msg}"
            )
        raise


class BenchmarkRunner:
    """Main benchmark runner for Qwen2.5 models."""
    
    def __init__(self, config_path: str, logger: logging.Logger):
        """
        Initialize benchmark runner.
        
        Args:
            config_path: Path to benchmark configuration file
            logger: Logger instance
        """
        self.logger = logger
        self.config = self._load_config(config_path)
        self.device_map = self._setup_device_map()
        self.memory_tracker = MemoryTracker(
            log_dir=self.config['logging']['log_dir'],
            log_file='benchmark_memory.log'
        )
        
        # Setup output directories
        self.output_dir = self._setup_output_dir()
        ensure_dir(self.output_dir)
        
        # Track results
        self.results = {
            'config': self.config,
            'system_info': self._get_system_info(),
            'benchmarks': {}
        }

        # Optional experiment tracking
        self._tb_writer = None
        self._wandb = None
        self._wandb_run = None
        self._setup_tracking()
        
        self.logger.info("=" * 80)
        self.logger.info("QWEN2.5 BENCHMARK RUNNER")
        self.logger.info("=" * 80)

    def _setup_tracking(self) -> None:
        """Initialize TensorBoard/W&B if enabled in config.

        Safe to call even if deps aren’t installed; will log and continue.
        """
        log_cfg = self.config.get('logging', {}) if isinstance(self.config.get('logging', {}), dict) else {}

        # TensorBoard
        tb_cfg = log_cfg.get('tensorboard', {}) if isinstance(log_cfg.get('tensorboard', {}), dict) else {}
        if bool(tb_cfg.get('enabled', False)):
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_dir = str(tb_cfg.get('log_dir') or os.path.join(log_cfg.get('log_dir', 'logs'), 'tensorboard'))
                ensure_dir(tb_dir)
                run_name = os.path.basename(self.output_dir.rstrip('/'))
                self._tb_writer = SummaryWriter(log_dir=os.path.join(tb_dir, run_name))
                self._tb_writer.add_text('config', json.dumps(self.config, indent=2), 0)
                self.logger.info(f"TensorBoard enabled. Logdir: {tb_dir}")
            except Exception as e:
                self.logger.warning(f"TensorBoard requested but not available: {e}")

        # Weights & Biases
        wb_cfg = log_cfg.get('wandb', {}) if isinstance(log_cfg.get('wandb', {}), dict) else {}
        if bool(wb_cfg.get('enabled', False)):
            try:
                import wandb

                mode = str(wb_cfg.get('mode') or os.environ.get('WANDB_MODE') or 'offline')
                os.environ.setdefault('WANDB_MODE', mode)
                if wb_cfg.get('project'):
                    os.environ.setdefault('WANDB_PROJECT', str(wb_cfg['project']))
                if wb_cfg.get('entity'):
                    os.environ.setdefault('WANDB_ENTITY', str(wb_cfg['entity']))

                self._wandb = wandb
                self._wandb_run = wandb.init(
                    project=str(wb_cfg.get('project') or os.environ.get('WANDB_PROJECT') or 'qwen-bench'),
                    entity=(str(wb_cfg['entity']) if wb_cfg.get('entity') else None),
                    name=(str(wb_cfg['run_name']) if wb_cfg.get('run_name') else None),
                    tags=list(wb_cfg.get('tags') or []),
                    config=self.config,
                )
                self.logger.info(f"wandb enabled (mode={mode}).")
            except Exception as e:
                self.logger.warning(f"wandb requested but not available: {e}")

    def _track_metrics(self, model_key: str, benchmark_name: str, metrics: Dict[str, Any], step: int) -> None:
        """Emit scalar metrics to TB/W&B."""
        if not isinstance(metrics, dict):
            return

        prefix = f"{model_key}/{benchmark_name}"
        # TensorBoard
        if self._tb_writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb_writer.add_scalar(f"{prefix}/{k}", float(v), step)

        # W&B
        if self._wandb is not None and self._wandb_run is not None:
            payload = {f"{prefix}/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
            if payload:
                self._wandb.log(payload, step=step)
        self.logger.info(f"Configuration loaded from: {config_path}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        self.logger.info("=" * 80)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_device_map(self) -> Dict:
        """Setup device map for multi-GPU support (1-8 GPUs)."""
        requested_gpus = int(self.config['gpu']['num_gpus'])
        device_map_strategy = self.config['gpu']['device_map']

        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if requested_gpus > 1 and available_gpus > 0 and requested_gpus > available_gpus:
            self.logger.warning(
                f"Requested gpu.num_gpus={requested_gpus} but only {available_gpus} GPUs are visible. "
                f"Clamping to {available_gpus}."
            )
            requested_gpus = available_gpus
            self.config['gpu']['num_gpus'] = requested_gpus

        num_gpus = requested_gpus
        
        if num_gpus > 1:
            self.logger.info(f"Setting up multi-GPU configuration with {num_gpus} GPUs")
            
            # Configure max memory per GPU
            max_memory = {}
            for i in range(num_gpus):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory
                # Use 90% of available memory
                max_memory[i] = f"{int(gpu_memory * 0.9 / 1024**3)}GB"
            
            self.logger.info(f"Max memory per GPU: {max_memory}")
            
            return {
                'device_map': device_map_strategy,
                'max_memory': max_memory,
                'offload_folder': self.config['advanced']['device_map_options']['offload_folder']
            }
        else:
            return {'device_map': 'auto'}
    
    def _setup_quantization(self) -> Optional[BitsAndBytesConfig]:
        """Setup 4-bit quantization configuration."""
        # Lazy import (see `_import_transformers`).
        _, _, BitsAndBytesConfig = _import_transformers(self.logger)
        if not self.config['gpu']['quantization']['enabled']:
            return None
        
        quant_config = self.config['gpu']['quantization']
        bits = quant_config['bits']
        
        self.logger.info(f"Setting up {bits}-bit quantization")
        
        # Determine compute dtype
        compute_dtype_str = quant_config.get('compute_dtype', 'float16')
        if compute_dtype_str == 'float16':
            compute_dtype = torch.float16
        elif compute_dtype_str == 'bfloat16':
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float32
        
        # Create BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(bits == 4),
            load_in_8bit=(bits == 8),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_config.get('double_quant', True),
            bnb_4bit_quant_type=quant_config.get('quant_type', 'nf4')
        )
        
        self.logger.info(f"Quantization config: {bits}-bit, {quant_config.get('quant_type', 'nf4')}, double_quant={quant_config.get('double_quant', True)}")
        
        return bnb_config
    
    def _setup_output_dir(self) -> str:
        """Setup output directory with timestamp."""
        base_dir = self.config['output']['base_dir']
        include_timestamp = self.config['output']['include_timestamp']
        
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(base_dir, f"benchmark_{timestamp}")
        else:
            output_dir = base_dir
        
        ensure_dir(output_dir)
        return output_dir
    
    def _get_system_info(self) -> Dict:
        """Get system information."""
        info = {
            'timestamp': datetime.now().isoformat(),
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'cuda_version': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(),
            'num_gpus': torch.cuda.device_count(),
            'gpu_info': []
        }
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info['gpu_info'].append({
                'device_id': i,
                'name': props.name,
                'total_memory': f"{props.total_memory / 1024**3:.2f} GB",
                'compute_capability': f"{props.major}.{props.minor}"
            })
        
        return info
    
    def load_model(self, model_config: Dict) -> tuple:
        """
        Load model and tokenizer with quantization support.
        
        Args:
            model_config: Model configuration dictionary
        
        Returns:
            Tuple of (model, tokenizer)
        """
        model_name = model_config['name']
        model_size = model_config['size']
        
        self.logger.info("=" * 80)
        self.logger.info(f"Loading model: {model_name} ({model_size})")
        self.logger.info("=" * 80)
        
        self.memory_tracker.log_memory(f"load_model_{model_size}", "Starting model load")
        start_time = time.time()
        
        # Quantization can be overridden per-model (e.g., baselines).
        # Values: "none" | "4bit" | "8bit" (default uses global config)
        quant_override = str(model_config.get('quantization', '')).lower().strip() if model_config.get('quantization') is not None else ''
        original_quant_enabled = bool(self.config['gpu']['quantization']['enabled'])
        original_bits = int(self.config['gpu']['quantization'].get('bits', 4))

        if quant_override in ("none", "fp16", "fp32"):
            self.config['gpu']['quantization']['enabled'] = False
        elif quant_override in ("4bit", "4", "int4"):
            self.config['gpu']['quantization']['enabled'] = True
            self.config['gpu']['quantization']['bits'] = 4
        elif quant_override in ("8bit", "8", "int8"):
            self.config['gpu']['quantization']['enabled'] = True
            self.config['gpu']['quantization']['bits'] = 8

        # Setup quantization (after overrides)
        quantization_config = self._setup_quantization()
        
        # Determine torch dtype (only used if not quantizing)
        dtype_str = self.config['gpu']['torch_dtype']
        if dtype_str == 'float16':
            torch_dtype = torch.float16
        elif dtype_str == 'bfloat16':
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
        
        # Lazy import transformers (avoid hard failure at module import time).
        AutoModelForCausalLM, AutoTokenizer, _ = _import_transformers(self.logger)

        # Load tokenizer
        self.logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.config['gpu']['trust_remote_code'],
            cache_dir=self.config['evaluation']['cache_dir']
        )
        
        # Set pad token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Decoder-only models should left-pad for batched generation.
        # Right padding can cause incorrect attention for some architectures.
        try:
            tokenizer.padding_side = "left"
        except Exception:
            pass
        
        # Load model with device map and quantization
        self.logger.info("Loading model...")
        model_kwargs = {
            'device_map': self.device_map['device_map'],
            'trust_remote_code': self.config['gpu']['trust_remote_code'],
            'low_cpu_mem_usage': self.config['advanced']['low_cpu_mem_usage'],
            'cache_dir': self.config['evaluation']['cache_dir'],
        }

        # Prefer FlashAttention2 if enabled.
        # If missing and auto-install is enabled, attempt to install it.
        # If still unavailable, fall back to default attention.
        prefer_flash2 = bool(self.config['advanced'].get('use_flash_attention', False)) or bool(
            self.config['gpu']['quantization'].get('use_flash_attention', False)
        )
        auto_install_flash2 = bool(self.config['advanced'].get('auto_install_flash_attention', False)) or (
            str(os.environ.get('AUTO_INSTALL_FLASH_ATTN', '')).strip() in ('1', 'true', 'True', 'yes', 'YES')
        )
        attn_impl = pick_attn_implementation(
            logger=self.logger,
            prefer_flash2=prefer_flash2,
            auto_install=auto_install_flash2,
        )
        if attn_impl:
            model_kwargs['attn_implementation'] = attn_impl
        
        # Add quantization config if enabled
        if quantization_config is not None:
            model_kwargs['quantization_config'] = quantization_config
            self.logger.info("Loading model with 4-bit quantization")
        else:
            model_kwargs['torch_dtype'] = torch_dtype
        
        # Add max_memory if specified
        if 'max_memory' in self.device_map:
            model_kwargs['max_memory'] = self.device_map['max_memory']
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
        except Exception as e:
            # Common failure mode: flash-attn not installed / unsupported attention.
            if 'attn_implementation' in model_kwargs:
                self.logger.warning(f"Model load failed with attn_implementation=flash_attention_2; retrying without it. Error: {e}")
                model_kwargs.pop('attn_implementation', None)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs
                )
            else:
                raise

        # Optional: load adapters (LoRA/PiSSA) on top of the base model.
        adapter_path = model_config.get('adapter_path')
        if adapter_path:
            # Allow adapter_path to be either a local directory or a HuggingFace Hub repo id.
            # If repo id, download snapshot into cache_dir.
            adapter_path = resolve_path_or_hf_repo(
                str(adapter_path),
                cache_dir=self.config.get('evaluation', {}).get('cache_dir'),
                logger=self.logger,
            )
            if not os.path.exists(str(adapter_path)):
                raise FileNotFoundError(
                    f"adapter_path does not exist after resolution: {adapter_path}. "
                    "Provide a local PEFT adapter directory or a Hub repo id like 'org/repo[@rev]'."
                )
            try:
                from peft import PeftModel
            except Exception as e:
                raise ImportError("peft is required to load adapters. Install `peft`.\n" + str(e))

            self.logger.info(f"Loading adapters from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
            model.eval()
        
        # Ensure eval mode for benchmarking.
        model.eval()
        
        # Inference optimizations.
        # NOTE: Do *not* enable gradient checkpointing during inference; it slows
        # generation and can disable KV cache.
        try:
            if hasattr(model.config, 'use_cache'):
                model.config.use_cache = True
        except Exception:
            pass

        # Restore global quantization config (avoid leaking baseline overrides).
        self.config['gpu']['quantization']['enabled'] = original_quant_enabled
        self.config['gpu']['quantization']['bits'] = original_bits
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Model loaded in {format_time(elapsed_time)}")
        self.memory_tracker.log_memory(f"load_model_{model_size}", f"Model loaded in {format_time(elapsed_time)}")
        
        return model, tokenizer

    def _expand_model_variants(self, model_config: Dict) -> List[Dict]:
        """Expand a base model config into runnable variants.

        We keep this *opt-in* via CLI (`--run_baselines`) to avoid turning the
        default 2-4h sweep into a 6-12h sweep.
        """
        # Default behavior: run the model config as-is.
        # When `--run_baselines` is enabled, we instead run a triad:
        #   1) pipeline_4bit ("our pipeline" in 4-bit; typically base 4-bit + our adapters)
        #   2) baseline_4bit (native 4-bit, no adapters)
        #   3) baseline_4bit_lora (4-bit base + LoRA adapters baseline)
        # This avoids accidentally benchmarking the same "no-adapter" model twice.

        baselines_cfg = self.config.get('baselines', {})

        if not getattr(self, 'run_baselines', False):
            main_cfg = dict(model_config)
            main_cfg.setdefault('variant', 'main')
            return [main_cfg]

        variants: List[Dict] = []

        def _resolve_adapter_path(v) -> Optional[str]:
            """Resolve adapter_path config.

            Supports either a string or a mapping like:
              {"default": "...", "7B": "...", "14B": "..."}
            """
            if v is None:
                return None
            if isinstance(v, dict):
                return v.get(model_config.get('size')) or v.get('default')
            return str(v)

        def _mk_variant(base: Dict, variant_id: str, bcfg: Dict) -> Dict:
            out = dict(base)
            out['variant'] = variant_id
            # Apply baseline overrides
            if 'use_adapters' in bcfg and not bool(bcfg['use_adapters']):
                out.pop('adapter_path', None)
            if bool(bcfg.get('use_adapters', False)):
                out['adapter_path'] = _resolve_adapter_path(bcfg.get('adapter_path'))
            if 'quantization' in bcfg:
                out['quantization'] = bcfg['quantization']
            return out

        # 1) Our pipeline run (in 4-bit).
        # Prefer per-model adapter_path; else fall back to baselines.pipeline_4bit.adapter_path.
        p = baselines_cfg.get('pipeline_4bit', {})
        pipeline_cfg = dict(model_config)
        pipeline_cfg['variant'] = 'pipeline_4bit'
        pipeline_cfg['quantization'] = '4bit'
        if not pipeline_cfg.get('adapter_path'):
            ap = _resolve_adapter_path(p.get('adapter_path'))
            if ap:
                pipeline_cfg['adapter_path'] = ap

        if p.get('enabled', True):
            if not pipeline_cfg.get('adapter_path'):
                self.logger.warning(
                    "pipeline_4bit enabled but no adapter_path configured. "
                    "Set models[].adapter_path OR baselines.pipeline_4bit.adapter_path. "
                    "(If you intended a no-adapter pipeline, ignore this.)"
                )
            variants.append(pipeline_cfg)

        # Native 4-bit quant baseline (no adapters)
        b1 = baselines_cfg.get('quantization_4bit', {})
        if b1.get('enabled', False):
            variants.append(_mk_variant(model_config, 'baseline_4bit', b1))

        # 4-bit LoRA baseline (4-bit base + adapters)
        b2 = baselines_cfg.get('quantization_4bit_lora', {})
        if b2.get('enabled', False):
            cand = _mk_variant(model_config, 'baseline_4bit_lora', b2)
            ap = cand.get('adapter_path')
            if not ap:
                self.logger.warning(
                    "Skipping baseline_4bit_lora because adapter_path is missing. "
                    f"Set baselines.quantization_4bit_lora.adapter_path to a valid PEFT adapter directory. Got: {ap}"
                )
            else:
                # If adapter_path is a remote repo id, allow it; it will be downloaded at load time.
                if os.path.exists(str(ap)) or is_probably_hf_repo_id(str(ap)):
                    variants.append(cand)
                else:
                    self.logger.warning(
                        "Skipping baseline_4bit_lora because adapter_path is neither an existing local path nor a valid Hub repo id. "
                        f"Got: {ap}"
                    )

        return variants
    
    def run_benchmark(
        self,
        model,
        tokenizer,
        model_config: Dict,
        benchmark_name: str,
        benchmark_config: Dict
    ) -> Dict:
        """
        Run a specific benchmark.
        
        Args:
            model: Loaded model
            tokenizer: Loaded tokenizer
            model_config: Model configuration
            benchmark_name: Name of the benchmark
            benchmark_config: Benchmark configuration
        
        Returns:
            Dictionary with benchmark results
        """
        self.logger.info("=" * 80)
        self.logger.info(f"Running benchmark: {benchmark_name}")
        self.logger.info(f"Model: {model_config['name']} ({model_config['size']})")
        self.logger.info("=" * 80)
        
        # Import the appropriate benchmark runner
        if benchmark_name == 'aime':
            from benchmarks.aime_benchmark import AIMEBenchmark
            benchmark = AIMEBenchmark(benchmark_config, self.logger)
        elif benchmark_name == 'math':
            from benchmarks.math_benchmark import MATHBenchmark
            benchmark = MATHBenchmark(benchmark_config, self.logger)
        elif benchmark_name == 'livecodebench':
            from benchmarks.livecodebench_benchmark import LiveCodeBenchBenchmark
            benchmark = LiveCodeBenchBenchmark(benchmark_config, self.logger)
        elif benchmark_name == 'swe_bench':
            from benchmarks.swe_bench_benchmark import SWEBenchBenchmark
            benchmark = SWEBenchBenchmark(benchmark_config, self.logger)
        elif benchmark_name == 'gpqa':
            from benchmarks.gpqa_benchmark import GPQABenchmark
            benchmark = GPQABenchmark(benchmark_config, self.logger)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        # Run benchmark
        self.memory_tracker.log_memory(f"benchmark_{benchmark_name}", "Starting benchmark")
        start_time = time.time()
        
        # Log memory before benchmark
        gpu_memory = self.memory_tracker.get_gpu_memory()
        self.logger.info(f"GPU Memory before {benchmark_name}: {gpu_memory['allocated_mb']:.2f}MB / {gpu_memory['total_mb']:.2f}MB ({gpu_memory['percent_used']:.1f}%)")
        
        results = benchmark.run(model, tokenizer, model_config)

        # Emit metrics to tracking backends (TB/W&B)
        try:
            if isinstance(results, dict) and isinstance(results.get('metrics'), dict):
                model_key = f"{model_config.get('size')}__{model_config.get('variant', 'main')}"
                step = int(time.time())
                self._track_metrics(model_key, benchmark_name, results['metrics'], step)
        except Exception:
            pass
        
        elapsed_time = time.time() - start_time
        results['elapsed_time'] = elapsed_time
        results['formatted_time'] = format_time(elapsed_time)
        
        # Log memory after benchmark
        gpu_memory_after = self.memory_tracker.get_gpu_memory()
        self.logger.info(f"GPU Memory after {benchmark_name}: {gpu_memory_after['allocated_mb']:.2f}MB / {gpu_memory_after['total_mb']:.2f}MB ({gpu_memory_after['percent_used']:.1f}%)")
        
        # Log inference speed metrics
        if 'metrics' in results:
            metrics = results['metrics']
            self.logger.info(f"Inference Speed Metrics for {benchmark_name}:")
            if 'avg_latency' in metrics:
                self.logger.info(f"  Average Latency: {metrics['avg_latency']:.4f}s per sample")
            if 'throughput' in metrics:
                self.logger.info(f"  Throughput: {metrics['throughput']:.2f} samples/second")
            if 'total_latency' in metrics:
                self.logger.info(f"  Total Latency: {metrics['total_latency']:.2f}s")
        
        self.logger.info(f"Benchmark {benchmark_name} completed in {format_time(elapsed_time)}")
        self.memory_tracker.log_memory(f"benchmark_{benchmark_name}", f"Completed in {format_time(elapsed_time)}")
        
        return results
    
    def run_all_benchmarks_for_model(self, model_config: Dict) -> Dict:
        """
        Run all enabled benchmarks for a specific model.
        
        Args:
            model_config: Model configuration
        
        Returns:
            Dictionary with all benchmark results for the model
        """
        model_name = model_config['name']
        model_size = model_config['size']
        variant = model_config.get('variant', 'main')
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"BENCHMARKING MODEL: {model_name} ({model_size}) [{variant}]")
        self.logger.info("=" * 80)
        
        def _max_memory_to_gb(v) -> float:
            """Parse a max_memory entry to GB.

            HuggingFace accepts either strings like "20GB" or integer bytes.
            Some environments/tooling end up passing ints here.
            """
            if isinstance(v, (int, float)):
                # Heuristic: values above ~1024 are almost certainly bytes.
                if float(v) > 1024.0:
                    return float(v) / (1024.0 ** 3)
                return float(v)

            s = str(v).strip().upper()
            try:
                if s.endswith("GB"):
                    return float(s[:-2].strip())
                if s.endswith("GIB"):
                    return float(s[:-3].strip())
                if s.endswith("MB"):
                    return float(s[:-2].strip()) / 1024.0
                if s.endswith("MIB"):
                    return float(s[:-3].strip()) / 1024.0
                if s.endswith("B") and s[:-1].strip().isdigit():
                    return float(s[:-1].strip()) / (1024.0 ** 3)
            except Exception:
                pass

            # Fallback: try direct float
            try:
                return float(s)
            except Exception:
                return 0.0

        # Apply 14B/72B-specific optimizations before loading model
        if model_size in ["14B", "72B"] and model_config.get('aggressive_optimization', False):
            self.logger.info(f"Applying aggressive optimizations for {model_size} model")
            # Increase GPU memory utilization for 14B/72B
            if 'max_memory' in self.device_map:
                for i in self.device_map['max_memory']:
                    # Use 95% instead of 90% for 14B/72B
                    current_gb = _max_memory_to_gb(self.device_map['max_memory'][i])
                    boosted_gb = int(max(1.0, current_gb * 1.056))  # 95/90 = 1.056
                    self.device_map['max_memory'][i] = f"{boosted_gb}GB"
                self.logger.info(f"Optimized GPU memory for {model_size}: {self.device_map['max_memory']}")
        
        # Load model
        model, tokenizer = self.load_model(model_config)
        
        # Run all enabled benchmarks
        model_results = {
            'model_name': model_name,
            'model_size': model_size,
            'variant': variant,
            'benchmarks': {}
        }
        
        for benchmark_name, benchmark_config in self.config['benchmarks'].items():
            if not benchmark_config.get('enabled', False):
                self.logger.info(f"Skipping disabled benchmark: {benchmark_name}")
                continue
            
            # Check if this benchmark should be skipped for this model
            skip_benchmarks = model_config.get('skip_benchmarks', [])
            if benchmark_name in skip_benchmarks:
                self.logger.info(f"Skipping {benchmark_name} for {model_size} (in skip list)")
                continue
            
            # Adjust num_samples for specific models
            adjusted_config = benchmark_config.copy()

            # Debug mode: use debug_samples and smaller generation for fast iteration.
            debug_cfg = self.config.get('debug', {})
            if debug_cfg.get('enabled', False) and debug_cfg.get('use_debug_samples', True):
                dbg_n = benchmark_config.get('debug_samples', debug_cfg.get('debug_sample_size', 3))
                adjusted_config['num_samples'] = int(dbg_n)
                # Reduce generation in debug runs.
                if 'max_new_tokens' in adjusted_config:
                    adjusted_config['max_new_tokens'] = int(min(adjusted_config['max_new_tokens'], 64))
                if 'timeout' in adjusted_config:
                    adjusted_config['timeout'] = int(min(adjusted_config['timeout'], 30))
            if model_size == "72B" and f"num_samples_72b" in benchmark_config:
                adjusted_config['num_samples'] = benchmark_config['num_samples_72b']
                self.logger.info(f"Using reduced sample count for 72B model: {adjusted_config['num_samples']}")
            elif model_size == "14B" and f"num_samples_14b" in benchmark_config:
                adjusted_config['num_samples'] = benchmark_config['num_samples_14b']
                self.logger.info(f"Using reduced sample count for 14B model: {adjusted_config['num_samples']}")
            
            # Apply 14B/72B-specific optimizations
            if model_size in ["14B", "72B"]:
                if model_config.get('use_dynamic_batching', False):
                    self.logger.info(f"Enabling dynamic batching for {model_size} model")
                if model_config.get('early_stopping', False):
                    self.logger.info(f"Enabling early stopping for {model_size} model")
                if model_config.get('optimize_generation', False):
                    self.logger.info(f"Using optimized generation parameters for {model_size} model")
                
                # Apply aggressive optimizations for LiveCodeBench
                if benchmark_name == "livecodebench":
                    self.logger.info(f"Applying aggressive optimizations for LiveCodeBench on {model_size}")
                    # Reduce timeout and test cases for 14B/72B
                    if model_size == "72B" and f"timeout_72b" in benchmark_config:
                        adjusted_config['timeout'] = benchmark_config['timeout_72b']
                    else:
                        adjusted_config['timeout'] = benchmark_config.get('timeout', 120)
                    
                    if model_size == "72B" and f"num_test_cases_72b" in benchmark_config:
                        adjusted_config['num_test_cases'] = benchmark_config['num_test_cases_72b']
                    else:
                        adjusted_config['num_test_cases'] = benchmark_config.get('num_test_cases', 5)
                    
                    self.logger.info(f"LiveCodeBench {model_size}: timeout={adjusted_config['timeout']}s, test_cases={adjusted_config['num_test_cases']}")
            
            try:
                results = self.run_benchmark(
                    model, tokenizer, model_config,
                    benchmark_name, adjusted_config
                )
                model_results['benchmarks'][benchmark_name] = results
                
                # Save intermediate results
                if self.config['evaluation']['save_intermediate_results']:
                    self._save_results(model_results, f"intermediate_{model_size}_{benchmark_name}")
            
            except Exception as e:
                self.logger.error(f"Error running benchmark {benchmark_name}: {e}")
                import traceback
                traceback.print_exc()
                model_results['benchmarks'][benchmark_name] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Cleanup
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
        return model_results
    
    def run_all_models(self, model_sizes: Optional[List[str]] = None):
        """
        Run benchmarks for all models or specific model sizes.
        Supports time budget and priority-based execution.
        
        Args:
            model_sizes: Optional list of model sizes to run (e.g., ['3B', '7B'])
        """
        self.memory_tracker.start_tracking()
        total_start_time = time.time()
        
        # Check time budget
        max_execution_time = self.config['execution'].get('max_execution_time', 4) * 3600  # Convert to seconds
        enable_time_budget = self.config['execution'].get('enable_time_budget', True)
        
        # Filter models by size if specified
        base_models_to_run = self.config['models']
        if model_sizes:
            base_models_to_run = [m for m in base_models_to_run if m['size'] in model_sizes]
            self.logger.info(f"Running benchmarks for model sizes: {model_sizes}")
        
        # Sort models by priority (smaller models first)
        model_priority = self.config['execution'].get('model_priority', [])
        base_models_to_run.sort(key=lambda m: model_priority.index(m['size']) if m['size'] in model_priority else len(model_priority))

        # Expand into variants (main + optional baselines)
        models_to_run: List[Dict] = []
        for m in base_models_to_run:
            models_to_run.extend(self._expand_model_variants(m))
        
        # Run benchmarks for each model
        for model_config in models_to_run:
            # Check time budget
            if enable_time_budget:
                elapsed_time = time.time() - total_start_time
                remaining_time = max_execution_time - elapsed_time
                
                if remaining_time <= 0:
                    self.logger.warning(f"Time budget exceeded, skipping remaining models")
                    break
                
                self.logger.info(f"Time remaining: {remaining_time/60:.1f} minutes")
            
            try:
                model_results = self.run_all_benchmarks_for_model(model_config)
                key = model_config['size']
                variant = model_config.get('variant', 'main')
                if variant != 'main':
                    key = f"{key}__{variant}"
                self.results['benchmarks'][key] = model_results
            except Exception as e:
                self.logger.error(f"Error benchmarking model {model_config['name']}: {e}")
                import traceback
                traceback.print_exc()
                self.results['benchmarks'][model_config['size']] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        total_elapsed_time = time.time() - total_start_time
        self.results['total_elapsed_time'] = total_elapsed_time
        self.results['total_formatted_time'] = format_time(total_elapsed_time)
        
        # Save final results
        self._save_results(self.results, 'final_results')

        # Save quick charts for clarity (optional; requires matplotlib/seaborn).
        self._save_quick_charts()
        
        # Stop memory tracking
        self.memory_tracker.stop_tracking()
        self.memory_tracker.print_summary()
        
        # Print summary
        self._print_summary()

        # Close tracking runs
        try:
            if self._tb_writer is not None:
                self._tb_writer.flush()
                self._tb_writer.close()
        except Exception:
            pass
        try:
            if self._wandb_run is not None:
                self._wandb_run.finish()
        except Exception:
            pass

    def _save_quick_charts(self) -> None:
        """Write a couple of quick PNG charts into the run output directory.

        This is a convenience so you can see results immediately without running
        the full report generator.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
        except Exception as e:
            self.logger.info(f"Plotting dependencies not available; skipping quick charts. ({e})")
            return

        rows = []
        for model_key, model_data in self.results.get('benchmarks', {}).items():
            if not isinstance(model_data, dict):
                continue
            benches = model_data.get('benchmarks', {})
            if not isinstance(benches, dict):
                continue
            for bench_name, bench_data in benches.items():
                if not isinstance(bench_data, dict):
                    continue
                metrics = bench_data.get('metrics', {})
                if not isinstance(metrics, dict):
                    continue
                row = {
                    'model': model_key,
                    'benchmark': bench_name,
                    'accuracy': metrics.get('accuracy', None),
                    'throughput': metrics.get('throughput', None),
                    'avg_latency': metrics.get('avg_latency', None),
                }
                rows.append(row)

        if not rows:
            return

        df = pd.DataFrame(rows)
        sns.set_theme(style='whitegrid')

        # Accuracy chart
        if df['accuracy'].notna().any():
            p = df.dropna(subset=['accuracy'])
            plt.figure(figsize=(12, 5))
            sns.barplot(data=p, x='benchmark', y='accuracy', hue='model')
            plt.title('Accuracy by benchmark (per model/variant)')
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            out_path = os.path.join(self.output_dir, 'quick_accuracy.png')
            plt.savefig(out_path, dpi=200)
            plt.close()
            self.logger.info(f"Quick chart saved to: {out_path}")

        # Throughput chart
        if df['throughput'].notna().any():
            p = df.dropna(subset=['throughput'])
            plt.figure(figsize=(12, 5))
            sns.barplot(data=p, x='benchmark', y='throughput', hue='model')
            plt.title('Throughput (samples/s) by benchmark (per model/variant)')
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            out_path = os.path.join(self.output_dir, 'quick_throughput.png')
            plt.savefig(out_path, dpi=200)
            plt.close()
            self.logger.info(f"Quick chart saved to: {out_path}")
    
    def _save_results(self, results: Dict, filename: str):
        """Save results to file."""
        output_format = self.config['output']['format']
        
        if output_format in ['json', 'both']:
            json_path = os.path.join(self.output_dir, f"{filename}.json")
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Results saved to: {json_path}")
        
        if output_format in ['csv', 'both']:
            csv_path = os.path.join(self.output_dir, f"{filename}.csv")
            self._save_results_csv(results, csv_path)
            self.logger.info(f"Results saved to: {csv_path}")
    
    def _save_results_csv(self, results: Dict, csv_path: str):
        """Save results to CSV format."""
        import csv
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['Model Size', 'Benchmark', 'Metric', 'Value'])
            
            # Write data
            for model_size, model_data in results.get('benchmarks', {}).items():
                if 'error' in model_data:
                    writer.writerow([model_size, 'ALL', 'ERROR', model_data['error']])
                    continue
                
                for benchmark_name, benchmark_data in model_data.get('benchmarks', {}).items():
                    if 'error' in benchmark_data:
                        writer.writerow([model_size, benchmark_name, 'ERROR', benchmark_data['error']])
                        continue
                    
                    for metric_name, metric_value in benchmark_data.get('metrics', {}).items():
                        writer.writerow([model_size, benchmark_name, metric_name, metric_value])
    
    def _print_summary(self):
        """Print benchmark summary."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("BENCHMARK SUMMARY")
        self.logger.info("=" * 80)
        
        for model_size, model_data in self.results['benchmarks'].items():
            if 'error' in model_data:
                self.logger.info(f"\n{model_size}: FAILED - {model_data['error']}")
                continue
            
            self.logger.info(f"\n{model_size}:")
            for benchmark_name, benchmark_data in model_data.get('benchmarks', {}).items():
                if 'error' in benchmark_data:
                    self.logger.info(f"  {benchmark_name}: FAILED - {benchmark_data['error']}")
                    continue
                
                self.logger.info(f"  {benchmark_name}:")
                for metric_name, metric_value in benchmark_data.get('metrics', {}).items():
                    self.logger.info(f"    {metric_name}: {metric_value}")
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"Total time: {self.results['total_formatted_time']}")
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info("=" * 80)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen2.5 models on difficult benchmarks"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/benchmark_config.yaml',
        help='Path to benchmark configuration file'
    )
    parser.add_argument(
        '--model_sizes',
        type=str,
        nargs='+',
        help='Model sizes to benchmark (e.g., 0.5B 7B 32B)'
    )
    parser.add_argument(
        '--benchmarks',
        type=str,
        nargs='+',
        help='Specific benchmarks to run (e.g., aime math gpqa)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Override output directory'
    )
    parser.add_argument(
        '--num_gpus',
        type=int,
        help='Number of GPUs to use (overrides config)'
    )
    parser.add_argument(
        '--quantize',
        action='store_true',
        help='Enable 4-bit quantization'
    )
    parser.add_argument(
        '--no_quantize',
        action='store_true',
        help='Disable quantization'
    )
    parser.add_argument(
        '--time_budget',
        type=float,
        help='Time budget in hours'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (uses benchmark debug_samples and can skip slow parts)'
    )

    parser.add_argument(
        '--run_baselines',
        action='store_true',
        help='Also run baseline variants (native 4-bit quant, 4-bit QLoRA) as configured in the YAML'
    )

    parser.add_argument(
        '--no_flash_attn',
        action='store_true',
        help='Force-disable FlashAttention2 usage and auto-install (use standard attention instead)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(log_dir='logs/benchmark', log_file='benchmark.log', logger_name='benchmark')
    
    # Create benchmark runner
    runner = BenchmarkRunner(args.config, logger)

    # Baseline execution is opt-in.
    runner.run_baselines = bool(args.run_baselines)

    # Enable debug mode
    if args.debug:
        runner.config.setdefault('debug', {})
        runner.config['debug']['enabled'] = True
        runner.config['debug'].setdefault('use_debug_samples', True)
        logger.info("Debug mode enabled")
    
    # Override output directory if specified
    if args.output_dir:
        runner.output_dir = args.output_dir
        ensure_dir(runner.output_dir)
    
    # Override number of GPUs if specified
    if args.num_gpus:
        runner.config['gpu']['num_gpus'] = args.num_gpus
        logger.info(f"Overriding GPU count to: {args.num_gpus}")
    
    # Override quantization settings if specified
    if args.quantize:
        runner.config['gpu']['quantization']['enabled'] = True
        logger.info("Enabling 4-bit quantization")
    elif args.no_quantize:
        runner.config['gpu']['quantization']['enabled'] = False
        logger.info("Disabling quantization")
    
    # Override time budget if specified
    if args.time_budget:
        runner.config['execution']['max_execution_time'] = args.time_budget
        logger.info(f"Setting time budget to: {args.time_budget} hours")
    
    # Disable benchmarks not specified
    if args.benchmarks:
        for benchmark_name in runner.config['benchmarks']:
            if benchmark_name not in args.benchmarks:
                runner.config['benchmarks'][benchmark_name]['enabled'] = False
    
    # Optional: force-disable FlashAttention2.
    # This is a safety valve for environments where flash-attn is unstable or
    # you want fully portable behavior.
    if args.no_flash_attn or (str(os.environ.get('DISABLE_FLASH_ATTN', '')).strip() in ('1', 'true', 'True', 'yes', 'YES')):
        runner.config.setdefault('advanced', {})
        runner.config['advanced']['use_flash_attention'] = False
        runner.config['advanced']['auto_install_flash_attention'] = False
        runner.config.setdefault('gpu', {}).setdefault('quantization', {})
        runner.config['gpu']['quantization']['use_flash_attention'] = False
        logger.info('FlashAttention2 disabled for this run')

    # Run benchmarks
    runner.run_all_models(model_sizes=args.model_sizes)
    
    logger.info("Benchmarking completed successfully")


if __name__ == '__main__':
    main()
