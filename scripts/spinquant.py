"""
SpinQuant quantization script for the residual base model.
Quantizes and freezes the residual model using SpinQuant method.
"""

import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import (
    setup_logging, get_device, print_model_size,
    ensure_dir, set_seed, format_time
)
from utils.memory_tracker import MemoryTracker
from utils.rotation_quant import RotationQuantConfig, learn_and_apply_rotations, apply_hadamard_rotations
import time


class SpinQuantizer:
    """Quantize model using SpinQuant method."""
    
    def __init__(self, config: dict, logger):
        """
        Initialize SpinQuantizer.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.device = get_device()
        
        # Initialize memory tracker
        self.memory_tracker = MemoryTracker(
            log_dir=config.get('log_dir', 'logs'),
            log_file='spinquant_memory.log'
        )
        
        # Set random seed
        set_seed(config.get('seed', 42))
        
        # Load residual model
        model_path = config['residual_model_path']
        self.logger.info(f"Loading residual model from {model_path}")
        self.memory_tracker.log_memory("SpinQuant", "Loading residual model...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        self.logger.info("Residual model loaded successfully")
        self.memory_tracker.log_memory("SpinQuant", "Residual model loaded")
        print_model_size(self.model, "Residual Model (Pre-Quantization)")

        # For reporting/visualization
        self.layer_summary = {}
    
    def prepare_calibration_data(self):
        """
        Prepare calibration data for SpinQuant.
        
        Returns:
            List of calibration examples
        """
        self.logger.info("Preparing calibration data...")
        
        # Sample calibration prompts
        calibration_prompts = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world.",
            "Machine learning models require large datasets.",
            "Quantization reduces model size and improves efficiency.",
            "Natural language processing enables human-computer interaction.",
            "Deep learning architectures have revolutionized AI.",
            "The future of technology depends on innovation.",
            "Data science combines statistics and programming.",
            "Neural networks learn patterns from data.",
            "Computer vision enables machines to see.",
        ]
        
        # Tokenize calibration data
        calibration_inputs = []
        for prompt in calibration_prompts:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            calibration_inputs.append(inputs)
        
        self.logger.info(f"Prepared {len(calibration_inputs)} calibration examples")
        return calibration_inputs

    def capture_linear_inputs(self, calibration_prompts: list, max_vectors_per_layer: int = 512):
        """Capture input activation vectors for target Linear layers.

        We use these vectors to weight the rotation objective toward preserving
        functional behavior (output error proxy) rather than pure weight MSE.

        Returns:
            Dict[layer_name, Tensor[N, in_features]] on CPU.
        """
        targets = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and any(
                s in name for s in self.config['spinquant'].get('target_modules', [
                    'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'
                ])
            ):
                targets.append((name, module))

        captured = {name: [] for name, _ in targets}

        hooks = []

        def make_hook(layer_name: str):
            def hook(module, inputs):
                if layer_name not in captured:
                    return
                if not inputs:
                    return
                x = inputs[0]
                # x can be [B, T, C] or [B, C]
                if x is None:
                    return
                if x.dim() == 3:
                    x = x.reshape(-1, x.shape[-1])
                elif x.dim() == 2:
                    pass
                else:
                    return

                # Move small sample to CPU.
                remaining = max_vectors_per_layer - sum(t.shape[0] for t in captured[layer_name])
                if remaining <= 0:
                    return
                take = min(remaining, x.shape[0])
                captured[layer_name].append(x[:take].detach().to('cpu', dtype=torch.float16))
            return hook

        for name, module in targets:
            hooks.append(module.register_forward_pre_hook(make_hook(name)))

        # Choose an input device (best-effort) for model-parallel setups.
        try:
            input_device = next(self.model.parameters()).device
        except StopIteration:
            input_device = torch.device("cpu")

        try:
            self.model.eval()
            batch = self.tokenizer(
                calibration_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config['spinquant'].get('calibration_max_length', 512),
            ).to(input_device)
            with torch.no_grad():
                _ = self.model(**batch)
        finally:
            for h in hooks:
                h.remove()

        # Concatenate
        out = {}
        for name, chunks in captured.items():
            if chunks:
                out[name] = torch.cat(chunks, dim=0)
        return out
    
    def apply_spinquant(self):
        """
        Apply SpinQuant quantization to the model.
        
        This method:
        1. Prepares calibration data
        2. Applies SpinQuant quantization with learned rotation matrices
        3. Freezes the quantized model
        
        NOTE:
        Official SpinQuant uses a Cayley-manifold optimization loop.

        This repo provides a SpinQuant-lite backend that learns *blockwise
        orthogonal rotations* (Givens sweeps) and optimizes a simple objective on
        weight reconstruction error after fake quantization.
        """
        spinquant_config = self.config['spinquant']
        
        backend = spinquant_config.get('backend', 'blockwise_givens')
        self.logger.info(f"Applying SpinQuant backend: {backend}")
        self.logger.info("=" * 60)
        self.logger.info("SPINQUANT-LITE (this repo):")
        self.logger.info("  - Learns blockwise orthogonal rotations via Givens sweeps")
        self.logger.info("  - Optimizes weight-domain reconstruction error after fake quant")
        self.logger.info("  - Applies rotation+fake-quant in-place to selected Linear layers")
        self.logger.info("=" * 60)
        self.memory_tracker.log_memory("SpinQuant", "Preparing calibration data...")
        start_time = time.time()
        
        # Prepare calibration data and (optionally) capture activation vectors.
        calibration_inputs = self.prepare_calibration_data()
        self.memory_tracker.log_memory("SpinQuant", "Calibration data prepared")

        if backend not in ('blockwise_givens', 'hadamard'):
            raise ValueError(
                f"Unsupported backend: {backend}. Supported: blockwise_givens, hadamard"
            )

        # Rotation learning can be expensive for large models.
        # Allow skipping it entirely and falling back to plain bnb quantization.
        skip_rotations = bool(spinquant_config.get('skip_rotations', False))
        if skip_rotations:
            self.logger.info("[SpinQuant-lite] skip_rotations=true; skipping rotation learning and using standard bitsandbytes quantization")
            self.layer_summary = {}
        else:
            rq_cfg = RotationQuantConfig(
                bits=int(spinquant_config.get('bits', 8)),
                block_size=int(spinquant_config.get('block_size', 64)),
                num_steps=int(spinquant_config.get('num_steps', 50)),
                lr=float(spinquant_config.get('lr', 5e-2)),
                num_sweeps=int(spinquant_config.get('num_sweeps', 2)),
                max_layers=int(spinquant_config.get('max_layers', 16)),
                use_activation_objective=bool(spinquant_config.get('use_activation_objective', True)),
                backend=str(spinquant_config.get('backend', 'blockwise_givens')),
            )

            if rq_cfg.backend == 'hadamard':
                layer_summary = apply_hadamard_rotations(
                    self.model,
                    rq_cfg,
                    logger=self.logger,
                )
            else:
                activation_samples = None
                if rq_cfg.use_activation_objective:
                    prompts = [
                        "Explain why 13 is a prime number.",
                        "Solve: If x+3=10, what is x?",
                        "Write a short Python function to add two numbers.",
                    ]
                    activation_samples = self.capture_linear_inputs(
                        prompts,
                        max_vectors_per_layer=int(spinquant_config.get('calibration_vectors_per_layer', 512)),
                    )

                layer_summary = learn_and_apply_rotations(
                    self.model,
                    rq_cfg,
                    logger=self.logger,
                    activation_samples=activation_samples,
                )
            self.logger.info(f"SpinQuant-lite updated layers: {len(layer_summary)}")
            self.layer_summary = layer_summary

        # Optional: convert to *real* bitsandbytes quantized modules for lower VRAM.
        # This more closely matches the benchmarking runner's 4-bit weight memory.
        use_bnb = bool(spinquant_config.get('use_bnb_quantization', True))
        if use_bnb:
            bits = int(spinquant_config.get('bits', 8))
            if bits not in (4, 8):
                raise ValueError(f"bits must be 4 or 8 for bnb quantization, got {bits}")

            # Optional: keep a few sensitive modules in fp16 even when quantizing.
            keep_fp16 = spinquant_config.get('keep_fp16_modules', ['input_embeddings', 'output_embeddings'])
            fp16_in_sd = None
            fp16_out_sd = None
            try:
                if 'input_embeddings' in keep_fp16 and hasattr(self.model, 'get_input_embeddings'):
                    emb = self.model.get_input_embeddings()
                    if emb is not None:
                        fp16_in_sd = {k: v.detach().to('cpu', dtype=torch.float16) for k, v in emb.state_dict().items()}
                if 'output_embeddings' in keep_fp16 and hasattr(self.model, 'get_output_embeddings'):
                    out_emb = self.model.get_output_embeddings()
                    if out_emb is not None:
                        fp16_out_sd = {k: v.detach().to('cpu', dtype=torch.float16) for k, v in out_emb.state_dict().items()}
            except Exception as e:
                self.logger.warning(f"Failed to snapshot fp16 embeddings before bnb reload: {e}")

            self.logger.info(f"Re-loading rotated model with bitsandbytes {bits}-bit quantization...")
            rotated_dir = os.path.join(self.config['output_dir'], 'rotated_residual_model')
            ensure_dir(rotated_dir)

            # Save rotated fp16 weights.
            self.model.save_pretrained(rotated_dir)
            self.tokenizer.save_pretrained(rotated_dir)

            # Load with BitsAndBytesConfig.
            quant_type = spinquant_config.get('quant_type', 'nf4')
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=(bits == 4),
                load_in_8bit=(bits == 8),
                bnb_4bit_use_double_quant=bool(spinquant_config.get('double_quant', True)),
                bnb_4bit_quant_type=quant_type,
                bnb_4bit_compute_dtype=torch.float16,
            )

            # Re-load into quantized modules.
            self.model = AutoModelForCausalLM.from_pretrained(
                rotated_dir,
                device_map="auto",
                trust_remote_code=True,
                quantization_config=bnb_cfg,
                low_cpu_mem_usage=True,
            )

            # Restore selected fp16 modules if requested.
            try:
                if fp16_in_sd is not None and hasattr(self.model, 'get_input_embeddings') and hasattr(self.model, 'set_input_embeddings'):
                    emb = self.model.get_input_embeddings()
                    if emb is not None:
                        emb.load_state_dict(fp16_in_sd, strict=True)
                        self.model.set_input_embeddings(emb)
                        self.logger.info("Restored fp16 input embeddings")
                if fp16_out_sd is not None and hasattr(self.model, 'get_output_embeddings') and hasattr(self.model, 'set_output_embeddings'):
                    out_emb = self.model.get_output_embeddings()
                    if out_emb is not None:
                        out_emb.load_state_dict(fp16_out_sd, strict=True)
                        self.model.set_output_embeddings(out_emb)
                        self.logger.info("Restored fp16 output embeddings (lm_head)")
            except Exception as e:
                self.logger.warning(f"Failed to restore fp16 modules after bnb reload: {e}")

            self.logger.info("Bitsandbytes-quantized model loaded")
        
        # Freeze all parameters after quantization
        for param in self.model.parameters():
            param.requires_grad = False
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"SpinQuant quantization completed in {format_time(elapsed_time)}")
        self.memory_tracker.log_memory("SpinQuant", "Quantization completed")
        
        print_model_size(self.model, "Quantized Residual Model")
        
        return self.model
    
    def save_quantized_model(self, output_dir: str):
        """
        Save the quantized model.
        
        Args:
            output_dir: Directory to save quantized model
        """
        ensure_dir(output_dir)
        
        self.logger.info(f"Saving quantized model to {output_dir}")
        
        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save quantization config
        import json
        quant_config_path = os.path.join(output_dir, 'quantization_config.json')
        with open(quant_config_path, 'w') as f:
            json.dump(self.config['spinquant'], f, indent=2)

        # Save SpinQuant-lite per-layer summary for paper figures.
        if getattr(self, 'layer_summary', None):
            summary_path = os.path.join(output_dir, 'spinquant_layer_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(self.layer_summary, f, indent=2)
            self.logger.info(f"SpinQuant layer summary saved to {summary_path}")
        
        self.logger.info("Quantized model saved successfully")
    
    def quantize_and_save(self):
        """
        Complete pipeline: quantize and save the model.
        """
        # Start memory tracking
        self.memory_tracker.start_tracking()
        
        # Apply SpinQuant
        self.apply_spinquant()
        
        # Save quantized model
        output_dir = os.path.join(
            self.config['output_dir'],
            'quantized_model'
        )
        self.memory_tracker.log_memory("SpinQuant", "Saving quantized model...")
        self.save_quantized_model(output_dir)
        
        self.logger.info("SpinQuant quantization completed successfully")
        
        # Stop memory tracking
        self.memory_tracker.stop_tracking()
        self.memory_tracker.print_summary()
        
        return {
            'quantized_model_path': output_dir
        }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Quantize model using SpinQuant")
    parser.add_argument(
        '--residual_model_path',
        type=str,
        required=True,
        help='Path to the residual model from PiSSA'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/quantized_models',
        help='Output directory for quantized model'
    )
    parser.add_argument(
        '--bits',
        type=int,
        default=8,
        choices=[4, 8],
        help='Quantization bits (4 or 8)'
    )
    parser.add_argument(
        '--double_quant',
        action='store_true',
        help='Use double quantization'
    )
    parser.add_argument(
        '--quant_type',
        type=str,
        default='nf4',
        choices=['nf4', 'fp4'],
        help='Quantization type (for 4-bit)'
    )

    # SpinQuant-lite backend knobs
    parser.add_argument(
        '--backend',
        type=str,
        default='blockwise_givens',
        choices=['blockwise_givens', 'hadamard'],
        help='SpinQuant backend implementation'
    )
    parser.add_argument('--block_size', type=int, default=64, help='Rotation block size')
    parser.add_argument('--num_steps', type=int, default=50, help='Optimization steps per layer')
    parser.add_argument('--lr', type=float, default=5e-2, help='Optimizer learning rate')
    parser.add_argument('--num_sweeps', type=int, default=2, help='Givens sweeps per block')
    parser.add_argument('--max_layers', type=int, default=16, help='Max Linear layers to optimize')
    parser.add_argument(
        '--no_activation_objective',
        action='store_true',
        help='Disable activation-weighted objective and use plain weight MSE'
    )
    parser.add_argument(
        '--calibration_vectors_per_layer',
        type=int,
        default=512,
        help='Max activation vectors captured per layer for the activation-weighted objective'
    )
    parser.add_argument(
        '--keep_fp16_modules',
        type=str,
        default='input_embeddings,output_embeddings',
        help='Comma-separated list of modules to keep fp16 when reloading with bnb (input_embeddings,output_embeddings)'
    )
    parser.add_argument(
        '--use_bnb_quantization',
        action='store_true',
        help='After learning rotations, reload the model with bitsandbytes quantized modules (recommended for VRAM)'
    )

    parser.add_argument(
        '--skip_rotations',
        action='store_true',
        help='Skip rotation learning and directly quantize with bitsandbytes (fast path)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(log_dir='logs', log_file='spinquant.log', logger_name='spinquant')
    logger.info("Starting SpinQuant quantization")
    
    # Build configuration
    config = {
        'residual_model_path': args.residual_model_path,
        'output_dir': args.output_dir,
        'seed': 42,
        'spinquant': {
            'bits': args.bits,
            'double_quant': args.double_quant,
            'quant_type': args.quant_type,
            'group_size': 128,
            'symmetric': False,
            'backend': args.backend,
            'block_size': args.block_size,
            'num_steps': args.num_steps,
            'lr': args.lr,
            'num_sweeps': args.num_sweeps,
            'max_layers': args.max_layers,
            'use_bnb_quantization': bool(args.use_bnb_quantization),
            'use_activation_objective': not bool(args.no_activation_objective),
            'calibration_vectors_per_layer': int(args.calibration_vectors_per_layer),
            'keep_fp16_modules': [s.strip() for s in args.keep_fp16_modules.split(',') if s.strip()],
            'skip_rotations': bool(args.skip_rotations),
        }
    }
    
    # Create quantizer and run
    quantizer = SpinQuantizer(config, logger)
    result = quantizer.quantize_and_save()
    
    logger.info(f"Quantization complete. Quantized model saved to: {result['quantized_model_path']}")


if __name__ == '__main__':
    main()
