"""
PiSSA (Principal Singular values and Singular Vectors Adaptation) adapter extraction script.
Extracts adapters and residuals from the Qwen2.5 model.
"""

import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import (
    setup_logging, get_device, print_model_size,
    ensure_dir, set_seed, count_parameters
)
from utils.memory_tracker import MemoryTracker


class PiSSAExtractor:
    """Extract adapters using PiSSA method."""
    
    def __init__(self, config: dict, logger):
        """
        Initialize PiSSA extractor.
        
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
            log_file='pissa_memory.log'
        )
        
        # Set random seed for reproducibility
        set_seed(config.get('seed', 42))
        
        # Load model and tokenizer
        self.logger.info(f"Loading model from {config['model_name']}")
        self.memory_tracker.log_memory("PiSSA", "Loading model...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config['model_name'],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['model_name'],
            trust_remote_code=True
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.logger.info(f"Model loaded successfully")
        self.memory_tracker.log_memory("PiSSA", "Model loaded")
        print_model_size(self.model, "Original Model")
    
    def apply_pissa(self):
        """
        Apply PiSSA to extract adapters and residuals.
        
        This method:
        1. Applies LoRA with PiSSA initialization
        2. Extracts the residual base model
        3. Saves both adapters and residuals
        """
        pissa_config = self.config['pissa']
        
        self.memory_tracker.log_memory("PiSSA", "Applying PiSSA...")
        
        # Configure LoRA with PiSSA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=pissa_config['rank'],
            lora_alpha=pissa_config['alpha'],
            lora_dropout=pissa_config['dropout'],
            target_modules=pissa_config['target_modules'],
            bias=pissa_config.get('bias', 'none'),
            init_lora_weights="pissa",  # PiSSA initialization
        )
        
        self.logger.info("Applying PiSSA to extract adapters...")
        self.model = get_peft_model(self.model, lora_config)
        
        self.memory_tracker.log_memory("PiSSA", "PiSSA applied")
        
        # Print model statistics after PiSSA
        print_model_size(self.model, "Model with PiSSA Adapters")
        
        # Enable gradient checkpointing for memory efficiency
        if self.config.get('gradient_checkpointing', False):
            self.model.gradient_checkpointing_enable()
            self.logger.info("Gradient checkpointing enabled")
        
        return self.model
    
    def save_adapters(self, output_dir: str):
        """
        Save the extracted adapters.
        
        Args:
            output_dir: Directory to save adapters
        """
        ensure_dir(output_dir)
        
        self.logger.info(f"Saving adapters to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        self.logger.info("Adapters saved successfully")
    
    def save_residual_model(self, output_dir: str):
        """
        Save the residual base model (model without adapters).
        
        Args:
            output_dir: Directory to save residual model
        """
        ensure_dir(output_dir)
        
        self.logger.info(f"Saving residual model to {output_dir}")
        
        # Get the base model without adapters
        base_model = self.model.base_model.model
        
        # Save the base model
        base_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        self.logger.info("Residual model saved successfully")
    
    def extract_and_save(self):
        """
        Complete pipeline: extract adapters and save both components.
        """
        # Start memory tracking
        self.memory_tracker.start_tracking()
        
        # Apply PiSSA
        self.apply_pissa()
        
        # Save adapters
        adapter_output_dir = os.path.join(
            self.config['output_dir'],
            'adapters'
        )
        self.memory_tracker.log_memory("PiSSA", "Saving adapters...")
        self.save_adapters(adapter_output_dir)
        
        # Save residual model
        residual_output_dir = os.path.join(
            self.config['output_dir'],
            'residual_model'
        )
        self.memory_tracker.log_memory("PiSSA", "Saving residual model...")
        self.save_residual_model(residual_output_dir)
        
        self.logger.info("PiSSA extraction completed successfully")
        
        # Stop memory tracking
        self.memory_tracker.stop_tracking()
        self.memory_tracker.print_summary()
        
        return {
            'adapter_path': adapter_output_dir,
            'residual_model_path': residual_output_dir
        }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Extract adapters using PiSSA")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/pissa_config.yaml',
        help='Path to PiSSA configuration file'
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
        default='outputs/adapters',
        help='Output directory for adapters and residuals'
    )
    parser.add_argument(
        '--rank',
        type=int,
        default=64,
        help='LoRA rank for PiSSA'
    )
    parser.add_argument(
        '--alpha',
        type=int,
        default=128,
        help='LoRA alpha for PiSSA'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(log_dir='logs', log_file='pissa.log', logger_name='pissa')
    logger.info("Starting PiSSA adapter extraction")
    
    # Build configuration
    config = {
        'model_name': args.model_name,
        'output_dir': args.output_dir,
        'seed': 42,
        'gradient_checkpointing': True,
        'pissa': {
            'rank': args.rank,
            'alpha': args.alpha,
            'dropout': 0.05,
            'target_modules': [
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj'
            ],
            'bias': 'none'
        }
    }
    
    # Create extractor and run
    extractor = PiSSAExtractor(config, logger)
    result = extractor.extract_and_save()
    
    logger.info(f"Extraction complete. Adapters saved to: {result['adapter_path']}")
    logger.info(f"Residual model saved to: {result['residual_model_path']}")


if __name__ == '__main__':
    main()
