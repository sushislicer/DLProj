"""
Inference script for the quantized Qwen2.5 model with trained adapters.
"""

import os
import argparse
import torch
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.flash_attention import patch_broken_flash_attn

# Patch around broken/incompatible flash-attn wheels (we don't require FA2 here).
patch_broken_flash_attn(logger=None)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils.helpers import get_device


def load_model(quantized_model_path: str, adapter_path: str, device: str = None):
    """
    Load the quantized model with trained adapters.
    
    Args:
        quantized_model_path: Path to quantized base model
        adapter_path: Path to trained adapters
        device: Device to load model on (auto-detect if None)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # NOTE: we keep `device_map="auto"` below (recommended for large models).
    # `device` is retained for CLI compatibility and future extensions.
    if device is None:
        device = str(get_device())
    
    print(f"Loading quantized base model from {quantized_model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        quantized_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    
    print(f"Loading trained adapters from {adapter_path}...")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=False
    )

    model.eval()
    
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        quantized_model_path,
        trust_remote_code=True
    )
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded successfully!")
    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True
):
    """
    Generate text using the model.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        prompt: Input prompt
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        do_sample: Whether to use sampling
    
    Returns:
        Generated text
    """
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text


def interactive_mode(model, tokenizer):
    """
    Run interactive inference mode.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
    """
    print("\n" + "=" * 60)
    print("Interactive Inference Mode")
    print("=" * 60)
    print("Type your prompt and press Enter to generate.")
    print("Type 'quit' or 'exit' to exit.")
    print("=" * 60 + "\n")
    
    while True:
        try:
            prompt = input("Prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break
            
            if not prompt:
                continue
            
            print("\nGenerating...")
            generated = generate_text(
                model,
                tokenizer,
                prompt,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            print("\nGenerated:")
            print(generated)
            print("\n" + "-" * 60 + "\n")
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def batch_inference(model, tokenizer, prompts: list, output_file: str = None):
    """
    Run batch inference on multiple prompts.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        prompts: List of prompts
        output_file: Optional file to save results
    """
    print(f"\nRunning batch inference on {len(prompts)} prompts...")
    
    results = []
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Processing: {prompt[:50]}...")
        
        generated = generate_text(
            model,
            tokenizer,
            prompt,
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        results.append({
            'prompt': prompt,
            'generated': generated
        })
        
        print(f"Generated: {generated[:100]}...")
    
    # Save results if output file specified
    if output_file:
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_file}")
    
    return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Inference with quantized Qwen2.5 model"
    )
    parser.add_argument(
        '--quantized_model_path',
        type=str,
        default='outputs/quantized_model',
        help='Path to quantized base model'
    )
    parser.add_argument(
        '--adapter_path',
        type=str,
        default='outputs/trained_adapters',
        help='Path to trained adapters'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['interactive', 'single', 'batch'],
        default='interactive',
        help='Inference mode'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        help='Prompt for single mode'
    )
    parser.add_argument(
        '--prompts_file',
        type=str,
        help='File containing prompts for batch mode (one per line)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help='Output file for batch results'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='Maximum generation length'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='Nucleus sampling parameter'
    )
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(
        args.quantized_model_path,
        args.adapter_path
    )
    
    # Run inference based on mode
    if args.mode == 'interactive':
        interactive_mode(model, tokenizer)
    
    elif args.mode == 'single':
        if not args.prompt:
            print("Error: --prompt is required for single mode")
            return
        
        print(f"\nPrompt: {args.prompt}")
        print("\nGenerating...")
        
        generated = generate_text(
            model,
            tokenizer,
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True
        )
        
        print("\nGenerated:")
        print(generated)
    
    elif args.mode == 'batch':
        if not args.prompts_file:
            print("Error: --prompts_file is required for batch mode")
            return
        
        # Load prompts from file
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        batch_inference(model, tokenizer, prompts, args.output_file)


if __name__ == '__main__':
    main()
