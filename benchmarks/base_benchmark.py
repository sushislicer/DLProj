"""
Base benchmark class for all benchmark implementations.
Provides common functionality for benchmark evaluation.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from tqdm import tqdm
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class _StopOnTokenSequences(torch.nn.Module):
    """Token-level early stopping for `transformers.generate()`.

    We implement this as a small callable to avoid hard-depending on a specific
    transformers version's stop-string support.

    The check is purely token based: stop when the *suffix* of any sequence in
    the batch matches one of the configured token-id sequences.
    """

    def __init__(self, stop_sequences: List[List[int]]):
        super().__init__()
        # Filter empty sequences
        self.stop_sequences = [seq for seq in stop_sequences if seq]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:  # type: ignore[override]
        if not self.stop_sequences:
            return False

        # input_ids: [batch, seq]
        for seq in self.stop_sequences:
            if input_ids.shape[1] < len(seq):
                continue
            tail = input_ids[:, -len(seq):]
            target = torch.tensor(seq, device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0)
            # Stop if any row matches.
            if torch.any(torch.all(tail == target, dim=1)).item():
                return True
        return False


class BaseBenchmark(ABC):
    """Base class for all benchmarks."""
    
    def __init__(self, config: Dict, logger: logging.Logger):
        """
        Initialize benchmark.
        
        Args:
            config: Benchmark configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.dataset = self._load_dataset()
        self.results = {
            'predictions': [],
            'metrics': {},
            'metadata': {
                'dataset_size': len(self.dataset),
                'num_samples': min(len(self.dataset), config.get('num_samples', len(self.dataset))),
                'config': config
            }
        }
    
    @abstractmethod
    def _load_dataset(self) -> List[Dict]:
        """Load benchmark dataset."""
        pass
    
    @abstractmethod
    def _format_prompt(self, sample: Dict) -> str:
        """Format sample into prompt for the model."""
        pass
    
    @abstractmethod
    def _extract_answer(self, response: str, sample: Dict) -> Any:
        """Extract answer from model response."""
        pass
    
    @abstractmethod
    def _evaluate_sample(self, prediction: Any, ground_truth: Any) -> bool:
        """Evaluate if prediction is correct."""
        pass
    
    def run(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        model_config: Dict
    ) -> Dict:
        """
        Run benchmark on model.
        
        Args:
            model: Loaded model
            tokenizer: Loaded tokenizer
            model_config: Model configuration
        
        Returns:
            Dictionary with benchmark results
        """
        self.logger.info(f"Running benchmark with {len(self.dataset)} samples")
        
        # Determine number of samples to run
        num_samples = min(
            len(self.dataset),
            self.config.get('num_samples', len(self.dataset))
        )
        
        # Use model-specific batch size
        batch_size = model_config.get('batch_size', 1)
        max_batch_size = model_config.get('max_batch_size', batch_size)
        min_batch_size = model_config.get('min_batch_size', 1)
        
        # Check if dynamic batching is enabled
        use_dynamic_batching = model_config.get('use_dynamic_batching', False)
        
        # Process samples
        correct = 0
        total = 0
        latencies = []
        
        samples_to_process = self.dataset[:num_samples]
        
        if use_dynamic_batching:
            # Dynamic batching: group samples by similar lengths
            batches = self._create_dynamic_batches(
                samples_to_process,
                min_batch_size,
                max_batch_size
            )
            self.logger.info(f"Using dynamic batching: {len(batches)} batches created")
        else:
            # Fixed batch size
            batches = [samples_to_process[i:i+batch_size]
                      for i in range(0, len(samples_to_process), batch_size)]
        
        for batch in tqdm(batches, desc="Processing"):
            # Format prompts
            prompts = [self._format_prompt(sample) for sample in batch]
            
            # Generate responses
            start_time = time.time()
            responses = self._generate_batch(model, tokenizer, prompts)
            batch_latency = time.time() - start_time
            latencies.append(batch_latency)
            
            # Evaluate each sample
            for idx_in_batch, (sample, response) in enumerate(zip(batch, responses)):
                prediction = self._extract_answer(response, sample)
                # Some benchmarks (e.g., code execution / patch generation) store
                # evaluation targets on the sample itself (test cases, commands).
                ground_truth = sample.get('answer', sample.get('solution', sample))
                
                is_correct = self._evaluate_sample(prediction, ground_truth)
                
                if is_correct:
                    correct += 1
                total += 1
                
                # Store prediction
                self.results['predictions'].append({
                    'sample_id': sample.get('id', total),
                    'prompt': prompts[idx_in_batch],
                    'response': response,
                    'prediction': prediction,
                    'ground_truth': ground_truth,
                    'correct': is_correct,
                    'latency': batch_latency / len(batch)
                })
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0.0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        
        self.results['metrics'] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'avg_latency': avg_latency,
            'total_latency': sum(latencies),
            'throughput': total / sum(latencies) if sum(latencies) > 0 else 0.0
        }
        
        # Calculate pass@k if needed
        if 'pass_at_1' in self.config.get('metrics', []):
            self.results['metrics']['pass_at_1'] = accuracy
        
        if 'pass_at_5' in self.config.get('metrics', []):
            self.results['metrics']['pass_at_5'] = self._calculate_pass_at_k(5)
        
        self.logger.info(f"Benchmark completed: {correct}/{total} correct ({accuracy:.2%})")
        
        return self.results
    
    def _create_dynamic_batches(
        self,
        samples: List[Dict],
        min_batch_size: int,
        max_batch_size: int
    ) -> List[List[Dict]]:
        """
        Create dynamic batches based on input lengths for better GPU utilization.
        
        Args:
            samples: List of samples to batch
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
        
        Returns:
            List of batches
        """
        # Calculate prompt lengths for each sample
        samples_with_lengths = []
        for sample in samples:
            prompt = self._format_prompt(sample)
            length = len(prompt)
            samples_with_lengths.append((sample, length))
        
        # Sort by length (descending) for better packing
        samples_with_lengths.sort(key=lambda x: x[1], reverse=True)
        
        # Create batches with similar lengths
        batches = []
        current_batch = []
        current_length = 0
        
        for sample, length in samples_with_lengths:
            # Check if adding this sample would exceed reasonable batch size
            if len(current_batch) >= max_batch_size:
                batches.append(current_batch)
                current_batch = []
                current_length = 0
            elif len(current_batch) >= min_batch_size and current_length > 0:
                # Check if this sample is too different in length
                avg_length = current_length / len(current_batch)
                if abs(length - avg_length) > avg_length * 0.5:  # More than 50% difference
                    batches.append(current_batch)
                    current_batch = []
                    current_length = 0
            
            current_batch.append(sample)
            current_length += length
        
        # Add the last batch if not empty
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _generate_batch(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prompts: List[str]
    ) -> List[str]:
        """
        Generate responses for a batch of prompts.
        
        Args:
            model: Loaded model
            tokenizer: Loaded tokenizer
            prompts: List of prompts
        
        Returns:
            List of generated responses
        """
        # Tokenize inputs
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.get('max_length', 2048)
        )

        # For sharded / device-mapped models, `model.device` can be misleading.
        # Put inputs on the device of the first parameter (typically the embed
        # layer). This matches common HF generation expectations.
        try:
            first_param_device = next(model.parameters()).device
        except Exception:
            first_param_device = getattr(model, 'device', torch.device('cpu'))

        inputs = {k: v.to(first_param_device) for k, v in inputs.items()}

        # Per-sample input lengths (avoid incorrectly stripping prompts when
        # samples are padded to a uniform length).
        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        
        # Get generation parameters
        gen_kwargs = {
            'max_new_tokens': self.config.get('max_new_tokens', 512),
            'temperature': self.config.get('temperature', 0.7),
            'top_p': self.config.get('top_p', 0.9),
            'top_k': self.config.get('top_k', 50),
            'do_sample': self.config.get('do_sample', True),
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'use_cache': True
        }
        
        # Add beam search parameters if needed
        if self.config.get('num_beams', 1) > 1:
            gen_kwargs['num_beams'] = self.config['num_beams']
            gen_kwargs['early_stopping'] = self.config.get('early_stopping', True)
        
        # Add token-level early stopping sequences if configured.
        # NOTE: We don't rely on `stop_strings` because it is tokenizer/version
        # dependent and not available in many `transformers` releases.
        early_stopping_tokens = self.config.get('early_stopping_tokens', None)
        if early_stopping_tokens:
            try:
                from transformers import StoppingCriteriaList

                stop_sequences: List[List[int]] = []
                for s in early_stopping_tokens:
                    tok = tokenizer(s, add_special_tokens=False, return_tensors=None)
                    ids = tok.get('input_ids', [])
                    # HF returns either list[int] or list[list[int]] depending on tokenizer call.
                    if isinstance(ids, list) and ids and isinstance(ids[0], list):
                        ids = ids[0]
                    if isinstance(ids, list):
                        stop_sequences.append([int(x) for x in ids])

                if stop_sequences:
                    gen_kwargs['stopping_criteria'] = StoppingCriteriaList([
                        _StopOnTokenSequences(stop_sequences)
                    ])
            except Exception as e:
                # Never fail the run due to early-stopping wiring.
                self.logger.warning(f"Failed to set early stopping criteria: {e}")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        
        # Decode responses
        responses = []
        for i, output in enumerate(outputs):
            # Remove input tokens from output using per-sample length.
            response_tokens = output[int(input_lengths[i]):]
            response = tokenizer.decode(response_tokens, skip_special_tokens=True)
            responses.append(response)
        
        return responses
    
    def _calculate_pass_at_k(self, k: int) -> float:
        """
        Calculate pass@k metric.
        
        Args:
            k: Number of attempts
        
        Returns:
            Pass@k score
        """
        # This is a simplified implementation
        # For actual pass@k, you would need multiple generations per sample
        predictions = self.results['predictions']
        correct = sum(1 for p in predictions if p['correct'])
        total = len(predictions)
        return correct / total if total > 0 else 0.0
    
    def save_predictions(self, output_path: str):
        """Save predictions to file."""
        with open(output_path, 'w') as f:
            json.dump(self.results['predictions'], f, indent=2)
        self.logger.info(f"Predictions saved to: {output_path}")
    
    def save_metrics(self, output_path: str):
        """Save metrics to file."""
        with open(output_path, 'w') as f:
            json.dump(self.results['metrics'], f, indent=2)
        self.logger.info(f"Metrics saved to: {output_path}")


class ExactMatchBenchmark(BaseBenchmark):
    """Benchmark that uses exact match evaluation."""
    
    def _evaluate_sample(self, prediction: Any, ground_truth: Any) -> bool:
        """Evaluate using exact match."""
        if prediction is None or ground_truth is None:
            return False
        
        # Convert to strings for comparison
        pred_str = str(prediction).strip().lower()
        truth_str = str(ground_truth).strip().lower()
        
        return pred_str == truth_str


class MultipleChoiceBenchmark(BaseBenchmark):
    """Benchmark for multiple choice questions."""
    
    def _evaluate_sample(self, prediction: Any, ground_truth: Any) -> bool:
        """Evaluate multiple choice answer."""
        if prediction is None or ground_truth is None:
            return False
        
        # Normalize to single letter (A, B, C, D)
        pred_str = str(prediction).strip().upper()
        truth_str = str(ground_truth).strip().upper()
        
        # Extract first letter if longer
        if len(pred_str) > 0:
            pred_str = pred_str[0]
        if len(truth_str) > 0:
            truth_str = truth_str[0]
        
        return pred_str == truth_str


class CodeExecutionBenchmark(BaseBenchmark):
    """Benchmark that requires code execution for evaluation."""
    
    def _evaluate_sample(self, prediction: Any, ground_truth: Any) -> bool:
        """Evaluate by executing code and comparing outputs."""
        # This would require a sandboxed code execution environment
        # For now, return False as placeholder
        self.logger.warning("Code execution evaluation not implemented")
        return False


class MathBenchmark(BaseBenchmark):
    """Benchmark for mathematical problems."""
    
    def _evaluate_sample(self, prediction: Any, ground_truth: Any) -> bool:
        """Evaluate mathematical answer."""
        if prediction is None or ground_truth is None:
            return False
        
        try:
            # Try to parse as numbers
            pred_num = float(str(prediction).strip())
            truth_num = float(str(ground_truth).strip())
            
            # Check if close (with tolerance for floating point)
            return abs(pred_num - truth_num) < 1e-6
        except (ValueError, TypeError):
            # Fall back to string comparison
            pred_str = str(prediction).strip().lower()
            truth_str = str(ground_truth).strip().lower()
            return pred_str == truth_str
