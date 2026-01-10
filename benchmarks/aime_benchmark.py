"""
AIME (American Invitational Mathematics Examination) benchmark implementation.
"""

import os
import json
import logging
from typing import Dict, List, Any
import re
from .base_benchmark import MathBenchmark


class AIMEBenchmark(MathBenchmark):
    """AIME benchmark for mathematical problem solving."""
    
    def __init__(self, config: Dict, logger: logging.Logger):
        """
        Initialize AIME benchmark.
        
        Args:
            config: Benchmark configuration
            logger: Logger instance
        """
        super().__init__(config, logger)
        self.logger.info("AIME Benchmark initialized")
    
    def _load_dataset(self) -> List[Dict]:
        """
        Load AIME dataset.
        
        Returns:
            List of AIME problems
        """
        dataset_path = self.config.get('dataset_path', 'datasets/aime')
        
        # Try to load from file
        if os.path.exists(dataset_path):
            if os.path.isfile(dataset_path):
                with open(dataset_path, 'r') as f:
                    data = json.load(f)
                return data
            else:
                # Load from directory
                json_files = [f for f in os.listdir(dataset_path) if f.endswith('.json')]
                data = []
                for json_file in json_files:
                    with open(os.path.join(dataset_path, json_file), 'r') as f:
                        data.extend(json.load(f))
                return data
        
        # If no dataset found, try to download
        self.logger.info(f"AIME dataset not found at {dataset_path}, attempting download...")
        data = download_aime_dataset(dataset_path)
        if data:
            return data

        if not bool(self.config.get('allow_sample_data', True)):
            raise FileNotFoundError(
                f"AIME dataset not found/downloadable at '{dataset_path}'. "
                "Sample fallback is disabled (allow_sample_data=false). "
                "Download datasets first (e.g., `python3 scripts/test_download.py`) or set a valid benchmarks.aime.dataset_path."
            )

        self.logger.warning("Download failed, using sample data")
        return self._create_sample_dataset()
    
    def _create_sample_dataset(self) -> List[Dict]:
        """Create sample AIME dataset for testing."""
        return [
            {
                'id': 1,
                'problem': 'Find the number of positive integers n ≤ 1000 such that n is a multiple of 3 or 5.',
                'answer': '467',
                'year': 2023,
                'problem_number': 1
            },
            {
                'id': 2,
                'problem': 'Find the sum of all positive integers n such that n^2 + 5n + 6 is a perfect square.',
                'answer': '10',
                'year': 2023,
                'problem_number': 2
            },
            {
                'id': 3,
                'problem': 'Let ABC be a triangle with AB = 13, BC = 14, and CA = 15. Let D be the foot of the altitude from A to BC. Find AD.',
                'answer': '12',
                'year': 2023,
                'problem_number': 3
            },
            {
                'id': 4,
                'problem': 'Find the number of ordered pairs (a, b) of positive integers such that a + b = 100 and a and b are relatively prime.',
                'answer': '40',
                'year': 2023,
                'problem_number': 4
            },
            {
                'id': 5,
                'problem': 'Find the value of x such that log_2(x) + log_4(x) + log_8(x) = 11.',
                'answer': '64',
                'year': 2023,
                'problem_number': 5
            }
        ]
    
    def _format_prompt(self, sample: Dict) -> str:
        """
        Format AIME problem into prompt.
        
        Args:
            sample: AIME problem sample
        
        Returns:
            Formatted prompt
        """
        problem = sample['problem']
        
        prompt = f"""Solve the following AIME problem step by step. Provide your final answer as a single integer.

Problem: {problem}

Solution:"""
        
        return prompt
    
    def _extract_answer(self, response: str, sample: Dict) -> Any:
        """
        Extract answer from model response.
        
        Args:
            response: Model response
            sample: Original sample
        
        Returns:
            Extracted answer
        """
        # Try to find the final answer in the response
        # Look for patterns like "Answer:", "Therefore,", "The answer is", etc.
        
        # Pattern 1: Look for "Answer:" or "answer:"
        answer_match = re.search(r'(?:Answer|answer|The answer is|Therefore, the answer is)[:\s]*([0-9]+)', response, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1)
        
        # Pattern 2: Look for final number at the end
        numbers = re.findall(r'\b\d+\b', response)
        if numbers:
            return numbers[-1]
        
        # Pattern 3: If no clear answer found, return the last number in the response
        if numbers:
            return numbers[-1]
        
        # If no number found, return None
        return None
    
    def _evaluate_sample(self, prediction: Any, ground_truth: Any) -> bool:
        """
        Evaluate AIME answer.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
        
        Returns:
            True if correct, False otherwise
        """
        if prediction is None or ground_truth is None:
            return False
        
        try:
            # Convert to integers
            pred_int = int(str(prediction).strip())
            truth_int = int(str(ground_truth).strip())
            
            return pred_int == truth_int
        except (ValueError, TypeError):
            # Fall back to string comparison
            pred_str = str(prediction).strip().lower()
            truth_str = str(ground_truth).strip().lower()
            return pred_str == truth_str


def load_aime_dataset(dataset_path: str = 'datasets/aime') -> List[Dict]:
    """
    Load AIME dataset from file or directory.
    
    Args:
        dataset_path: Path to AIME dataset
    
    Returns:
        List of AIME problems
    """
    if os.path.exists(dataset_path):
        if os.path.isfile(dataset_path):
            with open(dataset_path, 'r') as f:
                return json.load(f)
        else:
            json_files = [f for f in os.listdir(dataset_path) if f.endswith('.json')]
            data = []
            for json_file in json_files:
                with open(os.path.join(dataset_path, json_file), 'r') as f:
                    data.extend(json.load(f))
            return data
    
    return []


def download_aime_dataset(output_path: str = 'datasets/aime'):
    """
    Download AIME dataset from Hugging Face.
    
    Args:
        output_path: Path to save dataset
    """
    try:
        from datasets import load_dataset

        # NOTE:
        # The previous repo id ("EleutherAI/aime") may not exist anymore.
        # Try a configurable dataset id first, then fall back to a small list
        # of known/commonly-used community mirrors.
        dataset_id = os.environ.get('BENCH_AIME_HF_DATASET', '').strip()
        split = os.environ.get('BENCH_AIME_HF_SPLIT', 'train')

        candidates = [
            dataset_id,
            # Community mirrors (availability can change).
            'Maxwell-Jia/AIME_2024',
            'lighteval/aime',
        ]
        candidates = [c for c in candidates if c]

        dataset = None
        last_err: Exception | None = None
        for cand in candidates:
            try:
                dataset = load_dataset(cand, split=split)
                dataset_id = cand
                break
            except Exception as e:
                last_err = e
                continue

        if dataset is None:
            raise RuntimeError(
                "Could not download an AIME dataset from HuggingFace. "
                "Set BENCH_AIME_HF_DATASET to a valid dataset repo id, or provide a local JSON under datasets/aime. "
                f"Last error: {last_err}"
            )
        
        # Convert to list of dictionaries
        data = []
        for item in dataset:
            data.append({
                'id': item.get('id', len(data)),
                'problem': item['problem'],
                'answer': item['answer'],
                'year': item.get('year', 2023),
                'problem_number': item.get('problem_number', 1)
            })
        
        # Save to file
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, 'aime.json'), 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"AIME dataset downloaded ({dataset_id}:{split}) and saved to {output_path}")
        return data
    
    except Exception as e:
        print(f"Error downloading AIME dataset: {e}")
        return []
