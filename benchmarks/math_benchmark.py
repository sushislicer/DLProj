"""
MATH benchmark implementation for mathematical problem solving.
"""

import os
import json
import logging
from typing import Dict, List, Any
import re
from .base_benchmark import MathBenchmark


class MATHBenchmark(MathBenchmark):
    """MATH benchmark for mathematical problem solving."""
    
    def __init__(self, config: Dict, logger: logging.Logger):
        """
        Initialize MATH benchmark.
        
        Args:
            config: Benchmark configuration
            logger: Logger instance
        """
        super().__init__(config, logger)
        self.logger.info("MATH Benchmark initialized")
    
    def _load_dataset(self) -> List[Dict]:
        """
        Load MATH dataset.
        
        Returns:
            List of MATH problems
        """
        dataset_path = self.config.get('dataset_path', 'datasets/math')
        
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
        self.logger.info(f"MATH dataset not found at {dataset_path}, attempting download...")
        data = download_math_dataset(dataset_path)
        if data:
            return data

        if not bool(self.config.get('allow_sample_data', True)):
            raise FileNotFoundError(
                f"MATH dataset not found/downloadable at '{dataset_path}'. "
                "Sample fallback is disabled (allow_sample_data=false). "
                "Download datasets first (e.g., `python3 scripts/test_download.py`) or set a valid benchmarks.math.dataset_path."
            )

        self.logger.warning("Download failed, using sample data")
        return self._create_sample_dataset()
    
    def _create_sample_dataset(self) -> List[Dict]:
        """Create sample MATH dataset for testing."""
        return [
            {
                'id': 1,
                'problem': 'Find the value of x that satisfies the equation 2x + 3 = 11.',
                'answer': '4',
                'subject': 'Algebra',
                'level': 'Level 1',
                'solution': '2x + 3 = 11 => 2x = 8 => x = 4'
            },
            {
                'id': 2,
                'problem': 'Simplify the expression (x + 3)(x - 3).',
                'answer': 'x^2 - 9',
                'subject': 'Algebra',
                'level': 'Level 1',
                'solution': '(x + 3)(x - 3) = x^2 - 9'
            },
            {
                'id': 3,
                'problem': 'Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1.',
                'answer': '3x^2 + 4x - 5',
                'subject': 'Calculus',
                'level': 'Level 2',
                'solution': 'f\'(x) = 3x^2 + 4x - 5'
            },
            {
                'id': 4,
                'problem': 'Find the area of a triangle with base 10 and height 6.',
                'answer': '30',
                'subject': 'Geometry',
                'level': 'Level 1',
                'solution': 'Area = (1/2) * base * height = (1/2) * 10 * 6 = 30'
            },
            {
                'id': 5,
                'problem': 'Solve the equation x^2 - 5x + 6 = 0.',
                'answer': '2, 3',
                'subject': 'Algebra',
                'level': 'Level 1',
                'solution': 'x^2 - 5x + 6 = (x - 2)(x - 3) = 0 => x = 2 or x = 3'
            },
            {
                'id': 6,
                'problem': 'Find the integral of f(x) = 2x + 3.',
                'answer': 'x^2 + 3x + C',
                'subject': 'Calculus',
                'level': 'Level 2',
                'solution': '∫(2x + 3)dx = x^2 + 3x + C'
            },
            {
                'id': 7,
                'problem': 'Find the value of sin(30°).',
                'answer': '1/2',
                'subject': 'Trigonometry',
                'level': 'Level 1',
                'solution': 'sin(30°) = 1/2'
            },
            {
                'id': 8,
                'problem': 'Find the limit of (x^2 - 1)/(x - 1) as x approaches 1.',
                'answer': '2',
                'subject': 'Calculus',
                'level': 'Level 2',
                'solution': 'lim(x→1) (x^2 - 1)/(x - 1) = lim(x→1) (x + 1) = 2'
            },
            {
                'id': 9,
                'problem': 'Find the probability of rolling a 6 on a fair six-sided die.',
                'answer': '1/6',
                'subject': 'Probability',
                'level': 'Level 1',
                'solution': 'P(6) = 1/6'
            },
            {
                'id': 10,
                'problem': 'Find the sum of the first 10 natural numbers.',
                'answer': '55',
                'subject': 'Number Theory',
                'level': 'Level 1',
                'solution': 'Sum = n(n+1)/2 = 10*11/2 = 55'
            }
        ]
    
    def _format_prompt(self, sample: Dict) -> str:
        """
        Format MATH problem into prompt.
        
        Args:
            sample: MATH problem sample
        
        Returns:
            Formatted prompt
        """
        problem = sample['problem']
        subject = sample.get('subject', 'Mathematics')
        level = sample.get('level', 'Unknown')
        
        prompt = f"""Solve the following {subject} problem ({level}). Show your work step by step and provide your final answer.

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
        answer_match = re.search(r'(?:Answer|answer|The answer is|Therefore, the answer is|Final answer)[:\s]*([^\n]+)', response, re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).strip()
            # Remove trailing punctuation
            answer = re.sub(r'[.,;!?]+$', '', answer)
            return answer
        
        # Pattern 2: Look for "Therefore" followed by answer
        therefore_match = re.search(r'Therefore,?\s+([^\n]+)', response, re.IGNORECASE)
        if therefore_match:
            answer = therefore_match.group(1).strip()
            # Remove trailing punctuation
            answer = re.sub(r'[.,;!?]+$', '', answer)
            return answer
        
        # Pattern 3: Look for last equation or expression
        equations = re.findall(r'=([^\n]+)', response)
        if equations:
            answer = equations[-1].strip()
            # Remove trailing punctuation
            answer = re.sub(r'[.,;!?]+$', '', answer)
            return answer
        
        # Pattern 4: If no clear answer found, return the last line
        lines = response.strip().split('\n')
        if lines:
            answer = lines[-1].strip()
            # Remove trailing punctuation
            answer = re.sub(r'[.,;!?]+$', '', answer)
            return answer
        
        # If nothing found, return None
        return None
    
    def _evaluate_sample(self, prediction: Any, ground_truth: Any) -> bool:
        """
        Evaluate MATH answer.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
        
        Returns:
            True if correct, False otherwise
        """
        if prediction is None or ground_truth is None:
            return False
        
        # Normalize both answers
        pred_str = str(prediction).strip().lower()
        truth_str = str(ground_truth).strip().lower()
        
        # Remove common variations
        pred_str = re.sub(r'\s+', ' ', pred_str)
        truth_str = re.sub(r'\s+', ' ', truth_str)
        
        # Remove trailing punctuation
        pred_str = re.sub(r'[.,;!?]+$', '', pred_str)
        truth_str = re.sub(r'[.,;!?]+$', '', truth_str)
        
        # Check for exact match
        if pred_str == truth_str:
            return True
        
        # Check for numeric match
        try:
            pred_num = float(pred_str)
            truth_num = float(truth_str)
            if abs(pred_num - truth_num) < 1e-6:
                return True
        except (ValueError, TypeError):
            pass
        
        # Check for multiple answers (comma-separated)
        pred_answers = [a.strip() for a in pred_str.split(',')]
        truth_answers = [a.strip() for a in truth_str.split(',')]
        
        # Sort and compare
        pred_answers_sorted = sorted(pred_answers)
        truth_answers_sorted = sorted(truth_answers)
        
        if pred_answers_sorted == truth_answers_sorted:
            return True
        
        # Check if all predicted answers are in truth answers
        if all(p in truth_answers for p in pred_answers):
            return True
        
        return False


def load_math_dataset(dataset_path: str = 'datasets/math') -> List[Dict]:
    """
    Load MATH dataset from file or directory.
    
    Args:
        dataset_path: Path to MATH dataset
    
    Returns:
        List of MATH problems
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


def download_math_dataset(output_path: str = 'datasets/math'):
    """
    Download MATH dataset from Hugging Face.
    
    Args:
        output_path: Path to save dataset
    """
    try:
        from datasets import load_dataset
        
        # Load MATH dataset from Hugging Face
        dataset = load_dataset("EleutherAI/math", split="test")
        
        # Convert to list of dictionaries
        data = []
        for item in dataset:
            data.append({
                'id': item.get('id', len(data)),
                'problem': item['problem'],
                'answer': item['answer'],
                'subject': item.get('subject', 'Unknown'),
                'level': item.get('level', 'Unknown'),
                'solution': item.get('solution', '')
            })
        
        # Save to file
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, 'math.json'), 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"MATH dataset downloaded and saved to {output_path}")
        return data
    
    except Exception as e:
        print(f"Error downloading MATH dataset: {e}")
        return []
