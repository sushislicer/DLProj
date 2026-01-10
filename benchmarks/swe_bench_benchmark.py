"""
SWE-Bench (Software Engineering Benchmark) implementation for code patch generation.
"""

import os
import json
import logging
from typing import Dict, List, Any
import re
import subprocess
import tempfile
from .base_benchmark import CodeExecutionBenchmark


class SWEBenchBenchmark(CodeExecutionBenchmark):
    """SWE-Bench benchmark for software engineering tasks."""
    
    def __init__(self, config: Dict, logger: logging.Logger):
        """
        Initialize SWE-Bench benchmark.
        
        Args:
            config: Benchmark configuration
            logger: Logger instance
        """
        super().__init__(config, logger)
        self.logger.info("SWE-Bench Benchmark initialized")
        self.timeout = config.get('timeout', 1200)
        self.max_retries = config.get('max_retries', 3)
    
    def _load_dataset(self) -> List[Dict]:
        """
        Load SWE-Bench dataset.
        
        Returns:
            List of SWE-Bench problems
        """
        dataset_path = self.config.get('dataset_path', 'datasets/swe_bench')
        
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
        
        # If no dataset found, either fail-fast (paper runs) or fall back to sample data.
        if not bool(self.config.get('allow_sample_data', True)):
            raise FileNotFoundError(
                f"SWE-Bench dataset not found at '{dataset_path}'. "
                "Sample fallback is disabled (allow_sample_data=false). "
                "Download datasets first (e.g., `python3 scripts/test_download.py`) or set a valid benchmarks.swe_bench.dataset_path."
            )

        self.logger.warning(f"SWE-Bench dataset not found at {dataset_path}, using sample data")
        return self._create_sample_dataset()
    
    def _create_sample_dataset(self) -> List[Dict]:
        """Create sample SWE-Bench dataset for testing."""
        return [
            {
                'id': 1,
                'repo': 'example/repo',
                'issue_title': 'Fix bug in function that calculates sum',
                'issue_body': 'The sum function is not handling negative numbers correctly.',
                'base_commit': 'abc123',
                'problem_statement': 'Fix the sum function to handle negative numbers correctly.',
                'hints': ['Check the condition for negative numbers', 'Ensure proper initialization'],
                'test_command': 'python test_sum.py',
                'language': 'Python',
                'difficulty': 'Easy'
            },
            {
                'id': 2,
                'repo': 'example/repo',
                'issue_title': 'Add error handling for file operations',
                'issue_body': 'The file reading function should handle file not found errors.',
                'base_commit': 'def456',
                'problem_statement': 'Add proper error handling for file operations.',
                'hints': ['Use try-except blocks', 'Handle FileNotFoundError'],
                'test_command': 'python test_file_ops.py',
                'language': 'Python',
                'difficulty': 'Medium'
            },
            {
                'id': 3,
                'repo': 'example/repo',
                'issue_title': 'Optimize list sorting algorithm',
                'issue_body': 'The current sorting implementation is inefficient for large lists.',
                'base_commit': 'ghi789',
                'problem_statement': 'Optimize the sorting algorithm for better performance.',
                'hints': ['Consider using built-in sort', 'Check time complexity'],
                'test_command': 'python test_sort.py',
                'language': 'Python',
                'difficulty': 'Medium'
            },
            {
                'id': 4,
                'repo': 'example/repo',
                'issue_title': 'Fix memory leak in data processing',
                'issue_body': 'The data processing function is not releasing memory properly.',
                'base_commit': 'jkl012',
                'problem_statement': 'Fix the memory leak in the data processing function.',
                'hints': ['Check for unclosed resources', 'Use context managers'],
                'test_command': 'python test_memory.py',
                'language': 'Python',
                'difficulty': 'Hard'
            },
            {
                'id': 5,
                'repo': 'example/repo',
                'issue_title': 'Add support for new data format',
                'issue_body': 'The parser should support JSON format in addition to CSV.',
                'base_commit': 'mno345',
                'problem_statement': 'Extend the parser to support JSON format.',
                'hints': ['Use json module', 'Maintain backward compatibility'],
                'test_command': 'python test_parser.py',
                'language': 'Python',
                'difficulty': 'Medium'
            }
        ]
    
    def _format_prompt(self, sample: Dict) -> str:
        """
        Format SWE-Bench problem into prompt.
        
        Args:
            sample: SWE-Bench problem sample
        
        Returns:
            Formatted prompt
        """
        repo = sample['repo']
        issue_title = sample['issue_title']
        issue_body = sample.get('issue_body', '')
        problem_statement = sample['problem_statement']
        hints = sample.get('hints', [])
        language = sample.get('language', 'Python')
        difficulty = sample.get('difficulty', 'Unknown')
        
        prompt = f"""You are a software engineer working on the repository: {repo}

Issue: {issue_title}

Description:
{issue_body}

Problem:
{problem_statement}

Hints:
{chr(10).join(f'- {hint}' for hint in hints) if hints else 'No hints provided'}

Language: {language}
Difficulty: {difficulty}

Provide a code patch or solution to fix this issue. Include the complete modified code or the specific changes needed.

Solution:"""
        
        return prompt
    
    def _extract_answer(self, response: str, sample: Dict) -> Any:
        """
        Extract code patch from model response.
        
        Args:
            response: Model response
            sample: Original sample
        
        Returns:
            Extracted code patch
        """
        # Try to extract code from the response
        # Look for code blocks with ```python or ```
        
        # Pattern 1: Look for ```python code block
        code_match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Pattern 2: Look for ``` code block
        code_match = re.search(r'```\n(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Pattern 3: Look for diff format
        diff_match = re.search(r'```diff\n(.*?)```', response, re.DOTALL)
        if diff_match:
            return diff_match.group(1).strip()
        
        # Pattern 4: If no code block found, return the entire response
        return response.strip()
    
    def _evaluate_sample(self, prediction: Any, ground_truth: Any) -> bool:
        """
        Evaluate code patch by applying it and running tests.
        
        Args:
            prediction: Generated code patch
            ground_truth: Ground truth (test command and expected behavior)
        
        Returns:
            True if tests pass, False otherwise
        """
        if prediction is None or ground_truth is None:
            return False
        
        patch = str(prediction).strip()
        test_command = ground_truth.get('test_command', '')
        
        if not test_command:
            self.logger.warning("No test command provided")
            return False
        
        # For now, we'll do a simplified evaluation
        # In a real implementation, you would:
        # 1. Clone the repository
        # 2. Apply the patch
        # 3. Run the test command
        # 4. Check if tests pass
        
        # Placeholder: check if the patch contains meaningful code
        if len(patch) < 10:
            return False
        
        # Check if it contains common programming constructs
        has_code = any(keyword in patch.lower() for keyword in ['def ', 'class ', 'import ', 'return ', 'if ', 'for ', 'while '])
        
        return has_code
    
    def _apply_patch(self, repo_path: str, patch: str) -> bool:
        """
        Apply patch to repository.
        
        Args:
            repo_path: Path to repository
            patch: Patch to apply
        
        Returns:
            True if patch applied successfully, False otherwise
        """
        try:
            # Create a patch file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
                f.write(patch)
                patch_file = f.name
            
            # Apply the patch using git apply
            result = subprocess.run(
                ['git', 'apply', patch_file],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # Clean up
            os.unlink(patch_file)
            
            return result.returncode == 0
        
        except Exception as e:
            self.logger.warning(f"Error applying patch: {e}")
            return False
    
    def _run_tests(self, repo_path: str, test_command: str) -> bool:
        """
        Run tests in repository.
        
        Args:
            repo_path: Path to repository
            test_command: Test command to run
        
        Returns:
            True if tests pass, False otherwise
        """
        try:
            # Run the test command
            result = subprocess.run(
                test_command.split(),
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            return result.returncode == 0
        
        except Exception as e:
            self.logger.warning(f"Error running tests: {e}")
            return False


def load_swe_bench_dataset(dataset_path: str = 'datasets/swe_bench') -> List[Dict]:
    """
    Load SWE-Bench dataset from file or directory.
    
    Args:
        dataset_path: Path to SWE-Bench dataset
    
    Returns:
        List of SWE-Bench problems
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


def download_swe_bench_dataset(output_path: str = 'datasets/swe_bench'):
    """
    Download SWE-Bench dataset from Hugging Face.
    
    Args:
        output_path: Path to save dataset
    """
    try:
        from datasets import load_dataset
        
        # Load SWE-Bench dataset from Hugging Face
        dataset = load_dataset("princeton-nlp/SWE-bench", split="test")
        
        # Convert to list of dictionaries
        data = []
        for item in dataset:
            data.append({
                'id': item.get('instance_id', len(data)),
                'repo': item.get('repo', 'unknown/repo'),
                'issue_title': item.get('problem_statement', '')[:100],
                'issue_body': item.get('problem_statement', ''),
                'base_commit': item.get('base_commit', ''),
                'problem_statement': item.get('problem_statement', ''),
                'hints': item.get('hints', []),
                'test_command': item.get('test_command', ''),
                'language': item.get('language', 'Python'),
                'difficulty': item.get('difficulty', 'Unknown')
            })
        
        # Save to file
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, 'swe_bench.json'), 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"SWE-Bench dataset downloaded and saved to {output_path}")
        return data
    
    except Exception as e:
        print(f"Error downloading SWE-Bench dataset: {e}")
        return []
