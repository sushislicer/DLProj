"""
LiveCodeBench benchmark implementation for code generation and execution.
"""

import os
import json
import logging
from typing import Dict, List, Any
import re
import subprocess
import tempfile
import multiprocessing as mp
import ast
from .base_benchmark import CodeExecutionBenchmark


class LiveCodeBenchBenchmark(CodeExecutionBenchmark):
    """LiveCodeBench benchmark for code generation and execution."""
    
    def __init__(self, config: Dict, logger: logging.Logger):
        """
        Initialize LiveCodeBench benchmark.
        
        Args:
            config: Benchmark configuration
            logger: Logger instance
        """
        super().__init__(config, logger)
        self.logger.info("LiveCodeBench Benchmark initialized")
        self.timeout = config.get('timeout', 600)
        self.num_test_cases = config.get('num_test_cases', 10)
    
    def _load_dataset(self) -> List[Dict]:
        """
        Load LiveCodeBench dataset.
        
        Returns:
            List of LiveCodeBench problems
        """
        dataset_path = self.config.get('dataset_path', 'datasets/livecodebench')
        
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
        self.logger.info(f"LiveCodeBench dataset not found at {dataset_path}, attempting download...")
        data = download_livecodebench_dataset(dataset_path)
        if data:
            return data

        if not bool(self.config.get('allow_sample_data', True)):
            raise FileNotFoundError(
                f"LiveCodeBench dataset not found/downloadable at '{dataset_path}'. "
                "Sample fallback is disabled (allow_sample_data=false). "
                "Download datasets first (e.g., `python3 scripts/test_download.py`) or set a valid benchmarks.livecodebench.dataset_path."
            )

        self.logger.warning("Download failed, using sample data")
        return self._create_sample_dataset()
    
    def _create_sample_dataset(self) -> List[Dict]:
        """Create sample LiveCodeBench dataset for testing."""
        return [
            {
                'id': 1,
                'problem': 'Write a function that takes a list of integers and returns the sum of all even numbers.',
                'function_name': 'sum_even_numbers',
                'test_cases': [
                    {'input': [1, 2, 3, 4, 5, 6], 'expected': 12},
                    {'input': [2, 4, 6, 8], 'expected': 20},
                    {'input': [1, 3, 5, 7], 'expected': 0},
                    {'input': [], 'expected': 0},
                    {'input': [0, 1, 2], 'expected': 2}
                ],
                'difficulty': 'Easy',
                'language': 'Python'
            },
            {
                'id': 2,
                'problem': 'Write a function that checks if a string is a palindrome.',
                'function_name': 'is_palindrome',
                'test_cases': [
                    {'input': 'racecar', 'expected': True},
                    {'input': 'hello', 'expected': False},
                    {'input': 'A man a plan a canal Panama', 'expected': True},
                    {'input': '', 'expected': True},
                    {'input': 'a', 'expected': True}
                ],
                'difficulty': 'Easy',
                'language': 'Python'
            },
            {
                'id': 3,
                'problem': 'Write a function that finds the maximum element in a list of integers.',
                'function_name': 'find_max',
                'test_cases': [
                    {'input': [1, 2, 3, 4, 5], 'expected': 5},
                    {'input': [5, 4, 3, 2, 1], 'expected': 5},
                    {'input': [1], 'expected': 1},
                    {'input': [-1, -2, -3], 'expected': -1},
                    {'input': [1, 1, 1], 'expected': 1}
                ],
                'difficulty': 'Easy',
                'language': 'Python'
            },
            {
                'id': 4,
                'problem': 'Write a function that reverses a string.',
                'function_name': 'reverse_string',
                'test_cases': [
                    {'input': 'hello', 'expected': 'olleh'},
                    {'input': 'world', 'expected': 'dlrow'},
                    {'input': '', 'expected': ''},
                    {'input': 'a', 'expected': 'a'},
                    {'input': 'racecar', 'expected': 'racecar'}
                ],
                'difficulty': 'Easy',
                'language': 'Python'
            },
            {
                'id': 5,
                'problem': 'Write a function that calculates the factorial of a number.',
                'function_name': 'factorial',
                'test_cases': [
                    {'input': 5, 'expected': 120},
                    {'input': 0, 'expected': 1},
                    {'input': 1, 'expected': 1},
                    {'input': 10, 'expected': 3628800},
                    {'input': 3, 'expected': 6}
                ],
                'difficulty': 'Medium',
                'language': 'Python'
            }
        ]
    
    def _format_prompt(self, sample: Dict) -> str:
        """
        Format LiveCodeBench problem into prompt.
        
        Args:
            sample: LiveCodeBench problem sample
        
        Returns:
            Formatted prompt
        """
        problem = sample['problem']
        function_name = sample.get('function_name', 'solution')
        language = sample.get('language', 'Python')
        difficulty = sample.get('difficulty', 'Unknown')
        
        prompt = f"""Write a {language} function to solve the following problem ({difficulty}).

Problem: {problem}

Function name: {function_name}

Provide the complete function implementation with proper syntax and include any necessary imports.

Solution:"""
        
        return prompt
    
    def _extract_answer(self, response: str, sample: Dict) -> Any:
        """
        Extract code from model response.
        
        Args:
            response: Model response
            sample: Original sample
        
        Returns:
            Extracted code
        """
        # Try to extract Python code from the response
        # Look for code blocks with ```python or ```
        
        # Pattern 1: Look for ```python code block
        code_match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Pattern 2: Look for ``` code block
        code_match = re.search(r'```\n(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Pattern 3: Look for function definition
        function_match = re.search(r'def\s+\w+\s*\(.*?\):.*?(?=\ndef\s|\Z)', response, re.DOTALL)
        if function_match:
            return function_match.group(0).strip()
        
        # Pattern 4: Heuristic: start from first plausible code line.
        # Many models emit brief prose before code; executing prose will raise
        # SyntaxError. Prefer slicing from `def` / `import`.
        idx_def = response.find("def ")
        idx_imp = response.find("import ")
        candidates = [i for i in [idx_def, idx_imp] if i != -1]
        if candidates:
            return response[min(candidates):].strip()

        # Pattern 5: If no clear code found, return the entire response.
        return response.strip()

    @staticmethod
    def _sanitize_python(code: str) -> str:
        """Best-effort sanitize model output into valid python.

        Strategy:
        - Strip markdown fences if present.
        - If it doesn't parse, drop leading lines until it parses.
        """
        c = (code or "").strip()
        # Remove fenced blocks if the model returned them without being matched.
        c = re.sub(r"^```(?:python)?\s*\n", "", c, flags=re.IGNORECASE)
        c = re.sub(r"\n```\s*$", "", c)

        lines = c.splitlines()
        # Try progressively dropping leading lines to get a valid module.
        for start in range(0, min(len(lines), 15)):
            candidate = "\n".join(lines[start:]).strip()
            if not candidate:
                continue
            try:
                ast.parse(candidate)
                return candidate
            except SyntaxError:
                continue
        return c
    
    def _evaluate_sample(self, prediction: Any, ground_truth: Any) -> bool:
        """
        Evaluate code by executing it against test cases.
        
        Args:
            prediction: Generated code
            ground_truth: Ground truth (test cases)
        
        Returns:
            True if all test cases pass, False otherwise
        """
        if prediction is None or ground_truth is None:
            return False
        
        code = self._sanitize_python(str(prediction))
        test_cases = ground_truth.get('test_cases', [])
        function_name = ground_truth.get('function_name', 'solution')
        
        if not test_cases:
            self.logger.warning("No test cases provided")
            return False
        
        # Create a temporary file once per sample to reduce overhead.
        passed = 0
        total = len(test_cases)

        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            for test_case in test_cases[: self.num_test_cases]:
                try:
                    result = self._execute_code(temp_file, test_case, function_name=function_name)
                    if result == test_case.get('expected'):
                        passed += 1
                except Exception as e:
                    self.logger.warning(f"Error executing code: {e}")
                    continue
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
        
        # Calculate pass rate
        pass_rate = passed / total if total > 0 else 0.0
        
        # Consider it correct if all test cases pass
        return pass_rate == 1.0
    
    def _execute_code(self, code_file: str, test_case: Dict, function_name: str = 'solution') -> Any:
        """
        Execute code file with test case input.
        
        Args:
            code_file: Path to code file
            test_case: Test case with input and expected output
        
        Returns:
            Actual output from code execution
        """
        def _worker(q: mp.Queue):
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("test_module", code_file)
                module = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(module)

                # Get the function from the module
                if hasattr(module, function_name):
                    func = getattr(module, function_name)
                else:
                    # Try to find any function in the module
                    functions = [name for name in dir(module) if callable(getattr(module, name)) and not name.startswith('_')]
                    if functions:
                        func = getattr(module, functions[0])
                    else:
                        raise ValueError("No function found in code")

                input_data = test_case.get('input')

                # Handle different input types.
                if isinstance(input_data, dict):
                    result = func(**input_data)
                elif isinstance(input_data, (list, tuple)):
                    try:
                        import inspect
                        sig = inspect.signature(func)
                        params = [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
                        has_varargs = any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values())
                        if len(params) == 1 and not has_varargs:
                            result = func(input_data)
                        else:
                            result = func(*input_data)
                    except Exception:
                        result = func(input_data)
                else:
                    result = func(input_data)

                q.put((True, result))
            except Exception as e:
                q.put((False, str(e)))

        q: mp.Queue = mp.Queue(1)
        p = mp.Process(target=_worker, args=(q,))
        p.start()
        p.join(timeout=self.timeout)

        if p.is_alive():
            p.terminate()
            p.join(timeout=1)
            return None

        try:
            ok, payload = q.get_nowait()
        except Exception:
            return None

        if not ok:
            self.logger.warning(f"Error executing code: {payload}")
            return None

        return payload


def load_livecodebench_dataset(dataset_path: str = 'datasets/livecodebench') -> List[Dict]:
    """
    Load LiveCodeBench dataset from file or directory.
    
    Args:
        dataset_path: Path to LiveCodeBench dataset
    
    Returns:
        List of LiveCodeBench problems
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


def download_livecodebench_dataset(output_path: str = 'datasets/livecodebench'):
    """
    Download LiveCodeBench dataset from Hugging Face.
    
    Args:
        output_path: Path to save dataset
    """
    try:
        from datasets import load_dataset

        dataset_id = os.environ.get('BENCH_LIVECODEBENCH_HF_DATASET', '').strip()
        split = os.environ.get('BENCH_LIVECODEBENCH_HF_SPLIT', 'test')

        candidates = [
            dataset_id,
            # Commonly-referenced ids (availability may vary).
            'livecodebench/code_generation_lite',
            'livecodebench/livecodebench',
            'livecodebench/LiveCodeBench',
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
                "Could not download LiveCodeBench from HuggingFace. "
                "Set BENCH_LIVECODEBENCH_HF_DATASET to a valid dataset repo id, or provide a local JSON under datasets/livecodebench. "
                f"Last error: {last_err}"
            )
        
        # Convert to list of dictionaries
        data = []
        for item in dataset:
            data.append({
                'id': item.get('id', len(data)),
                'problem': item['problem'],
                'function_name': item.get('function_name', 'solution'),
                'test_cases': item.get('test_cases', []),
                'difficulty': item.get('difficulty', 'Unknown'),
                'language': item.get('language', 'Python')
            })
        
        # Save to file
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, 'livecodebench.json'), 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"LiveCodeBench dataset downloaded ({dataset_id}:{split}) and saved to {output_path}")
        return data
    
    except Exception as e:
        print(f"Error downloading LiveCodeBench dataset: {e}")
        return []
