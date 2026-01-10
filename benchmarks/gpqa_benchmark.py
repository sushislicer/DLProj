"""
GPQA (Graduate-Level Google-Proof Q&A) benchmark implementation.
"""

import os
import json
import logging
from typing import Dict, List, Any
import re
from .base_benchmark import MultipleChoiceBenchmark


class GPQABenchmark(MultipleChoiceBenchmark):
    """GPQA benchmark for graduate-level multiple choice questions."""
    
    def __init__(self, config: Dict, logger: logging.Logger):
        """
        Initialize GPQA benchmark.
        
        Args:
            config: Benchmark configuration
            logger: Logger instance
        """
        super().__init__(config, logger)
        self.logger.info("GPQA Benchmark initialized")
    
    def _load_dataset(self) -> List[Dict]:
        """
        Load GPQA dataset.
        
        Returns:
            List of GPQA questions
        """
        dataset_path = self.config.get('dataset_path', 'datasets/gpqa')
        
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
        self.logger.info(f"GPQA dataset not found at {dataset_path}, attempting download...")
        data = download_gpqa_dataset(dataset_path)
        if data:
            return data

        self.logger.warning(f"Download failed, using sample data")
        return self._create_sample_dataset()
    
    def _create_sample_dataset(self) -> List[Dict]:
        """Create sample GPQA dataset for testing."""
        return [
            {
                'id': 1,
                'question': 'What is the primary function of mitochondria in eukaryotic cells?',
                'choices': ['A) Protein synthesis', 'B) Energy production', 'C) DNA replication', 'D) Lipid storage'],
                'answer': 'B',
                'subject': 'Biology',
                'difficulty': 'Graduate',
                'explanation': 'Mitochondria are known as the powerhouses of the cell, responsible for ATP production through cellular respiration.'
            },
            {
                'id': 2,
                'question': 'Which quantum mechanical principle states that it is impossible to simultaneously know the exact position and momentum of a particle?',
                'choices': ['A) Pauli Exclusion Principle', 'B) Heisenberg Uncertainty Principle', 'C) Schrödinger Equation', 'D) Bohr Model'],
                'answer': 'B',
                'subject': 'Physics',
                'difficulty': 'Graduate',
                'explanation': 'The Heisenberg Uncertainty Principle, formulated by Werner Heisenberg, states that the more precisely the position of a particle is determined, the less precisely its momentum can be known, and vice versa.'
            },
            {
                'id': 3,
                'question': 'In organic chemistry, what is the mechanism of the SN2 reaction?',
                'choices': ['A) Two-step mechanism with carbocation intermediate', 'B) Concerted one-step mechanism with backside attack', 'C) Radical mechanism', 'D) Electrophilic addition'],
                'answer': 'B',
                'subject': 'Chemistry',
                'difficulty': 'Graduate',
                'explanation': 'SN2 (Substitution Nucleophilic Bimolecular) is a concerted reaction where the nucleophile attacks the electrophilic carbon from the backside, leading to inversion of configuration.'
            },
            {
                'id': 4,
                'question': 'What is the time complexity of the QuickSort algorithm in the average case?',
                'choices': ['A) O(n)', 'B) O(n log n)', 'C) O(n²)', 'D) O(log n)'],
                'answer': 'B',
                'subject': 'Computer Science',
                'difficulty': 'Graduate',
                'explanation': 'QuickSort has an average-case time complexity of O(n log n) due to its divide-and-conquer approach, though it can degrade to O(n²) in the worst case.'
            },
            {
                'id': 5,
                'question': 'Which statistical test is used to determine if there is a significant difference between the means of two independent groups?',
                'choices': ['A) Chi-square test', 'B) ANOVA', 'C) Independent t-test', 'D) Paired t-test'],
                'answer': 'C',
                'subject': 'Statistics',
                'difficulty': 'Graduate',
                'explanation': 'The independent t-test (or two-sample t-test) is used to compare the means of two independent groups to determine if there is a statistically significant difference between them.'
            },
            {
                'id': 6,
                'question': 'What is the role of the hippocampus in the human brain?',
                'choices': ['A) Motor control', 'B) Visual processing', 'C) Memory formation and consolidation', 'D) Language production'],
                'answer': 'C',
                'subject': 'Neuroscience',
                'difficulty': 'Graduate',
                'explanation': 'The hippocampus is crucial for the formation of new memories and spatial navigation. It plays a key role in consolidating short-term memories into long-term memories.'
            },
            {
                'id': 7,
                'question': 'In thermodynamics, what does the Second Law state about entropy?',
                'choices': ['A) Entropy always decreases in a closed system', 'B) Entropy remains constant in all processes', 'C) Entropy always increases in an isolated system', 'D) Entropy is unrelated to energy'],
                'answer': 'C',
                'subject': 'Physics',
                'difficulty': 'Graduate',
                'explanation': 'The Second Law of Thermodynamics states that the total entropy of an isolated system always increases over time, approaching a maximum value at equilibrium.'
            },
            {
                'id': 8,
                'question': 'What is the primary function of ribosomes in protein synthesis?',
                'choices': ['A) DNA replication', 'B) Transcription', 'C) Translation', 'D) RNA splicing'],
                'answer': 'C',
                'subject': 'Biology',
                'difficulty': 'Graduate',
                'explanation': 'Ribosomes are the molecular machines that perform translation, synthesizing proteins by reading mRNA sequences and assembling amino acids in the correct order.'
            },
            {
                'id': 9,
                'question': 'Which algorithm is commonly used for finding the shortest path in a weighted graph with non-negative edge weights?',
                'choices': ['A) Bellman-Ford', 'B) Dijkstra\'s algorithm', 'C) Floyd-Warshall', 'D) Breadth-First Search'],
                'answer': 'B',
                'subject': 'Computer Science',
                'difficulty': 'Graduate',
                'explanation': 'Dijkstra\'s algorithm is a greedy algorithm that finds the shortest path from a source node to all other nodes in a weighted graph with non-negative edge weights.'
            },
            {
                'id': 10,
                'question': 'What is the mechanism of action of selective serotonin reuptake inhibitors (SSRIs)?',
                'choices': ['A) Increase dopamine levels', 'B) Block serotonin receptors', 'C) Inhibit serotonin reuptake', 'D) Decrease serotonin production'],
                'answer': 'C',
                'subject': 'Pharmacology',
                'difficulty': 'Graduate',
                'explanation': 'SSRIs work by inhibiting the reuptake of serotonin into the presynaptic neuron, increasing the concentration of serotonin in the synaptic cleft and enhancing serotonergic neurotransmission.'
            }
        ]
    
    def _format_prompt(self, sample: Dict) -> str:
        """
        Format GPQA question into prompt.
        
        Args:
            sample: GPQA question sample
        
        Returns:
            Formatted prompt
        """
        question = sample['question']
        choices = sample['choices']
        subject = sample.get('subject', 'General')
        difficulty = sample.get('difficulty', 'Graduate')
        
        # Format choices
        choices_text = '\n'.join(choices)
        
        prompt = f"""Answer the following {subject} question ({difficulty} level). Select the best answer from the choices provided.

Question: {question}

Choices:
{choices_text}

Provide your answer as a single letter (A, B, C, or D) and briefly explain your reasoning.

Answer:"""
        
        return prompt
    
    def _extract_answer(self, response: str, sample: Dict) -> Any:
        """
        Extract answer from model response.
        
        Args:
            response: Model response
            sample: Original sample
        
        Returns:
            Extracted answer (letter A, B, C, or D)
        """
        # Try to extract the answer letter from the response
        # Look for patterns like "Answer: A", "The answer is B", etc.
        
        # Pattern 1: Look for "Answer:" or "answer:" followed by a letter
        answer_match = re.search(r'(?:Answer|answer|The answer is|Therefore, the answer is)[:\s]*([A-D])', response, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).upper()
        
        # Pattern 2: Look for standalone letter at the beginning or end
        standalone_match = re.search(r'\b([A-D])\b', response)
        if standalone_match:
            return standalone_match.group(1).upper()
        
        # Pattern 3: Look for "Option A", "Choice B", etc.
        option_match = re.search(r'(?:Option|Choice|Select)[:\s]*([A-D])', response, re.IGNORECASE)
        if option_match:
            return option_match.group(1).upper()
        
        # Pattern 4: If no clear answer found, return None
        return None
    
    def _evaluate_sample(self, prediction: Any, ground_truth: Any) -> bool:
        """
        Evaluate GPQA answer.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
        
        Returns:
            True if correct, False otherwise
        """
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
        
        # Check if valid choices
        valid_choices = {'A', 'B', 'C', 'D'}
        if pred_str not in valid_choices or truth_str not in valid_choices:
            return False
        
        return pred_str == truth_str


def load_gpqa_dataset(dataset_path: str = 'datasets/gpqa') -> List[Dict]:
    """
    Load GPQA dataset from file or directory.
    
    Args:
        dataset_path: Path to GPQA dataset
    
    Returns:
        List of GPQA questions
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


def download_gpqa_dataset(output_path: str = 'datasets/gpqa'):
    """
    Download GPQA dataset from Hugging Face.
    
    Args:
        output_path: Path to save dataset
    """
    try:
        from datasets import load_dataset
        
        # Load GPQA dataset from Hugging Face
        dataset = load_dataset("Idavidrein/gpqa", split="test")
        
        # Convert to list of dictionaries
        data = []
        for item in dataset:
            data.append({
                'id': item.get('id', len(data)),
                'question': item['question'],
                'choices': item.get('choices', []),
                'answer': item.get('answer', ''),
                'subject': item.get('subject', 'General'),
                'difficulty': item.get('difficulty', 'Graduate'),
                'explanation': item.get('explanation', '')
            })
        
        # Save to file
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, 'gpqa.json'), 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"GPQA dataset downloaded and saved to {output_path}")
        return data
    
    except Exception as e:
        print(f"Error downloading GPQA dataset: {e}")
        return []
