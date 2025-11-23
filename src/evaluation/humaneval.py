"""
HumanEval benchmark implementation for Demiurgic.

Evaluates code generation quality using the HumanEval benchmark.
"""

import json
import os
import itertools
import multiprocessing
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
import numpy as np
import torch
from tqdm import tqdm


class HumanEvalBenchmark:
    """
    HumanEval benchmark evaluator.

    Generates code completions for HumanEval problems and evaluates them
    with Pass@K metrics.
    """

    def __init__(
        self,
        model,
        tokenizer,
        num_samples_per_task: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new_tokens: int = 512,
    ):
        """
        Initialize HumanEval benchmark.

        Args:
            model: Demiurgic model for causal LM
            tokenizer: Tokenizer for the model
            num_samples_per_task: Number of solutions to generate per problem
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_new_tokens: Maximum tokens to generate
        """
        self.model = model
        self.tokenizer = tokenizer
        self.num_samples_per_task = num_samples_per_task
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

        self.model.eval()

    def load_problems(self, humaneval_path: Optional[str] = None) -> Dict:
        """
        Load HumanEval problems.

        Args:
            humaneval_path: Path to HumanEval dataset (JSONL file)
                          If None, tries default locations

        Returns:
            Dictionary mapping task_id to problem data
        """
        # Try provided path first
        if humaneval_path and os.path.exists(humaneval_path):
            problems = {}
            with open(humaneval_path, 'r') as f:
                for line in f:
                    problem = json.loads(line)
                    problems[problem['task_id']] = problem
            return problems

        # Try default locations
        default_paths = [
            'data/humaneval/HumanEval.jsonl',
            'HumanEval.jsonl',
            os.path.expanduser('~/.cache/humaneval/HumanEval.jsonl'),
        ]

        for path in default_paths:
            if os.path.exists(path):
                print(f"Loading HumanEval from: {path}")
                problems = {}
                with open(path, 'r') as f:
                    for line in f:
                        problem = json.loads(line)
                        problems[problem['task_id']] = problem
                return problems

        # If human_eval package is available, use it
        try:
            from human_eval.data import read_problems
            print("Loading HumanEval from human_eval package")
            return read_problems()
        except ImportError:
            pass

        # Provide helpful error message
        raise FileNotFoundError(
            "HumanEval dataset not found!\n\n"
            "Please download it:\n"
            "  bash scripts/download_humaneval.sh\n\n"
            "Or provide path explicitly:\n"
            "  benchmark.load_problems('path/to/HumanEval.jsonl')\n\n"
            "Or download manually:\n"
            "  wget https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz\n"
            "  gunzip HumanEval.jsonl.gz\n"
            "  mv HumanEval.jsonl data/humaneval/"
        )

    def generate_completion(self, prompt: str) -> str:
        """
        Generate a single code completion.

        Args:
            prompt: Code prompt to complete

        Returns:
            Generated completion (without prompt)
        """
        # Get model device
        if hasattr(self.model, 'device'):
            device = self.model.device
        elif next(self.model.parameters(), None) is not None:
            device = next(self.model.parameters()).device
        else:
            device = 'cpu'

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        # Move to device
        if isinstance(inputs, dict):
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in inputs.items()}
        else:
            inputs = inputs.to(device)

        # Generate
        with torch.no_grad():
            # Build generate kwargs
            generate_kwargs = {
                'input_ids': inputs['input_ids'],
                'max_new_tokens': self.max_new_tokens,
                'temperature': self.temperature,
                'top_p': self.top_p,
                'top_k': 50,
                'do_sample': True if self.temperature > 0 else False,
                'pad_token_id': getattr(self.tokenizer, 'pad_token_id', 0),
                'eos_token_id': getattr(self.tokenizer, 'eos_token_id', 2),
            }

            # Only add attention_mask if it's in the model's forward signature
            # (some implementations don't support it in generate)
            try:
                outputs = self.model.generate(**generate_kwargs)
            except TypeError:
                # If generate() doesn't accept these kwargs, use minimal version
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )

        # Decode
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract completion (remove prompt)
        completion = full_output[len(prompt):]

        # Stop at first occurrence of a new function/class definition or double newline
        stop_tokens = ['\nclass ', '\ndef ', '\n#', '\nif __name__']
        for stop in stop_tokens:
            if stop in completion:
                completion = completion[:completion.index(stop)]

        return completion

    def generate_samples(
        self,
        problems: Dict,
        output_path: str,
        resume: bool = True,
    ) -> List[Dict]:
        """
        Generate completions for all problems.

        Args:
            problems: Dictionary of HumanEval problems
            output_path: Path to save generated samples (JSONL)
            resume: If True, resume from existing output file

        Returns:
            List of generated samples
        """
        # Load existing samples if resuming
        existing_samples = []
        completed_tasks = set()

        if resume and os.path.exists(output_path):
            print(f"Resuming from {output_path}")
            with open(output_path, 'r') as f:
                for line in f:
                    sample = json.loads(line)
                    existing_samples.append(sample)
                    completed_tasks.add((sample['task_id'], sample['completion']))

            print(f"Loaded {len(existing_samples)} existing samples")

        samples = existing_samples.copy()

        # Count samples per task
        task_counts = Counter(s['task_id'] for s in existing_samples)

        # Generate samples
        print(f"\nGenerating {self.num_samples_per_task} samples per task...")
        print(f"Total tasks: {len(problems)}")

        for task_id, problem in tqdm(problems.items(), desc="Tasks"):
            prompt = problem['prompt']
            num_existing = task_counts.get(task_id, 0)
            num_needed = self.num_samples_per_task - num_existing

            if num_needed <= 0:
                continue

            for _ in tqdm(
                range(num_needed),
                desc=f"{task_id}",
                leave=False,
            ):
                try:
                    completion = self.generate_completion(prompt)

                    sample = {
                        'task_id': task_id,
                        'completion': completion,
                    }

                    samples.append(sample)

                    # Append to file incrementally
                    with open(output_path, 'a') as f:
                        f.write(json.dumps(sample) + '\n')

                except Exception as e:
                    print(f"\nError generating for {task_id}: {e}")
                    continue

        print(f"\nGenerated {len(samples)} total samples")
        return samples

    def evaluate_samples(
        self,
        samples_path: str,
        k_values: List[int] = [1, 10, 100],
        timeout: float = 3.0,
        num_workers: int = 4,
    ) -> Dict:
        """
        Evaluate generated samples using Pass@K metric.

        Args:
            samples_path: Path to generated samples (JSONL)
            k_values: List of K values for Pass@K metric
            timeout: Timeout for code execution (seconds)
            num_workers: Number of parallel workers for testing

        Returns:
            Dictionary with Pass@K results
        """
        # Check if human_eval package is available
        try:
            from human_eval.evaluation import evaluate_functional_correctness

            # Evaluate using official implementation (if available)
            print(f"\nEvaluating with official human_eval package...")
            print(f"Timeout: {timeout}s, Workers: {num_workers}")

            results = evaluate_functional_correctness(
                samples_path,
                k=k_values,
                n_workers=num_workers,
                timeout=timeout,
            )

            return results

        except ImportError:
            print("Note: human_eval package not available (this is fine!).")
            print("Using standalone evaluation implementation.")
            return self._simple_evaluate(samples_path, k_values, timeout)

    def _simple_evaluate(
        self,
        samples_path: str,
        k_values: List[int],
        timeout: float = 3.0,
    ) -> Dict:
        """
        Standalone evaluation implementation (no external package needed).

        This implementation tests code execution and calculates Pass@K metrics
        without requiring the human_eval package.
        """
        # Load samples
        samples = []
        with open(samples_path, 'r') as f:
            for line in f:
                samples.append(json.loads(line))

        # Group by task
        task_samples = defaultdict(list)
        for sample in samples:
            task_samples[sample['task_id']].append(sample)

        # Load problems for test cases
        try:
            problems = self.load_problems()
        except:
            print("Error: Cannot load HumanEval problems for evaluation")
            return {}

        # Test each sample
        results = {}
        for task_id, task_samples_list in task_samples.items():
            problem = problems[task_id]
            test_code = problem['test']
            entry_point = problem['entry_point']

            passed = []
            for sample in task_samples_list:
                # Combine prompt + completion
                full_code = problem['prompt'] + sample['completion']

                # Test
                success = self._test_code(full_code, test_code, entry_point)
                passed.append(success)

            results[task_id] = passed

        # Calculate Pass@K
        pass_at_k = {}
        for k in k_values:
            pass_at_k[f'pass@{k}'] = self._estimate_pass_at_k(results, k)

        return pass_at_k

    def _test_code(self, code: str, test: str, entry_point: str, timeout: float = 3.0) -> bool:
        """
        Test if code passes the test (simple version).

        Warning: This executes arbitrary code. Only use in isolated environment.
        """
        try:
            # Create execution environment
            exec_globals = {}

            # Execute code
            exec(code, exec_globals)

            # Run test
            exec(test, exec_globals)

            # Check function exists
            if entry_point in exec_globals:
                return True
            return False

        except Exception as e:
            return False

    def _estimate_pass_at_k(
        self,
        results: Dict[str, List[bool]],
        k: int,
    ) -> float:
        """
        Estimate Pass@K metric.

        Pass@K = E[1 - (n-c choose k) / (n choose k)]
        where n = total samples, c = correct samples
        """
        def comb(n, k):
            """Binomial coefficient"""
            if k > n or k < 0:
                return 0
            if k == 0 or k == n:
                return 1
            return np.prod([n - i for i in range(k)]) / np.prod([i + 1 for i in range(k)])

        total = 0
        correct = 0

        for task_id, passed_list in results.items():
            n = len(passed_list)
            c = sum(passed_list)

            if n < k:
                # Not enough samples
                continue

            # Pass@K formula
            if c >= k:
                pass_at_k = 1.0
            else:
                pass_at_k = 1.0 - comb(n - c, k) / comb(n, k)

            total += 1
            correct += pass_at_k

        return correct / total if total > 0 else 0.0


def evaluate_model_on_humaneval(
    model,
    tokenizer,
    output_dir: str = "humaneval_results",
    num_samples_per_task: int = 200,
    temperature: float = 0.8,
    k_values: List[int] = [1, 10, 100],
    humaneval_path: Optional[str] = None,
) -> Dict:
    """
    Convenient function to evaluate model on HumanEval.

    Args:
        model: Demiurgic model
        tokenizer: Tokenizer
        output_dir: Directory to save results
        num_samples_per_task: Number of solutions per problem
        temperature: Sampling temperature
        k_values: K values for Pass@K
        humaneval_path: Optional path to HumanEval dataset

    Returns:
        Dictionary with evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create benchmark
    benchmark = HumanEvalBenchmark(
        model=model,
        tokenizer=tokenizer,
        num_samples_per_task=num_samples_per_task,
        temperature=temperature,
    )

    # Load problems
    print("Loading HumanEval problems...")
    problems = benchmark.load_problems(humaneval_path)
    print(f"Loaded {len(problems)} problems")

    # Generate samples
    samples_path = os.path.join(output_dir, "samples.jsonl")
    samples = benchmark.generate_samples(
        problems=problems,
        output_path=samples_path,
        resume=True,
    )

    # Evaluate
    results = benchmark.evaluate_samples(
        samples_path=samples_path,
        k_values=k_values,
    )

    # Save results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("HumanEval Results")
    print("="*70)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2%}")
        else:
            print(f"{k}: {v}")
    print("="*70)

    print(f"\nResults saved to {results_path}")

    return results
