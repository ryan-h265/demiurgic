"""
Quick test of HumanEval evaluation system without full dataset.

This script demonstrates the evaluation system using a few sample problems.
For full evaluation, use scripts/run_humaneval.py with the real HumanEval dataset.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.model import DemiurgicForCausalLM, get_100m_config
from src.evaluation.humaneval import HumanEvalBenchmark


# Sample HumanEval-style problems for testing
SAMPLE_PROBLEMS = {
    "test/0": {
        "task_id": "test/0",
        "prompt": """from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"
""",
        "entry_point": "has_close_elements",
        "canonical_solution": """    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True

    return False
""",
        "test": """def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True

check(has_close_elements)
"""
    },
    "test/1": {
        "task_id": "test/1",
        "prompt": """

def is_palindrome(text: str) -> bool:
    \"\"\"
    Checks if given string is a palindrome
    >>> is_palindrome('racecar')
    True
    >>> is_palindrome('hello')
    False
    \"\"\"
""",
        "entry_point": "is_palindrome",
        "canonical_solution": """    return text == text[::-1]
""",
        "test": """def check(candidate):
    assert candidate('racecar') == True
    assert candidate('hello') == False
    assert candidate('a') == True
    assert candidate('') == True
    assert candidate('noon') == True

check(is_palindrome)
"""
    },
    "test/2": {
        "task_id": "test/2",
        "prompt": """from typing import List


def sum_positive(numbers: List[int]) -> int:
    \"\"\"
    Sum only the positive numbers in a list
    >>> sum_positive([1, -2, 3, -4, 5])
    9
    >>> sum_positive([-1, -2, -3])
    0
    \"\"\"
""",
        "entry_point": "sum_positive",
        "canonical_solution": """    return sum(n for n in numbers if n > 0)
""",
        "test": """def check(candidate):
    assert candidate([1, -2, 3, -4, 5]) == 9
    assert candidate([-1, -2, -3]) == 0
    assert candidate([10, 20, 30]) == 60
    assert candidate([]) == 0

check(sum_positive)
"""
    },
}


def main():
    print("="*70)
    print("HumanEval Quick Test")
    print("="*70)

    # Try to load model or create a test one
    print("\n1. Loading model...")
    try:
        model = DemiurgicForCausalLM.from_pretrained("checkpoints/distilled_10m/final")
        print("   ✓ Loaded 10M checkpoint")
    except:
        print("   ! No checkpoint found, creating small test model")
        config = get_100m_config()
        config.num_hidden_layers = 4
        config.hidden_size = 256
        config.intermediate_size = 688
        config.num_attention_heads = 4
        model = DemiurgicForCausalLM(config)
        print("   ✓ Created test model")

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create minimal tokenizer
    class DummyTokenizer:
        pad_token_id = 0
        eos_token_id = 2
        bos_token_id = 1

        def __call__(self, text, return_tensors=None, **kwargs):
            # Very basic tokenization
            tokens = [min(ord(c), 1000) for c in text[:512]]
            if return_tensors == "pt":
                return {
                    'input_ids': torch.tensor([tokens]),
                    'attention_mask': torch.ones(len(tokens)),
                }
            return tokens

        def decode(self, token_ids, **kwargs):
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            try:
                return ''.join([chr(min(t, 127)) for t in token_ids if t > 0])
            except:
                return ""

    tokenizer = DummyTokenizer()

    # Create benchmark
    print("\n2. Creating benchmark evaluator...")
    benchmark = HumanEvalBenchmark(
        model=model,
        tokenizer=tokenizer,
        num_samples_per_task=5,  # Just 5 samples for quick test
        temperature=0.8,
    )
    print("   ✓ Benchmark created")

    # Test generation on sample problems
    print("\n3. Testing generation on sample problems...")
    print("-"*70)

    for task_id, problem in SAMPLE_PROBLEMS.items():
        print(f"\n{task_id}: {problem['entry_point']}")
        prompt = problem['prompt']

        # Show prompt
        print(f"\nPrompt (first 150 chars):")
        print(f"{prompt[:150]}...")

        # Generate completion
        try:
            completion = benchmark.generate_completion(prompt)
            print(f"\nGenerated completion:")
            print(f"{completion[:200]}")

            # Try to test it
            full_code = prompt + completion
            success = benchmark._test_code(
                full_code,
                problem['test'],
                problem['entry_point']
            )

            status = "✓ PASS" if success else "✗ FAIL"
            print(f"\nResult: {status}")

        except Exception as e:
            print(f"\n✗ Error: {e}")

        print("-"*70)

    print("\n" + "="*70)
    print("Quick test complete!")
    print("="*70)
    print("\nNote: This test uses a dummy tokenizer and small/untrained model,")
    print("so results are not meaningful. For real evaluation:")
    print("  1. Train a proper BPE tokenizer")
    print("  2. Train the model on code data")
    print("  3. Run: python scripts/run_humaneval.py --model <checkpoint>")
    print("="*70)


if __name__ == "__main__":
    main()
