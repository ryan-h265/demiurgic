#!/usr/bin/env python3
"""
Run HumanEval benchmark on a Demiurgic model.

Usage:
    python scripts/run_humaneval.py --model checkpoints/distilled_10m/final
    python scripts/run_humaneval.py --model checkpoints/7b/final --samples 200
    python scripts/run_humaneval.py --help
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.model import DemiurgicForCausalLM
from src.evaluation import evaluate_model_on_humaneval


def load_tokenizer(tokenizer_path: str):
    """Load tokenizer (basic implementation for testing)."""
    # This is a placeholder - you'll need to implement proper tokenizer loading
    # For now, we'll create a minimal tokenizer interface

    class DummyTokenizer:
        """Minimal tokenizer for testing."""

        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 2
            self.bos_token_id = 1

        def encode(self, text, **kwargs):
            # Very basic - just split on spaces and use ord values
            # In production, use a proper BPE tokenizer
            tokens = [ord(c) % 1000 for c in text[:100]]
            return tokens

        def decode(self, token_ids, **kwargs):
            # Very basic reverse
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            try:
                text = ''.join([chr(t % 128 + 32) for t in token_ids if t > 0])
                return text
            except:
                return ""

        def __call__(self, text, return_tensors=None, **kwargs):
            tokens = self.encode(text)
            if return_tensors == "pt":
                return {
                    'input_ids': torch.tensor([tokens]),
                    'attention_mask': torch.ones(len(tokens)),
                }
            return tokens

    # Try to load real tokenizer
    if os.path.exists(tokenizer_path):
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(tokenizer_path)
        except:
            pass

    # Fallback to dummy
    print("Warning: Using dummy tokenizer. Results may not be meaningful.")
    print("For real evaluation, train a proper tokenizer first.")
    return DummyTokenizer()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Demiurgic model on HumanEval benchmark"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint directory",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer (if None, uses model path or dummy tokenizer)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="humaneval_results",
        help="Directory to save results",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of samples to generate per task (for Pass@K)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )

    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 10, 100],
        help="K values for Pass@K metric",
    )

    parser.add_argument(
        "--humaneval-path",
        type=str,
        default=None,
        help="Path to HumanEval dataset (JSONL). If None, uses human_eval package.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (10 samples per task for testing)",
    )

    args = parser.parse_args()

    # Quick mode adjustments
    if args.quick:
        args.samples = 10
        args.k_values = [1]
        print("🚀 Quick mode: 10 samples per task, Pass@1 only")

    print("="*70)
    print("HumanEval Benchmark for Demiurgic")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Tokenizer: {args.tokenizer or args.model}")
    print(f"  Output: {args.output_dir}")
    print(f"  Samples per task: {args.samples}")
    print(f"  Temperature: {args.temperature}")
    print(f"  K values: {args.k_values}")
    print(f"  Device: {args.device}")
    print()

    # Load model
    print("Loading model...")
    try:
        model = DemiurgicForCausalLM.from_pretrained(args.model, device=args.device)
        model.eval()
        print(f"✓ Model loaded successfully")
        print(f"  Parameters: {model.num_parameters():,}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer_path = args.tokenizer or args.model
    tokenizer = load_tokenizer(tokenizer_path)
    print(f"✓ Tokenizer loaded")

    # Check dependencies
    print("\nChecking dependencies...")
    try:
        import human_eval
        print("✓ human_eval package installed")
    except ImportError:
        print("⚠ Warning: human_eval package not installed")
        print("  Install with: pip install human-eval")
        print("  Falling back to simple evaluation (less accurate)")

    # Run evaluation
    print("\n" + "="*70)
    print("Starting Evaluation")
    print("="*70)

    try:
        results = evaluate_model_on_humaneval(
            model=model,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            num_samples_per_task=args.samples,
            temperature=args.temperature,
            k_values=args.k_values,
            humaneval_path=args.humaneval_path,
        )

        print("\n✓ Evaluation complete!")

        # Print comparison
        print("\n" + "="*70)
        print("Comparison with other models:")
        print("="*70)
        print("Model                    Size      Pass@1")
        print("-"*70)
        print("CodeGen-Mono             16B       29.3%")
        print("StarCoder                15B       33.6%")
        print("Code Llama               13B       36.0%")
        print("GPT-3.5-turbo            ?         48.1%")
        print("-"*70)

        if 'pass@1' in results:
            model_name = Path(args.model).parent.name
            model_size = "?"
            print(f"{model_name:20s}     {model_size:5s}     {results['pass@1']:.1%}")

        print("="*70)

        return 0

    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
