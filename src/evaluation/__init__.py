"""
Evaluation package for Demiurgic models.

Includes benchmarks for code generation quality.
"""

from .humaneval import HumanEvalBenchmark, evaluate_model_on_humaneval

__all__ = [
    "HumanEvalBenchmark",
    "evaluate_model_on_humaneval",
]
