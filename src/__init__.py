"""Utilities for fine-tuning and serving ChatGLM3 GGUF models.

This package now targets a llama.cpp-backed workflow for ChatGLM3
models. It includes helpers for:
- Preparing training data
- Running supervised fine-tuning (QLoRA or full precision)
- Querying a local GGUF build to generate fresh supervision data
- Lightweight evaluation harnesses for code-generation benchmarks
"""

__all__ = [
    "cli",
    "data",
    "distillation",
    "evaluation",
    "model",
    "training",
]
