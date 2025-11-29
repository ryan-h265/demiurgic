"""Configuration for collecting training data from a ChatGLM3 GGUF teacher."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class TeacherClientConfig:
    """How to connect to a local llama.cpp-powered ChatGLM3 model."""

    gguf_path: Path
    n_ctx: int = 4096
    n_gpu_layers: int = 0
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512


@dataclass
class DistillationRunConfig:
    """Settings for harvesting new supervision data."""

    output_path: Path
    prompts_per_chunk: int = 128
    num_chunks: int = 1
    system_prompt: str = (
        "You are ChatGLM3 tuned for tool use and concise coding help."
    )
    categories: List[str] = field(default_factory=lambda: ["code", "tools", "reasoning"])


__all__ = ["TeacherClientConfig", "DistillationRunConfig"]
