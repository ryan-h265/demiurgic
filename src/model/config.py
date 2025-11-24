"""Configuration helpers for ChatGLM3 fine-tuning and inference."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ChatGLM3Paths:
    """Paths used for training and GGUF inference."""

    hf_model: str = "THUDM/chatglm3-6b"
    gguf_path: Optional[Path] = None
    output_dir: Path = Path("checkpoints/chatglm3-sft")


@dataclass
class GenerationConfig:
    """Sampling configuration shared across scripts."""

    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    n_ctx: int = 4096
    n_gpu_layers: int = 0


__all__ = ["ChatGLM3Paths", "GenerationConfig"]
