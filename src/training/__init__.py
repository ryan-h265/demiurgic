"""High-level training entry points for ChatGLM3."""

from pathlib import Path
from typing import Optional

from ..distillation import ChatGLM3Trainer, SFTConfig


def train_chatglm3(config: Optional[SFTConfig] = None) -> None:
    """Kick off a supervised fine-tuning run with sane defaults."""
    cfg = config or SFTConfig()
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    trainer = ChatGLM3Trainer(cfg)
    trainer.train()


__all__ = ["train_chatglm3", "SFTConfig", "ChatGLM3Trainer"]
