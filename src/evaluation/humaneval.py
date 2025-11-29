"""Lightweight HumanEval-style harness for ChatGLM3 GGUF."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from llama_cpp import Llama

from ..cli import build_prompt


task_instructions = """
You are completing a coding task. Return only the final Python function.
""".strip()


@dataclass
class HumanEvalConfig:
    gguf_path: Path
    max_tokens: int = 256
    n_ctx: int = 4096
    n_gpu_layers: int = 0
    temperature: float = 0.2
    top_p: float = 0.9


class HumanEvalRunner:
    def __init__(self, config: HumanEvalConfig):
        self.model = Llama(
            model_path=str(config.gguf_path),
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
        )
        self.config = config

    def generate(self, prompt: str) -> str:
        full_prompt = build_prompt(task_instructions, [{"role": "user", "content": prompt}])
        output = self.model(
            full_prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        return output["choices"][0]["text"]

    def run_suite(self, problems: Iterable[str]) -> List[str]:
        return [self.generate(problem) for problem in problems]


__all__ = ["HumanEvalConfig", "HumanEvalRunner"]
