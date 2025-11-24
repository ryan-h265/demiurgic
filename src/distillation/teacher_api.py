"""Llama.cpp-backed teacher API for generating new training pairs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json

from llama_cpp import Llama

from .config import DistillationRunConfig, TeacherClientConfig
from .prompt_generator import PromptGenerator, generate_system_prompt
from ..cli import build_prompt


@dataclass
class TeacherAPI:
    config: TeacherClientConfig

    def __post_init__(self) -> None:
        self.client = Llama(
            model_path=str(self.config.gguf_path),
            n_ctx=self.config.n_ctx,
            n_gpu_layers=self.config.n_gpu_layers,
            embedding=False,
        )

    def generate_batch(self, prompts: List[str], system_prompt: str) -> List[Dict[str, str]]:
        """Generate responses for each prompt with consistent sampling params."""
        results: List[Dict[str, str]] = []
        for prompt in prompts:
            chat_prompt = build_prompt(system_prompt, [
                {"role": "user", "content": prompt},
            ])
            output = self.client(
                chat_prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            text = output["choices"][0]["text"]
            results.append({"prompt": prompt, "response": text})
        return results


def create_teacher_api(config: TeacherClientConfig) -> TeacherAPI:
    return TeacherAPI(config=config)


def harvest_supervision(run_config: DistillationRunConfig, teacher_config: TeacherClientConfig) -> None:
    """Generate a JSONL file of fresh supervision pairs."""
    generator = PromptGenerator()
    teacher = create_teacher_api(teacher_config)
    system_prompt = generate_system_prompt(run_config.categories)

    with Path(run_config.output_path).open("w", encoding="utf-8") as handle:
        for _ in range(run_config.num_chunks):
            prompts = [entry["prompt"] for entry in generator.sample(run_config.prompts_per_chunk)]
            for pair in teacher.generate_batch(prompts, system_prompt):
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")


__all__ = ["TeacherAPI", "create_teacher_api", "harvest_supervision"]
