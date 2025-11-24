"""Prompt generation tuned for ChatGLM3 data collection."""

from dataclasses import dataclass
from typing import Dict, Iterable, List
import random


@dataclass
class PromptTemplate:
    category: str
    template: str
    variables: Dict[str, List[str]]

    def fill(self) -> str:
        values = {key: random.choice(options) for key, options in self.variables.items()}
        return self.template.format(**values)


class PromptGenerator:
    """Produce lightweight prompts for code, tool use, and reasoning."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.templates = self._build_templates()

    def _build_templates(self) -> List[PromptTemplate]:
        return [
            PromptTemplate(
                category="code",
                template="Write a {language} function to {task}.",
                variables={
                    "language": ["Python", "TypeScript", "Rust", "Go"],
                    "task": [
                        "stream JSONL logs to stdout safely",
                        "summarize a list of error messages",
                        "retry an HTTP request with exponential backoff",
                        "parse environment variables with defaults",
                    ],
                },
            ),
            PromptTemplate(
                category="tools",
                template=(
                    "Given these tools: {tool_list}, decide which to call and show the JSON payload."
                ),
                variables={
                    "tool_list": [
                        "fetch_weather, set_timer",
                        "search_web, write_file",
                        "run_sql, send_email",
                    ]
                },
            ),
            PromptTemplate(
                category="reasoning",
                template="Explain step-by-step how you would {goal} without executing code.",
                variables={
                    "goal": [
                        "debug a failing API client",
                        "triage a slow database query",
                        "design a cache eviction strategy",
                    ]
                },
            ),
        ]

    def sample(self, count: int) -> List[Dict[str, str]]:
        """Return N prompts with categories for logging."""
        prompts: List[Dict[str, str]] = []
        for _ in range(count):
            template = random.choice(self.templates)
            prompts.append({
                "category": template.category,
                "prompt": template.fill(),
            })
        return prompts


def generate_system_prompt(categories: Iterable[str]) -> str:
    """Build a system prompt that keeps instructions concise and tool-aware."""
    categories_list = ", ".join(categories)
    return (
        "You are a ChatGLM3 teacher model. Focus on concise, factual answers, "
        "produce code when asked, and demonstrate tool calls. Categories: "
        f"{categories_list}."
    )


__all__ = ["PromptTemplate", "PromptGenerator", "generate_system_prompt"]
