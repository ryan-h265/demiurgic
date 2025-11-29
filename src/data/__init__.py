"""Data loading helpers for ChatGLM3 fine-tuning."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict
import json


@dataclass
class ConversationRecord:
    """A single instruction/response pair for supervised fine-tuning."""

    prompt: str
    response: str


def load_jsonl(path: str) -> List[ConversationRecord]:
    """Load JSONL data that contains `prompt` and `response` keys."""
    records: List[ConversationRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            records.append(ConversationRecord(
                prompt=payload["prompt"],
                response=payload["response"],
            ))
    return records


def to_sft_format(records: Iterable[ConversationRecord]) -> List[Dict[str, str]]:
    """Convert records into a simple text field for language-model SFT."""
    formatted: List[Dict[str, str]] = []
    for record in records:
        formatted.append({
            "text": f"<|user|>\n{record.prompt.strip()}\n<|assistant|>\n{record.response.strip()}<|/assistant|>",
        })
    return formatted


__all__ = ["ConversationRecord", "load_jsonl", "to_sft_format"]
