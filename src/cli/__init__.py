"""Prompt utilities for ChatGLM3 GGUF inference."""

from typing import List, Dict, Optional

CHAT_TEMPLATE = """<|system|>\n{system}\n<|/system|>\n{messages}\n<|assistant|>"""


def format_messages(messages: List[Dict[str, str]]) -> str:
    """Render chat messages into the ChatGLM3 chat template.

    Args:
        messages: A list of {"role": str, "content": str} dicts. Roles should
            be "user", "assistant", or "tool".

    Returns:
        A string formatted for ChatGLM3 generation.
    """
    rendered = []
    for message in messages:
        role = message["role"]
        content = message["content"].strip()
        rendered.append(f"<|{role}|>\n{content}\n<|/{role}|>")
    return "\n".join(rendered)


def build_prompt(system: str, messages: List[Dict[str, str]],
                 tools: Optional[List[Dict[str, str]]] = None) -> str:
    """Build a prompt that includes an optional tool schema section.

    Args:
        system: System prompt that sets behavior and constraints.
        messages: Conversation history to include in the prompt.
        tools: Optional list of tool specs (each should already be serialized).

    Returns:
        Prompt string ready to pass to llama_cpp or another inference backend.
    """
    tool_block = ""
    if tools:
        tool_schemas = "\n\n".join(tool["schema"] for tool in tools)
        tool_block = f"\n<|tools|>\n{tool_schemas}\n<|/tools|>"

    chat_body = format_messages(messages)
    return CHAT_TEMPLATE.format(system=system.strip(), messages=chat_body) + tool_block


__all__ = ["build_prompt", "format_messages", "CHAT_TEMPLATE"]
