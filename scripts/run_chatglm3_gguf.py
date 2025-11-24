"""Lightweight offline inference helper for the ChatGLM3-6B GGUF model.

This script loads a GGUF checkpoint with `llama_cpp` and runs a single-turn
prompt. Install the llama.cpp Python bindings first via `pip install -r
requirements-core.txt`. It mirrors the minimal pieces you can reuse inside a
custom Python agent (prompt assembly + generation call).
"""

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from llama_cpp import Llama


DEFAULT_MODEL_PATH = Path("models") / "chatglm3-6b.Q4_K_M.gguf"
DEFAULT_STOP: Sequence[str] = ("</s>", "User:", "Assistant:")


Conversation = Sequence[Tuple[str, str]]


def format_prompt(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    history: Optional[Conversation] = None,
) -> str:
    """Build a simple prompt string with optional history.

    The template keeps roles explicit so you can extend it for tool calls later.
    """

    parts: List[str] = []
    if system_prompt:
        parts.append(f"System: {system_prompt}\n")

    if history:
        for prior_user, prior_assistant in history:
            parts.append(f"User: {prior_user}\n")
            parts.append(f"Assistant: {prior_assistant}\n")

    parts.append(f"User: {user_prompt}\n")
    parts.append("Assistant: ")
    return "".join(parts)


def load_model(
    model_path: Path,
    n_ctx: int = 8192,
    n_threads: int = 8,
    n_gpu_layers: int = 0,
    seed: int = 0,
) -> Llama:
    """Load the GGUF model via llama.cpp bindings."""

    return Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        seed=seed,
        verbose=False,
    )


def generate(
    llm: Llama,
    prompt: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stop: Optional[Iterable[str]] = None,
) -> str:
    """Run a single completion call and return the text output."""

    response = llm.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop or DEFAULT_STOP,
    )
    return response["choices"][0]["text"].strip()


def parse_history(raw_history: Sequence[str]) -> Conversation:
    """Parse repeated `--history` flags formatted as `user|assistant`."""

    parsed: List[Tuple[str, str]] = []
    for item in raw_history:
        if "|" not in item:
            raise ValueError("History entries must be formatted as 'user|assistant'")
        user_turn, assistant_turn = item.split("|", maxsplit=1)
        parsed.append((user_turn.strip(), assistant_turn.strip()))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single prompt against chatglm3 GGUF via llama.cpp.")
    parser.add_argument("prompt", help="User prompt to send to the model.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the GGUF model file (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument("--system", default=None, help="Optional system instruction.")
    parser.add_argument(
        "--history",
        action="append",
        default=[],
        metavar="USER|ASSISTANT",
        help="Optional past turns as 'user|assistant'; can be repeated.",
    )
    parser.add_argument("--n-ctx", type=int, default=8192, help="Context window size.")
    parser.add_argument("--n-threads", type=int, default=8, help="Number of CPU threads for inference.")
    parser.add_argument("--n-gpu-layers", type=int, default=0, help="Number of layers to offload to GPU (if supported).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling parameter.")

    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {args.model_path}. Download it via scripts/download_chatglm3_gguf.py first."
        )

    history = parse_history(args.history) if args.history else None
    prompt = format_prompt(args.prompt, system_prompt=args.system, history=history)
    llm = load_model(
        model_path=args.model_path,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
        seed=args.seed,
    )
    output = generate(
        llm,
        prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(output)


if __name__ == "__main__":
    main()
