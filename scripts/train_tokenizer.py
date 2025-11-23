#!/usr/bin/env python3
"""
Train a Byte-Level BPE tokenizer on project data.

This is intended to become the standard tokenizer step before distillation
training. It streams JSONL examples to avoid holding everything in memory.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, List

from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast


def iter_jsonl_text(paths: List[Path], limit: int | None = None) -> Iterator[str]:
    """Yield text fields from JSONL files."""
    seen = 0
    for path in paths:
        with path.open() as f:
            for line in f:
                if limit is not None and seen >= limit:
                    return
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for key in ("prompt", "response"):
                    if key in obj and isinstance(obj[key], str):
                        seen += 1
                        yield obj[key]
                        if limit is not None and seen >= limit:
                            return


def train_tokenizer(
    input_paths: Iterable[Path],
    output_dir: Path,
    vocab_size: int,
    min_frequency: int,
    limit: int | None,
):
    """Train and save a byte-level BPE tokenizer."""
    output_dir.mkdir(parents=True, exist_ok=True)

    special_tokens = [
        "<|pad|>",
        "<|bos|>",
        "<|eos|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
    ]

    raw_tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)
    raw_tokenizer.train_from_iterator(
        iter_jsonl_text(list(input_paths), limit),
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        pad_token="<|pad|>",
        unk_token=None,
        additional_special_tokens=[
            "<|fim_prefix|>",
            "<|fim_middle|>",
            "<|fim_suffix|>",
        ],
    )

    tokenizer.save_pretrained(output_dir)
    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.txt"
    print(f"Saved tokenizer to {output_dir}")
    print(f"  vocab_size: {len(tokenizer)}")
    print(f"  vocab file: {vocab_path}")
    print(f"  merges:     {merges_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Byte-Level BPE tokenizer on distillation data",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=[
            "data/distillation/train.jsonl",
            "data/distillation/dev.jsonl",
        ],
        help="JSONL files containing prompt/response fields",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tokenizers/code-50k",
        help="Where to save the trained tokenizer",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50257,
        help="Vocabulary size (includes special tokens)",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum token frequency to be included",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of text fields to stream (for quick tests)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_paths = [Path(p) for p in args.input]
    train_tokenizer(
        input_paths=input_paths,
        output_dir=Path(args.output_dir),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
