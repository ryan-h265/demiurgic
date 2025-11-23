#!/bin/bash
# Download HumanEval dataset directly from GitHub
# No PyPI package needed!

set -e

echo "Downloading HumanEval dataset..."

DATASET_DIR="data/humaneval"
mkdir -p "$DATASET_DIR"

# Download directly from OpenAI's GitHub
echo "Fetching from GitHub..."
wget -O "$DATASET_DIR/HumanEval.jsonl.gz" \
    "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz" \
    2>&1 || curl -L -o "$DATASET_DIR/HumanEval.jsonl.gz" \
    "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"

# Decompress
echo "Decompressing..."
gunzip -f "$DATASET_DIR/HumanEval.jsonl.gz"

echo "✓ HumanEval dataset downloaded to $DATASET_DIR/HumanEval.jsonl"
echo "✓ $(wc -l < $DATASET_DIR/HumanEval.jsonl) problems loaded"
