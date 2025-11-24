# Quick Start - Getting Tools to Work

## TL;DR

The base ChatGLM3 model doesn't know how to use tools. We've added:
1. **Enhanced system prompts** with tool examples
2. **Few-shot learning** (automatic examples)
3. **Smart hints** in your messages

**Just run:** `python cli.py` and it should work better now!

## How to Test

```bash
# Start CLI
python cli.py

# Try these test messages:
You: Calculate fibonacci of 10 using Python

You: Create a file called test.txt with hello world

You: List files in the current directory

You: Read the README.md file
```

## Expected Behavior

The model should respond with JSON like:

```json
{
  "tool": "execute_code",
  "parameters": {
    "language": "python",
    "code": "def fib(n): ..."
  }
}
```

## If It Still Doesn't Work

### Option 1: Be More Explicit
```
You: Use the execute_code tool to run this Python code: print("Hello")
```

### Option 2: Show an Example First
```
You: When I ask you to run code, respond with JSON like this:

```json
{
  "tool": "execute_code",
  "parameters": {
    "language": "python",
    "code": "print('Hello')"
  }
}
```

Now please calculate 5 factorial.
```

### Option 3: Fine-Tune the Model (Best Solution)

Generate training data focused on tool usage:

```bash
# Create tool-focused training data
python scripts/generate_distillation_data.py \
    --provider anthropic \
    --model claude-3-5-sonnet-20241022 \
    --num-examples 2000 \
    --mode tool-focused \
    --output-dir data/tool_training

# Fine-tune ChatGLM3
# (Run on cloud GPU)
python scripts/train_chatglm3.py \
    --data data/tool_training/ \
    --model THUDM/chatglm3-6b \
    --output models/chatglm3-tools/
```

## Why This Happens

The base GGUF model you downloaded:
- ✅ Can chat and answer questions
- ✅ Can write code
- ❌ Wasn't trained to output JSON tool calls
- ❌ Doesn't know tool-calling format

Our enhancements **teach** it through:
- Detailed instructions in system prompt
- Working examples in conversation history
- Hints in your messages

But for **production quality**, fine-tuning is recommended.

## What's Been Added

| Component | What It Does | File |
|-----------|-------------|------|
| Enhanced System Prompt | Teaches tool format with examples | `src/cli/prompts.py` |
| Few-Shot Examples | Shows 2 working tool usage examples | `src/cli/prompts.py` |
| Smart Hints | Adds tool suggestions to messages | `src/cli/prompts.py` |
| Auto-Integration | All features enabled by default | `cli.py`, `interface.py` |

## Next Steps

1. **Test current enhancements**: `python cli.py`
2. **If not satisfied**: Fine-tune on tool usage data
3. **For production**: Train with 2000+ tool-focused examples

See `TOOL_USAGE_GUIDE.md` for complete details.
