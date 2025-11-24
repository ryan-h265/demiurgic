# Tool Usage Guide - Getting ChatGLM3 to Use Tools

## The Problem

The base ChatGLM3-6B model downloaded via GGUF **has not been trained to use tools**. It doesn't know:
- How to format tool calls
- When to use tools
- What tools are available

This is why the model doesn't use tools automatically.

## The Solution

We've implemented **three strategies** to teach the model tool usage:

### 1. Enhanced System Prompts ✅
The system prompt now includes:
- List of available tools with descriptions
- JSON format examples for each tool
- Clear instructions on when and how to use tools
- Multiple concrete examples

### 2. Few-Shot Examples ✅
The conversation automatically starts with working examples:
- User asks for code execution → Assistant uses `execute_code`
- User asks to create file → Assistant uses `write_file`
- Model learns from these examples in context

### 3. Smart Hints ✅
When you type a message, the system:
- Analyzes your intent
- Adds subtle hints about which tool to use
- Guides the model toward tool usage

## How to Use

### Option 1: Use Enhanced Prompts (Default)

The CLI now automatically:
```python
# Generates tool-aware system prompt
system_prompt = generate_tool_aware_system_prompt(available_tools)

# Includes few-shot examples
conversation.add_message("user", "Calculate fibonacci(10)")
conversation.add_message("assistant", "{tool: execute_code, ...}")
conversation.add_message("tool", "Result: 55")

# Adds hints to your messages
"run this code" → "run this code\n\n**Hint**: Use execute_code tool"
```

### Option 2: Train the Model (Recommended for Production)

For best results, **fine-tune ChatGLM3 on tool usage data**:

```bash
# Generate tool-usage training data
python scripts/generate_tool_training_data.py \
    --provider anthropic \
    --num-examples 2000 \
    --focus-tools

# This creates examples like:
# User: "Check if a number is prime"
# Assistant: {uses execute_code tool}
# Tool: {result}
# Assistant: {explains result}

# Then fine-tune on this data
python scripts/train_chatglm3.py \
    --data data/tool_usage/ \
    --output models/chatglm3-tool-tuned/
```

### Option 3: Manual Demonstration

If the model still doesn't use tools, **show it explicitly**:

```
You: I need you to use tools. Here's an example. When I say "run this code", you should respond with:

```json
{
  "tool": "execute_code",
  "parameters": {
    "language": "python",
    "code": "print('Hello')"
  }
}
```

Now, please calculate factorial of 5 using the execute_code tool.
```

## Tool Format Reference

The model must output tool calls in this exact format:

### execute_code
```json
{
  "tool": "execute_code",
  "parameters": {
    "language": "python",
    "code": "print('Hello, World!')"
  }
}
```

Supported languages: `python`, `javascript`, `bash`

### read_file
```json
{
  "tool": "read_file",
  "parameters": {
    "filepath": "example.txt"
  }
}
```

### write_file
```json
{
  "tool": "write_file",
  "parameters": {
    "filepath": "output.txt",
    "content": "Hello, World!",
    "mode": "write"
  }
}
```

Modes: `write` (overwrite) or `append`

### bash_command
```json
{
  "tool": "bash_command",
  "parameters": {
    "command": "ls -la"
  }
}
```

## Current Behavior

With the enhancements:

### ✅ What Works Now
- Enhanced system prompt teaches tool format
- Few-shot examples show working tool usage
- Hints guide the model toward appropriate tools
- Tool parsing detects JSON blocks in responses

### ⚠️ Current Limitations
- Base model may still prefer describing rather than using tools
- May need explicit reminders to use tools
- Tool calls might not be perfectly formatted
- Requires clear, direct requests

### 🎯 After Fine-Tuning
- Model naturally uses tools without prompting
- Automatically chooses appropriate tool for task
- Perfect JSON formatting
- Multi-step tool usage
- Professional coding assistant behavior

## Example Conversations

### Example 1: With Hints (Current)
```
You: Calculate the 10th fibonacci number