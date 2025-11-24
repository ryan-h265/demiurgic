# ChatGLM3 CLI - Interactive Assistant Guide

A powerful TUI-based CLI tool for interacting with your ChatGLM3 model. Similar to Claude Code and Codex, with code execution, file operations, and more.

## Features

✅ **Interactive TUI** - Rich terminal interface with syntax highlighting and markdown rendering
✅ **Multi-turn Conversations** - Context-aware chat with conversation history
✅ **Tool System** - Execute code, read/write files, run bash commands
✅ **Streaming Responses** - Real-time response generation
✅ **Conversation Management** - Save/load conversations, clear history
✅ **Configurable** - YAML-based configuration system
✅ **ChatGLM3 GGUF** - Uses llama.cpp for efficient local inference

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download ChatGLM3 model (if not already done):**
   ```bash
   python scripts/download_chatglm3_gguf.py
   ```

3. **Create default config (optional):**
   ```bash
   python cli.py --create-config
   ```
   This creates `~/.demiurgic/config.yaml` with default settings.

## Quick Start

### Basic Usage

```bash
# Start interactive chat
python cli.py

# Use custom model path
python cli.py --model-path models/chatglm3-6b.Q4_K_M.gguf

# Enable GPU acceleration (if available)
python cli.py --n-gpu-layers 35

# Use more CPU threads
python cli.py --n-threads 16

# Custom system prompt
python cli.py --system-prompt "You are an expert Python developer"

# Load previous conversation
python cli.py --load-conversation my_chat.json
```

### Interactive Commands

Once in the chat interface, you have access to these commands:

| Command | Description |
|---------|-------------|
| `/help` | Show help message with available commands |
| `/tools` | List all available tools |
| `/context` | Show conversation context statistics |
| `/clear` | Clear conversation history |
| `/save <file>` | Save conversation to JSON file |
| `/load <file>` | Load conversation from JSON file |
| `/config` | Show current generation settings |
| `/exit` or `/quit` | Exit the program |

## Available Tools

The CLI comes with several built-in tools that the model can use:

### 1. Code Execution (`execute_code`)

Execute Python, JavaScript, or Bash code and see the output.

**Example:**
```
You: Can you write and test a Python function to calculate fibonacci numbers?