# ChatGLM3 Interactive CLI

A powerful TUI-based CLI tool for interacting with ChatGLM3. Similar to Claude Code and GitHub Copilot CLI, with code execution, file operations, and intelligent conversation.

## ✨ Features

- 🖥️ **Beautiful TUI** - Rich terminal interface with markdown rendering and syntax highlighting
- 💬 **Interactive Chat** - Multi-turn conversations with full context awareness
- 🛠️ **Tool System** - Execute code, read/write files, run bash commands
- ⚡ **Streaming** - Real-time response generation with live updates
- 💾 **Persistence** - Save and load conversations
- ⚙️ **Configurable** - YAML-based configuration for all settings
- 🚀 **Fast** - Uses llama.cpp for efficient GGUF model inference

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Model (if not done)

```bash
python scripts/download_chatglm3_gguf.py
```

### 3. Run CLI

```bash
python cli.py
```

## 📖 Usage

### Basic Commands

```bash
# Start interactive chat
python cli.py

# Use custom model
python cli.py --model-path models/chatglm3-6b.Q4_K_M.gguf

# Enable GPU acceleration (if CUDA available)
python cli.py --n-gpu-layers 35

# More CPU threads for faster inference
python cli.py --n-threads 16

# Custom system prompt
python cli.py --system-prompt "You are a Python expert"

# Load previous conversation
python cli.py --load-conversation saved_chat.json

# Create default config file
python cli.py --create-config
```

### Interactive Commands

Once in the chat, use these commands:

| Command | Description |
|---------|-------------|
| `/help` | Show help and available commands |
| `/tools` | List all available tools |
| `/context` | Show conversation statistics |
| `/clear` | Clear conversation history |
| `/save <file>` | Save conversation to file |
| `/load <file>` | Load conversation from file |
| `/config` | Show generation settings |
| `/exit` or `/quit` | Exit program |

## 🛠️ Available Tools

The assistant can use these tools automatically:

### 1. execute_code
Execute code in Python, JavaScript, or Bash
```json
{
  "tool": "execute_code",
  "parameters": {
    "language": "python",
    "code": "print('Hello World')"
  }
}
```

### 2. read_file
Read contents from a file
```json
{
  "tool": "read_file",
  "parameters": {
    "filepath": "example.txt"
  }
}
```

### 3. write_file
Write or append to a file
```json
{
  "tool": "write_file",
  "parameters": {
    "filepath": "output.txt",
    "content": "Hello!",
    "mode": "write"
  }
}
```

### 4. bash_command
Execute bash commands (with safety restrictions)
```json
{
  "tool": "bash_command",
  "parameters": {
    "command": "ls -la"
  }
}
```

## ⚙️ Configuration

Configuration file location: `~/.demiurgic/config.yaml`

Create default config:
```bash
python cli.py --create-config
```

### Example Configuration

```yaml
model:
  model_path: models/chatglm3-6b.Q4_K_M.gguf
  n_ctx: 8192
  n_threads: 8
  n_gpu_layers: 0
  verbose: false

generation:
  max_tokens: 512
  temperature: 0.7
  top_p: 0.9
  top_k: 40
  repeat_penalty: 1.1
  stop_sequences:
    - </s>
    - <|user|>
    - <|system|>

conversation:
  max_context_length: 8192
  max_history_turns: 20
  system_prompt: null

tools:
  enabled: true
  enable_code_execution: true
  enable_file_operations: true
  enable_bash_commands: true
  allowed_paths:
    - "."
  code_execution_timeout: 30
  bash_timeout: 30
  max_file_size: 100000
  max_output_size: 10000
```

## 💡 Example Conversations

### Example 1: Code Generation + Execution

```
You: Write a Python function to calculate factorial and test it