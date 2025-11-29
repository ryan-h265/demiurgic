# ChatGLM3 CLI - Quick Start

## Install & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download model (if needed)
python scripts/download_chatglm3_gguf.py

# 3. Start CLI
python cli.py
```

## Common Commands

```bash
# GPU acceleration (faster)
python cli.py --n-gpu-layers 35

# More threads (faster on CPU)
python cli.py --n-threads 16

# Custom system prompt
python cli.py --system-prompt "You are a Python expert"

# Load previous chat
python cli.py --load-conversation chat.json
```

## In-Chat Commands

| Type this | It does this |
|-----------|--------------|
| `/help` | Show all commands |
| `/tools` | Show available tools |
| `/save chat.json` | Save conversation |
| `/load chat.json` | Load conversation |
| `/clear` | Clear chat history |
| `/context` | Show stats |
| `/exit` | Quit |

## Example Conversations

### Code Generation
```
You: Write a Python function to reverse a string