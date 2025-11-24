# ChatGLM3-6B GGUF setup

This repository now focuses on a lightweight path to work with `chatglm3-6b.Q4_K_M.gguf` using the existing llama.cpp-based tooling.

## Download the GGUF model
Use the helper script to grab the quantized weights locally (defaults to `models/chatglm3-6b.Q4_K_M.gguf`).

```bash
python scripts/download_chatglm3_gguf.py
```

Flags you might want:
- `--output-dir`: where to store the GGUF file if you prefer a different path.
- `--token`: pass a Hugging Face token if your network throttles anonymous downloads.
- `--force`: re-download even if the file already exists locally.

## Quick smoke test with llama.cpp bindings
After downloading the model, run a single prompt through the GGUF file with the lightweight helper. This is the same minimal
flow you can embed in your agent.

```bash
python scripts/run_chatglm3_gguf.py "Write a two-line haiku about tool use"
```

Flags you might tweak:
- `--system`: add a system instruction to steer behavior.
- `--history`: include prior turns as `user|assistant` pairs (flag can repeat).
- `--n-ctx`, `--n-threads`, `--n-gpu-layers`: resource controls for your hardware.
- `--max-tokens`, `--temperature`, `--top-p`: sampling controls.

## Plan: simple inference loader for your agent
1. **Loader**: Implement a thin `LlamaCppInferenceEngine` that wraps `llama_cpp.Llama` with `model_path` pointing to the downloaded GGUF file, plus tunables for `n_ctx`, `n_threads`, and `n_gpu_layers`.
2. **Prompt shaping**: Build a prompt assembler that accepts system text, conversation turns, and optional tool schemas/results. Concatenate them into the chatglm3-friendly template before calling the engine.
3. **Agent loop**: Expose a function like `generate(prompt, max_tokens, temperature, top_p)` that your Python agent can call, returning both the raw text and any parsed tool-call JSON if applicable.
4. **Lifecycle hooks**: Add small health checks (load, single-turn echo, tool-call stub) and a hot-reload flag to swap in refreshed GGUF exports without restarting the agent process.

## Optional next steps for training/refresh
- Reuse `scripts/train_with_distillation.py` with your conversation datasets to keep fine-tuning iteratively.
- After each training phase, re-quantize to GGUF using llama.cpp’s tools and swap the `model_path` in your loader.
