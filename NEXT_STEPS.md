# Next Steps for ChatGLM3 GGUF Workflow

Use this checklist to progress from the current helpers toward a fully trained and deployable ChatGLM3 GGUF agent stack.

1. **Verify local inference with the GGUF helper.**
   - Download weights via `scripts/download_chatglm3_gguf.py` if you have not already.
   - Install the core dependencies (including `llama-cpp-python`) with `pip install -r requirements-core.txt` to satisfy the
     llama.cpp binding import.
   - Run a smoke test with `scripts/run_chatglm3_gguf.py --model-path <path-to-gguf> --prompt "hello"` to confirm llama.cpp bindings work end-to-end.
2. **Harvest fresh supervision data from the quantized teacher.**
   - Configure `TeacherClientConfig` and `DistillationRunConfig` in `src/distillation/config.py` and call `harvest_supervision` to generate JSONL pairs in `run_config.output_path` for the ChatGLM3 prompt format.
3. **Prepare the SFT dataset.**
   - Convert generated records into the `<|user|> ... <|assistant|>` text format via `src/data/to_sft_format` before training so padding and truncation behave correctly.
4. **Launch supervised fine-tuning.**
   - Use `src/training/train_chatglm3` (or `ChatGLM3Trainer` directly) with `SFTConfig` to run full-precision or QLoRA training on the curated JSONL data; adjust `max_seq_length`, batch size, and logging cadence to your hardware.
5. **Quantize and validate new checkpoints.**
   - After training, export the updated checkpoint to GGUF using llama.cpp’s conversion tools, then run the HumanEval-style harness in `src/evaluation/humaneval.py` (or your own prompts) to confirm quality before adoption.
6. **Integrate with your agent/tooling layer.**
   - Reuse the prompt builder in `src/cli/build_prompt` to structure system instructions, conversation history, and tool schemas consistently between training and inference.

Track progress by checking off each step and iterating on the generated data and training hyperparameters as you refine the model.
