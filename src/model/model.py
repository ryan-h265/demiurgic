"""Utilities to load ChatGLM3 for training and inference."""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from llama_cpp import Llama

from .config import GenerationConfig


@dataclass
class TrainingLoadResult:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer


def load_for_training(
    model_name_or_path: str,
    use_qlora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> TrainingLoadResult:
    """Load ChatGLM3 with optional QLoRA adapters configured."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if use_qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto",
    )

    if use_qlora:
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query_key_value", "dense_h_to_4h", "dense_4h_to_h"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    return TrainingLoadResult(model=model, tokenizer=tokenizer)


def load_gguf_for_inference(gguf_path: str, gen_config: Optional[GenerationConfig] = None) -> Llama:
    """Load a llama.cpp-backed ChatGLM3 model for inference."""
    gen = gen_config or GenerationConfig()
    return Llama(
        model_path=gguf_path,
        n_ctx=gen.n_ctx,
        n_gpu_layers=gen.n_gpu_layers,
        embedding=False,
    )


def generate_with_llama(
    client: Llama,
    prompt: str,
    gen_config: Optional[GenerationConfig] = None,
) -> str:
    """Run a single completion against a llama.cpp model."""
    gen = gen_config or GenerationConfig()
    output = client(
        prompt,
        max_tokens=gen.max_tokens,
        temperature=gen.temperature,
        top_p=gen.top_p,
    )
    return output["choices"][0]["text"]


__all__ = [
    "TrainingLoadResult",
    "load_for_training",
    "load_gguf_for_inference",
    "generate_with_llama",
]
