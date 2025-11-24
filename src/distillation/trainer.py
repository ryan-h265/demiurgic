"""Supervised fine-tuning loop for ChatGLM3 using Hugging Face + PEFT."""

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from ..data import load_jsonl, to_sft_format


@dataclass
class SFTConfig:
    """Training hyperparameters for ChatGLM3 supervised fine-tuning."""

    model_name_or_path: str = "THUDM/chatglm3-6b"
    dataset_path: str = "data/distillation/train.jsonl"
    output_dir: str = "checkpoints/chatglm3-sft"
    use_qlora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_steps: int = 500
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    logging_steps: int = 10
    save_steps: int = 100
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    bf16: bool = True
    gradient_checkpointing: bool = True


class ChatGLM3Trainer:
    """Minimal trainer for SFT/QLoRA runs."""

    def __init__(self, config: SFTConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=True,
            padding_side="right",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_config = None
        if config.use_qlora:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=True,
            quantization_config=quant_config,
            device_map="auto",
        )

        if config.use_qlora:
            self.model = prepare_model_for_kbit_training(self.model)
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["query_key_value", "dense_h_to_4h", "dense_4h_to_h"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)

    def _load_dataset(self) -> Dataset:
        records = load_jsonl(self.config.dataset_path)
        formatted = to_sft_format(records)
        return Dataset.from_list(formatted)

    def _tokenize(self, sample: dict) -> dict:
        return self.tokenizer(
            sample["text"],
            max_length=self.config.max_seq_length,
            truncation=True,
            padding="max_length",
        )

    def train(self, formatting_fn: Optional[Callable[[dict], dict]] = None) -> Trainer:
        dataset = self._load_dataset()
        if formatting_fn:
            dataset = dataset.map(formatting_fn)

        tokenized = dataset.map(self._tokenize, batched=False)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        args = TrainingArguments(
            output_dir=self.config.output_dir,
            max_steps=self.config.max_steps,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=args,
            train_dataset=tokenized,
            data_collator=data_collator,
        )
        trainer.train()
        trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        return trainer


__all__ = ["SFTConfig", "ChatGLM3Trainer"]
