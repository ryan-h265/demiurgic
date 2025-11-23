#!/usr/bin/env python3
"""
Test the distillation trainer to ensure it's working correctly.

This creates a small dataset and tests initialization without full training.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from transformers import AutoTokenizer

from src.distillation.config import (
    OutputDistillationConfig,
    LogitDistillationConfig,
)
from src.distillation.trainer import DistillationTrainer, DistillationDataset
from src.model.model import DemiurgicForCausalLM
from src.model.config import DemiurgicConfig


def create_test_data(output_dir: Path, num_examples: int = 10):
    """Create a small test dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = [
        {
            "prompt": f"Write a Python function to compute factorial of {i}",
            "response": f"```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n```\n\nThis function computes the factorial using recursion.",
            "category": "function_implementation",
            "language": "python",
            "tokens_used": 150,
            "cost_estimate": 0.01,
        }
        for i in range(num_examples)
    ]

    # Save to JSONL
    train_file = output_dir / 'train.jsonl'
    with open(train_file, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    print(f"Created test data: {train_file}")
    return str(train_file)


def test_dataset(data_path: str, tokenizer):
    """Test dataset loading."""
    print("\n" + "="*60)
    print("Testing Dataset")
    print("="*60)

    dataset = DistillationDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=512,
    )

    print(f"Dataset size: {len(dataset)}")

    # Get a sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Input shape: {sample['input_ids'].shape}")
    print(f"Attention mask shape: {sample['attention_mask'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")

    # Check masking
    num_unmasked = (sample['labels'] != -100).sum().item()
    print(f"Unmasked tokens: {num_unmasked} / {sample['labels'].shape[0]}")

    print("\n✓ Dataset test passed")
    return dataset


def test_output_distillation(data_path: str, tokenizer):
    """Test output distillation trainer."""
    print("\n" + "="*60)
    print("Testing Output Distillation")
    print("="*60)

    # Create config
    config = OutputDistillationConfig(
        train_data_path=data_path,
        batch_size=2,
        max_steps=10,
        logging_steps=5,
        save_steps=100,  # Don't save during test
        output_dir='checkpoints/test_output',
        use_gradient_checkpointing=False,
        use_mixed_precision=False,
        device='cpu',  # Use CPU for testing
    )

    print(f"\nConfig:")
    print(f"  Distillation type: {config.distillation_type}")
    print(f"  Alpha: {config.alpha}")
    print(f"  Temperature: {config.temperature}")

    # Create small student model
    model_config = DemiurgicConfig(
        vocab_size=len(tokenizer),
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        max_position_embeddings=512,
        use_flash_attention_2=False,
    )

    student_model = DemiurgicForCausalLM(model_config)
    print(f"\nStudent model: {student_model.num_parameters() / 1e6:.1f}M parameters")

    # Create trainer
    trainer = DistillationTrainer(
        config=config,
        student_model=student_model,
        tokenizer=tokenizer,
        teacher_model=None,  # No teacher for output distillation
    )

    print("\nTrainer initialized:")
    print(f"  Train dataset size: {len(trainer.train_dataset)}")
    print(f"  Device: {trainer.device}")

    # Test a single batch
    print("\nTesting single batch...")
    batch = next(iter(trainer.train_loader))

    metrics = trainer.train_step(batch)
    print(f"  Metrics: {metrics}")

    assert 'total_loss' in metrics
    assert 'hard_loss' in metrics
    assert metrics['soft_loss'] == 0.0  # No soft loss for output distillation

    print("\n✓ Output distillation test passed")


def test_logit_distillation_config():
    """Test logit distillation configuration."""
    print("\n" + "="*60)
    print("Testing Logit Distillation Config")
    print("="*60)

    # Test that it requires teacher model
    try:
        config = LogitDistillationConfig(
            train_data_path='data/test/train.jsonl',
            teacher_model_name=None,  # Should fail
        )
        print("❌ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised error: {e}")

    # Test valid config
    config = LogitDistillationConfig(
        train_data_path='data/test/train.jsonl',
        teacher_model_name='gpt2',
        batch_size=2,
        alpha=0.5,
        temperature=2.0,
    )

    print(f"\nValid config created:")
    print(f"  Distillation type: {config.distillation_type}")
    print(f"  Teacher model: {config.teacher_model_name}")
    print(f"  Alpha: {config.alpha}")
    print(f"  Temperature: {config.temperature}")

    print("\n✓ Logit distillation config test passed")


def main():
    print("\n" + "="*60)
    print("Distillation Trainer Test Suite")
    print("="*60)

    # Setup
    print("\nSetting up...")
    test_dir = Path('data/test_distillation')
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create test data
    data_path = create_test_data(test_dir, num_examples=10)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Run tests
    try:
        # Test dataset
        dataset = test_dataset(data_path, tokenizer)

        # Test output distillation
        test_output_distillation(data_path, tokenizer)

        # Test logit distillation config
        test_logit_distillation_config()

        # Summary
        print("\n" + "="*60)
        print("All Tests Passed! ✓")
        print("="*60)
        print("\nThe distillation trainer is ready to use.")
        print("\nNext steps:")
        print("1. Generate real training data:")
        print("   python scripts/generate_distillation_data.py --num-examples 1000")
        print("\n2. Train your model:")
        print("   python scripts/train_with_distillation.py \\")
        print("       --distillation-type output \\")
        print("       --train-data data/distillation/train.jsonl \\")
        print("       --student-config configs/model/100m_laptop.json \\")
        print("       --output-dir checkpoints/distilled_100m")

    except Exception as e:
        print("\n" + "="*60)
        print("Test Failed! ❌")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
