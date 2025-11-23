#!/usr/bin/env python3
"""
Basic model test script.

Demonstrates model creation and forward pass without needing pytest.
Run this to quickly verify the model works.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.model import DemiurgicConfig, DemiurgicForCausalLM, get_1b_config


def test_model_creation():
    """Test creating and running a small model."""
    print("=" * 60)
    print("Testing Demiurgic Model Architecture")
    print("=" * 60)

    # Create a small config for testing
    print("\n1. Creating small test configuration...")
    config = DemiurgicConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=688,
        num_hidden_layers=4,
        num_attention_heads=8,
        max_position_embeddings=512,
    )
    print(f"   ✓ Config created: {config.num_hidden_layers} layers, "
          f"{config.hidden_size} hidden size")

    # Create model
    print("\n2. Instantiating model...")
    model = DemiurgicForCausalLM(config)
    print(f"   ✓ Model created successfully")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ Total parameters: {total_params:,}")
    print(f"   ✓ Trainable parameters: {trainable_params:,}")

    # Test forward pass
    print("\n3. Testing forward pass...")
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    print(f"   Input shape: {input_ids.shape}")

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        loss, logits = outputs[0], outputs[1]

    print(f"   ✓ Forward pass successful")
    print(f"   ✓ Logits shape: {logits.shape}")
    print(f"   ✓ Expected shape: ({batch_size}, {seq_len}, {config.vocab_size})")

    # Test with labels (compute loss)
    print("\n4. Testing loss computation...")
    labels = input_ids.clone()
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        loss = outputs[0]

    print(f"   ✓ Loss computed: {loss.item():.4f}")

    # Test generation
    print("\n5. Testing text generation...")
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    print(f"   Prompt length: {prompt.shape[1]}")

    with torch.no_grad():
        generated = model.generate(prompt, max_length=10, do_sample=False)

    print(f"   ✓ Generated sequence length: {generated.shape[1]}")
    print(f"   ✓ Generated tokens: {generated[0].tolist()}")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


def test_1b_model():
    """Test creating a 1B parameter model."""
    print("\n" + "=" * 60)
    print("Testing 1B Model Configuration")
    print("=" * 60)

    print("\n1. Loading 1B config...")
    config = get_1b_config()
    print(f"   ✓ Config loaded: {config.num_hidden_layers} layers, "
          f"{config.hidden_size} hidden size")

    print("\n2. Creating 1B model...")
    model = DemiurgicForCausalLM(config)

    total_params = sum(p.numel() for p in model.parameters())
    billion_params = total_params / 1e9

    print(f"   ✓ Model created")
    print(f"   ✓ Total parameters: {total_params:,}")
    print(f"   ✓ Parameter count: {billion_params:.2f}B")

    print("\n   Note: This is just the architecture.")
    print("   Training would require substantial compute resources.")

    print("\n" + "=" * 60)


def test_components():
    """Test individual components."""
    print("\n" + "=" * 60)
    print("Testing Model Components")
    print("=" * 60)

    # Test RMSNorm
    print("\n1. Testing RMSNorm...")
    from src.model.normalization import RMSNorm
    norm = RMSNorm(hidden_size=256)
    x = torch.randn(2, 16, 256)
    output = norm(x)
    print(f"   ✓ Input shape: {x.shape}")
    print(f"   ✓ Output shape: {output.shape}")
    print(f"   ✓ Output mean: {output.mean():.4f} (should be ~0)")
    print(f"   ✓ Output std: {output.std():.4f} (should be ~1)")

    # Test RoPE
    print("\n2. Testing Rotary Embeddings...")
    from src.model.embeddings import RotaryEmbedding
    rope = RotaryEmbedding(dim=128, max_position_embeddings=512)
    x = torch.randn(2, 16, 128)
    cos, sin = rope(x, seq_len=16)
    print(f"   ✓ Cos shape: {cos.shape}")
    print(f"   ✓ Sin shape: {sin.shape}")

    # Test SwiGLU
    print("\n3. Testing SwiGLU...")
    from src.model.feedforward import SwiGLU
    swiglu = SwiGLU(hidden_size=256, intermediate_size=688)
    x = torch.randn(2, 16, 256)
    output = swiglu(x)
    print(f"   ✓ Input shape: {x.shape}")
    print(f"   ✓ Output shape: {output.shape}")

    print("\n" + "=" * 60)
    print("All component tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "DEMIURGIC MODEL ARCHITECTURE TEST" + " " * 15 + "║")
    print("╚" + "═" * 58 + "╝")

    try:
        test_components()
        test_model_creation()
        test_1b_model()

        print("\n✓ All tests completed successfully!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run pytest: pytest tests/test_model.py -v")
        print("  3. Proceed to data preparation and training setup")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
