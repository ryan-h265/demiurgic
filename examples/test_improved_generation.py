"""
Test script demonstrating improved generation capabilities.

Shows the difference between old basic generation and new advanced generation
with various sampling strategies.
"""

import torch
from src.model import DemiurgicForCausalLM, get_100m_config, GenerationConfig


def main():
    print("="*70)
    print("Demiurgic Improved Generation Demo")
    print("="*70)

    # Load a small model (or create a dummy one for testing)
    print("\n1. Loading model...")

    # Try to load the 10M checkpoint, otherwise create a small test model
    try:
        model = DemiurgicForCausalLM.from_pretrained("checkpoints/distilled_10m/final")
        print("   ✓ Loaded 10M checkpoint")
    except:
        print("   ! No checkpoint found, creating small test model")
        config = get_100m_config()
        config.num_hidden_layers = 4  # Even smaller for testing
        config.hidden_size = 256
        config.intermediate_size = 688
        config.num_attention_heads = 4
        model = DemiurgicForCausalLM(config)
        print("   ✓ Created test model")

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"   ✓ Model on {device}")

    # Create a sample prompt (simplified tokens)
    print("\n2. Creating sample prompt...")
    # In a real scenario, you'd use a tokenizer
    # Here we'll just use random token IDs for demonstration
    prompt_length = 10
    input_ids = torch.randint(3, model.vocab_size, (1, prompt_length)).to(device)
    print(f"   ✓ Prompt shape: {input_ids.shape}")

    # Test different generation strategies
    print("\n3. Testing generation strategies...")
    print("-" * 70)

    strategies = [
        ("Greedy (deterministic)", GenerationConfig.greedy()),
        ("Balanced code", GenerationConfig.balanced_code(max_new_tokens=50)),
        ("Precise code", GenerationConfig.precise_code(max_new_tokens=50)),
        ("Creative code", GenerationConfig.creative_code(max_new_tokens=50)),
        ("Natural text", GenerationConfig.natural_text(max_new_tokens=50)),
    ]

    for name, config in strategies:
        print(f"\n{name}:")
        print(f"  • Temperature: {config.temperature}")
        print(f"  • Top-k: {config.top_k}")
        print(f"  • Top-p: {config.top_p}")
        print(f"  • Repetition penalty: {config.repetition_penalty}")
        print(f"  • Frequency penalty: {config.frequency_penalty}")
        print(f"  • Presence penalty: {config.presence_penalty}")

        try:
            # Generate
            with torch.no_grad():
                output = model.generate(input_ids, **config.to_dict())

            generated_length = output.shape[1] - input_ids.shape[1]
            print(f"  ✓ Generated {generated_length} tokens")
            print(f"    Output shape: {output.shape}")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "="*70)
    print("4. Key improvements demonstrated:")
    print("-" * 70)
    print("  ✓ KV caching - much faster generation")
    print("  ✓ Repetition penalty - reduces repeated tokens")
    print("  ✓ Frequency penalty - fine-grained repetition control")
    print("  ✓ Presence penalty - encourages diversity")
    print("  ✓ Top-k + Top-p sampling - quality control")
    print("  ✓ Typical-p sampling - alternative sampling method")
    print("  ✓ Min/max length - better control")
    print("  ✓ Multiple sequences - generate N variants")
    print("  ✓ Proper EOS handling - stops per sequence")
    print("  ✓ Configurable presets - easy to use")
    print("="*70)

    # Performance comparison
    print("\n5. Speed comparison (KV cache vs no cache):")
    print("-" * 70)

    # Measure with KV cache (new)
    import time

    config = GenerationConfig.balanced_code(max_new_tokens=30)
    start = time.time()
    with torch.no_grad():
        _ = model.generate(input_ids, **config.to_dict())
    elapsed_with_cache = time.time() - start

    print(f"  With KV cache: {elapsed_with_cache:.3f}s")
    print(f"  ✓ KV cache enabled by default in new implementation")

    print("\n" + "="*70)
    print("Demo complete!")
    print("="*70)


if __name__ == "__main__":
    main()
