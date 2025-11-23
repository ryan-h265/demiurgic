#!/usr/bin/env python3
"""
Laptop-friendly testing script for Demiurgic model.

This script tests model creation, forward pass, and generation
with memory profiling to help understand resource requirements.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import psutil
import os
from src.model import (
    DemiurgicForCausalLM,
    get_100m_config,
    get_350m_config,
    get_1b_config,
)


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


def format_size(bytes_size):
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def test_model(config_fn, name, batch_size=1, seq_len=32):
    """Test a model configuration with memory profiling."""
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print(f"{'='*60}")

    # Get starting memory
    mem_before = get_memory_usage()
    print(f"\n1. Memory before model creation: {mem_before:.1f} MB")

    # Create model
    print(f"\n2. Creating model...")
    config = config_fn()
    model = DemiurgicForCausalLM(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())

    mem_after_model = get_memory_usage()
    mem_model = mem_after_model - mem_before

    print(f"   ✓ Model created")
    print(f"   ✓ Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"   ✓ Trainable: {trainable_params:,}")
    print(f"   ✓ Model size: {format_size(param_size)}")
    print(f"   ✓ Memory used: {mem_model:.1f} MB")

    # Test forward pass
    print(f"\n3. Testing forward pass (batch={batch_size}, seq_len={seq_len})...")
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    mem_before_forward = get_memory_usage()

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs[1]

    mem_after_forward = get_memory_usage()
    mem_forward = mem_after_forward - mem_before_forward

    print(f"   ✓ Forward pass successful")
    print(f"   ✓ Input shape: {input_ids.shape}")
    print(f"   ✓ Output shape: {logits.shape}")
    print(f"   ✓ Memory for forward pass: {mem_forward:.1f} MB")

    # Test loss computation
    print(f"\n4. Testing loss computation...")
    labels = input_ids.clone()

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        loss = outputs[0]

    print(f"   ✓ Loss computed: {loss.item():.4f}")

    # Test generation
    print(f"\n5. Testing generation...")
    prompt = torch.randint(0, config.vocab_size, (1, 10))

    mem_before_gen = get_memory_usage()

    with torch.no_grad():
        generated = model.generate(prompt, max_length=20, do_sample=False)

    mem_after_gen = get_memory_usage()

    print(f"   ✓ Generated {generated.shape[1]} tokens")
    print(f"   ✓ Peak memory during generation: {mem_after_gen:.1f} MB")

    # Summary
    mem_final = get_memory_usage()
    print(f"\n6. Memory Summary:")
    print(f"   • Total memory used: {mem_final:.1f} MB")
    print(f"   • Peak memory: {max(mem_after_model, mem_after_forward, mem_after_gen):.1f} MB")

    # Cleanup
    del model, input_ids, logits, labels, outputs, generated
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'name': name,
        'params': total_params,
        'model_size_mb': param_size / 1024 / 1024,
        'peak_memory_mb': max(mem_after_model, mem_after_forward, mem_after_gen),
    }


def check_system_resources():
    """Check available system resources."""
    print("\n" + "="*60)
    print("System Resources")
    print("="*60)

    # RAM
    mem = psutil.virtual_memory()
    print(f"\nRAM:")
    print(f"  • Total: {mem.total / 1024**3:.1f} GB")
    print(f"  • Available: {mem.available / 1024**3:.1f} GB")
    print(f"  • Used: {mem.used / 1024**3:.1f} GB ({mem.percent}%)")

    # CPU
    print(f"\nCPU:")
    print(f"  • Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    print(f"  • Usage: {psutil.cpu_percent(interval=1)}%")

    # GPU
    print(f"\nGPU:")
    if torch.cuda.is_available():
        print(f"  • CUDA available: Yes")
        print(f"  • Device: {torch.cuda.get_device_name(0)}")
        print(f"  • CUDA version: {torch.version.cuda}")

        # GPU memory
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        print(f"  • GPU memory: {gpu_mem / 1024**3:.1f} GB")
    else:
        print(f"  • CUDA available: No (CPU-only mode)")

    # Recommendations
    print(f"\nRecommendations:")
    available_gb = mem.available / 1024**3

    if available_gb < 4:
        print(f"  ⚠️  Low memory ({available_gb:.1f} GB available)")
        print(f"     Recommended: Close other applications")
        print(f"     Max model: Very small models only")
    elif available_gb < 8:
        print(f"  ℹ️  Moderate memory ({available_gb:.1f} GB available)")
        print(f"     Max model: 100M-350M parameters")
    elif available_gb < 16:
        print(f"  ✓ Good memory ({available_gb:.1f} GB available)")
        print(f"     Max model: 350M-1B parameters (inference)")
    else:
        print(f"  ✓ Excellent memory ({available_gb:.1f} GB available)")
        print(f"     Max model: 1B+ parameters (inference)")


def main():
    """Main test function."""
    print("\n")
    print("╔" + "═"*58 + "╗")
    print("║" + " "*15 + "LAPTOP TESTING MODE" + " "*24 + "║")
    print("╚" + "═"*58 + "╝")

    # Check system resources
    check_system_resources()

    # Test models
    print("\n" + "="*60)
    print("Model Testing")
    print("="*60)
    print("\nTesting different model sizes to see what fits in memory...")

    results = []

    # Test 100M model
    try:
        result = test_model(get_100m_config, "100M Model (Laptop-Friendly)", batch_size=1, seq_len=32)
        results.append(result)
    except Exception as e:
        print(f"\n❌ 100M model failed: {e}")

    # Test 350M model
    try:
        result = test_model(get_350m_config, "350M Model (Laptop Training)", batch_size=1, seq_len=32)
        results.append(result)
    except Exception as e:
        print(f"\n❌ 350M model failed: {e}")

    # Test 1B model (inference only)
    mem = psutil.virtual_memory()
    if mem.available > 8 * 1024**3:  # More than 8GB available
        try:
            result = test_model(get_1b_config, "1B Model (Inference Only)", batch_size=1, seq_len=16)
            results.append(result)
        except Exception as e:
            print(f"\n❌ 1B model failed: {e}")
    else:
        print(f"\n⏭️  Skipping 1B model (insufficient memory)")

    # Summary
    if results:
        print(f"\n" + "="*60)
        print("Summary")
        print("="*60)
        print(f"\n{'Model':<30} {'Parameters':<15} {'Size':<12} {'Peak RAM':<12}")
        print("-"*60)

        for r in results:
            params_str = f"{r['params']/1e6:.1f}M"
            size_str = f"{r['model_size_mb']:.0f} MB"
            mem_str = f"{r['peak_memory_mb']:.0f} MB"
            print(f"{r['name']:<30} {params_str:<15} {size_str:<12} {mem_str:<12}")

    # Recommendations
    print(f"\n" + "="*60)
    print("What Can You Do On This Laptop?")
    print("="*60)

    mem = psutil.virtual_memory()
    available_gb = mem.available / 1024**3

    print(f"\n✓ Development & Testing:")
    print(f"  • Code the model architecture ✓")
    print(f"  • Test forward/backward passes ✓")
    print(f"  • Experiment with small models ✓")
    print(f"  • Debug and validate code ✓")

    print(f"\n✓ Inference (Running trained models):")
    if available_gb >= 16:
        print(f"  • 100M-350M models: Fast ✓")
        print(f"  • 1B model: Possible ✓")
        print(f"  • 7B+ models: Need quantization")
    elif available_gb >= 8:
        print(f"  • 100M-350M models: Yes ✓")
        print(f"  • 1B model: Slow but possible")
        print(f"  • 7B+ models: Not recommended")
    else:
        print(f"  • 100M model: Yes ✓")
        print(f"  • 350M+ models: Challenging")

    print(f"\n⚠️  Training:")
    if available_gb >= 16:
        print(f"  • 100M model: Yes, ~1-2 hours per epoch on small dataset")
        print(f"  • 350M model: Very slow, not practical")
        print(f"  • 1B+ models: Not possible on laptop")
    elif available_gb >= 8:
        print(f"  • 100M model: Possible but slow")
        print(f"  • 350M+ models: Not practical")
    else:
        print(f"  • Training: Not recommended (insufficient RAM)")

    print(f"\n💡 Recommendations:")
    print(f"  • Use this laptop for development and testing")
    print(f"  • Train small models (100M-350M) to validate pipeline")
    print(f"  • Use cloud compute for larger models (7B+)")
    print(f"  • Consider Google Colab for free GPU access")

    print(f"\n" + "="*60)
    print("Next Steps")
    print("="*60)
    print(f"""
1. Develop on laptop (you can do this now ✓):
   - Code model architecture
   - Test with 100M-350M models
   - Validate training pipeline
   - Debug and iterate

2. Small-scale training:
   - Train 100M model on laptop (slow but possible)
   - Validate entire training workflow
   - Test evaluation metrics

3. Production training:
   - Use cloud GPU (AWS/GCP) for 7B+ models
   - Or use Colab Pro for medium models
   - Or wait until you have access to better hardware

Your laptop is perfect for development! 🚀
""")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
