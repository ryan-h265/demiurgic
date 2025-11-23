#!/usr/bin/env python3
"""Quick test of save_pretrained and from_pretrained."""

import torch
import tempfile
from pathlib import Path
from src.model.model import DemiurgicForCausalLM
from src.model.config import DemiurgicConfig

# Create a small test model
config = DemiurgicConfig(
    vocab_size=1000,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    max_position_embeddings=512,
)

print("Creating model...")
model = DemiurgicForCausalLM(config)
print(f"Model parameters: {model.num_parameters() / 1e6:.1f}M")

# Create some dummy data and get initial output
input_ids = torch.randint(0, 1000, (1, 10))
print("\nGetting initial output...")
with torch.no_grad():
    initial_output = model(input_ids)
    initial_logits = initial_output[1][:, 0, :5]  # First 5 logits of first token
    print(f"Initial logits sample: {initial_logits}")

# Save the model
with tempfile.TemporaryDirectory() as tmpdir:
    save_path = Path(tmpdir) / "test_model"
    print(f"\nSaving model to {save_path}...")
    model.save_pretrained(save_path)

    # Check files were created
    print("\nFiles created:")
    for f in save_path.iterdir():
        print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")

    # Load the model back
    print("\nLoading model from checkpoint...")
    loaded_model = DemiurgicForCausalLM.from_pretrained(save_path)
    print(f"Loaded model parameters: {loaded_model.num_parameters() / 1e6:.1f}M")

    # Get output from loaded model
    print("\nGetting loaded model output...")
    with torch.no_grad():
        loaded_output = model(input_ids)
        loaded_logits = loaded_output[1][:, 0, :5]
        print(f"Loaded logits sample: {loaded_logits}")

    # Check they match
    diff = torch.abs(initial_logits - loaded_logits).max().item()
    print(f"\nMax difference: {diff}")
    if diff < 1e-5:
        print("✅ SUCCESS: Model save/load works perfectly!")
    else:
        print("❌ FAILED: Outputs don't match!")

print("\n✅ Test completed successfully!")
