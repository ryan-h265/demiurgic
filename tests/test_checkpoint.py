"""Tests for model checkpoint save/load functionality."""

import pytest
import torch
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.model import DemiurgicForCausalLM
from src.model.config import DemiurgicConfig


class TestCheckpointSaveLoad:
    """Test saving and loading model checkpoints."""

    @pytest.fixture
    def tiny_model(self):
        """Create a tiny model for testing."""
        config = DemiurgicConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=512,
            use_flash_attention_2=False,
        )
        return DemiurgicForCausalLM(config)

    def test_save_pretrained(self, tiny_model, temp_dir):
        """Test saving model with save_pretrained."""
        save_path = temp_dir / "test_model"

        tiny_model.save_pretrained(str(save_path))

        # Check that files were created
        assert (save_path / "pytorch_model.bin").exists()
        assert (save_path / "config.json").exists()

    def test_load_pretrained(self, tiny_model, temp_dir):
        """Test loading model with from_pretrained."""
        save_path = temp_dir / "test_model"

        # Save model
        tiny_model.save_pretrained(str(save_path))

        # Load model
        loaded_model = DemiurgicForCausalLM.from_pretrained(str(save_path))

        assert loaded_model is not None
        assert loaded_model.config.vocab_size == tiny_model.config.vocab_size
        assert loaded_model.config.hidden_size == tiny_model.config.hidden_size

    def test_save_load_weights_match(self, tiny_model, temp_dir):
        """Test that loaded weights match saved weights."""
        save_path = temp_dir / "test_model"

        # Get initial output
        input_ids = torch.randint(0, 1000, (1, 10))
        with torch.no_grad():
            initial_output = tiny_model(input_ids)
            initial_logits = initial_output[1]

        # Save model
        tiny_model.save_pretrained(str(save_path))

        # Load model
        loaded_model = DemiurgicForCausalLM.from_pretrained(str(save_path))

        # Get loaded model output
        with torch.no_grad():
            loaded_output = loaded_model(input_ids)
            loaded_logits = loaded_output[1]

        # Check outputs match
        assert torch.allclose(initial_logits, loaded_logits, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
