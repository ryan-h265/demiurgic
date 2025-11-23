"""
Tests for Demiurgic model.

Verifies model architecture, forward pass, and shape correctness.
"""

import json
import torch
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import (
    DemiurgicConfig,
    DemiurgicModel,
    DemiurgicForCausalLM,
    get_1b_config,
    get_7b_config,
)


class TestConfiguration:
    """Test model configuration."""

    def test_config_creation(self):
        """Test basic config creation."""
        config = DemiurgicConfig()
        assert config.vocab_size == 32000
        assert config.hidden_size == 4096
        assert config.num_hidden_layers == 32

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "vocab_size": 50000,
            "hidden_size": 2048,
            "num_hidden_layers": 24,
        }
        config = DemiurgicConfig.from_dict(config_dict)
        assert config.vocab_size == 50000
        assert config.hidden_size == 2048
        assert config.num_hidden_layers == 24

    def test_config_validation(self):
        """Test config validation."""
        with pytest.raises(ValueError):
            # hidden_size not divisible by num_attention_heads
            DemiurgicConfig(hidden_size=4095, num_attention_heads=32)

    def test_predefined_configs(self):
        """Test predefined configurations."""
        config_1b = get_1b_config()
        assert config_1b.hidden_size == 2048
        assert config_1b.num_hidden_layers == 24

        config_7b = get_7b_config()
        assert config_7b.hidden_size == 4096
        assert config_7b.num_hidden_layers == 32

    def test_head_dim_property(self):
        """Test head_dim computation."""
        config = DemiurgicConfig(hidden_size=4096, num_attention_heads=32)
        assert config.head_dim == 128

    def test_gqa_config(self):
        """Test Grouped-Query Attention config."""
        config = DemiurgicConfig(
            num_attention_heads=32,
            num_key_value_heads=8,
        )
        assert config.num_key_value_heads == 8
        assert config.num_attention_heads % config.num_key_value_heads == 0


class TestModelArchitecture:
    """Test model architecture and shapes."""

    @pytest.fixture
    def small_config(self):
        """Small config for fast testing."""
        return DemiurgicConfig(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=688,
            num_hidden_layers=4,
            num_attention_heads=8,
            max_position_embeddings=512,
        )

    def test_model_creation(self, small_config):
        """Test model instantiation."""
        model = DemiurgicModel(small_config)
        assert model is not None
        assert len(model.layers) == 4
        assert model.vocab_size == 1000

    def test_causal_lm_creation(self, small_config):
        """Test causal LM model creation."""
        model = DemiurgicForCausalLM(small_config)
        assert model is not None
        assert model.lm_head is not None

    def test_forward_pass_shapes(self, small_config):
        """Test forward pass produces correct shapes."""
        model = DemiurgicModel(small_config)
        model.eval()

        batch_size = 2
        seq_len = 16

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        with torch.no_grad():
            outputs = model(input_ids)

        hidden_states = outputs[0]
        assert hidden_states.shape == (batch_size, seq_len, small_config.hidden_size)

    def test_causal_lm_forward(self, small_config):
        """Test causal LM forward pass."""
        model = DemiurgicForCausalLM(small_config)
        model.eval()

        batch_size = 2
        seq_len = 16

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        with torch.no_grad():
            outputs = model(input_ids)

        loss, logits = outputs[0], outputs[1]
        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)
        assert loss is None  # No labels provided

    def test_causal_lm_with_labels(self, small_config):
        """Test causal LM with loss computation."""
        model = DemiurgicForCausalLM(small_config)
        model.eval()

        batch_size = 2
        seq_len = 16

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        labels = input_ids.clone()

        with torch.no_grad():
            outputs = model(input_ids, labels=labels)

        loss, logits = outputs[0], outputs[1]
        assert loss is not None
        assert loss.item() > 0  # Loss should be positive
        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)

    def test_gradient_flow(self, small_config):
        """Test that gradients flow through the model."""
        model = DemiurgicForCausalLM(small_config)
        model.train()

        batch_size = 2
        seq_len = 16

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        labels = input_ids.clone()

        outputs = model(input_ids, labels=labels)
        loss = outputs[0]

        loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_different_sequence_lengths(self, small_config):
        """Test model with different sequence lengths."""
        model = DemiurgicModel(small_config)
        model.eval()

        for seq_len in [8, 16, 32, 64]:
            input_ids = torch.randint(0, 1000, (1, seq_len))

            with torch.no_grad():
                outputs = model(input_ids)

            hidden_states = outputs[0]
            assert hidden_states.shape == (1, seq_len, small_config.hidden_size)

    def test_attention_mask(self, small_config):
        """Test forward pass with attention mask."""
        model = DemiurgicModel(small_config)
        model.eval()

        batch_size = 2
        seq_len = 16

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 10:] = 0  # Mask last tokens of first sequence

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        hidden_states = outputs[0]
        assert hidden_states.shape == (batch_size, seq_len, small_config.hidden_size)


class TestGeneration:
    """Test text generation capabilities."""

    @pytest.fixture
    def small_config(self):
        return DemiurgicConfig(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=688,
            num_hidden_layers=4,
            num_attention_heads=8,
            max_position_embeddings=512,
        )

    def test_greedy_generation(self, small_config):
        """Test greedy generation."""
        model = DemiurgicForCausalLM(small_config)
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 5))

        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_length=10,
                do_sample=False,
            )

        assert generated.shape[1] >= input_ids.shape[1]
        assert generated.shape[1] <= input_ids.shape[1] + 10

    def test_sampling_generation(self, small_config):
        """Test sampling generation."""
        model = DemiurgicForCausalLM(small_config)
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 5))

        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_length=10,
                do_sample=True,
                temperature=0.8,
            )

        assert generated.shape[1] >= input_ids.shape[1]
        assert generated.shape[1] <= input_ids.shape[1] + 10


class TestModelComponents:
    """Test individual model components."""

    def test_embeddings_shape(self):
        """Test embedding layer shapes."""
        from src.model.embeddings import RotaryEmbedding

        rope = RotaryEmbedding(dim=128, max_position_embeddings=512)

        x = torch.randn(2, 16, 128)
        cos, sin = rope(x, seq_len=16)

        assert cos.shape == (16, 128)
        assert sin.shape == (16, 128)

    def test_rmsnorm(self):
        """Test RMSNorm layer."""
        from src.model.normalization import RMSNorm

        norm = RMSNorm(hidden_size=256)
        x = torch.randn(2, 16, 256)

        output = norm(x)

        assert output.shape == x.shape
        # Check that output has roughly zero mean and unit variance
        assert output.mean().abs() < 0.5
        assert (output.std() - 1.0).abs() < 0.5

    def test_swiglu(self):
        """Test SwiGLU activation."""
        from src.model.feedforward import SwiGLU

        swiglu = SwiGLU(hidden_size=256, intermediate_size=688)
        x = torch.randn(2, 16, 256)

        output = swiglu(x)

        assert output.shape == x.shape


class TestConfigFiles:
    """Test loading from config files."""

    def test_load_1b_config(self):
        """Test loading 1B config file."""
        config_path = Path(__file__).parent.parent / "configs" / "model" / "1b_test.json"

        with open(config_path) as f:
            config_dict = json.load(f)

        config = DemiurgicConfig.from_dict(config_dict)

        assert config.hidden_size == 2048
        assert config.num_hidden_layers == 24
        assert config.vocab_size == 32000

    def test_load_7b_config(self):
        """Test loading 7B config file."""
        config_path = Path(__file__).parent.parent / "configs" / "model" / "7b.json"

        with open(config_path) as f:
            config_dict = json.load(f)

        config = DemiurgicConfig.from_dict(config_dict)

        assert config.hidden_size == 4096
        assert config.num_hidden_layers == 32
        assert config.vocab_size == 32000


class TestParameterCount:
    """Test parameter counting."""

    def test_1b_model_params(self):
        """Test that 1B model has approximately 1B parameters."""
        config = get_1b_config()
        model = DemiurgicForCausalLM(config)

        total_params = sum(p.numel() for p in model.parameters())
        billion_params = total_params / 1e9

        # Should be close to 1B (within 0.5B)
        assert 0.8 < billion_params < 1.5

    def test_7b_model_params(self):
        """Test that 7B model has approximately 7B parameters."""
        config = get_7b_config()
        model = DemiurgicForCausalLM(config)

        total_params = sum(p.numel() for p in model.parameters())
        billion_params = total_params / 1e9

        # Should be close to 7B (within 1B)
        assert 6 < billion_params < 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
