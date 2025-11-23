"""Tests for model configuration."""

import pytest
from src.model.config import DemiurgicConfig


class TestDemiurgicConfig:
    """Test model configuration."""

    def test_default_config(self):
        """Test that default config can be created."""
        config = DemiurgicConfig()
        assert config.vocab_size == 32000
        assert config.hidden_size == 4096
        assert config.num_hidden_layers == 32

    def test_custom_config(self, tiny_config):
        """Test custom config creation."""
        assert tiny_config.vocab_size == 1000
        assert tiny_config.hidden_size == 128
        assert tiny_config.num_hidden_layers == 2

    def test_head_dim_property(self, tiny_config):
        """Test head_dim calculation."""
        expected_head_dim = tiny_config.hidden_size // tiny_config.num_attention_heads
        assert tiny_config.head_dim == expected_head_dim
        assert tiny_config.head_dim == 32  # 128 / 4

    def test_gqa_validation(self):
        """Test Grouped-Query Attention validation."""
        # Should work: num_attention_heads divisible by num_key_value_heads
        config = DemiurgicConfig(
            num_attention_heads=16,
            num_key_value_heads=4,
        )
        assert config.num_key_value_heads == 4

    def test_invalid_head_configuration(self):
        """Test that invalid head configuration raises error."""
        with pytest.raises(ValueError, match="must be divisible"):
            DemiurgicConfig(
                hidden_size=128,
                num_attention_heads=5,  # 128 not divisible by 5
            )

    def test_to_dict(self, tiny_config):
        """Test config serialization to dict."""
        config_dict = tiny_config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['vocab_size'] == 1000
        assert config_dict['hidden_size'] == 128

    def test_from_dict(self, tiny_config):
        """Test config creation from dict."""
        config_dict = tiny_config.to_dict()
        new_config = DemiurgicConfig.from_dict(config_dict)
        assert new_config.vocab_size == tiny_config.vocab_size
        assert new_config.hidden_size == tiny_config.hidden_size


class TestPresetConfigs:
    """Test preset configurations."""

    def test_100m_config(self):
        """Test 100M parameter configuration."""
        from src.model.config import get_100m_config
        config = get_100m_config()
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.use_flash_attention_2 is False  # For CPU

    def test_1b_config(self):
        """Test 1B parameter configuration."""
        from src.model.config import get_1b_config
        config = get_1b_config()
        assert config.hidden_size == 2048
        assert config.num_hidden_layers == 24

    def test_7b_config(self):
        """Test 7B parameter configuration."""
        from src.model.config import get_7b_config
        config = get_7b_config()
        assert config.hidden_size == 4096
        assert config.num_hidden_layers == 32
