"""Pytest configuration and shared fixtures."""

import pytest
import torch
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.config import DemiurgicConfig
from src.model.model import DemiurgicForCausalLM


@pytest.fixture
def tiny_config():
    """Create a tiny model config for fast testing."""
    return DemiurgicConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        use_flash_attention_2=False,  # Disable for CPU testing
    )


@pytest.fixture
def small_config():
    """Create a small model config for testing."""
    return DemiurgicConfig(
        vocab_size=5000,
        hidden_size=256,
        intermediate_size=688,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=1024,
        use_flash_attention_2=False,
    )


@pytest.fixture
def tiny_model(tiny_config):
    """Create a tiny model instance."""
    return DemiurgicForCausalLM(tiny_config)


@pytest.fixture
def sample_input():
    """Create sample input tensor."""
    return torch.randint(0, 1000, (2, 10))  # batch_size=2, seq_len=10


@pytest.fixture
def sample_attention_mask():
    """Create sample attention mask."""
    return torch.ones(2, 10, dtype=torch.long)


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for tests."""
    return tmp_path
