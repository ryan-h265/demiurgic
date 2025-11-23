# Demiurgic Test Suite

Comprehensive test suite for the Demiurgic code model project.

## Test Structure

```
tests/
├── conftest.py              # Shared pytest fixtures
├── test_model_config.py     # Configuration tests
├── test_model.py            # Model architecture and forward pass tests
├── test_checkpoint.py       # Save/load checkpoint tests
└── README.md                # This file
```

## Running Tests

### Quick Start

```bash
# Run all tests
./run_tests.sh

# Run with verbose output
./run_tests.sh -v

# Run with coverage report
./run_tests.sh --coverage

# Run only quick tests (exclude slow ones)
./run_tests.sh --quick
```

### Using pytest directly

```bash
# Activate environment
source venv/bin/activate

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_model.py

# Run specific test class
pytest tests/test_model.py::TestModelForward

# Run specific test
pytest tests/test_model.py::TestModelForward::test_forward_pass

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Test Categories

### Configuration Tests (`test_model_config.py`)
- Default and custom configuration creation
- Configuration validation
- Preset configurations (100M, 1B, 7B)
- Configuration serialization (to_dict, from_dict)
- GQA (Grouped-Query Attention) configuration

### Model Tests (`test_model.py`)
- Model initialization
- Parameter counting
- Forward pass with various inputs
- Shape verification
- Gradient flow
- Attention masks
- Different sequence lengths
- Generation capabilities
- Individual components (RoPE, RMSNorm, SwiGLU)

### Checkpoint Tests (`test_checkpoint.py`)
- Saving models with `save_pretrained()`
- Loading models with `from_pretrained()`
- Weight preservation across save/load
- Saving after training
- Saving with resized embeddings
- Configuration serialization

## Writing New Tests

### Using Fixtures

The test suite provides several useful fixtures in `conftest.py`:

```python
def test_my_feature(tiny_model, sample_input):
    """Test using pre-configured fixtures."""
    outputs = tiny_model(sample_input)
    assert outputs is not None
```

Available fixtures:
- `tiny_config` - Minimal config for fast tests
- `small_config` - Small config for testing
- `tiny_model` - Pre-initialized tiny model
- `sample_input` - Sample input tensor
- `sample_attention_mask` - Sample attention mask
- `temp_dir` - Temporary directory for file tests

### Test Organization

Follow this structure for new test files:

```python
"""Brief description of what this file tests."""

import pytest
from src.model.model import DemiurgicForCausalLM


class TestFeatureCategory:
    """Test a category of related features."""

    def test_specific_behavior(self):
        """Test a specific behavior."""
        # Arrange
        model = create_model()

        # Act
        result = model.do_something()

        # Assert
        assert result == expected
```

## Continuous Integration

The test suite is designed to run in CI/CD environments:

```yaml
# Example GitHub Actions workflow
steps:
  - name: Run tests
    run: |
      source venv/bin/activate
      pytest tests/ --cov=src --cov-report=xml
```

## Performance Considerations

- Tests use tiny models (128-256 hidden size) for speed
- Gradient checkpointing is disabled in tests
- Flash Attention is disabled for CPU compatibility
- Most tests run in < 1 second
- Full test suite completes in < 30 seconds

## Troubleshooting

### Import Errors

If you see import errors, ensure the project root is in your Python path:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
pytest tests/
```

### CUDA/GPU Errors

Tests are designed to run on CPU. If you see CUDA errors:

```bash
# Force CPU-only mode
export CUDA_VISIBLE_DEVICES=""
pytest tests/
```

### Slow Tests

If tests are slow:

```bash
# Run only fast tests
pytest tests/ -m "not slow"

# Or use the quick mode
./run_tests.sh --quick
```

## Coverage Goals

Target coverage levels:
- Overall: > 80%
- Core model code (`src/model/`): > 90%
- Training code (`src/distillation/`): > 70%
- Utilities: > 60%

Generate coverage report:
```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```
