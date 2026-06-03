"""
Pytest configuration and fixtures.
"""

import pytest
from rocm_inference.config import ModelConfig


@pytest.fixture
def model_config():
    """Create a test model configuration."""
    return ModelConfig(
        model_name="gpt2",  # Small model for testing
        precision="fp32",
        max_batch_size=2,
        max_sequence_length=128,
        device="cpu",  # Use CPU for CI tests
        torch_compile=False,
        flash_attention=False,
    )
