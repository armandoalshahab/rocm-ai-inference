"""
Tests for configuration module.
"""

import pytest
from rocm_inference.config import ModelConfig


class TestModelConfig:
    """Test ModelConfig class."""
    
    def test_valid_config(self):
        """Test creating a valid configuration."""
        config = ModelConfig(
            model_name="test-model",
            precision="fp16",
            max_batch_size=8,
        )
        assert config.model_name == "test-model"
        assert config.precision == "fp16"
        assert config.max_batch_size == 8
    
    def test_invalid_precision(self):
        """Test that invalid precision raises error."""
        with pytest.raises(ValueError, match="Invalid precision"):
            ModelConfig(model_name="test", precision="invalid")
    
    def test_invalid_batch_size(self):
        """Test that invalid batch size raises error."""
        with pytest.raises(ValueError, match="max_batch_size must be >= 1"):
            ModelConfig(model_name="test", max_batch_size=0)
    
    def test_invalid_sequence_length(self):
        """Test that invalid sequence length raises error."""
        with pytest.raises(ValueError, match="max_sequence_length must be >= 1"):
            ModelConfig(model_name="test", max_sequence_length=-1)
    
    def test_torch_dtype_fp16(self):
        """Test torch dtype for fp16."""
        import torch
        config = ModelConfig(model_name="test", precision="fp16")
        assert config.torch_dtype == torch.float16
    
    def test_torch_dtype_bf16(self):
        """Test torch dtype for bf16."""
        import torch
        config = ModelConfig(model_name="test", precision="bf16")
        assert config.torch_dtype == torch.bfloat16
    
    def test_is_quantized(self):
        """Test quantization detection."""
        config_int8 = ModelConfig(model_name="test", precision="int8")
        config_fp16 = ModelConfig(model_name="test", precision="fp16")
        
        assert config_int8.is_quantized is True
        assert config_fp16.is_quantized is False
