"""
Model and inference configuration.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import os


@dataclass
class ModelConfig:
    """Configuration for model loading and inference.
    
    Attributes:
        model_name: HuggingFace model name or local path
        precision: Compute precision (fp16, bf16, int8)
        max_batch_size: Maximum batch size for inference
        max_sequence_length: Maximum input sequence length
        device: Device to use (cuda for ROCm)
        torch_compile: Enable torch.compile optimization
        flash_attention: Use Flash Attention 2 if available
        kv_cache: Enable KV caching for generation
        device_ids: GPU device IDs to use
    """
    model_name: str
    precision: str = "fp16"
    max_batch_size: int = 8
    max_sequence_length: int = 2048
    device: str = "cuda"
    torch_compile: bool = True
    flash_attention: bool = True
    kv_cache: bool = True
    device_ids: Optional[List[int]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_precisions = ["fp16", "bf16", "int8", "fp32"]
        if self.precision not in valid_precisions:
            raise ValueError(f"Invalid precision: {self.precision}. Must be one of {valid_precisions}")
        
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        
        if self.max_sequence_length < 1:
            raise ValueError("max_sequence_length must be >= 1")
        
        # Auto-detect ROCm
        if self.device == "cuda" and os.environ.get("ROCM_PATH"):
            self._is_rocm = True
        else:
            self._is_rocm = False
    
    @property
    def torch_dtype(self):
        """Get PyTorch dtype from precision string."""
        import torch
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "int8": torch.int8,
            "fp32": torch.float32,
        }
        return dtype_map[self.precision]
    
    @property
    def is_quantized(self) -> bool:
        """Check if using quantized inference."""
        return self.precision == "int8"
