"""
ROCm AI Inference Engine
========================

High-performance AI inference pipeline optimized for AMD Instinct GPUs.

Example:
    >>> from rocm_inference import InferenceEngine, ModelConfig
    >>> config = ModelConfig(model_name="meta-llama/Llama-2-7b-hf", precision="fp16")
    >>> engine = InferenceEngine(config)
    >>> result = engine.generate("Hello, world!")
    >>> print(result.text)
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from rocm_inference.engine import InferenceEngine
from rocm_inference.config import ModelConfig
from rocm_inference.batch import BatchProcessor
from rocm_inference.benchmark import BenchmarkSuite

__all__ = [
    "InferenceEngine",
    "ModelConfig",
    "BatchProcessor",
    "BenchmarkSuite",
]
