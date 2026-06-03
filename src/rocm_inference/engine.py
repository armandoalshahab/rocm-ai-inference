"""
Core inference engine with ROCm optimization.
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional, List, Union
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rocm_inference.config import ModelConfig
from rocm_inference.utils.gpu_utils import get_gpu_info, setup_rocm_env

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result from a single inference call.
    
    Attributes:
        text: Generated text output
        tokens_generated: Number of tokens generated
        latency_ms: Total latency in milliseconds
        tokens_per_second: Throughput metric
        prompt_tokens: Number of input tokens
    """
    text: str
    tokens_generated: int
    latency_ms: float
    tokens_per_second: float
    prompt_tokens: int


class InferenceEngine:
    """High-performance inference engine for AMD ROCm GPUs.
    
    This engine provides optimized inference for large language models
    on AMD Instinct GPUs using ROCm and PyTorch.
    
    Example:
        >>> config = ModelConfig(model_name="meta-llama/Llama-2-7b-hf", precision="fp16")
        >>> engine = InferenceEngine(config)
        >>> result = engine.generate("Explain quantum computing")
        >>> print(result.text)
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize the inference engine.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self._model = None
        self._tokenizer = None
        self._device = None
        self._setup()
    
    def _setup(self):
        """Setup device and load model."""
        # Setup ROCm environment
        setup_rocm_env()
        
        # Determine device
        if torch.cuda.is_available():
            self._device = torch.device(self.config.device)
            gpu_info = get_gpu_info()
            logger.info(f"Using GPU: {gpu_info.get('name', 'Unknown')}")
            logger.info(f"GPU Memory: {gpu_info.get('memory_total', 0) / 1e9:.1f} GB")
        else:
            logger.warning("No GPU available, falling back to CPU")
            self._device = torch.device("cpu")
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer from HuggingFace or local path."""
        logger.info(f"Loading model: {self.config.model_name}")
        logger.info(f"Precision: {self.config.precision}")
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Load model with appropriate settings
        load_kwargs = {
            "torch_dtype": self.config.torch_dtype,
            "device_map": "auto" if self._device.type == "cuda" else None,
            "trust_remote_code": True,
        }
        
        # Enable Flash Attention if requested
        if self.config.flash_attention and self._device.type == "cuda":
            load_kwargs["attn_implementation"] = "flash_attention_2"
        
        # Load with quantization if needed
        if self.config.is_quantized:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["quantization_config"] = quantization_config
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **load_kwargs
        )
        
        # Apply torch.compile if enabled
        if self.config.torch_compile and self._device.type == "cuda":
            logger.info("Applying torch.compile optimization...")
            self._model = torch.compile(self._model, mode="reduce-overhead")
        
        self._model.eval()
        logger.info("Model loaded successfully")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
    ) -> InferenceResult:
        """Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            repetition_penalty: Penalty for repetition
            
        Returns:
            InferenceResult with generated text and metrics
        """
        # Tokenize input
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_sequence_length - max_new_tokens,
        ).to(self._device)
        
        prompt_tokens = inputs["input_ids"].shape[1]
        
        # Generate
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                top_k=top_k if do_sample else 0,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        
        # Synchronize GPU
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        # Decode output
        generated_tokens = outputs[0][prompt_tokens:]
        generated_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Calculate metrics
        latency_ms = (end_time - start_time) * 1000
        tokens_generated = len(generated_tokens)
        tokens_per_second = tokens_generated / (latency_ms / 1000) if latency_ms > 0 else 0
        
        return InferenceResult(
            text=generated_text,
            tokens_generated=tokens_generated,
            latency_ms=latency_ms,
            tokens_per_second=tokens_per_second,
            prompt_tokens=prompt_tokens,
        )
    
    def embed(self, text: str) -> torch.Tensor:
        """Get embeddings for input text.
        
        Args:
            text: Input text
            
        Returns:
            Tensor of embeddings
        """
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_sequence_length,
        ).to(self._device)
        
        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)
            # Use last hidden state mean as embedding
            hidden_states = outputs.hidden_states[-1]
            embeddings = hidden_states.mean(dim=1)
        
        return embeddings.cpu()
    
    def unload(self):
        """Unload model and free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        
        logger.info("Model unloaded")
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
