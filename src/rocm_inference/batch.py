"""
Batch processing for high-throughput inference.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Callable

from rocm_inference.engine import InferenceEngine, InferenceResult

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result from batch processing.
    
    Attributes:
        results: List of individual inference results
        total_latency_ms: Total time for entire batch
        avg_latency_ms: Average latency per request
        total_tokens: Total tokens generated across all requests
        throughput: Overall throughput in tokens/second
    """
    results: List[InferenceResult]
    total_latency_ms: float
    avg_latency_ms: float
    total_tokens: int
    throughput: float


class BatchProcessor:
    """High-throughput batch processor for inference.
    
    Processes multiple prompts concurrently with configurable parallelism.
    
    Example:
        >>> engine = InferenceEngine(config)
        >>> processor = BatchProcessor(engine, max_concurrent=8)
        >>> batch_result = processor.process_batch(["prompt1", "prompt2", ...])
    """
    
    def __init__(
        self,
        engine: InferenceEngine,
        max_concurrent: int = 8,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """Initialize batch processor.
        
        Args:
            engine: Inference engine instance
            max_concurrent: Maximum concurrent requests
            progress_callback: Optional callback for progress updates (completed, total)
        """
        self.engine = engine
        self.max_concurrent = max_concurrent
        self.progress_callback = progress_callback
    
    def process_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> BatchResult:
        """Process a batch of prompts.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens per generation
            temperature: Sampling temperature
            **kwargs: Additional arguments passed to engine.generate()
            
        Returns:
            BatchResult with all results and metrics
        """
        import time
        
        results = []
        total_start = time.perf_counter()
        completed = 0
        
        # Process with thread pool for concurrent execution
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            futures = {
                executor.submit(
                    self.engine.generate,
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs,
                ): i
                for i, prompt in enumerate(prompts)
            }
            
            # Collect results in order
            ordered_results = [None] * len(prompts)
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    ordered_results[idx] = result
                except Exception as e:
                    logger.error(f"Failed to process prompt {idx}: {e}")
                    ordered_results[idx] = InferenceResult(
                        text=f"Error: {str(e)}",
                        tokens_generated=0,
                        latency_ms=0,
                        tokens_per_second=0,
                        prompt_tokens=0,
                    )
                
                completed += 1
                if self.progress_callback:
                    self.progress_callback(completed, len(prompts))
        
        results = ordered_results
        total_end = time.perf_counter()
        
        # Calculate batch metrics
        total_latency_ms = (total_end - total_start) * 1000
        total_tokens = sum(r.tokens_generated for r in results)
        avg_latency_ms = sum(r.latency_ms for r in results) / len(results) if results else 0
        throughput = total_tokens / (total_latency_ms / 1000) if total_latency_ms > 0 else 0
        
        logger.info(f"Batch complete: {len(results)} prompts, {total_tokens} tokens, {throughput:.1f} tok/s")
        
        return BatchResult(
            results=results,
            total_latency_ms=total_latency_ms,
            avg_latency_ms=avg_latency_ms,
            total_tokens=total_tokens,
            throughput=throughput,
        )
