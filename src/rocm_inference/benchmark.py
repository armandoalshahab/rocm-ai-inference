"""
Benchmarking suite for performance evaluation.
"""

import time
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from pathlib import Path

import torch

from rocm_inference.engine import InferenceEngine
from rocm_inference.config import ModelConfig
from rocm_inference.utils.gpu_utils import get_gpu_info

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run.
    
    Attributes:
        model_name: Name of the model tested
        precision: Compute precision used
        batch_size: Batch size for the test
        sequence_length: Input sequence length
        latency_ms: Average latency in milliseconds
        throughput: Tokens per second
        gpu_memory_mb: Peak GPU memory usage in MB
        iterations: Number of iterations run
    """
    model_name: str
    precision: str
    batch_size: int
    sequence_length: int
    latency_ms: float
    throughput: float
    gpu_memory_mb: float
    iterations: int


@dataclass
class BenchmarkReport:
    """Complete benchmark report with multiple results.
    
    Attributes:
        results: List of individual benchmark results
        gpu_info: GPU information at time of benchmark
        timestamp: When the benchmark was run
    """
    results: List[BenchmarkResult]
    gpu_info: Dict
    timestamp: str
    
    def print_summary(self):
        """Print a formatted summary of benchmark results."""
        print("\n" + "=" * 80)
        print("Benchmark Report")
        print("=" * 80)
        print(f"GPU: {self.gpu_info.get('name', 'Unknown')}")
        print(f"Timestamp: {self.timestamp}")
        print("-" * 80)
        print(f"{'Model':<20} {'Precision':<10} {'Batch':<8} {'Seq Len':<10} {'Latency':<12} {'Throughput':<15}")
        print("-" * 80)
        
        for r in self.results:
            print(f"{r.model_name:<20} {r.precision:<10} {r.batch_size:<8} {r.sequence_length:<10} {r.latency_ms:<12.2f} {r.throughput:<15.1f}")
        
        print("=" * 80)
    
    def save_csv(self, path: str):
        """Save results to CSV file.
        
        Args:
            path: Output file path
        """
        import csv
        
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model", "precision", "batch_size", "sequence_length",
                "latency_ms", "throughput", "gpu_memory_mb", "iterations"
            ])
            for r in self.results:
                writer.writerow([
                    r.model_name, r.precision, r.batch_size, r.sequence_length,
                    f"{r.latency_ms:.2f}", f"{r.throughput:.1f}",
                    f"{r.gpu_memory_mb:.1f}", r.iterations
                ])
        
        logger.info(f"Saved CSV report to {path}")
    
    def save_json(self, path: str):
        """Save results to JSON file.
        
        Args:
            path: Output file path
        """
        data = {
            "gpu_info": self.gpu_info,
            "timestamp": self.timestamp,
            "results": [asdict(r) for r in self.results],
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved JSON report to {path}")


class BenchmarkSuite:
    """Comprehensive benchmarking suite for inference engines.
    
    Runs standardized benchmarks across different configurations.
    
    Example:
        >>> engine = InferenceEngine(config)
        >>> suite = BenchmarkSuite(engine)
        >>> report = suite.run(batch_sizes=[1, 4, 8], iterations=50)
        >>> report.print_summary()
    """
    
    def __init__(self, engine: InferenceEngine):
        """Initialize benchmark suite.
        
        Args:
            engine: Inference engine to benchmark
        """
        self.engine = engine
    
    def run(
        self,
        batch_sizes: List[int] = [1, 4, 8, 16],
        sequence_lengths: List[int] = [128, 256, 512],
        iterations: int = 50,
        warmup_iterations: int = 5,
    ) -> BenchmarkReport:
        """Run comprehensive benchmarks.
        
        Args:
            batch_sizes: List of batch sizes to test
            sequence_lengths: List of sequence lengths to test
            iterations: Number of iterations per configuration
            warmup_iterations: Number of warmup iterations
            
        Returns:
            BenchmarkReport with all results
        """
        results = []
        
        # Get GPU info
        gpu_info = get_gpu_info()
        
        total_configs = len(batch_sizes) * len(sequence_lengths)
        config_num = 0
        
        for batch_size in batch_sizes:
            for seq_len in sequence_lengths:
                config_num += 1
                logger.info(f"Benchmarking config {config_num}/{total_configs}: batch={batch_size}, seq_len={seq_len}")
                
                result = self._benchmark_config(
                    batch_size=batch_size,
                    sequence_length=seq_len,
                    iterations=iterations,
                    warmup_iterations=warmup_iterations,
                )
                results.append(result)
        
        from datetime import datetime
        
        return BenchmarkReport(
            results=results,
            gpu_info=gpu_info,
            timestamp=datetime.now().isoformat(),
        )
    
    def _benchmark_config(
        self,
        batch_size: int,
        sequence_length: int,
        iterations: int,
        warmup_iterations: int,
    ) -> BenchmarkResult:
        """Benchmark a specific configuration."""
        import torch
        
        # Generate dummy prompt of specified length
        dummy_tokens = "hello " * (sequence_length // 2)
        
        # Warmup
        for _ in range(warmup_iterations):
            self.engine.generate(
                dummy_tokens,
                max_new_tokens=16,
                do_sample=False,
            )
        
        # Clear GPU memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Benchmark
        latencies = []
        total_tokens = 0
        
        for _ in range(iterations):
            result = self.engine.generate(
                dummy_tokens,
                max_new_tokens=16,
                do_sample=False,
            )
            latencies.append(result.latency_ms)
            total_tokens += result.tokens_generated
        
        # Get peak memory
        peak_memory_mb = 0
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6
        
        # Calculate metrics
        avg_latency = sum(latencies) / len(latencies)
        avg_tokens = total_tokens / len(latencies)
        throughput = avg_tokens / (avg_latency / 1000) if avg_latency > 0 else 0
        
        return BenchmarkResult(
            model_name=self.engine.config.model_name.split("/")[-1],
            precision=self.engine.config.precision,
            batch_size=batch_size,
            sequence_length=sequence_length,
            latency_ms=avg_latency,
            throughput=throughput,
            gpu_memory_mb=peak_memory_mb,
            iterations=iterations,
        )
