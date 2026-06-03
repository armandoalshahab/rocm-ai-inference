#!/usr/bin/env python3
"""
Benchmark example for ROCm AI Inference Engine.
"""

from rocm_inference import InferenceEngine, ModelConfig, BenchmarkSuite


def main():
    # Configure model
    config = ModelConfig(
        model_name="gpt2",
        precision="fp16",
        max_batch_size=8,
    )
    
    # Initialize engine
    print("Loading model...")
    engine = InferenceEngine(config)
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    suite = BenchmarkSuite(engine)
    
    report = suite.run(
        batch_sizes=[1, 2, 4],
        sequence_lengths=[64, 128],
        iterations=20,
        warmup_iterations=3,
    )
    
    # Print results
    report.print_summary()
    
    # Save results
    report.save_csv("benchmark_results.csv")
    report.save_json("benchmark_results.json")
    print("\nResults saved to benchmark_results.csv and benchmark_results.json")


if __name__ == "__main__":
    main()
