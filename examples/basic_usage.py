#!/usr/bin/env python3
"""
Basic usage example for ROCm AI Inference Engine.
"""

from rocm_inference import InferenceEngine, ModelConfig


def main():
    # Configure model
    config = ModelConfig(
        model_name="gpt2",  # Use small model for demo
        precision="fp16",
        max_batch_size=4,
        max_sequence_length=512,
    )
    
    # Initialize engine
    print("Loading model...")
    engine = InferenceEngine(config)
    
    # Generate text
    prompt = "The future of artificial intelligence is"
    print(f"\nPrompt: {prompt}")
    print("-" * 50)
    
    result = engine.generate(
        prompt,
        max_new_tokens=100,
        temperature=0.7,
    )
    
    print(f"Generated: {result.text}")
    print(f"\nMetrics:")
    print(f"  Tokens generated: {result.tokens_generated}")
    print(f"  Latency: {result.latency_ms:.2f} ms")
    print(f"  Throughput: {result.tokens_per_second:.1f} tokens/sec")


if __name__ == "__main__":
    main()
