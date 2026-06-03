# Getting Started

## Prerequisites

- AMD Instinct GPU (MI200/MI300 series recommended)
- ROCm 6.0+ installed
- Python 3.9+

## Installation

### From Source

```bash
git clone https://github.com/YOUR_USERNAME/rocm-ai-inference.git
cd rocm-ai-inference
pip install -e .
```

### Using Docker

```bash
docker pull rocm/pytorch:latest
docker run -it --device=/dev/kfd --device=/dev/dri rocm/pytorch:latest
```

## Quick Example

```python
from rocm_inference import InferenceEngine, ModelConfig

config = ModelConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    precision="fp16"
)

engine = InferenceEngine(config)
result = engine.generate("Hello, world!")
print(result.text)
```
