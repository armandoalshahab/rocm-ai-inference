# Contributing to ROCm AI Inference Engine

We love your input! We want to make contributing to this project as easy and transparent as possible.

## Development Process

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code lints
6. Issue that pull request!

## Setting Up Development Environment

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/rocm-ai-inference.git
cd rocm-ai-inference

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

## Code Style

We use:
- **Black** for code formatting
- **Ruff** for linting
- **isort** for import sorting

Run formatting:
```bash
black src/ tests/
isort src/ tests/
```

Run linting:
```bash
ruff check src/ tests/
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=rocm_inference --cov-report=html
```

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update the docs/ if you're changing APIs
3. The PR will be merged once you have the sign-off of at least one maintainer

## Any contributions you make will be under the MIT Software License

When you submit code changes, your submissions are understood to be under the same [MIT License](LICENSE) that covers the project.

## Report bugs using GitHub Issues

We use GitHub issues to track public bugs. Report a bug by opening a new issue.

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be, or stuff you tried)

## License

By contributing, you agree that your contributions will be licensed under its MIT License.
