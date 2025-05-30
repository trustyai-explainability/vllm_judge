# Installation

This guide covers the installation of vLLM Judge and its prerequisites.

## Prerequisites

### Python Version

vLLM Judge requires Python 3.8 or higher. You can check your Python version:

```bash
python --version
```

### vLLM Server

You need access to a vLLM server running your preferred model. If you don't have one:


```bash
# Install vLLM
pip install vllm

# Start a model server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8b-instruct \
    --port 8000
```

## Installing vLLM Judge

### Basic Installation

Install the core library with pip:

```bash
pip install vllm-judge
```

This installs the essential dependencies:

- `httpx` - Async HTTP client

- `pydantic` - Data validation

- `tenacity` - Retry logic

- `click` - CLI interface

### Optional Features

#### API Server

To run vLLM Judge as an API server:

```bash
pip install vllm-judge[api]
```

This adds:

- `fastapi` - Web framework

- `uvicorn` - ASGI server

- `websockets` - WebSocket support

#### Jinja2 Templates

For advanced template support:

```bash
pip install vllm-judge[jinja2]
```
    
This enables Jinja2 template engine for complex template logic.


#### Everything

Install all optional features:

```bash
pip install vllm-judge[dev]
```

### Installation from Source

To install the latest development version:

```bash
# Clone the repository
git clone https://github.com/saichandrapandraju/vllm-judge.git
cd vllm-judge

# Install in development mode
pip install -e .

# With all extras
pip install -e ".[dev]"
```

## Verifying Installation

### Basic Check

```python
# In Python
from vllm_judge import Judge
print("vLLM Judge installed successfully!")
```

### CLI Check

```bash
# Check CLI installation
vllm-judge --help
```

### Version Check

```python
import vllm_judge
print(f"vLLM Judge version: {vllm_judge.__version__}")
```

## Environment Setup

### Virtual Environment (Recommended)

It's recommended to use a virtual environment:

#### venv

```bash
# Create virtual environment
python -m venv vllm-judge-env

# Activate it
# On Linux/Mac:
source vllm-judge-env/bin/activate
# On Windows:
vllm-judge-env\Scripts\activate

# Install vLLM Judge
pip install vllm-judge
```

#### conda

```bash
# Create conda environment
conda create -n vllm-judge python=3.9
conda activate vllm-judge

# Install vLLM Judge
pip install vllm-judge
```

## 🎉 Next Steps

Congratulations! You've successfully installed vLLM Judge and ready for some evals. Here's what to explore next:

- **[Quick Start](quickstart.md)** - Get up and running with vLLM Judge in 5 minutes!