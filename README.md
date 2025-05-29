# vLLM Judge

A lightweight library for LLM-as-a-Judge evaluations using vLLM hosted models.

## Features

- 🚀 **Simple Interface**: Single `evaluate()` method that adapts to any use case
- 🎯 **Pre-built Metrics**: 20+ ready-to-use evaluation metrics
- 🔧 **Template Support**: Dynamic evaluations with template variables
- ⚡ **High Performance**: Optimized for vLLM with automatic batching
- 🌐 **API Mode**: Run as a REST API service
- 🔄 **Async Native**: Built for high-throughput evaluations

## Installation

```bash
# Basic installation
pip install vllm_judge

# With API support
pip install vllm_judge[api]

# With Jinja2 template support
pip install vllm_judge[jinja2]

# Everything
pip install vllm_judge[api,jinja2]
```

## Quick Start

```python
from vllm_judge import Judge

# Initialize with vLLM url
judge = await Judge.from_url("http://localhost:8000")

# Simple evaluation
result = await judge.evaluate(
    response="The Earth orbits around the Sun.",
    criteria="scientific accuracy"
)
print(f"Decision: {result.decision}")
print(f"Reasoning: {result.reasoning}")

# Using pre-built metrics
from vllm_judge import CODE_QUALITY

result = await judge.evaluate(
    response="def add(a, b): return a + b",
    metric=CODE_QUALITY
)

# With template variables
result = await judge.evaluate(
    response="Essay content here...",
    criteria="Evaluate this {doc_type} for {audience}",
    template_vars={
        "doc_type": "essay",
        "audience": "high school students"
    }
)
```

## API Server

Run Judge as a REST API:

```bash
vllm-judge serve --base-url http://localhost:8000 --port 9090 --host localhost
```

Then use the HTTP API:

```python
from vllm_judge.api import JudgeClient

client = JudgeClient("http://localhost:9090")
result = await client.evaluate(
    response="Python is great!",
    criteria="technical accuracy"
)
```

