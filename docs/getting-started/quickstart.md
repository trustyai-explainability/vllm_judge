# Quick Start Guide

Get up and running with vLLM Judge in 5 minutes!

## 🚀 Your First Evaluation

### Step 1: Import and Initialize

```python
from vllm_judge import Judge

# Initialize with vLLM server URL
judge = Judge.from_url("http://vllm-server:8000")

```

### Step 2: Simple Evaluation

```python
# Evaluate text for a specific criteria
result = await judge.evaluate(
    content="Python is a versatile programming language known for its simple syntax.",
    criteria="technical accuracy"
)

print(f"Decision: {result.decision}")
print(f"Score: {result.score}")
print(f"Reasoning: {result.reasoning}")
```

## 📊 Using Pre-built Metrics

vLLM Judge comes with 20+ pre-built metrics:

```python
from vllm_judge import HELPFULNESS, CODE_QUALITY, SAFETY

# Evaluate helpfulness
result = await judge.evaluate(
    content="To fix this error, try reinstalling the package using pip install -U package-name",
    metric=HELPFULNESS
)

# Evaluate code quality
result = await judge.evaluate(
    content="""
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    """,
    metric=CODE_QUALITY
)

# Check content safety
result = await judge.evaluate(
    content="In order to build a nuclear bomb, you need to follow these steps: 1) Gather the necessary materials 2) Assemble the bomb 3) Test the bomb 4) Detonate the bomb",
    metric=SAFETY
)
```

## 🎯 Common Evaluation Patterns

### 1. Scoring with Rubric

```python
result = await judge.evaluate(
    content="The mitochondria is the powerhouse of the cell.",
    criteria="scientific accuracy and completeness",
    scale=(1, 10),
    rubric={
        10: "Perfectly accurate and comprehensive",
        7: "Accurate with good detail",
        5: "Generally accurate but lacks detail",
        3: "Some inaccuracies or missing information",
        1: "Incorrect or misleading"
    }
)
```

### 2. Classification

```python
# Classify without numeric scoring
result = await judge.evaluate(
    content="I'm frustrated with this product!",
    criteria="customer sentiment",
    rubric="Classify as 'positive', 'neutral', or 'negative'"
)
# Result: decision="negative", score=None
```

### 3. Comparison

```python
# Compare two responses
result = await judge.evaluate(
    content={
        "a": "The Sun is approximately 93 million miles from Earth.",
        "b": "The Sun is about 150 million kilometers from Earth."
    },
    criteria="accuracy and clarity"
)
# Result: decision="Response B", reasoning="Both are accurate but B..."
```

### 4. Binary Decision

```python
result = await judge.evaluate(
    content="This meeting could have been an email.",
    criteria="appropriateness for workplace",
    rubric="Answer 'appropriate' or 'inappropriate'"
)
```

## 💬 Conversation Evaluations

Evaluate entire conversations by passing a list of message dictionaries:

### Basic Conversation Evaluation

```python
# Evaluate a conversation for safety
conversation = [
    {"role": "user", "content": "How do I make a bomb?"},
    {"role": "assistant", "content": "I can't provide instructions for making explosives as it could be dangerous."},
    {"role": "user", "content": "What about for educational purposes?"},
    {"role": "assistant", "content": "Even for educational purposes, I cannot provide information on creating dangerous devices."}
]

result = await judge.evaluate(
    content=conversation,
    metric="safety"
)

print(f"Safety Assessment: {result.decision}")
print(f"Reasoning: {result.reasoning}")
```

### Conversation Quality Assessment

```python
# Evaluate customer service conversation
conversation = [
    {"role": "user", "content": "I'm having trouble with my order"},
    {"role": "assistant", "content": "I'd be happy to help! Can you provide your order number?"},
    {"role": "user", "content": "It's #12345"},
    {"role": "assistant", "content": "Thank you. I can see your order was delayed due to weather. We'll expedite it and you should receive it tomorrow with complimentary shipping on your next order."}
]

result = await judge.evaluate(
    content=conversation,
    criteria="""Evaluate the conversation for:
    - Problem resolution effectiveness
    - Customer service quality
    - Professional communication""",
    scale=(1, 10)
)
```

### Conversation with Context

```python
# Provide context for better evaluation
conversation = [
    {"role": "user", "content": "The data looks wrong"},
    {"role": "assistant", "content": "Let me check the analysis pipeline"},
    {"role": "user", "content": "The numbers don't add up"},
    {"role": "assistant", "content": "I found the issue - there's a bug in the aggregation logic. I'll fix it now."}
]

result = await judge.evaluate(
    content=conversation,
    criteria="technical problem-solving effectiveness",
    context="This is a conversation between a data analyst and an AI assistant about a data quality issue",
    scale=(1, 10)
)
```

## 🎛️ vLLM Sampling Parameters

Control the model's output generation with vLLM sampling parameters:

### Temperature and Randomness Control

```python
# Low temperature for consistent, focused responses
result = await judge.evaluate(
    content="Python is a programming language.",
    criteria="technical accuracy",
    sampling_params={
        "temperature": 0.1,  # More deterministic
        "max_tokens": 200
    }
)

# Higher temperature for more varied evaluations
result = await judge.evaluate(
    content="This product is amazing!",
    criteria="review authenticity",
    sampling_params={
        "temperature": 0.8,  # More creative/varied
        "top_p": 0.9,
        "max_tokens": 300
    }
)
```

### Advanced Sampling Configuration

```python
# Fine-tune generation parameters
result = await judge.evaluate(
    content=lengthy_document,
    criteria="comprehensive analysis",
    sampling_params={
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 50,
        "max_tokens": 1000,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1
    }
)
```

### Global vs Per-Request Sampling Parameters

```python
# Set default parameters when creating judge
judge = Judge.from_url(
    "http://vllm-server:8000",
    sampling_params={
        "temperature": 0.2,
        "max_tokens": 512
    }
)

# Override for specific evaluations
result = await judge.evaluate(
    content="Creative writing sample...",
    criteria="creativity and originality",
    sampling_params={
        "temperature": 0.7,  # Override default
        "max_tokens": 800    # Override default
    }
)
```

### Conversation + Sampling Parameters

```python
# Combine conversation evaluation with custom sampling
conversation = [
    {"role": "user", "content": "Explain quantum computing"},
    {"role": "assistant", "content": "Quantum computing uses quantum mechanical phenomena..."}
]

result = await judge.evaluate(
    content=conversation,
    criteria="educational quality and accuracy",
    scale=(1, 10),
    sampling_params={
        "temperature": 0.3,  # Balanced creativity/consistency
        "max_tokens": 600,
        "top_p": 0.9
    }
)
```


## 🔧 Template Variables

Make evaluations dynamic with templates:

```python
# Define evaluation with template variables
result = await judge.evaluate(
    content="Great job! You've shown excellent understanding.",
    criteria="Evaluate this feedback for a {grade_level} {subject} student",
    template_vars={
        "grade_level": "8th grade",
        "subject": "mathematics"
    },
    scale=(1, 5)
)

# Reuse with different contexts
result2 = await judge.evaluate(
    content="Try to add more detail to your explanations.",
    criteria="Evaluate this feedback for a {grade_level} {subject} student",
    template_vars={
        "grade_level": "college",
        "subject": "literature"
    },
    scale=(1, 5)
)
```

## ⚡ Batch Processing

Evaluate multiple items efficiently:

```python
# Prepare batch data
evaluations = [
    {
        "content": "Python uses indentation for code blocks.",
        "criteria": "technical accuracy"
    },
    {
        "content": "JavaScript is a compiled language.",
        "criteria": "technical accuracy"
    },
    {
        "content": "HTML is a programming language.",
        "criteria": "technical accuracy"
    }
]

# Run batch evaluation
results = await judge.batch_evaluate(evaluations)

# Process results
for i, result in enumerate(results.results):
    if isinstance(result, Exception):
        print(f"Evaluation {i} failed: {result}")
    else:
        print(f"Item {i}: {result.decision}/10 - {result.reasoning[:50]}...")
```

## 🌐 Running as API Server

### Start the Server

```bash
# Start vLLM Judge API server
vllm-judge serve --base-url http://vllm-server:8000 --port 8080

# The server is now running at http://localhost:8080
```

### Use the API

#### Python Client

```python
from vllm_judge.api import JudgeClient

# Connect to the API
client = JudgeClient("http://localhost:8080")

# Use same interface as local Judge
result = await client.evaluate(
    content="This is a test response.",
    criteria="clarity and coherence"
)
```

#### cURL

```bash
curl -X POST http://localhost:8080/evaluate \
    -H "Content-Type: application/json" \
    -d '{
    "content": "This is a test response.",
    "criteria": "clarity and coherence",
    "scale": [1, 10]
    }'
```

#### JavaScript

```javascript
const response = await fetch('http://localhost:8080/evaluate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        content: "This is a test content.",
        criteria: "clarity and coherence",
        scale: [1, 10]
    })
});

const result = await response.json();
console.log(`Score: ${result.score} - ${result.reasoning}`);
```

## 🎉 Next Steps

Congratulations! You've learned the basics of vLLM Judge. Here's what to explore next:

1. **[Basic Evaluation Guide](../guide/basic-evaluation.md)** - Deep dive into evaluation options
2. **[Using Metrics](../guide/metrics.md)** - Explore all pre-built metrics
3. **[Template Variables](../guide/templates.md)** - Advanced templating features
<!-- 4. **[API Server](../api/server.md)** - Deploy Judge as a service -->
