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
    response="Python is a versatile programming language known for its simple syntax.",
    criteria="technical accuracy"
)

print(f"Decision: {result.decision}")
print(f"Score: {result.score}")
print(f"Reasoning: {result.reasoning}")
```

**Sample Output:**
```
Decision: 9
Score: 9.0
Reasoning: The statement is technically accurate. Python is indeed known for its 
versatility and simple, readable syntax, making it popular for various applications.
```

## 📊 Using Pre-built Metrics

vLLM Judge comes with 20+ pre-built metrics:

```python
from vllm_judge import HELPFULNESS, CODE_QUALITY, SAFETY

# Evaluate helpfulness
result = await judge.evaluate(
    response="To fix this error, try reinstalling the package using pip install -U package-name",
    metric=HELPFULNESS
)

# Evaluate code quality
result = await judge.evaluate(
    response="""
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    """,
    metric=CODE_QUALITY
)

# Check content safety
result = await judge.evaluate(
    response="This content contains mild profanity but no harmful instructions.",
    metric=SAFETY
)
```

## 🎯 Common Evaluation Patterns

### 1. Scoring with Rubric

```python
result = await judge.evaluate(
    response="The mitochondria is the powerhouse of the cell.",
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
    response="I'm frustrated with this product!",
    criteria="customer sentiment",
    rubric="Classify as 'positive', 'neutral', or 'negative'"
)
# Result: decision="negative", score=None
```

### 3. Comparison

```python
# Compare two responses
result = await judge.evaluate(
    response={
        "a": "The Sun is approximately 93 million miles from Earth.",
        "b": "The Sun is about 150 million kilometers from Earth."
    },
    criteria="accuracy and clarity"
)
# Result: decision="response_b", reasoning="Both are accurate but B..."
```

### 4. Binary Decision

```python
result = await judge.evaluate(
    response="This meeting could have been an email.",
    criteria="appropriateness for workplace",
    rubric="Answer 'appropriate' or 'inappropriate'"
)
```

## 🔧 Template Variables

Make evaluations dynamic with templates:

```python
# Define evaluation with template variables
result = await judge.evaluate(
    response="Great job! You've shown excellent understanding.",
    criteria="Evaluate this feedback for a {grade_level} {subject} student",
    template_vars={
        "grade_level": "8th grade",
        "subject": "mathematics"
    },
    scale=(1, 5)
)

# Reuse with different contexts
result2 = await judge.evaluate(
    response="Try to add more detail to your explanations.",
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
        "response": "Python uses indentation for code blocks.",
        "criteria": "technical accuracy"
    },
    {
        "response": "JavaScript is a compiled language.",
        "criteria": "technical accuracy"
    },
    {
        "response": "HTML is a programming language.",
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
    response="This is a test response.",
    criteria="clarity and coherence"
)
```

#### cURL

```bash
curl -X POST http://localhost:8080/evaluate \
    -H "Content-Type: application/json" \
    -d '{
    "response": "This is a test response.",
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
        response: "This is a test response.",
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
4. **[API Server](../api/server.md)** - Deploy Judge as a service
