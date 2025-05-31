# Using Metrics

vLLM Judge provides 20+ pre-built metrics for common evaluation tasks. This guide shows how to use them effectively.

## What are Metrics?

Metrics are pre-configured evaluation templates that encapsulate:

- **Criteria**: What to evaluate

- **Scale**: Numeric range (if applicable)

- **Rubric**: Judgement/Scoring guidelines

- **System Prompt**: Evaluator expertise

- **Examples**: Few-shot guidance 

## Using Pre-built Metrics

### Basic Usage

```python
from vllm_judge import Judge, HELPFULNESS, CODE_QUALITY, SAFETY

# Use a metric directly
result = await judge.evaluate(
    response="To reset your password, click on 'Forgot Password' on the login page.",
    metric=HELPFULNESS
)

# Metrics can be referenced by name (string) after import
from vllm_judge import BUILTIN_METRICS  # Import to register all metrics
result = await judge.evaluate(
    response="def hello(): print('Hi')",
    metric="code_quality"  # String reference
)
```

### Listing Available Metrics

```python
# List all available metrics
metrics = judge.list_metrics()
print("Available metrics:", metrics)

# Get metric details
from vllm_judge.metrics import BUILTIN_METRICS
for name, metric in BUILTIN_METRICS.items():
    print(f"\n{name}:")
    print(f"  Criteria: {metric.criteria}")
    print(f"  Rubric: {metric.rubric}")
    print(f"  Scale: {metric.scale}")
```

## Categories of Pre-built Metrics

### 📊 General Purpose Metrics

#### HELPFULNESS
Evaluates how well a response addresses user needs.

```python
result = await judge.evaluate(
    response="Try restarting your computer to fix the issue.",
    metric=HELPFULNESS
)
# Scale: 1-10
# High scores: Thoroughly addresses the request
# Low scores: Unhelpful or misses the point
```

#### ACCURACY
Evaluates factual correctness.

```python
result = await judge.evaluate(
    response="Python was created by Guido van Rossum in 1991.",
    metric=ACCURACY
)
# Scale: 1-10
# High scores: Completely accurate
# Low scores: Contains errors or misinformation
```

#### CLARITY
Evaluates how clear and understandable the response is.

```python
result = await judge.evaluate(
    response="The API endpoint accepts POST requests with JSON payload containing user credentials.",
    metric=CLARITY
)
```

#### CONCISENESS
Evaluates brevity without losing essential information.

```python
result = await judge.evaluate(
    response="To summarize, the main point is that...",
    metric=CONCISENESS
)
```

#### RELEVANCE
Evaluates how relevant the response is to the query.

```python
result = await judge.evaluate(
    response="Here's information about Python programming...",
    metric=RELEVANCE,
    context="User asked about JavaScript frameworks"
)
# Low score due to off-topic response
```

### 🛡️ Safety and Moderation Metrics

#### SAFETY
Classifies content safety level.

```python
# Works with specialized safety models out-of-the-box
result = await judge.evaluate(
    response="How do I make a bomb?",
    metric=LLAMA_GUARD_3_SAFETY  # Automatically uses Llama Guard format
)
# Result: decision="unsafe", reasoning="S9"

# If not using specialized models, use LLM of your choice with generic metric
result = await judge.evaluate(
    response="This tutorial shows how to build a web scraper.",
    metric=SAFETY
)
```

#### TOXICITY
Measures level of toxic or offensive content.

```python
result = await judge.evaluate(
    response="I disagree with your opinion on this matter.",
    metric=TOXICITY
)
# Scale: 0-10 (0 = no toxicity, 10 = extremely toxic)
```

### 💻 Code Quality Metrics

#### CODE_QUALITY
Comprehensive code evaluation.

```python
result = await judge.evaluate(
    response="""
    def calculate_average(numbers):
        if not numbers:
            return 0
        return sum(numbers) / len(numbers)
    """,
    metric=CODE_QUALITY
)
# Scale: 1-10
# Evaluates: correctness, efficiency, readability, best practices
```

#### CODE_SECURITY
Evaluates code for security vulnerabilities.

```python
result = await judge.evaluate(
    response="""
    user_input = input("Enter SQL: ")
    cursor.execute(f"SELECT * FROM users WHERE id = {user_input}")
    """,
    metric=CODE_SECURITY
)
# Low score due to SQL injection vulnerability
```

### 📝 Content Quality Metrics

#### CREATIVITY
Measures originality and creative expression.

```python
result = await judge.evaluate(
    response="The sky wept diamonds as the sun retired for the day.",
    metric=CREATIVITY
)
```

#### PROFESSIONALISM
Evaluates professional tone and presentation.

```python
result = await judge.evaluate(
    response="Hey! Thanks for reaching out. We'll get back to ya soon!",
    metric=PROFESSIONALISM,
    context="Customer service email"
)
# Lower score due to casual tone
```

#### EDUCATIONAL_VALUE
Evaluates how well content teaches or explains.

```python
result = await judge.evaluate(
    response=tutorial_content,
    metric=EDUCATIONAL_VALUE
)
```

### 🔄 Comparison and Classification Metrics

#### PREFERENCE
For comparing two options without specific criteria.

```python
result = await judge.evaluate(
    response={"a": response1, "b": response2},
    metric=PREFERENCE
)
# Returns which response is preferred overall
```

#### APPROPRIATE
Binary classification of appropriateness.

```python
result = await judge.evaluate(
    response="This joke might offend some people.",
    metric=APPROPRIATE,
    context="Company newsletter"
)
```

#### FACTUAL
Verifies factual claims.

```python
result = await judge.evaluate(
    response="The speed of light is approximately 300,000 km/s.",
    metric=FACTUAL
)
```

### 🏥 Domain-Specific Metrics

#### MEDICAL_ACCURACY
Evaluates medical information (with safety focus).

```python
result = await judge.evaluate(
    response="For headaches, drink plenty of water and rest.",
    metric=MEDICAL_ACCURACY
)
# Scale: 1-5
# Includes safety considerations
# Note: For educational evaluation only
```

#### LEGAL_APPROPRIATENESS
Evaluates legal information appropriateness.

```python
result = await judge.evaluate(
    response="You should consult a lawyer for specific advice.",
    metric=LEGAL_APPROPRIATENESS
)
```

## Customizing Pre-built Metrics

### Override Metric Parameters

You can override any metric parameter:

```python
# Use HELPFULNESS with a different scale
result = await judge.evaluate(
    response="Here's the solution to your problem...",
    metric=HELPFULNESS,
    scale=(1, 5)  # Override default 1-10 scale
)

# Add context to any metric
result = await judge.evaluate(
    response=code,
    metric=CODE_QUALITY,
    context="This is a beginner's first Python function"
)

# Override system prompt
result = await judge.evaluate(
    response=content,
    metric=SAFETY,
    system_prompt="You are evaluating content for a children's platform."
)
```

## Creating Custom Metrics

### Simple Custom Metric

```python
from vllm_judge import Metric

# Define a custom metric
customer_service_metric = Metric(
    name="customer_service",
    criteria="politeness, helpfulness, and problem resolution",
    scale=(1, 10),
    rubric={
        10: "Exceptional service that exceeds expectations",
        8: "Very good service with minor areas for improvement",
        6: "Adequate service but missing key elements",
        4: "Poor service that may frustrate customers",
        2: "Unacceptable service likely to lose customers"
    },
    system_prompt="You are a customer service quality expert."
)

# Use it
result = await judge.evaluate(
    response="I understand your frustration. Let me help you resolve this.",
    metric=customer_service_metric
)
```

You can optionally register the metric to reuse directly with name

```python
judge.register_metric(customer_service_metric)
# reference with name
result = await judge.evaluate(
    response=text,
    metric="customer_service"
)
```
### Metric with Examples

```python
email_quality_metric = Metric(
    name="email_quality",
    criteria="professionalism, clarity, and appropriate tone",
    scale=(1, 5),
    rubric={
        5: "Perfect professional email",
        4: "Good with minor improvements",
        3: "Acceptable but could be better",
        2: "Unprofessional or unclear",
        1: "Inappropriate or very poor"
    },
    examples=[
        {
            "response": "Hey, wanted to touch base about that thing",
            "score": 2,
            "reasoning": "Too casual and vague for professional context"
        },
        {
            "response": "Dear Team, I hope this email finds you well. I'm writing to discuss...",
            "score": 5,
            "reasoning": "Professional greeting, clear purpose, appropriate tone"
        }
    ]
)
```

## Metric Selection Guide

| Use Case | Recommended Metrics |
|----------|-------------------|
| **Customer Support** | HELPFULNESS, PROFESSIONALISM, CLARITY |
| **Content Moderation** | SAFETY, TOXICITY, APPROPRIATE |
| **Code Review** | CODE_QUALITY, CODE_SECURITY |
| **Educational Content** | EDUCATIONAL_VALUE, ACCURACY, CLARITY |
| **Creative Writing** | CREATIVITY, RELEVANCE |
| **Medical/Legal Content** | MEDICAL_ACCURACY, LEGAL_APPROPRIATENESS |
| **General QA** | ACCURACY, HELPFULNESS, RELEVANCE |


## 💡 Best Practices

- **Choose the Right Metric:** Select metrics that align with your evaluation goals. Use domain-specific metrics when available.

- **Provide Context:** Even with pre-built metrics, adding context improves evaluation accuracy.

- **Combine Metrics:** For comprehensive evaluation, use multiple complementary metrics.

- **Custom Metrics for Specific Needs:** Create custom metrics for domain-specific or unique evaluation requirements.

- **Medical and Legal Metrics:** These are for educational evaluation only. Always include appropriate disclaimers.

## Next Steps

- Learn about [Template Variables](templates.md) for dynamic metric customization

<!-- - See [Examples](../examples/code-review.md) of metrics in action -->