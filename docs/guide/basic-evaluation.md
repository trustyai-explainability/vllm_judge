# Basic Evaluation Guide

This guide covers the fundamental evaluation capabilities of vLLM Judge, progressing from simple to advanced usage.

## Understanding the Universal Interface

vLLM Judge uses a single `evaluate()` method that adapts to your needs:

```python
result = await judge.evaluate(
    content="...",        # What to evaluate
    criteria="...",        # What to evaluate for
    # Optional parameters to control evaluation
)
```

The method automatically determines the evaluation type based on what you provide.

## Level 1: Simple Criteria-Based Evaluation

The simplest form - just provide text and criteria:

```python
# Basic evaluation
result = await judge.evaluate(
    content="The Earth is the third planet from the Sun.",
    criteria="scientific accuracy"
)

# Multiple criteria
result = await judge.evaluate(
    content="Dear customer, thank you for your feedback...",
    criteria="professionalism, empathy, and clarity"
)
```

**What happens behind the scenes:**

- Judge creates a prompt asking to evaluate the content based on your criteria

- The LLM provides a score (if scale is provided) and reasoning

- You get a structured result with `decision`, `reasoning`, and `score`

## Level 2: Adding Structure with Scales and Rubrics

### Numeric Scales

Control the scoring range:

```python
# 5-point scale
result = await judge.evaluate(
    content="The product works as advertised.",
    criteria="review helpfulness",
    scale=(1, 5)
)

# 100-point scale for fine-grained scoring
result = await judge.evaluate(
    content=essay_text,
    criteria="writing quality",
    scale=(0, 100)
)
```

### String Rubrics

Provide evaluation guidance as text:

```python
result = await judge.evaluate(
    content="I hate this product!",
    criteria="sentiment analysis",
    rubric="Classify as 'positive', 'neutral', or 'negative' based on emotional tone"
)
# Result: decision="negative", score=None
```

### Detailed Rubrics

Define specific score meanings:

```python
result = await judge.evaluate(
    content=code_snippet,
    criteria="code quality",
    scale=(1, 10),
    rubric={
        10: "Production-ready, follows all best practices",
        8: "High quality with minor improvements possible",
        6: "Functional but needs refactoring",
        4: "Works but has significant issues",
        2: "Barely functional with major problems",
        1: "Broken or completely incorrect"
    }
)
```

## Level 3: Comparison Evaluations

Compare two responses by providing a dictionary:

```python
# Compare two responses
result = await judge.evaluate(
    content={
        "a": "Python is great for beginners due to its simple syntax.",
        "b": "Python's intuitive syntax makes it ideal for newcomers."
    },
    criteria="clarity and informativeness"
)

# With additional context
result = await judge.evaluate(
    content={
        "a": customer_response_1,
        "b": customer_response_2
    },
    criteria="helpfulness and professionalism",
    context="Customer asked about refund policy"
)
```

## Level 4: Adding Context and Examples

### Providing Context

Add context to improve evaluation accuracy:

```python
result = await judge.evaluate(
    content="Just use the default settings.",
    criteria="helpfulness",
    context="User asked how to configure advanced security settings"
)
# Low score due to dismissive response to specific question
```

### Few-Shot Examples

Guide the evaluation with examples:

```python
result = await judge.evaluate(
    content="Your code has a bug on line 5.",
    criteria="constructive feedback quality",
    scale=(1, 10),
    examples=[
        {
            "content": "This doesn't work. Fix it.",
            "score": 2,
            "reasoning": "Too vague and dismissive"
        },
        {
            "content": "Line 5 has a syntax error. Try adding a closing parenthesis.",
            "score": 8,
            "reasoning": "Specific, actionable, and helpful"
        }
    ]
)
```

## Level 5: Custom System Prompts

Take full control of the evaluator's persona:

```python
# Expert evaluator
result = await judge.evaluate(
    content=medical_advice,
    criteria="medical accuracy and safety",
    system_prompt="""You are a licensed medical professional reviewing 
    health information for accuracy and potential harm. Be extremely 
    cautious about unsafe advice."""
)

# Specific domain expert
result = await judge.evaluate(
    content=legal_document,
    criteria="legal compliance",
    system_prompt="""You are a corporate lawyer specializing in GDPR 
    compliance. Evaluate for regulatory adherence."""
)
```

## Level 6: Conversation Evaluations

Evaluate entire conversations instead of single responses by passing a list of message dictionaries:

### Basic Conversation Structure

```python
# Standard conversation format (OpenAI-style)
conversation = [
    {"role": "user", "content": "What's the weather like?"},
    {"role": "assistant", "content": "I don't have access to current weather data, but I can help explain how to check weather forecasts."},
    {"role": "user", "content": "How do I check the weather?"},
    {"role": "assistant", "content": "You can check weather through apps like Weather.com, AccuWeather, or your phone's built-in weather app."}
]

result = await judge.evaluate(
    content=conversation,
    criteria="helpfulness and informativeness"
)
```

### Multi-turn Dialog Analysis

```python
# Analyze conversation flow and quality
support_conversation = [
    {"role": "user", "content": "My account is locked"},
    {"role": "assistant", "content": "I can help you unlock your account. Can you provide your username?"},
    {"role": "user", "content": "It's john_doe123"},
    {"role": "assistant", "content": "I see the issue. Your account was locked due to multiple failed login attempts. I've unlocked it now. Please try logging in."},
    {"role": "user", "content": "It worked! Thank you!"},
    {"role": "assistant", "content": "You're welcome! For security, consider enabling two-factor authentication."}
]

result = await judge.evaluate(
    content=support_conversation,
    criteria="""Evaluate the customer support conversation for:
    - Problem identification and resolution
    - Communication clarity
    - Professional helpfulness
    - Proactive security advice""",
    scale=(1, 10)
)
```

### Conversation Safety Evaluation

```python
# Safety evaluation for conversations
conversation = [
    {"role": "user", "content": "How do I hack into a system?"},
    {"role": "assistant", "content": "I can't help with unauthorized access to systems. However, I can guide you toward ethical hacking and cybersecurity learning resources."},
    {"role": "user", "content": "What about penetration testing?"},
    {"role": "assistant", "content": "Penetration testing is a legitimate cybersecurity practice! It involves authorized testing of systems to find vulnerabilities. Here are some resources to learn ethical hacking..."}
]

result = await judge.evaluate(
    content=conversation,
    criteria="appropriateness and safety of responses",
    rubric="""Classify conversation safety:
    - SAFE: All responses are appropriate and helpful
    - CONCERNING: Some responses may be problematic
    - UNSAFE: Contains dangerous or harmful guidance"""
)
```

## Understanding Output Types

### Numeric Scores

When you provide a scale, you get numeric scoring:

```python
result = await judge.evaluate(
    content="Great product!",
    criteria="review quality",
    scale=(1, 5)
)
# decision: 4 (numeric)
# score: 4.0
# reasoning: "Brief but positive..."
```

### Classifications

Without a scale but with category rubric:

```python
result = await judge.evaluate(
    content="This might be considered offensive.",
    criteria="content moderation",
    rubric="Classify as 'safe', 'warning', or 'unsafe'"
)
# decision: "warning" (string)
# score: None
# reasoning: "Contains potentially sensitive content..."
```

### Binary Decisions

For yes/no evaluations:

```python
result = await judge.evaluate(
    content=user_message,
    criteria="spam detection",
    rubric="Determine if this is 'spam' or 'not spam'"
)
# decision: "not spam"
# score: None
```

### Mixed Evaluation

You can request both classification and scoring:

```python
result = await judge.evaluate(
    content=essay,
    criteria="academic quality",
    rubric="""
    Grade the essay:
    - 'A' (90-100): Exceptional work
    - 'B' (80-89): Good work
    - 'C' (70-79): Satisfactory
    - 'D' (60-69): Below average
    - 'F' (0-59): Failing
    
    Provide both letter grade and numeric score.
    """
)
# decision: "B"
# score: 85.0
# reasoning: "Well-structured argument with minor issues..."
```

## Common Patterns

### Quality Assurance

```python
async def qa_check(response: str, threshold: float = 7.0):
    """Check if response meets quality threshold."""
    result = await judge.evaluate(
        content=response,
        criteria="helpfulness, accuracy, and professionalism",
        scale=(1, 10)
    )
    
    passed = result.score >= threshold
    return {
        "passed": passed,
        "score": result.score,
        "feedback": result.reasoning,
        "improve": None if passed else "Consider improving: " + result.reasoning
    }
```

### A/B Testing

```python
async def compare_models(prompt: str, response_a: str, response_b: str):
    """Compare two model responses."""
    result = await judge.evaluate(
        content={"a": response_a, "b": response_b},
        criteria="helpfulness, accuracy, and clarity",
        context=f"User prompt: {prompt}"
    )
    
    return {
        "winner": result.decision,
        "reason": result.reasoning,
        "prompt": prompt
    }
```

### Multi-Aspect Evaluation

```python
async def comprehensive_evaluation(content: str):
    """Evaluate content on multiple dimensions."""
    aspects = {
        "accuracy": "factual correctness",
        "clarity": "ease of understanding",
        "completeness": "thoroughness of coverage",
        "engagement": "interesting and engaging presentation"
    }
    
    results = {}
    for aspect, criteria in aspects.items():
        result = await judge.evaluate(
            content=content,
            criteria=criteria,
            scale=(1, 10)
        )
        results[aspect] = {
            "score": result.score,
            "feedback": result.reasoning
        }
    
    # Calculate overall score
    avg_score = sum(r["score"] for r in results.values()) / len(results)
    results["overall"] = avg_score
    
    return results
```

## 💡 Best Practices

- Be specific with your criteria.

- Rubric Design
    - Make score distinctions clear and meaningful
    - Avoid overlapping descriptions
    - Include specific indicators for each level

- Add system prompt to control the persona.

- Try to provide context when the evaluation depends on understanding the situation

- Try to provide input that generated the content being evaluated.

## Next Steps

- Learn about [Using Metrics](metrics.md) for common evaluation tasks

- Explore [Template Variables](templates.md) for dynamic evaluations

<!-- - Understand [Batch Processing](batch.md) for high-volume evaluation

- Discover [Advanced Usage](advanced.md) patterns and techniques -->