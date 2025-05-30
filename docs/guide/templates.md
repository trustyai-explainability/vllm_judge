# Template Variables

Template variables make your evaluations dynamic and reusable. This guide shows how to use them effectively.

## Why Use Templates?

Templates allow you to:
- **Reuse** evaluation logic with different contexts
- **Parameterize** criteria for different scenarios
- **Scale** evaluations across varied use cases
- **Maintain** consistency while allowing flexibility

## Basic Template Usage

### Simple Variable Substitution

Use `{variable_name}` in your criteria, rubric, or prompts:

```python
# Basic template
result = await judge.evaluate(
    response=tutorial,
    criteria="Evaluate this tutorial for {audience}",
    template_vars={
        "audience": "beginners"
    }
)

# The criteria becomes: "Evaluate this tutorial for beginners"
```

### Multiple Variables

```python
result = await judge.evaluate(
    response=code_snippet,
    criteria="Review this {language} code for {purpose} following {company} standards",
    template_vars={
        "language": "Python",
        "purpose": "data processing",
        "company": "ACME Corp"
    }
)
```

### Templates in Rubrics

Templates work in rubric text and score descriptions:

```python
result = await judge.evaluate(
    response=essay,
    criteria="Evaluate this {doc_type} for {grade_level} students",
    scale=(1, 10),
    rubric={
        10: "Perfect {doc_type} for {grade_level} level",
        7: "Good {doc_type} with minor issues for {grade_level}",
        4: "Weak {doc_type} inappropriate for {grade_level}",
        1: "Completely unsuitable for {grade_level}"
    },
    template_vars={
        "doc_type": "essay",
        "grade_level": "8th grade"
    }
)
```

## Template Engines

### Format Engine (Default)

Uses Python's `str.format()` syntax:

```python
# Basic substitution
criteria="Evaluate {aspect} of this {content_type}"

# With format specifiers
criteria="Score must be at least {min_score:.1f}"

# Accessing nested values
template_vars = {"user": {"name": "Alice", "role": "student"}}
criteria="Evaluate for {user.name} who is a {user.role}"
```

### Jinja2 Engine

For advanced templating with logic:

```python
# Conditional content
result = await judge.evaluate(
    response=content,
    criteria="""
    Evaluate this content for {audience}.
    {% if technical_level == 'advanced' %}
    Pay special attention to technical accuracy and depth.
    {% else %}
    Focus on clarity and accessibility.
    {% endif %}
    """,
    template_vars={
        "audience": "developers",
        "technical_level": "advanced"
    },
    template_engine="jinja2"
)

# Loops in templates
result = await judge.evaluate(
    response=api_docs,
    criteria="""
    Check documentation for these aspects:
    {% for aspect in aspects %}
    - {{ aspect }}
    {% endfor %}
    """,
    template_vars={
        "aspects": ["completeness", "accuracy", "examples", "error handling"]
    },
    template_engine="jinja2"
)
```

## Creating Template-Based Metrics

### Basic Template Metric

```python
from vllm_judge import Metric

# Create a reusable template metric
review_metric = Metric(
    name="product_review",
    criteria="Evaluate this {product_type} review for {marketplace}",
    scale=(1, 5),
    rubric={
        5: "Excellent {product_type} review for {marketplace}",
        3: "Average review with basic information",
        1: "Poor review lacking detail"
    },
    template_vars={
        "marketplace": "online"  # Default value
    },
    required_vars=["product_type"]  # Must be provided during evaluation
)

# Use with different products
tech_result = await judge.evaluate(
    response="This laptop has great battery life...",
    metric=review_metric,
    template_vars={"product_type": "electronics"}
)

book_result = await judge.evaluate(
    response="Engaging plot with well-developed characters...",
    metric=review_metric,
    template_vars={
        "product_type": "book",
        "marketplace": "Amazon"  # Override default
    }
)
```

### Advanced Template Metric with Jinja2

```python
# Metric with conditional logic
assessment_metric = Metric(
    name="student_assessment",
    criteria="""
    Evaluate this {{ work_type }} from a {{ level }} student.
    
    {% if subject == 'STEM' %}
    Focus on:
    - Technical accuracy
    - Problem-solving approach
    - Use of formulas and calculations
    {% else %}
    Focus on:
    - Creativity and expression
    - Critical thinking
    - Argumentation quality
    {% endif %}
    
    Consider that this is {{ timepoint }} in the semester.
    """,
    scale=(0, 100),
    rubric="""
    {% if level == 'graduate' %}
    90-100: Publication-quality work
    80-89: Strong graduate-level work
    70-79: Acceptable with revisions needed
    Below 70: Does not meet graduate standards
    {% else %}
    90-100: Exceptional undergraduate work
    80-89: Very good understanding
    70-79: Satisfactory
    60-69: Passing but needs improvement
    Below 60: Failing
    {% endif %}
    """,
    required_vars=["work_type", "level", "subject", "timepoint"],
    template_engine="jinja2"
)
```

## Dynamic Evaluation Patterns

### Context-Aware Evaluation

```python
async def evaluate_customer_response(
    response: str,
    customer_type: str,
    issue_severity: str,
    channel: str
):
    """Evaluate response based on customer context."""
    
    # Adjust criteria based on severity
    urgency_phrase = "immediate resolution" if issue_severity == "high" else "timely assistance"
    
    result = await judge.evaluate(
        response=response,
        criteria="""Evaluate this {channel} response to a {customer_type} customer.
        The response should provide {urgency_phrase} for their {issue_severity} priority issue.""",
        template_vars={
            "channel": channel,
            "customer_type": customer_type,
            "urgency_phrase": urgency_phrase,
            "issue_severity": issue_severity
        },
        scale=(1, 10),
        rubric={
            10: f"Perfect response for {customer_type} via {channel}",
            5: "Adequate but could be improved",
            1: f"Inappropriate for {channel} communication"
        }
    )
    
    return result
```

### Multi-Language Support

```python
code_review_template = Metric(
    name="multilang_code_review",
    criteria="Review this {language} code for {purpose}",
    rubric="""
    10: Excellent {language} code - idiomatic and efficient
    7: Good code following most {language} conventions
    4: Functional but not idiomatic {language}
    1: Poor {language} code with major issues
    """,
    system_prompt="You are an expert {language} developer.",
    required_vars=["language", "purpose"]
)

# Use for different languages
python_result = await judge.evaluate(
    response=python_code,
    metric=code_review_template,
    template_vars={"language": "Python", "purpose": "data analysis"}
)

rust_result = await judge.evaluate(
    response=rust_code,
    metric=code_review_template,
    template_vars={"language": "Rust", "purpose": "systems programming"}
)
```

### Adaptive Rubrics

```python
# Rubric that adapts to scale
adaptive_metric = Metric(
    name="adaptive_quality",
    criteria="Evaluate {content_type} quality",
    template_engine="jinja2",
    rubric="""
    {% if scale_max == 5 %}
    5: Exceptional {content_type}
    4: Good quality
    3: Acceptable
    2: Below average
    1: Poor quality
    {% elif scale_max == 10 %}
    9-10: Outstanding {content_type}
    7-8: Very good
    5-6: Average
    3-4: Below average
    1-2: Poor quality
    {% else %}
    {{ scale_max }}: Perfect {content_type}
    {{ scale_max * 0.5 }}: Average
    0: Completely inadequate
    {% endif %}
    """,
    template_vars={"scale_max": 10}  # Default
)
```

## Template Variable Validation

### Required Variables

```python
# Define required variables
metric = Metric(
    name="context_eval",
    criteria="Evaluate {doc_type} for {audience} regarding {topic}",
    required_vars=["doc_type", "audience", "topic"]
)

# This will raise an error - missing 'topic'
try:
    result = await judge.evaluate(
        response="...",
        metric=metric,
        template_vars={"doc_type": "article", "audience": "general"}
    )
except InvalidInputError as e:
    print(f"Error: {e}")  # "Missing required template variables: topic"
```

### Validating Templates

```python
from vllm_judge.templating import TemplateProcessor

# Check what variables a template needs
template = "Evaluate {doc_type} for {audience} considering {aspects}"
required = TemplateProcessor.get_required_vars(template)
print(f"Required variables: {required}")  # {'doc_type', 'audience', 'aspects'}

# Validate before use
provided = {"doc_type": "essay", "audience": "students"}
missing = required - set(provided.keys())
if missing:
    print(f"Missing variables: {missing}")
```

## Best Practices

### 1. Use Descriptive Variable Names

```python
# Good - clear what each variable represents
template_vars = {
    "document_type": "technical specification",
    "target_audience": "senior engineers",
    "company_standards": "ISO 9001"
}

# Avoid - ambiguous names
template_vars = {
    "type": "tech",
    "level": "senior",
    "std": "ISO"
}
```

### 2. Provide Defaults for Optional Variables

```python
metric = Metric(
    name="flexible_eval",
    criteria="Evaluate {content} for {audience} with {strictness} standards",
    template_vars={
        "strictness": "moderate"  # Default
    },
    required_vars=["content", "audience"]  # Only these required
)
```

## Real-World Examples

### E-commerce Review Analysis

```python
# Template for different product categories
product_review_metric = Metric(
    name="product_review_analysis",
    criteria="""
    Analyze this {product_category} review for:
    - Authenticity (real customer vs fake)
    - Helpfulness to other {customer_segment} shoppers
    - Coverage of key {product_category} features: {key_features}
    - Balanced perspective (pros and cons)
    """,
    scale=(1, 10),
    rubric={
        10: "Exceptional review that perfectly helps {customer_segment} buyers",
        7: "Good review with useful information for {product_category}",
        4: "Basic review lacking important details",
        1: "Unhelpful or potentially fake review"
    },
    required_vars=["product_category", "customer_segment", "key_features"]
)

# Analyze electronics review
result = await judge.evaluate(
    response="This smartphone has amazing battery life...",
    metric=product_review_metric,
    template_vars={
        "product_category": "electronics",
        "customer_segment": "tech-savvy",
        "key_features": "battery, camera, performance, build quality"
    }
)
```

### Multi-Stage Document Review

```python
async def document_review_pipeline(document: str, doc_metadata: dict):
    """Multi-stage document review with progressive templates."""
    
    stages = [
        {
            "name": "initial_screen",
            "criteria": "Check if this {doc_type} meets basic {organization} standards",
            "pass_threshold": 6
        },
        {
            "name": "detailed_review",
            "criteria": """Review this {doc_type} for:
            - Compliance with {organization} {year} guidelines
            - Appropriate tone for {audience}
            - Completeness of required sections: {required_sections}""",
            "pass_threshold": 7
        },
        {
            "name": "final_approval",
            "criteria": """Final review of {doc_type} for {department} publication.
            Ensure it represents {organization} values and meets all {compliance_framework} requirements.""",
            "pass_threshold": 8
        }
    ]
    
    base_vars = {
        "doc_type": doc_metadata["type"],
        "organization": doc_metadata["org"],
        "year": "2024",
        **doc_metadata
    }
    
    for stage in stages:
        result = await judge.evaluate(
            response=document,
            criteria=stage["criteria"],
            template_vars=base_vars,
            scale=(1, 10)
        )
        
        print(f"{stage['name']}: {result.score}/10")
        
        if result.score < stage["pass_threshold"]:
            return {
                "passed": False,
                "failed_at": stage["name"],
                "feedback": result.reasoning
            }
    
    return {"passed": True, "scores": [s["name"] for s in stages]}
```

### API Documentation Evaluation

```python
# Comprehensive API docs evaluation
api_docs_metric = Metric(
    name="api_documentation",
    criteria="""
    Evaluate {api_type} API documentation for {api_name}:
    
    Required sections:
    - Authentication ({auth_type})
    - Endpoints ({endpoint_count} endpoints)
    - Request/Response formats
    - Error handling
    - Rate limiting
    
    Code examples should be in: {languages}
    
    {% if versioned %}
    Check version compatibility notes for v{version}
    {% endif %}
    
    {% if has_webhooks %}
    Verify webhook documentation completeness
    {% endif %}
    """,
    template_engine="jinja2",
    scale=(1, 10),
    required_vars=["api_type", "api_name", "auth_type", "endpoint_count", "languages"],
    template_vars={
        "versioned": False,
        "has_webhooks": False
    }
)

# Evaluate REST API docs
result = await judge.evaluate(
    response=api_documentation,
    metric=api_docs_metric,
    template_vars={
        "api_type": "REST",
        "api_name": "Payment Gateway",
        "auth_type": "OAuth 2.0",
        "endpoint_count": 25,
        "languages": "Python, JavaScript, Ruby",
        "versioned": True,
        "version": "2.0"
    }
)
```

## Troubleshooting Templates

### Common Issues

1. **Missing Variables**
```python
# Error: Missing required template variables
try:
    result = await judge.evaluate(
        response="...",
        criteria="Evaluate {missing_var}",
        template_vars={}  # Forgot to provide variables
    )
except InvalidInputError as e:
    print(f"Error: {e}")
    # Fix: Provide all required variables
```

2. **Typos in Variable Names**
```python
# Wrong variable name
template_vars = {"reponse_type": "email"}  # Typo: reponse vs response

# Template expects {response_type}
criteria = "Evaluate this {response_type}"  # Will fail
```

3. **Incorrect Template Engine**
```python
# Using Jinja2 syntax with format engine
result = await judge.evaluate(
    criteria="{% if condition %}...{% endif %}",  # Jinja2 syntax
    template_engine="format"  # Wrong engine!
)
# Fix: Use template_engine="jinja2"
```

### Debugging Templates

```python
# Test template rendering
from vllm_judge.templating import TemplateProcessor

template = "Evaluate {doc_type} for {audience}"
vars = {"doc_type": "report", "audience": "executives"}

# Preview the rendered template
rendered = TemplateProcessor.apply_template(template, vars)
print(f"Rendered: {rendered}")
# Output: "Evaluate report for executives"

# Check required variables
required = TemplateProcessor.get_required_vars(template)
print(f"Required: {required}")
# Output: {'doc_type', 'audience'}
```

## Performance Considerations

### Template Caching

When using the same template repeatedly:

```python
# Create metric once, reuse many times
metric = Metric(
    name="cached_evaluation",
    criteria="Complex template with {var1} and {var2}",
    # ... other settings
)

# Register for reuse
judge.register_metric(metric)

# Use many times efficiently
for item in items_to_evaluate:
    result = await judge.evaluate(
        response=item["response"],
        metric="cached_evaluation",  # Reference by name
        template_vars={
            "var1": item["var1"],
            "var2": item["var2"]
        }
    )
```

### Batch Processing with Templates

```python
# Prepare batch with templates
batch_data = [
    {
        "response": doc1,
        "criteria": "Evaluate {doc_type} quality",
        "template_vars": {"doc_type": "report"}
    },
    {
        "response": doc2,
        "criteria": "Evaluate {doc_type} quality",
        "template_vars": {"doc_type": "proposal"}
    }
]

# Process efficiently
results = await judge.batch_evaluate(batch_data)
```

## Summary

Template variables provide powerful flexibility for:


- **Reusable evaluations** across different contexts

- **Dynamic criteria** that adapt to your needs

- **Consistent evaluation** with parameterized variation

- **Complex logic** with Jinja2 templates

Key takeaways:
1. Start with simple `{variable}` substitution
2. Use template metrics for reusability
3. Leverage Jinja2 for complex logic
4. Always validate required variables
5. Document your templates clearly

<!-- ## Next Steps

- Explore [Batch Processing](batch.md) with templates
- See [Advanced Usage](advanced.md) for complex patterns
- Check [Examples](../examples/education.md) using templates -->