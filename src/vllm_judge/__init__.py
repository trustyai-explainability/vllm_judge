"""
vLLM Judge - LLM-as-a-Judge evaluations for vLLM hosted models.

A lightweight library for evaluating text responses using self-hosted language models
via vLLM's OpenAI-compatible API.
"""

__version__ = "0.1.0"

from .judge import Judge
from .models import (
    JudgeConfig,
    EvaluationResult,
    Metric,
    BatchResult
)
from .metrics import (
    # General metrics
    HELPFULNESS,
    ACCURACY,
    CLARITY,
    CONCISENESS,
    RELEVANCE,
    
    # Safety metrics
    SAFETY,
    TOXICITY,
    
    # Code metrics
    CODE_QUALITY,
    CODE_SECURITY,
    
    # Content metrics
    CREATIVITY,
    PROFESSIONALISM,
    EDUCATIONAL_VALUE,
    
    # Comparison metrics
    PREFERENCE,
    
    # Binary metrics
    APPROPRIATE,
    FACTUAL,
    
    # Domain metrics
    MEDICAL_ACCURACY,
    LEGAL_APPROPRIATENESS,
    
    # Utility
    BUILTIN_METRICS
)
from .exceptions import (
    VLLMJudgeError,
    ConfigurationError,
    ConnectionError,
    TimeoutError,
    ParseError,
    MetricNotFoundError,
    InvalidInputError,
    RetryExhaustedError
)

__all__ = [
    # Main classes
    "Judge",
    "JudgeConfig",
    "EvaluationResult",
    "Metric",
    "BatchResult",
    
    # Metrics
    "HELPFULNESS",
    "ACCURACY", 
    "CLARITY",
    "CONCISENESS",
    "RELEVANCE",
    "SAFETY",
    "TOXICITY",
    "CODE_QUALITY",
    "CODE_SECURITY",
    "CREATIVITY",
    "PROFESSIONALISM",
    "EDUCATIONAL_VALUE",
    "PREFERENCE",
    "APPROPRIATE",
    "FACTUAL",
    "MEDICAL_ACCURACY",
    "LEGAL_APPROPRIATENESS",
    "BUILTIN_METRICS",
    
    # Exceptions
    "VLLMJudgeError",
    "ConfigurationError",
    "ConnectionError",
    "TimeoutError",
    "ParseError",
    "MetricNotFoundError",
    "InvalidInputError",
    "RetryExhaustedError"
]