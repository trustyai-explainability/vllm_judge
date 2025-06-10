"""
vLLM Judge - LLM-as-a-Judge evaluations for vLLM hosted models.

A lightweight library for evaluating text responses using self-hosted language models
via vLLM's OpenAI-compatible API.
"""

__version__ = "0.1.5"

from vllm_judge.judge import Judge
from vllm_judge.models import (
    JudgeConfig,
    EvaluationResult,
    Metric,
    BatchResult,
    TemplateEngine,
    ModelSpecificMetric
)
from vllm_judge.templating import TemplateProcessor
from vllm_judge.metrics import (
    # General metrics
    HELPFULNESS,
    ACCURACY,
    CLARITY,
    CONCISENESS,
    RELEVANCE,
    COHERENCE,
    # Safety metrics
    SAFETY,
    TOXICITY,
    BIAS_DETECTION,
    LLAMA_GUARD_3_SAFETY,
    
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
    BUILTIN_METRICS,

    # Template metrics
    EDUCATIONAL_CONTENT_TEMPLATE,
    CODE_REVIEW_TEMPLATE,
    CUSTOMER_SERVICE_TEMPLATE,
    WRITING_QUALITY_TEMPLATE,
    PRODUCT_REVIEW_TEMPLATE,
    MEDICAL_INFO_TEMPLATE,
    API_DOCS_TEMPLATE,
    RAG_EVALUATION_TEMPLATE,
    AGENT_PERFORMANCE_TEMPLATE,

    # NLP metrics
    TRANSLATION_QUALITY,
    SUMMARIZATION_QUALITY,

)
from vllm_judge.exceptions import (
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
    "TemplateEngine",
    "TemplateProcessor",
    "ModelSpecificMetric",

    # Metrics
    "HELPFULNESS",
    "ACCURACY", 
    "CLARITY",
    "CONCISENESS",
    "RELEVANCE",
    "COHERENCE",
    "SAFETY",
    "TOXICITY",
    "BIAS_DETECTION",
    "LLAMA_GUARD_3_SAFETY",
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
    "EDUCATIONAL_CONTENT_TEMPLATE",
    "CODE_REVIEW_TEMPLATE",
    "CUSTOMER_SERVICE_TEMPLATE",
    "WRITING_QUALITY_TEMPLATE",
    "PRODUCT_REVIEW_TEMPLATE",
    "MEDICAL_INFO_TEMPLATE",
    "API_DOCS_TEMPLATE",
    "RAG_EVALUATION_TEMPLATE",
    "AGENT_PERFORMANCE_TEMPLATE",
    "TRANSLATION_QUALITY",
    "SUMMARIZATION_QUALITY",

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