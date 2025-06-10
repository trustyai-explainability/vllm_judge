from typing import Optional, Any, Dict, Union, List, Tuple, Callable
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum


class TemplateEngine(str, Enum):
    """Supported template engines."""
    FORMAT = "format"
    JINJA2 = "jinja2"


class EvaluationResult(BaseModel):
    """Standard output format for ALL evaluations."""
    decision: Union[str, bool, int, float] = Field(
        ..., description="The judgment (e.g., score, class, 'response_a')"
    )
    reasoning: str = Field(
        ..., description="Explanation for the decision"
    )
    score: Optional[float] = Field(
        None, description="Numeric score if applicable"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional information"
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "decision": "PROFESSIONAL",
                    "reasoning": "The response demonstrates strong professional tone...",
                    "score": 8.5,
                    "metadata": {"model": "llama-3-70b", "latency_ms": 450}
                },
                {
                    "decision": "response_a",
                    "reasoning": "Response A provides more comprehensive coverage...",
                    "score": None,
                    "metadata": {"comparison_type": "pairwise"}
                }
            ]
        }


class JudgeConfig(BaseModel):
    """Configuration for Judge client."""
    # Connection settings
    base_url: str = Field(..., description="vLLM server URL (e.g., http://localhost:8000)")
    model: str = Field(..., description="Model name/path")
    api_key: str = Field("dummy", description="API key (usually 'dummy' for vLLM)")
    
    # API settings
    use_chat_api: bool = Field(True, description="Use chat completions endpoint")
    timeout: float = Field(30.0, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum retry attempts")
    retry_delay: float = Field(1.0, description="Initial retry delay in seconds")
    
    # Model parameters
    temperature: float = Field(0.0, description="Sampling temperature")
    max_tokens: int = Field(256, description="Maximum tokens in response")
    # top_p: float = Field(0.95, description="Top-p sampling")
    
    # Batch settings
    max_concurrent: int = Field(50, description="Maximum concurrent requests")
        
    @staticmethod
    def _validate_url(url: str) -> str:
        if not url.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")
        return url.rstrip('/').removesuffix('/v1')

    @field_validator('base_url')
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        """Ensure base_url is properly formatted."""
        return cls._validate_url(v)
    
    @classmethod
    def from_url(cls, url: str, model: Optional[str] = None, **kwargs):
        """Convenience constructor."""
        url = cls._validate_url(url)
        if not model:
            from vllm_judge.client import detect_model_sync
            model = detect_model_sync(url)
        return cls(base_url=url, model=model, **kwargs)


class Metric:
    """Reusable evaluation configuration."""
    
    def __init__(
        self,
        name: str,
        criteria: str,
        rubric: Union[str, Dict[Union[int, float], str]] = None,
        scale: Optional[Tuple[int, int]] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        template_vars: Optional[Dict[str, Any]] = None,
        required_vars: Optional[List[str]] = None,
        template_engine: Union[str, TemplateEngine] = TemplateEngine.FORMAT,
        additional_instructions: Optional[str] = None
    ):
        """
        Initialize a reusable metric.
        
        Args:
            name: Metric identifier
            criteria: What to evaluate for (can contain template variables)
            rubric: Evaluation guide (can contain template variables)
            scale: Optional numeric scale (min, max)
            examples: Optional few-shot examples
            system_prompt: Optional custom system message (can contain template variables)
            template_vars: Default template variable values
            required_vars: List of required template variables, these are variables that are required to be provided by the user for every evaluation
            template_engine: Template engine to use ('format' or 'jinja2'), default is 'format'
        """
        self.name = name
        self.criteria = criteria
        self.rubric = rubric
        self.scale = scale
        # TODO: Create a dedicated class for examples for better handling
        self.examples = examples or []
        self.system_prompt = system_prompt
        self.template_vars = template_vars or {}
        self.required_vars = required_vars or []
        self.template_engine = TemplateEngine(template_engine)
        self.additional_instructions = additional_instructions
        # Auto-detect required variables if not specified
        if not self.required_vars and self.template_engine == TemplateEngine.FORMAT:
            self._auto_detect_required_vars()
    
    def _auto_detect_required_vars(self):
        """Auto-detect required variables from format strings."""
        import string
        
        texts_to_check = [self.criteria]
        if isinstance(self.rubric, str):
            texts_to_check.append(self.rubric)
        elif isinstance(self.rubric, dict):
            texts_to_check.extend(str(v) for v in self.rubric.values())
        if self.system_prompt:
            texts_to_check.append(self.system_prompt)
        
        all_vars = set()
        for text in texts_to_check:
            try:
                # Parse format string to find variable names
                formatter = string.Formatter()
                for _, field_name, _, _ in formatter.parse(text):
                    if field_name:
                        all_vars.add(field_name)
            except:
                pass  # If parsing fails, skip auto-detection
        
        # Required vars are those not in default template_vars
        self.required_vars = list(all_vars - set(self.template_vars.keys()))
    
    def __repr__(self):
        return f"Metric(name='{self.name}', criteria='{self.criteria}', template_engine='{self.template_engine}')"

# Base class for model-specific metrics
class ModelSpecificMetric(Metric):
    """Metric that bypasses our prompt formatting."""
    
    def __init__(self, name: str, model_pattern: str, parser_func: Callable[[str], EvaluationResult]):
        super().__init__(name=name, criteria="model-specific evaluation")
        self.model_pattern = model_pattern
        self.parser_func = parser_func
        # self.is_model_specific = True  # Flag for special handling

class BatchResult(BaseModel):
    """Result of batch evaluation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    results: List[Union[EvaluationResult, Exception]] = Field(
        ..., description="List of results or exceptions"
    )
    total: int = Field(..., description="Total number of evaluations")
    successful: int = Field(..., description="Number of successful evaluations")
    failed: int = Field(..., description="Number of failed evaluations")
    duration_seconds: float = Field(..., description="Total processing time")
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful / self.total if self.total > 0 else 0.0
    
    def get_failures(self) -> List[Tuple[int, Exception]]:
        """Get list of (index, exception) for failed evaluations."""
        failures = []
        for i, result in enumerate(self.results):
            if isinstance(result, Exception):
                failures.append((i, result))
        return failures