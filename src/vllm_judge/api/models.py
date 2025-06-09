from typing import Union, Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
from datetime import datetime


class EvaluateRequest(BaseModel):
    """Request model for single evaluation."""
    content: Union[str, Dict[str, str]] = Field(
        ..., 
        description="Content to evaluate (string or dict with 'a'/'b' for comparison)",
        examples=["This is a response", {"a": "Response A", "b": "Response B"}]
    )
    input: Optional[str] = Field(
        None,
        description="Optional input/question/prompt that the content responds to",
        examples=["What is the capital of France?", "Write a function to sort a list"]
    )
    criteria: Optional[str] = Field(
        None, description="What to evaluate for"
    )
    rubric: Optional[Union[str, Dict[Union[int, float], str]]] = Field(
        None, description="Evaluation guide"
    )
    scale: Optional[List[int]] = Field(
        None, description="Numeric scale as [min, max]"
    )
    metric: Optional[str] = Field(
        None, description="Pre-defined metric name"
    )
    context: Optional[str] = Field(
        None, description="Additional context"
    )
    system_prompt: Optional[str] = Field(
        None, description="Custom system prompt"
    )
    examples: Optional[List[Dict[str, Any]]] = Field(
        None, description="Few-shot examples"
    )
    template_vars: Optional[Dict[str, Any]] = Field(
        None, description="Template variables to substitute"
    )
    template_engine: Optional[str] = Field(
        None, description="Template engine to use ('format' or 'jinja2'), default is 'format'"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "Python is a high-level programming language...",
                "criteria": "technical accuracy for {audience}",
                "template_vars": {"audience": "beginners"},
                "scale": [1, 10]
            }
        }


class BatchEvaluateRequest(BaseModel):
    """Request model for batch evaluation."""
    data: List[Dict[str, Any]] = Field(
        ..., description="List of evaluation inputs"
    )
    max_concurrent: Optional[int] = Field(
        None, description="Maximum concurrent requests"
    )
    default_criteria: Optional[str] = Field(
        None, description="Default criteria for all evaluations"
    )
    default_metric: Optional[str] = Field(
        None, description="Default metric for all evaluations"
    )


class AsyncBatchRequest(BaseModel):
    """Request model for async batch evaluation."""
    data: List[Dict[str, Any]] = Field(
        ..., description="List of evaluation inputs"
    )
    callback_url: Optional[str] = Field(
        None, description="URL to POST results when complete"
    )
    max_concurrent: Optional[int] = Field(
        None, description="Maximum concurrent requests"
    )


class EvaluationResponse(BaseModel):
    """Response model for evaluation results."""
    decision: Union[str, bool, int, float]
    reasoning: str
    score: Optional[float]
    metadata: Dict[str, Any] = {}
    
    # API-specific fields
    evaluation_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    duration_ms: Optional[int] = None


class BatchResponse(BaseModel):
    """Response model for batch results."""
    total: int
    successful: int
    failed: int
    success_rate: float
    duration_seconds: float
    results: List[Union[EvaluationResponse, Dict[str, str]]]


class AsyncBatchResponse(BaseModel):
    """Response model for async batch initiation."""
    job_id: str
    status: str = "pending"
    total_items: int
    created_at: datetime
    estimated_duration_seconds: Optional[float] = None


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: Dict[str, int]  # {"completed": 50, "total": 100}
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result_url: Optional[str] = None
    error: Optional[str] = None


class MetricInfo(BaseModel):
    """Information about a metric."""
    name: str
    criteria: str
    has_scale: bool
    scale: Optional[Tuple[int, int]] = None
    has_rubric: bool
    rubric_type: Optional[str] = None  # "string" or "dict"
    has_examples: bool
    example_count: int = 0
    has_system_prompt: bool
    has_template_vars: bool = False
    template_vars: Optional[Dict[str, Any]] = None
    required_vars: Optional[List[str]] = None
    template_engine: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str
    model: str
    base_url: str
    uptime_seconds: float
    total_evaluations: int
    active_connections: int
    metrics_available: int


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)