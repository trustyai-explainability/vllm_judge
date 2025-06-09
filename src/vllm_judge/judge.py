import json
import re
from typing import Union, Dict, List, Optional, Tuple, Any, Callable

from vllm_judge.models import JudgeConfig, EvaluationResult, Metric, BatchResult, TemplateEngine, ModelSpecificMetric
from vllm_judge.client import VLLMClient
from vllm_judge.prompts import PromptBuilder
from vllm_judge.batch import BatchProcessor
from vllm_judge.metrics import BUILTIN_METRICS
from vllm_judge.templating import TemplateProcessor
from vllm_judge.exceptions import (
    ParseError,
    InvalidInputError,
    MetricNotFoundError,
    VLLMJudgeError
)
import logging

logger = logging.getLogger(__name__)


class Judge:
    """Main class for LLM-as-a-Judge evaluations."""
    
    def __init__(self, config: JudgeConfig):
        """
        Initialize Judge with configuration.
        
        Args:
            config: Judge configuration
        """
        self.config = config
        self.client = VLLMClient(config)
        self.metrics: Dict[str, Metric] = {}
    
    @classmethod
    def from_url(cls, base_url: str, model: Optional[str] = None, **kwargs) -> 'Judge':
        """
        Create Judge from URL.
        
        Args:
            base_url: vLLM server URL
            model: Model name (optional, can be auto-detected)
            **kwargs: Additional configuration
            
        Returns:
            Judge instance
        """
        config = JudgeConfig.from_url(base_url, model=model, **kwargs)
        return cls(config)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close client connections."""
        await self.client.close()
    
    async def evaluate(
        self,
        content: Union[str, Dict[str, str]],
        input: Optional[str] = None,
        criteria: str = None,
        rubric: Union[str, Dict[Union[int, float], str]] = None,
        scale: Optional[Tuple[int, int]] = None,
        examples: List[Dict[str, Any]] = None,
        metric: Union[Metric, str] = None,
        system_prompt: str = None,
        context: str = None,
        template_vars: Dict[str, Any] = None,
        template_engine: Union[str, TemplateEngine] = TemplateEngine.FORMAT,
        **kwargs
    ) -> EvaluationResult:
        """
        Universal evaluation method that adapts to use case.
        
        Args:
            content: String for single evaluation, dict {"a": ..., "b": ...} for comparison
            input: Optional input/question/prompt that the content is responding to
            criteria: What to evaluate for (can contain template variables)
            rubric: Instructions for evaluation, can be string or dict containing mapping of score to description (can contain template variables)
            scale: Optional numeric scale (min, max)
            examples: Optional few-shot examples
            metric: Pre-defined Metric object or registered metric name
            system_prompt: Optional custom system message (can contain template variables)
            context: Optional context for the evaluation
            template_vars: Variables to substitute in templates
            template_engine: Template engine to use ('format' or 'jinja2'), default is 'format'
            **kwargs: Additional parameters
            
        Returns:
            EvaluationResult with decision, reasoning, and optional score
            
        Raises:
            InvalidInputError: If inputs are invalid or template vars missing
            MetricNotFoundError: If metric name not found
            ParseError: If unable to parse model response
        """
        # Handle model-specific metrics
        if isinstance(metric, ModelSpecificMetric):
            assert isinstance(content, str), "Model-specific metrics only support string content for now"

            # logger.info(f"Evaluating model-specific metric {metric.name}.")
            logger.info(f"We assume you're using {metric.model_pattern} type model. If not, please do not use this metric and use a normal metric instead.")
            # Skip ALL our formatting
            messages = [{"role": "user", "content": content}]
            
            # vLLM applies model's chat template automatically
            llm_response = await self._call_model(messages)
            
            # Use metric's parser
            return metric.parser_func(llm_response)
        
        # Handle normal metrics
        # Handle metric parameter
        metric_template_vars = {}
        
        if metric:
            if isinstance(metric, str):
                metric = self.get_metric(metric)
            # Use metric defaults but allow overrides
            criteria = criteria or metric.criteria
            rubric = rubric or metric.rubric
            scale = scale or metric.scale
            examples = examples or metric.examples
            system_prompt = system_prompt or metric.system_prompt
            metric_template_vars = metric.template_vars
            if metric.template_engine:
                template_engine = metric.template_engine
        
        # Validate inputs
        if not criteria:
            raise InvalidInputError("Either 'criteria' or 'metric' must be provided")
        
        # Determine template engine
        engine = TemplateEngine(template_engine)
        
        # Merge template variables (metric defaults + user provided)
        all_template_vars = {**metric_template_vars, **(template_vars or {})}
        # Add input to template variables if provided
        if input:
            all_template_vars["input"] = input
        
        # Process templates
        criteria = TemplateProcessor.apply_template(
            criteria, all_template_vars, engine, strict=True
        )
        rubric = TemplateProcessor.apply_template(
            rubric, all_template_vars, engine, strict=True
        )
        system_prompt = TemplateProcessor.apply_template(
            system_prompt, all_template_vars, engine, strict=True
        )
        context = TemplateProcessor.apply_template(
            context, all_template_vars, engine, strict=True
        )
        input = TemplateProcessor.apply_template(
            input, all_template_vars, engine, strict=True
        )
        
        # Build messages
        messages = PromptBuilder.build_messages(
            content=content,
            input=input,
            criteria=criteria,
            rubric=rubric,
            scale=scale,
            examples=examples,
            system_prompt=system_prompt,
            context=context,
            **kwargs
        )
        
        # Get LLM response
        llm_response = await self._call_model(messages)
        
        # Parse response
        result = self._parse_response(llm_response)
        
        # Add template info to metadata if used
        if all_template_vars:
            result.metadata["template_vars"] = all_template_vars
            result.metadata["template_engine"] = engine.value
        
        return result
    
    async def _call_model(self, messages: List[Dict[str, str]]) -> str:
        """
        Call the model with the given messages.
        """
        try:
            if self.config.use_chat_api:
                llm_response = await self.client.chat_completion(messages)
            else:
                prompt = PromptBuilder.format_messages_as_text(messages)
                llm_response = await self.client.completion(prompt)
            return llm_response
        except Exception as e:
            raise VLLMJudgeError(f"Failed to get model response: {e}")

    
    def _parse_response(self, response: str) -> EvaluationResult:
        """
        Parse LLM response into EvaluationResult.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed EvaluationResult
            
        Raises:
            ParseError: If unable to parse response
        """
        # Try to parse as JSON
        try:
            # First attempt: direct JSON parsing
            data = json.loads(response.strip())
        except json.JSONDecodeError:
            # Second attempt: extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            else:
                # Third attempt: find JSON-like structure
                json_match = re.search(r'({[^{}]*"decision"[^{}]*})', response, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        raise ParseError(
                            "Failed to parse JSON from response",
                            raw_response=response
                        )
                else:
                    raise ParseError(
                        "No JSON structure found in response",
                        raw_response=response
                    )
        
        # Validate required fields
        if "decision" not in data:
            raise ParseError(
                "Response missing required 'decision' field",
                raw_response=response
            )
        
        if "reasoning" not in data:
            # Try to extract reasoning from other fields
            data["reasoning"] = data.get("reason", data.get("explanation", "No reasoning provided"))
        
        # Create result
        return EvaluationResult(
            decision=data["decision"],
            reasoning=data["reasoning"],
            score=data.get("score"),
            metadata={
                "model": self.config.model,
                "raw_response": response,
                **data.get("metadata", {})
            }
        )
    
    # Convenience methods
    async def score(
        self,
        criteria: str,
        content: str,
        input: Optional[str] = None,
        scale: Tuple[int, int] = (1, 10),
        **kwargs
    ) -> EvaluationResult:
        """
        Quick scoring evaluation.
        
        Args:
            criteria: What to evaluate
            content: Response to evaluate
            input: Optional input/question/prompt that the response addresses
            scale: Numeric scale (default 1-10)
            **kwargs: Additional parameters
            
        Returns:
            EvaluationResult with numeric score
        """
        return await self.evaluate(
            content=content,
            input=input,
            criteria=criteria,
            scale=scale,
            **kwargs
        )
    async def qa_evaluate(
        self,
        question: str,
        answer: str,
        criteria: str = "accuracy and completeness",
        scale: Tuple[int, int] = (1, 10),
        **kwargs
    ) -> EvaluationResult:
        """
        Convenience method for QA evaluation.
        
        Args:
            question: The question being answered
            answer: The answer to evaluate
            criteria: Evaluation criteria (default: "accuracy and completeness")
            scale: Numeric scale (default 1-10)
            **kwargs: Additional parameters
            
        Returns:
            EvaluationResult with QA assessment
        """
        return await self.evaluate(
            content=answer,
            input=question,
            criteria=criteria,
            scale=scale,
            **kwargs
        )
    
    async def compare(
        self,
        response_a: str,
        response_b: str,
        criteria: str,
        input: Optional[str] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Quick comparison evaluation.
        
        Args:
            response_a: First response
            response_b: Second response
            criteria: What to compare on
            input: Optional input/question that both responses address
            **kwargs: Additional parameters
            
        Returns:
            EvaluationResult with decision of 'response_a' or 'response_b'
        """
        return await self.evaluate(
            content={"a": response_a, "b": response_b},
            input=input,
            criteria=criteria,
            **kwargs
        )
    
    async def classify(
        self,
        content: str,
        categories: List[str],
        criteria: str = None,
        input: Optional[str] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Quick classification evaluation.
        
        Args:
            content: Content to classify
            categories: List of categories
            criteria: Classification criteria
            input: Optional input/question that the response addresses
            **kwargs: Additional parameters
            
        Returns:
            EvaluationResult with category decision
        """
        if not criteria:
            criteria = "appropriate category"
        
        rubric = f"Classify into one of these categories: {', '.join(categories)}"
        
        return await self.evaluate(
            content=content,
            input=input,
            criteria=criteria,
            rubric=rubric,
            **kwargs
        )
    
    # Metric management
    def register_metric(self, metric: Metric):
        """
        Register a metric for reuse.
        
        Args:
            metric: Metric to register
        """
        self.metrics[metric.name] = metric
    
    def get_metric(self, name: str) -> Metric:
        """
        Get registered metric by name.
        
        Args:
            name: Metric name
            
        Returns:
            Metric instance
            
        Raises:
            MetricNotFoundError: If metric not found
        """
        # Check user-registered metrics first
        if name in self.metrics:
            return self.metrics[name]
        
        # Check built-in metrics
        if name in BUILTIN_METRICS:
            return BUILTIN_METRICS[name]
        
        # List available metrics in error
        available = list(self.metrics.keys()) + list(BUILTIN_METRICS.keys())
        raise MetricNotFoundError(
            f"Metric '{name}' not found. Available metrics: {', '.join(available)}"
        )
    
    def list_metrics(self) -> List[str]:
        """List all available metric names."""
        return list(self.metrics.keys()) + list(BUILTIN_METRICS.keys())
    
    # Batch processing
    async def batch_evaluate(
        self,
        data: List[Dict[str, Any]],
        max_concurrent: int = None,
        progress_callback: Callable[[int, int], None] = None,
        **default_kwargs
    ) -> BatchResult:
        """
        Batch evaluation with high concurrency.
        
        Args:
            data: List of evaluation inputs (each must have 'content' key)
            max_concurrent: Maximum concurrent requests
            progress_callback: Optional callback for progress updates
            **default_kwargs: Default parameters for all evaluations
            
        Returns:
            BatchResult with all results
            
        Example:
            results = await judge.batch_evaluate([
                {"content": "Text 1", "criteria": "clarity"},
                {"content": "Paris", "input": "What is the capital of France?", "criteria": "accuracy"},
                {"content": {"a": "A", "b": "B"}, "criteria": "quality"},
                {"content": "Text 3", "metric": "safety"}
            ])
        """
        processor = BatchProcessor(self, max_concurrent or self.config.max_concurrent)
        return await processor.process(data, progress_callback, **default_kwargs)
    
    async def batch_score(
        self,
        responses: List[str],
        criteria: str,
        scale: Tuple[int, int] = (1, 10),
        **kwargs
    ) -> List[EvaluationResult]:
        """
        Convenience method for batch scoring.
        
        Args:
            responses: List of responses to score
            criteria: Scoring criteria
            scale: Numeric scale
            **kwargs: Additional parameters
            
        Returns:
            List of EvaluationResults
        """
        data = [
            {"response": resp, "criteria": criteria, "scale": scale, **kwargs}
            for resp in responses
        ]
        batch_result = await self.batch_evaluate(data)
        
        # Extract results, raising first error if any
        results = []
        for r in batch_result.results:
            if isinstance(r, Exception):
                raise r
            results.append(r)
        return results