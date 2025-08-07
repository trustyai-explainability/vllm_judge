import json
import re
from typing import Union, Dict, List, Optional, Tuple, Any, Callable

from vllm_judge.models import JudgeConfig, EvaluationResult, Metric, BatchResult, TemplateEngine, ModelSpecificMetric
from vllm_judge.client import VLLMClient
from vllm_judge.prompt_builder import PromptBuilder
from vllm_judge.batch import BatchProcessor
from vllm_judge.builtin_metrics import BUILTIN_METRICS
from vllm_judge.templating import TemplateProcessor
from vllm_judge.exceptions import (
    ParseError,
    InvalidInputError,
    MetricNotFoundError,
    VLLMJudgeError
)
import logging

logger = logging.getLogger(__name__)

DECISION_ALTERNATIVES = ["label", "judgment", "result", "output", "prediction", "response"]
REASONING_ALTERNATIVES = ["reason", "explanation", "justification", "rationale", "thought", "thinking"]
SCORE_ALTERNATIVES = ["confidence", "probability", "prob", "grade", "rating", "score_value", "value"]
REQUIRED_FIELDS = ["decision", "reasoning"]

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
        content: Union[str, Dict[str, str], List[Dict[str, str]]],
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
        sampling_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Universal evaluation method that adapts to use case.
        
        Args:
            content: String for single evaluation, list of dicts for conversation, dict {"a": ..., "b": ...} for comparison
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
            sampling_params: Optional sampling parameters for vLLM
            **kwargs: Additional instructions for the model (e.g., with `additional_instructions` key)
            
        Returns:
            EvaluationResult with decision, reasoning, and optional score
            
        Raises:
            InvalidInputError: If inputs are invalid or template vars missing
            MetricNotFoundError: If metric name not found
            ParseError: If unable to parse model response
        """
        # Resolve metric if string
        resolved_metric = self._resolve_metric(metric)
        
        # Handle model-specific metrics early
        if isinstance(resolved_metric, ModelSpecificMetric):
            return await self._evaluate_model_specific_metric(
                resolved_metric, content, sampling_params
            )
        
        # Process normal evaluation
        evaluation_params = self._prepare_evaluation_params(
            resolved_metric, criteria, rubric, scale, examples, 
            system_prompt, template_engine
        )
        
        # Process templates
        processed_params = self._process_templates(
            evaluation_params, template_vars, input, context
        )
        
        # Build and execute evaluation
        return await self._execute_evaluation(
            content, processed_params, sampling_params, **kwargs
        )
    
    def _resolve_metric(self, metric: Union[Metric, str, None]) -> Optional[Metric]:
        """Resolve metric string to Metric object."""
        if metric and isinstance(metric, str):
            return self.get_metric(metric)
        return metric
    
    async def _evaluate_model_specific_metric(
        self, 
        metric: ModelSpecificMetric, 
        content: Union[str, List[Dict[str, str]]], 
        sampling_params: Optional[Dict[str, Any]]
    ) -> EvaluationResult:
        """Handle evaluation for model-specific metrics."""
        # Validate content for model-specific metrics
        if isinstance(content, dict):
            raise InvalidInputError(
                "Model-specific metrics only support string and list of dicts as content for now"
            )
        
        if isinstance(content, list) and len(content) == 0:
            raise InvalidInputError("Conversation content cannot be an empty list.")
        
        # Validate conversation format
        is_conversation = (
            isinstance(content, list) and 
            all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in content)
        )
        if isinstance(content, list) and not is_conversation:
            raise InvalidInputError(
                "Invalid content structure for conversation. "
                "Please provide a list of dicts with role and content fields."
            )
        
        # Prepare messages
        if is_conversation:
            messages = content
        else:
            messages = [{"role": "user", "content": content}]

        logger.info(
            f"We assume you're using {metric.model_pattern} type model. "
            f"If not, please do not use this metric and use a normal metric instead."
        )
        
        # Get model response and parse
        llm_response = await self._call_model(messages, sampling_params, return_choices=False)
        return metric.parser_func(llm_response)
    
    def _prepare_evaluation_params(
        self,
        metric: Optional[Metric],
        criteria: Optional[str],
        rubric: Union[str, Dict[Union[int, float], str], None],
        scale: Optional[Tuple[int, int]],
        examples: Optional[List[Dict[str, Any]]],
        system_prompt: Optional[str],
        template_engine: Union[str, TemplateEngine]
    ) -> Dict[str, Any]:
        """Prepare evaluation parameters, merging metric defaults with user overrides."""
        params = {
            "criteria": criteria,
            "rubric": rubric,
            "scale": scale,
            "examples": examples,
            "system_prompt": system_prompt,
            "template_engine": template_engine,
            "metric_template_vars": {}
        }
        
        if metric:
            # Use metric defaults but allow overrides
            params["criteria"] = criteria or metric.criteria
            params["rubric"] = rubric or metric.rubric
            params["scale"] = scale or metric.scale
            params["examples"] = examples or metric.examples
            params["system_prompt"] = system_prompt or metric.system_prompt
            params["metric_template_vars"] = metric.template_vars
            if metric.template_engine:
                params["template_engine"] = metric.template_engine
        
        # Validate required parameters
        if not params["criteria"]:
            raise InvalidInputError("Either 'criteria' or 'metric' must be provided")
        
        return params
    
    def _process_templates(
        self,
        params: Dict[str, Any],
        template_vars: Optional[Dict[str, Any]],
        input_text: Optional[str],
        context: Optional[str]
    ) -> Dict[str, Any]:
        """Process all template variables and return processed parameters."""
        # Determine template engine
        engine = TemplateEngine(params["template_engine"])
        
        # Merge template variables (metric defaults + user provided)
        all_template_vars = {**params["metric_template_vars"], **(template_vars or {})}
        if input_text:
            all_template_vars["input"] = input_text
        
        # Process templates for all relevant fields
        template_fields = ["criteria", "rubric", "system_prompt"]
        processed = {}
        
        for field in template_fields:
            processed[field] = TemplateProcessor.apply_template(
                params[field], all_template_vars, engine, strict=True
            )
        
        # Process additional fields
        processed["context"] = TemplateProcessor.apply_template(
            context, all_template_vars, engine, strict=True
        )
        processed["input"] = TemplateProcessor.apply_template(
            input_text, all_template_vars, engine, strict=True
        )
        
        # Copy other parameters
        processed.update({
            "scale": params["scale"],
            "examples": params["examples"],
            "template_vars": all_template_vars,
            "template_engine": engine
        })
        
        return processed
    
    async def _execute_evaluation(
        self,
        content: Union[str, Dict[str, str], List[Dict[str, str]]],
        params: Dict[str, Any],
        sampling_params: Optional[Dict[str, Any]],
        **kwargs
    ) -> EvaluationResult:
        """Execute the evaluation with processed parameters."""
        # Build messages
        messages = PromptBuilder.build_messages(
            content=content,
            input=params["input"],
            criteria=params["criteria"],
            rubric=params["rubric"],
            scale=params["scale"],
            examples=params["examples"],
            system_prompt=params["system_prompt"],
            context=params["context"],
            **kwargs
        )
        
        # Get LLM response
        llm_response = await self._call_model(messages, sampling_params, return_choices=False)
        
        # Parse response
        result = self._parse_response(llm_response)
        
        # Add template info to metadata if used
        if params["template_vars"]:
            result.metadata["template_vars"] = params["template_vars"]
            result.metadata["template_engine"] = params["template_engine"].value
        
        return result
    
    async def _call_model(self, messages: List[Dict[str, str]], 
                          sampling_params: Optional[Dict[str, Any]] = None,
                          return_choices: bool = False) -> Union[str, List[Dict[str, Any]]]:
        """
        Call the model with the given messages.

        Args:
            messages: List of messages
            sampling_params: Sampling parameters
            return_choices: Whether to return choices

        Returns:
            str model response if return_choices is False, otherwise List[Dict[str, Any]]
        """
        if sampling_params and 'n' in sampling_params and sampling_params['n'] > 1:
            raise InvalidInputError("n > 1 is not supported for now")
        
        # Merge sampling params
        final_sampling_params = {**self.config.sampling_params}
        if sampling_params:
            final_sampling_params.update(sampling_params)
        try:
            if self.config.use_chat_api:
                llm_response = await self.client.chat_completion(
                    messages,
                    sampling_params=final_sampling_params,
                    return_choices=return_choices)
            else:
                prompt = PromptBuilder.format_messages_as_text(messages)
                llm_response = await self.client.completion(
                    prompt,
                    sampling_params=final_sampling_params,
                    return_choices=return_choices)
            return llm_response
        except Exception as e:
            raise VLLMJudgeError(f"Failed to get model response: {e}")

    
    def _parse_response(self, response: str) -> EvaluationResult:
        """
        Parse LLM response into EvaluationResult.
        
        Uses multiple parsing strategies in order of preference:
        1. Direct JSON parsing
        2. JSON extraction from markdown code blocks  
        3. Regex-based JSON structure detection
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed EvaluationResult
            
        Raises:
            ParseError: If unable to parse response or missing required fields
        """
        logger.debug(f"Parsing response: {response[:100]}...")
        
        # Try each parsing strategy
        parsing_strategies = [
            ("direct JSON", self._parse_direct_json),
            ("markdown JSON", self._parse_markdown_json), 
            ("regex JSON", self._parse_regex_json)
        ]
        
        data = None
        for strategy_name, strategy_func in parsing_strategies:
            data = strategy_func(response)
            if data is not None:
                logger.debug(f"Successfully parsed using {strategy_name}")
                break
        
        if data is None:
            raise ParseError(
                "Unable to extract valid JSON from response using any parsing strategy",
                raw_response=response
            )
        
        # Validate and normalize the data
        try:
            data = self._validate_and_normalize_data(data, response)
        except ParseError:
            raise  # Re-raise ParseError as-is
        except Exception as e:
            raise ParseError(
                f"Data validation failed: {e}",
                raw_response=response
            )
        
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
    
    def _parse_direct_json(self, response: str) -> Optional[Dict[str, Any]]:
        """Attempt direct JSON parsing."""
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parsing failed: {e}")
            return None

    def _parse_markdown_json(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from markdown code blocks."""
        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError as e:
                logger.debug(f"Markdown JSON parsing failed: {e}")
                return None
        return None
    
    def _parse_regex_json(self, response: str) -> Optional[Dict[str, Any]]:
        """Find JSON-like structure using regex."""
        # Look for JSON containing "decision" field - more flexible pattern
        json_match = re.search(r'(\{[^{}]*"decision"[^{}]*\})', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError as e:
                logger.debug(f"Regex JSON parsing failed: {e}")
                return None
        return None
    

    def _validate_and_normalize_data(self, data: Dict[str, Any], response: str) -> Dict[str, Any]:
        """Validate and normalize parsed data."""

        # Handle missing decision field
        if "decision" not in data:
            for alt_field in DECISION_ALTERNATIVES:
                if alt_field in data:
                    data["decision"] = data[alt_field]
                    logger.debug(f"Used '{alt_field}' field for decision")
                    break
        
        # Handle missing reasoning field with fallbacks
        if "reasoning" not in data:
            for alt_field in REASONING_ALTERNATIVES:
                if alt_field in data:
                    data["reasoning"] = data[alt_field]
                    logger.debug(f"Used '{alt_field}' field for reasoning")
                    break
            else:
                data["reasoning"] = "=== No reasoning provided ==="
                logger.warning("No reasoning field found, using default")
        
        # Handle missing score field with fallbacks
        if "score" not in data:
            for alt_field in SCORE_ALTERNATIVES:
                if alt_field in data:
                    data["score"] = data[alt_field]
                    logger.debug(f"Used '{alt_field}' field for score")
                    break
            else:
                data["score"] = None
                logger.warning("No score field found, setting to None")
        
        # Check for required fields
        for field in REQUIRED_FIELDS:
            if field not in data:
                raise ParseError(
                    f"Response missing required '{field}' field",
                    raw_response=response
                )
        
        # Validate field types
        if not isinstance(data["decision"], (str, bool, int,float)):
            logger.warning(f"Decision field has unexpected type: {type(data['decision'])}")
        
        if not isinstance(data["reasoning"], str):
            data["reasoning"] = str(data["reasoning"])
            logger.debug("Converted reasoning to string")
        
        # Validate score if present
        if "score" in data and data["score"] is not None:
            if not isinstance(data["score"], (int, float)):
                try:
                    data["score"] = float(data["score"])
                    logger.debug("Converted score to float")
                except (ValueError, TypeError):
                    logger.warning(f"Invalid score value: {data['score']}, setting to None")
                    data["score"] = None
        
        return data

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
        sampling_params: Optional[Dict[str, Any]] = None,
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
        return await processor.process(data, progress_callback, sampling_params, **default_kwargs)
    
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