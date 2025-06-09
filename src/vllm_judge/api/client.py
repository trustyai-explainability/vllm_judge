"""
HTTP client for vLLM Judge API.
"""
import asyncio
from typing import Union, Dict, List, Optional, Tuple, Any, AsyncIterator
import httpx
import websockets
import json

from vllm_judge.models import EvaluationResult, BatchResult
from vllm_judge.exceptions import VLLMJudgeError, ConnectionError
from vllm_judge.api.models import (
    EvaluateRequest,
    BatchEvaluateRequest,
    AsyncBatchRequest,
    MetricInfo
)


class JudgeClient:
    """HTTP client for vLLM Judge API."""
    
    def __init__(
        self,
        api_url: str,
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        """
        Initialize Judge API client.
        
        Args:
            api_url: Base URL of Judge API server
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.api_url = api_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = httpx.AsyncClient(
            base_url=self.api_url,
            timeout=httpx.Timeout(timeout)
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close HTTP session."""
        await self.session.aclose()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        try:
            response = await self.session.get("/health")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise ConnectionError(f"Health check failed: {e}")
    
    async def evaluate(
        self,
        content: Union[str, Dict[str, str]],
        input: Optional[str] = None,
        criteria: str = None,
        rubric: Union[str, Dict[Union[int, float], str]] = None,
        scale: Optional[Tuple[int, int]] = None,
        metric: str = None,
        context: str = None,
        system_prompt: str = None,
        examples: List[Dict[str, Any]] = None,
        template_vars: Dict[str, Any] = None,
        template_engine: str = "format",
        **kwargs
    ) -> EvaluationResult:
        """
        Perform single evaluation via API.
        
        Args:
            Same as Judge.evaluate() including template support
            
        Returns:
            EvaluationResult
        """
        request = EvaluateRequest(
            content=content,
            input=input,
            criteria=criteria,
            rubric=rubric,
            scale=list(scale) if scale else None,
            metric=metric,
            context=context,
            system_prompt=system_prompt,
            examples=examples,
            template_vars=template_vars,
            template_engine=template_engine
        )
        
        try:
            api_response = await self.session.post(
                "/evaluate",
                json=request.model_dump()
            )
            api_response.raise_for_status()
            data = api_response.json()
            
            return EvaluationResult(
                decision=data["decision"],
                reasoning=data["reasoning"],
                score=data.get("score"),
                metadata=data.get("metadata", {})
            )
            
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json().get("detail", str(e))
            raise VLLMJudgeError(f"Evaluation failed: {error_detail}")
        except httpx.HTTPError as e:
            raise ConnectionError(f"API request failed: {e}")
    
    async def batch_evaluate(
        self,
        data: List[Dict[str, Any]],
        max_concurrent: int = None,
        default_criteria: str = None,
        default_metric: str = None,
        **kwargs
    ) -> BatchResult:
        """
        Perform synchronous batch evaluation.
        
        Args:
            data: List of evaluation inputs
            max_concurrent: Maximum concurrent requests
            default_criteria: Default criteria for all evaluations
            default_metric: Default metric for all evaluations
            
        Returns:
            BatchResult
        """
        request = BatchEvaluateRequest(
            data=data,
            max_concurrent=max_concurrent,
            default_criteria=default_criteria,
            default_metric=default_metric
        )
        
        try:
            response = await self.session.post(
                "/batch",
                json=request.model_dump(),
                timeout=None  # No timeout for batch operations
            )
            response.raise_for_status()
            data = response.json()
            
            # Convert results
            results = []
            for r in data["results"]:
                if "error" in r:
                    results.append(VLLMJudgeError(r["error"]))
                else:
                    results.append(EvaluationResult(
                        decision=r["decision"],
                        reasoning=r["reasoning"],
                        score=r.get("score"),
                        metadata=r.get("metadata", {})
                    ))
            
            return BatchResult(
                results=results,
                total=data["total"],
                successful=data["successful"],
                failed=data["failed"],
                duration_seconds=data["duration_seconds"]
            )
            
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json().get("detail", str(e))
            raise VLLMJudgeError(f"Batch evaluation failed: {error_detail}")
        except httpx.HTTPError as e:
            raise ConnectionError(f"API request failed: {e}")
    
    async def async_batch_evaluate(
        self,
        data: List[Dict[str, Any]],
        callback_url: str = None,
        max_concurrent: int = None,
        poll_interval: float = 1.0
    ) -> BatchResult:
        """
        Start async batch evaluation and wait for completion.
        
        Args:
            data: List of evaluation inputs
            callback_url: Optional callback URL
            max_concurrent: Maximum concurrent requests
            poll_interval: Seconds between status checks
            
        Returns:
            BatchResult when complete
        """
        # Start async job
        request = AsyncBatchRequest(
            data=data,
            callback_url=callback_url,
            max_concurrent=max_concurrent
        )
        
        response = await self.session.post(
            "/batch/async",
            json=request.model_dump()
        )
        response.raise_for_status()
        job_data = response.json()
        job_id = job_data["job_id"]
        
        # Poll for completion
        while True:
            status = await self.get_job_status(job_id)
            
            if status["status"] == "completed":
                return await self.get_job_result(job_id)
            elif status["status"] == "failed":
                raise VLLMJudgeError(f"Job failed: {status.get('error', 'Unknown error')}")
            
            await asyncio.sleep(poll_interval)
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of async job."""
        response = await self.session.get(f"/jobs/{job_id}")
        response.raise_for_status()
        return response.json()
    
    async def get_job_result(self, job_id: str) -> BatchResult:
        """Get result of completed async job."""
        response = await self.session.get(f"/jobs/{job_id}/result")
        response.raise_for_status()
        data = response.json()
        
        # Convert to BatchResult
        results = []
        for r in data["results"]:
            if "error" in r:
                results.append(VLLMJudgeError(r["error"]))
            else:
                results.append(EvaluationResult(
                    decision=r["decision"],
                    reasoning=r["reasoning"],
                    score=r.get("score"),
                    metadata=r.get("metadata", {})
                ))
        
        return BatchResult(
            results=results,
            total=data["total"],
            successful=data["successful"],
            failed=data["failed"],
            duration_seconds=data["duration_seconds"]
        )
    
    async def list_metrics(self) -> List[MetricInfo]:
        """List all available metrics."""
        response = await self.session.get("/metrics")
        response.raise_for_status()
        return [MetricInfo(**m) for m in response.json()]
    
    async def get_metric(self, metric_name: str) -> Dict[str, Any]:
        """Get details of a specific metric."""
        response = await self.session.get(f"/metrics/{metric_name}")
        response.raise_for_status()
        return response.json()
    
    # Convenience methods matching Judge interface
    async def score(
        self,
        criteria: str,
        content: str,
        input: Optional[str] = None,
        scale: Tuple[int, int] = (1, 10),
        **kwargs
    ) -> EvaluationResult:
        """Quick scoring evaluation."""
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
        Convenience method for QA evaluation via API.
        
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
        """Quick comparison evaluation."""
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
        """Quick classification evaluation."""
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
    
    async def evaluate_streaming(
        self,
        content: Union[str, Dict[str, str]],
        input: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        WebSocket-based streaming evaluation.
        
        Yields partial results as they arrive.
        """
        ws_url = self.api_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/ws/evaluate"
        
        async with websockets.connect(ws_url) as websocket:
            # Send request
            request_data = {
                "content": content,
                "input": input,
                **kwargs
            }
            await websocket.send(json.dumps(request_data))
            
            # Receive result
            result_data = await websocket.recv()
            result = json.loads(result_data)
            
            if result["status"] == "success":
                yield json.dumps(result["result"])
            else:
                raise VLLMJudgeError(f"Streaming evaluation failed: {result.get('error')}")