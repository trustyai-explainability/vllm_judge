import asyncio
import time
from typing import List, Dict, Any, Callable, Optional, Union
from vllm_judge.models import EvaluationResult, BatchResult
from vllm_judge.exceptions import VLLMJudgeError


class BatchProcessor:
    """High-concurrency batch processing for evaluations."""
    
    def __init__(self, judge, max_concurrent: int = 50):
        """
        Initialize batch processor.
        
        Args:
            judge: Judge instance
            max_concurrent: Maximum concurrent requests
        """
        self.judge = judge
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self.progress_lock = asyncio.Lock()
        self.completed = 0
    
    async def process(
        self,
        data: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        **default_kwargs
    ) -> BatchResult:
        """
        Process batch of evaluations.
        
        Args:
            data: List of evaluation inputs
            progress_callback: Optional callback for progress updates
            **default_kwargs: Default parameters for all evaluations
            
        Returns:
            BatchResult with all results
        """
        start_time = time.time()
        self.completed = 0
        total = len(data)
        
        # Create tasks
        tasks = []
        for i, item in enumerate(data):
            # Merge default kwargs with item-specific kwargs
            eval_kwargs = {**default_kwargs, **item}
            
            task = self._process_item(
                eval_kwargs,
                i,
                total,
                progress_callback,
                sampling_params
            )
            tasks.append(task)
        
        # Process all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate statistics
        successful = sum(1 for r in results if isinstance(r, EvaluationResult))
        failed = total - successful
        duration = time.time() - start_time
        
        return BatchResult(
            results=results,
            total=total,
            successful=successful,
            failed=failed,
            duration_seconds=duration
        )
    
    async def _process_item(
        self,
        eval_kwargs: Dict[str, Any],
        index: int,
        total: int,
        progress_callback: Optional[Callable],
        sampling_params: Optional[Dict[str, Any]]
    ) -> Union[EvaluationResult, Exception]:
        """Process single item with concurrency control."""
        async with self.semaphore:
            try:
                # Extract response from kwargs
                content = eval_kwargs.pop('content', None)
                if not content:
                    raise ValueError(f"Item {index} missing 'content' field")
                
                # Perform evaluation
                result = await self.judge.evaluate(content=content, sampling_params=sampling_params, **eval_kwargs)
                
                # Update progress
                async with self.progress_lock:
                    self.completed += 1
                    if progress_callback:
                        progress_callback(self.completed, total)
                
                # Add index to metadata
                result.metadata['batch_index'] = index
                return result
                
            except Exception as e:
                # Update progress even for failures
                async with self.progress_lock:
                    self.completed += 1
                    if progress_callback:
                        progress_callback(self.completed, total)
                
                # Return exception with context
                error = VLLMJudgeError(f"Item {index} failed: {str(e)}")
                error.batch_index = index
                error.original_error = e
                return error
    
    async def process_streaming(
        self,
        data: List[Dict[str, Any]],
        callback: Callable[[int, Union[EvaluationResult, Exception]], None],
        sampling_params: Optional[Dict[str, Any]] = None,
        **default_kwargs
    ):
        """
        Process batch with streaming results.
        
        Args:
            data: List of evaluation inputs
            callback: Called with (index, result) as results complete
            **default_kwargs: Default parameters for all evaluations
        """
        async def process_and_callback(item, index):
            result = await self._process_item(
                {**default_kwargs, **item},
                index,
                len(data),
                None,
                sampling_params
            )
            callback(index, result)
            return result
        
        tasks = [
            process_and_callback(item, i)
            for i, item in enumerate(data)
        ]
        
        # Process tasks as they complete
        for coro in asyncio.as_completed(tasks):
            await coro