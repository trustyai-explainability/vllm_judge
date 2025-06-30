import pytest
import asyncio
from unittest.mock import AsyncMock, Mock
from vllm_judge.batch import BatchProcessor
from vllm_judge.models import EvaluationResult, BatchResult


class TestBatchProcessor:
    """Test BatchProcessor functionality."""
    
    @pytest.fixture
    def mock_judge(self):
        """Mock Judge for batch processing tests."""
        judge = Mock()
        judge.evaluate = AsyncMock()
        judge.evaluate.return_value = EvaluationResult(
            decision="GOOD", reasoning="Test reasoning"
        )
        return judge
    
    async def test_batch_processor_init(self, mock_judge):
        """Test BatchProcessor initialization."""
        processor = BatchProcessor(mock_judge, max_concurrent=10)
        assert processor.judge == mock_judge
        assert processor.max_concurrent == 10
    
    async def test_batch_process_success(self, mock_judge):
        """Test successful batch processing."""
        processor = BatchProcessor(mock_judge, max_concurrent=2)
        
        data = [
            {"content": "Text 1", "criteria": "quality"},
            {"content": "Text 2", "criteria": "accuracy"},
            {"content": "Text 3", "criteria": "clarity"}
        ]
        
        result = await processor.process(data)
        
        assert isinstance(result, BatchResult)
        assert result.total == 3
        assert result.successful == 3
        assert result.failed == 0
        assert len(result.results) == 3
        
        # Check that all results are EvaluationResult instances
        for res in result.results:
            assert isinstance(res, EvaluationResult)
    
    async def test_batch_conversation_evaluation(self, mock_judge):
        """Test batch processing of conversations."""
        processor = BatchProcessor(mock_judge, max_concurrent=2)

        conversations = [
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            [
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well, thanks!"}
            ]
        ]
        
        data = [
            {"content": conv, "criteria": "conversation quality"}
            for conv in conversations
        ]
        
        result = await processor.process(data)
        assert isinstance(result, BatchResult)
        assert result.total == 2
        assert result.successful == 2
        assert result.failed == 0
        assert len(result.results) == 2
        
        # Check that all results are EvaluationResult instances
        for res in result.results:
            assert isinstance(res, EvaluationResult)
    
    async def test_batch_process_with_failures(self, mock_judge):
        """Test batch processing with some failures."""
        processor = BatchProcessor(mock_judge, max_concurrent=2)
        
        # Make the second call fail
        mock_judge.evaluate.side_effect = [
            EvaluationResult(decision="GOOD", reasoning="Success"),
            Exception("Evaluation failed"),
            EvaluationResult(decision="OK", reasoning="Success")
        ]
        
        data = [
            {"content": "Text 1", "criteria": "quality"},
            {"content": "Text 2", "criteria": "accuracy"},  # This will fail
            {"content": "Text 3", "criteria": "clarity"}
        ]
        
        result = await processor.process(data)
        
        assert result.total == 3
        assert result.successful == 2
        assert result.failed == 1
        assert result.success_rate == 2/3
        
        # Check failures
        failures = result.get_failures()
        assert len(failures) == 1
        assert failures[0][0] == 1  # Second item (index 1) failed
        assert isinstance(failures[0][1], Exception)
    
    async def test_batch_process_with_progress_callback(self, mock_judge):
        """Test batch processing with progress callback."""
        processor = BatchProcessor(mock_judge, max_concurrent=1)
        
        progress_calls = []
        def progress_callback(completed, total):
            progress_calls.append((completed, total))
        
        data = [
            {"content": "Text 1", "criteria": "quality"},
            {"content": "Text 2", "criteria": "accuracy"}
        ]
        
        await processor.process(data, progress_callback=progress_callback)
        
        # Should have recorded progress
        assert len(progress_calls) >= 2
        assert progress_calls[-1] == (2, 2)  # Final call should be (completed, total)
    
    async def test_batch_process_default_kwargs(self, mock_judge):
        """Test batch processing with default kwargs."""
        processor = BatchProcessor(mock_judge, max_concurrent=1)
        
        data = [
            {"content": "Text 1"},  # No criteria specified
            {"content": "Text 2", "criteria": "specific_criteria"}  # Override
        ]
        
        default_kwargs = {"criteria": "default_criteria"}
        
        await processor.process(data, **default_kwargs)
        
        # Check that evaluate was called with merged kwargs
        calls = mock_judge.evaluate.call_args_list
        assert len(calls) == 2
        
        # First call should use default criteria
        assert calls[0][1]["criteria"] == "default_criteria"
        # Second call should use specific criteria (override)
        assert calls[1][1]["criteria"] == "specific_criteria"
    
    async def test_batch_process_concurrency_limit(self, mock_judge):
        """Test that concurrency limit is respected."""
        max_concurrent = 2
        processor = BatchProcessor(mock_judge, max_concurrent=max_concurrent)
        
        call_times = []
        
        async def mock_evaluate(**kwargs):
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.1)  # Simulate some processing time
            return EvaluationResult(decision="GOOD", reasoning="Test")
        
        mock_judge.evaluate.side_effect = mock_evaluate
        
        data = [{"content": f"Text {i}", "criteria": "test"} for i in range(5)]
        
        start_time = asyncio.get_event_loop().time()
        await processor.process(data)
        end_time = asyncio.get_event_loop().time()
        
        # With max_concurrent=2 and 0.1s per call, 5 calls should take at least 0.3s
        # (first 2 in parallel, then next 2 in parallel, then last 1)
        assert end_time - start_time >= 0.25  # Allow some margin for timing