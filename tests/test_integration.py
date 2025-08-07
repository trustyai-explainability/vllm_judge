import pytest
from unittest.mock import AsyncMock, Mock, patch
from vllm_judge import Judge, JudgeConfig, Metric, EvaluationResult



class TestIntegration:
    """Integration tests for the complete workflow."""
    
    async def test_end_to_end_evaluation(self):
        """Test complete evaluation workflow."""
        # Mock the HTTP client
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": '{"decision": "EXCELLENT", "reasoning": "The response is comprehensive and accurate.", "score": 9.2}'
                    }
                }]
            }
            mock_response.raise_for_status.return_value = None
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            # Create judge
            config = JudgeConfig(
                base_url="http://localhost:8000",
                model="test-model"
            )
            judge = Judge(config)
            
            # Perform evaluation
            result = await judge.evaluate(
                content="Python is a versatile programming language used for web development, data science, and automation.",
                criteria="accuracy and completeness of the technical description"
            )
            
            assert isinstance(result, EvaluationResult)
            assert result.decision == "EXCELLENT"
            assert result.reasoning == "The response is comprehensive and accurate."
            assert result.score == 9.2
    
    async def test_conversation_evaluation_flow(self):
        """Test complete conversation evaluation flow."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": '{"decision": "EDUCATIONAL", "reasoning": "Conversation shows good educational progression", "score": 9.0}'
                    }
                }]
            }
            mock_response.raise_for_status.return_value = None
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            config = JudgeConfig(
                base_url="http://localhost:8000",
                model="test-model"
            )
            
            conversation = [
                {"role": "user", "content": "Help me with my homework"},
                {"role": "assistant", "content": "I'd be happy to help! What subject?"},
                {"role": "user", "content": "Math - I need help with fractions"},
                {"role": "assistant", "content": "Great! Let's start with the basics..."}
            ]
            
            judge = Judge(config)
            
            result = await judge.evaluate(
                content=conversation,
                criteria="educational value and appropriateness"
            )
            
            assert result.decision == "EDUCATIONAL"
            assert result.reasoning == "Conversation shows good educational progression"
            assert result.score == 9.0
    
    async def test_workflow_with_custom_metric(self):
        """Test workflow with custom registered metric."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": '{"decision": "GOOD", "reasoning": "Meets custom criteria.", "score": 8.0}'
                    }
                }]
            }
            mock_response.raise_for_status.return_value = None
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            config = JudgeConfig(base_url="http://localhost:8000", model="test-model")
            judge = Judge(config)
            
            # Register custom metric
            custom_metric = Metric(
                name="custom_technical_accuracy",
                criteria="Evaluate technical accuracy for software documentation",
                rubric="Rate from 1-10 based on factual correctness",
                scale=(1, 10)
            )
            judge.register_metric(custom_metric)
            
            # Use custom metric
            result = await judge.evaluate(
                content="Python uses garbage collection for memory management.",
                metric="custom_technical_accuracy"
            )
            
            assert isinstance(result, EvaluationResult)
            assert result.decision == "GOOD"
    
    async def test_workflow_with_templates(self):
        """Test workflow with template variables."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": '{"decision": "APPROPRIATE", "reasoning": "Content is suitable for the target audience."}'
                    }
                }]
            }
            mock_response.raise_for_status.return_value = None
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            config = JudgeConfig(base_url="http://localhost:8000", model="test-model")
            judge = Judge(config)
            
            result = await judge.evaluate(
                content="Python is easy to learn and has simple syntax.",
                criteria="Evaluate this {content_type} for {audience} level understanding",
                template_vars={
                    "content_type": "explanation",
                    "audience": "beginner"
                }
            )
            
            assert isinstance(result, EvaluationResult)
            assert "template_vars" in result.metadata
            assert result.metadata["template_vars"]["content_type"] == "explanation"
    
    async def test_comparison_workflow(self):
        """Test comparison evaluation workflow."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": '{"decision": "response_a", "reasoning": "Response A is more comprehensive and accurate."}'
                    }
                }]
            }
            mock_response.raise_for_status.return_value = None
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            config = JudgeConfig(base_url="http://localhost:8000", model="test-model")
            judge = Judge(config)
            
            result = await judge.compare(
                response_a="Python is a high-level, interpreted programming language with dynamic semantics.",
                response_b="Python is a programming language.",
                criteria="technical completeness and accuracy"
            )
            
            assert isinstance(result, EvaluationResult)
            assert result.decision == "response_a"
    
    async def test_batch_evaluation_workflow(self):
        """Test batch evaluation workflow."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": '{"decision": "GOOD", "reasoning": "Satisfactory response."}'
                    }
                }]
            }
            mock_response.raise_for_status.return_value = None
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            config = JudgeConfig(base_url="http://localhost:8000", model="test-model")
            judge = Judge(config)
            
            data = [
                {"content": "Python is versatile.", "criteria": "accuracy"},
                {"content": "Java is fast.", "criteria": "accuracy"},
                {"content": "JavaScript runs in browsers.", "criteria": "accuracy"}
            ]
            
            # Mock BatchProcessor
            with patch('vllm_judge.judge.BatchProcessor') as mock_batch_processor:
                from vllm_judge.models import BatchResult
                mock_processor_instance = AsyncMock()
                mock_processor_instance.process.return_value = BatchResult(
                    results=[
                        EvaluationResult(decision="GOOD", reasoning="Accurate"),
                        EvaluationResult(decision="GOOD", reasoning="Accurate"),
                        EvaluationResult(decision="EXCELLENT", reasoning="Very accurate")
                    ],
                    total=3,
                    successful=3,
                    failed=0,
                    duration_seconds=2.5
                )
                mock_batch_processor.return_value = mock_processor_instance
                
                result = await judge.batch_evaluate(data)
                
                assert result.total == 3
                assert result.successful == 3
                assert result.failed == 0
    
    async def test_error_handling_workflow(self):
        """Test error handling in complete workflow."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = Exception("Network error")
            mock_client_class.return_value = mock_client
            
            config = JudgeConfig(base_url="http://localhost:8000", model="test-model")
            judge = Judge(config)
            
            # Should handle network errors gracefully
            with pytest.raises(Exception):
                await judge.evaluate(
                    content="Test content",
                    criteria="test criteria"
                )
    
    def test_metric_registration_workflow(self):
        """Test metric registration and retrieval workflow."""
        config = JudgeConfig(base_url="http://localhost:8000", model="test-model")
        judge = Judge(config)
        
        # Register multiple metrics
        metrics = [
            Metric(name="metric1", criteria="criteria1"),
            Metric(name="metric2", criteria="criteria2"),
            Metric(name="metric3", criteria="criteria3")
        ]
        
        for metric in metrics:
            judge.register_metric(metric)
        
        # Verify all metrics are registered
        all_metrics = judge.list_metrics()
        for metric in metrics:
            assert metric.name in all_metrics
            retrieved = judge.get_metric(metric.name)
            assert retrieved == metric
    
    async def test_convenience_methods_workflow(self):
        """Test all convenience methods work together."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": '{"decision": "GOOD", "reasoning": "Test response."}'
                    }
                }]
            }
            mock_response.raise_for_status.return_value = None
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            config = JudgeConfig(base_url="http://localhost:8000", model="test-model")
            judge = Judge(config)
            
            # Test score method
            score_result = await judge.score(
                criteria="clarity",
                content="Clear explanation",
                scale=(1, 10)
            )
            assert isinstance(score_result, EvaluationResult)
            
            # Test QA evaluate method
            qa_result = await judge.qa_evaluate(
                question="What is Python?",
                answer="Python is a programming language",
                criteria="accuracy"
            )
            assert isinstance(qa_result, EvaluationResult)
            
            # Test classify method
            classify_result = await judge.classify(
                content="This looks like spam",
                categories=["spam", "not_spam"],
                criteria="content classification"
            )
            assert isinstance(classify_result, EvaluationResult)