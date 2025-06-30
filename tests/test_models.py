import pytest
from pydantic import ValidationError
from vllm_judge.models import (
    JudgeConfig, 
    EvaluationResult, 
    Metric, 
    BatchResult, 
    TemplateEngine,
    ModelSpecificMetric
)


class TestJudgeConfig:
    """Test JudgeConfig model."""
    
    def test_judge_config_creation(self):
        """Test basic JudgeConfig creation."""
        config = JudgeConfig(
            base_url="http://localhost:8000",
            model="llama-2-7b"
        )
        assert config.base_url == "http://localhost:8000"
        assert config.model == "llama-2-7b"
        assert config.api_key == "dummy"
        assert config.timeout == 30.0
    
    def test_judge_config_url_validation(self):
        """Test URL validation."""
        # Valid URLs
        config = JudgeConfig(base_url="http://localhost:8000", model="test")
        assert config.base_url == "http://localhost:8000"
        
        config = JudgeConfig(base_url="https://api.example.com/", model="test") 
        assert config.base_url == "https://api.example.com"
        
        # Invalid URLs
        with pytest.raises(ValidationError):
            JudgeConfig(base_url="invalid-url", model="test")
    
    def test_judge_config_from_url(self, monkeypatch):
        """Test from_url class method."""
        # Mock detect_model_sync
        monkeypatch.setattr(
            "vllm_judge.client.detect_model_sync", 
            lambda url: "auto-detected-model"
        )
        
        config = JudgeConfig.from_url("http://localhost:8000")
        assert config.base_url == "http://localhost:8000"
        assert config.model == "auto-detected-model"
        
        # With explicit model
        config = JudgeConfig.from_url("http://localhost:8000", model="explicit-model")
        assert config.model == "explicit-model"
    
    def test_default_sampling_params(self):
        """Test default sampling parameters are set correctly."""
        config = JudgeConfig(
            base_url="http://localhost:8000",
            model="test-model"
        )
        
        assert "sampling_params" in config.model_dump()
        assert config.sampling_params["temperature"] == 0.0
        assert config.sampling_params["max_tokens"] == 256
    
    def test_custom_sampling_params(self):
        """Test custom sampling parameters."""
        custom_params = {
            "temperature": 0.7,
            "max_tokens": 512,
            "top_p": 0.9,
            "top_k": 50,
            "frequency_penalty": 0.1
        }
        
        config = JudgeConfig(
            base_url="http://localhost:8000",
            model="test-model",
            sampling_params=custom_params
        )
        
        assert config.sampling_params == custom_params
        assert config.sampling_params["temperature"] == 0.7
        assert config.sampling_params["top_p"] == 0.9
    
    def test_partial_sampling_params_override(self):
        """Test partial override of default sampling parameters."""
        config = JudgeConfig(
            base_url="http://localhost:8000",
            model="test-model",
            sampling_params={"temperature": 0.8}  # Only override temperature
        )
        
        # Should only have what we explicitly set
        assert config.sampling_params["temperature"] == 0.8
        assert len(config.sampling_params) == 1  # Only temperature
    
    def test_from_url_with_sampling_params(self, monkeypatch):
        """Test from_url with custom sampling parameters."""
        monkeypatch.setattr(
            "vllm_judge.client.detect_model_sync", 
            lambda url: "detected-model"
        )
        
        custom_params = {
            "temperature": 1.0,
            "max_tokens": 1024,
            "repetition_penalty": 1.1
        }
        
        config = JudgeConfig.from_url(
            "http://localhost:8000",
            sampling_params=custom_params
        )
        
        assert config.sampling_params == custom_params
        assert config.model == "detected-model"
    
    def test_empty_sampling_params(self):
        """Test empty sampling parameters."""
        config = JudgeConfig(
            base_url="http://localhost:8000",
            model="test-model",
            sampling_params={}
        )
        
        assert config.sampling_params == {}
    
    def test_sampling_params_validation(self):
        """Test sampling parameters type validation."""
        # Should accept dict
        config = JudgeConfig(
            base_url="http://localhost:8000",
            model="test-model",
            sampling_params={"temperature": 0.5}
        )
        assert isinstance(config.sampling_params, dict)
        
        # Invalid types should be caught by Pydantic
        with pytest.raises(ValueError):
            JudgeConfig(
                base_url="http://localhost:8000",
                model="test-model",
                sampling_params="invalid"  # Should be dict
            )


class TestEvaluationResult:
    """Test EvaluationResult model."""
    
    def test_evaluation_result_creation(self):
        """Test basic EvaluationResult creation."""
        result = EvaluationResult(
            decision="GOOD",
            reasoning="The response is clear and helpful.",
            score=8.5,
            metadata={"model": "llama-2-7b"}
        )
        
        assert result.decision == "GOOD"
        assert result.reasoning == "The response is clear and helpful."
        assert result.score == 8.5
        assert result.metadata == {"model": "llama-2-7b"}
    
    def test_evaluation_result_required_fields(self):
        """Test that decision and reasoning are required."""
        # Should work with minimal fields
        result = EvaluationResult(
            decision="PASS",
            reasoning="Good response"
        )
        assert result.decision == "PASS"
        assert result.reasoning == "Good response"
        assert result.score is None
        assert result.metadata == {}
        
        # Should fail without decision
        with pytest.raises(ValidationError):
            EvaluationResult(reasoning="Good response")
        
        # Should fail without reasoning
        with pytest.raises(ValidationError):
            EvaluationResult(decision="PASS")
    
    def test_evaluation_result_types(self):
        """Test different types for decision field."""
        # String decision
        result = EvaluationResult(decision="GOOD", reasoning="Test")
        assert isinstance(result.decision, str)
        
        # Boolean decision
        result = EvaluationResult(decision=True, reasoning="Test")
        assert isinstance(result.decision, bool)
        
        # Integer decision
        result = EvaluationResult(decision=5, reasoning="Test")
        assert isinstance(result.decision, int)
        
        # Float decision
        result = EvaluationResult(decision=7.5, reasoning="Test")
        assert isinstance(result.decision, float)


class TestMetric:
    """Test Metric class."""
    
    def test_metric_creation(self):
        """Test basic Metric creation."""
        metric = Metric(
            name="test_metric",
            criteria="Test for accuracy",
            rubric="Score from 1-10",
            scale=(1, 10)
        )
        
        assert metric.name == "test_metric"
        assert metric.criteria == "Test for accuracy"
        assert metric.rubric == "Score from 1-10"
        assert metric.scale == (1, 10)
        assert metric.template_engine == TemplateEngine.FORMAT
    
    def test_metric_with_templates(self):
        """Test Metric with template variables."""
        metric = Metric(
            name="template_metric",
            criteria="Evaluate {content_type} for {audience}",
            template_vars={"content_type": "essay", "audience": "students"},
            required_vars=["specific_topic"]
        )
        
        assert metric.criteria == "Evaluate {content_type} for {audience}"
        assert metric.template_vars == {"content_type": "essay", "audience": "students"}
        assert "specific_topic" in metric.required_vars
    
    def test_metric_auto_detect_vars(self):
        """Test auto-detection of required variables."""
        metric = Metric(
            name="auto_detect",
            criteria="Evaluate {topic} for {audience}",
            rubric="Focus on {quality_aspect}",
            template_vars={"audience": "general"}  # topic and quality_aspect should be auto-detected
        )
        
        # Should auto-detect topic and quality_aspect as required
        assert "topic" in metric.required_vars
        assert "quality_aspect" in metric.required_vars
        assert "audience" not in metric.required_vars  # Already has default value


class TestModelSpecificMetric:
    """Test ModelSpecificMetric class."""
    
    def test_model_specific_metric_creation(self):
        """Test ModelSpecificMetric creation."""
        def dummy_parser(response: str):
            return EvaluationResult(decision="safe", reasoning="Test")
        
        metric = ModelSpecificMetric(
            name="llama_guard",
            model_pattern="llama-guard",
            parser_func=dummy_parser
        )
        
        assert metric.name == "llama_guard"
        assert metric.model_pattern == "llama-guard"
        assert metric.parser_func == dummy_parser
        assert metric.criteria == "model-specific evaluation"


class TestBatchResult:
    """Test BatchResult model."""
    
    def test_batch_result_creation(self):
        """Test BatchResult creation."""
        results = [
            EvaluationResult(decision="GOOD", reasoning="Good response"),
            EvaluationResult(decision="BAD", reasoning="Poor response"),
            Exception("Failed evaluation")
        ]
        
        batch_result = BatchResult(
            results=results,
            total=3,
            successful=2,
            failed=1,
            duration_seconds=5.0
        )
        
        assert batch_result.total == 3
        assert batch_result.successful == 2
        assert batch_result.failed == 1
        assert batch_result.success_rate == 2/3
    
    def test_batch_result_get_failures(self):
        """Test getting failures from batch result."""
        error = Exception("Test error")
        results = [
            EvaluationResult(decision="GOOD", reasoning="Good"),
            error,
            EvaluationResult(decision="OK", reasoning="OK")
        ]
        
        batch_result = BatchResult(
            results=results,
            total=3,
            successful=2,
            failed=1,
            duration_seconds=1.0
        )
        
        failures = batch_result.get_failures()
        assert len(failures) == 1
        assert failures[0] == (1, error)