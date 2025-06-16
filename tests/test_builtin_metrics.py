from vllm_judge.builtin_metrics import (
    HELPFULNESS, ACCURACY, SAFETY, CODE_QUALITY,
    BUILTIN_METRICS, LLAMA_GUARD_3_SAFETY
)
from vllm_judge.models import Metric, ModelSpecificMetric


class TestBuiltinMetrics:
    """Test built-in metrics."""
    
    def test_builtin_metrics_are_metrics(self):
        """Test that built-in metrics are Metric instances."""
        assert isinstance(HELPFULNESS, Metric)
        assert isinstance(ACCURACY, Metric)
        assert isinstance(SAFETY, Metric)
        assert isinstance(CODE_QUALITY, Metric)
    
    def test_builtin_metrics_have_names(self):
        """Test that built-in metrics have proper names."""
        assert HELPFULNESS.name.upper() == "HELPFULNESS"
        assert ACCURACY.name.upper() == "ACCURACY" 
        assert SAFETY.name.upper() == "SAFETY"
        assert CODE_QUALITY.name.upper() == "CODE_QUALITY"
    
    def test_builtin_metrics_have_criteria(self):
        """Test that built-in metrics have criteria defined."""
        assert HELPFULNESS.criteria is not None
        assert len(HELPFULNESS.criteria) > 0
        
        assert ACCURACY.criteria is not None
        assert len(ACCURACY.criteria) > 0
    
    def test_builtin_metrics_dict(self):
        """Test BUILTIN_METRICS dictionary."""
        assert isinstance(BUILTIN_METRICS, dict)
        assert "HELPFULNESS".lower() in BUILTIN_METRICS
        assert "ACCURACY".lower() in BUILTIN_METRICS
        assert BUILTIN_METRICS["HELPFULNESS".lower()] == HELPFULNESS
    
    def test_model_specific_metrics(self):
        """Test model-specific metrics like Llama Guard."""
        assert isinstance(LLAMA_GUARD_3_SAFETY, ModelSpecificMetric)
        assert LLAMA_GUARD_3_SAFETY.name.upper() == "LLAMA_GUARD_3_SAFETY"
        assert LLAMA_GUARD_3_SAFETY.model_pattern is not None
        assert LLAMA_GUARD_3_SAFETY.parser_func is not None
    
    def test_metrics_with_scales(self):
        """Test metrics that have defined scales."""
        # Some metrics should have scales defined
        scale_metrics = [m for m in BUILTIN_METRICS.values() if m.scale is not None]
        assert len(scale_metrics) > 0
        
        # Check scale format
        for metric in scale_metrics:
            assert isinstance(metric.scale, tuple)
            assert len(metric.scale) == 2
            assert metric.scale[0] < metric.scale[1]
    
    def test_template_metrics(self):
        """Test metrics that use templates."""
        from vllm_judge.builtin_metrics import (
            EDUCATIONAL_CONTENT_TEMPLATE,
            CODE_REVIEW_TEMPLATE
        )
        
        assert isinstance(EDUCATIONAL_CONTENT_TEMPLATE, Metric)
        assert isinstance(CODE_REVIEW_TEMPLATE, Metric)
        
        # These should have template variables
        assert len(EDUCATIONAL_CONTENT_TEMPLATE.required_vars) > 0
        assert len(CODE_REVIEW_TEMPLATE.required_vars) > 0