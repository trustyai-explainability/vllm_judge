import pytest
from vllm_judge.templating import TemplateProcessor
from vllm_judge.models import TemplateEngine
from vllm_judge.exceptions import InvalidInputError

def _has_jinja2() -> bool:
    """Check if Jinja2 is available."""
    try:
        import jinja2
        return True
    except ImportError:
        return False

class TestTemplateProcessor:
    """Test TemplateProcessor functionality."""
    
    def test_apply_template_string_format(self):
        """Test applying format string templates."""
        template = "Hello {name}, you are {age} years old"
        variables = {"name": "Alice", "age": 30}
        
        result = TemplateProcessor.apply_template(
            template, variables, TemplateEngine.FORMAT
        )
        
        assert result == "Hello Alice, you are 30 years old"
    
    def test_apply_template_none(self):
        """Test applying template to None."""
        result = TemplateProcessor.apply_template(
            None, {"key": "value"}, TemplateEngine.FORMAT
        )
        assert result is None
    
    def test_apply_template_no_variables(self):
        """Test applying template with no variables."""
        template = "Static text"
        result = TemplateProcessor.apply_template(
            template, {}, TemplateEngine.FORMAT
        )
        assert result == "Static text"
    
    def test_apply_template_dict_rubric_format(self):
        """Test applying template to dictionary rubric with format strings."""
        rubric = {
            1: "Poor {quality}",
            5: "Excellent {quality}"
        }
        variables = {"quality": "writing"}
        
        result = TemplateProcessor.apply_template(
            rubric, variables, TemplateEngine.FORMAT
        )
        
        assert result == {1: "Poor writing", 5: "Excellent writing"}
    
    def test_apply_template_missing_variable_strict(self):
        """Test missing variable in strict mode."""
        template = "Hello {name}, you are {age} years old"
        variables = {"name": "Alice"}  # missing 'age'
        
        with pytest.raises(InvalidInputError):
            TemplateProcessor.apply_template(
                template, variables, TemplateEngine.FORMAT, strict=True
            )
    
    def test_apply_template_missing_variable_non_strict(self):
        """Test missing variable in non-strict mode."""
        template = "Hello {name}, you are {age} years old"
        variables = {"name": "Alice"}  # missing 'age'
        
        # Should not raise error in non-strict mode
        result = TemplateProcessor.apply_template(
            template, variables, TemplateEngine.FORMAT, strict=False
        )
        # Result should contain the unsubstituted placeholder
        assert "{age}" in result
    
    @pytest.mark.skipif(
        not _has_jinja2(), 
        reason="Jinja2 not available"
    )
    def test_apply_template_jinja2(self):
        """Test applying Jinja2 templates."""
        template = "Hello {{ name }}, you are {{ age }} years old"
        variables = {"name": "Bob", "age": 25}
        
        result = TemplateProcessor.apply_template(
            template, variables, TemplateEngine.JINJA2
        )
        
        assert result == "Hello Bob, you are 25 years old"
    
    @pytest.mark.skipif(
        not _has_jinja2(),
        reason="Jinja2 not available"
    )
    def test_apply_template_jinja2_missing_variable(self):
        """Test Jinja2 template with missing variable."""
        template = "Hello {{ name }}, you are {{ age }} years old"
        variables = {"name": "Bob"}  # missing 'age'
        
        with pytest.raises(InvalidInputError):
            TemplateProcessor.apply_template(
                template, variables, TemplateEngine.JINJA2, strict=True
            )
    
    def test_apply_template_jinja2_not_available(self, monkeypatch):
        """Test Jinja2 template when Jinja2 not installed."""
        # Save original import function
        original_import = __import__
        
        # Mock Jinja2 import to fail
        def mock_import(name, *args, **kwargs):
            if name == "jinja2":
                raise ImportError("No module named 'jinja2'")
            return original_import(name, *args, **kwargs)
        
        monkeypatch.setattr("builtins.__import__", mock_import)
        
        template = "Hello {{ name }}"
        variables = {"name": "Test"}
        
        with pytest.raises(ImportError):
            TemplateProcessor.apply_template(
                template, variables, TemplateEngine.JINJA2
            )
