import pytest
import json
import numpy as np
from unittest.mock import patch
from vllm_judge.parsers import (
    parse_llama_guard_3,
    parse_granite_guardian_3_2,
    get_probabilities
)
from vllm_judge.models import EvaluationResult


class TestLlamaGuard3Parser:
    """Test Llama Guard 3 parser functionality."""
    
    def test_parse_safe_response_single_line(self):
        """Test parsing a safe response with single line."""
        response = "safe"
        
        result = parse_llama_guard_3(response)
        
        assert isinstance(result, EvaluationResult)
        assert result.decision == "safe"
        assert result.reasoning == "No violations detected"
        assert result.score is None
        assert result.metadata["model_type"] == "llama_guard_3"
    
    def test_parse_unsafe_response_with_violations(self):
        """Test parsing unsafe response with violation details."""
        response = """unsafe
S1: Violent Crimes
S2: Non-Violent Crimes"""
        
        result = parse_llama_guard_3(response)
        
        assert result.decision == "unsafe"
        assert result.reasoning == "S1: Violent Crimes"
        assert result.score is None
        assert result.metadata["model_type"] == "llama_guard_3"
    
    def test_parse_unsafe_response_multiline_reasoning(self):
        """Test parsing unsafe response with detailed reasoning."""
        response = """unsafe
The content contains potentially harmful information that could be used to cause harm. Specifically, it includes instructions for dangerous activities."""
        
        result = parse_llama_guard_3(response)
        
        assert result.decision == "unsafe"
        assert "potentially harmful information" in result.reasoning
        assert result.score is None
    
    def test_parse_safe_response_with_explanation(self):
        """Test parsing safe response with explanation."""
        response = """safe
The content is appropriate and does not violate any safety guidelines."""
        
        result = parse_llama_guard_3(response)
        
        assert result.decision == "safe"
        assert result.reasoning == "The content is appropriate and does not violate any safety guidelines."
    
    def test_parse_response_with_whitespace(self):
        """Test parsing response with various whitespace."""
        response = "  safe  \n  No issues found  \n"
        
        result = parse_llama_guard_3(response)
        
        assert result.decision == "safe"
        assert result.reasoning == "No issues found"
    
    def test_parse_empty_response(self):
        """Test parsing empty response."""
        response = ""
        
        result = parse_llama_guard_3(response)
        
        assert result.decision == ""
        assert result.reasoning == "No violations detected"
    
    def test_parse_response_case_preservation(self):
        """Test that case is preserved in decision."""
        response = "UNSAFE\nVIOLATION DETECTED"
        
        result = parse_llama_guard_3(response)
        
        assert result.decision == "unsafe"  # Should be lowercased
        assert result.reasoning == "VIOLATION DETECTED"
    
    def test_parse_response_with_multiple_lines(self):
        """Test parsing response with multiple reasoning lines."""
        response = """unsafe
S1: Violent Crimes
S2: Non-Violent Crimes
Additional context about the violation"""
        
        result = parse_llama_guard_3(response)
        
        assert result.decision == "unsafe"
        # Should only take the first reasoning line
        assert result.reasoning == "S1: Violent Crimes"
    
    def test_parse_response_unicode_content(self):
        """Test parsing response with Unicode characters."""
        response = """safe
内容安全，没有违规行为"""
        
        result = parse_llama_guard_3(response)
        
        assert result.decision == "safe"
        assert "内容安全" in result.reasoning
    
    def test_parse_response_special_characters(self):
        """Test parsing response with special characters."""
        response = """unsafe
Content contains: <script>alert('xss')</script> and other harmful elements"""
        
        result = parse_llama_guard_3(response)
        
        assert result.decision == "unsafe"
        assert "<script>" in result.reasoning


class TestGraniteGuardian32Parser:
    """Test Granite Guardian 3.2 parser functionality."""
    
    def test_parse_safe_response_with_logprobs(self):
        """Test parsing safe response with logprobs."""
        choices = [
            {
                "message": {
                    "content": "Yes\n<confidence> high </confidence>"
                },
                "logprobs": {
                    "content": [
                        {
                            "top_logprobs": [
                                {"token": "Yes", "logprob": -0.1},
                                {"token": "No", "logprob": -2.3}
                            ]
                        }
                    ]
                }
            }
        ]
        
        result = parse_granite_guardian_3_2(choices)
        
        assert isinstance(result, EvaluationResult)
        assert result.decision == "Yes"
        assert result.reasoning == "Confidence level: high"
        assert result.score is not None
        assert result.score > 0.5  # Should be high probability for "Yes"
        assert result.metadata["model_type"] == "granite_guardian_3_2"
    
    def test_parse_risky_response_with_logprobs(self):
        """Test parsing risky response with logprobs."""
        choices = [
            {
                "message": {
                    "content": "No\n<confidence> medium </confidence>"
                },
                "logprobs": {
                    "content": [
                        {
                            "top_logprobs": [
                                {"token": "No", "logprob": -0.2},
                                {"token": "Yes", "logprob": -1.8}
                            ]
                        }
                    ]
                }
            }
        ]
        
        result = parse_granite_guardian_3_2(choices)
        
        assert result.decision == "No"
        assert result.reasoning == "Confidence level: medium"
        assert result.score is not None
        assert result.score > 0.5  # Should be high probability for "No"
    
    def test_parse_response_without_logprobs(self):
        """Test parsing response without logprobs."""
        choices = [
            {
                "message": {
                    "content": "Yes\n<confidence> low </confidence>"
                },
                "logprobs": None
            }
        ]
        
        result = parse_granite_guardian_3_2(choices)
        
        assert result.decision == "Yes"
        assert result.reasoning == "Confidence level: low"
        assert result.score is None
    
    def test_parse_response_without_confidence(self):
        """Test parsing response without confidence tag."""
        choices = [
            {
                "message": {
                    "content": "No"
                },
                "logprobs": None
            }
        ]
        
        result = parse_granite_guardian_3_2(choices)
        
        assert result.decision == "No"
        assert result.reasoning == "Confidence level: Unknown"
        assert result.score is None
    
    def test_parse_string_input_valid_json(self):
        """Test parsing when input is a JSON string."""
        choices_json = json.dumps([
            {
                "message": {
                    "content": "Yes\n<confidence> high </confidence>"
                },
                "logprobs": None
            }
        ])
        
        result = parse_granite_guardian_3_2(choices_json)
        
        assert result.decision == "Yes"
        assert result.reasoning == "Confidence level: high"
    
    def test_parse_string_input_invalid_json(self):
        """Test parsing when input is invalid JSON string."""
        invalid_json = "not valid json"
        
        result = parse_granite_guardian_3_2(invalid_json)
        
        assert result.decision == "Failed"
        assert "JSON parsing error" in result.reasoning
        assert result.score is None
        assert result.metadata["model_type"] == "granite_guardian_3_2"
    
    def test_parse_empty_choices(self):
        """Test parsing with empty choices list."""
        choices = []
        
        result = parse_granite_guardian_3_2(choices)
        
        assert result.decision == "Failed"
        assert result.reasoning == "Empty choices list"
        assert result.score is None
    
    def test_parse_ambiguous_response(self):
        """Test parsing response that doesn't match Yes/No pattern."""
        choices = [
            {
                "message": {
                    "content": "Maybe\n<confidence> uncertain </confidence>"
                },
                "logprobs": None
            }
        ]
        
        result = parse_granite_guardian_3_2(choices)
        
        assert result.decision == "Maybe"
        assert result.reasoning == "Confidence level: uncertain"
    
    def test_parse_response_case_insensitive(self):
        """Test parsing with different case variations."""
        choices = [
            {
                "message": {
                    "content": "YES\n<confidence> high </confidence>"
                },
                "logprobs": {
                    "content": [
                        {
                            "top_logprobs": [
                                {"token": "yes", "logprob": -0.1},  # lowercase in logprobs
                                {"token": "no", "logprob": -2.0}
                            ]
                        }
                    ]
                }
            }
        ]
        
        result = parse_granite_guardian_3_2(choices)
        
        assert result.decision == "Yes"  # Should normalize to "Yes"
        assert result.score is not None
    
    def test_parse_complex_confidence_pattern(self):
        """Test parsing complex confidence patterns."""
        choices = [
            {
                "message": {
                    "content": "No\nSome explanation here\n<confidence> very high </confidence>\nMore text"
                },
                "logprobs": None
            }
        ]
        
        result = parse_granite_guardian_3_2(choices)
        
        assert result.decision == "No"
        assert result.reasoning == "Confidence level: very high"
    
    def test_parse_response_with_multiline_content(self):
        """Test parsing response with multiline content."""
        choices = [
            {
                "message": {
                    "content": """Yes
This is a detailed explanation
of why the content is safe.
<confidence> high </confidence>"""
                },
                "logprobs": None
            }
        ]
        
        result = parse_granite_guardian_3_2(choices)
        
        assert result.decision == "Yes"
        assert result.reasoning == "Confidence level: high"
    
    def test_parse_response_unicode_content(self):
        """Test parsing response with Unicode characters."""
        choices = [
            {
                "message": {
                    "content": "Yes\n<confidence> 高 </confidence>"
                },
                "logprobs": None
            }
        ]
        
        result = parse_granite_guardian_3_2(choices)
        
        assert result.decision == "Yes"
        assert result.reasoning == "Confidence level: 高"


class TestGetProbabilities:
    """Test the get_probabilities helper function."""
    
    def test_get_probabilities_basic(self):
        """Test basic probability calculation."""
        logprobs = {
            "content": [
                {
                    "top_logprobs": [
                        {"token": "Yes", "logprob": -0.1},
                        {"token": "No", "logprob": -2.0}
                    ]
                }
            ]
        }
        
        probs = get_probabilities(logprobs)
        
        assert isinstance(probs, np.ndarray)
        assert len(probs) == 2
        assert probs[0] > probs[1]  # "Yes" should have higher probability
        assert np.isclose(np.sum(probs), 1.0)  # Should sum to 1
    
    def test_get_probabilities_case_insensitive(self):
        """Test that probability calculation is case insensitive."""
        logprobs = {
            "content": [
                {
                    "top_logprobs": [
                        {"token": "yes", "logprob": -0.1},  # lowercase
                        {"token": "NO", "logprob": -1.5}   # uppercase
                    ]
                }
            ]
        }
        
        probs = get_probabilities(logprobs)
        
        assert probs[0] > probs[1]  # "yes" should map to safe token
        assert np.isclose(np.sum(probs), 1.0)
    
    def test_get_probabilities_with_whitespace(self):
        """Test probability calculation with whitespace in tokens."""
        logprobs = {
            "content": [
                {
                    "top_logprobs": [
                        {"token": " Yes ", "logprob": -0.2},
                        {"token": " No ", "logprob": -1.8}
                    ]
                }
            ]
        }
        
        probs = get_probabilities(logprobs)
        
        assert probs[0] > probs[1]
        assert np.isclose(np.sum(probs), 1.0)
    
    def test_get_probabilities_multiple_tokens(self):
        """Test probability calculation with multiple token instances."""
        logprobs = {
            "content": [
                {
                    "top_logprobs": [
                        {"token": "Yes", "logprob": -0.1},
                        {"token": "No", "logprob": -2.0}
                    ]
                },
                {
                    "top_logprobs": [
                        {"token": "Yes", "logprob": -0.3},  # Another instance
                        {"token": "Other", "logprob": -1.5}
                    ]
                }
            ]
        }
        
        probs = get_probabilities(logprobs)
        
        # Should accumulate probabilities from multiple instances
        assert np.isclose(np.sum(probs), 1.0)
    
    def test_get_probabilities_no_matching_tokens(self):
        """Test probability calculation when no matching tokens found."""
        logprobs = {
            "content": [
                {
                    "top_logprobs": [
                        {"token": "Other", "logprob": -0.1},
                        {"token": "Random", "logprob": -0.2}
                    ]
                }
            ]
        }
        
        probs = get_probabilities(logprobs)
        
        # Should return default small probabilities
        assert np.isclose(np.sum(probs), 1.0)
        assert np.isclose(probs[0], probs[1])  # Should be roughly equal (both very small)
    
    def test_get_probabilities_numerical_stability(self):
        """Test numerical stability with very small probabilities."""
        logprobs = {
            "content": [
                {
                    "top_logprobs": [
                        {"token": "Yes", "logprob": -50.0},  # Very small probability
                        {"token": "No", "logprob": -100.0}   # Even smaller
                    ]
                }
            ]
        }
        
        probs = get_probabilities(logprobs)
        
        assert not np.any(np.isnan(probs))  # Should not be NaN
        assert not np.any(np.isinf(probs))  # Should not be infinite
        assert np.isclose(np.sum(probs), 1.0)
        assert probs[0] > probs[1]  # Relative ordering should be preserved


class TestParserIntegration:
    """Test integration scenarios with parsers."""
    
    def test_llama_guard_realistic_responses(self):
        """Test Llama Guard parser with realistic model responses."""
        test_cases = [
            ("safe", "safe", "No violations detected"),
            ("unsafe\nS1: Violent Crimes", "unsafe", "S1: Violent Crimes"),
            ("safe\nContent is appropriate for all audiences", "safe", "Content is appropriate for all audiences"),
            ("unsafe\nS2: Non-Violent Crimes\nS3: Sex Crimes", "unsafe", "S2: Non-Violent Crimes"),
        ]
        
        for response, expected_decision, expected_reasoning in test_cases:
            result = parse_llama_guard_3(response)
            assert result.decision == expected_decision
            assert result.reasoning == expected_reasoning
    
    def test_granite_guardian_realistic_responses(self):
        """Test Granite Guardian parser with realistic model responses."""
        # Safe response
        safe_choices = [
            {
                "message": {
                    "content": "Yes\n\nThe user is asking for help with a legitimate programming task. This is a safe request that doesn't involve any harmful activities.\n\n<confidence> high </confidence>"
                },
                "logprobs": {
                    "content": [
                        {
                            "top_logprobs": [
                                {"token": "Yes", "logprob": -0.05},
                                {"token": "No", "logprob": -3.2}
                            ]
                        }
                    ]
                }
            }
        ]
        
        result = parse_granite_guardian_3_2(safe_choices)
        assert result.decision == "Yes"
        assert result.reasoning == "Confidence level: high"
        assert result.score > 0.9  # Should be very confident
        
        # Risky response
        risky_choices = [
            {
                "message": {
                    "content": "No\n\nThis request asks for information that could be used to cause harm. The content involves dangerous activities that should not be shared.\n\n<confidence> very high </confidence>"
                },
                "logprobs": {
                    "content": [
                        {
                            "top_logprobs": [
                                {"token": "No", "logprob": -0.02},
                                {"token": "Yes", "logprob": -4.1}
                            ]
                        }
                    ]
                }
            }
        ]
        
        result = parse_granite_guardian_3_2(risky_choices)
        assert result.decision == "No"
        assert result.reasoning == "Confidence level: very high"
        assert result.score > 0.95  # Should be very confident
    
    def test_parser_error_handling_robustness(self):
        """Test that parsers handle edge cases gracefully."""
        # Test with None input
        with pytest.raises(AttributeError):
            parse_llama_guard_3(None)
        
        # Test with malformed data
        malformed_choices = [
            {
                "message": {},  # Missing content
                "logprobs": None
            }
        ]
        
        # Should handle gracefully by extracting what it can
        try:
            result = parse_granite_guardian_3_2(malformed_choices)
            # Should not crash, but may return Failed or unusual results
            assert isinstance(result, EvaluationResult)
        except (KeyError, AttributeError):
            # These exceptions are acceptable for malformed input
            pass
    
    def test_parser_metadata_consistency(self):
        """Test that parser metadata is consistent across calls."""
        # Test Llama Guard
        result1 = parse_llama_guard_3("safe")
        result2 = parse_llama_guard_3("unsafe\nS1: Violent Crimes")
        
        assert result1.metadata["model_type"] == result2.metadata["model_type"]
        assert result1.metadata["model_type"] == "llama_guard_3"
        
        # Test Granite Guardian
        choices = [{"message": {"content": "Yes"}, "logprobs": None}]
        result3 = parse_granite_guardian_3_2(choices)
        result4 = parse_granite_guardian_3_2([])  # Empty choices
        
        assert result3.metadata["model_type"] == result4.metadata["model_type"]
        assert result3.metadata["model_type"] == "granite_guardian_3_2"
