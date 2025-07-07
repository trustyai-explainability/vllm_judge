import pytest
import json
from unittest.mock import patch
import time
from vllm_judge import EvaluationResult
from vllm_judge.exceptions import ParseError


class TestResponseParsingHappyPath:
    """Test successful parsing scenarios."""
    
    def test_parse_direct_json_all_fields(self, mock_judge):
        """Test parsing valid JSON with all fields."""
        response = json.dumps({
            "decision": "EXCELLENT",
            "reasoning": "Clear and comprehensive explanation",
            "score": 9.5
        })
        
        result = mock_judge._parse_response(response)
        
        assert result.decision == "EXCELLENT"
        assert result.reasoning == "Clear and comprehensive explanation"
        assert result.score == 9.5
        assert result.metadata["model"] == mock_judge.config.model
        assert result.metadata["raw_response"] == response
    
    def test_parse_direct_json_minimal_fields(self, mock_judge):
        """Test parsing JSON with only required fields."""
        response = json.dumps({
            "decision": "PASS",
            "reasoning": "Meets requirements"
        })
        
        result = mock_judge._parse_response(response)
        
        assert result.decision == "PASS"
        assert result.reasoning == "Meets requirements"
        assert result.score is None
    
    def test_parse_boolean_decision(self, mock_judge):
        """Test parsing with boolean decision."""
        response = json.dumps({
            "decision": True,
            "reasoning": "Condition met",
            "score": 1
        })
        
        result = mock_judge._parse_response(response)
        
        assert result.decision is True
        assert result.reasoning == "Condition met"
        assert result.score == 1
    
    def test_parse_markdown_json_with_marker(self, mock_judge):
        """Test parsing JSON from markdown with json marker."""
        response = """
        Based on my evaluation:
        
        ```json
        {
            "decision": "GOOD",
            "reasoning": "Well-structured response",
            "score": 7.5
        }
        ```
        
        That's my assessment.
        """
        
        result = mock_judge._parse_response(response)
        
        assert result.decision == "GOOD"
        assert result.reasoning == "Well-structured response"
        assert result.score == 7.5
    
    def test_parse_markdown_json_without_marker(self, mock_judge):
        """Test parsing JSON from markdown without json marker."""
        response = """
        Here's the evaluation:
        
        ```
        {
            "decision": "FAIR",
            "reasoning": "Adequate but could improve"
        }
        ```
        """
        
        result = mock_judge._parse_response(response)
        
        assert result.decision == "FAIR"
        assert result.reasoning == "Adequate but could improve"
    
    def test_parse_regex_json_simple(self, mock_judge):
        """Test regex-based JSON extraction."""
        response = 'After analysis, I conclude {"decision": "EXCELLENT", "reasoning": "Outstanding work"} based on criteria.'
        
        result = mock_judge._parse_response(response)
        
        assert result.decision == "EXCELLENT"
        assert result.reasoning == "Outstanding work"
    
    def test_parse_with_extra_metadata(self, mock_judge):
        """Test parsing JSON with extra metadata fields."""
        response = json.dumps({
            "decision": "GOOD",
            "reasoning": "Quality response",
            "score": 8,
            "metadata": {
                "confidence": 0.95,
                "category": "technical"
            },
            "extra_field": "should be ignored in result but preserved in metadata"
        })
        
        result = mock_judge._parse_response(response)
        
        assert result.decision == "GOOD"
        assert result.reasoning == "Quality response"
        assert result.score == 8
        assert result.metadata["confidence"] == 0.95
        assert result.metadata["category"] == "technical"
        assert "extra_field" not in result.metadata



class TestResponseParsingFieldHandling:
    """Test handling of missing or alternative fields."""
    
    def test_parse_missing_reasoning_with_reason(self, mock_judge):
        """Test fallback to 'reason' when 'reasoning' missing."""
        response = json.dumps({
            "decision": "PASS",
            "reason": "Meets all criteria"
        })
        
        with patch('vllm_judge.judge.logger') as mock_logger:
            result = mock_judge._parse_response(response)
            
            assert result.decision == "PASS"
            assert result.reasoning == "Meets all criteria"
            mock_logger.debug.assert_called_with("Used 'reason' field for reasoning")
    
    def test_parse_missing_reasoning_with_explanation(self, mock_judge):
        """Test fallback to 'explanation' when 'reasoning' missing."""
        response = json.dumps({
            "decision": "FAIL",
            "explanation": "Does not meet standards"
        })
        
        with patch('vllm_judge.judge.logger') as mock_logger:
            result = mock_judge._parse_response(response)
            
            assert result.decision == "FAIL"
            assert result.reasoning == "Does not meet standards"
            mock_logger.debug.assert_called_with("Used 'explanation' field for reasoning")
    
    def test_parse_missing_reasoning_no_alternatives(self, mock_judge):
        """Test default reasoning when no alternatives found."""
        response = json.dumps({
            "decision": "UNCLEAR"
        })

        result = mock_judge._parse_response(response)
        assert result.decision == "UNCLEAR"
        assert "No reasoning provided" in result.reasoning
    
    def test_parse_null_score(self, mock_judge):
        """Test handling of explicit null score."""
        response = json.dumps({
            "decision": "GOOD",
            "reasoning": "Quality work",
            "score": None
        })
        
        result = mock_judge._parse_response(response)
        
        assert result.decision == "GOOD"
        assert result.score is None


class TestResponseParsingTypeValidation:
    """Test type validation and conversion."""
    
    def test_parse_non_string_reasoning_conversion(self, mock_judge):
        """Test conversion of non-string reasoning to string."""
        response = json.dumps({
            "decision": "GOOD",
            "reasoning": 42  # Should be string
        })
        
        with patch('vllm_judge.judge.logger') as mock_logger:
            result = mock_judge._parse_response(response)
            
            assert result.reasoning == "42"
            mock_logger.debug.assert_called_with("Converted reasoning to string")
    
    def test_parse_string_score_conversion(self, mock_judge):
        """Test conversion of string score to float."""
        response = json.dumps({
            "decision": "GOOD",
            "reasoning": "Quality work",
            "score": "8.5"  # String that should convert
        })
        
        with patch('vllm_judge.judge.logger') as mock_logger:
            result = mock_judge._parse_response(response)
            
            assert result.score == 8.5
            assert isinstance(result.score, float)
            mock_logger.debug.assert_called_with("Converted score to float")
    
    def test_parse_invalid_score_fallback(self, mock_judge):
        """Test fallback for invalid score values."""
        response = json.dumps({
            "decision": "GOOD",
            "reasoning": "Quality work",
            "score": "not-a-number"
        })
        
        with patch('vllm_judge.judge.logger') as mock_logger:
            result = mock_judge._parse_response(response)
            
            assert result.score is None
            mock_logger.warning.assert_called_with("Invalid score value: not-a-number, setting to None")
    
    def test_parse_integer_score(self, mock_judge):
        """Test that integer scores are preserved as integers."""
        response = json.dumps({
            "decision": "EXCELLENT",
            "reasoning": "Perfect work",
            "score": 10
        })
        
        result = mock_judge._parse_response(response)
        
        assert result.score == 10
        assert isinstance(result.score, float)


class TestResponseParsingErrorCases:
    """Test error handling scenarios."""
    
    def test_parse_missing_decision_field(self, mock_judge):
        """Test error when decision field is missing."""
        response = json.dumps({
            "reasoning": "Missing decision field",
            "score": 5
        })
        
        with pytest.raises(ParseError) as exc_info:
            mock_judge._parse_response(response)
        
        assert "Response missing required 'decision' field" in str(exc_info.value)
        assert exc_info.value.raw_response == response
    
    def test_parse_completely_invalid_json(self, mock_judge):
        """Test error with completely malformed JSON."""
        response = "This is not JSON at all, just plain text."
        
        with pytest.raises(ParseError) as exc_info:
            mock_judge._parse_response(response)
        
        assert "Unable to extract valid JSON from response" in str(exc_info.value)
        assert exc_info.value.raw_response == response
    
    def test_parse_empty_response(self, mock_judge):
        """Test error with empty response."""
        response = ""
        
        with pytest.raises(ParseError) as exc_info:
            mock_judge._parse_response(response)
        
        assert "Unable to extract valid JSON from response" in str(exc_info.value)
    
    def test_parse_only_whitespace(self, mock_judge):
        """Test error with whitespace-only response."""
        response = "   \n\t  "
        
        with pytest.raises(ParseError) as exc_info:
            mock_judge._parse_response(response)
        
        assert "Unable to extract valid JSON from response" in str(exc_info.value)
    
    def test_parse_json_syntax_error_in_markdown(self, mock_judge):
        """Test handling of malformed JSON in markdown."""
        response = """
        ```json
        {
            "decision": "GOOD",
            "reasoning": "Missing closing quote
        }
        ```
        """
        
        with pytest.raises(ParseError):
            mock_judge._parse_response(response)
    
    def test_parse_no_decision_in_regex_json(self, mock_judge):
        """Test regex parsing when JSON doesn't contain decision."""
        response = 'Here is some text {"reasoning": "No decision field", "score": 5} more text.'
        
        with pytest.raises(ParseError):
            mock_judge._parse_response(response)
    
    def test_parse_data_validation_error(self, mock_judge):
        """Test handling of data validation errors."""
        response = json.dumps({
            "decision": "GOOD",
            "reasoning": "Valid response"
        })
        
        # Mock validation to raise an exception
        with patch.object(mock_judge, '_validate_and_normalize_data') as mock_validate:
            mock_validate.side_effect = ValueError("Validation failed")
            
            with pytest.raises(ParseError) as exc_info:
                mock_judge._parse_response(response)
            
            assert "Data validation failed: Validation failed" in str(exc_info.value)


class TestResponseParsingEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_parse_with_leading_trailing_whitespace(self, mock_judge):
        """Test parsing with whitespace around JSON."""
        response = '  \n\t  {"decision": "GOOD", "reasoning": "Clean response"}  \n  '
        
        result = mock_judge._parse_response(response)
        
        assert result.decision == "GOOD"
        assert result.reasoning == "Clean response"
    
    def test_parse_large_response(self, mock_judge):
        """Test parsing with very large response content."""
        large_reasoning = "X" * 10000  # 10k character reasoning
        response = json.dumps({
            "decision": "GOOD",
            "reasoning": large_reasoning,
            "score": 8
        })
        
        result = mock_judge._parse_response(response)
        
        assert result.decision == "GOOD"
        assert result.reasoning == large_reasoning
        assert len(result.reasoning) == 10000
    
    def test_parse_multiple_json_objects_takes_first(self, mock_judge):
        """Test that first valid JSON is used when multiple exist."""
        response = '''
        First attempt: {"decision": "GOOD", "reasoning": "First JSON"}
        
        Second attempt: {"decision": "EXCELLENT", "reasoning": "Second JSON"}
        '''
        
        # This should not fail and return the first valid JSON
        result = mock_judge._parse_response(response)
        assert result.decision == "GOOD"
        assert result.reasoning == "First JSON"
    
    def test_parse_json_with_special_characters(self, mock_judge):
        """Test parsing JSON with special characters."""
        response = json.dumps({
            "decision": "GOOD",
            "reasoning": "Response with émojis 🎯 and quotes \"nested\" and newlines\nand tabs\t",
            "score": 7.5
        })
        
        result = mock_judge._parse_response(response)
        
        assert result.decision == "GOOD"
        assert "émojis 🎯" in result.reasoning
        assert "\"nested\"" in result.reasoning
        assert "\n" in result.reasoning
    
    def test_parse_unicode_content(self, mock_judge):
        """Test parsing with Unicode content."""
        response = json.dumps({
            "decision": "优秀",  # Chinese characters
            "reasoning": "评估结果：这是一个很好的回答。",
            "score": 9
        }, ensure_ascii=False)
        
        result = mock_judge._parse_response(response)
        
        assert result.decision == "优秀"
        assert "评估结果" in result.reasoning


class TestResponseParsingLogging:
    """Test logging behavior during parsing."""
    
    def test_parse_logs_strategy_success(self, mock_judge):
        """Test that successful parsing strategy is logged."""
        response = json.dumps({"decision": "GOOD", "reasoning": "Clear"})
        
        with patch('vllm_judge.judge.logger') as mock_logger:
            mock_judge._parse_response(response)
            
            # Should log the response preview and successful strategy
            mock_logger.debug.assert_any_call(f"Parsing response: {response[:100]}...")
            mock_logger.debug.assert_any_call("Successfully parsed using direct JSON")
    
    def test_parse_logs_markdown_strategy(self, mock_judge):
        """Test logging when markdown strategy succeeds."""
        response = '```json\n{"decision": "GOOD", "reasoning": "Clear"}\n```'
        
        with patch('vllm_judge.judge.logger') as mock_logger:
            mock_judge._parse_response(response)
            
            mock_logger.debug.assert_any_call("Successfully parsed using markdown JSON")
    
    def test_parse_logs_regex_strategy(self, mock_judge):
        """Test logging when regex strategy succeeds."""
        # First two strategies should fail, regex should succeed
        response = 'Analysis: {"decision": "GOOD", "reasoning": "Found it"}'
        
        with patch('vllm_judge.judge.logger') as mock_logger:
            # Mock the first two strategies to return None
            with patch.object(mock_judge, '_parse_direct_json', return_value=None), \
                 patch.object(mock_judge, '_parse_markdown_json', return_value=None):
                
                mock_judge._parse_response(response)
                
                mock_logger.debug.assert_any_call("Successfully parsed using regex JSON")
    
    def test_parse_logs_failed_attempts(self, mock_judge):
        """Test that failed parsing attempts are logged at debug level."""
        response = "invalid json content"
        
        with patch('vllm_judge.judge.logger') as mock_logger:
            with pytest.raises(ParseError):
                mock_judge._parse_response(response)
            
            # Should have debug logs for each failed strategy
            debug_calls = [call.args[0] for call in mock_logger.debug.call_args_list]
            assert any("Direct JSON parsing failed" in call for call in debug_calls)

class TestParseHelperMethods:
    """Test individual parsing helper methods."""
    
    def test_parse_direct_json_valid(self, mock_judge):
        """Test _parse_direct_json with valid JSON."""
        response = json.dumps({"decision": "GOOD", "reasoning": "Clear"})
        
        result = mock_judge._parse_direct_json(response)
        
        assert result is not None
        assert result["decision"] == "GOOD"
        assert result["reasoning"] == "Clear"
    
    def test_parse_direct_json_invalid(self, mock_judge):
        """Test _parse_direct_json with invalid JSON."""
        response = "not json"
        
        with patch('vllm_judge.judge.logger') as mock_logger:
            result = mock_judge._parse_direct_json(response)
            
            assert result is None
            mock_logger.debug.assert_called_once()
    
    def test_parse_direct_json_with_whitespace(self, mock_judge):
        """Test _parse_direct_json handles whitespace correctly."""
        response = '  \n {"decision": "GOOD", "reasoning": "Clean"} \t  '
        
        result = mock_judge._parse_direct_json(response)
        
        assert result is not None
        assert result["decision"] == "GOOD"
    
    def test_parse_markdown_json_with_marker(self, mock_judge):
        """Test _parse_markdown_json with json marker."""
        response = '''
        ```json
        {"decision": "EXCELLENT", "reasoning": "Great work"}
        ```
        '''
        
        result = mock_judge._parse_markdown_json(response)
        
        assert result is not None  
        assert result["decision"] == "EXCELLENT"
    
    def test_parse_markdown_json_without_marker(self, mock_judge):
        """Test _parse_markdown_json without json marker."""
        response = '''
        ```
        {"decision": "FAIR", "reasoning": "Adequate"}
        ```
        '''
        
        result = mock_judge._parse_markdown_json(response)
        
        assert result is not None
        assert result["decision"] == "FAIR"
    
    def test_parse_markdown_json_invalid_json(self, mock_judge):
        """Test _parse_markdown_json with invalid JSON in markdown."""
        response = '''
        ```json
        {"decision": "GOOD", "reasoning": invalid}
        ```
        '''
        
        with patch('vllm_judge.judge.logger') as mock_logger:
            result = mock_judge._parse_markdown_json(response)
            
            assert result is None
            mock_logger.debug.assert_called_once()
    
    def test_parse_markdown_json_no_code_block(self, mock_judge):
        """Test _parse_markdown_json with no code blocks."""
        response = "Just regular text with no code blocks"
        
        result = mock_judge._parse_markdown_json(response)
        
        assert result is None
    
    def test_parse_regex_json_simple(self, mock_judge):
        """Test _parse_regex_json with simple case."""
        response = 'Analysis: {"decision": "GOOD", "reasoning": "Works well"} end'
        
        result = mock_judge._parse_regex_json(response)
        
        assert result is not None
        assert result["decision"] == "GOOD"
    
    def test_parse_regex_json_no_decision_field(self, mock_judge):
        """Test _parse_regex_json when JSON lacks decision field."""
        response = 'Text {"reasoning": "No decision here", "score": 5} more text'
        
        result = mock_judge._parse_regex_json(response)
        
        assert result is None
    
    def test_parse_regex_json_invalid_json(self, mock_judge):
        """Test _parse_regex_json with malformed JSON."""
        response = 'Text {"decision": "GOOD", "reasoning": invalid} more'
        
        with patch('vllm_judge.judge.logger') as mock_logger:
            result = mock_judge._parse_regex_json(response)
            
            assert result is None
            mock_logger.debug.assert_called_once()
    
    def test_validate_and_normalize_data_valid(self, mock_judge):
        """Test _validate_and_normalize_data with valid data."""
        data = {
            "decision": "GOOD",
            "reasoning": "Clear explanation",
            "score": 8.5
        }
        response = "test response"
        
        result = mock_judge._validate_and_normalize_data(data, response)
        
        assert result["decision"] == "GOOD"
        assert result["reasoning"] == "Clear explanation"
        assert result["score"] == 8.5
    
    def test_validate_and_normalize_data_missing_decision(self, mock_judge):
        """Test _validate_and_normalize_data with missing decision."""
        data = {"reasoning": "No decision"}
        response = "test response"
        
        with pytest.raises(ParseError) as exc_info:
            mock_judge._validate_and_normalize_data(data, response)
        
        assert "missing required 'decision' field" in str(exc_info.value)
    
    def test_validate_and_normalize_data_reason_fallback(self, mock_judge):
        """Test _validate_and_normalize_data with reason fallback."""
        data = {
            "decision": "PASS",
            "reason": "Used reason field"
        }
        response = "test response"
        
        with patch('vllm_judge.judge.logger') as mock_logger:
            result = mock_judge._validate_and_normalize_data(data, response)
            
            assert result["reasoning"] == "Used reason field"
            mock_logger.debug.assert_called_with("Used 'reason' field for reasoning")
    
    def test_validate_and_normalize_data_score_conversion(self, mock_judge):
        """Test _validate_and_normalize_data score type conversion."""
        data = {
            "decision": "GOOD", 
            "reasoning": "Test",
            "score": "7.5"  # String score
        }
        response = "test response"
        
        with patch('vllm_judge.judge.logger') as mock_logger:
            result = mock_judge._validate_and_normalize_data(data, response)
            
            assert result["score"] == 7.5
            assert isinstance(result["score"], float)
            mock_logger.debug.assert_called_with("Converted score to float")

class TestResponseParsingIntegration:
    """Test parsing integration with full Judge workflow."""
    
    def test_parse_includes_metadata(self, mock_judge):
        """Test that parsing adds metadata."""
        response = json.dumps({
            "decision": "EXCELLENT",
            "reasoning": "Outstanding work",
            "score": 9.5,
            "metadata": {"custom": "value"}
        })
        
        result = mock_judge._parse_response(response)
        
        # Should have model metadata
        assert result.metadata["model"] == mock_judge.config.model
        # Should preserve raw response
        assert result.metadata["raw_response"] == response
        # Should add metadata
        assert result.metadata["custom"] == "value"
    
    def test_parse_response_end_to_end(self, mock_judge):
        """Test a realistic end-to-end parsing scenario."""
        # Simulate a realistic LLM response
        response = """
        I'll evaluate this response based on the given criteria.

        Looking at the content, I can see that it addresses the main points effectively.
        The explanation is clear and well-structured.

        ```json
        {
            "decision": "GOOD",
            "reasoning": "The response demonstrates good understanding of the topic and provides clear explanations. The structure is logical and easy to follow.",
            "score": 8.2
        }
        ```

        This evaluation considers both accuracy and clarity of the response.
        """
        
        result = mock_judge._parse_response(response)
        
        assert result.decision == "GOOD"
        assert "good understanding" in result.reasoning
        assert result.score == 8.2
        assert isinstance(result, EvaluationResult)
