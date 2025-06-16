from vllm_judge.exceptions import (
    VLLMJudgeError,
    ConfigurationError,
    ConnectionError,
    TimeoutError,
    ParseError,
    MetricNotFoundError,
    InvalidInputError,
    RetryExhaustedError
)


class TestExceptions:
    """Test custom exceptions."""
    
    def test_base_exception(self):
        """Test base VLLMJudgeError."""
        error = VLLMJudgeError("Base error")
        assert str(error) == "Base error"
        assert isinstance(error, Exception)
    
    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Invalid config")
        assert str(error) == "Invalid config"
        assert isinstance(error, VLLMJudgeError)
    
    def test_connection_error(self):
        """Test ConnectionError."""
        error = ConnectionError("Cannot connect to server")
        assert str(error) == "Cannot connect to server"
        assert isinstance(error, VLLMJudgeError)
    
    def test_timeout_error(self):
        """Test TimeoutError."""
        error = TimeoutError("Request timed out")
        assert str(error) == "Request timed out"
        assert isinstance(error, VLLMJudgeError)
    
    def test_parse_error(self):
        """Test ParseError with raw response."""
        raw_response = "Invalid JSON response"
        error = ParseError("Cannot parse response", raw_response=raw_response)
        
        assert str(error) == "Cannot parse response"
        assert error.raw_response == raw_response
        assert isinstance(error, VLLMJudgeError)
    
    def test_parse_error_without_raw_response(self):
        """Test ParseError without raw response."""
        error = ParseError("Cannot parse response")
        assert str(error) == "Cannot parse response"
        assert error.raw_response is None
    
    def test_metric_not_found_error(self):
        """Test MetricNotFoundError."""
        error = MetricNotFoundError("Metric 'unknown' not found")
        assert str(error) == "Metric 'unknown' not found"
        assert isinstance(error, VLLMJudgeError)
    
    def test_invalid_input_error(self):
        """Test InvalidInputError."""
        error = InvalidInputError("Invalid input parameters")
        assert str(error) == "Invalid input parameters"
        assert isinstance(error, VLLMJudgeError)
    
    def test_retry_exhausted_error(self):
        """Test RetryExhaustedError."""
        last_error = Exception("Last attempt failed")
        error = RetryExhaustedError("All retries failed", last_error=last_error)
        
        assert str(error) == "All retries failed"
        assert error.last_error == last_error
        assert isinstance(error, VLLMJudgeError)
    
    def test_retry_exhausted_error_without_last_error(self):
        """Test RetryExhaustedError without last error."""
        error = RetryExhaustedError("All retries failed")
        assert str(error) == "All retries failed"
        assert error.last_error is None