class VLLMJudgeError(Exception):
    """Base exception for all vLLM Judge errors."""
    pass


class ConfigurationError(VLLMJudgeError):
    """Raised when configuration is invalid."""
    pass


class ConnectionError(VLLMJudgeError):
    """Raised when unable to connect to vLLM server."""
    pass


class TimeoutError(VLLMJudgeError):
    """Raised when request times out."""
    pass


class ParseError(VLLMJudgeError):
    """Raised when unable to parse LLM response."""
    def __init__(self, message: str, raw_response: str = None):
        super().__init__(message)
        self.raw_response = raw_response


class MetricNotFoundError(VLLMJudgeError):
    """Raised when requested metric is not found."""
    pass


class InvalidInputError(VLLMJudgeError):
    """Raised when input parameters are invalid."""
    pass


class RetryExhaustedError(VLLMJudgeError):
    """Raised when all retry attempts are exhausted."""
    def __init__(self, message: str, last_error: Exception = None):
        super().__init__(message)
        self.last_error = last_error