import pytest
from unittest.mock import AsyncMock, Mock, patch
from vllm_judge import Judge, JudgeConfig, EvaluationResult


@pytest.fixture
def mock_config():
    """Mock JudgeConfig for testing."""
    return JudgeConfig(
        base_url="http://localhost:8000",
        model="test-model",
        api_key="test-key",
        timeout=30.0,
        max_retries=3
    )


@pytest.fixture
def sample_evaluation_result():
    """Sample EvaluationResult for testing."""
    return EvaluationResult(
        decision="GOOD",
        reasoning="The response is clear and accurate.",
        score=8.5,
        metadata={"model": "test-model"}
    )


@pytest.fixture
def mock_vllm_response():
    """Mock vLLM API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": '{"decision": "GOOD", "reasoning": "The response is clear and accurate.", "score": 8.5}'
                }
            }
        ]
    }


@pytest.fixture
def mock_models_response():
    """Mock models list response."""
    return {
        "data": [
            {"id": "test-model-1"},
            {"id": "test-model-2"}
        ]
    }


@pytest.fixture
async def mock_judge(mock_config):
    """Mock Judge instance with mocked HTTP client."""
    judge = Judge(mock_config)
    
    # Mock the HTTP client
    mock_session = AsyncMock()
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"decision": "GOOD", "reasoning": "Test reasoning", "score": 8.0}'
                }
            }
        ]
    }
    mock_response.raise_for_status.return_value = None
    mock_session.post.return_value = mock_response
    
    judge.client.session = mock_session
    
    return judge


@pytest.fixture
def mock_httpx_client(monkeypatch):
    """Mock httpx client for testing client functionality."""
    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"decision": "GOOD", "reasoning": "Test reasoning"}'
                }
            }
        ]
    }
    mock_response.raise_for_status.return_value = None
    mock_client.post.return_value = mock_response
    
    # Mock httpx.AsyncClient to return our mock
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: mock_client)
    return mock_client

@pytest.fixture
def mock_judge_client_session():
    """Mock httpx.AsyncClient session for JudgeClient testing."""
    mock_session = AsyncMock()
    mock_response = Mock()
    mock_response.json.return_value = {
        "decision": "GOOD",
        "reasoning": "Test reasoning",
        "score": 8.0,
        "metadata": {}
    }
    mock_response.raise_for_status.return_value = None
    mock_session.post.return_value = mock_response
    mock_session.get.return_value = mock_response
    
    with patch('httpx.AsyncClient', return_value=mock_session):
        yield mock_session