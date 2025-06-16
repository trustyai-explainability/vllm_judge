import pytest
from unittest.mock import AsyncMock, Mock, patch
from vllm_judge.api.client import JudgeClient
from vllm_judge.models import EvaluationResult

def _has_fastapi() -> bool:
    """Check if FastAPI is available."""
    try:
        import fastapi
        return True
    except ImportError:
        return False

class TestJudgeClient:
    """Test JudgeClient for API interactions."""
    
    def test_judge_client_init(self):
        """Test JudgeClient initialization."""
        client = JudgeClient("http://localhost:9090")
        assert client.api_url == "http://localhost:9090"  # Changed from base_url to api_url
    
    async def test_judge_client_evaluate(self):
        """Test JudgeClient.evaluate method."""
        with patch('httpx.AsyncClient') as mock_client_class:
            # Create proper async mock
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
            
            # Set up the AsyncClient mock properly
            mock_client_class.return_value = mock_session
            
            client = JudgeClient("http://localhost:9090")
            result = await client.evaluate(
                content="Test content",
                criteria="Test criteria"
            )
            
            assert isinstance(result, EvaluationResult)
            assert result.decision == "GOOD"
            assert result.reasoning == "Test reasoning"
            assert result.score == 8.0
    
    async def test_judge_client_batch_evaluate(self):
        """Test JudgeClient.batch_evaluate method."""
        with patch('httpx.AsyncClient') as mock_client_class:
            # Create proper async mock
            mock_session = AsyncMock()
            mock_response = Mock()
            mock_response.json.return_value = {
                "results": [
                    {
                        "decision": "GOOD",
                        "reasoning": "Test reasoning 1",
                        "score": 8.0,
                        "metadata": {}
                    },
                    {
                        "decision": "EXCELLENT", 
                        "reasoning": "Test reasoning 2",
                        "score": 9.0,
                        "metadata": {}
                    }
                ],
                "total": 2,
                "successful": 2,
                "failed": 0,
                "duration_seconds": 1.5
            }
            mock_response.raise_for_status.return_value = None
            mock_session.post.return_value = mock_response
            
            # Set up the AsyncClient mock properly
            mock_client_class.return_value = mock_session
            
            client = JudgeClient("http://localhost:9090")
            data = [
                {"content": "Text 1", "criteria": "quality"},
                {"content": "Text 2", "criteria": "accuracy"}
            ]
            
            result = await client.batch_evaluate(data)
            
            # Check that we got a BatchResult
            assert result is not None
            assert result.total == 2
            assert result.successful == 2
            assert result.failed == 0


@pytest.mark.skipif(
    not _has_fastapi(),
    reason="FastAPI not available"
)
class TestAPIServer:
    """Test API server functionality."""
    
    def test_api_server_imports(self):
        """Test that API server components can be imported."""
        try:
            from vllm_judge.api.server import app
            from vllm_judge.api.models import EvaluateRequest
            assert app is not None
        except ImportError:
            pytest.skip("API components not available")
    
    def test_api_server_health_check_with_mock_judge(self):
        """Test API server health check endpoint with mocked judge."""
        try:
            from fastapi.testclient import TestClient
            from vllm_judge.api.server import app
            from vllm_judge import JudgeConfig
            
            # Mock the global judge in the server module
            with patch('vllm_judge.api.server.judge') as mock_judge:
                # Set up mock judge
                mock_config = Mock()
                mock_config.model = "test-model"
                mock_config.base_url = "http://localhost:8000"
                mock_judge.config = mock_config
                mock_judge.list_metrics.return_value = []
                
                # Mock app_start_time
                with patch('vllm_judge.api.server.app_start_time', 1234567890.0):
                    with patch('vllm_judge.api.server.time.time', return_value=1234567900.0):
                        client = TestClient(app)
                        response = client.get("/health")
                        assert response.status_code == 200
                        data = response.json()
                        assert data["status"] == "healthy"
                        assert data["model"] == "test-model"
        except ImportError:
            pytest.skip("FastAPI test client not available")
