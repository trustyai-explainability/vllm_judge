import pytest
import httpx
from unittest.mock import AsyncMock, Mock, patch
from vllm_judge.client import VLLMClient, detect_model_sync
from vllm_judge.exceptions import ConnectionError, TimeoutError, ParseError


class TestVLLMClient:
    """Test VLLMClient functionality."""
    
    def test_client_initialization(self, mock_config):
        """Test VLLMClient initialization."""
        client = VLLMClient(mock_config)
        assert client.config == mock_config
        assert client.session is not None
    
    async def test_client_context_manager(self, mock_config):
        """Test VLLMClient as async context manager."""
        async with VLLMClient(mock_config) as client:
            assert client is not None
    
    async def test_chat_completion_success(self, mock_config, mock_httpx_client):
        """Test successful chat completion."""
        client = VLLMClient(mock_config)
        messages = [{"role": "user", "content": "Test message"}]
        
        response = await client.chat_completion(messages)
        assert response == '{"decision": "GOOD", "reasoning": "Test reasoning"}'
    
    async def test_chat_completion_invalid_response(self, mock_config, monkeypatch):
        """Test chat completion with invalid response format."""
        client = VLLMClient(mock_config)
        
        # Mock response without choices
        mock_session = AsyncMock()
        mock_response = Mock()
        mock_response.json.return_value = {"error": "no choices"}
        mock_response.raise_for_status.return_value = None
        mock_session.post.return_value = mock_response
        client.session = mock_session
        
        messages = [{"role": "user", "content": "Test"}]
        
        with pytest.raises(ParseError):
            await client.chat_completion(messages)
    
    async def test_completion_success(self, mock_config, mock_httpx_client):
        """Test successful completion."""
        # Modify mock to return completion format
        mock_httpx_client.post.return_value.json.return_value = {
            "choices": [{"text": "Completion response"}]
        }
        
        client = VLLMClient(mock_config)
        response = await client.completion("Test prompt")
        assert response == "Completion response"
    
    async def test_connection_error(self, mock_config, monkeypatch):
        """Test connection error handling."""
        client = VLLMClient(mock_config)
        
        # Mock connection error
        mock_session = AsyncMock()
        mock_session.post.side_effect = httpx.ConnectError("Connection failed")
        client.session = mock_session
        
        messages = [{"role": "user", "content": "Test"}]
        
        with pytest.raises(ConnectionError):
            await client.chat_completion(messages)
    
    async def test_timeout_error(self, mock_config, monkeypatch):
        """Test timeout error handling."""
        client = VLLMClient(mock_config)
        
        # Mock timeout error
        mock_session = AsyncMock()
        mock_session.post.side_effect = httpx.TimeoutException("Request timed out")
        client.session = mock_session
        
        messages = [{"role": "user", "content": "Test"}]
        
        with pytest.raises((TimeoutError, ConnectionError)):
            await client.chat_completion(messages)
    
    async def test_list_models(self, mock_config, mock_httpx_client):
        """Test listing models."""
        # Mock models response
        mock_httpx_client.post.return_value.json.return_value = {
            "data": [
                {"id": "model-1"},
                {"id": "model-2"}
            ]
        }
        
        client = VLLMClient(mock_config)
        models = await client.list_models()
        assert models == ["model-1", "model-2"]
    
    async def test_detect_model(self, mock_config, mock_httpx_client):
        """Test auto-detecting model."""
        # Mock models response
        mock_httpx_client.post.return_value.json.return_value = {
            "data": [{"id": "auto-detected-model"}]
        }
        
        client = VLLMClient(mock_config)
        model = await client.detect_model()
        assert model == "auto-detected-model"
    
    async def test_detect_model_no_models(self, mock_config, mock_httpx_client):
        """Test detect model when no models available."""
        # Mock empty models response
        mock_httpx_client.post.return_value.json.return_value = {"data": []}
        
        client = VLLMClient(mock_config)
        
        with pytest.raises(ConnectionError):
            await client.detect_model()


class TestDetectModelSync:
    """Test synchronous model detection."""
    
    def test_detect_model_sync_success(self, monkeypatch):
        """Test successful synchronous model detection."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [{"id": "sync-detected-model"}]
        }
        mock_response.raise_for_status.return_value = None
        
        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        
        with patch('httpx.Client', return_value=mock_client):
            model = detect_model_sync("http://localhost:8000")
            assert model == "sync-detected-model"
    
    def test_detect_model_sync_no_models(self, monkeypatch):
        """Test sync model detection with no models."""
        mock_response = Mock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status.return_value = None
        
        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        
        with patch('httpx.Client', return_value=mock_client):
            with pytest.raises(ConnectionError):
                detect_model_sync("http://localhost:8000")
    
    def test_detect_model_sync_http_error(self, monkeypatch):
        """Test sync model detection with HTTP error."""
        mock_client = Mock()
        mock_client.get.side_effect = httpx.HTTPError("HTTP error")
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        
        with patch('httpx.Client', return_value=mock_client):
            with pytest.raises(ConnectionError):
                detect_model_sync("http://localhost:8000")