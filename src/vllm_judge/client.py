from typing import List, Dict, Any, Optional, Union
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    # before_retry
)

from vllm_judge.models import JudgeConfig
from vllm_judge.exceptions import (
    ConnectionError,
    TimeoutError,
    ParseError,
    RetryExhaustedError
)

CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"
COMPLETIONS_ENDPOINT = "/v1/completions"
MODELS_ENDPOINT = "/v1/models"

class VLLMClient:
    """Async client for vLLM endpoints."""
    
    def __init__(self, config: JudgeConfig):
        """
        Initialize vLLM client.
        
        Args:
            config: Judge configuration
        """
    
        if not config.model:
            config.model = detect_model_sync(config.base_url)
        self.config = config
        self.session = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(config.timeout),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20
            ),
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json"
            }
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close the HTTP session."""
        await self.session.aclose()
    
    def _log_retry(self, retry_state):
        """Log retry attempts."""
        attempt = retry_state.attempt_number
        if attempt > 1:
            print(f"Retry attempt {attempt} after error: {retry_state.outcome.exception()}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, ConnectionError, TimeoutError)),
        # before=before_retry(lambda retry_state: retry_state.outcome and print(
        #     f"Retrying after error: {retry_state.outcome.exception()}"
        # ))
    )
    async def _request_with_retry(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic.
        
        Args:
            endpoint: API endpoint
            **kwargs: Request parameters
            
        Returns:
            Parsed JSON response
            
        Raises:
            ConnectionError: If unable to connect
            TimeoutError: If request times out
            RetryExhaustedError: If all retries fail
        """
        try:
            response = await self.session.post(endpoint, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError as e:
            raise ConnectionError(f"Failed to connect to {self.config.base_url}: {e}")
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Request timed out after {self.config.timeout}s: {e}")
        except httpx.HTTPStatusError as e:
            # Parse error message from response if available
            try:
                error_detail = e.response.json().get('detail', str(e))
            except:
                error_detail = str(e)
            raise ConnectionError(f"HTTP {e.response.status_code}: {error_detail}")
        except Exception as e:
            raise ConnectionError(f"Unexpected error: {e}")
    
    async def chat_completion(self, messages: List[Dict[str, str]], 
                              sampling_params: Optional[Dict[str, Any]] = None,
                              return_choices: bool = False) -> Union[str, List[Dict[str, Any]]]:
        """
        Use chat completions endpoint (handles templates automatically).
        
        Args:
            messages: List of chat messages
            
        Returns:
            String model response if 'n' (no. of generations) is 1 (default), otherwise the choices (list of dicts) field of the response
            
        Raises:
            ConnectionError: If request fails
            ParseError: If response parsing fails
        """
        request_data = {
            "model": self.config.model,
            "messages": messages,
        }

        if sampling_params:
            request_data.update(sampling_params)
        
        try:
            response = await self._request_with_retry(
                CHAT_COMPLETIONS_ENDPOINT,
                json=request_data
            )
            
            # Extract content from response
            if "choices" not in response or not response["choices"]:
                raise ParseError("Invalid response format: missing choices")
            
            if return_choices:
                return response["choices"]
            else:
                return response["choices"][0]["message"]["content"]
            
        except RetryExhaustedError:
            raise
        except Exception as e:
            if isinstance(e, (ConnectionError, TimeoutError, ParseError)):
                raise
            raise ConnectionError(f"Chat completion failed: {e}")
    
    async def completion(self, prompt: str, 
                         sampling_params: Optional[Dict[str, Any]] = None,
                         return_choices: bool = False) -> Union[str, List[Dict[str, Any]]]:
        """
        Use completions endpoint for edge cases.
        
        Args:
            prompt: Text prompt
            
        Returns:
            String model response if 'n' (no. of generations) is 1 (default), otherwise the choices (list of dicts) field of the response
            
        Raises:
            ConnectionError: If request fails
            ParseError: If response parsing fails
        """
        request_data = {
            "model": self.config.model,
            "prompt": prompt,
        }

        if sampling_params:
            request_data.update(sampling_params)
        
        try:
            response = await self._request_with_retry(
                COMPLETIONS_ENDPOINT,
                json=request_data
            )
            
            # Extract text from response
            if "choices" not in response or not response["choices"]:
                raise ParseError("Invalid response format: missing choices")
            
            if return_choices:
                return response["choices"]
            else:
                return response["choices"][0]["text"]
            
        except RetryExhaustedError:
            raise
        except Exception as e:
            if isinstance(e, (ConnectionError, TimeoutError, ParseError)):
                raise
            raise ConnectionError(f"Completion failed: {e}")
    
    async def list_models(self) -> List[str]:
        """
        List available models.
        
        Returns:
            List of model names
            
        Raises:
            ConnectionError: If request fails
        """
        try:
            response = await self._request_with_retry(MODELS_ENDPOINT)
            models = response.get("data", [])
            return [model["id"] for model in models]
        except Exception as e:
            if isinstance(e, ConnectionError):
                raise
            raise ConnectionError(f"Failed to list models: {e}")
    
    async def detect_model(self) -> str:
        """
        Auto-detect the first available model.
        
        Returns:
            Model name
            
        Raises:
            ConnectionError: If no models found
        """
        models = await self.list_models()
        if not models:
            raise ConnectionError("No models available on vLLM server")
        return models[0]
        

def detect_model_sync(base_url: str, timeout: float = 30.0) -> str:
    """
    Synchronously detect the first available model.
    
    Args:
        base_url: vLLM server URL
        timeout: Request timeout
        
    Returns:
        Model name
        
    Raises:
        ConnectionError: If no models found
    """
    url = f"{base_url}{MODELS_ENDPOINT}"
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json().get("data", [])
            models = [model["id"] for model in data]
            
            if not models:
                raise ConnectionError("No models available on vLLM server")
            
            model = models[0]
            return model
            
    except httpx.HTTPError as e:
        raise ConnectionError(f"Failed to detect model: {e}")