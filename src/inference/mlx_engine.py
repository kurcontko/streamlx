# src/inference/mlx_engine.py
from typing import Iterator, List, Dict, Tuple
import requests
import json
from ..core.cache import cache_with_ttl
from ..config.model_config import ModelConfig


class MLXInference:
    """MLX-LM inference engine connection using basic requests."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        """Initialize the MLX-LM inference engine."""
        self.base_url = base_url.rstrip('/')

    @cache_with_ttl(ttl=60)
    def check_server_health(self) -> Tuple[bool, str]:
        """Check if the MLX-LM server is running by testing models endpoint."""
        try:
            response = requests.get(
                f"{self.base_url}/v1/models",
                timeout=ModelConfig.REQUEST_TIMEOUT
            )
            is_healthy = response.status_code == 200
            message = "MLX-LM Server is running" if is_healthy else "MLX-LM Server is not responding properly"
            return is_healthy, message
        except requests.exceptions.RequestException as e:
            return False, f"Cannot connect to MLX-LM Server: {str(e)}"

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True,
        max_tokens: int = ModelConfig.DEFAULT_MAX_TOKENS,
        temperature: float = ModelConfig.DEFAULT_TEMPERATURE
    ) -> Iterator[str]:
        """Get completion from MLX-LM server."""
        try:
            payload = {
                "messages": messages,
                "temperature": temperature,
                "stream": stream
            }
            if max_tokens:
                payload["max_tokens"] = max_tokens

            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                stream=stream,
                timeout=None
            )
            response.raise_for_status()

            if stream:
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]
                            if data == '[DONE]':
                                break
                            try:
                                chunk = json.loads(data)
                                if chunk.get("choices") and chunk["choices"][0].get("delta"):
                                    content = chunk["choices"][0]["delta"].get("content", "")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
            else:
                data = response.json()
                if data.get("choices") and data["choices"][0].get("message"):
                    yield data["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            yield f"Error: Could not connect to MLX server: {str(e)}"