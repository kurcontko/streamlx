# src/config/model_config.py
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration settings for the MLX-LM model."""
    DEFAULT_MAX_TOKENS: int = 4096
    DEFAULT_TEMPERATURE: float = 0.3
    DEFAULT_MAX_HISTORY: int = 50
    REQUEST_TIMEOUT: int = 30
