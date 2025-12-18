# start src/config.py
"""Configuration management for LinkedIn Bias Auditor.

Provides Pydantic models for type-safe configuration loading from YAML files.
Configuration is cached using lru_cache for efficient repeated access.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class AppConfig(BaseModel):
    """Application-level configuration.

    Attributes:
        name: Display name of the application.
        log_level: Logging verbosity level.
    """

    name: str
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class LMStudioConfig(BaseModel):
    """LM Studio API connection configuration.

    Attributes:
        base_url: Base URL for the LM Studio API (e.g., http://127.0.0.1:1234/v1).
        model_id: Model identifier to use for embeddings.
        timeout_seconds: HTTP request timeout in seconds.
        max_retries: Maximum retry attempts for failed requests.
    """

    base_url: str
    model_id: str
    timeout_seconds: int
    max_retries: int


class PathConfig(BaseModel):
    """File path configuration.

    Attributes:
        input_dir: Directory containing input data files.
        output_dir: Directory for output files and database.
    """

    input_dir: Path
    output_dir: Path


class EmbeddingConfig(BaseModel):
    """Local LLaMA embedding configuration.

    Attributes:
        model_path: Path to the local LLaMA model directory.
        device: Device to run inference on ("auto", "mps", "cuda", "cpu").
    """

    model_path: str = "models/Llama-3.2-3B-bf16"
    device: str = "auto"


class Settings(BaseSettings):
    """Main application settings container.

    Aggregates all configuration sections and provides loading from YAML.

    Attributes:
        app: Application-level settings.
        lm_studio: LM Studio API settings (legacy, optional).
        paths: File path settings.
        embedding: Local LLaMA embedding settings.
    """

    app: AppConfig
    lm_studio: LMStudioConfig | None = None
    paths: PathConfig
    embedding: EmbeddingConfig = EmbeddingConfig()

    @classmethod
    def load_from_yaml(cls, path: str = "config.yml") -> "Settings":
        """Load settings from a YAML configuration file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Settings instance populated from the file.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            yaml.YAMLError: If the file contains invalid YAML.
            pydantic.ValidationError: If values don't match expected types.
        """
        with open(path, "r") as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings.

    Loads settings from config.yml on first call and returns cached
    instance on subsequent calls.

    Returns:
        Cached Settings instance.
    """
    return Settings.load_from_yaml()


# end src/config.py
