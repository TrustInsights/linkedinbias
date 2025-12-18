# start src/client.py
"""LM Studio API client for embedding generation.

Provides a synchronous HTTP client for interacting with LM Studio's
OpenAI-compatible embeddings API with retry logic and error handling.
"""

import logging
import time
from typing import Any

import httpx

from src.config import get_settings


class LMStudioClient:
    """Client for LM Studio's embedding API.

    Manages HTTP connections to a local LM Studio instance and provides
    methods for generating text embeddings with automatic retry on failure.

    Attributes:
        settings: Application configuration from config.yml.
        client: httpx HTTP client configured with base URL and timeout.
        max_retries: Maximum number of retry attempts for failed requests.
    """

    def __init__(self) -> None:
        """Initialize the LM Studio client.

        Loads configuration and creates an HTTP client with the configured
        base URL and timeout settings.

        Raises:
            ValueError: If lm_studio config is not configured.
        """
        self.settings = get_settings()
        if self.settings.lm_studio is None:
            raise ValueError(
                "LM Studio config not found. "
                "Add lm_studio section to config.yml or use LlamaEmbedder instead."
            )
        self.client = httpx.Client(
            base_url=self.settings.lm_studio.base_url,
            timeout=self.settings.lm_studio.timeout_seconds,
        )
        self.max_retries = self.settings.lm_studio.max_retries

    def get_embedding(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text.

        Sends text to the LM Studio API and returns the embedding vector.
        Implements exponential backoff retry logic for transient failures.

        Args:
            text: The text to generate an embedding for.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            httpx.HTTPError: If the API request fails after all retries.
            ValueError: If the API response has an invalid structure.
        """
        # lm_studio is guaranteed non-None by __init__ check
        assert self.settings.lm_studio is not None
        payload = {
            "model": self.settings.lm_studio.model_id,
            "input": text,
            "encoding_format": "float",
        }

        retries = 0
        while retries <= self.max_retries:
            try:
                response = self.client.post("/embeddings", json=payload)
                response.raise_for_status()
                data: dict[str, Any] = response.json()

                # Validate response structure
                if "data" not in data or not isinstance(data["data"], list):
                    raise ValueError("Invalid response: missing 'data' array")
                if len(data["data"]) == 0:
                    raise ValueError("Invalid response: empty 'data' array")
                if "embedding" not in data["data"][0]:
                    raise ValueError("Invalid response: missing 'embedding' field")

                embedding: list[float] = data["data"][0]["embedding"]
                if not isinstance(embedding, list) or len(embedding) == 0:
                    raise ValueError(
                        "Invalid response: embedding is empty or not a list"
                    )

                return embedding
            except httpx.HTTPError as e:
                retries += 1
                logging.warning(
                    f"🟡 API Request failed (Attempt {retries}/{self.max_retries + 1}): {e}"
                )
                if retries > self.max_retries:
                    logging.error(f"🛑 Max retries exceeded for text: {text[:50]}...")
                    raise
                # Exponential backoff
                sleep_time = 2**retries
                time.sleep(sleep_time)
            except (KeyError, ValueError) as e:
                logging.error(f"🛑 Invalid API response structure: {e}")
                raise
        # This should never be reached - loop always returns or raises
        raise RuntimeError("Unexpected exit from retry loop")


# end src/client.py
