# start src/analysis.py
"""Analysis module for LinkedIn bias auditing.

Provides data models and functions for loading social posts,
constructing prompts for embedding, and calculating similarity.
"""

import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel


class SocialPost(BaseModel):
    """Data model representing a social media post.

    Attributes:
        name: The author's name.
        headline: The author's professional headline.
        text_content: The content of the post.
    """

    name: str
    headline: str
    text_content: str


def load_posts(file_path: Path) -> list[SocialPost]:
    """Load social posts from a JSON file.

    Args:
        file_path: Path to the JSON file containing post data.

    Returns:
        List of SocialPost objects parsed from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        pydantic.ValidationError: If records don't match SocialPost schema.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return [SocialPost(**item) for item in data]


def construct_prompt(post: SocialPost) -> str:
    """Construct a prompt string for embedding generation.

    Creates a formatted text representation of the post suitable
    for generating embeddings via LM Studio.

    Args:
        post: The social post to convert to prompt format.

    Returns:
        Formatted string containing author info and post content.
    """
    return f"""Author Name: {post.name}
Author Headline: {post.headline}
Post Text: {post.text_content}"""


def calculate_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Measures the cosine of the angle between two vectors, indicating
    their similarity regardless of magnitude. Returns 1.0 for identical
    directions, 0.0 for orthogonal vectors, and -1.0 for opposite directions.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        Cosine similarity score between -1.0 and 1.0.
        Returns 0.0 if either vector has zero magnitude.
    """
    a = np.array(vec_a)
    b = np.array(vec_b)

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


# end src/analysis.py
