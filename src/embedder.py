# start src/embedder.py
"""LLaMA-3 embedding extraction via hidden state mean pooling.

Implements LinkedIn's methodology from arXiv:2510.14223 by extracting
hidden states from LLaMA-3 and applying mean pooling to create embeddings.
"""

import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LlamaEmbedder:
    """Extract embeddings from LLaMA-3 hidden states.

    Replicates LinkedIn's dual-encoder approach by:
    1. Feeding text through LLaMA-3 as a causal language model
    2. Extracting hidden states from the last transformer layer
    3. Applying mean pooling over all token representations

    Attributes:
        tokenizer: HuggingFace tokenizer for the model.
        model: LLaMA-3 model with hidden state output enabled.
        device: Device the model is running on (mps, cuda, cpu).
    """

    def __init__(self, model_path: str, device: str = "auto") -> None:
        """Initialize the LLaMA embedder.

        Args:
            model_path: Path to the local LLaMA model directory.
            device: Device to run on ("auto", "mps", "cuda", "cpu").
        """
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model not found at: {model_path}")

        logging.info(f"🔄 Loading LLaMA model from {model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)  # type: ignore[no-untyped-call]

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()  # type: ignore[no-untyped-call]

        # Determine actual device
        if hasattr(self.model, "device"):
            self.device = self.model.device
        else:
            self.device = next(self.model.parameters()).device

        logging.info(f"✅ Model loaded on device: {self.device}")

    def get_embedding(self, text: str) -> list[float]:
        """Extract embedding via mean pooling of last layer hidden states.

        This replicates the LinkedIn paper's methodology:
        - Feed text through the model
        - Get hidden states from the last transformer layer
        - Mean pool over the sequence dimension

        Args:
            text: Text to generate embedding for.

        Returns:
            List of floats representing the embedding vector (3072 dimensions).
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=False,
        )

        # Move inputs to model device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get model outputs with hidden states (request hidden states at inference time)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Extract last layer hidden states: (batch_size, seq_len, hidden_size)
        # hidden_states is a tuple of all layer outputs; [-1] is the last layer
        hidden_states = outputs.hidden_states[-1]

        # Mean pooling over sequence length: (batch_size, hidden_size)
        embedding = hidden_states.mean(dim=1)

        # Squeeze batch dimension and convert to list
        embedding = embedding.squeeze(0)

        # Convert to float32 for numpy compatibility and return as list
        result: list[float] = embedding.cpu().float().numpy().tolist()
        return result


# end src/embedder.py
