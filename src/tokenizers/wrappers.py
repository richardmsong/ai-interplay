"""Unified tokenizer wrappers for round-trip distortion experiments.

Each wrapper implements the same interface:
    encode(text: str) -> list[int]    # text to token IDs
    decode(ids: list[int]) -> str     # token IDs back to text

The round-trip test is: serialize numbers to text -> encode -> decode -> parse numbers.
We're measuring what the tokenizer *channel* does to numeric information.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Serialization formats: how numbers become text before tokenization
# ---------------------------------------------------------------------------

def serialize_csv(vec: np.ndarray) -> str:
    """Comma-separated values: '1.234, 5.678, 9.012'"""
    return ", ".join(f"{x}" for x in vec)


def serialize_json(vec: np.ndarray) -> str:
    """JSON array: '[1.234, 5.678, 9.012]'"""
    return "[" + ", ".join(f"{x}" for x in vec) + "]"


def serialize_scientific(vec: np.ndarray) -> str:
    """Scientific notation: '1.234e+00, 5.678e+00'"""
    return ", ".join(f"{x:.6e}" for x in vec)


def serialize_natural_language(vec: np.ndarray) -> str:
    """Natural language: 'The values are 1.234, 5.678, and 9.012.'"""
    if len(vec) == 1:
        return f"The value is {vec[0]}."
    parts = ", ".join(f"{x}" for x in vec[:-1])
    return f"The values are {parts}, and {vec[-1]}."


def serialize_digit_by_digit(vec: np.ndarray) -> str:
    """Space-separated digits: '1 . 2 3 4 | 5 . 6 7 8'"""
    parts = []
    for x in vec:
        s = f"{x}"
        parts.append(" ".join(s))
    return " | ".join(parts)


SERIALIZATION_FORMATS = {
    "csv": serialize_csv,
    "json": serialize_json,
    "scientific": serialize_scientific,
    "natural_language": serialize_natural_language,
    "digit_by_digit": serialize_digit_by_digit,
}


# ---------------------------------------------------------------------------
# Deserialization: parse numbers back from decoded text
# ---------------------------------------------------------------------------

def deserialize_numbers(text: str) -> np.ndarray:
    """Extract all floating-point numbers from a string.

    This is intentionally permissive — it handles all serialization formats
    by finding anything that looks like a number.
    """
    pattern = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
    matches = re.findall(pattern, text)
    if not matches:
        return np.array([], dtype=np.float64)
    return np.array([float(m) for m in matches], dtype=np.float64)


# ---------------------------------------------------------------------------
# Tokenizer wrappers
# ---------------------------------------------------------------------------

@dataclass
class RoundTripResult:
    """Result of a single round-trip through a tokenizer."""
    original_text: str
    token_ids: list[int]
    decoded_text: str
    num_tokens: int


class TokenizerWrapper(ABC):
    """Base class for tokenizer wrappers."""

    name: str

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        ...

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        ...

    def round_trip(self, text: str) -> RoundTripResult:
        ids = self.encode(text)
        decoded = self.decode(ids)
        return RoundTripResult(
            original_text=text,
            token_ids=ids,
            decoded_text=decoded,
            num_tokens=len(ids),
        )


class LlamaTokenizer(TokenizerWrapper):
    """Llama-3 tokenizer (BPE via HuggingFace transformers)."""

    name = "llama3"

    def __init__(self, model_id: str = "meta-llama/Llama-3.1-8B-Instruct"):
        from transformers import AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(model_id)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)


class MistralTokenizer(TokenizerWrapper):
    """Mistral tokenizer (SentencePiece via HuggingFace transformers)."""

    name = "mistral"

    def __init__(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        from transformers import AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(model_id)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)


class GPT4oTokenizer(TokenizerWrapper):
    """GPT-4o tokenizer (cl200k_base via tiktoken)."""

    name = "gpt4o"

    def __init__(self, encoding_name: str = "o200k_base"):
        import tiktoken
        self._enc = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)


class PythonStringBaseline(TokenizerWrapper):
    """Control: Python str() round-trip. No tokenization — just string serialization.

    This isolates the loss from float->str->float conversion, which is the
    baseline loss before any tokenizer gets involved.
    """

    name = "python_str"

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8")


class Float16Baseline(TokenizerWrapper):
    """Control: float64->float16->float64. Known quantization loss for calibration.

    This isn't a real tokenizer — it operates on the numeric vector directly
    and is handled specially in the experiment pipeline.
    """

    name = "float16"

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8")

    def round_trip_numeric(self, vec: np.ndarray) -> np.ndarray:
        """Direct numeric round-trip: float64 -> float16 -> float64."""
        return vec.astype(np.float16).astype(np.float64)


def get_tokenizer(name: str) -> TokenizerWrapper:
    """Factory function. Returns a tokenizer wrapper by name."""
    registry = {
        "llama3": LlamaTokenizer,
        "mistral": MistralTokenizer,
        "gpt4o": GPT4oTokenizer,
        "python_str": PythonStringBaseline,
        "float16": Float16Baseline,
    }
    if name not in registry:
        raise ValueError(f"Unknown tokenizer: {name}. Available: {list(registry.keys())}")
    return registry[name]()


def get_all_tokenizer_names() -> list[str]:
    return ["llama3", "mistral", "gpt4o", "python_str", "float16"]
