"""Numeric vector generators for the translation tax experiments.

Generates vectors with controlled statistical properties for systematic
testing of tokenizer distortion across magnitudes, precisions, and distributions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VectorBatch:
    """A batch of test vectors with metadata about their generation."""
    vectors: np.ndarray          # shape (n_vectors, dim)
    distribution: str            # e.g., "uniform", "normal", "lognormal"
    params: dict                 # distribution parameters
    dim: int
    precision_digits: int | None  # None means full float64 precision


def generate_uniform(
    n: int, dim: int, low: float, high: float, seed: int, precision_digits: int | None = None
) -> VectorBatch:
    rng = np.random.default_rng(seed)
    vecs = rng.uniform(low, high, size=(n, dim))
    if precision_digits is not None:
        vecs = np.round(vecs, precision_digits)
    return VectorBatch(
        vectors=vecs,
        distribution="uniform",
        params={"low": low, "high": high},
        dim=dim,
        precision_digits=precision_digits,
    )


def generate_normal(
    n: int, dim: int, mean: float, std: float, seed: int, precision_digits: int | None = None
) -> VectorBatch:
    rng = np.random.default_rng(seed)
    vecs = rng.normal(mean, std, size=(n, dim))
    if precision_digits is not None:
        vecs = np.round(vecs, precision_digits)
    return VectorBatch(
        vectors=vecs,
        distribution="normal",
        params={"mean": mean, "std": std},
        dim=dim,
        precision_digits=precision_digits,
    )


def generate_lognormal(
    n: int, dim: int, mean: float, sigma: float, seed: int, precision_digits: int | None = None
) -> VectorBatch:
    rng = np.random.default_rng(seed)
    vecs = rng.lognormal(mean, sigma, size=(n, dim))
    if precision_digits is not None:
        vecs = np.round(vecs, precision_digits)
    return VectorBatch(
        vectors=vecs,
        distribution="lognormal",
        params={"mean": mean, "sigma": sigma},
        dim=dim,
        precision_digits=precision_digits,
    )


def generate_integers(
    n: int, dim: int, low: int, high: int, seed: int
) -> VectorBatch:
    rng = np.random.default_rng(seed)
    vecs = rng.integers(low, high, size=(n, dim)).astype(np.float64)
    return VectorBatch(
        vectors=vecs,
        distribution="integers",
        params={"low": low, "high": high},
        dim=dim,
        precision_digits=0,
    )


# ---------------------------------------------------------------------------
# Standard test suite: the configurations we run for Phase 0
# ---------------------------------------------------------------------------

def get_phase0_configs(seed: int = 42) -> list[VectorBatch]:
    """Generate the full Phase 0 test suite.

    Returns ~40 batches covering the experimental design:
    - 3 distributions x 3 magnitude ranges x 4 precision levels x 4 dimensions
    - Plus integer and log-normal edge cases

    Each batch has 250 vectors (10K total across ~40 batches).
    """
    n_per_batch = 250
    dims = [1, 10, 100]
    batches = []

    # Uniform distributions at different magnitudes
    for dim in dims:
        for low, high, label in [(0, 1, "small"), (0, 100, "medium"), (0, 1e6, "large")]:
            for prec in [None, 2, 4, 8]:
                batches.append(generate_uniform(n_per_batch, dim, low, high, seed, prec))

    # Normal distributions at different spreads
    for dim in dims:
        for std in [0.01, 1.0, 100.0]:
            for prec in [None, 2, 4, 8]:
                batches.append(generate_normal(n_per_batch, dim, 0.0, std, seed, prec))

    # Log-normal (tests dynamic range)
    for dim in dims:
        for sigma in [0.5, 1.0, 2.0]:
            batches.append(generate_lognormal(n_per_batch, dim, 0.0, sigma, seed))

    # Integers at different ranges
    for dim in dims:
        for low, high in [(0, 10), (0, 1000), (0, 1_000_000)]:
            batches.append(generate_integers(n_per_batch, dim, low, high, seed))

    return batches
