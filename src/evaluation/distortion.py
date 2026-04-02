"""Distortion measurement suite for tokenizer round-trip experiments.

Three levels of measurement:
1. Element-wise: absolute error, relative error, significant digits preserved
2. Vector-wise: normalized L2, cosine similarity, rank correlation
3. Metric structure: Mantel test on pairwise distance matrices
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform


# ---------------------------------------------------------------------------
# 1. Element-wise metrics
# ---------------------------------------------------------------------------

@dataclass
class ElementWiseMetrics:
    absolute_error: np.ndarray       # |x - x_hat|
    relative_error: np.ndarray       # |x - x_hat| / |x|, NaN where x=0
    significant_digits: np.ndarray   # floor(-log10(relative_error)), capped

    # Aggregates
    mean_abs_error: float
    median_abs_error: float
    mean_rel_error: float
    median_rel_error: float
    mean_sig_digits: float
    perfect_recovery_rate: float     # fraction where x == x_hat exactly


def compute_element_wise(original: np.ndarray, recovered: np.ndarray) -> ElementWiseMetrics:
    """Compare original and recovered arrays element-by-element."""
    abs_err = np.abs(original - recovered)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.where(original != 0, abs_err / np.abs(original), np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        sig_digits = np.where(
            rel_err > 0,
            np.clip(np.floor(-np.log10(rel_err)), 0, 16),
            16.0,  # perfect recovery = 16 significant digits (float64 limit)
        )
    sig_digits = np.where(np.isnan(rel_err), np.nan, sig_digits)

    valid_rel = rel_err[~np.isnan(rel_err)]

    return ElementWiseMetrics(
        absolute_error=abs_err,
        relative_error=rel_err,
        significant_digits=sig_digits,
        mean_abs_error=float(np.mean(abs_err)),
        median_abs_error=float(np.median(abs_err)),
        mean_rel_error=float(np.mean(valid_rel)) if len(valid_rel) > 0 else float("nan"),
        median_rel_error=float(np.median(valid_rel)) if len(valid_rel) > 0 else float("nan"),
        mean_sig_digits=float(np.nanmean(sig_digits)),
        perfect_recovery_rate=float(np.mean(abs_err == 0)),
    )


# ---------------------------------------------------------------------------
# 2. Vector-wise metrics
# ---------------------------------------------------------------------------

@dataclass
class VectorWiseMetrics:
    normalized_l2: np.ndarray         # per-vector: ||x - x_hat|| / ||x||
    cosine_similarity: np.ndarray     # per-vector
    spearman_rho: np.ndarray          # per-vector rank correlation

    mean_normalized_l2: float
    mean_cosine_similarity: float
    mean_spearman_rho: float


def compute_vector_wise(original: np.ndarray, recovered: np.ndarray) -> VectorWiseMetrics:
    """Compare original and recovered as vectors (row-wise).

    Both inputs should be 2D: (n_vectors, dim). For dim=1, cosine similarity
    and rank correlation are not meaningful — we skip them.
    """
    n, d = original.shape

    # Normalized L2
    diff_norms = np.linalg.norm(original - recovered, axis=1)
    orig_norms = np.linalg.norm(original, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_l2 = np.where(orig_norms > 0, diff_norms / orig_norms, np.nan)

    # Cosine similarity (only meaningful for dim > 1)
    if d > 1:
        dot = np.sum(original * recovered, axis=1)
        rec_norms = np.linalg.norm(recovered, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            cos_sim = np.where(
                (orig_norms > 0) & (rec_norms > 0),
                dot / (orig_norms * rec_norms),
                np.nan,
            )
    else:
        cos_sim = np.full(n, np.nan)

    # Spearman rank correlation (only meaningful for dim > 2)
    if d > 2:
        spearman = np.array([
            stats.spearmanr(original[i], recovered[i]).statistic
            if not np.allclose(original[i], original[i][0])  # skip constant vectors
            else np.nan
            for i in range(n)
        ])
    else:
        spearman = np.full(n, np.nan)

    return VectorWiseMetrics(
        normalized_l2=norm_l2,
        cosine_similarity=cos_sim,
        spearman_rho=spearman,
        mean_normalized_l2=float(np.nanmean(norm_l2)),
        mean_cosine_similarity=float(np.nanmean(cos_sim)),
        mean_spearman_rho=float(np.nanmean(spearman)),
    )


# ---------------------------------------------------------------------------
# 3. Metric structure: Mantel test
# ---------------------------------------------------------------------------

@dataclass
class MantelResult:
    correlation: float       # Pearson r between distance matrices
    p_value: float           # permutation-based p-value
    n_pairs: int             # number of pairwise distances compared
    n_permutations: int


def mantel_test(
    original: np.ndarray,
    recovered: np.ndarray,
    n_permutations: int = 10_000,
    max_samples: int = 1000,
    seed: int = 42,
) -> MantelResult:
    """Mantel test: correlation between pairwise distance matrices.

    This is the key test for metric structure preservation. If the tokenizer
    preserves the metric structure of numeric space, then pairs of vectors that
    are close in original space should also be close after round-tripping.

    Args:
        original: (n, d) array of original vectors
        recovered: (n, d) array of recovered vectors
        n_permutations: number of permutations for significance test
        max_samples: subsample to this many vectors (Mantel is O(n^2))
        seed: random seed for subsampling and permutations
    """
    rng = np.random.default_rng(seed)
    n = original.shape[0]

    # Subsample if needed (Mantel is O(n^2) in memory and compute)
    if n > max_samples:
        idx = rng.choice(n, max_samples, replace=False)
        original = original[idx]
        recovered = recovered[idx]
        n = max_samples

    # Compute pairwise distance vectors (condensed form)
    d_orig = pdist(original, metric="euclidean")
    d_rec = pdist(recovered, metric="euclidean")

    # Handle degenerate cases
    if np.std(d_orig) == 0 or np.std(d_rec) == 0:
        return MantelResult(
            correlation=float("nan"),
            p_value=1.0,
            n_pairs=len(d_orig),
            n_permutations=n_permutations,
        )

    # Observed correlation
    r_obs = float(np.corrcoef(d_orig, d_rec)[0, 1])

    # Permutation test: shuffle rows of one matrix, recompute correlation
    count_ge = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        d_perm = pdist(recovered[perm], metric="euclidean")
        r_perm = float(np.corrcoef(d_orig, d_perm)[0, 1])
        if r_perm >= r_obs:
            count_ge += 1

    p_value = (count_ge + 1) / (n_permutations + 1)  # +1 to include observed

    return MantelResult(
        correlation=r_obs,
        p_value=p_value,
        n_pairs=len(d_orig),
        n_permutations=n_permutations,
    )


# ---------------------------------------------------------------------------
# Convenience: run all measurements
# ---------------------------------------------------------------------------

@dataclass
class FullDistortionReport:
    element_wise: ElementWiseMetrics
    vector_wise: VectorWiseMetrics
    mantel: MantelResult | None  # None if dim=1 (distance matrix is trivial)

    tokenizer_name: str
    serialization_format: str
    distribution: str
    dim: int
    precision_digits: int | None
    n_vectors: int
    n_tokens_total: int          # total tokens used across all vectors
    length_mismatch_rate: float  # fraction of vectors where len(recovered) != len(original)


def run_full_analysis(
    original: np.ndarray,
    recovered: np.ndarray,
    n_tokens_total: int,
    n_length_mismatches: int,
    tokenizer_name: str,
    serialization_format: str,
    distribution: str,
    dim: int,
    precision_digits: int | None,
    mantel_permutations: int = 10_000,
    mantel_max_samples: int = 250,
    seed: int = 42,
) -> FullDistortionReport:
    """Run all three levels of distortion measurement."""

    ew = compute_element_wise(original.ravel(), recovered.ravel())
    vw = compute_vector_wise(original, recovered)

    # Mantel test only for dim > 1 (pairwise distances need multi-dim vectors)
    mantel = None
    if dim > 1 and original.shape[0] >= 10:
        mantel = mantel_test(
            original, recovered,
            n_permutations=mantel_permutations,
            max_samples=mantel_max_samples,
            seed=seed,
        )

    return FullDistortionReport(
        element_wise=ew,
        vector_wise=vw,
        mantel=mantel,
        tokenizer_name=tokenizer_name,
        serialization_format=serialization_format,
        distribution=distribution,
        dim=dim,
        precision_digits=precision_digits,
        n_vectors=original.shape[0],
        n_tokens_total=n_tokens_total,
        length_mismatch_rate=n_length_mismatches / original.shape[0],
    )
