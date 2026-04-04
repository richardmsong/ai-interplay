"""Run Phase 0: Tokenizer Round-Trip Distortion experiment."""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

from src.tokenizers.wrappers import (
    get_tokenizer, get_all_tokenizer_names,
    SERIALIZATION_FORMATS, deserialize_numbers,
    Float16Baseline,
)
from src.evaluation.vectors import get_phase0_configs, VectorBatch
from src.evaluation.distortion import run_full_analysis, FullDistortionReport

RESULTS_DIR = Path("results/phase0")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load tokenizers
print("Loading tokenizers...")
tokenizers = {}
for name in get_all_tokenizer_names():
    try:
        tokenizers[name] = get_tokenizer(name)
        print(f"  OK  {name}")
    except Exception as e:
        print(f"  SKIP {name}: {type(e).__name__}")

# Generate test vectors
print("\nGenerating test vectors...")
batches = get_phase0_configs(seed=42)
total_vectors = sum(b.vectors.shape[0] for b in batches)
print(f"  {len(batches)} batches, {total_vectors} vectors total")

# Run experiments
def run_batch_experiment(batch, tokenizer_name, fmt_name):
    tok = tokenizers[tokenizer_name]
    serialize = SERIALIZATION_FORMATS[fmt_name]

    if tokenizer_name == "float16":
        assert isinstance(tok, Float16Baseline)
        recovered = tok.round_trip_numeric(batch.vectors)
        return run_full_analysis(
            original=batch.vectors, recovered=recovered,
            n_tokens_total=0, n_length_mismatches=0,
            tokenizer_name=tokenizer_name, serialization_format="native",
            distribution=batch.distribution, dim=batch.dim,
            precision_digits=batch.precision_digits, mantel_permutations=1000,
        )

    recovered_vecs = []
    total_tokens = 0
    length_mismatches = 0

    for vec in batch.vectors:
        text = serialize(vec)
        rt = tok.round_trip(text)
        total_tokens += rt.num_tokens
        recovered = deserialize_numbers(rt.decoded_text)

        if len(recovered) != len(vec):
            length_mismatches += 1
            if len(recovered) < len(vec):
                recovered = np.pad(recovered, (0, len(vec) - len(recovered)), constant_values=np.nan)
            else:
                recovered = recovered[:len(vec)]
        recovered_vecs.append(recovered)

    recovered_arr = np.nan_to_num(np.array(recovered_vecs), nan=0.0)

    return run_full_analysis(
        original=batch.vectors, recovered=recovered_arr,
        n_tokens_total=total_tokens, n_length_mismatches=length_mismatches,
        tokenizer_name=tokenizer_name, serialization_format=fmt_name,
        distribution=batch.distribution, dim=batch.dim,
        precision_digits=batch.precision_digits, mantel_permutations=1000,
    )


real_tokenizer_names = [n for n in tokenizers if n != "float16"]
fmt_names = list(SERIALIZATION_FORMATS.keys())
n_experiments = len(batches) * (
    len(real_tokenizer_names) * len(fmt_names) + (1 if "float16" in tokenizers else 0)
)

print(f"\nRunning {n_experiments} experiments...")
results = []
done = 0

for batch in batches:
    if "float16" in tokenizers:
        report = run_batch_experiment(batch, "float16", "csv")
        if report:
            results.append(report)
        done += 1

    for tok_name, fmt_name in product(real_tokenizer_names, fmt_names):
        report = run_batch_experiment(batch, tok_name, fmt_name)
        if report:
            results.append(report)
        done += 1
        if done % 100 == 0:
            print(f"  {done}/{n_experiments}...")

print(f"  {done}/{n_experiments} done.")

# Compile results
rows = []
for r in results:
    rows.append({
        "tokenizer": r.tokenizer_name,
        "format": r.serialization_format,
        "distribution": r.distribution,
        "dim": r.dim,
        "precision_digits": r.precision_digits,
        "n_vectors": r.n_vectors,
        "n_tokens_total": r.n_tokens_total,
        "length_mismatch_rate": r.length_mismatch_rate,
        "mean_abs_error": r.element_wise.mean_abs_error,
        "median_abs_error": r.element_wise.median_abs_error,
        "mean_rel_error": r.element_wise.mean_rel_error,
        "median_rel_error": r.element_wise.median_rel_error,
        "mean_sig_digits": r.element_wise.mean_sig_digits,
        "perfect_recovery_rate": r.element_wise.perfect_recovery_rate,
        "mean_normalized_l2": r.vector_wise.mean_normalized_l2,
        "mean_cosine_sim": r.vector_wise.mean_cosine_similarity,
        "mean_spearman_rho": r.vector_wise.mean_spearman_rho,
        "mantel_r": r.mantel.correlation if r.mantel else None,
        "mantel_p": r.mantel.p_value if r.mantel else None,
    })

df = pd.DataFrame(rows)
df.to_csv(RESULTS_DIR / "round_trip_results.csv", index=False)
print(f"\nResults: {len(df)} rows → {RESULTS_DIR / 'round_trip_results.csv'}")

# Summary
print("\n" + "=" * 60)
print("PHASE 0 SUMMARY: Tokenizer Round-Trip Distortion")
print("=" * 60 + "\n")

for tok_name in sorted(df["tokenizer"].unique()):
    subset = df[df["tokenizer"] == tok_name]
    pr = subset["perfect_recovery_rate"].mean()
    sd = subset["mean_sig_digits"].mean()
    mr = subset["mantel_r"].mean() if subset["mantel_r"].notna().any() else float("nan")
    print(f"{tok_name:15s}  perfect_recovery={pr:.3f}  sig_digits={sd:.1f}  mantel_r={mr:.4f}")

print()
print("- perfect_recovery: fraction of numbers recovered exactly (1.0 = lossless)")
print("- sig_digits: mean significant digits preserved (16 = float64 max)")
print("- mantel_r: metric structure preservation (1.0 = perfect, <1.0 = distortion)")

mantel_df = df[df["mantel_r"].notna()]
real_toks = mantel_df[~mantel_df["tokenizer"].isin(["float16", "python_str"])]
if len(real_toks) > 0:
    worst = real_toks["mantel_r"].min()
    mean = real_toks["mantel_r"].mean()
    print(f"\nWorst Mantel r across real tokenizers: {worst:.4f}")
    print(f"Mean Mantel r across real tokenizers:  {mean:.4f}")

    # Breakdown by format
    print("\nMantel r by serialization format:")
    for fmt in sorted(real_toks["format"].unique()):
        fmt_subset = real_toks[real_toks["format"] == fmt]
        print(f"  {fmt:20s}  mean={fmt_subset['mantel_r'].mean():.4f}  min={fmt_subset['mantel_r'].min():.4f}")

    # Breakdown by distribution
    print("\nMantel r by distribution:")
    for dist in sorted(real_toks["distribution"].unique()):
        dist_subset = real_toks[real_toks["distribution"] == dist]
        print(f"  {dist:20s}  mean={dist_subset['mantel_r'].mean():.4f}  min={dist_subset['mantel_r'].min():.4f}")

    # Breakdown by precision
    print("\nMantel r by precision:")
    for prec in sorted(real_toks["precision_digits"].unique(), key=lambda x: x if x is not None else 99):
        prec_subset = real_toks[real_toks["precision_digits"] == prec] if prec is not None else real_toks[real_toks["precision_digits"].isna()]
        label = str(prec) if prec is not None else "full"
        print(f"  {label:20s}  mean={prec_subset['mantel_r'].mean():.4f}  min={prec_subset['mantel_r'].min():.4f}")

    print()
    if mean > 0.999:
        print("CONCLUSION: Tokenizers preserve metric structure well.")
        print("  The translation tax may be minimal at the tokenization level.")
        print("  Proceed to Phase 1 to test whether the MODEL introduces distortion.")
    elif mean > 0.99:
        print("CONCLUSION: Small but measurable metric structure distortion.")
        print("  Proceed to Phase 1 — the distortion may compound across agent hops.")
    else:
        print("CONCLUSION: Significant metric structure distortion detected!")
        print("  This directly supports the thesis at the tokenization level.")
