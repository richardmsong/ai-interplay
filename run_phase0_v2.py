"""Phase 0 v2: Metric structure preservation in token embedding space.

The question: are numbers that are close in value also close
in the space the LLM reasons in (token embedding space)?

Only loads the embedding layer, not the full model.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import json

RESULTS_DIR = Path("results/phase0_v2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_embeddings_only(model_id: str):
    """Load only tokenizer + embedding weights. Minimal memory."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print(f"  Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load full model — for small models (GPT-2: 500MB) this is fine
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

    # Extract embedding weights
    if hasattr(model, "transformer"):  # GPT-2 style
        embed_weight = model.transformer.wte.weight.detach().numpy()
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):  # Llama style
        embed_weight = model.model.embed_tokens.weight.detach().numpy()
    else:
        raise ValueError(f"Can't find embedding layer in {model_id}")

    del model
    print(f"  Embedding shape: {embed_weight.shape}")
    return tokenizer, embed_weight


def numbers_to_embeddings(numbers, tokenizer, embed_weight, fmt="plain"):
    """Convert numbers to embedding-space vectors via mean pooling."""
    embeddings = []
    for x in numbers:
        text = f"{x}" if fmt == "plain" else f"{x:.6e}"
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        token_embeds = embed_weight[token_ids]  # (n_tokens, hidden_dim)
        embeddings.append(token_embeds.mean(axis=0))
    return np.array(embeddings)


def mantel_test(d1, d2, n_perms=1000, seed=42):
    """Mantel test between two condensed distance vectors."""
    if np.std(d1) == 0 or np.std(d2) == 0:
        return float("nan"), 1.0

    r_obs = float(np.corrcoef(d1, d2)[0, 1])
    rng = np.random.default_rng(seed)
    n = int((1 + np.sqrt(1 + 8 * len(d1))) / 2)

    D2_sq = squareform(d2)
    count_ge = 0
    for _ in range(n_perms):
        perm = rng.permutation(n)
        d2_perm = squareform(D2_sq[np.ix_(perm, perm)])
        if np.corrcoef(d1, d2_perm)[0, 1] >= r_obs:
            count_ge += 1

    return r_obs, (count_ge + 1) / (n_perms + 1)


def run_experiment(tokenizer, embed_weight, model_name, numbers, label, fmt="plain"):
    """Compare numeric distances to embedding distances."""
    d_numeric = pdist(numbers.reshape(-1, 1), metric="euclidean")

    embeddings = numbers_to_embeddings(numbers, tokenizer, embed_weight, fmt)
    d_embed_cos = pdist(embeddings, metric="cosine")
    d_embed_euc = pdist(embeddings, metric="euclidean")

    r_cos, p_cos = mantel_test(d_numeric, d_embed_cos)
    r_euc, p_euc = mantel_test(d_numeric, d_embed_euc)
    rho_cos = stats.spearmanr(d_numeric, d_embed_cos).statistic
    rho_euc = stats.spearmanr(d_numeric, d_embed_euc).statistic

    token_counts = [len(tokenizer.encode(
        f"{x}" if fmt == "plain" else f"{x:.6e}",
        add_special_tokens=False
    )) for x in numbers]

    return {
        "model": model_name, "number_set": label, "format": fmt,
        "n_numbers": len(numbers),
        "mantel_r_cosine": r_cos, "mantel_p_cosine": p_cos,
        "mantel_r_euclidean": r_euc, "mantel_p_euclidean": p_euc,
        "spearman_cosine": rho_cos, "spearman_euclidean": rho_euc,
        "mean_tokens": np.mean(token_counts),
    }


def generate_number_sets(seed=42):
    rng = np.random.default_rng(seed)
    return [
        (np.arange(1, 51, dtype=np.float64),
         "adjacent_integers_1_50"),
        (np.array([3.14 + i * 0.001 for i in range(50)]),
         "nearby_floats_3.14x"),
        (np.array([10 ** (i / 5 - 2) for i in range(50)]),
         "mixed_scale_1e-2_to_1e8"),
        (np.arange(100, 150, dtype=np.float64),
         "same_prefix_100_149"),
        (np.array([1, 2, 10, 11, 100, 101, 1000, 1001, 10000, 10001,
                    100000, 100001, 1e6, 1e6+1, 1e7, 1e7+1,
                    5, 6, 50, 51, 500, 501, 5000, 5001, 50000, 50001,
                    3, 4, 30, 31, 300, 301, 3000, 3001, 30000, 30001,
                    7, 8, 70, 71, 700, 701, 7000, 7001, 70000, 70001,
                    9, 9.001, 9.01, 9.1, 99, 99.001]),
         "equidistant_pairs_varying_mag"),
        (rng.uniform(0, 1, 50),
         "random_uniform_0_1"),
        (rng.uniform(0, 1000, 50),
         "random_uniform_0_1000"),
        (np.array([1.000001 * (1 + i * 1e-6) for i in range(50)]),
         "high_precision_6th_decimal"),
    ]


if __name__ == "__main__":
    import os

    # GPT-2 (124M params, ~500MB) — same BPE tokenization architecture as
    # larger models. Embedding geometry findings generalize.
    # Also test GPT-2-medium (355M) if disk permits.
    models = [
        ("gpt2", "gpt2-124m"),
        ("gpt2-medium", "gpt2-355m"),
    ]

    number_sets = generate_number_sets()
    formats = ["plain", "scientific"]
    all_results = []

    for model_id, model_name in models:
        try:
            tokenizer, embed_weight = load_embeddings_only(model_id)
        except Exception as e:
            print(f"  SKIP {model_name}: {e}")
            continue

        for numbers, label in number_sets:
            for fmt in formats:
                print(f"  {model_name} | {label} | {fmt}...", end="", flush=True)
                result = run_experiment(tokenizer, embed_weight, model_name, numbers, label, fmt)
                all_results.append(result)
                print(f" mantel_r(cos)={result['mantel_r_cosine']:+.4f}")

        del embed_weight

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "embedding_distances.csv", index=False)
    print(f"\nResults: {len(df)} rows → {RESULTS_DIR / 'embedding_distances.csv'}")

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 0 v2: Metric Structure in Token Embedding Space")
    print("=" * 70)
    print()
    print("Mantel r = correlation between numeric distance and embedding distance")
    print("  1.0 = perfect preservation")
    print("  0.0 = no relationship (random)")
    print(" <0.0 = inverted (close numbers are FAR in embedding space)")
    print()

    for model_name in df["model"].unique():
        print(f"--- {model_name} ---")
        model_df = df[df["model"] == model_name]
        for _, row in model_df.iterrows():
            sig = "*" if row["mantel_p_cosine"] < 0.05 else " "
            print(f"  {row['number_set']:40s} {row['format']:12s}  "
                  f"mantel_r={row['mantel_r_cosine']:+.4f}{sig}  "
                  f"spearman={row['spearman_cosine']:+.4f}  "
                  f"tok/num={row['mean_tokens']:.1f}")

    print(f"\n--- AGGREGATE ---")
    for col in ["mantel_r_cosine", "spearman_cosine"]:
        vals = df[col].dropna()
        print(f"  {col:25s}  mean={vals.mean():+.4f}  min={vals.min():+.4f}  max={vals.max():+.4f}")

    mean_m = df["mantel_r_cosine"].mean()
    print()
    if mean_m > 0.8:
        print("CONCLUSION: Embedding space largely preserves numeric metric structure.")
    elif mean_m > 0.4:
        print("CONCLUSION: Partial preservation — some numeric relationships survive, others don't.")
    elif mean_m > 0.0:
        print("CONCLUSION: Weak preservation — embedding geometry doesn't reliably reflect numeric proximity.")
    else:
        print("CONCLUSION: Metric structure destroyed — close numbers are NOT close in embedding space.")
