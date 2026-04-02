# Infrastructure & Budget Plan

**Project:** AI Interplay Research
**Date:** April 2, 2026

---

## Compute Strategy

**Notebooks-first:** Prototype in Jupyter, extract reusable code into `src/` as it stabilizes. Notebooks remain the execution and analysis layer.

**LLM serving:** vLLM on cloud GPU (spot instances) for throughput-heavy runs. Local Mac (llama.cpp/MLX) for development and debugging. Final results should all run on the same backend for reproducibility — quantization differences between MLX and vLLM can cause divergence.

**Cloud provider:** TBD per phase. AWS g5.xlarge (A10G, ~$1/hr spot) for 7B/8B models. p4d.24xlarge (A100) for 70B if needed.

---

## Cost Estimates

| Phase | What runs | GPU needed | Est. hours | Est. cost |
|-------|----------|-----------|------------|-----------|
| 0: Tokenizer Round-Trip | Tokenizer encode/decode on 10K vectors | None (CPU) | Minutes | $0 |
| 1: Agent Relay + Precision | ~30M tokens inference (7B/8B models) | 1x A10G | 10-20 hrs | $20-40 |
| 2: Main Experiments (7B/8B) | Full suite across 3 domains, 10 seeds | 1x A10G | 40-80 hrs | $100-200 |
| 2: Main Experiments (+70B) | 70B model runs | 2x A100 | 20-40 hrs | $240-480 |
| 2: Traditional ML | XGBoost, ARIMA, Prophet, IsoForest, LSTMs | CPU / 1x GPU | <5 hrs | <$10 |
| 3: Ablations | ~50-100% of Phase 2 | Variable | Variable | $50-400 |
| 4: Scale sensitivity | 70B + optional Claude API | A100 + API | Variable | $250-500 |
| **Total** | | | | **$200-1,600** |

**Optional Claude API addition:** Claude Sonnet at ~$3/M input + $15/M output tokens. Full Phase 2 run: ~$50-150.

**Storage:** Negligible. Datasets < 1GB total. Results are CSVs.

---

## Phase-by-Phase Infrastructure

### Phase 0 — Local / Free Colab
- CPU only. Tokenizer libraries are small downloads (~5MB each).
- Dependencies: `transformers`, `tiktoken`, `numpy`, `scipy`, `matplotlib`

### Phase 1 — Local Mac (dev) + Cloud GPU (final runs)
- 7B/8B models via llama.cpp/MLX locally for iteration
- vLLM on g5.xlarge for reproducible final runs
- Additional deps: `vllm`, `torch`, `llama-cpp-python` or `mlx-lm`

### Phase 2 — Cloud GPU
- vLLM server on spot instance, experiments hit it via API
- Traditional ML runs on CPU (same instance or local)
- Additional deps: `statsmodels`, `prophet`, `xgboost`, `scikit-learn`, `pytorch-lightning`

### Phase 3-4 — Same as Phase 2, scaled as needed

---

## Project Structure

```
notebooks/
  phase0_tokenizer_roundtrip.ipynb
  phase1_agent_relay.ipynb
  phase1_precision_curves.ipynb
  phase2_time_series.ipynb
  phase2_anomaly_detection.ipynb
  phase2_tabular.ipynb
  analysis/
    statistical_tests.ipynb
    visualizations.ipynb

src/
  tokenizers/          # Tokenizer wrappers
  agents/              # Agent implementations (LLM, traditional, hybrid)
  benchmarks/          # Dataset loaders, preprocessing, checksums
  evaluation/          # Metrics, statistical tests, visualization

configs/               # Experiment configs (seeds, datasets, arms)
results/               # Raw CSVs, one per experiment run
docs/                  # Literature review, experimental design, this file
```

---

## Reproducibility Notes

- Pin all dependencies in `pyproject.toml`
- Docker container for final runs
- All LLM model versions pinned to specific HuggingFace revisions
- Final results must all run on the same inference backend
- Download scripts for datasets with SHA-256 checksums
