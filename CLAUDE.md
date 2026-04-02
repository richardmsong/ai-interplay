# AI Interplay Research Project

## Overview
Research project investigating whether hybrid multi-agent systems (LLM + traditional AI agents) outperform pure LLM multi-agent systems on quantitative/numeric tasks. This is a research-first project — the goal is to produce defensible, conference-grade experimental results, not production-ready software.

## Hypothesis
- **H0 (Null):** Adding traditional AI components (statistical ML, classical DL) to an LLM-based multi-agent system provides no statistically significant improvement on numeric tasks.
- **H1 (Alternative):** Hybrid LLM + traditional AI multi-agent systems significantly outperform pure LLM multi-agent systems on tasks involving numeric signal interpretation.

## Experimental Domains
1. **Time-series forecasting** — Predicting future values from historical numeric sequences
2. **Anomaly detection** — Identifying outliers in numeric data streams
3. **Tabular classification/regression** — Structured data prediction tasks

## Architecture
- **Agent orchestration pattern** — Multiple agents coordinating, where some are LLM-powered and some are traditional ML models
- Compare: Pure LLM multi-agent vs. Hybrid (LLM + traditional) multi-agent

## Technical Constraints
- **LLMs:** Open-source primary (Llama, Mistral) for reproducibility. Design to be model-agnostic so Claude can also be tested.
- **Traditional AI:** Both statistical ML (XGBoost, ARIMA, Prophet, Random Forest) and classical DL (LSTMs, task-specific transformers)
- **Compute:** Cloud budget available
- **Rigor:** Conference-grade — proper train/val/test splits, statistical significance tests, ablation studies, multiple seeds
- **Data:** Open-source datasets only (Monash, ETT, OpenML, NAB, ADBench)

## Project Structure
```
docs/              # Literature review, experiment design docs
experiments/       # Experiment code, configs, and results
  time_series/
  anomaly_detection/
  tabular/
benchmarks/        # Benchmark dataset loaders and preprocessing
agents/            # Agent implementations (LLM and traditional)
evaluation/        # Metrics, statistical tests, visualization
```

## Principles
- **Reproducibility first** — All experiments must be fully reproducible with fixed seeds, pinned dependencies, open data
- **Fair comparison** — Compute-matched, data-matched comparisons. No comparing zero-shot LLM vs. fully-tuned baselines without noting it
- **Unbiased design** — Pre-register hypotheses and evaluation criteria before running experiments
- **Academic rigor** — Follow PRISMA-like methodology for benchmark selection. Use proper statistical tests (Wilcoxon signed-rank, bootstrap confidence intervals)
- **Push back critically** — Challenge assumptions, play devil's advocate, defend methodology choices
