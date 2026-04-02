# Experimental Design: The Translation Tax

**Project:** AI Interplay Research
**Date:** April 2, 2026
**Status:** Pre-registration document — deviations from this protocol must be documented

---

## 1. Thesis

**Language representations destroy the metric structure of numeric spaces.**

When numeric data passes through an LLM's tokenization and language processing, it undergoes a "translation tax": information is lost non-uniformly and unpredictably. Two numbers close in value (3.141 vs 3.142) may be far apart in token space, while numbers far in value may share token prefixes. Traditional AI models operate in representations where numeric distance is meaningful — vectors, distributions, embeddings — preserving the metric structure that LLMs destroy.

**Hypothesis (H1):** Hybrid multi-agent systems (LLM agents for language tasks + traditional AI agents for numeric tasks) significantly outperform pure LLM multi-agent systems on quantitative tasks, because they avoid routing numeric signals through language representations.

**Null Hypothesis (H0):** Adding traditional AI components to an LLM-based multi-agent system provides no statistically significant improvement on numeric tasks.

---

## 2. Execution Order

| Phase | Experiment | Purpose | Cost |
|-------|-----------|---------|------|
| 0 | 4A: Tokenizer Round-Trip | Validate premise — is tokenization lossy for numbers? | Minimal (no LLM inference) |
| 1 | 4B: Agent Relay | Does info degrade across LLM agent boundaries? | Moderate (LLM inference) |
| 1 | 4C: Precision Curves | At what precision does language representation fail? | Moderate |
| 2 | 1: Time-Series Forecasting | Main domain test | High |
| 2 | 2: Anomaly Detection | Main domain test | High |
| 2 | 3: Tabular Prediction | Main domain test | High |
| 3 | Ablations | Isolate causal mechanisms | Variable |
| 4 | Scale Sensitivity | Does model scale close the gap? | High |

**Critical gate:** If Phase 0 shows tokenizers perfectly preserve metric structure AND Phase 1 shows no communication degradation, the premise is weakened. Report honestly and reframe around model reasoning limitations rather than representation limitations.

---

## 3. Phase 0-1: Translation Tax Experiments

### 3.1 Experiment 4A — Tokenizer Round-Trip Distortion

**Question:** Does the act of tokenizing and detokenizing numbers destroy the metric structure of numeric spaces?

**Method:**
1. Generate 10,000 numeric vectors of dimension d ∈ {1, 10, 100, 1000} from controlled distributions:
   - Uniform on [0, 1], [0, 100], [0, 1e6] — tests magnitude sensitivity
   - Normal(0, σ) for σ ∈ {0.01, 1, 100} — tests spread sensitivity
   - Log-normal — tests dynamic range
   - Integers, floats with 2/4/8/16 decimal places — tests precision directly

2. For each vector, perform tokenizer round-trip:
   - Serialize to string (multiple formats: CSV, JSON array, natural language, scientific notation, digit-by-digit)
   - Tokenize with each tokenizer: Llama-3 (BPE), Mistral (SentencePiece), GPT-4o (cl100k_base), Chronos (quantization)
   - Detokenize back to string
   - Parse back to float

3. Measure distortion at three levels:

   **Element-wise:**
   - Absolute error |x - x̂|
   - Relative error |x - x̂| / |x|
   - Significant digits preserved

   **Vector-wise:**
   - Normalized L2 distance: ‖x - x̂‖₂ / ‖x‖₂
   - Cosine similarity
   - Spearman rank correlation (does ordering survive?)

   **Metric structure (the key test):**
   - For N=1000 random pairs from the same batch, compute pairwise Euclidean distance in original space → matrix D_orig
   - Compute pairwise distance in post-round-trip space → matrix D_rt
   - **Mantel test:** Pearson correlation between D_orig and D_rt, with permutation-based significance (10,000 permutations)
   - A correlation of 1.0 means metric structure is perfectly preserved. Below 1.0 = quantified destruction.

4. Controls:
   - float32 → float16 → float32 (known quantization, calibrates what "acceptable" loss looks like)
   - float64 → str() → float64 in Python (isolates tokenization loss from string serialization loss)

**Analysis:** Plot Mantel correlation as a function of (magnitude, precision, serialization format, tokenizer). Identify the regimes where metric structure breaks down.

**What a result means:**
- Mantel r ≈ 1.0 for all conditions → tokenization is not the bottleneck. Proceed to 4B/4C to test whether the model itself is the issue.
- Mantel r < 1.0, especially at high precision or large magnitude → direct evidence for the thesis. Quantify how much structure is lost and under what conditions.

---

### 3.2 Experiment 4B — Agent-to-Agent Communication Fidelity

**Question:** When numeric results pass between agents in a multi-agent system, does the communication channel (language vs. native) degrade the signal?

**Method:**
1. Design a relay task:
   - Ground truth: a 50-dimensional float vector from a known distribution
   - Agent Alpha "observes" the vector and communicates it to Agent Beta
   - Agent Beta uses the received vector to perform a downstream binary classification (is this vector from distribution A or distribution B?)

2. Four communication channels:

   | Channel | Alpha | Medium | Beta |
   |---------|-------|--------|------|
   | Natural language | LLM describes values in prose | Text string | LLM extracts values |
   | Structured JSON | LLM outputs JSON | JSON string | LLM parses JSON |
   | Code-mediated | LLM writes Python outputting numpy array | numpy array | Traditional model consumes array |
   | Pure native | Traditional model outputs array | numpy array | Traditional model consumes array |

3. Chain multiple hops: 1, 2, 3, 5 relay agents in sequence. At each hop, the receiving agent must re-transmit to the next.

4. Measure:
   - Reconstruction error: MSE between ground-truth vector and what the final agent receives
   - Downstream accuracy: classification accuracy of the final agent
   - Error growth rate: fit error = a · hops^b. If b > 1 → superlinear (compounding). If b ≈ 1 → linear. If b ≈ 0 → no degradation.

5. Test with multiple LLMs: Llama-3-8B, Llama-3-70B, Mistral-7B to see if scale mitigates.

**What a result means:**
- Superlinear error growth in language channels, near-zero in native → multi-agent LLM systems have a systemic architectural weakness for numeric communication that gets worse with more agents.
- All channels degrade similarly → the channel isn't the issue; something else is.

---

### 3.3 Experiment 4C — Precision-Dependent Degradation Curve

**Question:** Holding task difficulty constant, does LLM numeric accuracy degrade as precision requirements increase?

**Method:**
1. Generate data from known functions: y = x², y = sin(x), y = 2x + 3 (trivially learnable by all methods)
2. Provide identical training examples to all methods (20 examples, enough for any model to learn perfectly)
3. Evaluate at precision levels: integer, 1 decimal, 2 decimals, 4 decimals, 8 decimals

4. Methods compared:
   - LLM (few-shot, examples shown at target precision)
   - Linear regression
   - XGBoost
   - 2-layer MLP

5. Metric: Normalized MAE = MAE / precision_granularity. A random guess within the precision bucket gives Normalized MAE ≈ 1.0. A perfect predictor gives 0.0.

6. Plot degradation curves: x-axis = required precision (log scale), y-axis = Normalized MAE.

**What a result means:**
- LLM curve rises steeply after some precision threshold while traditional models stay flat → the representation can't carry precise numbers, even when the task is trivially learnable. This is the visual centerpiece of the thesis.
- All methods degrade similarly → precision limitations are shared across representations, thesis weakened.
- LLM stays flat too → LLMs handle precision fine (at least for simple functions), thesis needs refinement.

---

## 4. Phase 2: Main Domain Experiments

### 4.1 Time-Series Forecasting

**Datasets:**
- ETTh1, ETTh2, ETTm1, ETTm2 (Zhou et al., 2021) — de facto standard. Horizons: {96, 192, 336, 720}
- Monash Archive (6 datasets): `electricity_hourly`, `traffic_hourly`, `weather`, `tourism_monthly`, `hospital`, `solar_10_minutes`
- M5 competition — retail sales, count data stress test

**Arms:**

| Arm | Description | What it tests |
|-----|-------------|---------------|
| A1: Pure-LLM MAS | 3 LLM agents (Planner, Forecaster, Evaluator), all communicate in natural language. History serialized as text. | Baseline: can LLM agents handle forecasting end-to-end? |
| A2: Hybrid MAS | LLM Planner selects strategy → traditional Forecaster (ARIMA/Prophet/LSTM via bandit) → LLM Interpreter explains results. Numeric data stays native. | Core test: does keeping numbers in numeric representation help? |
| A3: Pure-Traditional | AutoML: auto-ARIMA, ETS, Prophet, LSTM, N-BEATS. Best selected on validation. No LLM. | Upper bound: what does pure numeric AI achieve? |
| A4: Hybrid-DL | Same as A2 but Forecaster is classical DL (LSTM/TFT/PatchTST). | Separates statistical ML from neural within "traditional." |

**Metrics:**
- Primary: MSE, MAE, MASE (scale-free, enables cross-dataset comparison)
- Secondary: CRPS (probabilistic), 90% prediction interval coverage
- Efficiency: wall-clock time, token count, USD cost

**Meaningful result:**
- Statistically significant difference (p < 0.05, Wilcoxon signed-rank, Holm-corrected) between A1 and A2, with Cliff's delta reported
- If A2 ≈ A3 → the hybrid architecture matches pure-traditional while gaining interpretability
- If A1 > A2 on any subset → characterize what data properties favor pure-LLM

### 4.2 Anomaly Detection

**Datasets:**
- NAB: 58 real-world streaming time series (standard NAB scoring)
- ADBench: 15 datasets (10 classical/tabular + 5 time-series)
- Synthetic injection set: 3 clean Monash series with injected anomalies of 4 types (point, contextual, collective, trend-change) at controlled magnitudes

**Arms:**

| Arm | Description | What it tests |
|-----|-------------|---------------|
| B1: Pure-LLM MAS | LLM Analyst describes statistics → LLM Detector classifies → LLM Aggregator resolves. All text. | Can LLM agents detect anomalies from text-described statistics? |
| B2: Hybrid MAS | LLM Analyst provides context → traditional Detector (IsoForest/LOF/LSTM-AE via meta-learner) produces scores → LLM Explainer interprets. Scores stay native. | Core test: traditional detection + LLM interpretation |
| B3: Pure-Traditional | Ensemble: IsoForest + LSTM-AE + Matrix Profile. Threshold via EVT. | Upper bound without LLM involvement |

**Metrics:**
- Primary: F1 (point-adjusted), NAB score, AUROC, AUPRC
- Secondary: detection latency, FPR at fixed recall (0.9)
- Qualitative: interpretability of explanations (B1 and B2 only, 2 judges, 1-5 Likert)

**Key sub-analysis:** On synthetic injection set, measure detection rate as a function of anomaly magnitude. Prediction: LLM detection degrades non-linearly as anomaly magnitude approaches tokenization granularity.

### 4.3 Tabular Classification/Regression

**Datasets:**
- 20 from OpenML-CC18 (classification), stratified by numeric feature density
- 8 from OpenML-CTR23 (regression)
- Total: 28 datasets

**Arms:**

| Arm | Description | What it tests |
|-----|-------------|---------------|
| C1: Pure-LLM MAS | LLM Analyst + LLM Feature Engineer + LLM Predictor (in-context). Features serialized as text. | Can LLMs do tabular prediction from text? |
| C2: Hybrid MAS | LLM Analyst + XGBoost Predictor (Bayesian-tuned) + LLM Interpreter. Data stays native. | Core test: LLM reasoning + traditional prediction |
| C3: Hybrid-FE | LLM generates candidate features (OCTree-style) → XGBoost on augmented features. | Does LLM add value as feature engineer? |
| C4: Pure-Traditional | AutoGluon/auto-sklearn. Full automated pipeline. | Upper bound |

**Metrics:**
- Classification: balanced accuracy, F1-macro, AUROC, log-loss
- Regression: RMSE, MAE, R²
- Efficiency: training time, inference time, cost

**Key comparisons:**
- C1 vs C2 → core hypothesis
- C2 vs C3 → does LLM feature engineering help beyond traditional prediction?
- C3 vs C4 → does LLM feature engineering beat automated feature engineering?

---

## 5. Ablation Studies (Phase 3)

| Ablation | What we remove/change | What it isolates |
|----------|----------------------|-----------------|
| Remove LLM Planner | Replace Planner with fixed model selection or meta-learning router | Is LLM routing valuable, or is it the numeric pathway that matters? |
| Remove LLM Interpreter | Output raw numbers, no explanation | Confirms accuracy comes from numeric compute, not interpretation |
| Vary communication channel | Pure-LLM with text → JSON → code execution | Isolates communication format from model capability |
| Scale LLM | 7B → 13B → 70B → frontier API | Does scale close the gap? (If yes, thesis needs nuance) |
| Vary precision requirements | Evaluate forecasting at rounded precision levels | Does hybrid advantage grow with precision? (Connects back to 4C) |

---

## 6. Statistical Methodology

### 6.1 Runs and Seeds
- 10 random seeds per condition per dataset: {42, 137, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}
- LLM arms: additionally 3 prompt orderings per seed. Per-seed observation = mean over orderings.
- Total: 10 paired observations per dataset for signed-rank test

### 6.2 Statistical Tests
- **Pairwise:** Wilcoxon signed-rank test (two-sided, paired by dataset)
- **Effect size:** Cliff's delta (non-parametric) + bootstrap 95% CIs (10K resamples)
- **Multi-arm:** Critical difference diagram with Nemenyi post-hoc test (Demšar, 2006)
- **Multiple comparisons:** Holm-Bonferroni correction within each experiment

### 6.3 Power
- N=28 (tabular): 0.80 power for medium effects (r=0.3)
- N=14 (time-series): adequate for large effects (r=0.5) only. Sensitivity analysis: pool across horizons (4×10=40) noting non-independence.
- If budget allows, extend Monash from 6 to 12 datasets.

### 6.4 Reporting
- All tables: mean, SD, median, IQR across seeds
- All tests: statistic, raw p-value, corrected p-value, effect size, 95% CI
- Complete per-dataset results in appendix (no cherry-picking)
- Raw results published as CSV alongside the paper

---

## 7. Confound Mitigations

| Confound | Risk | Mitigation |
|----------|------|-----------|
| **Model capability vs. representation** | Hybrid wins because XGBoost is better, not because representation is better | Translation Tax experiments (4A-4C) isolate representation. Ablation 3 (code exec) tests if native compute in pure-LLM closes gap. |
| **Compute budget unfairness** | Comparing $100 of LLM inference vs. $0.01 of XGBoost training | Report cost for all arms. Present Pareto frontier (accuracy vs. cost). Optional LoRA fine-tune arm for one experiment. |
| **Hyperparameter tuning fairness** | Traditional gets 50 Optuna trials, LLM gets one prompt | 5 prompt templates for LLM (select best on validation). Document tuning budget explicitly. |
| **Data leakage in LLMs** | LLMs may have seen benchmark data in pretraining | Contamination check (prompt for dataset values). Include synthetic datasets. Report with/without potentially contaminated sets. |
| **Prompt engineering quality** | Bad prompts artificially handicap LLM arms | 5 prompt variants per arm, report best. Release all prompts. Use established patterns (CoT, few-shot). |
| **Serialization format** | Bad number formatting handicaps LLM | Use best format from Experiment 4A. Report sensitivity. |
| **Context window limitation** | LLM can't fit all data, traditional can | Document truncation. Test multiple context budgets. Note as legitimate architectural difference. |

---

## 8. Reproducibility Protocol

- **Code:** Public GitHub repository (github.com/richardmsong/ai-interplay)
- **Dependencies:** `pyproject.toml` with pinned versions. Docker container provided.
- **Data:** All open-source. Download scripts with SHA-256 checksums.
- **Seeds:** All listed explicitly (Section 6.1)
- **LLM versions:** Pin exact model versions (e.g., `meta-llama/Llama-3.1-8B-Instruct` at specific revision)
- **Prompts:** All versioned in repository
- **Pre-registration:** This document, committed before experiments run. Deviations noted in final paper.
- **Results:** Raw CSVs (every run, every seed, every dataset) published with the paper
