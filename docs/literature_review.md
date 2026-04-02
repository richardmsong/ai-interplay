# Literature Review: Hybrid LLM + Traditional AI Multi-Agent Systems for Numeric Tasks

**Project:** AI Interplay Research  
**Date:** April 2, 2026  
**Status:** Living document — will be updated as research progresses

---

## 1. Introduction

The rapid adoption of Large Language Models (LLMs) has led to their application far beyond natural language — including time-series forecasting, anomaly detection, and tabular data prediction. A central question has emerged: **do LLMs add genuine value to numeric reasoning tasks, or are they an expensive intermediary that loses information by translating numeric signals into language?**

This review surveys the literature across four intersecting research areas:
1. LLMs applied to time-series and numeric tasks
2. Fundamental limitations of LLM numeric reasoning
3. Traditional AI baselines and their continued dominance on structured data
4. Multi-agent systems and neuro-symbolic architectures that combine both paradigms

Our goal is to establish the theoretical and empirical foundation for testing whether hybrid multi-agent systems (LLM agents + traditional AI agents) outperform pure LLM multi-agent systems on quantitative tasks.

---

## 2. LLMs for Time-Series Forecasting

### 2.1 Foundation Models for Time Series

The past two years have seen an explosion of work applying LLM architectures to time-series data. The approaches differ fundamentally in how they bridge the modality gap between language and numeric sequences.

**TimeGPT-1** (Garza & Mergenthaler-Canseco, 2023) was among the first foundation models specifically designed for time-series forecasting, trained on over 100 billion data points. It supports zero-shot inference, anomaly detection, and conformal prediction, with zero-shot performance matching or exceeding statistical, ML, and deep learning baselines across retail, electricity, finance, and IoT domains. However, it is available only as a proprietary API, limiting reproducibility ([arXiv:2310.03589](https://arxiv.org/abs/2310.03589)).

**Chronos** (Ansari et al., 2024) took a different approach: tokenizing time-series values via scaling and quantization into a fixed vocabulary, then training T5-family transformers (20M–710M parameters) using cross-entropy loss. Pretrained on public datasets augmented with synthetic Gaussian-process-generated data, Chronos achieves strong zero-shot probabilistic forecasting. The follow-up Chronos-Bolt (November 2024) achieved 5% lower error with up to 250x faster inference. Published at ICML 2024 ([arXiv:2403.07815](https://arxiv.org/abs/2403.07815)).

**Lag-Llama** (Rasul et al., 2024) applies the decoder-only LLaMA architecture to univariate probabilistic time-series forecasting using lagged values as covariates. Pretrained on diverse time-series corpora, it demonstrates strong zero-shot generalization to unseen datasets, outperforming many task-specific models without fine-tuning ([arXiv:2310.08278](https://arxiv.org/abs/2310.08278)).

**MOMENT** (Goswami et al., 2024) from CMU's Auton Lab addresses the absence of large public time-series repositories by introducing the "Time-Series Pile." The resulting pretrained models show strong performance in anomaly detection and classification, with all models and data released as open-source. Published at ICML 2024 ([arXiv:2402.03885](https://arxiv.org/abs/2402.03885)).

### 2.2 Reprogramming and Prompting Approaches

Rather than training time-series-native models, several papers attempt to repurpose existing language models.

**Time-LLM** (Jin et al., 2024) proposes a reprogramming framework that keeps the LLM backbone frozen and aligns time-series inputs with text prototypes before feeding them into the model. Combined with "Prompt-as-Prefix" (PaP) for enriching context, Time-LLM achieves strong forecasting performance without modifying pretrained weights. Published at ICLR 2024, cited over 1,000 times ([arXiv:2310.01728](https://arxiv.org/abs/2310.01728)).

**LLMs Are Zero-Shot Time Series Forecasters** (Gruver et al., 2023) showed that by encoding time-series as strings of numerical digits, GPT-3 and LLaMA-2 can zero-shot extrapolate time series at levels comparable to purpose-built models. The success is attributed to LLMs' ability to represent multimodal distributions combined with biases for simplicity and repetition. Published at NeurIPS 2023 ([arXiv:2310.07820](https://arxiv.org/abs/2310.07820)).

**PromptCast** (Xue & Salim, 2023) transforms numerical input/output into natural language prompts, framing forecasting as a sentence-to-sentence task. Introduces the PISA dataset and shows better zero-shot generalization than conventional numerical methods. Published in IEEE TKDE ([arXiv:2210.08964](https://arxiv.org/abs/2210.08964)).

**ChatTS** (Xie et al., 2024/2025) from ByteDance is a time-series multimodal LLM focused on *understanding and reasoning* about time series rather than point forecasting. Fine-tuned on Qwen2.5-14B-Instruct, it achieves 46–76% gains on categorical alignment and 80–113% on numerical alignment tasks versus GPT-4o. Uses synthetic data generation to align time-series modality with LLM representations. Accepted at VLDB 2025 ([arXiv:2412.03104](https://arxiv.org/abs/2412.03104)).

**One Fits All** (Zhou et al., 2023) demonstrates that a single frozen GPT-2 backbone with lightweight adaptation layers can handle forecasting, classification, imputation, and anomaly detection, suggesting pretrained language models encode generalizable temporal patterns. Published at NeurIPS 2023 ([arXiv:2302.11939](https://arxiv.org/abs/2302.11939)).

### 2.3 Critical Perspectives: Do LLMs Actually Help?

A growing body of work challenges whether the LLM component is actually responsible for observed performance gains.

**"Are Transformers Effective for Time Series Forecasting?"** (Zeng et al., 2023) is the landmark skeptical paper. It demonstrates that transformer self-attention is permutation-invariant and inevitably loses temporal information despite positional encoding. The authors propose DLinear — decomposition plus two single-layer linear networks — which outperforms all sophisticated transformer-based models (Autoformer, FEDformer, etc.) on nine benchmarks. Published as an Oral at AAAI 2023 ([arXiv:2205.13504](https://arxiv.org/abs/2205.13504)).

**"Are Language Models Actually Useful for Time Series Forecasting?"** (Tan et al., 2024) delivers the most direct challenge. The authors show that removing the LLM component from LLM-based forecasters, or replacing it with a basic attention layer, does not degrade — and often *improves* — forecasting performance. They propose PAttn, a simple patching + attention baseline that outperforms most LLM-based methods. The conclusion: performance gains attributed to LLMs actually originate from time-series encoding techniques (patching), not the language model. Published at NeurIPS 2024 ([arXiv:2406.16964](https://arxiv.org/abs/2406.16964)).

**2025 Benchmark Studies** present mixed evidence: GIFT-Eval favors time-series foundation models, OpenTS shows statistical models outperforming deep learning on univariate data, and FoundTS finds supervised baselines on par with foundation models. Unlike NLP's "BERT moment," time-series foundation models still often require full fine-tuning to be competitive ([arXiv:2410.11802](https://arxiv.org/abs/2410.11802)).

**"What Can Large Language Models Tell Us about Time Series Analysis?"** (Jin et al., 2024) is a critical position paper at ICML 2024 arguing that many LLM-for-time-series papers suffer from unfair comparisons (e.g., zero-shot LLM vs. non-tuned baselines). They propose a fair evaluation framework considering compute budget, data availability, and task specificity ([arXiv:2402.02713](https://arxiv.org/abs/2402.02713)).

**"An Evaluation of Standard Statistical Models and LLMs on Time Series Forecasting"** (Cao & Wang, 2024) finds that LLM-based approaches (specifically LLMTIME) significantly underperform traditional ARIMA on time series containing both periodic and trend components or complex frequency mixtures ([arXiv:2408.04867](https://arxiv.org/abs/2408.04867)).

> **Summary:** There is substantial evidence that while LLMs *can* be adapted for time-series tasks, the LLM component itself may contribute little beyond what simpler architectures achieve. The strongest results come from clever input encoding (patching, tokenization, reprogramming) rather than from the language model's pretrained knowledge. This supports our hypothesis that numeric-native models may be more appropriate for the numeric reasoning component of a multi-agent system.

---

## 3. Fundamental Limitations of LLM Numeric Reasoning

### 3.1 Arithmetic and Mathematical Reasoning

**GSM-Symbolic** (Mirzadeh et al., 2024) from Apple creates a symbolic benchmark that generates diverse question variants from templates. All tested LLMs show performance decline when only numerical values are changed, suggesting pattern matching rather than genuine reasoning. Adding a single irrelevant-but-plausible clause causes drops of up to 65%. Accepted at ICLR 2025 ([arXiv:2410.05229](https://arxiv.org/abs/2410.05229)).

**"Faith and Fate: Limits of Transformers on Compositionality"** (Dziri et al., 2023) from Allen AI investigates transformer failures on multi-digit multiplication, logic grid puzzles, and dynamic programming. Finds that transformers reduce compositional tasks to linearized subgraph matching without systematic problem-solving. Published at NeurIPS 2023 ([arXiv:2305.18654](https://arxiv.org/abs/2305.18654)).

**"Tokenization Counts"** (Singh & Strouse, 2024) demonstrates that tokenization strategy creates up to 20% accuracy differences in LLM arithmetic. GPT-3.5/4's multi-digit left-to-right tokenization creates stereotyped error patterns. Using commas to enforce right-to-left tokenization significantly improves performance, highlighting that fundamental architectural choices — not just model capability — drive arithmetic failures ([arXiv:2402.14903](https://arxiv.org/abs/2402.14903)).

### 3.2 Error Detection and Self-Correction

**"LLMs Cannot Find Reasoning Errors, but Can Correct Them Given the Error Location"** (Tyen et al., 2024) demonstrates that poor self-correction stems from inability to *locate* logical mistakes, not inability to fix them. Published in Findings of ACL 2024 ([arXiv:2311.08516](https://arxiv.org/abs/2311.08516)).

### 3.3 Scaling with Numerical Complexity

**"Mathematical Reasoning in LLMs: Assessing Errors across Wide Numerical Ranges"** (2025) documents up to 14 percentage-point increases in logical error rates as numerical magnitudes increase, with recurring calculation errors, counting errors, and formula confusion ([arXiv:2502.08680](https://arxiv.org/abs/2502.08680)).

**"Large Language Models and Mathematical Reasoning Failures"** (2025) argues that LLM mathematical reasoning combines probabilistic noisy reasoning with memorization, and that the more steps required, the more likely memorization interferes with reasoning ([arXiv:2502.11574](https://arxiv.org/abs/2502.11574)).

> **Summary:** LLMs have well-documented and fundamental limitations in numeric reasoning: tokenization artifacts, pattern matching instead of genuine computation, degradation with numerical magnitude, and inability to self-detect errors. These limitations are architectural, not just capability gaps to be closed with scale. This directly motivates using purpose-built numeric models for numeric sub-tasks rather than routing everything through an LLM.

---

## 4. Traditional AI Baselines: Still Dominant on Structured Data

### 4.1 Tabular Data

**"Why Do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data?"** (Grinsztajn et al., 2022) benchmarks 45 datasets and finds tree-based models (XGBoost, Random Forests) remain state-of-the-art on medium-sized tabular data. The gap stems from fundamental inductive bias differences: tabular data's heterogeneous features, small samples, and extreme values are poorly suited to neural network invariances. Published at NeurIPS 2022 ([arXiv:2207.08815](https://arxiv.org/abs/2207.08815)).

**"Tabular Data: Deep Learning is Not All You Need"** (Shwartz-Ziv & Armon, 2022) shows XGBoost outperforms deep learning across datasets — including datasets used in the papers proposing those deep models. However, ensembles of deep models and XGBoost perform better than either alone, suggesting hybrid approaches have merit. Published in Information Fusion ([arXiv:2106.03253](https://arxiv.org/abs/2106.03253)).

**"LLMs on Tabular Data: A Survey"** (Fang et al., 2024) confirms GBDT algorithms still outperform deep learning and LLM methods on most tabular datasets, with additional benefits of fast training and interpretability. LLMs show advantages primarily with very large datasets, categorical-dominant features, or few-shot scenarios ([arXiv:2402.17944](https://arxiv.org/abs/2402.17944)).

**UniPredict** (Wang et al., 2023) shows a single GPT-2 trained on 169 tabular datasets can serve as a universal classifier with 5.4–13.4% advantage over tree-based methods, but primarily in low-resource settings. The advantage diminishes with more training data ([arXiv:2310.03266](https://arxiv.org/abs/2310.03266)).

**TabPFN** (Hollmann et al., 2023) trains a transformer for in-context tabular classification, matching tuned GBDTs on datasets with ~1000 samples. Published at ICLR 2023 ([arXiv:2207.01848](https://arxiv.org/abs/2207.01848)).

### 4.2 Time-Series Forecasting

The M5 Competition (Makridakis et al., 2022) using hierarchical Walmart sales data (42,840 series) confirmed that gradient boosted trees (LightGBM) dominated the accuracy track. Published in International Journal of Forecasting ([doi:10.1016/j.ijforecast.2021.11.013](https://doi.org/10.1016/j.ijforecast.2021.11.013)).

### 4.3 Anomaly Detection

**"Anomaly Detection in Time Series: A Comprehensive Evaluation"** (Schmidl et al., 2022) evaluates 71 algorithms on 976 time series and establishes that no single method dominates — and simpler methods (k-NN, Isolation Forest variants) often outperform deep learning. Published in PVLDB ([doi:10.14778/3538598.3538602](https://doi.org/10.14778/3538598.3538602)).

> **Summary:** Tree-based models and classical statistical methods remain dominant on structured/tabular and time-series data respectively, particularly when reasonable training data is available. The pattern is consistent: LLMs show promise mainly in zero-shot/few-shot regimes, while traditional methods excel with in-domain data. This suggests a clear division of labor in hybrid systems.

---

## 5. Hybrid Approaches and Multi-Agent Systems

### 5.1 LLMs Enhancing Traditional Models

**OCTree** (Nam et al., 2024) uses LLM reasoning to identify effective feature generation rules guided by decision tree explanations, achieving 7.9% relative error reduction for XGBoost. This represents a genuinely hybrid approach: LLMs handle creative feature discovery while tree models handle prediction. Published at NeurIPS 2024 ([arXiv:2406.08527](https://arxiv.org/abs/2406.08527)).

### 5.2 Multi-Agent Frameworks

**AutoGen** (Wu et al., 2023) is an open-source framework where agents are customizable and conversable, operating in modes that combine LLMs, human inputs, and tools. Critically, agents need not all be LLM-powered — the framework supports heterogeneous agents including code executors and tool-using agents ([arXiv:2308.08155](https://arxiv.org/abs/2308.08155)).

**MetaGPT** (Hong et al., 2023) encodes Standardized Operating Procedures from human workflows into LLM multi-agent collaboration. By assigning specialized roles and requiring structured artifacts at each stage, it reduces cascading hallucinations. Published at ICLR 2024 ([arXiv:2308.00352](https://arxiv.org/abs/2308.00352)).

**CAMEL** (Li et al., 2023) proposes role-playing communicative agents using "inception prompting," foundational for studying cooperative behaviors in multi-agent LLM systems. Published at NeurIPS 2023 ([arXiv:2303.17760](https://arxiv.org/abs/2303.17760)).

**"Multi-Agent Collaboration Mechanisms: A Survey"** (Tran et al., 2025) provides a taxonomy of LLM-based multi-agent collaboration across actors, structures, strategies, and protocols. Notes the trend toward heterogeneous agent teams mixing LLM and non-LLM components ([arXiv:2501.06322](https://arxiv.org/abs/2501.06322)).

**X-MAS** (2025) demonstrates that combining heterogeneous agents consistently outperforms homogeneous single-model configurations. Assessed 28 LLMs across 5 domains with over 1.7 million evaluations ([arXiv:2505.16997](https://arxiv.org/abs/2505.16997)).

### 5.3 Neuro-Symbolic AI

**AlphaGeometry** (Trinh et al., 2024) combines a neural language model with a rule-based symbolic deduction engine to solve 25 of 30 IMO geometry problems (previous best AI: 10/30). Published in Nature ([doi:10.1038/s41586-023-06747-5](https://www.nature.com/articles/s41586-023-06747-5)).

**ToRA** (Gou et al., 2024) interweaves natural language reasoning with program-based tool use (computation libraries, symbolic solvers). ToRA-7B reaches 44.6% on MATH, surpassing the 10x-larger WizardMath-70B by 22% absolute. Published at ICLR 2024 ([arXiv:2309.17452](https://arxiv.org/abs/2309.17452)).

**"Neuro-Symbolic AI in 2024: A Systematic Review"** (Colelough & Regli, 2025) analyzes 167 papers following PRISMA methodology, documenting the growing integration of LLMs into symbolic frameworks ([arXiv:2501.05435](https://arxiv.org/abs/2501.05435)).

### 5.4 LLMs for Anomaly Detection

**LLMAD** (Wang et al., 2024) proposes an LLM-based agent for detecting anomalies in IoT sensor data. The LLM interprets statistical features and produces anomaly verdicts with explanations, showing comparable detection to Isolation Forest and LSTM-autoencoders with added interpretability ([arXiv:2405.14014](https://arxiv.org/abs/2405.14014)).

**"LLMs Are Zero-Shot Time Series Anomaly Detectors"** (Alnegheimish et al., 2024) finds LLMs can detect obvious point anomalies but struggle with subtle contextual anomalies and are highly sensitive to prompt design. Traditional methods outperform on standard benchmarks when trained on in-domain data ([arXiv:2405.14755](https://arxiv.org/abs/2405.14755)).

> **Summary:** Multi-agent frameworks already support heterogeneous agent compositions, and neuro-symbolic research demonstrates that hybrid neural + classical systems can dramatically exceed either component alone. The gap in the literature is rigorous, controlled comparison of pure LLM multi-agent systems vs. hybrid systems specifically on numeric tasks across multiple domains.

---

## 6. Benchmarks and Evaluation Infrastructure

### 6.1 Time-Series Forecasting
- **Monash Time Series Forecasting Archive** (Godahewa et al., 2021): 30 datasets across energy, finance, nature, web traffic. NeurIPS 2021 ([arXiv:2105.06643](https://arxiv.org/abs/2105.06643); [forecastingdata.org](https://forecastingdata.org/))
- **ETT Datasets** (Zhou et al., 2021): ETTh1, ETTh2, ETTm1, ETTm2 — de facto standard for long-sequence forecasting evaluation. AAAI 2021 ([arXiv:2012.07436](https://arxiv.org/abs/2012.07436))
- **M5 Competition** (Makridakis et al., 2022): 42,840 hierarchical Walmart sales series ([doi:10.1016/j.ijforecast.2021.11.013](https://doi.org/10.1016/j.ijforecast.2021.11.013))

### 6.2 Tabular Data
- **OpenML Benchmarking Suites** (Bischl et al., 2021): CC18 (72 classification datasets), CTR-23 (regression). NeurIPS 2021 ([openml.org](https://www.openml.org/))

### 6.3 Anomaly Detection
- **NAB** — Numenta Anomaly Benchmark (Lavin & Ahmad, 2015): 58 labeled real-world streaming time series. IEEE ICMLA ([arXiv:1510.03336](https://arxiv.org/abs/1510.03336); [github.com/numenta/NAB](https://github.com/numenta/NAB))
- **ADBench** (Han et al., 2022): 57 datasets, 30 algorithms. NeurIPS 2022 ([arXiv:2206.09426](https://arxiv.org/abs/2206.09426))
- **ODDS** (Rayana, 2016): Multi-dimensional outlier detection datasets ([odds.cs.stonybrook.edu](http://odds.cs.stonybrook.edu/))

### 6.4 Foundation Model Evaluation
- **TSFM-Bench** (2025): Comprehensive evaluation of time-series foundation models. KDD 2025 ([arXiv:2410.11802](https://arxiv.org/abs/2410.11802))
- **Foundation Models for Time Series: A Tutorial and Survey** (Fan et al., 2024): Taxonomy and evaluation framework. KDD 2024 ([arXiv:2403.14735](https://arxiv.org/abs/2403.14735))

---

## 7. Identified Gaps and Our Research Contribution

Based on this review, we identify the following gaps:

1. **No systematic comparison at the agent-system level.** Existing work compares individual models (LLM vs. XGBoost on dataset X). No work systematically compares *multi-agent orchestration patterns* where some agents are LLMs and others are traditional models, against systems where all agents are LLMs.

2. **No cross-domain evaluation of hybrid agent systems.** Papers evaluate within a single domain (forecasting OR tabular OR anomaly detection). We test across all three to see if hybrid advantages generalize.

3. **Fair evaluation is rare.** Most LLM-for-time-series papers compare zero-shot LLM against fully-tuned baselines, or fail to match compute budgets. We design compute-matched, data-matched comparisons.

4. **The "agent as traditional model" pattern is underexplored.** Multi-agent frameworks support non-LLM agents, but research focuses on all-LLM agent teams. The question of *when* to route to a traditional model agent vs. an LLM agent is largely unaddressed.

5. **Tokenization and representation losses are documented but not quantified in agent contexts.** We know LLMs struggle with numbers (Section 3), but this hasn't been measured in the context of agent-to-agent numeric communication — does information degrade as numbers pass through LLM agents?

---

## 8. References

1. Ansari, A.F. et al. (2024). Chronos: Learning the Language of Time Series. ICML 2024. [arXiv:2403.07815](https://arxiv.org/abs/2403.07815)
2. Alnegheimish, S. et al. (2024). Large Language Models Are Zero-Shot Time Series Anomaly Detectors. [arXiv:2405.14755](https://arxiv.org/abs/2405.14755)
3. Bischl, B. et al. (2021). OpenML Benchmarking Suites. NeurIPS 2021.
4. Cao, R. & Wang, Q. (2024). An Evaluation of Standard Statistical Models and LLMs on Time Series Forecasting. [arXiv:2408.04867](https://arxiv.org/abs/2408.04867)
5. Colelough, B.C. & Regli, W. (2025). Neuro-Symbolic AI in 2024: A Systematic Review. [arXiv:2501.05435](https://arxiv.org/abs/2501.05435)
6. Dziri, N. et al. (2023). Faith and Fate: Limits of Transformers on Compositionality. NeurIPS 2023. [arXiv:2305.18654](https://arxiv.org/abs/2305.18654)
7. Fang, X. et al. (2024). Large Language Models on Tabular Data: A Survey. [arXiv:2402.17944](https://arxiv.org/abs/2402.17944)
8. Fan, Y. et al. (2024). Foundation Models for Time Series Analysis: A Tutorial and Survey. KDD 2024. [arXiv:2403.14735](https://arxiv.org/abs/2403.14735)
9. Garza, A. & Mergenthaler-Canseco, M. (2023). TimeGPT-1. [arXiv:2310.03589](https://arxiv.org/abs/2310.03589)
10. Godahewa, R. et al. (2021). Monash Time Series Forecasting Archive. NeurIPS 2021. [arXiv:2105.06643](https://arxiv.org/abs/2105.06643)
11. Goswami, M. et al. (2024). MOMENT: A Family of Open Time-Series Foundation Models. ICML 2024. [arXiv:2402.03885](https://arxiv.org/abs/2402.03885)
12. Gou, Z. et al. (2024). ToRA: A Tool-Integrated Reasoning Agent. ICLR 2024. [arXiv:2309.17452](https://arxiv.org/abs/2309.17452)
13. Grinsztajn, L. et al. (2022). Why Do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data? NeurIPS 2022. [arXiv:2207.08815](https://arxiv.org/abs/2207.08815)
14. Gruver, N. et al. (2023). Large Language Models Are Zero-Shot Time Series Forecasters. NeurIPS 2023. [arXiv:2310.07820](https://arxiv.org/abs/2310.07820)
15. Han, S. et al. (2022). ADBench: Anomaly Detection Benchmark. NeurIPS 2022. [arXiv:2206.09426](https://arxiv.org/abs/2206.09426)
16. Hollmann, N. et al. (2023). TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second. ICLR 2023. [arXiv:2207.01848](https://arxiv.org/abs/2207.01848)
17. Hong, S. et al. (2023). MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework. ICLR 2024. [arXiv:2308.00352](https://arxiv.org/abs/2308.00352)
18. Jin, M. et al. (2024a). Time-LLM: Time Series Forecasting by Reprogramming Large Language Models. ICLR 2024. [arXiv:2310.01728](https://arxiv.org/abs/2310.01728)
19. Jin, M. et al. (2024b). What Can Large Language Models Tell Us about Time Series Analysis? ICML 2024. [arXiv:2402.02713](https://arxiv.org/abs/2402.02713)
20. Jin, H. et al. (2024). Large Language Models for Anomaly Detection in Computational Workflows. [arXiv:2409.19156](https://arxiv.org/abs/2409.19156)
21. Lavin, A. & Ahmad, S. (2015). Evaluating Real-Time Anomaly Detection Algorithms — The Numenta Anomaly Benchmark. IEEE ICMLA. [arXiv:1510.03336](https://arxiv.org/abs/1510.03336)
22. Li, G. et al. (2023). CAMEL: Communicative Agents for "Mind" Exploration. NeurIPS 2023. [arXiv:2303.17760](https://arxiv.org/abs/2303.17760)
23. Makridakis, S. et al. (2022). M5 Competition. International Journal of Forecasting 38(4).
24. Mirzadeh, I. et al. (2024). GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in LLMs. ICLR 2025. [arXiv:2410.05229](https://arxiv.org/abs/2410.05229)
25. Nam, J. et al. (2024). OCTree: Optimized Feature Generation for Tabular Data via LLMs. NeurIPS 2024. [arXiv:2406.08527](https://arxiv.org/abs/2406.08527)
26. Rasul, K. et al. (2024). Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting. [arXiv:2310.08278](https://arxiv.org/abs/2310.08278)
27. Schmidl, S. et al. (2022). Anomaly Detection in Time Series: A Comprehensive Evaluation. PVLDB 15(9). [doi:10.14778/3538598.3538602](https://doi.org/10.14778/3538598.3538602)
28. Shwartz-Ziv, R. & Armon, A. (2022). Tabular Data: Deep Learning is Not All You Need. Information Fusion 81. [arXiv:2106.03253](https://arxiv.org/abs/2106.03253)
29. Singh, A.K. & Strouse, D. (2024). Tokenization Counts: The Impact of Tokenization on Arithmetic in Frontier LLMs. [arXiv:2402.14903](https://arxiv.org/abs/2402.14903)
30. Tan, M. et al. (2024). Are Language Models Actually Useful for Time Series Forecasting? NeurIPS 2024. [arXiv:2406.16964](https://arxiv.org/abs/2406.16964)
31. Tran, K.-T. et al. (2025). Multi-Agent Collaboration Mechanisms: A Survey of LLMs. [arXiv:2501.06322](https://arxiv.org/abs/2501.06322)
32. Trinh, T.H. et al. (2024). Solving Olympiad Geometry Without Human Demonstrations. Nature. [doi:10.1038/s41586-023-06747-5](https://www.nature.com/articles/s41586-023-06747-5)
33. Tyen, G. et al. (2024). LLMs Cannot Find Reasoning Errors, but Can Correct Them. ACL 2024. [arXiv:2311.08516](https://arxiv.org/abs/2311.08516)
34. Wang, R. et al. (2023). UniPredict: Large Language Models are Universal Tabular Classifiers. [arXiv:2310.03266](https://arxiv.org/abs/2310.03266)
35. Wang, S. et al. (2024). LLMAD: Large Language Model-based IoT Agent for Anomaly Detection. [arXiv:2405.14014](https://arxiv.org/abs/2405.14014)
36. Wu, Q. et al. (2023). AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. [arXiv:2308.08155](https://arxiv.org/abs/2308.08155)
37. X-MAS (2025). Towards Building Multi-Agent Systems with Heterogeneous LLMs. [arXiv:2505.16997](https://arxiv.org/abs/2505.16997)
38. Xie, S. et al. (2025). ChatTS: Aligning Time Series with LLMs via Synthetic Data. VLDB 2025. [arXiv:2412.03104](https://arxiv.org/abs/2412.03104)
39. Xue, H. & Salim, F.D. (2023). PromptCast: A New Prompt-Based Learning Paradigm for Time Series Forecasting. IEEE TKDE 36(11). [arXiv:2210.08964](https://arxiv.org/abs/2210.08964)
40. Zeng, A. et al. (2023). Are Transformers Effective for Time Series Forecasting? AAAI 2023. [arXiv:2205.13504](https://arxiv.org/abs/2205.13504)
41. Zhou, T. et al. (2023). One Fits All: Power General Time Series Analysis by Pretrained LM. NeurIPS 2023. [arXiv:2302.11939](https://arxiv.org/abs/2302.11939)
42. Zhou, H. et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. AAAI 2021. [arXiv:2012.07436](https://arxiv.org/abs/2012.07436)
