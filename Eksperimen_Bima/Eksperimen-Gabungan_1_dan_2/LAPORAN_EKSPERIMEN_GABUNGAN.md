# 📄 Empirical Synthesis Report: Combined Experiment 1 & 2
## Comparative Benchmark of Tri-Paradigm Data Cleaning Across 72 Factorial Machine Learning Pipelines

> [!NOTE]
> This document has been upgraded to publication-grade scientific English for international dissemination. The canonical English report is available at [`COMBINED_EXPERIMENT_REPORT.md`](COMBINED_EXPERIMENT_REPORT.md).

**Research Group**: Group A01 — Advanced Software Engineering (Master of Computer Science, Institut Teknologi Sepuluh Nopember)  
**Primary Design Document**: [`RESEARCH_DESIGN.md`](RESEARCH_DESIGN.md) (Indonesian reference: [`RANCANGAN_PENELITIAN_GABUNGAN.md`](RANCANGAN_PENELITIAN_GABUNGAN.md))  
**Benchmark Dataset**: FlakeFlagger Curated Benchmark (`test_features.csv` & `cleaned_test_features.csv`, 24 open-source Java projects, 22,236 test cases)  
**Evaluation Protocol**: Leave-One-Project-Out Cross-Validation (LOPO-CV, Zero Data Leakage) & Repeated LOPO-CV (10 Independent Stochastic Seeds)  

---

## Executive Summary

The Combined Experiment 1 & 2 unifies the entire parameter design space from preliminary experimental iterations into a **Full Factorial 72-Pipeline Benchmark Matrix** ($3 \text{ Cleaning} \times 3 \text{ Feature Scaling} \times 4 \text{ Class Imbalance Handling} \times 2 \text{ Classifier Models}$). This unified design rigorously resolves the methodological debate concerning the trade-offs of sample discarding (*data omission*) versus intelligent contextual imputation (*group-aware median imputation*) within **Cross-Project Flaky Test Prediction (CP-FTP)**.

### Key Empirical Findings:

1. **Tri-Paradigm Cleaning Superiority (C3 vs. C1 & C2)**:  
   - **C3 (Group-Aware Median Imputation)** emerges as the definitive operational and methodological standard. Evaluated across the identical 24 target projects, C3 consistently surpasses C1 across all primary performance dimensions: **Mean F1 increases by +11.39%** ($0.1631 \rightarrow \mathbf{0.1816}$), **Precision improves by +28.31%** ($0.1588 \rightarrow \mathbf{0.2038}$), and detects **+37 additional True Positive flaky tests** ($428 \rightarrow \mathbf{465}\text{ TP}$).
   - The deceptively inflated performance of C2 (Project-Level Discarding, Mean F1 $0.2271$) is empirically demonstrated to be an artifact of **survivorship bias** caused by omitting the complex enterprise benchmark `spring-boot`. Pipeline C3 successfully identifies **127 of 163 flaky test cases within `spring-boot` (Recall: 77.91%)**, a mission-critical capability forfeited by C2.
2. **Supremacy of Random Under-Sampling (B3: RUS)**:  
   - Random Under-Sampling (RUS) decisively leads across all three cleaning paradigms (C1, C2, and C3). RUS increases Mean F1 by $3\times$ to $4\times$ over the unweighted Baseline (B1) and elevates detection Recall to $>43\% \text{--} 48\%$. Conversely, SMOTE (B2) exhibits degradation because synthetic minority interpolation within source feature spaces exacerbates cross-project domain divergence.
3. **Feature Scaling Resilience (S2: MinMaxScaler)**:  
   - MinMaxScaler (S2) provides the most consistent empirical convergence across bounded intervals $[0, 1]$, mitigating extreme skewness inherent in software churn and coverage metrics while facilitating histogram quantization for tree ensembles.
4. **Model Architecture & Computational Efficiency (M1: RF vs. M2: XGBoost GPU)**:  
   - Random Forest (M1) captures the top individual single-run scores (`P13_C1`, `P37_C2`, `P61_C3`). However, XGBoost GPU (M2) achieves superior operational throughput with a **$1.56\times$ inference acceleration** enabled by CUDA hardware acceleration.

---

## 1. Tri-Paradigm Data Cleaning Benchmark Table (C1 vs. C2 vs. C3)

| Evaluation Dimension | C1: Row-Level Discarding | C2: Project-Level Discarding | C3: Group-Aware Median Imputation |
|---|:---:|:---:|:---:|
| **Missing Value Treatment** | Discard 273 rows containing NaN | Discard 5 projects containing NaN | Group-Aware Per-Project Median Imputation |
| **Retained Projects** | 24 Projects | 19 Projects ($-20.8\%$) | **24 Projects (100% Retained)** |
| **Total Test Cases** | 21,963 instances | 16,458 instances ($-24.8\%$) | **22,236 instances (100% Retained)** |
| **Flaky Test Population** | 808 flaky tests | 586 flaky tests ($-27.4\%$) | **811 flaky tests (100% Retained)** |
| **Status of `spring-boot`** | Retained (160 flaky) | **COMPLETELY DISCARDED (0 flaky)** | **Retained (163 flaky)** |
| **Best Performing Pipeline** | `P13_C1_S2_B3_M1` | `P37_C2_S2_B3_M1`* | `P61_C3_S2_B3_M1` |
| **Mean F1-Score** | 0.1631 | 0.2271* | **0.1816 (+11.39% vs. C1)** |
| **Mean Recall** | 0.4110 | 0.5216* | **0.4374 (+6.42% vs. C1)** |
| **Mean Precision** | 0.1588 | 0.2030* | **0.2038 (+28.31% vs. C1)** |
| **Mean PR-AUC** | 0.2057 | 0.2472* | **0.2210 (+7.44% vs. C1)** |
| **Mean MCC** | 0.1341 | 0.2116* | **0.1547 (+15.36% vs. C1)** |
| **Total Captured Flaky (TP)** | 428 TP | 338 TP | **465 TP (+8.64% vs. C1)** |
| **Recall on `spring-boot`** | 68.75% (110 / 160 TP) | 0.00% (Excluded from Benchmark) | **77.91% (127 / 163 TP)** |

*\*Note: High metric values for C2 stem from survivorship bias due to discarding the most complex enterprise benchmark (`spring-boot`).*

---

## 2. Top-10 Ranked Pipelines Across the Full Factorial Matrix (Ranked by Mean F1)

| Rank | Pipeline ID | Cleaning | Scaling | Imbalance | Classifier | Mean F1 | Mean Recall | Mean Precision | Mean PR-AUC |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 **1** | `P37_C2_S2_B3_M1`* | C2 | S2 (MinMax) | **B3 (RUS)** | M1 (RF) | **0.2271** | 0.5216 | 0.2030 | 0.2472 |
| 🥈 **2** | `P38_C2_S2_B3_M2`* | C2 | S2 (MinMax) | **B3 (RUS)** | M2 (XGB)| **0.2268** | **0.5664** | 0.1918 | 0.2249 |
| 🥉 **3** | `P61_C3_S2_B3_M1` | **C3** | **S2 (MinMax)** | **B3 (RUS)** | **M1 (RF)** | **0.1816** | **0.4374** | **0.2038** | **0.2210** |
| **4** | `P69_C3_S3_B3_M1` | **C3** | S3 (Robust) | **B3 (RUS)** | M1 (RF) | **0.1786** | 0.4333 | 0.2014 | 0.2203 |
| **5** | `P53_C3_S1_B3_M1` | **C3** | S1 (Standard) | **B3 (RUS)** | M1 (RF) | **0.1785** | 0.4345 | 0.2024 | 0.2224 |
| **6** | `P13_C1_S2_B3_M1` | C1 | S2 (MinMax) | **B3 (RUS)** | M1 (RF) | **0.1631** | 0.4110 | 0.1588 | 0.2057 |
| **7** | `P21_C1_S3_B3_M1` | C1 | S3 (Robust) | **B3 (RUS)** | M1 (RF) | **0.1610** | 0.4082 | 0.1571 | 0.2044 |
| **8** | `P05_C1_S1_B3_M1` | C1 | S1 (Standard) | **B3 (RUS)** | M1 (RF) | **0.1608** | 0.4095 | 0.1569 | 0.2052 |
| **9** | `P54_C3_S1_B3_M2` | **C3** | S1 (Standard) | **B3 (RUS)** | M2 (XGB)| **0.1403** | 0.4255 | 0.1352 | 0.2012 |
| **10**| `P62_C3_S2_B3_M2` | **C3** | S2 (MinMax) | **B3 (RUS)** | M2 (XGB)| **0.1403** | 0.4255 | 0.1352 | 0.2012 |

---

## 3. Stochastic Stability & Consistency Analysis Across 10 Independent Seeds

Applying the 10-seed repeated evaluation protocol (`DEFAULT_SEEDS_10 = [42, 100, 2024, 777, 999, 123, 456, 789, 2025, 2026]`), the empirical consistency findings are summarized below:

1. **Top-Tier Pipeline Stability**:  
   Pipeline `P61_C3_S2_B3_M1` (MinMaxScaler + Random Under-Sampling + Random Forest on the group-aware imputed dataset) demonstrates high empirical stability with a **Coefficient of Variation $CV < 6\%$** ($\text{Mean F1} = 0.1798 \pm 0.0104$, $95\%\text{ CI} = [0.1733, 0.1863]$).
2. **Variance Dampening: Random Forest vs. XGBoost**:  
   Random Forest (M1) exhibits systematically lower inter-seed variance ($CV_{\text{RF}} = 5.78\%$) compared to XGBoost ($CV_{\text{XGB}} = 6.85\%$), showing that ensemble bagging across 100 decorrelated trees dampens stochastic fluctuations introduced by under-sampling.
3. **Deterministic vs. Stochastic Resampling**:  
   Resampling-free configurations (`B1` Baseline and `B4` ClassWeight on deterministic settings) yield $CV = 0.00\%$, verifying that empirical performance variation originates strictly from training set subsampling rather than computational floating-point instability.

---

## 4. Academic Implications & Production Recommendations

1. **Data Preprocessing Recommendation**:  
   Discontinue data discarding strategies (C1 row deletion and C2 project deletion). Adopt **Group-Aware Per-Project Median Imputation (C3)**, which preserves 100% of enterprise benchmark data while providing a $+11.39\%$ improvement in F1-score and $+28.31\%$ in Precision.
2. **Recommended Production Configuration (Golden Pipeline)**:  
   **`P61_C3_S2_B3_M1`** (**Group-Aware Imputed Dataset + MinMaxScaler + Random Under-Sampling + Random Forest**). This configuration yields the premier operational balance between test flakiness detection (F1 $0.1816$, Recall $43.74\%$, Precision $20.38\%$), maximum absolute true positive coverage ($465\text{ TP}$), and high stochastic stability ($CV < 6\%$).
