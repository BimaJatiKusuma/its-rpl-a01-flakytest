# 📑 Comprehensive Research Design: Combined Experiment 1 & 2
## Cross-Project Flaky Test Prediction (Full Factorial 72-Pipeline Matrix & 10x Repeated LOPO-CV)

**Research Group**: Group A01 — Advanced Software Engineering (Master of Computer Science, Institut Teknologi Sepuluh Nopember)  
**Primary Master Plan**: `Eksperimen_Bima/Rencana_Penelitian_Flaky_Test.md`  
**Foundational Dataset**: FlakeFlagger Curated Benchmark (`test_features.csv` & `cleaned_test_features.csv`, 24 open-source Java projects, 22,236 test cases)  
**Target Hardware Acceleration**: NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM, CUDA 13.4) & Multi-threaded CPU parallelization  

---

## 1. Background & Motivation for Experimental Synthesis

Cross-Project Flaky Test Prediction (CP-FTP) initially faced a foundational dilemma concerning the treatment of missing feature values across 273 instances in the FlakeFlagger dataset (Alshammari et al., ICSE 2021):
- **Experiment 1** explored two data omission paradigms:
  - **C1 (Row-Level Discarding)**: Omitted 273 rows containing NaNs while retaining all 24 projects (21,963 test cases, 808 flaky tests).
  - **C2 (Project-Level Discarding)**: Omitted 5 entire projects containing NaNs, reducing the corpus to 19 projects (16,458 test cases, 586 flaky tests). However, omitting the massive enterprise system `spring-boot` eliminated ~20% of the authentic industrial flaky test population.
- **Experiment 2** introduced contextual data imputation:
  - **C3 (Group-Aware Per-Project Median Imputation)**: Preserved 100% of the dataset (24 projects, 22,236 test cases, 811 flaky tests) with zero sample omission.

**Urgency of Experimental Synthesis**:  
To enable a fair, controlled (*ceteris paribus*) comparison among the three data cleaning paradigms (**C1 vs. C2 vs. C3**), all three strategies are synthesized into a single full factorial experimental matrix evaluated across multiple variations of **Feature Scaling (S1-S3)**, **Class Imbalance Handling (B1-B4)**, and **Classifier Architectures (M1-M2)** using a rigorous **Leave-One-Project-Out Cross-Validation (LOPO-CV)** protocol and **10x Repeated Stochastic Trials** to assess empirical stability and consistency.

---

## 2. Full Factorial Experimental Design Matrix (72 Pipelines)

The experimental matrix is structured across 4 independent design factors:
$$\text{Total Pipelines} = 3 \text{ Cleaning} \times 3 \text{ Scaling} \times 4 \text{ Imbalance} \times 2 \text{ Models} = \mathbf{72 \text{ Pipelines}}$$

| Factor | Notation & Levels | Operational Specification & Zero Data Leakage Guarantee |
|---|---|---|
| **Data Cleaning (3)** | **C1**: Row-Level Discarding<br>**C2**: Project-Level Discarding<br>**C3**: Group-Aware Median Imputation | **C1**: 24 projects, 21,963 instances, 808 flaky.<br>**C2**: 19 projects, 16,458 instances, 586 flaky.<br>**C3**: 24 projects, 22,236 instances, 811 flaky (100% intact). |
| **Feature Scaling (3)** | **S1**: StandardScaler<br>**S2**: MinMaxScaler<br>**S3**: RobustScaler | Scalers are fitted **EXCLUSIVELY on training folds ($N-1$ projects)** and applied to unseen target project test folds. |
| **Imbalance Handling (4)** | **B1**: Baseline (Unweighted)<br>**B2**: SMOTE<br>**B3**: Random Under-Sampling (RUS)<br>**B4**: ClassWeight (Cost-Sensitive) | Applied **EXCLUSIVELY to training folds**. Test fold target distributions remain natural and unmodified. XGBoost B4 uses `scale_pos_weight = neg / pos`. |
| **Classifier Models (2)** | **M1**: Random Forest CPU<br>**M2**: XGBoost GPU CUDA | **M1**: `n_estimators=100`, CPU parallel (`n_jobs=-1`).<br>**M2**: `n_estimators=100`, `max_depth=6`, GPU histogram algorithm (`device="cuda"`, `tree_method="hist"`). |

---

## 3. Validation Protocols & 10x Consistency Framework

### A. Leave-One-Project-Out Cross-Validation (LOPO-CV) Scheme
- In each iteration, exactly $1$ project serves as the unseen test fold, while the remaining $N-1$ projects form the training set.
- For C1 and C3: 24 evaluation folds per pipeline.
- For C2: 19 evaluation folds per pipeline.
- Total single-run evaluations: $(24 \times 24) + (24 \times 19) + (24 \times 24) = 576 + 456 + 576 = \mathbf{1,608 \text{ Model Folds}}$.

### B. Repeated Stochastic Trials Framework (10x Random Seeds)
To evaluate variance introduced by stochastic resampling (RUS / SMOTE) and tree partitioning (RF / XGBoost), each pipeline is evaluated across 10 standardized random seeds:
```python
DEFAULT_SEEDS_10 = [42, 100, 2024, 777, 999, 123, 456, 789, 2025, 2026]
```
Total 10x Repeated Evaluations: $10 \times 1,608 = \mathbf{16,080 \text{ Model Folds}}$.

---

## 4. Evaluation Metrics & Statistical Analysis Framework

1. **Operational Performance Metrics**:
   - **F1-Score**: Harmonic mean of precision and recall on the minority class (flaky tests).
   - **Recall (Sensitivity)**: Proportion of actual flaky tests identified by the model.
   - **Precision**: Proportion of positive flaky predictions that are true flaky tests.
   - **PR-AUC (Average Precision)**: Area under the Precision-Recall curve, ideal for severe class imbalance.
   - **Matthews Correlation Coefficient (MCC)**: Balanced metric unaffected by class imbalance.
   - **Balanced Accuracy**: Mean of sensitivity and specificity.
2. **Empirical Consistency & Stability Metrics**:
   - Sample Mean ($\mu$) and Standard Deviation ($\sigma$) across 10 independent seeds.
   - **Coefficient of Variation (CV %)**: $CV = \frac{\sigma}{\mu} \times 100\%$.
   - **95% Confidence Interval**: $\text{CI}_{95} = \mu \pm 1.96 \frac{\sigma}{\sqrt{10}}$.
   - **Performance Range**: $\text{Range} = \text{Max} - \text{Min}$.
3. **Statistical Significance Testing (ACM/SIGSOFT Guidelines)**:
   - **Friedman Test**: Omnibus non-parametric test evaluating the null hypothesis of equal performance across 72 pipelines.
   - **Nemenyi Post-hoc Test**: Computes Critical Difference (CD) to isolate significantly different pipeline pairs.
   - **Kruskal-Wallis Test**: Validates the statistical significance of differences among C1, C2, and C3.

---

## 5. Formal Research Questions (RQs)

The study addresses four primary research questions:
- **RQ1 (Data Cleaning & Flaky Sample Preservation)**: Examines whether row-level cleaning (C1) or project-level cleaning (C2) better preserves flaky samples and supports cross-project generalization, benchmarked against contextual group-aware median imputation (C3).
- **RQ2 (Feature Scaling & Distribution Shift)**: Investigates which scaling method (S1: StandardScaler, S2: MinMaxScaler, or S3: RobustScaler) best mitigates cross-project distribution shift.
- **RQ3 (Class Imbalance & Precision-Recall Trade-Off)**: Evaluates which imbalance treatment (B1: Baseline, B2: SMOTE, B3: Random Under-Sampling, or B4: Cost-Sensitive / ClassWeight) provides the most useful precision-recall trade-off.
- **RQ4 (Model Families & Cross-Project Transferability)**: Compares the transferability of different model families (M1: Random Forest [Bagging CPU] vs. M2: XGBoost [Gradient Boosting CUDA GPU]) across unseen projects.
