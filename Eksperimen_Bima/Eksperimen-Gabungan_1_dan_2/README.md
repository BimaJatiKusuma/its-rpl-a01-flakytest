# 🔬 Unified Cross-Project Flaky Test Prediction (Combined Experiment 1 & 2)
## Full Factorial Benchmark (72 Machine Learning Pipelines) & 10x Repeated LOPO-CV

This experimental repository provides the unified, reproducible benchmark synthesizing **Experiment 1** (data omission paradigms) and **Experiment 2** (contextual median imputation). It evaluates the **Tri-Paradigm Data Cleaning Framework (C1: Row Discarding vs. C2: Project Discarding vs. C3: Group-Aware Median Imputation)** across 72 machine learning pipelines under strict **Leave-One-Project-Out Cross-Validation (LOPO-CV, Zero Data Leakage)** augmented with a **10x Repeated Stochastic Trial Protocol (16,080 Total Model Folds)**.

---

## 📋 Full Factorial 72-Pipeline Benchmark Matrix

The benchmark matrix systematically crosses 4 independent experimental factors ($3 \times 3 \times 4 \times 2 = \mathbf{72 \text{ Machine Learning Pipelines}}$):

| Experimental Factor | Levels & Variations | Operational Specification |
|---|---|---|
| **Data Cleaning (3)** | **C1**: Row-Level Discarding (24 projects, 21,963 rows)<br>**C2**: Project-Level Discarding (19 projects, 16,458 rows)<br>**C3**: Group-Aware Median Imputation (24 projects, 22,236 rows, 100% intact) | Compares row deletion vs. project deletion vs. 100% sample preservation via contextual median imputation. |
| **Feature Scaling (3)** | **S1**: StandardScaler<br>**S2**: MinMaxScaler<br>**S3**: RobustScaler | Fitted **EXCLUSIVELY on training folds ($N-1$ projects)** and applied to unseen test project folds. |
| **Imbalance Resampling (4)** | **B1**: Baseline (Unweighted)<br>**B2**: SMOTE<br>**B3**: Random Under-Sampling (RUS)<br>**B4**: ClassWeight (Cost-Sensitive) | Applied **EXCLUSIVELY on training folds** to preserve natural target project imbalance. |
| **Classifier Models (2)** | **M1**: Random Forest CPU (`n_estimators=100`, `n_jobs=-1`)<br>**M2**: XGBoost GPU CUDA (`device="cuda"`, `tree_method="hist"`) | Evaluates decorrelated bagging ensembles against CUDA GPU-accelerated gradient boosting. |

---

## 🗂️ Directory Structure

```
Eksperimen-Gabungan_1_dan_2/
├── README.md                                 # Primary repository overview & execution guide
├── COMBINED_EXPERIMENT_REPORT.md             # Comprehensive empirical synthesis report (scientific English)
├── RESEARCH_DESIGN.md                        # Academic research design and methodology specification
├── LAPORAN_EKSPERIMEN_GABUNGAN.md            # Cross-referenced report document
├── RANCANGAN_PENELITIAN_GABUNGAN.md          # Cross-referenced research design document
├── requirements.txt                          # Python package dependencies
├── configs/
│   └── experiment_matrix.json                # Parameter definitions for all 72 pipeline configurations
├── data/
│   ├── README.md                             # Data dictionary, feature taxonomy, & curation protocols
│   ├── test_features.csv                     # Raw FlakeFlagger feature matrix (for C1 & C2)
│   ├── cleaned_test_features.csv             # Contextually imputed feature matrix (for C3)
│   ├── test_results.csv                      # Test execution logs (10,000x reruns per test case)
│   └── Project_Info.csv                      # Metadata across 24 curated open-source Java projects
├── src/
│   ├── __init__.py
│   ├── check_gpu.py                          # GPU environment & CUDA runtime diagnostic utility
│   ├── data_loader.py                        # Unified data loader for C1, C2, and C3 paradigms
│   ├── pipeline_builder.py                   # Modular pipeline constructor (scaler + sampler + classifier)
│   ├── lopo_runner.py                        # Single-run LOPO-CV execution engine (1,608 folds)
│   ├── repeated_lopo_runner.py               # Repeated LOPO-CV engine (10x seeds = 16,080 folds)
│   ├── consistency_analyzer.py               # Empirical consistency & stability analyzer (CV%, CI95)
│   ├── statistical_tests.py                  # Statistical significance tests (Friedman, Nemenyi CD, Kruskal-Wallis)
│   └── visualizer.py                         # Publication-grade visualization generator (300 DPI IEEE/ACM standard)
├── notebooks/
│   └── 01_run_repeated_lopo_10x_and_consistency.ipynb # All-in-one interactive experimental notebook
└── results/
    ├── raw_predictions/                      # Per-fold, per-seed prediction records (720 CSV files)
    ├── per_seed_summaries/                   # Aggregated metric summaries per seed
    ├── all_seed_pipeline_runs.csv            # Master benchmark execution records across 10 seeds
    ├── consistency_stability_analysis.csv    # Consistency metrics (Mean, Std, CV%, CI95) per pipeline
    ├── consistency_factor_*.csv              # Factorial stability breakdowns (Cleaning, Scaling, Imbalance, Model)
    ├── summary_metrics_72_pipelines.csv      # Unified single-run evaluation summary
    └── plots/                                # 300 DPI publication figures (boxplots, heatmaps, CI95 bars)
```

---

## ⚡ Execution Guide

### 1. Single-Run LOPO-CV Benchmark (72 Factorial Pipelines):
- **Smoke Test (2 Folds)**:
  ```powershell
  python src/lopo_runner.py --quick_test
  ```
- **Execute All 72 Pipelines**:
  ```powershell
  python src/lopo_runner.py --all
  ```
- **Execute a Specific Pipeline (e.g., P61)**:
  ```powershell
  python src/lopo_runner.py --pipeline_id P61_C3_S2_B3_M1
  ```

### 2. Repeated Stochastic Benchmark (10x Seeds Consistency Test):
- **Smoke Test (3 Pipelines across C1/C2/C3, 2 Seeds, 2 Folds)**:
  ```powershell
  python src/repeated_lopo_runner.py --quick_test
  ```
- **Top 12 Ranked Pipelines Across 10 Seeds (High Efficiency)**:
  ```powershell
  python src/repeated_lopo_runner.py --n_repeats 10 --top_pipelines 12
  ```
- **Full Benchmark (All 72 Pipelines Across 10 Seeds = 16,080 Folds)**:
  ```powershell
  python src/repeated_lopo_runner.py --n_repeats 10 --all
  ```

### 3. Consistency Analysis & Statistical Significance Testing:
```powershell
python src/consistency_analyzer.py
python src/statistical_tests.py
python src/visualizer.py
```

### 4. Interactive Jupyter Notebook:
- Launch [`notebooks/01_run_repeated_lopo_10x_and_consistency.ipynb`](notebooks/01_run_repeated_lopo_10x_and_consistency.ipynb) for end-to-end interactive execution, statistical tables, and live rendering of publication-grade figures.

---

## 🔬 Key Empirical Highlights

- **Tri-Paradigm Benchmark**: Imputing missing values with Group-Aware Median Imputation (**C3**) elevates **Mean F1 by +11.39%** and **Precision by +28.31%** over row omission (**C1**), identifying **465 True Positive flaky tests** while preserving the complex enterprise benchmark `spring-boot` (Recall: 77.91%).
- **Golden Pipeline Recommendation**: **`P61_C3_S2_B3_M1`** (**C3 Imputation + MinMaxScaler + Random Under-Sampling + Random Forest**) delivers the premier operational balance (F1: 0.1816, Recall: 43.74%, Precision: 20.38%) alongside robust stochastic stability ($CV < 6\%$).
