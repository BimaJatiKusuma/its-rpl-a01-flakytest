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
├── LAPORAN_EKSPERIMEN_GABUNGAN.md            # Laporan komprehensif temuan eksperimen (Bahasa Indonesia)
├── RANCANGAN_PENELITIAN_GABUNGAN.md          # Dokumen perancangan metodologi riset (Bahasa Indonesia)
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
│   ├── generate_cleaned_dataset.py           # Standalone dataset preparation & imputation generator
│   ├── data_loader.py                        # Unified data loader for C1, C2, and C3 paradigms
│   ├── pipeline_builder.py                   # Modular pipeline constructor (scaler + sampler + classifier)
│   ├── lopo_runner.py                        # Single-run LOPO-CV execution engine (1,608 folds)
│   ├── repeated_lopo_runner.py               # Repeated LOPO-CV engine (10x seeds = 16,080 folds)
│   ├── consistency_analyzer.py               # Empirical consistency & stability analyzer (CV%, CI95)
│   ├── statistical_tests.py                  # Statistical tests (Friedman, Nemenyi CD, Kruskal-Wallis)
│   └── visualizer.py                         # Publication-grade visualization generator (300 DPI IEEE/ACM standard)
├── notebooks/
│   ├── 00_eda_and_data_cleaning.ipynb        # Phase 1: Interactive Data Preparation, Imputation & EDA
│   ├── 01_run_main_72_pipelines.ipynb        # Phase 2: Interactive Single-Run LOPO-CV Benchmark (72 Pipelines)
│   └── 02_run_repeated_lopo_10x_and_consistency.ipynb # Phase 3: Interactive 10x Repeated LOPO-CV & Consistency Benchmark
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

## 📊 Dataset Ingestion & Preparation Protocols

Before running benchmarks, understanding the dataset structure and preparation workflows ensures full experimental reproducibility:

### 1. Dataset Provenance & Characteristics
- **Raw Dataset (`data/test_features.csv`)**:
  - Originates from the canonical **FlakeFlagger** benchmark across 24 open-source Java projects (22,236 test cases, 811 flaky tests).
  - Contains **273 missing values** localized in 3 continuous execution/coverage features (`testLength`, `numAsserts`, `numCoveredLines`) across 5 specific projects: `wildfly` (206), `logback` (38), `spring-boot` (12), `wro4j` (10), and `handlebars.java` (7).
- **Imputed Cleaned Dataset (`data/cleaned_test_features.csv`) & Provenance**:
  - Originally developed by Ali in [`Experiment_Ali/`](../../Experiment_Ali/) during Phase 1 Data Cleaning & EDA via notebook [`01_eda_and_feature_inspection.ipynb`](../../Experiment_Ali/notebook/01_eda_and_feature_inspection.ipynb) (generated via `Experiment_Ali/script/build_full_notebook.py`).
  - Replicated deterministically in this repository via [`src/generate_cleaned_dataset.py`](src/generate_cleaned_dataset.py) and [`notebooks/00_eda_and_data_cleaning.ipynb`](notebooks/00_eda_and_data_cleaning.ipynb).
  - **Cryptographic Hash Verification**: The generated file is **100% bit-for-bit identical** to Ali's original cleaned dataset:
    - `Experiment_Ali/data/cleaned_test_features.csv`: `SHA-256: BEAE188AAAEAA976C0D5F2E015045C928C5EC406D7C965601DF8FE29D33CC3B1`
    - `Eksperimen-Gabungan_1_dan_2/data/cleaned_test_features.csv`: `SHA-256: BEAE188AAAEAA976C0D5F2E015045C928C5EC406D7C965601DF8FE29D33CC3B1`
- **Canonical Feature Space (23 Predictive Features)**:
  - **8 Test Smells (Binary)**: `assertion-roulette`, `conditional-test-logic`, `eager-test`, `fire-and-forget`, `indirect-testing`, `mystery-guest`, `resource-optimism`, `test-run-war`.
  - **7 Execution & Coverage Metrics (Continuous)**: `testLength`, `numAsserts`, `numCoveredLines`, `ExecutionTime`, `projectSourceLinesCovered`, `projectSourceClassesCovered`, `num_third_party_libs`.
  - **8 Code Churn Metrics (Continuous)**: Sliding H-Index modifications per covered line (`window5` through `window10000`).

### 2. Tri-Paradigm Data Cleaning Formulations
The experiment evaluates three distinct data preparation paradigms:
1. **Paradigm C1 — Row-Level Discarding (Experiment 1)**:
   - Rows containing any NaN across the 23 predictive features are dropped.
   - Result: **21,963 test cases** across all **24 projects** (808 flaky tests, 3 flaky tests lost).
2. **Paradigm C2 — Project-Level Discarding (Experiment 1)**:
   - Any project containing at least one missing value is entirely discarded (`wildfly`, `logback`, `spring-boot`, `wro4j`, `handlebars.java`).
   - Result: **16,458 test cases** across **19 projects** ($-20.8\%$ projects, $-24.8\%$ test cases). **Fatal flaw**: Discards 222 ground-truth flaky tests (including 163 from `spring-boot`).
3. **Paradigm C3 — Group-Aware Median Imputation (Experiment 2)**:
   - Missing features are imputed using the **median of the respective project group** ($X_{i, j} \leftarrow \text{median}(\{X_{k, j} \mid \text{project}_k = \text{project}_i\})$), followed by probabilistic fallback sampling if an entire project group is null.
   - Result: **22,236 test cases** across **24 projects** (0 missing values, **100% data preservation**, 811 intact flaky tests).

### 3. How to Prepare & Reproduce the Datasets
Both CLI script and interactive notebook methods are provided:

- **Method A: Standalone CLI Script (`src/generate_cleaned_dataset.py`)**:
  Generates `data/cleaned_test_features.csv` directly from `data/test_features.csv` with a single command:
  ```powershell
  python src/generate_cleaned_dataset.py
  ```
  *(Options: `--input <path>`, `--output <path>`, `--seed 42`)*.

- **Method B: Interactive Jupyter Notebook (`notebooks/00_eda_and_data_cleaning.ipynb`)**:
  Provides complete step-by-step EDA, missing value visualization, group-aware imputation logic, correlation heatmaps, and dataset saving.

- **Automated Ingestion (`src/data_loader.py`)**:
  When executing models, `load_dataset('C1')`, `load_dataset('C2')`, and `load_dataset('C3')` automatically handle row filtering, project pruning, and reading imputed data seamlessly.

---

## ⚡ Dual Execution Guide: Notebooks vs. Python CLI

The repository supports two complete, symmetrical execution workflows:
- **Modality A: Interactive Jupyter Notebooks (`.ipynb`)** — Best for interactive analysis, inspecting dataframes, step-by-step debugging, and immediate inline visualization.
- **Modality B: Python CLI Scripts (`src/*.py`)** — Best for headless automation, background batch execution, remote servers, and CLI arguments.

```
                    ┌────────────────────────────────────────────────────────┐
                    │               FlakeFlagger Raw Data                     │
                    │              (data/test_features.csv)                  │
                    └──────────────────────────┬─────────────────────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
     【 MODALITY A: JUPYTER NOTEBOOKS 】              【 MODALITY B: PYTHON SCRIPTS CLI 】
                       │                                               │
  Step 0: notebooks/00_eda_and_data_cleaning.ipynb   Step 0: python src/generate_cleaned_dataset.py
          (Imputation & Paradigm Verification)               (Generates cleaned_test_features.csv)
                       │                                               │
  Step 1: notebooks/01_run_main_72_pipelines.ipynb   Step 1: python src/lopo_runner.py --all
          (Single-Run LOPO-CV 72 Pipelines)                  (Executes 1,608 model folds)
                       │                                               │
  Step 2: notebooks/02_run_repeated_lopo_10x_...     Step 2: python src/repeated_lopo_runner.py --all
          (10x Repeated LOPO-CV, Consistency,                (Executes 16,080 folds across 10 seeds)
           Statistical Tests & Inline Plots)                           │
                                                     Step 3: python src/consistency_analyzer.py
                                                             python src/statistical_tests.py
                                                             python src/visualizer.py
```

---

### 🖥️ Modality A: Interactive Execution via Jupyter Notebooks (`.ipynb`)

Launch Jupyter Lab or Jupyter Notebook from the workspace root or repository root:
```powershell
jupyter lab
# or: jupyter notebook
```

Navigate to the `notebooks/` directory and execute the following sequential notebooks:

#### 1. `notebooks/00_eda_and_data_cleaning.ipynb` (Phase 1: Preparation & EDA)
- **Purpose**: Explores raw dataset, audits 273 missing values, performs Group-Aware Per-Project Median Imputation, generates correlation heatmaps, and verifies the 3 paradigms (C1: 21,963 rows, C2: 16,458 rows, C3: 22,236 rows).
- **Output**: Saves `data/cleaned_test_features.csv`.

#### 2. `notebooks/01_run_main_72_pipelines.ipynb` (Phase 2: Single-Run LOPO-CV Benchmark)
- **Purpose**: Validates hardware (CUDA GPU), ingests C1/C2/C3 datasets, executes or loads single-run LOPO-CV across 72 pipelines (1,608 folds), displays top-10 pipelines ranked by Mean F1, and runs Kruskal-Wallis cleaning paradigm tests.
- **Output**: Generates `results/summary_metrics_72_pipelines.csv`.

#### 3. `notebooks/02_run_repeated_lopo_10x_and_consistency.ipynb` (Phase 3: 10x Repeated Stochastic Benchmark)
- **Purpose**: The master all-in-one experimental notebook.
  - Verifies RTX 4050 CUDA GPU acceleration.
  - Executes full 10x repeated LOPO-CV across 10 stochastic seeds (`[42, 100, 2024, 777, 999, 123, 456, 789, 2025, 2026]`) or loads existing records from disk.
  - Computes Mean ($\mu$), Standard Deviation ($\sigma$), Coefficient of Variation ($CV\%$), and 95% Confidence Intervals.
  - Executes Friedman omnibus test and Nemenyi critical difference (CD) post-hoc tests.
  - Renders all 6 IEEE/ACM publication-grade figures (300 DPI) directly inline.
- **Output**: Generates all CSV summaries and plots in `results/`.

*(Note: `notebooks/01_run_repeated_lopo_10x_and_consistency.ipynb` is also retained as a direct alias for backwards compatibility).*

---

### ⌨️ Modality B: Headless Execution via Python CLI Scripts (`src/*.py`)

Run scripts directly from the `Eksperimen-Gabungan_1_dan_2` directory in PowerShell:

#### Step 0: Hardware Diagnostic & Dataset Preparation
```powershell
# 1. Verify NVIDIA GPU & CUDA environment
python src/check_gpu.py

# 2. Reproduce / verify cleaned dataset (Group-Aware Median Imputation)
python src/generate_cleaned_dataset.py
```

#### Step 1: Single-Run LOPO-CV Benchmark (72 Pipelines = 1,608 Folds)
```powershell
# Fast Smoke Test (2 Folds for sanity verification)
python src/lopo_runner.py --quick_test

# Execute a single pipeline configuration (e.g., P61)
python src/lopo_runner.py --pipeline_id P61_C3_S2_B3_M1

# Full Benchmark: Execute all 72 pipelines across 24 projects
python src/lopo_runner.py --all
```

#### Step 2: Repeated Stochastic Benchmark (10x Seeds = 16,080 Folds)
```powershell
# Fast Smoke Test (3 Pipelines across C1/C2/C3, 2 Seeds, 2 Folds)
python src/repeated_lopo_runner.py --quick_test

# High-Efficiency Mode: Top 12 Ranked Pipelines across 10 Seeds
python src/repeated_lopo_runner.py --n_repeats 10 --top_pipelines 12

# Full Benchmark: All 72 Pipelines across 10 Seeds (16,080 Folds)
python src/repeated_lopo_runner.py --n_repeats 10 --all
```

#### Step 3: Consistency Analysis, Statistical Hypothesis Testing & Plotting
```powershell
# 1. Compute stability metrics (CV%, CI95, Mean, Std)
python src/consistency_analyzer.py

# 2. Run Friedman Omnibus, Nemenyi CD, and Kruskal-Wallis tests
python src/statistical_tests.py

# 3. Generate all 6 publication-grade figures (300 DPI) into results/plots/
python src/visualizer.py
```

---

## 🔬 Key Empirical Highlights & Findings

- **Tri-Paradigm Benchmark**: Imputing missing values with Group-Aware Median Imputation (**C3**) elevates **Mean F1 by +11.39%** and **Precision by +28.31%** over row omission (**C1**), identifying **465 True Positive flaky tests** while preserving the complex enterprise benchmark `spring-boot` (Recall: 77.91%).
- **Fatal Defect of Project Deletion (C2)**: Discarding 5 projects removes **222 ground-truth flaky tests** (27.4% of all flaky tests), producing artificial metric inflation and crippling cross-project transferability.
- **Golden Pipeline Recommendation**: **`P61_C3_S2_B3_M1`** (**C3 Imputation + MinMaxScaler + Random Under-Sampling + Random Forest**) delivers the premier operational balance (F1: 0.1816, Recall: 43.74%, Precision: 20.38%) alongside robust stochastic stability ($CV < 6\%$).
