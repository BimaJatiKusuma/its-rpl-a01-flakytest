# 🚀 Replikasi Eksperimen 3: Multi-Dataset Cross-Project Flaky Test Prediction dengan TrAdaBoost Target Adaptation

> **Folder Penelitian**: `Eksperimen_Kelompok/bima-eksperimen-3-all-dataset-tradaboost`  
> **Peneliti**: Bima Jati Kusuma (Kelompok A01 — RPL A Magister Teknik Informatika ITS)  
> **Rujukan Utama**:
> - **Alshammari et al. (ICSE 2021)** — *FlakeFlagger: Predicting Flakiness Without Rerunning Tests*
> - **Afeltra et al. (IEEE Access 2024)** — *A Large-Scale Empirical Investigation Into Cross-Project Flaky Test Prediction*
> - **Dai et al. (ICML 2007)** — *Boosting for Transfer Learning (TrAdaBoost)*

---

## 📌 Ringkasan Proyek

Folder ini memuat replikasi komprehensif dari **Eksperimen 3 Bima** yang diperluas ke **dua dataset benchmark independen** dengan menerapkan algoritma **TrAdaBoost** (*Transfer AdaBoost*) pada berbagai variasi porsi data berlabel target yang disuntikkan ke dalam model.

Eksperimen menguji **1.230 evaluasi model (*model folds*)** di bawah protokol **Repeated Leave-One-Project-Out Cross-Validation (Repeated LOPO-CV)** pada **5 random seeds independen** (`[42, 100, 2024, 777, 999]`):
1. **FlakeFlagger Benchmark**: 24 proyek Java, 22.236 test cases, 811 flaky tests (3,65% flakiness rate).
2. **IDoFT Benchmark**: 17 proyek Java, 12.626 test cases, 264 flaky tests (2,09% flakiness rate).

---

## 🔬 Matriks Kombinasi Perlakuan Pipeline & Posisi TrAdaBoost

Dalam pipeline eksperimen ini, **TrAdaBoost** beroperasi sebagai **Ensemble Meta-Learner (Transfer Learning Wrapper)** di atas model dasar (`DecisionTreeClassifier(max_depth=4)`). TrAdaBoost memadukan data proyek sumber yang telah di-resample (**RUS**) dengan porsi data target adaptasi (**25%, 50%, 75%**) untuk menyaring instans sumber yang tidak relevan secara iteratif (20 iterasi boosting).

| ID | Nama Perlakuan | Porsi Data Target (Sampling) | Perlakuan Imbalance (Resampling) | Posisi / Status TrAdaBoost | Model Dasar (Estimator) | Set Pengujian (Test Set) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **T1** | **Baseline Zero-Shot (RF)** | **0%** Target Train<br>(LOPO Murni) | **RUS** pada $N-1$ Proyek Sumber | **Tidak Aktif**<br>(Model Konvensional Non-Transfer) | **Random Forest**<br>(100 trees, `class_weight='balanced'`) | 100% Data Proyek Target |
| **T1b** | **Baseline Zero-Shot (XGB)** | **0%** Target Train<br>(LOPO Murni) | **RUS** pada $N-1$ Proyek Sumber | **Tidak Aktif**<br>(Model Konvensional Non-Transfer) | **XGBoost GPU**<br>(100 trees, `max_depth=6`, `hist`) | 100% Data Proyek Target |
| **T2** | **TrAdaBoost 25%** | **25%** Target Train<br>(Stratified Split) | **RUS** pada $N-1$ Proyek Sumber | **Aktif** (Meta-Learner Pembobot Instans, 20 iterasi) | **Decision Tree**<br>(`max_depth=4`, weak learner) | 75% Sisa Data Proyek Target |
| **T3** | **TrAdaBoost 50%** | **50%** Target Train<br>(Stratified Split) | **RUS** pada $N-1$ Proyek Sumber | **Aktif** (Meta-Learner Pembobot Instans, 20 iterasi) | **Decision Tree**<br>(`max_depth=4`, weak learner) | 50% Sisa Data Proyek Target |
| **T4** | **TrAdaBoost 75%** | **75%** Target Train<br>(Stratified Split) | **RUS** pada $N-1$ Proyek Sumber | **Aktif** (Meta-Learner Pembobot Instans, 20 iterasi) | **Decision Tree**<br>(`max_depth=4`, weak learner) | 25% Sisa Data Proyek Target |
| **T5** | **Within-Project (Upper Bound)** | **100%** Data Target Lokal<br>(Batas Atas Teoritis) | **None**<br>(Distribusi Alami Terstratifikasi) | **Tidak Aktif**<br>(Klasifikasi Terisolasi Lokal) | **Random Forest**<br>(5-Fold Stratified CV, 100 trees) | 20% Data Proyek Target per Fold |

---

## 📊 Ringkasan Hasil Empiris Utama

| Dataset | Perlakuan | Unweighted Macro F1 | Active Macro F1 | Flaky-Weighted F1 | Global Micro F1 | Precision | Recall | PR-AUC | Stabilitas CV% |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **FlakeFlagger** | Baseline Zero-Shot (RF) | **0.1769** | 0.1846 | 0.3751 | 0.1637 | 0.1728 | 0.4126 | 0.2128 | $5,33\%$ (Sangat Tinggi) |
| **FlakeFlagger** | TrAdaBoost 25% | **0.2810** | 0.2932 | 0.5878 | 0.5747 | 0.3075 | 0.3545 | 0.2366 | $10,64\%$ (Moderat) |
| **FlakeFlagger** | TrAdaBoost 50% | **0.3474** | 0.3625 | 0.6372 | 0.6338 | 0.4042 | 0.3875 | 0.3262 | $8,10\%$ (Sangat Tinggi) |
| **FlakeFlagger** | **TrAdaBoost 75%** | **0.3773** | **0.3937** | **0.6490** | **0.6155** | **0.5219** | 0.4035 | **0.3891** | $10,62\%$ (Moderat) |
| **FlakeFlagger** | Within-Project | **0.4487** | 0.4682 | 0.7331 | 0.7495 | 0.5531 | 0.4317 | 0.4494 | $5,55\%$ (Sangat Tinggi) |
| **IDoFT** | Baseline Zero-Shot (RF) | **0.0475** | 0.0621 | 0.1046 | 0.0326 | 0.0536 | 0.1972 | 0.0791 | $63,52\%$ (Sensitif) |
| **IDoFT** | TrAdaBoost 25% | **0.1719** | 0.2248 | 0.4475 | 0.4133 | 0.2323 | 0.1901 | 0.1483 | $13,80\%$ (Moderat) |
| **IDoFT** | TrAdaBoost 50% | **0.2136** | 0.2793 | 0.5081 | 0.4933 | 0.3234 | 0.2231 | 0.2066 | $15,28\%$ (Moderat) |
| **IDoFT** | **TrAdaBoost 75%** | **0.2085** | **0.2726** | **0.5367** | **0.5184** | **0.4914** | 0.2313 | **0.2473** | $22,23\%$ (Moderat) |
| **IDoFT** | Within-Project | **0.3046** | 0.3983 | 0.6273 | 0.6462 | 0.5476 | 0.3016 | 0.3216 | $2,09\%$ (Sangat Tinggi) |

---

## 📂 Struktur Direktori Proyek

```text
bima-eksperimen-3-all-dataset-tradaboost/
├── configs/
│   └── experiment_matrix.json             # Konfigurasi JSON eksperimen
├── data/
│   ├── dataset_metadata.json              # Metadata demografi kedua dataset
│   ├── cleaned_flakeflagger_features.csv  # Dataset FlakeFlagger (22.236 baris, 24 proyek)
│   └── cleaned_idoft_features.csv         # Dataset IDoFT (12.626 baris, 17 proyek)
├── src/
│   ├── __init__.py
│   ├── data_loader.py                     # Pemuat data dan verifikasi 23 fitur prediktif (Imputed & Row-Drop)
│   ├── tradaboost.py                      # Algoritma TrAdaBoost & Stratified target splitting
│   ├── pipeline_builder.py                # Preprocessing (S1, S2, S3), Samplers (RUS, SMOTE, None), Classifiers (M1, M2, M3)
│   ├── dual_dataset_tradaboost_runner.py  # Engine Repeated LOPO-CV 5 seeds lintas 2 dataset (Base 6 Treatments)
│   ├── comprehensive_pipeline_runner.py   # Runner Multi-Faktorial 24 Pipeline (4.665 Evaluasi Model)
│   ├── comprehensive_analyzer.py          # Analyzer 6 faktor pipeline & generator tabel komparatif
│   ├── generate_comprehensive_plots.py    # Generator 4 plot publikasi multi-faktorial 300 DPI
│   ├── consistency_analyzer.py            # Kalkulator Mean, Std, CV%, CI95 lintas seeds
│   ├── dataset_profiler.py                # Profiling proyek, 4 skema agregasi, korelasi
│   ├── statistical_tests.py               # Uji Wilcoxon, Cliff's delta, & Friedman Nemenyi
│   ├── visualizer.py                      # Pembuat grafik visualisasi publikasi 300 DPI
│   └── main_executor.py                   # Orkestrator end-to-end eksekusi
├── results/
│   ├── raw/
│   │   ├── flakeflagger_repeated_folds_predictions.csv  # 720 baris fold
│   │   ├── idoft_repeated_folds_predictions.csv         # 510 baris fold
│   │   ├── all_datasets_combined_predictions.csv        # 1.230 baris base
│   │   └── comprehensive_all_pipelines_predictions.csv  # 4.665 baris multi-faktorial lengkap
│   ├── summary/
│   │   ├── comprehensive_pipeline_benchmark_summary.csv # Rangkuman 24 konfigurasi pipeline
│   │   ├── factor_analysis_imbalance.csv                # Evaluasi B1, B2, B3, B4 pada TrAdaBoost
│   │   ├── factor_analysis_scaling.csv                  # Evaluasi S1, S2, S3 pada TrAdaBoost
│   │   ├── factor_analysis_base_estimators.csv          # Evaluasi DT Depth 2, 4, 6 vs RF Base
│   │   ├── factor_analysis_tradaboost_configs.csv       # Evaluasi iterasi (10, 20, 30) & skema voting
│   │   ├── factor_analysis_cleaning.csv                 # Evaluasi Row-Drop (C1) vs Imputasi (C2)
│   │   ├── consistency_stability_analysis.csv           # Ringkasan CV% & CI95
│   │   ├── project_capability_profile.csv               # Profil kapabilitas per proyek
│   │   ├── macro_vs_weighted_f1_summary.csv             # 4 skema agregasi matematis
│   │   ├── statistical_tests_results.csv                # Hasil uji Wilcoxon & Cliff's d
│   │   └── statistical_tests_results_friedman.csv       # Hasil uji Friedman & peringkat
│   └── plots/
│       ├── side_by_side_tradaboost_progression.png      # Plot progresi metrik 0% -> 75%
│       ├── boxplot_tradaboost_portions_both_datasets.png # Boxplot F1 per proyek
│       ├── heatmap_flakeflagger_tradaboost_f1.png       # Heatmap 24 proyek FlakeFlagger
│       ├── heatmap_idoft_tradaboost_f1.png              # Heatmap 17 proyek IDoFT
│       ├── flakiness_rate_vs_target_benefit_scatter.png # Scatter korelasi flakiness vs delta
│       ├── radar_or_bar_comparison_treatments.png       # Grouped bar 4 skema agregasi
│       ├── factor_imbalance_handling_tradaboost.png     # Plot komparasi Imbalance B1, B2, B3, B4
│       ├── factor_base_estimator_depth_tradaboost.png   # Plot komparasi DT depth vs RF Base
│       ├── factor_data_cleaning_tradaboost.png          # Plot komparasi Row-Drop vs Imputasi
│       └── factor_scaling_and_voting_tradaboost.png     # Plot komparasi Iterasi & Voting
├── notebooks/
│   ├── 01_run_all_dataset_tradaboost_experiment.ipynb   # Jupyter Notebook interaktif
│   └── build_notebook.py
├── RANCANGAN_PENELITIAN_ALL_DATASET_TRADABOOST.md       # Protokol & desain empiris
├── LAPORAN_EKSPERIMEN_3_ALL_DATASET_TRADABOOST.md       # Laporan ilmiah komprehensif
└── README.md                                            # Dokumentasi navigasi ini
```

---

## ⚡ Cara Menjalankan (Quick Start)

Untuk mereplikasi seluruh pipeline dari awal:

```bash
# Masuk ke direktori eksperimen
cd "Eksperimen_Kelompok/bima-eksperimen-3-all-dataset-tradaboost"

# Menjalankan end-to-end pipeline (data loader, runner, analyzer, stats, plots)
python src/main_executor.py
```

Untuk membuka dan mengeksekusi Jupyter Notebook:
```bash
jupyter notebook notebooks/01_run_all_dataset_tradaboost_experiment.ipynb
```

---

## 📖 Laporan Lengkap

Baca laporan ilmiah lengkap dengan ulasan mendalam per Research Question pada:  
👉 [`LAPORAN_EKSPERIMEN_3_ALL_DATASET_TRADABOOST.md`](file:///e:/Magister%20ITS/SEMESTER%20I/Rekayasa%20Perangkat%20Lunak%20A/Bell%20Labs%20hehe/Eksperimen_Kelompok/bima-eksperimen-3-all-dataset-tradaboost/LAPORAN_EKSPERIMEN_3_ALL_DATASET_TRADABOOST.md)
