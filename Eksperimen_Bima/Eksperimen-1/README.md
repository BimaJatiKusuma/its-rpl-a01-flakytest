# 🔬 Eksperimen 1: Cross-Project Flaky Test Prediction (Full Factorial 48 Pipelines)

Repositori ini berisi implementasi teknis dan eksekusi empiris dari **Rencana Penelitian Utama** yang dirumuskan pada [`../Rencana_Penelitian_Flaky_Test.md`](../Rencana_Penelitian_Flaky_Test.md), kini dilengkapi dengan **Framework Uji Konsistensi 10x Perulangan (Repeated LOPO-CV lintas 10 Random Seeds)**.

---

## 📋 Ikhtisar Eksperimen
- **Skenario Evaluasi**: Leave-One-Project-Out Cross-Validation (LOPO-CV) lintas 24 repositori Java open-source.
- **Hardware Acceleration**: GPU NVIDIA GeForce RTX 4050 Laptop (6GB VRAM, CUDA 13.4) & CPU multi-threading.
- **Dataset**: Dataset empiris FlakeFlagger (`test_features.csv`, `test_results.csv`, `Project_Info.csv`).
- **Total Pipelines**: **48 Pipelines** ($2 \times 3 \times 4 \times 2$):
  - **Data Cleaning (2)**: C1 (Row-Level Drop, 24 proyek) vs C2 (Project-Level Drop, 19 proyek)
  - **Feature Scaling (3)**: S1 (StandardScaler), S2 (MinMaxScaler), S3 (RobustScaler)
  - **Imbalance Handling (4)**: B1 (Baseline/None), B2 (SMOTE), B3 (Random Under-Sampling), B4 (Cost-Sensitive/Class Weight)
  - **Classifiers (2)**: M1 (Random Forest) vs M2 (XGBoost GPU)
- **Uji Konsistensi (Repeated Trials)**: **10 Random Seeds Independen** (`[42, 100, 2024, 777, 999, 123, 456, 789, 2025, 2026]`) untuk mengukur variansi stokastik, Coefficient of Variation ($CV = \frac{\sigma}{\mu} \times 100\%$), serta 95% Confidence Interval.

---

## 🗂️ Struktur Direktori

```
Eksperimen_Bima/Eksperimen-1/
├── README.md                                 # Dokumentasi ikhtisar & petunjuk eksekusi
├── PROMPT_EKSEKUSI_EKSPERIMEN_1.md          # Master prompt komprehensif untuk memulai riset
├── LAPORAN_EKSPERIMEN_1.md                   # Laporan komprehensif hasil temuan empiris
├── requirements.txt                          # Dependensi paket Python yang dibutuhkan
├── configs/
│   └── experiment_matrix.json                # Definisi parameter 48 konfigurasi pipeline
├── data/
│   ├── README.md                             # Dokumentasi kamus data & strategi C1/C2
│   ├── test_features.csv                     # Dataset utama FlakeFlagger (22.236 baris, 29 kolom)
│   ├── test_results.csv                      # Data eksekusi rerun 10.000x per test case
│   └── Project_Info.csv                      # Metadata 24 repositori Java open-source
├── src/
│   ├── __init__.py
│   ├── check_gpu.py                          # Verifikasi GPU RTX 4050 & CUDA
│   ├── data_loader.py                        # Modul ingestion dataset & cleaning C1/C2
│   ├── pipeline_builder.py                   # Modul scaler, sampler, classifier & specs
│   ├── lopo_runner.py                        # Engine eksekusi LOPO-CV (single run)
│   ├── repeated_lopo_runner.py               # [BARU] Engine Repeated LOPO-CV (hingga 10x Seeds)
│   ├── consistency_analyzer.py               # [BARU] Analisis konsistensi & stabilitas (CV, CI95)
│   ├── statistical_tests.py                  # Uji Friedman & Nemenyi post-hoc CD
│   └── visualizer.py                         # Pembuatan diagram, boxplot, dan heatmap
├── notebooks/
│   ├── 01_run_main_48_pipelines.ipynb        # Notebook interaktif untuk eksekusi single-run
│   └── 02_run_repeated_lopo_10x_and_consistency.ipynb # [BARU] Notebook eksekusi 10x perulangan
└── results/
    ├── raw_predictions/                      # Log prediksi mentah per fold & seed
    ├── per_seed_summaries/                   # Ringkasan agregat per seed
    ├── all_seed_pipeline_runs.csv            # Master seluruh runs lintas repetisi
    ├── consistency_stability_analysis.csv    # Rekapitulasi Mean, Std, CV%, CI95 per pipeline
    ├── consistency_factor_*.csv              # Analisis stabilitas per faktor (Cleaning, Scaling, dll.)
    ├── summary_metrics_48_pipelines.csv      # Ringkasan single-run awal
    └── plots/                                # Boxplot, heatmap F1, dan diagram CD
```

---

## ⚡ Langkah Eksekusi Uji Konsistensi (10x Perulangan)

### 1. Eksekusi Skrip CLI:
- **Mode Smoke Test Cepat (2 Fold, 2 Seeds)**:
  ```powershell
  python src/repeated_lopo_runner.py --quick_test
  ```
- **Mode Top 10 Pipelines (10x Seeds, Rekomendasi Cepat & Akurat)**:
  ```powershell
  python src/repeated_lopo_runner.py --n_repeats 10 --top_pipelines 10
  ```
- **Mode Pipeline Tunggal (10x Seeds)**:
  ```powershell
  python src/repeated_lopo_runner.py --pipeline_id P37_C2_S2_B3_M1 --n_repeats 10
  ```
- **Mode Penuh (Seluruh 48 Pipelines × 10 Seeds)**:
  ```powershell
  python src/repeated_lopo_runner.py --n_repeats 10 --all
  ```

### 2. Menjalankan Analisis Konsistensi:
Setelah proses perulangan selesai, analisis konsistensi akan otomatis dijalankan, atau dapat dipicu manual:
```powershell
python src/consistency_analyzer.py
```

### 3. Melalui Jupyter Notebook Interaktif:
Buka [`notebooks/02_run_repeated_lopo_10x_and_consistency.ipynb`](notebooks/02_run_repeated_lopo_10x_and_consistency.ipynb) untuk menjalankan eksekusi dan melihat visualisasi kestabilan boxplot secara langsung.
