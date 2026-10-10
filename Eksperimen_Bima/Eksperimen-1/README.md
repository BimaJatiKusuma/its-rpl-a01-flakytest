# 🔬 Eksperimen 1: Cross-Project Flaky Test Prediction (Full Factorial 48 Pipelines)

Repositori ini berisi implementasi teknis dan eksekusi empiris dari **Rencana Penelitian Utama** yang dirumuskan pada [`../Rencana_Penelitian_Flaky_Test.md`](../Rencana_Penelitian_Flaky_Test.md).

---

## 📋 Ikhtisar Eksperimen
- **Skenario Evaluasi**: Leave-One-Project-Out Cross-Validation (LOPO-CV) lintas 24 repositori Java open-source.
- **Hardware Acceleration**: GPU NVIDIA GeForce RTX 4050 Laptop (6GB VRAM, CUDA 13.4) & CPU multi-threading.
- **Dataset**: Dataset empiris FlakeFlagger (`test_features.csv`, `test_results.csv`, `Project_Info.csv`).
- **Referensi Kode Asli**: [`../../FlakeFlagger/`](../../FlakeFlagger/)
- **Total Pipelines**: **48 Pipelines** ($2 \times 3 \times 4 \times 2$):
  - **Data Cleaning (2)**: C1 (Row-Level Drop) vs C2 (Project-Level Drop)
  - **Feature Scaling (3)**: S1 (StandardScaler), S2 (MinMaxScaler), S3 (RobustScaler)
  - **Imbalance Handling (4)**: B1 (Baseline/None), B2 (SMOTE), B3 (Random Under-Sampling), B4 (Cost-Sensitive/Class Weight)
  - **Classifiers (2)**: M1 (Random Forest) vs M2 (XGBoost GPU)

---

## 🗂️ Struktur Direktori

```
Eksperimen_Bima/Eksperimen-1/
├── README.md                                 # Dokumentasi ikhtisar & petunjuk eksekusi
├── PROMPT_EKSEKUSI_EKSPERIMEN_1.md          # Master prompt komprehensif untuk memulai riset
├── configs/
│   └── experiment_matrix.json                # Definisi parameter 48 konfigurasi pipeline
├── src/
│   ├── __init__.py
│   ├── check_gpu.py                          # Verifikasi GPU RTX 4050 & CUDA
│   ├── data_loader.py                        # Modul ingestion dataset & cleaning C1/C2
│   ├── pipeline_builder.py                   # Modul scaler, sampler, dan classifier
│   ├── lopo_runner.py                        # Engine eksekusi LOPO-CV 48 pipelines
│   ├── statistical_tests.py                  # Uji Friedman & Nemenyi post-hoc CD
│   └── visualizer.py                         # Pembuatan diagram, boxplot, dan heatmap
├── notebooks/
│   └── 01_run_main_48_pipelines.ipynb        # Notebook interaktif untuk eksekusi terpadu
└── results/
    ├── raw_predictions/                      # Log prediksi mentah per target project & fold
    ├── summary_metrics_48_pipelines.csv      # Rekapitulasi metrik (PR-AUC, F1, MCC, ROC-AUC, dll.)
    ├── friedman_nemenyi_results.csv          # Hasil uji statistik signifikansi
    └── plots/                                # Boxplot, heatmap F1, dan diagram Critical Difference
```

---

## ⚡ Langkah Eksekusi Menggunakan Virtual Environment (.venv)

1. **Aktivasi Lingkungan Virtual `.venv`**:
   - PowerShell:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - Command Prompt (cmd):
     ```cmd
     .\.venv\Scripts\activate.bat
     ```

2. **Verifikasi Ketersediaan GPU & Pustaka ML**:
   ```powershell
   & ".\.venv\Scripts\python.exe" src/check_gpu.py
   ```

3. **Jalankan Pipeline Eksperimen**:
   - Melalui skrip runner:
     ```powershell
     & ".\.venv\Scripts\python.exe" src/lopo_runner.py
     ```
   - Atau melalui Jupyter Notebook:
     Buka [`notebooks/01_run_main_48_pipelines.ipynb`](notebooks/01_run_main_48_pipelines.ipynb) dan pilih kernel **Python (.venv)**.

4. **Analisis Jawaban RQ**:
   Buka file laporan sintesis akademik pada `LAPORAN_EKSPERIMEN_1.md`.
