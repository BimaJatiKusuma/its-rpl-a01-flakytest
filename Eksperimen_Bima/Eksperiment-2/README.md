# 🔬 Eksperimen 2: Cross-Project Flaky Test Prediction pada Dataset Terimputasi (Full Factorial 24 Pipelines)

Repositori ini berisi perancangan, implementasi teknis, dan eksekusi empiris dari **Eksperimen 2**, yang melanjutkan dan menyempurnakan temuan dari [Eksperimen 1](../Eksperimen-1/), kini dilengkapi dengan **Framework Uji Konsistensi 10x Perulangan (Repeated LOPO-CV lintas 10 Random Seeds)**.

---

## 📋 Ikhtisar & Latar Belakang Perubahan dari Eksperimen 1

Pada [Eksperimen 1](../Eksperimen-1/), dua strategi pembersihan data diuji untuk mengatasi 273 baris missing value pada dataset mentah:
1. **C1 (Row-Level Drop)**: Membuang 273 baris (tersisa 21.963 baris data, 24 proyek).
2. **C2 (Project-Level Drop)**: Membuang seluruh 5 proyek yang memiliki missing value (`wildfly`, `logback`, `spring-boot`, `wro4j`, `handlebars.java`), menyisakan 19 proyek (16.458 baris data). Dampak fatalnya: **222 flaky test riil terbuang (termasuk 163 flaky test dari `spring-boot`, yaitu ~20% dari seluruh flaky test)**.

### Mengapa Konsep C1 & C2 Tidak Diperlukan di Eksperimen 2?
Pada **Eksperimen 2**, kita menggunakan dataset yang telah melalui preprocessing tingkat lanjut:
`data/cleaned_test_features.csv`.
- **0 Missing Values**: Seluruh 273 missing values telah diimputasi secara cerdas menggunakan *Group-Aware Per-Project Median & Probabilistic Imputation*.
- **100% Data Preservation**: Seluruh **22.236 baris** dan seluruh **24 proyek Java** dipertahankan utuh.
- **811 Flaky Test Terjaga**: Tidak ada proyek maupun sampel flaky yang dibuang.
- **Konsep C1 dan C2 Tereliminasi**: Karena dataset sudah bersih dan lengkap, faktor manipulasi cleaning tidak lagi diperlukan, sehingga matriks eksperimen difokuskan pada perbandingan murni: **Feature Scaling $\times$ Imbalance Handling $\times$ Classifier Models** pada 24 proyek lengkap.
- **Uji Konsistensi (Repeated Trials)**: **10 Random Seeds Independen** (`[42, 100, 2024, 777, 999, 123, 456, 789, 2025, 2026]`) untuk menguji variansi stokastik dan stabilitas metrik (Mean, Std Deviasi, CV%, dan 95% Confidence Interval).

---

## ⚙️ Matriks Faktorial 24 Pipelines Eksperimen 2

Eksperimen 2 mengevaluasi $3 \times 4 \times 2 = \mathbf{24\text{ Pipelines}}$ secara Leave-One-Project-Out Cross-Validation (LOPO-CV):

| Faktor | Notasi & Variasi | Deskripsi Operasional |
|---|---|---|
| **Dataset Dasar** | **Cleaned Dataset** | `cleaned_test_features.csv` (22.236 baris, 24 proyek, 811 flaky). |
| **Feature Scaling (3)** | **S1**: StandardScaler<br>**S2**: MinMaxScaler<br>**S3**: RobustScaler | Di-fit **HANYA pada data latih** ($N-1$ proyek) dan di-transform ke target fold (Zero Data Leakage). |
| **Imbalance Handling (4)** | **B1**: Baseline (None)<br>**B2**: SMOTE<br>**B3**: RUS (Random Under-Sampling)<br>**B4**: ClassWeight (Cost-Sensitive) | Diterapkan **HANYA pada data latih** ($N-1$ proyek) untuk mencegah kebocoran distribusi target. |
| **Classifier Model (2)** | **M1**: Random Forest<br>**M2**: XGBoost GPU | M1: `n_estimators=100`, CPU multi-threaded (`n_jobs=-1`).<br>M2: `n_estimators=100`, `max_depth=6`, GPU accelerated (`device='cuda'`, `tree_method='hist'`). |

Total Evaluasi Per Repetisi: $24 \text{ Pipelines} \times 24 \text{ Folds} = \mathbf{576\text{ Model Folds}}$.  
Total Evaluasi 10x Repetisi: $10 \times 576 = \mathbf{5.760\text{ Model Folds}}$.

---

## 🗂️ Struktur Direktori Eksperimen 2

```
Eksperimen_Bima/Eksperiment-2/
├── README.md                                 # Dokumentasi ikhtisar & petunjuk eksekusi
├── PROMPT_EKSEKUSI_EKSPERIMEN_2.md          # Master prompt AI untuk eksekusi mandiri
├── RANCANGAN_PENELITIAN_EKSPERIMEN_2.md      # Dokumen akademik rancangan penelitian komprehensif
├── LAPORAN_EKSPERIMEN_2.md                   # Laporan komprehensif hasil temuan empiris
├── configs/
│   └── experiment_matrix.json                # Definisi parameter 24 konfigurasi pipeline
├── data/
│   ├── README.md                             # Dokumentasi dataset & metodologi imputasi
│   └── cleaned_test_features.csv             # Dataset bersih (22.236 baris, 0 missing, 24 proyek utuh)
├── src/
│   ├── __init__.py
│   ├── check_gpu.py                          # Verifikasi GPU RTX 4050 & lingkungan ML
│   ├── data_loader.py                        # Loader dataset cleaned_test_features.csv
│   ├── generate_cleaned_dataset.py           # Script penghasil cleaned_test_features.csv mandiri
│   ├── build_full_notebook.py                # Script pembuat notebook imputasi & EDA Phase 1
│   ├── pipeline_builder.py                   # Generator pipeline scaler + sampler + classifier
│   ├── lopo_runner.py                        # Engine eksekusi LOPO-CV (single run)
│   ├── repeated_lopo_runner.py               # [BARU] Engine Repeated LOPO-CV (hingga 10x Seeds)
│   ├── consistency_analyzer.py               # [BARU] Analisis konsistensi & stabilitas (CV, CI95)
│   ├── statistical_tests.py                  # Uji signifikansi statistik Friedman & Nemenyi post-hoc CD
│   └── visualizer.py                         # Pembuat boxplot faktor dan heatmap performa
├── notebooks/
│   ├── 00_eda_and_data_cleaning.ipynb        # Jupyter Notebook interaktif Phase 1 (Data Cleaning & EDA)
│   ├── 01_run_main_24_pipelines.ipynb        # Jupyter Notebook interaktif eksekusi single-run
│   └── 02_run_repeated_lopo_10x_and_consistency.ipynb # [BARU] Notebook eksekusi 10x perulangan
└── results/
    ├── raw_predictions/                      # Log prediksi mentah per fold & seed
    ├── per_seed_summaries/                   # Ringkasan agregat per seed
    ├── all_seed_pipeline_runs.csv            # Master seluruh runs lintas repetisi
    ├── consistency_stability_analysis.csv    # Rekapitulasi Mean, Std, CV%, CI95 per pipeline
    ├── consistency_factor_*.csv              # Analisis stabilitas per faktor (Scaling, Imbalance, Model)
    ├── summary_metrics_24_pipelines.csv      # Ringkasan single-run awal
    └── plots/                                # Boxplot faktor dan heatmap visualisasi
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
  python src/repeated_lopo_runner.py --pipeline_id P13_S2_B3_M1 --n_repeats 10
  ```
- **Mode Penuh (Seluruh 24 Pipelines × 10 Seeds)**:
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
