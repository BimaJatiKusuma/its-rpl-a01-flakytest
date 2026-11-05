# 🔬 Eksperimen Gabungan 1 & 2: Cross-Project Flaky Test Prediction
## Unified Full Factorial Matrix (72 Pipelines) & 10x Repeated LOPO-CV

Repositori ini merupakan penyatuan terpadu dan penyempurnaan dari **Eksperimen 1** dan **Eksperimen 2**, yang mengevaluasi secara komprehensif **Tri-Paradigma Data Cleaning (C1: Row Drop vs C2: Project Drop vs C3: Group-Aware Median Imputation)** dalam satu kerangka kerja validasi *Leave-One-Project-Out Cross-Validation (LOPO-CV)* yang dilengkapi dengan **Framework Uji Konsistensi 10x Random Seeds**.

---

## 📋 Ikhtisar Matriks Faktorial 72 Pipelines

Matriks eksperimen gabungan dibentuk dari 4 faktor variabel bebas ($3 \times 3 \times 4 \times 2 = \mathbf{72 \text{ Pipelines}}$):

| Faktor | Variasi | Deskripsi Operasional |
|---|---|---|
| **Data Cleaning (3)** | **C1**: Row-Level Drop (24 proyek, 21.963 baris)<br>**C2**: Project-Level Drop (19 proyek, 16.458 baris)<br>**C3**: Group-Aware Median Imputation (24 proyek, 22.236 baris, 100% utuh) | Membandingkan dampak penghapusan baris vs penghapusan proyek vs preservasi 100% data via imputasi cerdas. |
| **Feature Scaling (3)** | **S1**: StandardScaler<br>**S2**: MinMaxScaler<br>**S3**: RobustScaler | Di-fit **HANYA pada fold latih ($N-1$ proyek)** dan di-transformasikan ke fold target uji (Zero Data Leakage). |
| **Imbalance Handling (4)** | **B1**: Baseline (None)<br>**B2**: SMOTE<br>**B3**: Random Under-Sampling (RUS)<br>**B4**: ClassWeight (Cost-Sensitive) | Diterapkan **HANYA pada fold latih** untuk mencegah kebocoran distribusi target. |
| **Classifier Model (2)** | **M1**: Random Forest CPU (`n_estimators=100`, `n_jobs=-1`)<br>**M2**: XGBoost GPU CUDA (`device="cuda"`, `tree_method="hist"`) | Menguji bagging ensemble vs gradient boosting bertenaga GPU. |

---

## 🗂️ Struktur Direktori

```
Eksperimen-Gabungan_1_dan_2/
├── README.md                                 # Dokumentasi ikhtisar & petunjuk eksekusi
├── RANCANGAN_PENELITIAN_GABUNGAN.md          # Dokumen akademik rancangan penelitian komprehensif
├── LAPORAN_EKSPERIMEN_GABUNGAN.md            # Laporan komprehensif hasil temuan empiris
├── requirements.txt                          # Dependensi paket Python yang dibutuhkan
├── configs/
│   └── experiment_matrix.json                # Definisi parameter 72 konfigurasi pipeline
├── data/
│   ├── README.md                             # Dokumentasi kamus data & metodologi tri-paradigma
│   ├── test_features.csv                     # Dataset mentah FlakeFlagger (untuk C1 & C2)
│   ├── cleaned_test_features.csv             # Dataset bersih terimputasi (untuk C3)
│   ├── test_results.csv                      # Data eksekusi rerun 10.000x per test case
│   └── Project_Info.csv                      # Metadata 24 repositori Java open-source
├── src/
│   ├── __init__.py
│   ├── check_gpu.py                          # Verifikasi GPU RTX 4050 & lingkungan ML
│   ├── data_loader.py                        # Loader terpadu untuk C1, C2, dan C3
│   ├── pipeline_builder.py                   # Generator pipeline scaler + sampler + classifier (72 specs)
│   ├── lopo_runner.py                        # Engine eksekusi LOPO-CV 72 pipelines (single-run)
│   ├── repeated_lopo_runner.py               # Engine Repeated LOPO-CV (hingga 10x Random Seeds)
│   ├── consistency_analyzer.py               # Analisis konsistensi & stabilitas empiris (CV%, CI95)
│   ├── statistical_tests.py                  # Uji signifikansi statistik Friedman, Nemenyi & Kruskal-Wallis
│   └── visualizer.py                         # Pembuat boxplot faktor, tri-paradigma, dan heatmap
├── notebooks/
│   ├── 01_run_unified_72_pipelines.ipynb     # Jupyter Notebook interaktif eksekusi single-run 72 pipelines
│   └── 02_run_repeated_lopo_10x_and_consistency.ipynb # Jupyter Notebook interaktif eksekusi 10x perulangan
└── results/
    ├── raw_predictions/                      # Log prediksi mentah per fold & per seed
    ├── per_seed_summaries/                   # Ringkasan agregat per seed
    ├── all_seed_pipeline_runs.csv            # Master seluruh runs lintas repetisi
    ├── consistency_stability_analysis.csv    # Rekapitulasi Mean, Std, CV%, CI95 per pipeline
    ├── consistency_factor_*.csv              # Analisis stabilitas per faktor (Cleaning, Scaling, dll.)
    ├── summary_metrics_72_pipelines.csv      # Ringkasan single-run 72 pipelines
    └── plots/                                # Boxplot faktor dan tri-paradigma cleaning
```

---

## ⚡ Panduan Eksekusi

### 1. Eksekusi Single-Run (72 Pipelines Faktorial):
- **Smoke Test Cepat (2 Folds)**:
  ```powershell
  python src/lopo_runner.py --quick_test
  ```
- **Eksekusi Penuh Seluruh 72 Pipelines**:
  ```powershell
  python src/lopo_runner.py --all
  ```
- **Eksekusi 1 Pipeline Spesifik**:
  ```powershell
  python src/lopo_runner.py --pipeline_id P61_C3_S2_B3_M1
  ```

### 2. Eksekusi Repeated Trials (Uji Konsistensi 10x Seeds):
- **Smoke Test Cepat (3 Pipelines C1/C2/C3, 2 Seeds, 2 Folds)**:
  ```powershell
  python src/repeated_lopo_runner.py --quick_test
  ```
- **Top 12 Pipelines Lintas 10 Seeds (Rekomendasi Cepat & Akurat)**:
  ```powershell
  python src/repeated_lopo_runner.py --n_repeats 10 --top_pipelines 12
  ```
- **Uji 1 Strategi Cleaning Spesifik Lintas 10 Seeds (misal C3)**:
  ```powershell
  python src/repeated_lopo_runner.py --n_repeats 10 --cleaning C3
  ```
- **Eksekusi Penuh Seluruh 72 Pipelines Lintas 10 Seeds**:
  ```powershell
  python src/repeated_lopo_runner.py --n_repeats 10 --all
  ```

### 3. Analisis Konsistensi & Uji Statistik:
```powershell
python src/consistency_analyzer.py
python src/statistical_tests.py
python src/visualizer.py
```

### 4. Eksekusi Interaktif melalui Jupyter Notebook:
- Buka [`notebooks/01_run_unified_72_pipelines.ipynb`](notebooks/01_run_unified_72_pipelines.ipynb) untuk single-run 72 pipelines.
- Buka [`notebooks/02_run_repeated_lopo_10x_and_consistency.ipynb`](notebooks/02_run_repeated_lopo_10x_and_consistency.ipynb) untuk uji konsistensi 10x perulangan.
