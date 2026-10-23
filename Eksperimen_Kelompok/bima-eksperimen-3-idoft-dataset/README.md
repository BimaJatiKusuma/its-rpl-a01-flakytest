# 🔬 Replikasi Eksperimen 3: Repeated LOPO-CV pada Dataset IDoFT (Afeltra et al., 2024)

Repositori ini memuat implementasi teknis lengkap, eksekusi empiris berulang, dan analisis komprehensif dari **Replikasi Eksperimen 3** yang menggunakan **Dataset IDoFT (Illinois Dataset of Flaky Tests)** dalam skenario prediksi uji flaky lintas proyek (*Cross-Project Flaky Test Prediction*), sebagaimana diteliti dalam paper referensi utama:
> **Afeltra, A. et al. (IEEE Access 2024)** — *A Large-Scale Empirical Investigation Into Cross-Project Flaky Test Prediction*.

---

## 📋 Ikhtisar & Latar Belakang Riset

Pada [Eksperimen 3 Eksperimen_Bima](../../Eksperimen_Bima/Eksperiment-3/), 24 konfigurasi pipeline dievaluasi menggunakan *Leave-One-Project-Out Cross-Validation (LOPO-CV)* berulang kali (5 *independent random seeds*) pada dataset FlakeFlagger (22.236 baris data, 24 proyek Java, 811 flaky tests).

Replikasi ini dirancang untuk menjawab pertanyaan fundamental:
> **"Apakah temuan stabilitas, konsistensi stokastik, dominasi Random Under-Sampling (RUS), serta karakteristik sinkhole non-deteksi pada Eksperimen 3 tetap konsisten ketika diuji pada kurasi dataset yang berbeda, yaitu Dataset IDoFT?"**

### Perbandingan Karakteristik Dataset:
| Parameter Dataset | Eksperimen 3 (FlakeFlagger Set) | Replikasi Eksperimen 3 (IDoFT Set) |
|---|---|---|
| **Sumber Acuan** | Alshammari et al. (ICSE 2021) | Afeltra et al. (IEEE Access 2024) / IDoFT |
| **Total Instans Data** | 22.236 test cases | 12.626 test cases |
| **Jumlah Proyek Java** | 24 Proyek | 17 Proyek |
| **Flaky Tests Ground Truth**| 811 flaky tests (3,65%) | 264 flaky tests (2,09%) |
| **Tingkat Ketimpangan** | Sangat Timpang (~1 : 27) | Ekstrem Timpang (~1 : 47) |
| **Fitur Prediktif** | 23 Fitur (Smells, Coverage, Churn) | 23 Fitur (Smells, Coverage, Churn) |
| **Missing Values** | 0 (Group-aware median imputed) | 0 (Cleaned & imputed) |
| **Total Evaluasi Model** | 2.880 Model Folds | 2.040 Model Folds |

---

## ⚙️ Desain Eksperimen: Repeated LOPO-CV (2.040 Evaluasi Model)

Eksperimen menerapkan pengujian berulang sebanyak **5 Repetisi Independen** dengan variasi *random seeds*:
$$\text{Seeds} = [42, 100, 2024, 777, 999]$$

| Dimensi Eksperimen | Parameter & Konfigurasi |
|---|---|
| **Dataset Dasar** | `data/cleaned_idoft_test_features.csv` (12.626 baris, 17 proyek Java, 264 flaky test). |
| **Feature Scaling (3)** | **S1**: StandardScaler \| **S2**: MinMaxScaler \| **S3**: RobustScaler (Fit HANYA pada train). |
| **Imbalance Handling (4)** | **B1**: Baseline (None) \| **B2**: SMOTE \| **B3**: RUS (Random Under-Sampling) \| **B4**: ClassWeight. |
| **Classifier Models (2)** | **M1**: Random Forest CPU (`n_jobs=-1`) \| **M2**: XGBoost GPU CUDA (hist method). |
| **Skema Validasi** | **Repeated LOPO-CV**: 5 Repetisi $\times$ 24 Pipelines $\times$ 17 Folds = **2.040 Model Folds**. |
| **Metrik Evaluasi** | PR-AUC, ROC-AUC, F1-Score, Recall, Precision, MCC, Balanced Accuracy, Confusion Matrix. |
| **Ukuran Konsistensi** | Mean, Standard Deviation ($\sigma$), Coefficient of Variation ($CV = \frac{\sigma}{\mu} \times 100\%$), 95% Confidence Interval ($CI_{95}$). |

---

## 🗂️ Struktur Direktori

```
Eksperimen_Kelompok/bima-eksperimen-3-idoft-dataset/
├── README.md                                    # Dokumentasi ikhtisar & petunjuk eksekusi
├── RANCANGAN_PENELITIAN_EKSPERIMEN_3_IDOFT.md   # Dokumen akademik metodologi penelitian lengkap
├── LAPORAN_EKSPERIMEN_3_IDOFT.md               # Laporan komprehensif temuan empiris & analisa dataset
├── configs/
│   └── experiment_matrix.json                   # Konfigurasi parameter 24 pipelines & 5 random seeds
├── data/
│   ├── generate_idoft_dataset.py                # Skrip ingestion & matching dataset IDoFT
│   ├── cleaned_idoft_test_features.csv          # Dataset master IDoFT (12.626 baris, 17 proyek)
│   └── dataset_metadata.json                    # Demografi proyek & statistik flakiness
├── src/
│   ├── __init__.py
│   ├── check_gpu.py                             # Verifikasi GPU RTX 4050 & library ML
│   ├── data_loader.py                           # Loader dataset cleaned_idoft_test_features.csv
│   ├── pipeline_builder.py                      # Pembangun pipeline dinamis berbasis random_state
│   ├── repeated_lopo_runner.py                  # Runner 5 repetisi LOPO-CV (2.040 evaluasi model)
│   ├── consistency_analyzer.py                  # Analisis variansi, CV %, dan interval kepercayaan
│   ├── dataset_profiler.py                      # Profiling 17 dataset & root cause analysis akurasi
│   ├── statistical_tests.py                     # Uji Friedman & Nemenyi post-hoc lintas repetisi
│   └── visualizer.py                            # Generator grafik publikasi resolusi tinggi (300 DPI)
├── notebooks/
│   ├── build_notebook.py                        # Skrip pembuat notebook otomatis
│   └── 01_run_experiment_3_idoft_repeated_lopo.ipynb # Notebook interaktif eksekusi eksperimen
└── results/
    ├── raw_predictions/                         # Log prediksi fold per pipeline & per seed (CSV)
    ├── per_seed_summaries/                      # Ringkasan metrik per repetisi seed
    ├── all_repeated_folds_predictions.csv       # Master CSV 2.040 evaluasi fold
    ├── all_seed_pipeline_runs.csv               # Data mentah 120 run per pipeline lintas seed
    ├── summary_metrics_repeated_24_pipelines.csv# Rata-rata & std lintas repetisi per pipeline
    ├── consistency_stability_analysis.csv       # Tabel stabilitas & CV % ke-24 pipeline
    ├── consistency_factor_scaling.csv           # Analisis faktor feature scaling
    ├── consistency_factor_imbalance.csv         # Analisis faktor penanganan ketimpangan
    ├── consistency_factor_model.csv             # Analisis faktor algoritma classifier
    ├── dataset_capability_profile.csv           # Profil mendalam ke-17 proyek target IDoFT
    ├── matrix_pipeline_vs_dataset_f1.csv        # Matriks 24 Pipeline x 17 Proyek F1 & rata-rata
    ├── macro_vs_weighted_f1_explanation.csv     # Komparasi Macro vs Active vs Weighted vs Micro F1
    ├── friedman_nemenyi_f1_repeated_results.csv # Hasil uji signifikansi statistik F1
    ├── friedman_nemenyi_prauc_repeated_results.csv # Hasil uji signifikansi statistik PR-AUC
    └── plots/                                   # 6 visualisasi ilmiah publikasi 300 DPI
```

---

## ⚡ Petunjuk Menjalankan Eksperimen

1. **Aktivasi Virtual Environment**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Verifikasi Lingkungan ML & Akselerasi GPU**:
   ```powershell
   & ".\.venv\Scripts\python.exe" Eksperimen_Kelompok/bima-eksperimen-3-idoft-dataset/src/check_gpu.py
   ```

3. **Eksekusi Pengujian Berulang (Repeated LOPO-CV - 2.040 Folds)**:
   ```powershell
   & ".\.venv\Scripts\python.exe" Eksperimen_Kelompok/bima-eksperimen-3-idoft-dataset/src/repeated_lopo_runner.py
   ```

4. **Eksekusi Analisis Konsistensi, Profiling Dataset, & Uji Statistik**:
   ```powershell
   & ".\.venv\Scripts\python.exe" Eksperimen_Kelompok/bima-eksperimen-3-idoft-dataset/src/consistency_analyzer.py
   & ".\.venv\Scripts\python.exe" Eksperimen_Kelompok/bima-eksperimen-3-idoft-dataset/src/dataset_profiler.py
   & ".\.venv\Scripts\python.exe" Eksperimen_Kelompok/bima-eksperimen-3-idoft-dataset/src/statistical_tests.py
   & ".\.venv\Scripts\python.exe" Eksperimen_Kelompok/bima-eksperimen-3-idoft-dataset/src/visualizer.py
   ```
