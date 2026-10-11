# 🔬 Eksperimen 2: Cross-Project Flaky Test Prediction pada Dataset Terimputasi (Full Factorial 24 Pipelines)

Repositori ini berisi perancangan, implementasi teknis, dan eksekusi empiris dari **Eksperimen 2**, yang melanjutkan dan menyempurnakan temuan dari [Eksperimen 1](../Eksperimen-1/).

---

## 📋 Ikhtisar & Latar Belakang Perubahan dari Eksperimen 1

Pada [Eksperimen 1](../Eksperimen-1/), dua strategi pembersihan data diuji untuk mengatasi 273 baris missing value pada dataset mentah:
1. **C1 (Row-Level Drop)**: Membuang 273 baris (tersisa 21.963 baris data, 24 proyek).
2. **C2 (Project-Level Drop)**: Membuang seluruh 5 proyek yang memiliki missing value (`wildfly`, `logback`, `spring-boot`, `wro4j`, `handlebars.java`), menyisakan 19 proyek (16.458 baris data). Dampak fatalnya: **222 flaky test riil terbuang (termasuk 163 flaky test dari `spring-boot`, yaitu ~20% dari seluruh flaky test)**.

### Mengapa Konsep C1 & C2 Tidak Diperlukan di Eksperimen 2?
Pada **Eksperimen 2**, kita menggunakan dataset yang telah melalui preprocessing tingkat lanjut oleh Ali:
`Experiment_Ali/data/cleaned_test_features.csv`.
- **0 Missing Values**: Seluruh 273 missing values telah diimputasi secara cerdas menggunakan *Group-Aware Per-Project Median & Probabilistic Imputation*.
- **100% Data Preservation**: Seluruh **22.236 baris** dan seluruh **24 proyek Java** dipertahankan utuh.
- **811 Flaky Test Terjaga**: Tidak ada proyek maupun sampel flaky yang dibuang.
- **Konsep C1 dan C2 Tereliminasi**: Karena dataset sudah bersih dan lengkap, faktor manipulasi cleaning tidak lagi diperlukan, sehingga matriks eksperimen difokuskan pada perbandingan murni: **Feature Scaling $\times$ Imbalance Handling $\times$ Classifier Models** pada 24 proyek lengkap.

---

## ⚙️ Matriks Faktorial 24 Pipelines Eksperimen 2

Eksperimen 2 mengevaluasi $3 \times 4 \times 2 = \mathbf{24\text{ Pipelines}}$ secara Leave-One-Project-Out Cross-Validation (LOPO-CV):

| Faktor | Notasi & Variasi | Deskripsi Operasional |
|---|---|---|
| **Dataset Dasar** | **Cleaned Dataset** | `Experiment_Ali/data/cleaned_test_features.csv` (22.236 baris, 24 proyek, 811 flaky). |
| **Feature Scaling (3)** | **S1**: StandardScaler<br>**S2**: MinMaxScaler<br>**S3**: RobustScaler | Di-fit **HANYA pada data latih** ($N-1$ proyek) dan di-transform ke target fold (Zero Data Leakage). |
| **Imbalance Handling (4)** | **B1**: Baseline (None)<br>**B2**: SMOTE<br>**B3**: RUS (Random Under-Sampling)<br>**B4**: ClassWeight (Cost-Sensitive) | Diterapkan **HANYA pada data latih** ($N-1$ proyek) untuk mencegah kebocoran distribusi target. |
| **Classifier Model (2)** | **M1**: Random Forest<br>**M2**: XGBoost GPU | M1: `n_estimators=100`, CPU multi-threaded (`n_jobs=-1`).<br>M2: `n_estimators=100`, `max_depth=6`, GPU accelerated (`device='cuda'`, `tree_method='hist'`). |

Total Evaluasi: $24 \text{ Pipelines} \times 24 \text{ Folds} = \mathbf{576\text{ Model Folds}}$.

---

## 🎯 4 Research Questions (RQ) Eksperimen 2

1. **RQ1 (Dampak Imputasi Cerdas vs Pembuangan Data C1/C2)**:  
   *Bagaimana dampak penggunaan dataset terimputasi lengkap (24 proyek, 22.236 baris, 811 flaky test) terhadap kemampuan generalisasi model lintas proyek jika dibandingkan dengan strategi Row-Level Drop (C1) dan Project-Level Drop (C2) pada Eksperimen 1?*
2. **RQ2 (Efektivitas Feature Scaling pada Data Terimputasi)**:  
   *Di antara StandardScaler (S1), MinMaxScaler (S2), dan RobustScaler (S3), teknik scaling manakah yang paling efektif memitigasi domain shift pada fitur yang telah diimputasi?*
3. **RQ3 (Konsistensi Keunggulan Random Under-Sampling / RUS)**:  
   *Apakah Random Under-Sampling (B3: RUS) tetap mempertahankan dominasi mutlaknya (sebagaimana pada Eksperimen 1) dibandingkan SMOTE (B2), ClassWeight (B4), dan Baseline (B1) ketika seluruh 24 proyek dan 811 flaky test dilatih secara utuh?*
4. **RQ4 (Komparasi Ketangguhan Random Forest vs XGBoost GPU)**:  
   *Apakah XGBoost bertenaga GPU tetap mengungguli Random Forest dalam metrik operasional (F1-Score, Recall, Precision, MCC) pada populasi data penuh?*

---

## 🗂️ Struktur Direktori Eksperimen 2

```
Eksperimen_Bima/Eksperiment-2/
├── README.md                                 # Dokumentasi ikhtisar & petunjuk eksekusi
├── PROMPT_EKSEKUSI_EKSPERIMEN_2.md          # Master prompt AI untuk eksekusi mandiri
├── RANCANGAN_PENELITIAN_EKSPERIMEN_2.md      # Dokumen akademik rancangan penelitian komprehensif
├── configs/
│   └── experiment_matrix.json                # Definisi parameter 24 konfigurasi pipeline
├── src/
│   ├── __init__.py
│   ├── check_gpu.py                          # Verifikasi GPU RTX 4050 & lingkungan ML
│   ├── data_loader.py                        # Loader dataset cleaned_test_features.csv
│   ├── pipeline_builder.py                   # Generator pipeline scaler + sampler + classifier
│   ├── lopo_runner.py                        # Engine eksekusi LOPO-CV 24 pipelines lintas 24 fold
│   ├── statistical_tests.py                  # Uji signifikansi statistik Friedman & Nemenyi post-hoc CD
│   └── visualizer.py                         # Pembuat boxplot faktor dan heatmap performa
├── notebooks/
│   └── 01_run_main_24_pipelines.ipynb        # Jupyter Notebook interaktif eksekusi eksperimen
└── results/
    ├── raw_predictions/                      # CSV log metrik per fold proyek target
    ├── summary_metrics_24_pipelines.csv      # Ringkasan rata-rata, standar deviasi, dan median
    ├── friedman_nemenyi_f1_results.csv       # Peringkat signifikansi statistik F1-Score
    ├── friedman_nemenyi_prauc_results.csv    # Peringkat signifikansi statistik PR-AUC
    └── plots/                                # Boxplot faktor dan heatmap visualisasi
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
   & ".\.venv\Scripts\python.exe" Eksperimen_Bima/Eksperiment-2/src/check_gpu.py
   ```

3. **Jalankan Pipeline Eksperimen**:
   - Melalui skrip Python runner:
     ```powershell
     & ".\.venv\Scripts\python.exe" Eksperimen_Bima/Eksperiment-2/src/lopo_runner.py
     ```
   - Atau melalui Jupyter Notebook:
     Buka [`notebooks/01_run_main_24_pipelines.ipynb`](notebooks/01_run_main_24_pipelines.ipynb) dan pilih kernel **Python (.venv)**.

4. **Uji Statistik & Visualisasi**:
   ```powershell
   & ".\.venv\Scripts\python.exe" Eksperimen_Bima/Eksperiment-2/src/statistical_tests.py
   & ".\.venv\Scripts\python.exe" Eksperimen_Bima/Eksperiment-2/src/visualizer.py
   ```

---

## 🏆 Ringkasan Hasil Eksekusi & Top Pipelines

Seluruh 24 pipeline (576 model folds) telah selesai dieksekusi secara penuh. Laporan sintesis akademik lengkap dapat diakses pada:
👉 **[LAPORAN_EKSPERIMEN_2.md](LAPORAN_EKSPERIMEN_2.md)**

### Top 5 Pipelines Eksperimen 2 (Berdasarkan Mean F1-Score):
| Rank | Pipeline ID | Scaling | Imbalance | Model | Mean F1 | Mean Recall | Mean Precision | Mean PR-AUC | Mean MCC |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 **1** | `P13_S2_B3_M1` | **MinMaxScaler (S2)** | **RUS (B3)** | **Random Forest (M1)** | **0.1816** | **0.4374** | **0.2038** | **0.2210** | **0.1547** |
| 🥈 **2** | `P21_S3_B3_M1` | RobustScaler (S3) | **RUS (B3)** | Random Forest (M1) | **0.1786** | 0.4333 | 0.2014 | 0.2203 | 0.1509 |
| 🥉 **3** | `P05_S1_B3_M1` | StandardScaler (S1) | **RUS (B3)** | Random Forest (M1) | **0.1785** | 0.4345 | 0.2024 | **0.2224** | 0.1510 |
| **4** | `P06_S1_B3_M2` | StandardScaler (S1) | **RUS (B3)** | **XGBoost GPU (M2)** | **0.1403** | 0.4255 | 0.1352 | 0.2012 | 0.1011 |
| **5** | `P14_S2_B3_M2` | MinMaxScaler (S2) | **RUS (B3)** | **XGBoost GPU (M2)** | **0.1403** | 0.4255 | 0.1352 | 0.2012 | 0.1011 |

### Hasil Uji Signifikansi Statistik:
- **Friedman Test (F1-Score)**: $\chi^2 = 173.2039$, $p\text{-value} = 5.17 \times 10^{-25} \ll 0.05$ (Sangat Signifikan).
- **Nemenyi Critical Difference (CD)**: $\text{CD} = 7.4097$ ($\alpha=0.05$). Klaster teratas didominasi secara mutlak oleh Random Under-Sampling (`B3`).

