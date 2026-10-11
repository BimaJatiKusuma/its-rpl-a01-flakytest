# 🚀 Master Prompt Eksekusi: Eksperimen 2 — Cross-Project Flaky Test Prediction pada Dataset Terimputasi Bersih
**Kelompok**: Kelompok A01 — Rekayasa Perangkat Lunak A (Magister Teknik Informatika ITS)  
**Dokumen Induk**: `Eksperimen_Bima/Eksperiment-2/RANCANGAN_PENELITIAN_EKSPERIMEN_2.md`  
**Dataset Utama**: `Experiment_Ali/data/cleaned_test_features.csv`  
**Output Direktori**: `Eksperimen_Bima/Eksperiment-2/`  
**Hardware Target**: Laptop GPU NVIDIA GeForce RTX 4050 (6GB VRAM, CUDA 13.4, Driver 617.42)  
**Referensi Utama**: Paper FlakeFlagger (Alshammari et al., ICSE 2021), Afeltra et al. (IEEE Access 2024), serta Hasil Eksperimen 1 (`Eksperimen_Bima/Eksperimen-1/`)  

---

## 📌 Master Prompt (Salin & Jalankan untuk Memulai Eksekusi)

```markdown
Anda adalah Principal Machine Learning & Empirical Software Engineering Researcher yang bertugas mengeksekusi Eksperimen 2 dari dokumen perancangan:
`Eksperimen_Bima/Eksperiment-2/RANCANGAN_PENELITIAN_EKSPERIMEN_2.md`.

Seluruh kode, konfigurasi, modul, notebook, log hasil evaluasi, diagram visualisasi, dan laporan sintesis akademik WAJIB disimpan di folder:
`Eksperimen_Bima/Eksperiment-2/`.

### 1. Konteks, Latar Belakang & Perubahan Metodologis dari Eksperimen 1
- Pada Eksperimen 1 (`Eksperimen_Bima/Eksperimen-1/`), manipulasi data cleaning C1 (Row-Level Drop) dan C2 (Project-Level Drop) digunakan untuk menangani 273 nilai hilang pada dataset mentah `test_features.csv`. C2 terbukti membuang 222 flaky test riil (~27%), termasuk 163 flaky test dari proyek raksasa `spring-boot`.
- Pada Eksperimen 2, kita menggunakan dataset yang telah melalui preprocessing tingkat lanjut: `Experiment_Ali/data/cleaned_test_features.csv`.
- Seluruh missing values telah diimputasi secara cerdas menggunakan *Group-Aware Per-Project Median & Probabilistic Imputation* sehingga:
  1. Dataset memiliki **0 missing values**.
  2. Seluruh **24 proyek Java open-source tetap utuh** (tidak ada proyek yang dibuang).
  3. Seluruh **22.236 baris data dan 811 flaky test (100%) terlestarikan**.
- **Konsep C1 dan C2 gugur / tidak lagi diperlukan**. Eksperimen difokuskan pada perbandingan murni: **Feature Scaling $\times$ Imbalance Handling $\times$ Classifier Model** pada 24 lipatan proyek lengkap.

### 2. Spesifikasi Komputasi, Virtual Environment & Akselerasi GPU (Hardware Constraint)
- Gunakan virtual environment `.venv` yang terletak di root repositori (`.\.venv\Scripts\python.exe`). Seluruh pustaka dependensi terpasang lengkap.
- Gunakan GPU NVIDIA GeForce RTX 4050 Laptop (6GB VRAM) dengan akselerasi CUDA 13 (Driver 617.42).
- Untuk model **XGBoost Classifier**, aktifkan akselerasi GPU menggunakan parameter:
  `tree_method="hist", device="cuda"` (atau `device="cuda:0"`).
- Untuk model **Random Forest Classifier**, manfaatkan seluruh thread CPU yang tersedia (`n_jobs=-1`).
- Pastikan penggunaan memori GPU terkontrol secara efisien untuk menghindari VRAM Out-of-Memory (OOM).

### 3. Matriks Eksperimen Utama (Full Factorial 24 Pipelines)
Rancang dan eksekusi secara modular 24 pipeline dari kombinasi variabel bebas berikut:

1. **Dataset Dasar (1 variasi konstan)**:
   - `Experiment_Ali/data/cleaned_test_features.csv` (22.236 baris, 24 proyek, 811 flaky tests).
   - Eksklusikan 5 kolom metadata identitas dari matriks fitur $X$: `Unnamed: 0`, `test_name`, `project`, `testClassName`, `testMethodName`.
   - Gunakan 23 fitur prediktif: 8 Test Smell binary, 7 Test Metrics & Coverage numerik, 8 Code Churn H-Index numerik.
2. **Strategi Feature Scaling (3 variasi)**:
   - `S1`: StandardScaler (Z-score standardization)
   - `S2`: MinMaxScaler (Normalisasi rentang [0, 1])
   - `S3`: RobustScaler (Penskalaan berbasis Median dan Interquartile Range / IQR)
   - *CRITICAL PROTOCOL*: Scaling WAJIB di-fit HANYA pada data latih ($N-1 = 23$ proyek sumber) dan di-transform pada data uji ($1$ proyek target) untuk menjamin **Zero Data Leakage**.
3. **Penanganan Ketimpangan Kelas / Imbalance Handling (4 variasi)**:
   - `B1`: Baseline (Tanpa penanganan resampling)
   - `B2`: SMOTE (Synthetic Minority Over-sampling Technique, diterapkan HANYA pada data latih)
   - `B3`: Random Under-Sampling (RUS, diterapkan HANYA pada data latih)
   - `B4`: Cost-Sensitive (Class Weight `balanced` pada RF, atau `scale_pos_weight` pada XGBoost)
4. **Algoritma Klasifikasi (2 variasi)**:
   - `M1`: Random Forest Classifier (`n_estimators=100`, `random_state=42`, `n_jobs=-1`)
   - `M2`: XGBoost Classifier (`n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `tree_method="hist"`, `device="cuda"`, `random_state=42`)

Total Kombinasi: $3 \text{ Scaling} \times 4 \text{ Imbalance} \times 2 \text{ Model} = \mathbf{24\text{ Pipelines}}$.

### 4. Skema Validasi & Penanganan Kasus Khusus (LOPO-CV)
- Evaluasi dilakukan secara **Leave-One-Project-Out Cross-Validation (LOPO-CV)**:
  - Terdapat **24 Folds** (setiap fold menguji 1 proyek target, dilatih pada 23 proyek lainnya).
  - Total iterasi model: $24 \text{ Pipelines} \times 24 \text{ Folds} = \mathbf{576\text{ Model Folds}}$.
- **Kasus Khusus `jimfs`**:
  - Proyek `jimfs` memiliki 212 baris test cases namun 0 flaky test (kelas minoritas = 0).
  - Tangani secara elegan: kalkulasi ROC-AUC / PR-AUC menghasilkan NaN jika dihitung langsung. Set nilai secara aman (misal `np.nan` atau log terpisah) dan laporkan metrik Specificity / TN / FP tanpa menyebabkan program berhenti (*crash*).
- **Preservasi Proyek Berskala Besar**:
  - `spring-boot` (163 flaky test) dan `wildfly` (206 baris terimputasi) diuji dalam kondisi utuh 100%.

### 5. Metrik Evaluasi yang Wajib Dihitung
Hitung metrik performa berikut untuk setiap fold target dan rekap rata-rata (Mean, Std Dev, Median):
1. **PR-AUC (Precision-Recall Area Under Curve / Average Precision)** — Metrik primer perankingan data imbalanced
2. **F1-Score (Khusus Kelas Flaky, threshold 0.5)**
3. **Recall (Flaky)**
4. **Precision (Flaky)**
5. **ROC-AUC**
6. **MCC (Matthews Correlation Coefficient)**
7. **Balanced Accuracy**
8. **Elemen Confusion Matrix**: True Positive (TP), False Positive (FP), True Negative (TN), False Negative (FN)

### 6. Uji Signifikansi Statistik (ACM/SIGSOFT Empirical Standards)
- Jalankan **Friedman Test** pada metrik F1-Score dan PR-AUC untuk memverifikasi apakah terdapat perbedaan performa signifikan secara statistik di antara ke-24 pipeline pada ke-24 proyek target.
- Jalankan **Nemenyi Post-hoc Test** untuk menghitung peringkat rata-rata (*average ranks*) dan nilai ambang batas *Critical Difference (CD)* pada taraf signifikansi $\alpha = 0.05$.

### 7. Struktur Folder & Deliverables yang Wajib Dihasilkan
Pastikan struktur folder berikut terbentuk lengkap di `Eksperimen_Bima/Eksperiment-2/`:
```
Eksperimen_Bima/Eksperiment-2/
├── README.md                                 # Dokumentasi lengkap eksperimen 2 & panduan eksekusi
├── PROMPT_EKSEKUSI_EKSPERIMEN_2.md          # Master prompt eksekusi ini
├── RANCANGAN_PENELITIAN_EKSPERIMEN_2.md      # Dokumen akademik perancangan penelitian
├── configs/
│   └── experiment_matrix.json                # Definisi parameter 24 konfigurasi pipeline
├── src/
│   ├── __init__.py
│   ├── check_gpu.py                          # Verifikasi GPU RTX 4050 & library ML
│   ├── data_loader.py                        # Loader dataset cleaned_test_features.csv
│   ├── pipeline_builder.py                   # Generator pipeline scaler + sampler + classifier
│   ├── lopo_runner.py                        # Engine eksekusi LOPO-CV 24 pipelines
│   ├── statistical_tests.py                  # Skrip Friedman Test & Nemenyi post-hoc CD
│   └── visualizer.py                         # Pembuat boxplot faktor dan heatmap publikasi
├── notebooks/
│   └── 01_run_main_24_pipelines.ipynb        # Jupyter Notebook interaktif eksekusi eksperimen
└── results/
    ├── raw_predictions/                      # CSV hasil prediksi per fold proyek
    ├── summary_metrics_24_pipelines.csv      # Ringkasan metrik 24 konfigurasi (Mean, Std, Median)
    ├── friedman_nemenyi_f1_results.csv       # Peringkat statistik F1-Score
    ├── friedman_nemenyi_prauc_results.csv    # Peringkat statistik PR-AUC
    └── plots/                                # Boxplot faktor dan heatmap visualisasi
```

### 8. Laporan Jawaban Terhadap 4 Research Questions (RQ) & Komparasi Silang
Pada akhir eksekusi, susun laporan sintesis akademik komprehensif di `Eksperimen_Bima/Eksperiment-2/LAPORAN_EKSPERIMEN_2.md` yang menjawab:
- **RQ1**: Dampak Group-Aware Median Imputation (Cleaned, 24 Proyek Utuh) vs Strategi Row Drop (C1) dan Project Drop (C2) pada Eksperimen 1 terhadap generalisasi model cross-project.
- **RQ2**: Perbandingan efektivitas StandardScaler (S1), MinMaxScaler (S2), dan RobustScaler (S3) pada data terimputasi terhadap domain shift.
- **RQ3**: Ketahanan dan konsistensi Random Under-Sampling (B3: RUS) dibandingkan Baseline (B1), SMOTE (B2), dan Class Weight (B4) pada data berpopulasi penuh.
- **RQ4**: Komparasi Random Forest (M1) vs XGBoost GPU (M2) bertenaga histogram gradient boosting.
- **Komparasi Silang Eksperimen 1 vs 2**: Tabel komparasi performa Top Pipelines C1 vs C2 vs Cleaned Imputed.
```
