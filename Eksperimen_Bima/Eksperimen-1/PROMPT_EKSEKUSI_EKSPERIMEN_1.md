# 🚀 Master Prompt Eksekusi: Eksperimen 1 — Rencana Penelitian Utama
**Kelompok**: Kelompok A01 — Rekayasa Perangkat Lunak A (Magister Teknik Informatika ITS)  
**Dokumen Induk**: `Eksperimen_Bima/Rencana_Penelitian_Flaky_Test.md`  
**Output Direktori**: `Eksperimen_Bima/Eksperimen-1/`  
**Hardware Target**: Laptop GPU NVIDIA GeForce RTX 4050 (6GB VRAM, CUDA 13.4, Driver 617.42)  
**Referensi Utama**: Paper FlakeFlagger (Alshammari et al., ICSE 2021) & Source Code pada folder `FlakeFlagger/`  

---

## 📌 Master Prompt (Salin & Jalankan untuk Memulai Eksekusi)

```markdown
Anda adalah Principal Machine Learning & Empirical Software Engineering Researcher yang bertugas mengeksekusi Rencana Penelitian Utama dari dokumen:
`Eksperimen_Bima/Rencana_Penelitian_Flaky_Test.md`.

Seluruh kode, konfigurasi, modul, notebook, log hasil evaluasi, dan visualisasi grafik WAJIB disimpan di folder:
`Eksperimen_Bima/Eksperimen-1/`.

### 1. Konteks & Latar Belakang Masalah
- Paper acuan FlakeFlagger (Alshammari et al., ICSE 2021) menggunakan evaluasi Within-Project / Pooled CV (80/20 atau 90/10) sehingga rentan terhadap data leakage dan over-optimistic performance.
- Penelitian ini mengevaluasi skenario Leave-One-Project-Out Cross-Validation (LOPO-CV) lintas proyek (Cross-Project Flaky Test Prediction) pada dataset resmi FlakeFlagger (`test_features.csv` dan `test_results.csv`).
- Folder `FlakeFlagger/` berisi source code resmi penulis sebagai pembanding dan referensi logika fitur serta ekstraksi metrik.

### 2. Spesifikasi Komputasi, Virtual Environment & Akselerasi GPU (Hardware Constraint)
- Gunakan virtual environment `.venv` yang terletak di root repositori (`.\.venv\Scripts\python.exe`). Seluruh dependensi tersimpan di file `requirements.txt`.
- Gunakan GPU NVIDIA GeForce RTX 4050 Laptop (6GB VRAM) dengan akselerasi CUDA 13 (Driver 617.42).
- Untuk model **XGBoost Classifier**, aktifkan akselerasi GPU menggunakan parameter:
  `tree_method="hist", device="cuda"` (atau `device="cuda:0"`).
- Untuk model **Random Forest Classifier**, manfaatkan utilisasi seluruh thread CPU (`n_jobs=-1`).
- Pastikan penggunaan memori GPU terkontrol dan tidak terjadi VRAM Out-of-Memory (OOM).

### 3. Matriks Eksperimen Utama (Full Factorial 48 Pipelines)
Rancang dan eksekusi secara modular 48 pipeline dari kombinasi variabel bebas berikut:

1. **Strategi Data Cleaning (2 variasi)**:
   - `C1 (Row-Level Drop)`: Menghapus 273 baris yang memiliki nilai NaN pada `testLength`, `numAsserts`, atau `numCoveredLines`. Mempertahankan seluruh 24 proyek (21.963 baris data).
   - `C2 (Project-Level Drop)`: Menghapus 5 proyek yang memiliki baris NaN (`wildfly`, `logback`, `spring-boot`, `wro4j`, `handlebars.java`). Tersisa 19 proyek bersih tanpa missing values.
2. **Strategi Data Scaling (3 variasi)**:
   - `S1`: StandardScaler (Z-score standardization)
   - `S2`: MinMaxScaler (Normalisasi rentang [0, 1])
   - `S3`: RobustScaler (Penskalaan berbasis Median dan Interquartile Range / IQR)
   - *CRITICAL*: Scaling WAJIB di-fit HANYA pada Training Folds ($N-1$ proyek) dan di-transform pada Test Fold (Target Proyek) untuk menjamin **Zero Data Leakage**.
3. **Penanganan Ketimpangan Kelas / Imbalance Handling (4 variasi)**:
   - `B1`: Baseline (Tanpa penanganan imbalance)
   - `B2`: SMOTE (Synthetic Minority Over-sampling Technique, diterapkan HANYA pada data latih)
   - `B3`: Random Under-Sampling (RUS, diterapkan HANYA pada data latih)
   - `B4`: Cost-Sensitive (Class Weight `balanced` pada loss function / estimasi bobot kelas)
4. **Algoritma Klasifikasi (2 variasi)**:
   - `M1`: Random Forest Classifier (`n_estimators=100`, `random_state=42`, `n_jobs=-1`)
   - `M2`: XGBoost Classifier (`n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `tree_method="hist"`, `device="cuda"`, `random_state=42`)

Total Kombinasi: $2 \times 3 \times 4 \times 2 = 48 \text{ Pipelines}$.

### 4. Skema Validasi & Penanganan Kasus Khusus (LOPO-CV)
- Evaluasi dilakukan secara Leave-One-Project-Out Cross-Validation:
  - Pada C1: 24 Folds (setiap fold menguji 1 proyek target, dilatih pada 23 proyek lainnya).
  - Pada C2: 19 Folds (setiap fold menguji 1 proyek target, dilatih pada 18 proyek lainnya).
- **Kasus Khusus `jimfs`**:
  - Proyek `jimfs` memiliki 212 baris test cases namun 0 flaky test (kelas minoritas = 0).
  - Tangani secara elegan: kalkulasi ROC-AUC / PR-AUC menghasilkan NaN jika dihitung langsung. Set nilai secara aman (misal NaN atau log terpisah) dan laporkan metrik Specificity / TN / FP tanpa menyebabkan script error/crash.
- **Kasus Khusus `spring-boot`**:
  - Proyek ini menyumbang 163 flaky test (~20% dari seluruh flaky test). Analisis secara mendalam perbandingannya antara C1 (dipertahankan) vs C2 (dibuang).

### 5. Metrik Evaluasi yang Wajib Dihitung
Hitung metrik performa berikut untuk setiap fold target dan rekap rata-rata (Mean ± Std Dev):
1. **PR-AUC (Precision-Recall Area Under Curve / Average Precision)** — Metrik primer data imbalanced
2. **F1-Score (Khusus Kelas Flaky)**
3. **Recall (Flaky)**
4. **Precision (Flaky)**
5. **ROC-AUC**
6. **MCC (Matthews Correlation Coefficient)**
7. **Balanced Accuracy**
8. **Confusion Matrix Elements**: True Positive (TP), False Positive (FP), True Negative (TN), False Negative (FN)

### 6. Uji Signifikansi Statistik (Empirical Standards)
- Jalankan **Friedman Test** untuk memverifikasi apakah terdapat perbedaan signifikan secara statistik di antara ke-48 pipeline pada seluruh fold proyek.
- Jalankan **Nemenyi Post-hoc Test** untuk menghitung peringkat rata-rata (*average ranks*) dan menghasilkan data diagram *Critical Difference (CD)*.

### 7. Struktur Folder & Deliverables yang Wajib Dihasilkan
Pastikan struktur folder berikut terbentuk lengkap di `Eksperimen_Bima/Eksperimen-1/`:
```
Eksperimen_Bima/Eksperimen-1/
├── README.md                                 # Dokumentasi lengkap eksperimen 1 & cara menjalankan
├── configs/
│   └── experiment_matrix.json                # Definisi parameter 48 pipeline eksperimen
├── src/
│   ├── __init__.py
│   ├── check_gpu.py                          # Skrip verifikasi CUDA GPU RTX 4050 & XGBoost
│   ├── data_loader.py                        # Modul load data test_features.csv & cleaning C1/C2
│   ├── pipeline_builder.py                   # Modul pembuat pipeline scaler + sampler + model
│   ├── lopo_runner.py                        # Engine eksekusi LOPO-CV 48 pipelines
│   ├── statistical_tests.py                  # Skrip Friedman Test & Nemenyi post-hoc ranking
│   └── visualizer.py                         # Pembuatan heatmap, boxplot, dan grafik CD
├── notebooks/
│   └── 01_run_main_48_pipelines.ipynb        # Jupyter Notebook interaktif eksekusi eksperimen
└── results/
    ├── raw_predictions/                      # CSV hasil prediksi per fold
    ├── summary_metrics_48_pipelines.csv      # Ringkasan metrik 48 konfigurasi (Mean, Std, Median)
    ├── friedman_nemenyi_results.csv          # Hasil uji signifikansi statistik
    └── plots/                                # Boxplot metrik, heatmap F1/PR-AUC, dan Critical Difference plot
```

### 8. Laporan Jawaban Terhadap 4 Research Questions (RQ)
Pada akhir eksekusi, susun laporan sintesis akademik komprehensif di `Eksperimen_Bima/Eksperimen-1/LAPORAN_EKSPERIMEN_1.md` yang menjawab:
- **RQ1**: Dampak Row-Level Drop (C1) vs Project-Level Drop (C2) terhadap transferability model.
- **RQ2**: Perbandingan efektivitas StandardScaler (S1), MinMaxScaler (S2), dan RobustScaler (S3) terhadap domain shift.
- **RQ3**: Efektivitas Baseline (B1), SMOTE (B2), RUS (B3), dan Class Weight (B4) pada skenario cross-project.
- **RQ4**: Komparasi Random Forest (M1) vs XGBoost (M2) bertenaga GPU.
```
