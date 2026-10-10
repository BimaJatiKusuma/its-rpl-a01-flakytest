# 🚀 Master Prompt Eksekusi: Alternatif 3 — Project-Agnostic Feature Engineering & Normalisasi Relatif
**Kelompok Peneliti**: Kelompok A01 — Rekayasa Perangkat Lunak A (Magister Teknik Informatika ITS)  
**Dokumen Induk**: `Eksperimen_Bima/Rencana_Penelitian_Flaky_Test.md`  
**Output Direktori**: `Eksperimen_Bima/Eksperimen-alternatif/Alternatif-3_Project_Agnostic_Feature_Engineering/`  
**Hardware Akselerasi**: Laptop GPU NVIDIA GeForce RTX 4050 (6GB GDDR6 VRAM, CUDA 13.4, Driver 617.42)  
**Rujukan Paper Utama**: 
1. Alshammari et al. (ICSE 2021) — *FlakeFlagger: Predicting Flakiness Without Rerunning Tests*
2. Afeltra et al. (IEEE Access 2024) — *A Large-Scale Empirical Investigation Into Cross-Project Flaky Test Prediction*
3. Pontillo et al. (EMSE 2022) — *Static Test Flakiness Prediction: How Far Can We Go?*

---

## 📌 Master Prompt (Salin & Jalankan untuk Memulai Eksekusi)

```markdown
Anda adalah Principal Machine Learning & Empirical Software Engineering Researcher yang bertugas mengeksekusi Studi Lanjutan Alternatif 3:
"Project-Agnostic Feature Engineering & Normalisasi Relatif untuk Mitigasi Bias Skala Proyek pada Cross-Project Flaky Test Prediction".

Seluruh kode program, modul python, notebook eksperimen, konfigurasi, log metrik, dan laporan evaluasi WAJIB disimpan di folder:
`Eksperimen_Bima/Eksperimen-alternatif/Alternatif-3_Project_Agnostic_Feature_Engineering/`.

### 1. Konteks Akademik & Permasalahan
- Pada dataset FlakeFlagger (22 fitur), terdapat banyak fitur metrik absolut seperti `testLength`, `numAsserts`, `numCoveredLines`, dan `projectSourceLinesCovered`.
- Dalam skenario prediksi Within-Project (1 proyek tunggal), metrik absolut ini efektif. Namun pada skenario Cross-Project (LOPO-CV), metrik absolut ini sangat bias terhadap ukuran proyek (skala ratusan ribu baris kode pada `spring-boot` vs ribuan baris kode pada pustaka kecil). Hal ini mengakibatkan model gagal mentransfer pola prediktif ke proyek baru.
- Selain itu, 8 window fitur *code churn* `hIndex` memiliki multikolinearitas ekstrem ($r > 0.90$), yang memicu distorsi bobot pada model linier dan tree-based.
- Riset ini bertujuan merekayasa set fitur baru yang bersifat **Project-Agnostic (kebal terhadap variasi skala repositori)** melalui transformasi rasio, log-transformasi, dan reduksi dimensi PCA.

### 2. Lingkungan Komputasi & Akselerasi GPU RTX 4050 (CUDA 13)
- Gunakan virtual environment Python:
  `.\.venv\Scripts\python.exe`
- Akselerasi komputasi WAJIB memanfaatkan GPU NVIDIA GeForce RTX 4050 Laptop (6GB VRAM, CUDA 13.4, Driver 617.42):
  1. **XGBoost GPU Classifier**: Aktifkan `tree_method="hist"`, `device="cuda"` (atau `device="cuda:0"`).
  2. Pelatihan model pada ruang fitur baru dipercepat penuh menggunakan GPU RTX 4050 dengan memori VRAM yang efisien.
  3. Gunakan multi-threading CPU (`n_jobs=-1`) untuk pelatihan pembanding Random Forest.

### 3. Desain Rekayasa Fitur & Metodologi
Rancang modul rekayasa fitur `feature_engineer.py` yang mentransformasikan 22 fitur asli menjadi set fitur baru:

1. **Transformasi Fitur Rasio Relatif (Project-Agnostic Ratios)**:
   - Kerapatan Asersi: $\text{assert\_density} = \frac{\text{numAsserts}}{\text{testLength} + 1}$
   - Rasio Cakupan Baris: $\text{coverage\_ratio} = \frac{\text{numCoveredLines}}{\text{testLength} + 1}$
   - Rasio Cakupan Kelas Proyek: $\text{class\_coverage\_ratio} = \frac{\text{projectSourceClassesCovered}}{\text{num\_third\_party\_libs} + 1}$
   - Waktu Eksekusi per Asersi: $\text{time\_per\_assert} = \frac{\text{ExecutionTime}}{\text{numAsserts} + 1}$
   - Waktu Eksekusi per Baris Ter-cover: $\text{time\_per\_line} = \frac{\text{ExecutionTime}}{\text{numCoveredLines} + 1}$

2. **Transformasi Logaritmik ($\log(1+x)$)**:
   - Terapkan $\log(1+x)$ pada fitur yang memiliki skewness tinggi: `ExecutionTime`, `testLength`, `numCoveredLines`, `projectSourceLinesCovered`.

3. **Reduksi Dimensi PCA untuk Multikolinearitas Churn (`hIndex`)**:
   - 8 window fitur `hIndex` (hIndex_1 s.d hIndex_8) dianalisis menggunakan *Principal Component Analysis (PCA)*.
   - Fit PCA HANYA pada Training Folds ($N-1$ proyek) dan transform pada Test Fold untuk mencegah kebocoran data.
   - Ambil sejumlah $k$ komponen utama yang menjelaskan $\ge 95\%$ cumulative explained variance (biasanya cukup 1-2 komponen).

4. **Kumpulan Set Fitur yang Diuji**:
   - **Set F1 (Original Baseline)**: 22 fitur asli FlakeFlagger.
   - **Set F2 (Ratios Only)**: Fitur asli + 5 fitur rasio relatif.
   - **Set F3 (Project-Agnostic Reformed)**: Fitur rasio + fitur Test Smells (karena Test Smells bersifat biner dan independen dari ukuran proyek) + Log-transformed execution metrics + PCA Churn components (menghapus fitur ukuran absolut mentah).
   - **Set F4 (Feature Selection on F3)**: F3 yang disaring menggunakan SelectKBest / Mutual Information / Feature Importance threshold.

### 4. Skema Validasi LOPO-CV & Penanganan Kasus
- Terapkan Leave-One-Project-Out CV pada seluruh 24 proyek (Data Cleaning C1: Row-Level Drop).
- Scaling (RobustScaler / StandardScaler) dan PCA WAJIB dihitung HANYA pada data latih $N-1$ proyek, lalu di-transform pada data uji target (Zero Data Leakage).
- Tangani proyek `jimfs` (0 flaky tests) secara aman tanpa crash.

### 5. Metrik Evaluasi yang Wajib Dihitung
Bandingkan performa ke-4 set fitur (F1, F2, F3, F4) pada classifier XGBoost GPU dan Random Forest:
1. **PR-AUC (Precision-Recall Area Under Curve)** — Metrik utama.
2. **F1-Score (Khusus Kelas Flaky)**.
3. **Recall (Flaky)** & **Precision (Flaky)**.
4. **ROC-AUC** & **MCC (Matthews Correlation Coefficient)**.
5. **Analisis Variansi Multikolinearitas (VIF / Variance Inflation Factor)** sebelum dan sesudah PCA.
6. **Uji Signifikansi Statistik**: Wilcoxon Signed-Rank Test untuk membuktikan signifikansi perbedaan performa ($p < 0.05$).

### 6. Struktur Deliverables yang Wajib Dihasilkan
Pastikan direktori berikut terbangun lengkap di `Eksperimen_Bima/Eksperimen-alternatif/Alternatif-3_Project_Agnostic_Feature_Engineering/`:
```
Alternatif-3_Project_Agnostic_Feature_Engineering/
├── README.md                                 # Panduan eksperimen & dokumentasi teknis
├── PROMPT_EKSEKUSI_ALTERNATIF_3.md          # Dokumen prompt ini
├── configs/
│   └── feature_config.json                   # Definisi rumus fitur rasio & ambang variansi PCA
├── src/
│   ├── __init__.py
│   ├── check_gpu.py                          # Verifikasi GPU RTX 4050 & CUDA 13
│   ├── data_loader.py                        # Pemuatan test_features.csv & test_results.csv C1
│   ├── feature_engineer.py                   # Transformasi rasio, log-transform, PCA
│   ├── lopo_feature_runner.py                # Runner LOPO-CV komparasi set fitur F1 s.d F4
│   ├── statistical_tests.py                  # Uji Wilcoxon Signed-Rank Test & VIF calculation
│   └── visualizer.py                         # Heatmap korelasi Pearson, PCA scree plot, boxplot F1
├── notebooks/
│   └── 01_run_feature_engineering.ipynb      # Jupyter Notebook interaktif
└── results/
    ├── raw_predictions/                      # Log prediksi per fold
    ├── feature_comparison_summary.csv        # Rekap metrik (Mean ± Std) untuk set F1 s.d F4
    ├── vif_comparison.csv                    # VIF sebelum dan sesudah reduksi dimensi
    ├── pca_components.csv                    # Bobot loading tiap komponen PCA
    └── plots/                                # Correlation heatmap, scree plot, & boxplot komparasi
```

### 7. Laporan Akademik Akhir
Buat dokumen laporan komprehensif di `LAPORAN_ALTERNATIF_3.md` yang memuat:
1. Apakah representasi Project-Agnostic Features (F3) berhasil mengungguli 22 fitur mentah (F1) dalam skenario lintas proyek?
2. Analisis penurunan multikolinearitas dan efektivitas PCA pada fitur `hIndex`.
3. Fitur baru manakah di antara rasio asersi, rasio cakupan, atau log-transformed execution time yang memberikan kontribusi prediksi tertinggi?
4. Dampak akselerasi GPU RTX 4050 CUDA 13 terhadap kecepatan pelatihan model XGBoost dengan variasi set fitur baru.
```
