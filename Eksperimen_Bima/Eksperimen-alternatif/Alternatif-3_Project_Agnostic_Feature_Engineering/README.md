# 🌟 Alternatif 3: Project-Agnostic Feature Engineering & Normalisasi Relatif

**Kelompok Peneliti**: Kelompok A01 — Rekayasa Perangkat Lunak A (Magister Teknik Informatika ITS)  
**Dokumen Induk**: [`../../Rencana_Penelitian_Flaky_Test.md`](../../Rencana_Penelitian_Flaky_Test.md)  
**Hardware Akselerasi**: Laptop GPU NVIDIA GeForce RTX 4050 (6GB GDDR6 VRAM, CUDA 13.4, Driver 617.42)  
**Dataset**: FlakeFlagger (22.236 test cases dari 24 repositori Java, C1 Row-Level Drop: 21.963 baris)  
**Rujukan Paper Utama**:
1. Alshammari et al. (ICSE 2021) — *FlakeFlagger: Predicting Flakiness Without Rerunning Tests*
2. Afeltra et al. (IEEE Access 2024) — *A Large-Scale Empirical Investigation Into Cross-Project Flaky Test Prediction*
3. Pontillo et al. (EMSE 2022) — *Static Test Flakiness Prediction: How Far Can We Go?*

---

## 📌 1. Latar Belakang & Permasalahan

Dalam prediksi flaky test lintas proyek (*Cross-Project Flaky Test Prediction* dengan validasi LOPO-CV), model machine learning rentan mengalami degradasi performa akibat **bias skala repositori (project size bias)**:
1. **Bias Metrik Ukuran Absolut**: Fitur absolut FlakeFlagger seperti `testLength`, `numCoveredLines`, dan `projectSourceLinesCovered` memiliki variasi skala masif (ratusan ribu baris pada enterprise seperti `spring-boot` vs ratusan baris pada pustaka mikro seperti `jimfs`). Karakteristik absolut ini memicu *domain shift* ekstrem antar proyek sumber dan proyek target.
2. **Multikolinearitas Ekstrem Code Churn**: Delapan window fitur *code churn* `hIndex` (`window5` s.d `window10000`) memiliki koefisien korelasi Pearson sangat tinggi ($r > 0.90$), memicu redundansi representasi dan instabilitas bobot model linier maupun *tree-based*.
3. **Distribusi Ekor Panjang (*Right-Skewed*)**: Fitur waktu eksekusi (`ExecutionTime`) dan ukuran kode memiliki skewness tinggi yang memerlukan kompresi logaritmik $\log(1+x)$.

---

## 🔬 2. Desain Solusi & Rekayasa Fitur Project-Agnostic

Eksperimen ini merancang representasi fitur yang **Project-Agnostic (kebal terhadap skala proyek)**:

### A. Fitur Rasio Relatif (Project-Agnostic Ratios)
- **Kerapatan Asersi**: $\text{assert\_density} = \frac{\text{numAsserts}}{\text{testLength} + 1.0}$
- **Rasio Cakupan Baris**: $\text{coverage\_ratio} = \frac{\text{numCoveredLines}}{\text{testLength} + 1.0}$
- **Rasio Cakupan Kelas Proyek**: $\text{class\_coverage\_ratio} = \frac{\text{projectSourceClassesCovered}}{\text{num\_third\_party\_libs} + 1.0}$
- **Waktu Eksekusi per Asersi**: $\text{time\_per\_assert} = \frac{\text{ExecutionTime}}{\text{numAsserts} + 1.0}$
- **Waktu Eksekusi per Baris Tercover**: $\text{time\_per\_line} = \frac{\text{ExecutionTime}}{\text{numCoveredLines} + 1.0}$

### B. Transformasi Logaritmik ($\log(1+x)$)
Mereduksi skewness ekstrim pada variabel ukuran dan durasi eksekusi:
- `log_ExecutionTime`, `log_testLength`, `log_numCoveredLines`, `log_projectSourceLinesCovered`, `log_projectSourceClassesCovered`, `log_numAsserts`.

### C. Reduksi Dimensi PCA untuk Churn (Zero Data Leakage)
- 8 window `hIndex` distandarisasi menggunakan `StandardScaler` dan dianalisis melalui *Principal Component Analysis (PCA)*.
- **Protokol Ketat Zero Leakage**: Scaler dan PCA di-fit HANYA pada data latih $N-1$ proyek, lalu mentransformasi fold data uji target.
- Mengambil sejumlah komponen utama dengan variansi kumulatif $\ge 95\%$ (menghasilkan komponen ortogonal sempurna $r = 0.00$, $\text{VIF} = 1.00$).

### D. Kumpulan 4 Set Fitur yang Diuji
- **Set F1 (Original Baseline)**: 23 fitur FlakeFlagger mentah.
- **Set F2 (Original + Ratios)**: 23 fitur asli + 5 rasio relatif baru (28 fitur).
- **Set F3 (Project-Agnostic Reformed)**: 5 rasio relatif + 8 Test Smells biner + Log-transformed size/execution metrics + PCA churn components (menghapus metrik mentah absolut: 26 fitur).
- **Set F4 (Feature Selection on F3)**: Top-12 fitur paling diskriminatif dari F3 yang diseleksi menggunakan ANOVA F-test pada data latih fold.

---

## ⚡ 3. Akselerasi Komputasi GPU NVIDIA RTX 4050 & CUDA 13.4

- **XGBoost GPU**: Menggunakan algoritma histogram GPU canggih:
  `tree_method="hist"`, `device="cuda"`, `eval_metric="logloss"`.
- **Random Forest**: Pelatihan model ensemble pohon CPU multi-threaded dengan `n_jobs=-1`.

---

## 🗂️ 4. Struktur Direktori Proyek

```
Alternatif-3_Project_Agnostic_Feature_Engineering/
├── README.md                                 # Panduan eksperimen & dokumentasi teknis
├── PROMPT_EKSEKUSI_ALTERNATIF_3.md          # Master prompt eksekusi AI agent
├── LAPORAN_ALTERNATIF_3.md                   # Laporan empiris akademik komprehensif
├── configs/
│   └── feature_config.json                   # Definisi rumus fitur baru & parameter PCA
├── src/
│   ├── __init__.py
│   ├── check_gpu.py                          # Verifikasi GPU RTX 4050 & CUDA 13
│   ├── data_loader.py                        # Pemuatan data FlakeFlagger C1 (21.963 baris)
│   ├── feature_engineer.py                   # Transformasi rasio, log-transform, PCA, dan VIF
│   ├── lopo_feature_runner.py                # Runner LOPO-CV 24 proyek untuk Set F1 s.d F4
│   ├── statistical_tests.py                  # Uji Wilcoxon Signed-Rank Test & Cliff's Delta
│   └── visualizer.py                         # Modul visualisasi publikasi IEEE/ACM 300 DPI
├── notebooks/
│   └── 01_run_feature_engineering.ipynb      # Jupyter Notebook interaktif
└── results/
    ├── raw_predictions/                      # Log probabilitas & prediksi sampel per fold
    ├── lopo_feature_folds.csv                # Detail seluruh metrik evaluasi tiap fold proyek
    ├── feature_comparison_summary.csv        # Rekap performa Mean ± Std Dev untuk F1 s.d F4
    ├── vif_comparison.csv                    # VIF sebelum vs sesudah transformasi & PCA
    ├── pca_components.csv                    # Loading weights dan variansi tiap komponen PCA
    ├── wilcoxon_test_results.csv             # Hasil uji signifikansi statistik Wilcoxon
    └── plots/                                # Grafik publikasi (Heatmap, Scree, Boxplot, Bars, VIF)
```

---

## 🚀 5. Cara Menjalankan Eksperimen

1. **Verifikasi Lingkungan Komputasi & GPU**:
   ```bash
   .\.venv\Scripts\python.exe src/check_gpu.py
   ```

2. **Eksekusi Penuh Validasi LOPO-CV**:
   ```bash
   .\.venv\Scripts\python.exe src/lopo_feature_runner.py
   ```

3. **Eksplorasi Interaktif melalui Jupyter Notebook**:
   Buka `notebooks/01_run_feature_engineering.ipynb` pada Jupyter Lab atau VS Code untuk menjalankan tahapan rekayasa fitur secara modular.
