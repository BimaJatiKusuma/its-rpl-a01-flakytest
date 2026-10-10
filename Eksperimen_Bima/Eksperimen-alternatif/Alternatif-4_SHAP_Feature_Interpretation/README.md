# 🌟 Alternatif 4: Interpretasi Fitur Universal Menggunakan SHAP (SHapley Additive exPlanations)

**Kelompok Peneliti**: Kelompok A01 — Rekayasa Perangkat Lunak A (Magister Teknik Informatika ITS)  
**Dokumen Induk**: [`../../Rencana_Penelitian_Flaky_Test.md`](../../Rencana_Penelitian_Flaky_Test.md)  
**Laporan Lengkap**: [`./LAPORAN_ALTERNATIF_4.md`](./LAPORAN_ALTERNATIF_4.md)  
**Hardware Akselerasi**: Laptop GPU NVIDIA GeForce RTX 4050 (6GB GDDR6 VRAM, CUDA 13.4, Driver 617.42)  
**Paper Rujukan**:
1. Lundberg & Lee (NeurIPS 2017) — *A Unified Approach to Interpreting Model Predictions (SHAP)*
2. Alshammari et al. (ICSE 2021) — *FlakeFlagger: Predicting Flakiness Without Rerunning Tests*
3. Afeltra et al. (IEEE Access 2024) — *A Large-Scale Empirical Investigation Into Cross-Project Flaky Test Prediction*

---

## 📌 Ringkasan Riset & Temuan Kunci

Model Machine Learning pendeteksi flaky test kerap dipandang sebagai *black-box*. Studi Lanjutan Alternatif 4 menerapkan kerangka teori permainan kooperatif **SHAP (SHapley Additive exPlanations)** melalui **TreeExplainer** untuk membongkar faktor penentu flakiness secara transparan pada model **XGBoost GPU** dan **Random Forest** (21.963 baris data uji valid pada 24 proyek open-source Java).

### 🌟 Temuan Kunci:
1. **Fitur Universal Teratas**:
   - `ExecutionTime` (Waktu Eksekusi) terbukti menjadi prediktor flaky test paling dominan dan universal, menyumbang **26,75%** dari total kontribusi SHAP pada skenario Cross-Project ($E[|\text{SHAP}|] = 1.1519$) dan **26,31%** pada Within-Project ($E[|\text{SHAP}|] = 1.1330$).
   - Diikuti oleh riwayat modifikasi jangka menengah `hIndexModificationsPerCoveredLine_window100` (**9,24%**) serta metrik ukuran proyek yang ter-cover `projectSourceClassesCovered` (**9,02%**) dan `projectSourceLinesCovered` (**7,61%**).
2. **Korelasi Peringkat Fitur yang Sangat Kuat**:
   - Koefisien korelasi peringkat Spearman antara Within-Project dan Cross-Project mencapai **$\rho = 0.9704$ ($p = 1.98 \times 10^{-14}$)**, membuktikan bahwa peringkat fitur bersifat konsisten secara global, namun dengan pergeseran lokal penting.
3. **Dinamika Pergeseran Peringkat (Rank Shift)**:
   - Metrik riwayat modifikasi jangka panjang `hIndex_window10000` melesat naik $+4$ peringkat (dari posisi #10 di Within ke #6 di Cross).
   - Metrik ketergantungan library eksternal `num_third_party_libs` melesat naik $+3$ peringkat (dari #12 ke #9).
   - Di antara Test Smells, `mystery-guest` naik $+2$ peringkat (dari #19 ke #17), dan `fire-and-forget` memimpin sebagai Test Smell paling berdampak (#11 dengan kontribusi 3,26%).
4. **Efisiensi Akselerasi GPU RTX 4050 (CUDA 13.4)**:
   - Pelatihan dan ekstraksi SHAP pada **XGBoost GPU** untuk seluruh 24 fold LOPO-CV (21.963 baris data) tuntas hanya dalam **4,96 detik** (0,20 detik/fold).
   - Random Forest CPU memerlukan **365,12 detik** untuk skenario yang sama.
   - Akselerasi GPU XGBoost menghasilkan **speedup hingga 73,6x - 107,7x** dibanding CPU multi-core.

---

## 🗂️ Struktur Direktori Proyek

```
Alternatif-4_SHAP_Feature_Interpretation/
├── README.md                                 # Ringkasan riset & dokumentasi
├── PROMPT_EKSEKUSI_ALTERNATIF_4.md          # Master prompt eksekusi AI agent
├── LAPORAN_ALTERNATIF_4.md                   # Dokumen laporan ilmiah komprehensif
├── configs/
│   └── shap_config.json                      # Konfigurasi parameter model & taksonomi fitur
├── src/
│   ├── __init__.py                           # Package init
│   ├── check_gpu.py                          # Verifikasi GPU RTX 4050 & CUDA 13.4
│   ├── data_loader.py                        # Pemuatan data FlakeFlagger C1 (21.963 baris)
│   ├── shap_explainer.py                     # Mesin ekstraksi TreeExplainer (XGBoost GPU & RF)
│   ├── rank_comparator.py                    # Analisis Spearman & Rank Shift Within vs Cross
│   ├── visualizer.py                         # Plot Beeswarm, Waterfall, Dependence, dan Slopegraph
│   └── run_experiment.py                     # Master runner pipeline eksperimen
├── notebooks/
│   └── 01_run_shap_interpretation.ipynb      # Jupyter Notebook interaktif & visualisasi
└── results/
    ├── shap_summary_cross_project.csv        # Rata-rata nilai SHAP LOPO-CV (XGBoost GPU)
    ├── shap_summary_within_project.csv       # Rata-rata nilai SHAP Pooled CV (XGBoost GPU)
    ├── shap_summary_cross_project_rf.csv     # Rata-rata nilai SHAP LOPO-CV (Random Forest)
    ├── shap_summary_within_project_rf.csv    # Rata-rata nilai SHAP Pooled CV (Random Forest)
    ├── feature_rank_comparison.csv           # Peringkat komparatif, kontribusi %, & rank shift
    ├── rank_comparison_stats.json            # Metrik statistik Spearman, Pearson, & Taksonomi
    ├── spring_boot_case_studies.json         # Sampel penjelasan lokal kasus spring-boot
    ├── shap_cache_xgb.pkl                    # Cache matriks nilai SHAP & probabilitas
    └── plots/                                # 15 Gambar grafik publikasi 300 DPI
```

---

## 🚀 Panduan Eksekusi

### 1. Verifikasi Lingkungan GPU
```bash
.\.venv\Scripts\python.exe src/check_gpu.py
```

### 2. Jalankan Pipeline Ekstraksi SHAP Penuh
```bash
.\.venv\Scripts\python.exe src/run_experiment.py
```

### 3. Eksplorasi Interaktif melalui Jupyter Notebook
Buka dan jalankan notebook:
[`notebooks/01_run_shap_interpretation.ipynb`](notebooks/01_run_shap_interpretation.ipynb)
