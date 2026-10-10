# 🚀 Master Prompt Eksekusi: Alternatif 4 — Interpretasi Fitur Universal Menggunakan SHAP (SHapley Additive exPlanations)
**Kelompok Peneliti**: Kelompok A01 — Rekayasa Perangkat Lunak A (Magister Teknik Informatika ITS)  
**Dokumen Induk**: `Eksperimen_Bima/Rencana_Penelitian_Flaky_Test.md`  
**Output Direktori**: `Eksperimen_Bima/Eksperimen-alternatif/Alternatif-4_SHAP_Feature_Interpretation/`  
**Hardware Akselerasi**: Laptop GPU NVIDIA GeForce RTX 4050 (6GB GDDR6 VRAM, CUDA 13.4, Driver 617.42)  
**Rujukan Paper Utama**: 
1. Lundberg & Lee (NeurIPS 2017) — *A Unified Approach to Interpreting Model Predictions (SHAP)*
2. Alshammari et al. (ICSE 2021) — *FlakeFlagger: Predicting Flakiness Without Rerunning Tests*
3. Afeltra et al. (IEEE Access 2024) — *A Large-Scale Empirical Investigation Into Cross-Project Flaky Test Prediction*

---

## 📌 Master Prompt (Salin & Jalankan untuk Memulai Eksekusi)

```markdown
Anda adalah Principal Machine Learning & Empirical Software Engineering Researcher yang bertugas mengeksekusi Studi Lanjutan Alternatif 4:
"Interpretasi Fitur Universal Berbasis SHAP (SHapley Additive exPlanations): Mengungkap Faktor Penentu Flakiness Lintas Proyek vs Dalam Proyek".

Seluruh kode program, modul python, notebook eksperimen, konfigurasi, log metrik, dan laporan evaluasi WAJIB disimpan di folder:
`Eksperimen_Bima/Eksperimen-alternatif/Alternatif-4_SHAP_Feature_Interpretation/`.

### 1. Konteks Akademik & Permasalahan
- Model Machine Learning untuk deteksi flaky test sering dipandang sebagai *black box*. Pengembang software tidak hanya ingin tahu skor probabilitas suatu test case, namun memerlukan pemahaman: *Mengapa test ini dikategorikan flaky? Indikator apa yang paling bertanggung jawab?*
- Pada penelitian FlakeFlagger (Alshammari et al., 2021) dengan evaluasi Within-Project (Pooled), fitur seperti `ExecutionTime` dan cakupan baris (`numCoveredLines`) dilaporkan paling dominan.
- Namun, pada skenario Cross-Project (Afeltra et al., 2024), apakah fitur yang sama tetap menjadi prediktor paling universal? Ataukah fitur kategori *Test Smells* (seperti `fire-and-forget`, `sleepy`, `indirectTesting`) yang secara semantik tidak tergantung pada skala proyek menjadi jauh lebih penting dan dapat ditransfer lintas proyek?
- Riset ini menerapkan framework teori permainan kooperatif **SHAP (SHapley Additive exPlanations)** untuk membedah signifikansi fitur pada model pohon (Random Forest dan XGBoost GPU).

### 2. Lingkungan Komputasi & Akselerasi GPU RTX 4050 (CUDA 13)
- Gunakan virtual environment Python:
  `.\.venv\Scripts\python.exe`
- Pastikan pustaka `shap` terpasang (jika belum, tambahkan ke environment via `pip install shap`).
- Akselerasi komputasi WAJIB memanfaatkan GPU NVIDIA GeForce RTX 4050 Laptop (6GB VRAM, CUDA 13.4, Driver 617.42):
  1. **XGBoost GPU Classifier**: Latih model dengan `tree_method="hist"`, `device="cuda"`.
  2. **SHAP TreeExplainer**: Gunakan `shap.TreeExplainer(model)` untuk model pohon teroptimasi. Manfaatkan paralelisasi CPU (`check_additivity=False` jika diperlukan efisiensi) dan akselerasi GPU pada pelatihan pohon XGBoost.
  3. Kelola komputasi nilai Shapley secara ter-batch untuk 22.000 data agar tidak melebihi kapasitas memori VRAM 6GB atau RAM sistem.

### 3. Desain Metodologi & Tahapan Eksperimen
Rancang eksperimen analisis interpretasi fitur secara sistematis:

1. **Persiapan Data & Model Baseline**:
   - Gunakan dataset FlakeFlagger dengan data cleaning C1 (Row-Level Drop, 21.963 baris pada 24 proyek).
   - Latih dua model terbaik:
     - Model A: XGBoost GPU Classifier (`device="cuda"`, `tree_method="hist"`)
     - Model B: Random Forest Classifier (`n_estimators=100`, `n_jobs=-1`)

2. **Eksperimen Komparasi 1: Within-Project SHAP vs Cross-Project SHAP**:
   - **Skenario Within-Project**: Latih model pada skenario pooled 90/10 cross-validation (mereplikasi FlakeFlagger). Hitung rata-rata nilai absolut SHAP ($E[|\text{SHAP}|]$) untuk setiap fitur.
   - **Skenario Cross-Project**: Latih model pada LOPO-CV (24 proyek). Untuk setiap fold LOPO, hitung nilai SHAP pada data uji proyek target. Rata-ratakan nilai SHAP lintas seluruh 24 lipatan LOPO.
   - Bandingkan **Peringkat Fitur (Feature Importance Ranking)** antara Within-Project vs Cross-Project:
     - Apakah terdapat pergeseran signifikan (Rank Shift)?
     - Uji hipotesis: Apakah kategori *Test Smells* naik peringkatnya pada skenario Cross-Project dibanding Within-Project?

3. **Eksperimen Komparasi 2: Global Explanations (Analisis Global)**:
   - Buat **SHAP Summary Beeswarm Plot** yang memperlihatkan arah pengaruh nilai fitur (fitur tinggi vs rendah) terhadap probabilitas flakiness.
   - Kelompokkan fitur ke dalam 3 dimensi taksonomi:
     - Dimensi 1: *Test Smells* (8 fitur: `fire-and-forget`, `sleepy`, dll.)
     - Dimensi 2: *Execution & Coverage Metrics* (6 fitur: `ExecutionTime`, `numAsserts`, dll.)
     - Dimensi 3: *Code Churn Metrics* (8 fitur window `hIndex`)
   - Hitung persentase kontribusi total SHAP untuk masing-masing dimensi.

4. **Eksperimen Komparasi 3: Local Explanations (Studi Kasus Proyek Spesifik)**:
   - Pilih proyek yang kaya akan flaky tests (misalnya `spring-boot` dengan 163 flaky tests dan `wildfly`).
   - Buat **SHAP Waterfall Plot** dan **Force Plot** untuk 3 contoh test case flaky paling representatif dan 3 contoh test case non-flaky.
   - Identifikasi pemicu flakiness lokal pada kasus-kasus tersebut.

5. **Eksperimen Komparasi 4: SHAP Dependence Plots & Rekomendasi Developer**:
   - Plot **SHAP Dependence Plots** untuk fitur top-3 (misal `ExecutionTime`, `testLength`, atau smell dominan) beserta fitur interaksinya.
   - Identifikasi titik infleksi (*inflection point / risk threshold*), misalnya: *"Pada ExecutionTime > X ms atau numAsserts > Y, risiko flakiness melonjak tajam"*.

### 4. Metrik & Analisis Statistik yang Wajib Dihasilkan
1. **Mean Absolute SHAP Value ($|\text{SHAP}|$)** per fitur untuk kedua skenario (Within vs Cross).
2. **Spearman’s Rank Correlation Coefficient ($\rho$)**: Ukur korelasi urutan pentingnya fitur antara Within-Project dan Cross-Project.
3. **Persentase Kontribusi Dimensi Fitur**: Bar chart proporsi pengaruh Test Smells vs Coverage vs Churn.

### 5. Struktur Deliverables yang Wajib Dihasilkan
Pastikan direktori berikut terbangun lengkap di `Eksperimen_Bima/Eksperimen-alternatif/Alternatif-4_SHAP_Feature_Interpretation/`:
```
Alternatif-4_SHAP_Feature_Interpretation/
├── README.md                                 # Panduan eksekusi & penjelasan teori SHAP
├── PROMPT_EKSEKUSI_ALTERNATIF_4.md          # Dokumen prompt ini
├── configs/
│   └── shap_config.json                      # Konfigurasi parameter model & sample background
├── src/
│   ├── __init__.py
│   ├── check_gpu.py                          # Verifikasi GPU RTX 4050 & CUDA 13
│   ├── data_loader.py                        # Pemuatan data FlakeFlagger C1
│   ├── shap_explainer.py                     # Mesin ekstraksi TreeExplainer (XGBoost GPU & RF)
│   ├── rank_comparator.py                    # Analisis korelasi Spearman peringkat fitur Within vs Cross
│   └── visualizer.py                         # Plot Beeswarm, Waterfall, Dependence, dan Bar summary
├── notebooks/
│   └── 01_run_shap_interpretation.ipynb      # Jupyter Notebook interaktif
└── results/
    ├── shap_summary_cross_project.csv        # Nilai SHAP rata-rata per fitur pada LOPO-CV
    ├── shap_summary_within_project.csv       # Nilai SHAP rata-rata per fitur pada Pooled CV
    ├── feature_rank_comparison.csv           # Peringkat komparatif & rank shift
    └── plots/                                # Beeswarm plots, waterfall plots, dependence plots
```

### 6. Laporan Akademik Akhir
Buat dokumen laporan komprehensif di `LAPORAN_ALTERNATIF_4.md` yang memuat:
1. Temuan Peringkat Fitur: Fitur apa sajakah yang terbukti menjadi prediktor flaky test paling universal dalam skenario lintas proyek?
2. Jawaban atas hipotesis: Apakah terjadi pergeseran ranking fitur dari Within-Project ke Cross-Project? Seberapa besar kenaikan peran Test Smells?
3. Analisis studi kasus local explanation pada flaky tests di repositori `spring-boot`.
4. *Actionable Advice for Practitioners*: Rekomendasi konkret bagi tim software engineering untuk mendesain unit test yang minim flakiness berdasarkan kurva ketergantungan SHAP.
5. Efisiensi komputasi ekstraksi SHAP dengan akselerasi GPU RTX 4050 CUDA 13.
```
