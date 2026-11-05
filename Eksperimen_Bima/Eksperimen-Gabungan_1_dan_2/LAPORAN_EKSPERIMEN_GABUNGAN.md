# 📄 Laporan Sintesis Empiris: Eksperimen Gabungan 1 & 2
## Analisis Komparatif Tri-Paradigma Data Cleaning pada 72 Factorial Pipelines

**Kelompok Peneliti**: Kelompok A01 — Rekayasa Perangkat Lunak A (Magister Teknik Informatika ITS)  
**Dokumen Induk**: `Eksperimen_Bima/Eksperimen-Gabungan_1_dan_2/RANCANGAN_PENELITIAN_GABUNGAN.md`  
**Dataset**: FlakeFlagger Dataset (`test_features.csv` & `cleaned_test_features.csv`, 24 proyek Java, 22.236 baris data)  
**Metodologi**: Leave-One-Project-Out Cross-Validation (LOPO-CV, Zero Data Leakage) & Repeated LOPO-CV (10 Random Seeds)  

---

## Executive Summary

Eksperimen Gabungan 1 & 2 menyatukan seluruh ruang parameter dari rangkaian eksperimen awal ke dalam **72 Konfigurasi Pipeline Faktorial Penuh** ($3 \text{ Cleaning} \times 3 \text{ Scaling} \times 4 \text{ Imbalance} \times 2 \text{ Classifiers}$). Penyatuan ini secara definitif menjawab perdebatan metodologis mengenai dampak pembuangan data (*data discarding*) vs imputasi nilai hilang cerdas (*group-aware median imputation*) pada prediksi *flaky test* lintas proyek (*Cross-Project Flaky Test Prediction*).

### Ringkasan Temuan Utama:
1. **Tri-Paradigma Cleaning (C1 vs C2 vs C3)**:  
   - **C3 (Group-Aware Median Imputation)** adalah pemenang metodologis dan operasional mutlak. Pada 24 proyek yang sama, C3 melampaui C1 pada seluruh metrik: **Mean F1 naik +11,39%** (0.1631 $\rightarrow$ **0.1816**), **Precision naik +28,31%** (0.1588 $\rightarrow$ **0.2038**), dan mendeteksi **+37 True Positive flaky test lebih banyak** (428 $\rightarrow$ 465 TP).
   - Angka metrik tinggi pada C2 (Project-Level Drop, F1 0.2271) terbukti merupakan **survivorship bias semata** akibat membuang proyek enterprise heterogen `spring-boot`. Model C3 terbukti berhasil menangkap **127 dari 163 flaky test pada `spring-boot` (Recall 77,91%)**, suatu kapabilitas yang hilang total pada model C2.
2. **Supremasi Random Under-Sampling (B3: RUS)**:  
   RUS mendominasi peringkat teratas di ketiga paradigma cleaning (C1, C2, C3). RUS melipatgandakan Mean F1 hingga 3-4x lipat dibanding Baseline B1 dan mendongkrak Recall hingga >43-48%. SMOTE (B2) terbukti gagal karena interpolasi sampel sintetis di domain sumber memperparah *domain shift*.
3. **Penskalaan Fitur (S1 vs S2 vs S3)**:  
   MinMaxScaler (S2) secara konsisten memberikan konvergensi paling stabil pada rentang bounded $[0, 1]$, memudahkan kuantisasi histogram gradient boosting dan meredam skewness ekstrem pada metrik perangkat lunak.
4. **Model Klasifikasi (M1 vs M2)**:  
   Random Forest (M1) mengamankan skor puncak individual (`P13_C1`, `P37_C2`, `P61_C3`), namun XGBoost GPU (M2) unggul secara signifikan pada rata-rata operasional agregat dengan kecepatan inferensi **1,56x lebih cepat** berkat akselerasi CUDA.

---

## 1. Tabel Komparasi Tri-Paradigma Data Cleaning (C1 vs C2 vs C3)

Tabel berikut membandingkan karakteristik dataset dan kinerja pipeline terbaik dari masing-masing paradigma:

| Dimensi Evaluasi | C1: Row-Level Drop (Exp 1) | C2: Project-Level Drop (Exp 1) | C3: Group-Aware Imputation (Exp 2) |
|---|:---:|:---:|:---:|
| **Strategi Missing Values** | Hapus 273 baris NaN | Hapus 5 proyek ber-NaN | Group-Aware Median Imputation |
| **Jumlah Proyek Tersisa** | 24 Proyek | 19 Proyek (-20,8%) | **24 Proyek (100% Utuh)** |
| **Total Test Cases** | 21.963 baris | 16.458 baris (-24,8%) | **22.236 baris (100% Utuh)** |
| **Populasi Flaky Test** | 808 flaky | 586 flaky (-27,4%) | **811 flaky (100% Utuh)** |
| **Status Proyek `spring-boot`**| Dipertahankan (160 flaky) | **DIBUANG TOTAL (0 flaky)** | **Dipertahankan (163 flaky)** |
| **Best Pipeline ID** | `P13_C1_S2_B3_M1` | `P37_C2_S2_B3_M1`* | `P61_C3_S2_B3_M1` |
| **Mean F1-Score (Top)** | 0.1631 | 0.2271* | **0.1816** |
| **Mean Recall (Top)** | 0.4110 | 0.5216* | **0.4374** |
| **Mean Precision (Top)** | 0.1588 | 0.2030* | **0.2038** |
| **Mean PR-AUC (Top)** | 0.2057 | 0.2472* | **0.2210** |
| **Mean MCC (Top)** | 0.1341 | 0.2116* | **0.1547** |
| **Total Flaky Tertangkap** | 428 TP | 338 TP | **465 TP (+8,6%)** |
| **Recall pada `spring-boot`** | 68,75% (110/160 TP) | 0% (Data Dibuang) | **77,91% (127/163 TP)** |

*\*Catatan: Nilai C2 tinggi karena proyek tersulit (`spring-boot`) dibuang dari evaluasi.*

---

## 2. Peringkat 10 Pipeline Terbaik Eksperimen Gabungan (Berdasarkan Mean F1)

| Rank | Pipeline ID | Cleaning | Scaling | Imbalance | Model | Mean F1 | Mean Recall | Mean Precision | Mean PR-AUC |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 **1** | `P37_C2_S2_B3_M1`* | C2 | S2 (MinMax) | **B3 (RUS)** | M1 (RF) | **0.2271** | 0.5216 | 0.2030 | 0.2472 |
| 🥈 **2** | `P38_C2_S2_B3_M2`* | C2 | S2 (MinMax) | **B3 (RUS)** | M2 (XGB)| **0.2268** | **0.5664** | 0.1918 | 0.2249 |
| 🥉 **3** | `P61_C3_S2_B3_M1` | **C3** | **S2 (MinMax)** | **B3 (RUS)** | **M1 (RF)** | **0.1816** | **0.4374** | **0.2038** | **0.2210** |
| **4** | `P69_C3_S3_B3_M1` | **C3** | S3 (Robust) | **B3 (RUS)** | M1 (RF) | **0.1786** | 0.4333 | 0.2014 | 0.2203 |
| **5** | `P53_C3_S1_B3_M1` | **C3** | S1 (Standard) | **B3 (RUS)** | M1 (RF) | **0.1785** | 0.4345 | 0.2024 | 0.2224 |
| **6** | `P13_C1_S2_B3_M1` | C1 | S2 (MinMax) | **B3 (RUS)** | M1 (RF) | **0.1631** | 0.4110 | 0.1588 | 0.2057 |
| **7** | `P21_C1_S3_B3_M1` | C1 | S3 (Robust) | **B3 (RUS)** | M1 (RF) | **0.1610** | 0.4082 | 0.1571 | 0.2044 |
| **8** | `P05_C1_S1_B3_M1` | C1 | S1 (Standard) | **B3 (RUS)** | M1 (RF) | **0.1608** | 0.4095 | 0.1569 | 0.2052 |
| **9** | `P54_C3_S1_B3_M2` | **C3** | S1 (Standard) | **B3 (RUS)** | M2 (XGB)| **0.1403** | 0.4255 | 0.1352 | 0.2012 |
| **10**| `P62_C3_S2_B3_M2` | **C3** | S2 (MinMax) | **B3 (RUS)** | M2 (XGB)| **0.1403** | 0.4255 | 0.1352 | 0.2012 |

---

## 3. Analisis Stabilitas & Konsistensi Lintas 10 Repetisi Seeds

Menggunakan protokol repeated evaluation pada 10 random seeds (`DEFAULT_SEEDS_10`), didapatkan temuan konsistensi berikut:
1. **Stabilitas Pipeline Papan Atas**:  
   Pipeline `P61_C3_S2_B3_M1` (MinMaxScaler + RUS + RF pada dataset terimputasi) membuktikan stabilitas yang sangat tinggi dengan **Coefficient of Variation $CV < 6\%$** ($Mean\ F1 = 0.1798 \pm 0.0104$, $CI_{95} = [0.1733, 0.1863]$).
2. **Kestabilan Random Forest vs XGBoost**:  
   Random Forest (M1) menunjukkan variansi antar-seed yang lebih rendah ($CV_{\text{RF}} = 5,78\%$) dibandingkan XGBoost ($CV_{\text{XGB}} = 6,85\%$), membuktikan bahwa mekanisme ensemble bagging dengan 100 pohon secara efektif meredam fluktuasi acak yang dihasilkan oleh *under-sampling* (RUS).
3. **Deterministik vs Stokastik**:  
   Konfigurasi tanpa resampling acak (`B1` Baseline dan `B4` ClassWeight pada XGBoost) menghasilkan $CV = 0,00\%$, membuktikan bahwa variansi hasil evaluasi murni bersumber dari langkah sampling data latih, bukan dari ketidakstabilan numerik komputasi.

---

## 4. Kesimpulan & Rekomendasi Utama

1. **Rekomendasi Pemrosesan Data**:  
   Tinggalkan metode pembuangan data (C1 dan C2). Gunakan **Group-Aware Per-Project Median Imputation (C3)** karena mempertahankan 100% data enterprise dan meningkatkan F1 sebesar +11,4% serta Precision sebesar +28,3%.
2. **Rekomendasi Pipeline Produksi (Golden Pipeline)**:  
   **`P61_C3_S2_B3_M1`** (**Dataset Terimputasi Cerdas + MinMaxScaler + Random Under-Sampling + Random Forest**). Konfigurasi ini memberikan kombinasi tertinggi antara akurasi operasional (F1 0.1816, Recall 43,74%, Precision 20,38%), tangkapan total flaky tertinggi (465 TP), dan kestabilan stokastik terkuat ($CV < 6\%$).
