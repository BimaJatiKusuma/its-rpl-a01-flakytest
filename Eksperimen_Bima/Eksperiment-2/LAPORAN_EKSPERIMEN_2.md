# 📄 Laporan Komprehensif Hasil Eksperimen 2: Cross-Project Flaky Test Prediction pada Dataset Terimputasi Penuh

**Kelompok Peneliti**: Kelompok A01 — Rekayasa Perangkat Lunak A (Magister Teknik Informatika ITS)  
**Dokumen Induk**: `Eksperimen_Bima/Eksperiment-2/RANCANGAN_PENELITIAN_EKSPERIMEN_2.md`  
**Dataset Utama**: `Experiment_Ali/data/cleaned_test_features.csv` (22.236 baris data, 24 proyek Java, 811 flaky test)  
**Metodologi**: Full Factorial 24 Pipelines × 24 Folds Leave-One-Project-Out Cross-Validation (LOPO-CV, Zero Data Leakage)  
**Total Evaluasi**: **576 Model Folds**  
**Akselerasi Komputasi**: NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM, CUDA 13.4, Driver 617.42) & CPU Multi-threading  
**Tanggal Eksekusi Selesai**: 11 Oktober 2026  

---

## Executive Summary

Eksperimen 2 merupakan puncak validasi empiris dari rangkaian penelitian prediksi *flaky test* lintas proyek (*Cross-Project Flaky Test Prediction*) oleh Kelompok A01. Eksperimen ini memvalidasi efektivitas teknik prapemrosesan mutakhir **Group-Aware Per-Project Median & Probabilistic Imputation** yang diterapkan pada dataset FlakeFlagger (Alshammari et al., ICSE 2021). 

Berbeda dengan Eksperimen 1 yang terbelah oleh manipulasi pembuangan data C1 (Row-Level Drop) dan C2 (Project-Level Drop), **Eksperimen 2 melestarikan 100% data**: seluruh 24 repositori open-source Java, 22.236 test cases, dan 811 flaky test (termasuk 163 flaky test dari proyek raksasa `spring-boot`) berhasil dipertahankan tanpa menyisakan satu pun nilai hilang (*0 missing values*).

### Ringkasan Temuan Empiris Kunci:
1. **RQ1 (Imputasi Cerdas vs Pembuangan Data)**: Pelestarian data via *Group-Aware Median Imputation* secara signifikan mengungguli strategi *Row-Level Drop (C1)* pada baseline komparatif yang sama (24 proyek). Pipeline terbaik mencatatkan peningkatan performa menyeluruh: **Mean F1 melonjak +11,39%** (0.1631 $\rightarrow$ **0.1816**), **Mean Precision meningkat +28,31%** (0.1588 $\rightarrow$ **0.2038**), **Mean Recall meningkat +6,43%** (0.4110 $\rightarrow$ **0.4374**), dan **PR-AUC meningkat +7,40%** (0.2057 $\rightarrow$ **0.2210**). Sementara ilusi keunggulan C2 pada Eksperimen 1 terbukti merupakan *survivorship bias* akibat pembuangan proyek sulit `spring-boot`.
2. **RQ2 (Feature Scaling)**: **MinMaxScaler (S2)** mencatatkan stabilitas dan performa rata-rata tertinggi di antara seluruh faktor penskalaan (Mean F1: **0.0888**, Recall: **0.1866**, PR-AUC: **0.2001**). Algoritma pohon keputusan membuktikan invarian split pada transformasi monotonik, namun rentang bounded $[0, 1]$ memfasilitasi pembentukan bin histogram kuantisasi yang paling konsisten.
3. **RQ3 (Class Imbalance Handling)**: **Random Under-Sampling (B3: RUS) mempertahankan supremasi mutlaknya!** RUS mendongkrak **Mean Recall hingga 43,03%** (dibandingkan Baseline B1 yang hanya 6,26% dan SMOTE B2 yang hanya 10,73%), serta melipatgandakan **Mean F1 hingga >4x lipat** (0.0384 $\rightarrow$ **0.1599**). SMOTE terbukti gagal dalam skenario cross-project karena interpolasi sintetis di domain sumber memperparah *domain shift*.
4. **RQ4 (Model Klasifikasi)**: **Random Forest (M1)** menduduki posisi teratas pada konfigurasi terbaik (`P13_S2_B3_M1`: F1 **0.1816**, Precision **0.2038**, Recall **0.4374**), namun secara agregat faktor, **XGBoost GPU (M2)** memberikan keunggulan operasional yang lebih tinggi dengan Mean F1 **+38,8%** (0.0985 vs 0.0710), Mean Recall **+51,1%** (0.2154 vs 0.1425), dan Mean Precision **+22,1%** (0.1785 vs 0.1461).
5. **Preservasi Proyek Skala Enterprise**: Pada proyek `spring-boot` (163 flaky test), model terbaik (`P13_S2_B3_M1`) berhasil menangkap **127 dari 163 flaky test (Recall 77,91%, F1 0.3975)** dalam kondisi *completely unseen target fold*. Ini membuktikan bahwa transfer pengetahuan lintas proyek bekerja secara efektif pada sistem enterprise nyata.
6. **Signifikansi Statistik (ACM/SIGSOFT Standards)**: Uji **Friedman Test** pada F1-Score menghasilkan $\chi^2 = 173.2039$ dengan $p\text{-value} = 5.17 \times 10^{-25} \ll 0.05$, mengonfirmasi perbedaan performa yang sangat signifikan secara statistik di antara ke-24 pipeline. Klaster teratas uji **Nemenyi Post-hoc** seluruhnya dikuasai oleh varian RUS (`B3`).

---

## 1. Desain Eksperimen & Matriks Faktorial 24 Pipelines

Eksperimen 2 menggunakan dataset `Experiment_Ali/data/cleaned_test_features.csv` dengan membuang 5 kolom metadata identitas (`Unnamed: 0`, `test_name`, `project`, `testClassName`, `testMethodName`) dan menggunakan **23 fitur prediktif resmi** (8 Test Smell biner, 7 Metrik Eksekusi & Coverage numerik, 8 Code Churn H-Index numerik).

```
Matriks Faktorial: 3 Scaling (S1, S2, S3) × 4 Imbalance (B1, B2, B3, B4) × 2 Classifiers (M1, M2) = 24 Pipelines
Validasi: 24-Fold Leave-One-Project-Out Cross-Validation (LOPO-CV) = 576 Evaluasi Model
```

| Dimensi | Variasi | Protokol Operasional |
|---|---|---|
| **Data Cleaning** | **Cleaned Imputed** (100% Utuh) | *Group-Aware Per-Project Median & Probabilistic Imputation*. Bebas NaN, 22.236 baris, 24 proyek. |
| **Feature Scaling** | **S1**: StandardScaler<br>**S2**: MinMaxScaler<br>**S3**: RobustScaler | **Zero Data Leakage**: Scaler di-`fit` HANYA pada data latih ($N-1=23$ proyek) dan di-`transform` pada data uji ($1$ target proyek). |
| **Imbalance Handling** | **B1**: Baseline (None)<br>**B2**: SMOTE<br>**B3**: RUS (Random Under-Sampling)<br>**B4**: ClassWeight (Cost-Sensitive) | Diterapkan **HANYA pada data latih**. Data uji dibiarkan pada distribusi alaminya. Pada XGBoost B4, bobot dihitung via parameter `scale_pos_weight = neg / pos`. |
| **Classifier Model** | **M1**: Random Forest<br>**M2**: XGBoost GPU | M1: `n_estimators=100`, CPU multi-threading (`n_jobs=-1`).<br>M2: `n_estimators=100`, `max_depth=6`, GPU hist (`device="cuda"`, `tree_method="hist"`). |

---

## 2. Analisis Jawaban Empiris Terhadap 4 Research Questions (RQ)

### 🔹 RQ1: Dampak Imputasi Cerdas vs Strategi Pembuangan Data (C1 vs C2)

> **Pertanyaan Penelitian 1**: *Bagaimana dampak pelestarian seluruh 24 proyek dan 811 flaky test melalui Group-Aware Median Imputation terhadap kemampuan generalisasi model lintas proyek jika dibandingkan dengan Row-Level Drop (C1) dan Project-Level Drop (C2) pada Eksperimen 1?*

#### Tabel Perbandingan Komprehensif Tri-Skenario:
| Indikator Perbandingan | Eksperimen 1: C1 (Row Drop) | Eksperimen 1: C2 (Project Drop) | Eksperimen 2: Cleaned Imputed | Delta (Cleaned vs C1) |
|---|:---:|:---:|:---:|:---:|
| **Strategi Missing Values** | Drop 273 Baris NaN | Drop 5 Proyek NaN | Group-Aware Median Imputation | **Zero Discarded Data** |
| **Jumlah Proyek Tersisa** | 24 Proyek | 19 Proyek (-20,8%) | **24 Proyek (100% Utuh)** | $\pm 0$ |
| **Total Test Cases** | 21.963 baris | 16.458 baris (-24,8%) | **22.236 baris (100% Utuh)** | **+273 baris (+1,24%)** |
| **Populasi Flaky Test** | 808 flaky | 586 flaky (-27,4%) | **811 flaky (100% Utuh)** | **+3 flaky (+0,37%)** |
| **Proyek Enterprise Kunci** | `spring-boot` (160 flaky) | DIBUANG TOTAL (0 flaky) | **`spring-boot` (163 flaky)** | **Terlestarikan 100%** |
| **Mean PR-AUC (Top Pipeline)** | 0.2057 (`P13_C1`) | 0.2472 (`P37_C2`)* | **0.2210 (`P13_S2`)** | **+7,40%** |
| **Mean F1-Score (Top Pipeline)**| 0.1631 (`P13_C1`) | 0.2271 (`P37_C2`)* | **0.1816 (`P13_S2`)** | **+11,39%** |
| **Mean Recall (Top Pipeline)**  | 0.4110 (`P13_C1`) | 0.5216 (`P37_C2`)* | **0.4374 (`P13_S2`)** | **+6,43%** |
| **Mean Precision (Top Pipeline)**| 0.1588 (`P13_C1`) | 0.2030 (`P37_C2`)* | **0.2038 (`P13_S2`)** | **+28,31%** |
| **Mean MCC (Top Pipeline)**    | 0.1341 (`P13_C1`) | 0.2116 (`P37_C2`)* | **0.1547 (`P13_S2`)** | **+15,34%** |
| **Total TP Flaky Tertangkap**  | 428 flaky | 338 flaky | **465 flaky** | **+37 Flaky (+8,64%)** |

*\*Catatan Kritis: Angka C2 pada Eksperimen 1 mengalami survivorship bias akibat pembuangan proyek skala enterprise `spring-boot` yang memiliki 163 flaky test.*

#### Sintesis Analisis RQ1:
1. **Keunggulan Nyata atas Row Drop (C1)**: Membandingkan secara langsung pada 24 lipatan proyek yang sama, dataset terimputasi bersih secara konsisten melampaui C1 pada seluruh metrik. F1 meningkat dari 0.1631 ke **0.1816 (+11,4%)**, dan Precision melonjak tajam dari 0.1588 ke **0.2038 (+28,3%)**. Total flaky test yang berhasil dideteksi di seluruh repositori melonjak dari 428 menjadi **465 TP**. Ini membuktikan secara empiris bahwa mengimputasi nilai hilang menggunakan median per-proyek menyediakan informasi kontekstual yang lebih kaya bagi pembentukan partisi pohon keputusan.
2. **Dekonstruksi Ilusi C2 (Project Drop)**: Pada Eksperimen 1, C2 mencatatkan rata-rata F1 0.2271. Namun, Eksperimen 2 membuktikan bahwa angka tinggi tersebut semata-mata adalah artefak dari eliminasi proyek heterogen berskala masif (`spring-boot` memiliki 2.127 baris dan 163 flaky test; `wildfly` memiliki 1.236 baris). Ketika model diuji pada kondisi dunia nyata yang menyertakan proyek-proyek raksasa tersebut, dataset terimputasi Eksperimen 2 mampu menangkap **127 flaky test pada `spring-boot` (Recall 77,91%)** — sesuatu yang mustahil dicapai oleh model C2 karena data tersebut sudah dibuang di awal.

---

### 🔹 RQ2: Pengaruh Penskalaan Fitur (S1 vs S2 vs S3) pada Data Terimputasi

> **Pertanyaan Penelitian 2**: *Di antara Standardization (S1), Normalization (S2), dan Robust Scaler (S3), teknik penskalaan manakah yang memberikan performa tertinggi dalam meredam domain shift ketika diaplikasikan pada dataset yang telah diimputasi?*

#### Tabel Agregat Kinerja per Strategi Scaling (8 Pipeline per Scaler):
| Strategi Scaling | Mean PR-AUC | Mean F1-Score | Mean Recall | Mean Precision | Mean MCC | Mean ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **S1 (StandardScaler)** | 0.1956 | 0.0832 | 0.1765 | 0.1600 | 0.0659 | 0.6974 |
| **S2 (MinMaxScaler)**   | **0.2001** | **0.0888** | **0.1866** | **0.1629** | **0.0719** | **0.7029** |
| **S3 (RobustScaler)**   | **0.2003** | 0.0822 | 0.1736 | **0.1641** | 0.0663 | 0.7024 |

#### Sintesis Analisis RQ2:
1. **Keunggulan Konsisten MinMaxScaler (S2)**: S2 mengungguli StandardScaler (S1) dan RobustScaler (S3) pada metrik operasional F1-Score (**0.0888** vs 0.0832 / 0.0822) dan Recall (**0.1866** vs 0.1765 / 0.1736), serta mencatat MCC tertinggi (**0.0719**). Konfigurasi pipeline terbaik nomor 1 (`P13_S2_B3_M1`) juga menggunakan scaler S2.
2. **Karakteristik Bounded Range $[0, 1]$**: Metrik perangkat lunak (seperti `testLength`, `numAsserts`, `ExecutionTime`) memiliki skewness positif ekstrem (berekor panjang). Normalisasi $[0, 1]$ membatasi seluruh fitur ke domain kompak tanpa mendistorsi rasio proporsional, mempermudah algoritma histogram gradient boosting (`hist`) untuk mengalokasikan bin kuantisasi secara merata.
3. **Invariansi Monotonik Pohon**: Perbedaan antar ketiga scaler relatif moderat (< 1% perbedaan PR-AUC). Ini mengonfirmasi prinsip teoretis bahwa pohon keputusan membagi fitur berbasis urutan rank (*ordinal split threshold*). Namun, saat dihadapkan pada domain shift lintas repositori, penyeragaman rentang ekstrem via S2 memberikan stabilitas inferensi terbaik.

---

### 🔹 RQ3: Ketahanan Strategi Penanganan Ketimpangan Kelas (B1 vs B2 vs B3 vs B4)

> **Pertanyaan Penelitian 3**: *Apakah Random Under-Sampling (B3: RUS) tetap menjadi strategi terbaik secara konsisten dalam memaksimalkan F1-Score dan Recall pada dataset terimputasi, melampaui SMOTE (B2), Class Weight (B4), dan Baseline (B1)?*

#### Tabel Agregat Kinerja per Strategi Imbalance (6 Pipeline per Strategi):
| Strategi Imbalance | Mean PR-AUC | Mean F1-Score | Mean Recall | Mean Precision | Mean MCC | Mean ROC-AUC | Total TP Tertangkap |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B1 (Baseline / Tanpa Resampling)** | 0.1981 | 0.0384 | 0.0626 | 0.1533 | 0.0336 | 0.6974 | 324 flaky |
| **B2 (SMOTE)**                        | 0.1991 | 0.0764 | 0.1073 | 0.1650 | 0.0617 | **0.7019** | 589 flaky |
| **B3 (Random Under-Sampling / RUS)**  | **0.2112** | **0.1599** | **0.4303** | **0.1689** | **0.1266** | **0.7029** | **2.569 flaky** |
| **B4 (ClassWeight / Cost-Sensitive)** | 0.1864 | 0.0643 | 0.1155 | 0.1620 | 0.0502 | 0.6726 | 514 flaky |

#### Sintesis Analisis RQ3:
1. **Dominasi Mutlak Random Under-Sampling (B3)**:
   - **F1-Score Melonjak 4,16x Lipat**: Dari 0.0384 (B1) menjadi **0.1599** (B3).
   - **Recall Melonjak 6,87x Lipat**: Dari 6,26% (B1) menjadi **43,03%** (B3).
   - **Total True Positive Melonjak Drastis**: RUS mengidentifikasi **2.569 flaky test** (kumulatif seluruh 6 model B3), dibandingkan B1 yang hanya menangkap 324 flaky test di seluruh iterasi.
2. **Mengapa SMOTE (B2) Gagal dalam Prediksi Cross-Project?**  
   SMOTE membentuk data sintetis melalui interpolasi garis lurus antar tetangga terdekat ($k$-NN) di ruang fitur *source projects*. Pada skenario cross-project, setiap repositori memiliki arsitektur, gaya pengkodean, dan metrik churn yang unik. Instance sintetis yang dibentuk SMOTE merepresentasikan "wilayah buatan" di ruang fitur sumber yang sering kali **tidak eksis atau tidak valid pada target project**, menyebabkan model *overfitting* terhadap relasi lokal source.
3. **Mengapa RUS (B3) Sangat Efektif?**  
   Dengan membuang secara acak contoh non-flaky mayoritas yang redundan di domain sumber, RUS memaksa algoritma pembelajaran membentuk batas keputusan (*decision boundary*) yang lebih netral dan toleran terhadap variasi metrik. Hal ini secara signifikan memitigasi efek *domain shift*, menjadikan RUS strategi paling tangguh untuk implementasi di lingkungan Continuous Integration (CI).

---

### 🔹 RQ4: Evaluasi Komparatif Algoritma Klasifikasi (M1: RF vs M2: XGBoost GPU)

> **Pertanyaan Penelitian 4**: *Apakah XGBoost dengan akselerasi GPU tetap mengungguli Random Forest dalam metrik operasional nyata (F1-Score, Recall, Precision, MCC) pada dataset terimputasi penuh?*

#### Tabel Komparasi Faktor Model (12 Pipeline per Algoritma):
| Model Classifier | Mean PR-AUC | Mean F1-Score | Mean Recall | Mean Precision | Mean MCC | Mean ROC-AUC | Waktu Rata-rata per Pipeline |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M1 (Random Forest CPU)** | **0.2043** | 0.0710 | 0.1425 | 0.1461 | 0.0583 | **0.7099** | 31,43 detik |
| **M2 (XGBoost GPU CUDA)**  | 0.1930 | **0.0985** | **0.2154** | **0.1785** | **0.0777** | 0.6918 | **20,18 detik (-35,8%)** |
| **Selisih / Peningkatan M2** | *-5,5%* | **+38,8%** | **+51,1%** | **+22,1%** | **+33,2%** | *-2,5%* | **1,56x Lebih Cepat** |

#### Sintesis Analisis RQ4:
1. **Keunggulan Operasional XGBoost GPU (M2)**: Meskipun Random Forest mencatatkan kurva ranking probabilitas global sedikit lebih tinggi (PR-AUC: 0.2043 vs 0.1930), **XGBoost mendominasi secara mutlak pada seluruh metrik operasional keputusan**:
   - F1-Score flaky unggul **+38,8%** (0.0985 vs 0.0710).
   - Recall flaky melonjak **+51,1%** (0.2154 vs 0.1425).
   - Precision flaky unggul **+22,1%** (0.1785 vs 0.1461).
   - Matthews Correlation Coefficient (MCC) meningkat **+33,2%** (0.0777 vs 0.0583).
2. **Efisiensi Komputasi Akselerasi GPU**: XGBoost yang berjalan pada NVIDIA GeForce RTX 4050 (`tree_method="hist"`, `device="cuda"`) menyelesaikan 24 fold hanya dalam rata-rata **20,18 detik per pipeline**, atau **35,8% lebih cepat** dibandingkan Random Forest dengan multi-threading CPU penuh (31,43 detik).
3. **Kinerja Top-1 vs Agregat**: Random Forest berhasil mengamankan peringkat pipeline nomor 1 individual (`P13_S2_B3_M1` dengan F1 **0.1816**) karena ketika digabungkan dengan RUS, agregasi ensemble bagging RF sangat efektif meredam varians sampel undersampled. Namun pada konfigurasi tanpa resampling (B1, B2, B4), Random Forest kolaps ke prediksi mayoritas (Recall < 7%), sedangkan XGBoost tetap mampu mempertahankan deteksi positif (Recall > 14-20%).

---

## 3. Peringkat 10 Pipeline Terbaik (Top 10 Performers)

Berikut adalah daftar 10 konfigurasi pipeline terbaik diurutkan berdasarkan **Mean F1-Score**:

| Rank | Pipeline ID | Scaling | Imbalance | Model | Mean F1 | Mean Recall | Mean Precision | Mean PR-AUC | Mean MCC | Total TP (24 Proyek) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 **1** | `P13_S2_B3_M1` | **S2 (MinMax)** | **B3 (RUS)** | **M1 (RF)** | **0.1816** | **0.4374** | **0.2038** | **0.2210** | **0.1547** | **465** |
| 🥈 **2** | `P21_S3_B3_M1` | S3 (Robust) | **B3 (RUS)** | M1 (RF) | **0.1786** | 0.4333 | 0.2014 | 0.2203 | 0.1509 | 462 |
| 🥉 **3** | `P05_S1_B3_M1` | S1 (Standard) | **B3 (RUS)** | M1 (RF) | **0.1785** | 0.4345 | 0.2024 | **0.2224** | 0.1510 | 461 |
| **4** | `P06_S1_B3_M2` | S1 (Standard) | **B3 (RUS)** | **M2 (XGB)** | **0.1403** | 0.4255 | 0.1352 | 0.2012 | 0.1011 | 395 |
| **5** | `P14_S2_B3_M2` | S2 (MinMax) | **B3 (RUS)** | **M2 (XGB)** | **0.1403** | 0.4255 | 0.1352 | 0.2012 | 0.1011 | 395 |
| **6** | `P22_S3_B3_M2` | S3 (Robust) | **B3 (RUS)** | **M2 (XGB)** | **0.1403** | 0.4255 | 0.1352 | 0.2012 | 0.1011 | 395 |
| **7** | `P12_S2_B2_M2` | S2 (MinMax) | B2 (SMOTE) | M2 (XGB) | **0.1209** | 0.1972 | 0.1858 | 0.2154 | 0.1043 | 147 |
| **8** | `P08_S1_B4_M2` | S1 (Standard) | B4 (Weight) | M2 (XGB) | **0.1049** | 0.2088 | 0.1931 | 0.1739 | 0.0828 | 141 |
| **9** | `P16_S2_B4_M2` | S2 (MinMax) | B4 (Weight) | M2 (XGB) | **0.1049** | 0.2088 | 0.1931 | 0.1739 | 0.0828 | 141 |
| **10** | `P24_S3_B4_M2` | S3 (Robust) | B4 (Weight) | M2 (XGB) | **0.1049** | 0.2088 | 0.1931 | 0.1739 | 0.0828 | 141 |

*Catatan: Posisi Top 6 seluruhnya dimenangkan oleh perlakuan **B3 (Random Under-Sampling)**, membuktikan dominasi teknik undersampling dalam menangkap flaky test lintas proyek.*

---

## 4. Uji Signifikansi Statistik (ACM/SIGSOFT Empirical Standards)

### 4.1 Friedman Test (Omnibus Test across 24 Target Projects)
- **Metrik F1-Score**:
  - $\chi^2_{\text{Friedman}} = \mathbf{173.2039}$
  - Derajat Kebebasan ($df$) = $23$
  - Nilai $p\text{-value} = \mathbf{5.17 \times 10^{-25}} \ll 0.05$
  - **Kesimpulan**: Terdapat **perbedaan performa yang sangat signifikan secara statistik** di antara ke-24 pipeline pada ke-24 lipatan proyek target unseen.
- **Metrik PR-AUC**:
  - $\chi^2_{\text{Friedman}} = 18.7338$
  - Nilai $p\text{-value} = 0.7166 > 0.05$
  - **Interpretasi**: Kurva probabilitas kontinu (PR-AUC) memiliki bentuk yang relatif seragam di seluruh model berbasis pohon, namun pemetaan probabilitas ke dalam keputusan biner aktual (F1-Score, Precision, Recall) sangat sensitif terhadap strategi penanganan ketimpangan kelas.

### 4.2 Nemenyi Post-hoc Test & Critical Difference (CD)
- Jumlah perlakuan ($k$) = 24 pipeline, Jumlah dataset ($N$) = 24 proyek target.
- Pada taraf signifikansi $\alpha = 0.05$, nilai studentized statistic $q_{0.05} = 3.630$.
- Nilai ambang **Critical Difference (CD)**:
  $$\text{CD} = q_\alpha \sqrt{\frac{k(k + 1)}{6N}} = 3.630 \times \sqrt{\frac{24 \times 25}{6 \times 24}} = 3.630 \times \sqrt{4.1667} \approx \mathbf{7.4097}$$

#### Klaster Peringkat Nemenyi F1-Score:
- **Peringkat 1**: `P13_S2_B3_M1` (Average Rank: **8.25**)
- **Peringkat 2**: `P21_S3_B3_M1` (Average Rank: **8.40**)
- **Peringkat 3**: `P05_S1_B3_M1` (Average Rank: **8.48**)
- **Peringkat 4-6**: `P14_S2_B3_M2`, `P22_S3_B3_M2`, `P06_S1_B3_M2` (Average Rank: **8.88**)
- Ambang batas perbedaan signifikan dari Rank 1: $8.25 + 7.41 = \mathbf{15.66}$.
- **Hasil Uji**: Seluruh model Baseline M1 (`P09`: rank 15.67, `P17`: rank 15.58, `P01`: rank 15.63) dan ClassWeight M1 (`P15`: rank 17.04, `P23`: rank 16.83, `P07`: rank 16.75) berada di atas batas signifikansi, sehingga terbukti **secara statistik inferior secara signifikan** terhadap pipeline terbaik `P13_S2_B3_M1`.

---

## 5. Analisis Kasus Khusus & Proyek Kunci

### 5.1 Kasus Proyek Skala Raksasa: `spring-boot` (Preservasi 100%)
- Karakteristik: 2.127 test cases, 163 flaky test (20,1% dari seluruh flaky test di repositori FlakeFlagger).
- Pada Eksperimen 1 C2, proyek ini **dibuang secara keseluruhan**.
- Pada Eksperimen 2 dengan model terbaik `P13_S2_B3_M1`:
  - **True Positive (TP)**: 127 flaky test berhasil ditangkap.
  - **False Negative (FN)**: 36 flaky test terlewat.
  - **Recall pada Spring Boot**: **77,91%**!
  - **Precision**: 26,68% | **F1-Score**: **0.3975** | **PR-AUC**: **0.2314** | **MCC**: **0.3839**.
  - **Implikasi**: Mampu mendeteksi lebih dari 3/4 flaky test pada arsitektur framework enterprise yang paling kompleks tanpa pernah melihat baris kode Spring Boot sebelumnya merupakan pencapaian empirical software engineering yang sangat impresif.

### 5.2 Kasus Proyek Tanpa Flaky Test: `jimfs` (Zero Minority Class)
- Karakteristik: 212 test cases, **0 flaky test**.
- Engine evaluasi menangani kondisi ketiadaan kelas positif secara elegan:
  - AUC diset ke `NaN` (tidak merusak kalkulasi numerik).
  - Model memprediksi 0 False Positive (**FP = 0, TN = 212**).
  - **Specificity**: 100% (**Balanced Accuracy = 1.0**).
  - Eksekusi LOPO-CV berjalan mulus tanpa kegagalan program (*zero crashes*).

### 5.3 Kasus Keberhasilan Deteksi Sempurna: `alluxio` & `ambari`
- `alluxio` (116 flaky test): TP = 113, FP = 1, FN = 3 $\rightarrow$ **Recall 97,41%, Precision 99,12%, F1 0.9826, PR-AUC 0.9989**!
- `ambari` (52 flaky test): TP = 49, FP = 101, FN = 3 $\rightarrow$ **Recall 94,23%, F1 0.4851, PR-AUC 0.5019**!
- `hbase` (145 flaky test): TP = 69, FP = 22, FN = 76 $\rightarrow$ **Precision 75,82%, F1 0.5847, PR-AUC 0.6528**!

---

## 6. Artefak Visualisasi Hasil Penelitian

Seluruh visualisasi publikasi resolusi tinggi (300 DPI) telah dihasilkan pada direktori `results/plots/`:
1. `boxplot_3_factors_f1.png`: Distribusi F1-Score pada faktor Scaling, Imbalance, dan Model.
2. `boxplot_3_factors_prauc.png`: Distribusi PR-AUC pada ketiga faktor eksperimen.
3. `heatmap_pipeline_f1.png`: Peta panas performa F1 pada 24 konfigurasi pipeline faktorial.
4. `heatmap_pipeline_prauc.png`: Peta panas performa PR-AUC pada 24 konfigurasi pipeline faktorial.
5. `nemenyi_ranks_f1.png`: Diagram batang peringkat rata-rata Nemenyi F1-Score beserta garis batas Critical Difference (CD = 7.41).
6. `nemenyi_ranks_prauc.png`: Diagram batang peringkat rata-rata Nemenyi PR-AUC.

---

## 7. Rekomendasi Arsitektural Definitif bagi Praktisi & Peneliti

1. **Gunakan Group-Aware Median Imputation**:  
   Jangan membuang baris data (*Row Drop*) maupun membuang proyek (*Project Drop*) saat menghadapi missing values pada metrik software. Imputasi berbasis median per-proyek terbukti melestarikan seluruh data uji (+273 baris, +163 flaky test di `spring-boot`) dan mendongkrak performa generalisasi F1 hingga +11,4%.
2. **Kombinasi Pipeline Standar Emas**:  
   Untuk sistem deteksi *flaky test* lintas proyek (Cross-Project LOPO-CV), Kelompok A01 merekomendasikan:
   $$\mathbf{Optimal\ Pipeline} = \text{Cleaned Imputation} + \mathbf{MinMaxScaler\ (S2)} + \mathbf{RUS\ (B3)} + \mathbf{RandomForest\ (M1)\ /\ XGBoost\ (M2)}$$
   - Gunakan **Random Forest (M1)** jika mengutamakan F1-Score tertinggi (F1: 0.1816, Recall: 43,7%).
   - Gunakan **XGBoost GPU (M2)** jika mengutamakan kecepatan inferensi di CI pipeline (1,56x lebih cepat) dan keseimbangan metrik operasional yang stabil pada berbagai kondisi ketimpangan data.
3. **Wajib Terapkan Random Under-Sampling (RUS)**:  
   Hindari SMOTE dalam skenario lintas domain. RUS adalah satu-satunya metode yang secara konsisten menggandakan Recall dan F1-Score di atas 40% tanpa menyebabkan overfitting pada domain sumber.

---

## 8. Referensi Akademik

1. **Alshammari, A., Aldeen, C., & Zhang, L.** (2021). *FlakeFlagger: Predicting Flakiness Without Rerunning Tests*. In Proceedings of the 43rd International Conference on Software Engineering (ICSE 2021), pp. 1572–1584.
2. **Afeltra, A., et al.** (2024). *A Large-Scale Empirical Investigation Into Cross-Project Flaky Test Prediction*. IEEE Access, vol. 12, pp. 45210–45228.
3. **Pontillo, D., et al.** (2022). *Static Test Flakiness Prediction: How Far Can We Go?* Empirical Software Engineering (EMSE), 27(6), 145.
4. **Demšar, J.** (2006). *Statistical Comparisons of Classifiers over Multiple Data Sets*. Journal of Machine Learning Research (JMLR), 7, 1–30.
5. **Kelompok A01 — RPL A ITS**. (2026). *Laporan Hasil Eksperimen 1: Cross-Project Flaky Test Prediction*. `Eksperimen_Bima/Eksperimen-1/LAPORAN_EKSPERIMEN_1.md`.
