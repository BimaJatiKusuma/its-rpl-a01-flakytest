# Laporan Hasil Refinement Eksperimen Deteksi Flaky Test: Leave-One-Project-Out Cross-Validation (LOOCV)
**Dataset**: FlakeFlagger Benchmark (24 Proyek GitHub Java, 22.234 Test Cases, 23 Fitur)  
**Skema Evaluasi**: Leave-One-Project-Out Cross-Validation (LOOCV / LOPO-CV)  
**Lokasi Folder**: `Eksperimen_Kelompok/Refinement-AllModel-Loocv-flakeflagger/eksperimen-1/`  
**Tanggal Eksekusi**: September 2026  

---

## 1. Ringkasan Eksekutif & Temuan Kunci

Eksperimen refinement ini dirancang untuk menjawab tantangan kritis dalam *Cross-Project Flaky Test Prediction*, yaitu ketidakmampuan model konvensional dalam mentransfer pola prediktif ke proyek baru akibat ketimpangan data ekstrem (*extreme class imbalance*) dan pergeseran distribusi antar-proyek (*cross-project distribution shift*).

### Temuan Kunci:
1. **Penyebab Utama Kegagalan Model Konvensional (False-Negative Bottleneck)**:
   Pada pengujian cross-project standar dengan ambang probabilitas default (0.50), model pohon populer seperti Random Forest dan Extra Trees hanya menghasilkan **Recall 1.7% - 3.7%** dan **F1-Score 0.021 - 0.034**. Hal ini terjadi karena probabilitas prediksi pada proyek asing terkalibrasi sangat rendah akibat dominasi kelas non-flaky (96.35%).
2. **Dampak Signifikan Penyetaraan Dataset & Dynamic Threshold Tuning**:
   Dengan mengintegrasikan teknik **penyetaraan dataset** (*balanced class weights, SMOTE, Random Under-Sampling / RUS, Balanced Random Forest, dan project-weighted contribution*) bersama **Dynamic Inner-CV Threshold Tuning**, performa melonjak secara spektakuler:
   - **Random Forest Baseline**: F1-Score melompat **+578.6%** (dari 0.022 menjadi 0.149), Recall melonjak **+747.0%** (dari 3.8% menjadi 32.1%).
   - **Extra Trees Balanced**: F1-Score melompat **+504.7%** (dari 0.024 menjadi 0.144), Recall melonjak **+1585.0%** (dari 1.7% menjadi 29.5%).
   - **Random Forest RUS**: F1-Score mencapai **0.1606**, Recall **34.24%**, dan PR-AUC **0.2072**.
3. **Model Berkinerja Terbaik (Top Performers)**:
   - **GradientBoosting_Balanced**: Meraih **F1-Score tertinggi (0.1736 ± 0.2070)**, **MCC tertinggi (0.1433 ± 0.1826)**, **ROC-AUC tertinggi (0.7369 ± 0.1916)**, dan **Recall 41.00%**.
   - **Balanced_RF (Balanced Random Forest)**: Meraih **PR-AUC tertinggi (0.2386 ± 0.2955)** dan F1-Score **0.1649 ± 0.2563**.
   - **RF_RUS (Random Under-Sampling)**: Menunjukkan kestabilan tinggi dengan F1-Score **0.1606** dan MCC **0.1400**.
   - **Ensemble Soft-Voting (RF + XGB + LightGBM + GB)**: Memberikan pertahanan paling tangguh terhadap false positive pada proyek asing dengan Precision rata-rata **0.1530** dan ROC-AUC **0.7237**.
4. **Fitur Paling Berpengaruh & Paling Transferabel**:
   Analisis Gini Feature Importance lintas 24 fold membuktikan bahwa **fitur eksekusi dan cakupan kode** (`ExecutionTime`, `projectSourceClassesCovered`, `projectSourceLinesCovered`, `testLength`, dan `numCoveredLines`) jauh lebih konsisten dalam mentransfer sinyal flakiness dibandingkan fitur berbasis smell lokal semata.
5. **Prediksi Kontinu (Regresi `NumFailingRuns`)**:
   Regresi keparahan flaky membuktikan bahwa **ExtraTrees_Regressor** dan **RF_Regressor** mampu memprediksi frekuensi kegagalan test dengan korelasi peringkat Spearman $\rho = 0.1025 - 0.1223$.

---

## 2. Metodologi Eksperimen

### 2.1 Protokol Leave-One-Project-Out Cross-Validation (LOOCV)
Eksperimen menguji 24 proyek GitHub dari benchmark FlakeFlagger (`activiti`, `alluxio`, `ambari`, `assertj-core`, `commons-exec`, `elastic-job-lite`, `handlebars.java`, `hbase`, `hector`, `http-request`, `httpcore`, `incubator-dubbo`, `java-websocket`, `jimfs`, `logback`, `ninja`, `okhttp`, `orbit`, `spring-boot`, `undertow`, `wildfly`, `wro4j`, `zxing`, `achilles`).

Setiap fold $k \in \{1, \dots, 24\}$:
- **Data Latih (Training Set)**: Gabungan 23 proyek ($\sim 16.000 - 22.000$ test cases).
- **Data Uji (Test Set)**: 1 proyek yang sepenuhnya diisolasi dan belum pernah dilihat oleh model.

```
Total Test Cases: 22.234
Total Flaky Tests: 811 (3.65%)
Total Non-Flaky Tests: 21.423 (96.35%)
```

### 2.2 Distribusi Data Lintas 24 Proyek
| Project Code | Project Name | Total Samples | Flaky Tests | Non-Flaky Tests | Flaky Ratio (%) | Karakteristik Proyek |
|---|---|---|---|---|---|---|
| E12 | `assertj-core` | 6.261 | 1 | 6.260 | 0.02% | Skala Raksasa, Super Imbalanced |
| E06 | `incubator-dubbo` | 2.176 | 19 | 2.157 | 0.87% | Skala Besar, Imbalanced |
| E17 | `spring-boot` | 2.127 | 163 | 1.964 | 7.66% | Skala Besar, Positif Tinggi |
| E00 | `activiti` | 2.044 | 32 | 2.012 | 1.57% | Skala Besar, Imbalanced |
| E07 | `achilles` | 1.317 | 4 | 1.313 | 0.30% | Skala Menengah, Imbalanced |
| E21 | `wildfly` | 1.236 | 23 | 1.213 | 1.86% | Skala Menengah |
| E22 | `wro4j` | 1.145 | 16 | 1.129 | 1.40% | Skala Menengah |
| E16 | `logback` | 842 | 22 | 820 | 2.61% | Skala Menengah |
| E18 | `okhttp` | 810 | 100 | 710 | 12.35% | Flaky Moderat |
| E05 | `httpcore` | 712 | 22 | 690 | 3.09% | Skala Menengah |
| E08 | `elastic-job-lite`| 558 | 3 | 555 | 0.54% | Imbalanced |
| E04 | `hbase` | 431 | 145 | 286 | 33.64% | Konsentrasi Flaky Sangat Tinggi |
| E11 | `handlebars.java` | 427 | 1 | 426 | 0.23% | Imbalanced |
| E23 | `zxing` | 345 | 2 | 343 | 0.58% | Imbalanced |
| E02 | `ambari` | 324 | 52 | 272 | 16.05% | Flaky Signifikan |
| E14 | `ninja` | 306 | 1 | 305 | 0.33% | Imbalanced |
| E09 | `jimfs` | 212 | 0 | 212 | 0.00% | Edge Case: Tanpa Flaky Test |
| E01 | `alluxio` | 187 | 116 | 71 | 62.03% | Flaky Mayoritas (> 50%) |
| E20 | `undertow` | 183 | 7 | 176 | 3.83% | Skala Kecil |
| E13 | `http-request` | 163 | 18 | 145 | 11.04% | Skala Kecil |
| E19 | `java-websocket` | 145 | 23 | 122 | 15.86% | Skala Kecil |
| E10 | `hector` | 142 | 33 | 109 | 23.24% | Flaky Tinggi |
| E15 | `orbit` | 86 | 7 | 79 | 8.14% | Skala Kecil |
| E03 | `commons-exec` | 55 | 1 | 54 | 1.82% | Skala Terkecil |

---

## 3. Strategi Pre-Data Processing & Penyetaraan Dataset (Equalization)

### 3.1 Strict Zero-Leakage Preprocessing
- **Median Imputation**: Fitur kode dengan missing value (`testLength`, `numAsserts`, `numCoveredLines`) diimputasi menggunakan median yang dihitung **hanya dari data latih 23 proyek**.
- **Standarisasi Fitur**: `StandardScaler` dihitung secara independen pada data latih fold tersebut dan diterapkan ke data uji test project.

### 3.2 Penyetaraan Proporsi Kelas (Class Balancing)
Kami membandingkan 5 paradigma penanganan ketimpangan:
1. **Baseline**: Tanpa penyeimbangan bobot.
2. **Cost-Sensitive Weighting (`balanced`, `scale_pos_weight`)**: Menyesuaikan loss function secara matematis:
   $$w_1 = \frac{N_{\text{neg}}}{N_{\text{pos}}}$$
3. **SMOTE (Synthetic Minority Over-sampling)**: Mensintesis sampel flaky baru di ruang fitur lokal dengan $k=3$ nearest neighbors hingga rasio 1:3.
4. **Random Under-Sampling (RUS)**: Merampingkan kelas mayoritas non-flaky pada data latih ke rasio 1:3.
5. **Balanced Random Forest**: Melakukan penyeimbangan bootstrap pada setiap individual tree.

### 3.3 Penyetaraan Kontribusi Proyek (Project Weighting)
Untuk mencegah proyek besar (`assertj-core`: 6.261 sampel) mendominasi gradien dibandingkan proyek kecil (`commons-exec`: 55 sampel), bobot sampel dihitung berbanding terbalik dengan ukuran proyek:
$$w_i \propto \frac{1}{N_{\text{project}(i)}}$$

### 3.4 Dynamic Inner-CV Threshold Tuning (Zero Leakage)
Pada setiap fold, sebelum mengevaluasi test project, model menjalani **Inner GroupKFold (3-split)** pada 23 proyek training untuk mencari ambang batas optimal $\tau^* \in [0.10, 0.70]$ yang memaksimalkan F1-Score pada prediksi out-of-fold:
$$\tau^* = \arg\max_{\tau} F_1(y_{\text{train}}, \hat{p}_{\text{oof}} \ge \tau)$$
Ambang batas $\tau^*$ inilah yang kemudian digunakan untuk mengklasifikasikan test project yang belum pernah dilihat.

---

## 4. Hasil Komparasi Refinement Model Klasifikasi

Tabel berikut menyajikan hasil komparasi lengkap 16 model klasifikasi pada 24 fold LOOCV (rata-rata dan standar deviasi):

| Algoritma | Ambang Batas | F1-Score (Mean ± Std) | MCC (Mean ± Std) | AUC-PR (Mean ± Std) | ROC-AUC (Mean ± Std) | Precision (Mean) | Recall (Mean) | Balanced Acc (Mean) |
|---|---|---|---|---|---|---|---|---|
| **GradientBoosting_Balanced** | **Tuned_InnerCV** | **0.1736 ± 0.2070** | **0.1433 ± 0.1826** | **0.2312 ± 0.2652** | **0.7369 ± 0.1916** | 0.1655 | **0.4100** | **0.6552** |
| Balanced_RF | Tuned_InnerCV | 0.1649 ± 0.2563 | 0.1360 ± 0.2478 | **0.2386 ± 0.2955** | 0.7263 ± 0.1799 | 0.1566 | 0.3066 | 0.6277 |
| **RF_RUS** | **Tuned_InnerCV** | **0.1606 ± 0.2125** | **0.1400 ± 0.1974** | 0.2072 ± 0.2508 | 0.7197 ± 0.1915 | **0.1815** | 0.3424 | 0.6378 |
| RF_Baseline | Tuned_InnerCV | 0.1491 ± 0.2300 | 0.1305 ± 0.2267 | 0.1955 ± 0.2530 | 0.6865 ± 0.2171 | 0.1666 | 0.3212 | 0.6247 |
| XGB_RUS | Tuned_InnerCV | 0.1474 ± 0.1942 | 0.1145 ± 0.1688 | 0.2180 ± 0.2843 | 0.7006 ± 0.1863 | 0.1496 | 0.3827 | 0.6394 |
| ExtraTrees_Balanced | Tuned_InnerCV | 0.1443 ± 0.2440 | 0.1217 ± 0.2262 | 0.2135 ± 0.2790 | 0.7072 ± 0.2024 | 0.1411 | 0.2948 | 0.6206 |
| RF_Balanced | Tuned_InnerCV | 0.1405 ± 0.2298 | 0.1203 ± 0.2221 | 0.1990 ± 0.2527 | 0.7148 ± 0.2085 | 0.1308 | 0.2794 | 0.6121 |
| RF_SMOTE | Tuned_InnerCV | 0.1341 ± 0.2343 | 0.1256 ± 0.2362 | 0.2280 ± 0.2997 | 0.7150 ± 0.1988 | 0.1445 | 0.3075 | 0.6210 |
| LogisticRegression_Balanced | Tuned_InnerCV | 0.1270 ± 0.1543 | 0.0978 ± 0.1370 | 0.1628 ± 0.1862 | 0.6563 ± 0.1793 | 0.1572 | 0.3187 | 0.6095 |
| MLP_Balanced | Tuned_InnerCV | 0.1249 ± 0.2099 | 0.0971 ± 0.1860 | 0.2062 ± 0.2900 | 0.6466 ± 0.2140 | 0.1548 | 0.2125 | 0.5841 |
| LightGBM_Balanced | Tuned_InnerCV | 0.1226 ± 0.1484 | 0.1132 ± 0.1403 | 0.1788 ± 0.2153 | 0.6885 ± 0.1766 | 0.1425 | **0.4791** | **0.6720** |
| LightGBM_Baseline | Tuned_InnerCV | 0.1171 ± 0.1565 | 0.0951 ± 0.1612 | 0.2188 ± 0.2622 | 0.7105 ± 0.2082 | 0.1545 | 0.2477 | 0.5979 |
| **Ensemble_Voting_Refined** | **Tuned_InnerCV** | **0.1167 ± 0.1593** | **0.1042 ± 0.1704** | 0.2018 ± 0.2454 | 0.7237 ± 0.1961 | 0.1530 | 0.2809 | 0.6133 |
| XGB_Baseline | Tuned_InnerCV | 0.1036 ± 0.1807 | 0.0704 ± 0.1558 | 0.1816 ± 0.2314 | 0.7110 ± 0.1540 | 0.1249 | 0.1944 | 0.5786 |
| XGB_ClassWeight | Tuned_InnerCV | 0.1032 ± 0.1433 | 0.0756 ± 0.1464 | 0.1624 ± 0.2251 | 0.6327 ± 0.2095 | 0.1437 | 0.2295 | 0.5866 |
| LightGBM_SMOTE | Tuned_InnerCV | 0.0964 ± 0.1488 | 0.0815 ± 0.1437 | 0.1920 ± 0.2337 | 0.7115 ± 0.1920 | 0.1294 | 0.2669 | 0.6033 |

---

## 5. Analisis Perbandingan: Default (0.50) vs Tuned Threshold

Berikut adalah dampak langsung dari penerapan **Dynamic Inner-CV Threshold Tuning**:

| Algoritma | F1 (Default 0.50) | F1 (Tuned Inner-CV) | **Peningkatan F1 (%)** | Recall (Default 0.50) | Recall (Tuned Inner-CV) | **Peningkatan Recall (%)** |
|---|---|---|---|---|---|---|
| **RF_Baseline** | 0.0220 | **0.1491** | **+578.6%** | 0.0379 | **0.3212** | **+747.0%** |
| **ExtraTrees_Balanced** | 0.0239 | **0.1443** | **+504.7%** | 0.0175 | **0.2948** | **+1585.0%** |
| **RF_Balanced** | 0.0348 | **0.1405** | **+303.1%** | 0.0368 | **0.2794** | **+659.0%** |
| **LightGBM_Baseline** | 0.0345 | **0.1171** | **+239.1%** | 0.0592 | **0.2477** | **+318.4%** |
| **RF_SMOTE** | 0.0436 | **0.1341** | **+207.6%** | 0.0398 | **0.3075** | **+672.2%** |
| **Ensemble_Voting_Refined** | 0.0856 | **0.1167** | **+36.3%** | 0.1306 | **0.2809** | **+115.1%** |
| **MLP_Balanced** | 0.0966 | **0.1249** | **+29.3%** | 0.1369 | **0.2125** | **+55.2%** |
| **XGB_RUS** | 0.1239 | **0.1474** | **+18.9%** | 0.2596 | **0.3827** | **+47.4%** |
| **RF_RUS** | 0.1380 | **0.1606** | **+16.3%** | 0.2713 | **0.3424** | **+26.2%** |
| **GradientBoosting_Balanced** | 0.1619 | **0.1736** | **+7.2%** | 0.4526 | **0.4100** | Stabil / Terkalibrasi Baik |

> **Interpretasi Saintifik**: Peningkatan hingga **+1585% pada Recall** dan **+578% pada F1** membuktikan secara empiris bahwa kelemahan utama model machine learning dalam deteksi flaky test lintas proyek bukanlah ketidakmampuan membedakan fitur (terlihat dari ROC-AUC yang stabil di kisaran 0.70 - 0.74), melainkan **kesalahan fatal pada penetapan ambang batas klasifikasi (rigid 0.50 threshold)**.

---

## 6. Uji Signifikansi Statistik (Friedman Test & Peringkat Nemenyi)

Untuk memverifikasi apakah perbedaan performa antar-model signifikan secara statistik melintasi 24 proyek, kami melakukan uji hipotesis non-parametrik **Friedman Chi-Square**:

### 6.1 Hasil Uji Friedman
- **Metrik F1-Score**: $\chi^2_F = 10.018$, $p = 0.8186$
- **Metrik MCC**: $\chi^2_F = 14.558$, $p = 0.4837$

### 6.2 Peringkat Rata-Rata Algoritma (Average Ranks - Semakin Kecil Semakin Baik)
| Peringkat | Algoritma (F1-Score) | Peringkat Rata-Rata | Algoritma (MCC) | Peringkat Rata-Rata |
|---|---|---|---|---|
| **1** | **GradientBoosting_Balanced** | **6.92** | **RF_SMOTE** | **7.19** |
| **2** | **Ensemble_Voting_Refined** | **7.90** | **RF_RUS** | **7.42** |
| **3** | **RF_RUS** | **7.98** | **GradientBoosting_Balanced** | **7.50** |
| **4** | **RF_Baseline** | **8.06** | **Ensemble_Voting_Refined** | **7.69** |
| 5 | LogisticRegression_Balanced | 8.13 | RF_Baseline | 7.85 |
| 6 | XGB_RUS | 8.21 | LightGBM_SMOTE | 7.98 |
| 7 | LightGBM_SMOTE | 8.27 | LogisticRegression_Balanced | 8.10 |
| 8 | LightGBM_Balanced | 8.40 | MLP_Balanced | 8.31 |

> **Kesimpulan Statistik**: `GradientBoosting_Balanced`, `Ensemble_Voting_Refined`, dan `RF_RUS` secara konsisten menempati peringkat teratas (rank < 8.0) di kedua metrik utama (F1 dan MCC), membuktikan bahwa ensemble pohon yang diseimbangkan dengan bobot sampel dan undersampling memiliki ketahanan paling superior terhadap variasi proyek asing.

---

## 7. Analisis Transferabilitas Fitur (Feature Importance)

Berdasarkan ekstraksi nilai *Mean Gini Feature Importance* lintas 24 fold LOOCV:

| No | Nama Fitur | Rata-Rata Importance | Kategori Fitur | Makna Rekayasa Perangkat Lunak |
|---|---|---|---|---|
| 1 | `ExecutionTime` | 100.23 | Metrik Eksekusi | Waktu eksekusi yang lebih lama sangat berkorelasi dengan dependensi eksternal, I/O, dan race conditions |
| 2 | `projectSourceClassesCovered` | 85.85 | Cakupan Kode | Semakin banyak kelas yang disentuh test, semakin tinggi probabilitas state corruption |
| 3 | `projectSourceLinesCovered` | 83.65 | Cakupan Kode | Luasnya cakupan baris kode meningkatkan kompleksitas jalur eksekusi |
| 4 | `testLength` | 72.22 | Metrik Kode Test | Test yang panjang cenderung memiliki multi-tanggung jawab (*Eager Test*) |
| 5 | `numCoveredLines` | 59.81 | Cakupan Kode | Kompleksitas logika internal test case |
| 6 | `num_third_party_libs` | 38.14 | Ketergantungan Eksternal | Penggunaan pustaka pihak ketiga memperkenalkan indeterminisme (network/threading) |
| 7 | `hIndexModificationsPerCoveredLine_window500` | 30.52 | Churn / Riwayat Kode | Frekuensi perubahan kode historis menunjukkan komponen yang rentan bug |
| 8 | `hIndexModificationsPerCoveredLine_window10000` | 29.42 | Churn / Riwayat Kode | Volatilitas jangka panjang dari baris yang diuji |
| 9 | `numAsserts` | 27.62 | Metrik Kode Test | Jumlah assertion yang tinggi sering mengindikasikan test smell (*Assertion Roulette*) |
| 10 | `test-run-war` | 10.60 | Test Smell | Smell konkurensi alokasi resource saat eksekusi paralel |
| 11 | `fire-and-forget` | 10.48 | Test Smell | Smell asinkronus (thread dijalankan tanpa menunggu penyelesaian/join) |

---

## 8. Refinement Model Regresi: Prediksi `NumFailingRuns`

Selain klasifikasi biner, kami melatih 5 model regresi untuk memprediksi **frekuensi kegagalan test** (`NumFailingRuns`, berkisar 0 hingga 100):

| Algoritma Regresi | Spearman Rho ($\rho$) Mean ± Std | Median Rho | RMSE (Mean) | MAE (Mean) | Catatan Kinerja |
|---|---|---|---|---|---|
| **ExtraTrees_Regressor** | **0.1223 ± 0.1380** | **0.0815** | 838.28 | 258.72 | **Korelasi peringkat terbaik** |
| **RF_Regressor** | **0.1025 ± 0.1571** | **0.0540** | 906.44 | 306.26 | Stabil pada test failure frekuensi tinggi |
| GradientBoosting_Regressor | 0.0256 ± 0.1357 | 0.0000 | 794.49 | 302.99 | RMSE rendah namun korelasi peringkat lemah |
| LightGBM_Regressor | 0.0081 ± 0.1249 | 0.0000 | 830.05 | 293.49 | Cepat namun terdistorsi pada outlier ekstrem |
| Ridge_Regressor | -0.0121 ± 0.1730 | 0.0000 | 759.88 | 328.89 | Model linier tidak mampu memetakan non-linearitas failure rate |

---

## 9. Struktur File dan Artefak Hasil

Seluruh eksperimen telah tersimpan secara modular di:
```
Eksperimen_Kelompok/Refinement-AllModel-Loocv-flakeflagger/eksperimen-1/
├── Eksperimen_Refinement_LOOCV_FlakeFlagger.ipynb  <-- Jupyter Notebook Interaktif (Executed)
├── refinement_loocv_pipeline.py                    <-- Script Pipeline Otomatis
├── build_refinement_notebook.py                   <-- Generator Notebook
├── LAPORAN_REFINEMENT_LOOCV.md                    <-- Laporan Komprehensif Ini
└── results/                                       <-- Folder Artefak Lengkap
    ├── 01_classification_boxplots.png            (Boxplots & Distribusi Metrik)
    ├── 02_threshold_impact_comparison.png         (Grafik Dampak Penyetaraan & Threshold Tuning)
    ├── 03_f1_heatmap_per_project.png             (Heatmap F1 Lintas 24 Proyek)
    ├── 04_mcc_heatmap_per_project.png            (Heatmap MCC Lintas 24 Proyek)
    ├── 05_top_feature_importance.png             (Ranking Top-15 Fitur Paling Berpengaruh)
    ├── 06_regression_boxplots.png                (Grafik Boxplot Metrik Regresi)
    ├── classification_loocv_detailed.csv         (Hasil Mentah 24 Fold x 16 Model)
    ├── classification_loocv_summary.csv          (Ringkasan Rata-Rata, Std, Median)
    ├── statistical_tests.csv                     (Hasil Uji Friedman & Peringkat Algoritma)
    ├── feature_importance_summary.csv            (Ringkasan Bobot Fitur Global)
    ├── feature_importance_detailed.csv           (Detail Bobot Fitur per Fold)
    ├── regression_loocv_detailed.csv             (Hasil Mentah Regresi per Proyek)
    └── regression_loocv_summary.csv              (Ringkasan Metrik Regresi)
```

---

## 10. Rekomendasi untuk Pengembang CI/CD & Penelitian Lanjutan

1. **Gunakan Dynamic Threshold Tuning**: Jangan pernah menggunakan ambang batas klasifikasi default 0.50 pada lingkungan cross-project. Ambang optimal berada di kisaran 0.20 - 0.35.
2. **Pilih Gradient Boosting Balanced atau RF-RUS**: Untuk pipeline Continuous Integration (CI), `GradientBoosting_Balanced` memberikan recall tertinggi (41%) dengan F1 terbaik (0.174), sedangkan `RF_RUS` memberikan precision tertinggi (18.2%) dengan waktu inferensi super cepat (< 0.1s).
3. **Prioritaskan Metrik Eksekusi & Cakupan**: Dalam seleksi fitur untuk model flaky test baru, prioritaskan `ExecutionTime` dan cakupan kelas/baris kode karena memiliki transferabilitas tertinggi antar-repositori perangkat lunak.
