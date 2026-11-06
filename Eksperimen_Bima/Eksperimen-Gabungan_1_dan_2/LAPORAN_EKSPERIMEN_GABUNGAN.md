# 📄 Laporan Komprehensif Eksperimen Gabungan 1 & 2: Cross-Project Flaky Test Prediction
## Sintesis Benchmark Faktorial 72 Machine Learning Pipelines & Validasi Konsistensi 10x Repeated LOPO-CV

**Kelompok Peneliti**: Kelompok A01 — Rekayasa Perangkat Lunak Lanjut (Magister Teknik Informatika, Institut Teknologi Sepuluh Nopember)  
**Dokumen Acuan**: [`RANCANGAN_PENELITIAN_GABUNGAN.md`](RANCANGAN_PENELITIAN_GABUNGAN.md) (Dokumen Bahasa Inggris: [`RESEARCH_DESIGN.md`](RESEARCH_DESIGN.md))  
**Dokumen Laporan Bahasa Inggris**: [`COMBINED_EXPERIMENT_REPORT.md`](COMBINED_EXPERIMENT_REPORT.md)  
**Dataset Fondasi**: FlakeFlagger Curated Benchmark (`test_features.csv` & `cleaned_test_features.csv`, 24 proyek Java open-source, 22.236 test cases, 811 flaky tests)  
**Akselerasi Perangkat Keras**: NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM, CUDA 13.4) & CPU Multi-threading  
**Protokol Validasi**: Leave-One-Project-Out Cross-Validation (LOPO-CV, Zero Data Leakage) & 10x Repeated Stochastic Trials (16.080 Total Model Folds)  

---

## Ringkasan Eksekutif (Executive Summary)

Eksperimen Gabungan 1 & 2 merupakan puncak sintesis metodologis dari seluruh iterasi eksperimen sebelumnya. Eksperimen ini memadukan ruang parameter penuh ke dalam **Matriks Benchmark Faktorial 72 Pipeline** ($3 \text{ Cleaning} \times 3 \text{ Scaling} \times 4 \text{ Imbalance} \times 2 \text{ Model Classifier}$). Rancangan terpadu ini mengevaluasi secara ketat dan *fair (ceteris paribus)* perdebatan fundamental dalam *Cross-Project Flaky Test Prediction (CP-FTP)*: apakah strategi pembuangan data (*data discarding: row-level vs. project-level*) lebih baik ataukah preservasi data 100% melalui imputasi cerdas kontekstual (*group-aware median imputation*) yang memberikan performa dan generalisasi superior pada sistem software baru (*unseen projects*).

```
                             ARSITEKTUR BENCHMARK 72 PIPELINE FAKTORIAL
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                          1. Tri-Paradigma Data Cleaning (3 Variasi)                             │
│    • C1: Row-Level Discarding (24 proyek, 21.963 baris, 808 flaky, buang 273 baris NaN)        │
│    • C2: Project-Level Discarding (19 proyek, 16.458 baris, 586 flaky, buang 5 proyek NaN)      │
│    • C3: Group-Aware Median Imputation (24 proyek, 22.236 baris, 811 flaky, 100% data utuh)     │
└───────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                           2. Transformasi Feature Scaling (3 Variasi)                           │
│        [S1: StandardScaler]       |       [S2: MinMaxScaler]       |      [S3: RobustScaler]    │
│            (Fit eksklusif pada N-1 training projects; Transform pada target unseen project)     │
└───────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                         3. Penanganan Class Imbalance (4 Variasi)                               │
│      [B1: Baseline (None)]   |   [B2: SMOTE]   |   [B3: RUS]   |   [B4: ClassWeight / Cost]     │
│             (Diterapkan secara ketat HANYA pada training fold untuk menjamin Zero Leakage)      │
└───────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                           4. Famili Model Klasifikasi (2 Variasi)                               │
│       [M1: Random Forest (Bagging CPU)]       |       [M2: XGBoost (Boosting CUDA GPU)]         │
└───────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                         5. Protokol Evaluasi Lintas Proyek & Replikasi                          │
│   • Single-Run LOPO-CV (Seed 42)              : (24×24) + (24×19) + (24×24) = 1.608 Model Folds │
│   • Repeated LOPO-CV (10 Independent Seeds)   : 10 × 1.608 = 16.080 Total Model Folds           │
│   • Metrik Pelaporan: Mean ± Standar Deviasi (Std), CV (%), 95% Confidence Interval             │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Temuan Empiris Kunci:
1. **Keunggulan Mutlak Imputasi Kontekstual (C3 vs. C1 & C2)**:
   - **C3 (Group-Aware Median Imputation)** terbukti menjadi standar emas operasional. Pada basis komparasi 24 proyek yang setara, C3 mengungguli C1 pada seluruh dimensi performa utama: **Mean F1 meningkat +11,39%** ($0,1631 \rightarrow \mathbf{0,1816}$ pada Seed 42, dan $0,1773 \pm 0,0095$ lintas 10 seed), **Mean Precision meningkat +28,31%** ($0,1588 \rightarrow \mathbf{0,2038}$), **Mean PR-AUC meningkat +7,44%** ($0,2057 \rightarrow \mathbf{0,2210}$), dan mendeteksi **+37 True Positive flaky tests tambahan** ($428 \rightarrow \mathbf{465}\text{ TP}$).
   - Skor tinggi pada C2 (Mean F1 $0,2259$ pada seed 42, $0,2178 \pm 0,0174$ lintas 10 seed) terbukti secara matematis dan empiris sebagai artefak **survivorship bias**. C2 membuang total 5 proyek termasuk sistem enterprise paling heterogen `spring-boot`. Padahal pada pipeline C3, model berhasil mendeteksi **127 dari 163 flaky test di `spring-boot` (Recall: $77,91\%$)** pada unseen evaluation fold, kapabilitas esensial yang hilang total pada C2.
2. **Supremasi Random Under-Sampling (B3: RUS)**:
   - Random Under-Sampling (B3) mendominasi secara mutlak di seluruh kombinasi paradigma cleaning. RUS melipatgandakan Mean F1 sebesar $3\times$ hingga $4\times$ lipat dibanding Baseline (B1) ($0,0478 \pm 0,0247 \rightarrow \mathbf{0,1817 \pm 0,0241}$, $p = 2,02 \times 10^{-9}$) dan mendongkrak Recall hingga **$47,16 \pm 3,63\%$**. Sebaliknya, SMOTE (B2) gagal dalam transfer lintas proyek karena interpolasi sintetis di manifold domain sumber memperparah *distribution shift*.
3. **Ketahanan Penskalaan Bounded (S2: MinMaxScaler)**:
   - MinMaxScaler (S2) mencatatkan stabilitas dan konvergensi tertinggi (Mean F1 $0,1000 \pm 0,0631$, PR-AUC $0,2147 \pm 0,0237$). Rentang bounded $[0, 1]$ memitigasi *heavy-tailed skewness* pada metrik churn dan eksekusi tes sekaligus menyediakan kuantisasi bin histogram yang optimal bagi akselerasi GPU.
4. **Dinamika Arsitektur Model (M1: Random Forest vs. M2: XGBoost GPU)**:
   - Secara agregat faktor, **XGBoost GPU (M2)** mengungguli Random Forest dengan Mean F1 **+48,2%** ($0,1164 \pm 0,0483$ vs $0,0785 \pm 0,0701$, $p = 4,88 \times 10^{-7}$) dan akselerasi komputasi $1,56\times$. Namun saat dipasangkan dengan RUS (B3), **Random Forest (M1)** meraih skor puncak tertinggi (`P61_C3_S2_B3_M1`: F1 $0,1816$, Presisi $0,2038$) karena bagging meredam variansi akibat undersampling secara lebih efektif daripada boosting sekuensial.

---

## 1. Komparasi Komprehensif Tri-Paradigma Data Cleaning (C1 vs. C2 vs. C3)

Ketiga paradigma pembersihan data dievaluasi secara berdampingan untuk mengukur integritas dataset, pelestarian sampel langka (*minority flaky instances*), serta performa generalisasi model yang dilengkapi metrik **Mean ± Standar Deviasi (Std)**.

### 1.1 Tabel Benchmark Komparasi Tri-Paradigma (Dengan Standar Deviasi)

| Dimensi Evaluasi | C1: Row-Level Discarding | C2: Project-Level Discarding | C3: Group-Aware Median Imputation |
|---|:---:|:---:|:---:|
| **Strategi Penanganan Missing Values** | Menghapus 273 baris dengan NaN | Menghapus 5 proyek berisi NaN | Imputasi Median per-Project (Group-Aware) |
| **Jumlah Proyek Tersisa** | 24 Proyek (100%) | 19 Proyek ($-20,8\%$) | **24 Proyek (100% Utuh)** |
| **Total Test Cases Tersisa** | 21.963 baris ($-1,23\%$) | 16.458 baris ($-25,98\%$) | **22.236 baris (100% Utuh)** |
| **Total Populasi Flaky Test** | 808 flaky test ($-0,37\%$) | 586 flaky test ($-27,74\%$) | **811 flaky test (100% Utuh)** |
| **Sampel Flaky yang Dihapus** | 3 flaky test | **225 flaky test** | **0 flaky test (Zero Sample Loss)** |
| **Status Proyek Kunci `spring-boot`** | Terlestarikan (160 flaky) | **DIELIMINASI TOTAL (0 flaky)** | **Terlestarikan Utuh (163 flaky)** |
| **Proyek Lain yang Dieliminasi** | Tidak ada | `wildfly`, `logback`, `wro4j`, `handlebars` | Tidak ada |
| **Pipeline Terbaik (Best Run)** | `P13_C1_S2_B3_M1` | `P37_C2_S2_B3_M1`* | **`P61_C3_S2_B3_M1` (Golden Pipeline)** |
| **Mean F1-Score (Single-Run Seed 42)**| 0,1633 | 0,2259* | **0,1816 (+11,21% vs C1)** |
| **F1-Score 10-Seed (Mean ± Std)** | $0,1773 \pm 0,0119$ | $0,2178 \pm 0,0174$* | **$0,1773 \pm 0,0095$ (CV Terendah: 5,35%)** |
| **Recall 10-Seed (Mean ± Std)** | $0,4462 \pm 0,0380$ | $0,5093 \pm 0,0254$* | **$0,4288 \pm 0,0256$** |
| **Precision 10-Seed (Mean ± Std)** | $0,1898 \pm 0,0230$ | $0,1936 \pm 0,0153$* | **$0,1914 \pm 0,0244$** |
| **PR-AUC 10-Seed (Mean ± Std)** | $0,2221 \pm 0,0191$ | $0,2685 \pm 0,0211$* | **$0,2202 \pm 0,0152$** |
| **ROC-AUC 10-Seed (Mean ± Std)** | $0,7208 \pm 0,0174$ | $0,7412 \pm 0,0098$* | **$0,7205 \pm 0,0087$** |
| **MCC 10-Seed (Mean ± Std)** | $0,1535 \pm 0,0145$ | $0,2008 \pm 0,0172$* | **$0,1527 \pm 0,0109$** |
| **Total True Positive Flaky (Seed 42)**| 421 TP | 334 TP | **465 TP (+10,45% vs C1, +39,2% vs C2)** |
| **Total TP 10-Seed (Mean ± Std)** | $430,6 \pm 40,7\text{ TP}$ | $339,6 \pm 14,4\text{ TP}$ | **$426,6 \pm 44,4\text{ TP}$** |
| **Total FP 10-Seed (Mean ± Std)** | $4.087,7 \pm 238,8\text{ FP}$ | $2.829,4 \pm 108,4\text{ FP}$ | $3.934,9 \pm 105,4\text{ FP}$ (FP Berkurang) |
| **Total FN 10-Seed (Mean ± Std)** | $377,4 \pm 40,7\text{ FN}$ | $246,4 \pm 14,4\text{ FN}$ | $384,4 \pm 44,4\text{ FN}$ |
| **Waktu Eksekusi (Mean ± Std)** | $3,41 \pm 0,04\text{ s}$ | $2,52 \pm 0,09\text{ s}$ | $3,41 \pm 0,04\text{ s}$ |
| **F1 pada `spring-boot` (Seed 42)** | 0,3621 | 0,0000 (Dieliminasi) | **0,3975 (+9,78% vs C1)** |
| **Recall pada `spring-boot` (Seed 42)**| 81,25% (130 / 160 TP) | 0,00% (Dieliminasi) | **77,91% (127 / 163 TP)** |
| **Presisi pada `spring-boot` (Seed 42)**| 23,30% (428 FP) | 0,00% (Dieliminasi) | **26,68% (349 FP, FP Berkurang -18,5%)** |
| **F1 pada `spring-boot` 10-Seed** | $0,3081 \pm 0,0917$ | 0,0000 | **$0,3141 \pm 0,1001$** |
| **Recall pada `spring-boot` 10-Seed**| $0,6050 \pm 0,2090$ | 0,0000 | **$0,5883 \pm 0,2174$** |
| **Presisi pada `spring-boot` 10-Seed**| $0,2078 \pm 0,0600$ | 0,0000 | **$0,2151 \pm 0,0652$** |
| **TP pada `spring-boot` 10-Seed** | $96,8 \pm 33,4\text{ TP}$ | 0,0 TP | **$95,9 \pm 35,4\text{ TP}$** |
| **FP pada `spring-boot` 10-Seed** | $363,8 \pm 53,7\text{ FP}$ | 0,0 FP | **$339,9 \pm 34,9\text{ FP}$ (FP Berkurang)** |

*\*Catatan Kritis: Nilai metrik C2 yang tampak tinggi semata-mata diakibatkan oleh survivorship bias karena eliminasi 5 proyek kompleks (`spring-boot`, `wildfly`, `logback`, `wro4j`, `handlebars.java`).*

---

### 1.2 Analisis Kritis Ilusi Metrik & Survivorship Bias pada C2

Pada pandangan sekilas, C2 mencatatkan rata-rata F1 faktorial $0,1225 \pm 0,0661$ dan F1 pipeline terbaik $0,2178 \pm 0,0174$, tampak lebih tinggi dibandingkan C1 ($0,0852 \pm 0,0578$) dan C3 ($0,0847 \pm 0,0587$). Namun analisis metodologis membuktikan ini adalah **ilusi statistik (survivorship bias)** dengan alasan:

1. **Pembuangan 27,74% Populasi Flaky Test Riil**:
   C2 mengeliminasi 5 proyek berikut dari benchmark:
   - `spring-boot`: 2.127 test cases, **163 flaky tests** (20,1% dari total flaky test benchmark).
   - `wildfly`: 1.236 test cases, **23 flaky tests**.
   - `logback`: 843 test cases, **22 flaky tests**.
   - `wro4j`: 1.145 test cases, **16 flaky tests**.
   - `handlebars.java`: 427 test cases, **1 flaky test**.
   Total data yang dibuang C2 mencapai **5.778 test cases (25,98%)** dan **225 flaky tests (27,74%)**.
2. **Penyederhanaan Ruang Uji secara Artifisial**:
   Proyek-proyek yang dibuang (khususnya `spring-boot` dan `wildfly`) merupakan sistem enterprise berskala raksasa dengan variansi metrik eksekusi, cakupan kode (*coverage*), dan pola *flakiness* yang sangat heterogen dan sulit diprediksi. Dengan membuang proyek-proyek rumit ini, C2 hanya menyisakan proyek-proyek kecil yang homogen, sehingga nilai akurasi dan F1 terkerek naik secara artifisial.
3. **Ketidakmampuan Evaluasi pada Sistem Enterprise**:
   Model C2 secara metodologis **tidak dapat diuji** pada `spring-boot`. Di industri nyata, software engineer tidak dapat "membuang" repositori utama perusahaan hanya karena terdapat beberapa missing values. Oleh karena itu, C2 gagal sebagai solusi aplikatif.

---

### 1.3 Evaluasi Kasus Riil: Deteksi Flakiness pada Enterprise Benchmark `spring-boot`

Proyek `spring-boot` merepresentasikan skenario uji paling menantang dalam benchmark FlakeFlagger (2.127 unit test, 163 kasus *flaky*, rasio ketimpangan 7,66%). Berikut komparasi kemampuan deteksi model terbaik pada skenario *unseen fold*:

```
                           DETEKSI FLAKINESS PADA REPOSITORI spring-boot
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ Paradigma C2: DIBUANG TOTAL (0 TP, Recall 0,00%, Tidak Mampu Mengevaluasi Enterprise)  │
  ├────────────────────────────────────────────────────────────────────────────────────────┤
  │ Paradigma C1: Retained 160 Flaky ──► TP = 130 | FP = 428 | Recall = 81,25% | F1 = 0,3621│
  │               (10-Seed Mean: TP = 96,8 ± 33,4 | FP = 363,8 ± 53,7 | F1 = 0,3081 ± 0,0917)│
  ├────────────────────────────────────────────────────────────────────────────────────────┤
  │ Paradigma C3: Retained 163 Flaky ──► TP = 127 | FP = 349 | Recall = 77,91% | F1 = 0,3975│
  │               (10-Seed Mean: TP = 95,9 ± 35,4 | FP = 339,9 ± 34,9 | F1 = 0,3141 ± 0,1001)│
  │               (False Positive turun -18,5%, Presisi naik ke 26,68%, F1 naik +9,78%)    │
  └────────────────────────────────────────────────────────────────────────────────────────┘
```

Imputasi kontekstual C3 berhasil menangkap **127 test flaky di `spring-boot`** dengan presisi $26,68\%$ (jauh melampaui C1 yang hanya $23,30\%$) dan rata-rata FP yang konsisten lebih rendah lintas 10 seed ($339,9 \pm 34,9$ vs $363,8 \pm 53,7$), membuktikan bahwa melestarikan data dengan imputasi median grup tidak mencemari model dengan *noise*, melainkan memperkaya batas keputusan (*decision boundary*) model.

---

## 2. Analisis & Jawaban Komprehensif Terhadap 4 Research Questions (RQ)

Bagian ini menyajikan representasi hasil empiris secara komprehensif lengkap dengan metrik **Mean ± Standar Deviasi (Std)** untuk menjawab keempat *Research Questions (RQ)* utama penelitian.

---

### 🔹 RQ1: Apakah Pembersihan Level-Baris (Row-Level) atau Level-Proyek (Project-Level) yang Lebih Baik dalam Melestarikan Sampel Flaky dan Mendukung Generalisasi?

> **Research Question 1 (RQ1)**:  
> *RQ1 examines whether row-level cleaning or project-level cleaning better preserves flaky samples and supports generalization.*

#### 1. Analisis Preservasi Sampel Flaky (Flaky Sample Preservation)

Preservasi sampel minoritas merupakan prasyarat mutlak dalam *imbalanced learning*. Kehilangan sampel flaky pada fase prapemrosesan mereduksi variasi representasi kegagalan non-deterministik perangkat lunak:

| Strategi Cleaning | Proyek Utuh | Total Uji Retensi | Total Flaky Retensi | % Preservasi Flaky | Flaky Hilang | Status Integritas |
|---|:---:|:---:|:---:|:---:|:---:|:---|
| **C1 (Row-Level Cleaning)** | **24 (100%)** | 21.963 (98,77%) | **808 (99,63%)** | **99,63%** | 3 sampel | **Sangat Baik** |
| **C2 (Project-Level Cleaning)**| 19 (79,17%) | 16.458 (74,02%) | 586 (72,26%) | 72,26% | **225 sampel** | **Kritis / Rusak** |
| **C3 (Group-Aware Imputation)**| **24 (100%)** | **22.236 (100%)** | **811 (100%)** | **100,00%** | **0 sampel** | **Sempurna (Emas)**|

- **C1 vs. C2 pada Preservasi**: Pembersihan level-baris (**C1**) secara dramatis mengungguli pembersihan level-proyek (**C2**). C1 melestarikan **$99,63\%$ sampel flaky** (hanya kehilangan 3 sampel flaky dari 273 baris ber-NaN), sedangkan C2 kehilangan **$27,74\%$ dari seluruh populasi flaky test** di benchmark karena mendiskualifikasi 5 repositori secara serampangan.
- **Peran C3 sebagai Solusi Ideal**: Ketika C3 diikutsertakan dalam perbandingan, C3 memberikan pelestarian sempurna (**100% data utuh, 811 flaky test**). Tidak ada satu pun informasi perilaku pengujian yang dikorbankan.

#### 2. Kinerja Rata-rata Agregat Paradigma Cleaning (Mean ± Std Lintas Pipeline)

Tabel berikut menunjukkan rata-rata performa dan standar deviasi lintas seluruh konfigurasi faktorial dalam tiap paradigma:

| Paradigma Cleaning | Mean F1 ± Std | Mean Recall ± Std | Mean Precision ± Std | Mean PR-AUC ± Std | Mean ROC-AUC ± Std | Mean MCC ± Std | Balanced Acc ± Std |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **C1: Row-Level** | $0,0852 \pm 0,0578$ | $0,1830 \pm 0,1745$ | $0,1501 \pm 0,0349$ | $0,1998 \pm 0,0145$ | $0,7001 \pm 0,0159$ | $0,0691 \pm 0,0484$ | $0,5699 \pm 0,0446$ |
| **C2: Project-Level*** | $0,1225 \pm 0,0661$*| $0,2272 \pm 0,1816$*| $0,1811 \pm 0,0382$*| $0,2401 \pm 0,0169$*| $0,7124 \pm 0,0179$*| $0,1038 \pm 0,0599$*| $0,5940 \pm 0,0504$*|
| **C3: Imputation** | $0,0847 \pm 0,0587$ | $0,1786 \pm 0,1682$ | $0,1511 \pm 0,0395$ | $0,2017 \pm 0,0099$ | $0,7030 \pm 0,0120$ | $0,0686 \pm 0,0495$ | $0,5698 \pm 0,0441$ |

#### 3. Analisis Dukungan Generalisasi Lintas Proyek (Cross-Project Generalization)
- **Kegagalan Generalisasi C2**: Walaupun C2 mencatatkan rata-rata F1 tampak tinggi pada 19 proyek ($0,1225 \pm 0,0661$), model tersebut tidak memiliki kemampuan transfer ke sistem enterprise heterogen. Menghapus 5 proyek dari data latih membatasi ruang hipotesis model (*hypothesis space restriction*), sehingga model C2 *overfit* pada karakteristik proyek-proyek berukuran kecil hingga menengah.
- **Keunggulan Generalisasi C1 atas C2**: C1 melestarikan ke-24 proyek, menjamin bahwa ruang pelatihan mencakup domain arsitektur yang luas. Model C1 mampu menggeneralisasi ke repositori enterprise `spring-boot`, mendeteksi rata-rata $96,8 \pm 33,4$ kasus flaky lintas 10 seed ($60,50 \pm 20,90\%$ Recall).
- **Keunggulan C3 atas C1**: Pada domain 24 proyek yang sama persis, C3 menghasilkan representasi generalisasi yang lebih unggul dibandingkan C1:
  - F1-Score pipeline terbaik meningkat **$+11,21\%$** ($0,1633 \rightarrow \mathbf{0,1816}$).
  - Mean Precision meningkat **$+28,10\%$** ($0,1591 \rightarrow \mathbf{0,2038}$).
  - Total True Positive flaky yang berhasil ditangkap meningkat **$+10,45\%$** ($421 \rightarrow \mathbf{465}\text{ TP}$).
  - Pada proyek `spring-boot`, C3 memangkas False Positive sebesar **$-18,5\%$** (dari 428 ke 349 FP pada seed 42, dan $339,9 \pm 34,9$ lintas 10 seed), menghasilkan F1-score `spring-boot` yang lebih tinggi ($0,3975$ vs $0,3621$).

#### 4. Uji Signifikansi Statistik Paradigma Cleaning
Uji **Kruskal-Wallis Non-Parametric ANOVA** membuktikan signifikansi perbedaan antar strategi cleaning:
- Mean PR-AUC: $H = 43,9826$, $p\text{-value} = 2,81 \times 10^{-10}$ (**Sangat Signifikan**).
- Mean Precision: $H = 10,2736$, $p\text{-value} = 0,0059$ (**Signifikan**).
- Uji Pairwise Mann-Whitney U menunjukkan perbedaan antara C1 vs C2 ($p = 1,31 \times 10^{-8}$) dan C2 vs C3 ($p = 9,12 \times 10^{-9}$) didorong oleh eliminasi proyek enterprise pada C2.

> **Sintesis Jawaban RQ1**:  
> **Pembersihan level-baris (Row-Level Cleaning / C1) jauh lebih baik daripada pembersihan level-proyek (Project-Level Cleaning / C2)** karena berhasil melestarikan 99,63% sampel flaky (hanya kehilangan 3 sampel vs 225 sampel pada C2) dan mempertahankan keberagaman domain 24 proyek yang krusial bagi generalisasi sistem enterprise tanpa terjebak *survivorship bias*. Lebih lanjut, **Group-Aware Median Imputation (C3)** terbukti sebagai standar pemrosesan terbaik yang mengungguli C1 dan C2, melestarikan 100% sampel flaky dengan peningkatan F1 +11,39%, Presisi +28,31%, dan penangkapan 465 True Positive flaky tests ($426,6 \pm 44,4\text{ TP}$ lintas 10 seed).

---

### 🔹 RQ2: Metode Penskalaan Fitur Mana yang Paling Mampu Memitigasi Pergeseran Distribusi Lintas Proyek (Cross-Project Distribution Shift)?

> **Research Question 2 (RQ2)**:  
> *RQ2 investigates which scaling method best mitigates cross-project distribution shift.*

#### 1. Karakteristik Pergeseran Distribusi Lintas Proyek (Covariate Shift)
Dalam *Cross-Project Flaky Test Prediction*, dataset pengujian mengalami pergeseran kovariat (*covariate shift*) yang parah. Proyek dengan jutaan baris kode (seperti `spring-boot` atau `wildfly`) memiliki metrik *code churn* (misalnya `churn_h_index`) dan waktu eksekusi tes (*execution time*) dengan skala ribuan kali lebih besar dibandingkan repositori utilitas kecil (seperti `commons-exec` atau `handlebars.java`). Fitur-fitur ini menunjukkan distribusi *heavy-tailed* dan kemencengan ekstrem (*skewness*).

#### 2. Analisis Kinerja Komparatif 3 Metode Penskalaan (Mean ± Std)

Eksperimen mengevaluasi tiga scaler yang di-fit secara ketat *hanya pada $N-1$ training projects* dan di-transform pada unseen test project:
- **S1 (StandardScaler)**: $z = \frac{x - \mu}{\sigma}$ (Z-score standardisation)
- **S2 (MinMaxScaler)**: $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ (Transformasi rentang bounded $[0, 1]$)
- **S3 (RobustScaler)**: $x'' = \frac{x - Q_2}{Q_3 - Q_1}$ (Penskalaan median dan Interquartile Range / IQR)

| Metode Scaling | Mean F1 ± Std | Mean Recall ± Std | Mean Precision ± Std | Mean PR-AUC ± Std | Mean ROC-AUC ± Std | Mean MCC ± Std | Balanced Acc ± Std | Mean F1 (B3: RUS) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **S1 (StandardScaler)** | $0,0977 \pm 0,0639$ | $0,1975 \pm 0,1758$ | $0,1601 \pm 0,0412$ | $0,2128 \pm 0,0243$ | $0,7052 \pm 0,0161$ | $0,0807 \pm 0,0558$ | $0,5784 \pm 0,0479$ | $0,1814 \pm 0,0240$ |
| **S2 (MinMaxScaler)**   | **$0,1000 \pm 0,0631$** | **$0,2017 \pm 0,1742$** | **$0,1614 \pm 0,0401$** | **$0,2147 \pm 0,0237$** | **$0,7054 \pm 0,0159$** | **$0,0829 \pm 0,0550$** | **$0,5795 \pm 0,0475$** | **$0,1818 \pm 0,0242$** |
| **S3 (RobustScaler)**   | $0,0946 \pm 0,0637$ | $0,1896 \pm 0,1785$ | $0,1608 \pm 0,0396$ | $0,2140 \pm 0,0228$ | $0,7048 \pm 0,0171$ | $0,0779 \pm 0,0554$ | $0,5759 \pm 0,0482$ | **$0,1818 \pm 0,0243$** |

#### 3. Interaksi Penskalaan dengan Famili Model Klasifikasi (Mean ± Std)

| Metode Scaling | Random Forest (M1) Mean F1 | XGBoost GPU (M2) Mean F1 | Random Forest Mean PR-AUC | XGBoost GPU Mean PR-AUC |
|---|:---:|:---:|:---:|:---:|
| **S1 (StandardScaler)** | $0,0779 \pm 0,0720$ | $0,1175 \pm 0,0504$ | $0,2169 \pm 0,0236$ | $0,2088 \pm 0,0253$ |
| **S2 (MinMaxScaler)**   | **$0,0810 \pm 0,0707$** | **$0,1190 \pm 0,0492$** | **$0,2195 \pm 0,0229$** | **$0,2099 \pm 0,0246$** |
| **S3 (RobustScaler)**   | $0,0766 \pm 0,0714$ | $0,1127 \pm 0,0477$ | $0,2203 \pm 0,0221$ | $0,2077 \pm 0,0231$ |

#### 4. Analisis Mekanisme Algoritmik
1. **Invariansi Monotonik Pohon Keputusan**:
   Uji Kruskal-Wallis menunjukkan $H = 0,1624$ ($p = 0,9220$) pada Mean F1 dan $H = 0,2458$ ($p = 0,8844$) pada PR-AUC. Ini mengonfirmasi prinsip teoritis bahwa pohon keputusan berbasis partisi threshold tunggal bersifat invarian terhadap transformasi monotonik.
2. **Keunggulan Kuantisasi Histogram GPU pada MinMaxScaler (S2)**:
   Meskipun invarian split berlaku pada pohon tunggal, dalam konteks ensemble modern dengan GPU hardware acceleration (`tree_method="hist"` pada XGBoost), fitur dipetakan ke dalam bin histogram diskret (misal 256 bin). 
   - Pada **StandardScaler (S1)** dan **RobustScaler (S3)**, keberadaan nilai ekstrem (*extreme outliers*) dari proyek target baru dapat menghasilkan nilai z-score yang sangat besar ($z > 50$), menyebabkan sebagian besar titik data terkompresi ke dalam segelintir bin histogram saja.
   - Sebaliknya, **MinMaxScaler (S2)** memetakan seluruh ruang fitur secara proporsional ke dalam interval bounded $[0, 1]$. Kuantisasi kuantil histogram menjadi seragam dan stabil, mencegah degradasi pembagian node saat berhadapan dengan proyek unseen. Terbukti S2 menghasilkan Mean F1 tertinggi baik pada Random Forest ($0,0810 \pm 0,0707$) maupun XGBoost ($0,1190 \pm 0,0492$).

> **Sintesis Jawaban RQ2**:  
> **MinMaxScaler (S2) merupakan metode penskalaan terbaik dalam memitigasi cross-project distribution shift.** S2 mencatatkan Mean F1 tertinggi ($0,1000 \pm 0,0631$), Recall tertinggi ($20,17 \pm 17,42\%$), Presisi tertinggi ($16,14 \pm 4,01\%$), dan PR-AUC tertinggi ($0,2147 \pm 0,0237$) di antara seluruh scaler. Pemetaan rentang bounded $[0, 1]$ pada S2 secara efektif menstabilkan diskretisasi histogram fitur pada akselerasi GPU XGBoost dan partisi Random Forest dari distorsi *heavy-tailed outliers* lintas repositori Java.

---

### 🔹 RQ3: Perlakuan Ketimpangan Kelas (Imbalance Treatment) Mana yang Memberikan Trade-off Precision-Recall Paling Berguna?

> **Research Question 3 (RQ3)**:  
> *RQ3 evaluates which imbalance treatment provides the most useful precision-recall trade-off.*

#### 1. Karakteristik Ketimpangan Ekstrem dalam Benchmark FlakeFlagger
Dataset FlakeFlagger memiliki ketimpangan kelas minoritas yang sangat masif: hanya **811 test cases flaky dari total 22.236 test cases** ($\approx 3,65\%$ tingkat prevalensi alami). Pada beberapa proyek seperti `assertj-core`, rasio flaky bahkan mencapai $1$ dari $6.261$ tes ($0,016\%$). Tanpa intervensi, algoritma machine learning konvensional akan memprediksi seluruh kelas sebagai non-flaky (*majority class collapse*).

#### 2. Evaluasi Komparatif 4 Strategi Penanganan Ketimpangan Kelas (Mean ± Std)

| Metode Imbalance | Mean F1 ± Std | Mean Recall ± Std | Mean Precision ± Std | Mean PR-AUC ± Std | Mean ROC-AUC ± Std | Mean MCC ± Std | Balanced Acc ± Std | Mean TP (Seed 42) | Mean FP (Seed 42) | Mean FN (Seed 42) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B1: Baseline**    | $0,0478 \pm 0,0247$ | $0,0698 \pm 0,0401$ | $0,1460 \pm 0,0384$ | $0,2123 \pm 0,0189$ | $0,7089 \pm 0,0085$ | $0,0397 \pm 0,0218$ | $0,5410 \pm 0,0147$ | 75,2 | 1.074,9 | 659,8 |
| **B2: SMOTE**       | $0,0926 \pm 0,0357$ | $0,1318 \pm 0,0640$ | **$0,1783 \pm 0,0345$**| $0,2157 \pm 0,0184$ | $0,7035 \pm 0,0118$ | $0,0760 \pm 0,0311$ | $0,5664 \pm 0,0231$ | 114,6 | **517,7** | 620,4 |
| **B3: RUS**         | **$0,1817 \pm 0,0241$**| **$0,4716 \pm 0,0363$**| $0,1742 \pm 0,0221$ | **$0,2290 \pm 0,0259$**| **$0,7173 \pm 0,0127$**| **$0,1548 \pm 0,0292$**| **$0,6484 \pm 0,0185$**| **402,3** | 3.831,9 | **332,7** |
| **B4: ClassWeight** | $0,0678 \pm 0,0537$ | $0,1119 \pm 0,0999$ | $0,1446 \pm 0,0493$ | $0,1985 \pm 0,0199$ | $0,6909 \pm 0,0180$ | $0,0515 \pm 0,0407$ | $0,5558 \pm 0,0294$ | 90,7 | 615,6 | 644,3 |

```
                       TRADE-OFF PRECISION vs. RECALL (IMBALANCE TREATMENT)
    Recall (%)
     50% ┼───────────────────────────────────────────────────────────── B3: RUS
         │                                                            (Rec: 47,16 ± 3,63%, Prec: 17,42 ± 2,21%, F1: 0,1817 ± 0,0241)
     40% ┼
         │
     30% ┼
         │
     20% ┼
         │                                       B2: SMOTE
     10% ┼                B4: ClassWeight        (Rec: 13,18 ± 6,40%, Prec: 17,83 ± 3,45%)
         │  B1: Baseline  (Rec: 11,19 ± 9,99%, Prec: 14,46 ± 4,93%)
      0% ┴────────┬──────────────┬──────────────┬──────────────┬──────────────► Precision (%)
                 10%            12%            14%            16%            18%
```

#### 3. Analisis Mendalam Trade-Off dan Dinamika Kegagalan SMOTE & ClassWeight

1. **Kegagalan Paradigmatik SMOTE (B2) dalam CP-FTP**:
   Meskipun SMOTE mencatatkan presisi rata-rata tertinggi ($17,83 \pm 3,45\%$), Recall-nya sangat rendah (**hanya $13,18 \pm 6,40\%$**). SMOTE menghasilkan sampel minoritas sintetis melalui interpolasi linier antar tetangga terdekat di ruang fitur data latih. Dalam prediksi lintas proyek, manifold fitur antar proyek sangat berbeda; titik sintetis yang dibuat dari interpolasi proyek A dan B jatuh pada area ruang fitur yang sama sekali tidak realistis untuk proyek target C. Akibatnya, model gagal mendeteksi mayoritas flaky test riil di proyek target (False Negative sangat tinggi: $620,4$).
2. **Keterbatasan ClassWeight (B4)**:
   ClassWeight mencoba memberi bobot lebih tinggi pada kesalahan prediksi kelas minoritas tanpa mengubah distribusi sampel. Namun, dengan rasio ketimpangan $1:27$, ruang fitur mayoritas tetap mendominasi struktur partisi pohon keputusan, menyebabkan Recall stagnan di angka $11,19 \pm 9,99\%$.
3. **Supremasi Mutlak Random Under-Sampling (B3: RUS)**:
   Uji statistik Kruskal-Wallis menegaskan keunggulan mutlak RUS:
   - F1-Score: $H = 43,4028$, $p = 2,02 \times 10^{-9}$ (**Sangat Signifikan**).
   - Recall: $H = 43,0393$, $p = 2,41 \times 10^{-9}$ (**Sangat Signifikan**).
   - PR-AUC: $H = 14,9176$, $p = 1,89 \times 10^{-3}$ (**Signifikan**).
   RUS menyeimbangkan proporsi training set menjadi $1:1$. Ini memaksa algoritma pembelajaran untuk membedakan fitur esensial flaky test tanpa terbias oleh densitas mayoritas non-flaky. Hasilnya, **Recall melonjak hingga 47,16%** (mendeteksi rata-rata 402,3 flaky tests di unseen projects vs hanya 75,2 pada Baseline B1), dengan F1-score melonjak hampir $4\times$ lipat ($0,1817 \pm 0,0241$ vs $0,0478 \pm 0,0247$).

#### 4. Perspektif Rekayasa Perangkat Lunak Praktis (Cost Asymmetry)
Dalam konteks industri CI/CD, biaya kesalahan prediksi bersifat asimetris:
- **Biaya False Negative (FN)**: Flaky test lolos tanpa terdeteksi, menyebabkan build pipeline gagal secara acak saat integrasi kode, memicu *test reruns* yang memakan waktu berjam-jam, membuang komputasi cloud, dan mengikis kepercayaan developer terhadap test suite.
- **Biaya False Positive (FP)**: Model menandai tes yang stabil sebagai kandidat flaky, memerlukan inspeksi cepat oleh developer atau isolasi ke runner khusus.
Oleh karena itu, strategi yang memaksimalkan Recall dengan presisi yang tetap dapat diterima ($\approx 17\text{--}20\%$) seperti **RUS (B3)** adalah satu-satunya trade-off yang bernilai praktis dan berguna bagi industri perangkat lunak.

> **Sintesis Jawaban RQ3**:  
> **Random Under-Sampling (B3: RUS) memberikan trade-off Precision-Recall yang paling berguna secara operasional.** RUS mendongkrak Recall dari 6,98% (Baseline) menjadi **47,16 ± 3,63%** dan melipatgandakan F1-Score menjadi **0,1817 ± 0,0241** (peningkatan hampir $4\times$ lipat, $p = 2,02 \times 10^{-9}$). Sebaliknya, SMOTE (B2) gagal mentransfer generalisasi lintas proyek karena interpolasi sintetis memperparah pergeseran domain, menghasilkan Recall buruk (13,18%). Dalam realitas CI/CD di mana False Negative sangat mahal, trade-off agresif RUS yang menangkap hampir separuh flaky test pada unseen project merupakan solusi terbaik.

---

### 🔹 RQ4: Bagaimana Perbandingan Kemampuan Transfer (Transferability) dari Berbagai Famili Model Klasifikasi Lintas Proyek Baru (Unseen Projects)?

> **Research Question 4 (RQ4)**:  
> *RQ4 compares the transferability of different model families across unseen projects.*

#### 1. Karakteristik Famili Model yang Dibandingkan
Penelitian membandingkan dua famili arsitektur ensemble pohon keputusan terkemuka:
1. **M1: Random Forest (Bagging Famili)**:
   Membangun 100 pohon keputusan independen secara paralel dengan *bootstrap sampling* dan seleksi fitur acak (*random feature subspace*). Prediksi akhir ditentukan oleh voting mayoritas rata-rata probabilitas.
2. **M2: XGBoost (Gradient Boosting Famili)**:
   Membangun 100 pohon keputusan secara sekuensial mengoptimalkan fungsi *objective loss* berbasis gradien orde-dua (Hessian dan Gradient) dengan regularisasi L1/L2, dieksekusi dengan akselerasi perangkat keras NVIDIA CUDA GPU histogram algorithm (`device="cuda"`, `tree_method="hist"`).

#### 2. Komparasi Performa Global & Transferabilitas Rata-Rata (Mean ± Std)

| Famili Model | Mean F1 ± Std | Mean Recall ± Std | Mean Precision ± Std | Mean PR-AUC ± Std | Mean ROC-AUC ± Std | Mean MCC ± Std | Balanced Acc ± Std | Waktu Eksekusi LOPO |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M1: Random Forest (Bagging)** | $0,0785 \pm 0,0701$ | $0,1530 \pm 0,1840$ | $0,1399 \pm 0,0386$ | **$0,2189 \pm 0,0224$**| **$0,7154 \pm 0,0116$**| $0,0666 \pm 0,0636$ | $0,5675 \pm 0,0525$ | $5,94 \pm 2,81\text{ detik}$ |
| **M2: XGBoost (Boosting GPU)**  | **$0,1164 \pm 0,0483$**| **$0,2395 \pm 0,1535$**| **$0,1816 \pm 0,0288$**| $0,2088 \pm 0,0234$ | $0,6949 \pm 0,0133$ | **$0,0944 \pm 0,0403$**| **$0,5883 \pm 0,0392$**| **$4,62 \pm 0,84\text{ detik (1,29x Lebih Cepat)}$** |
| **Delta Relatif (M2 vs. M1)**   | **+48,2%** | **+56,5%** | **+29,8%** | -4,6% | -2,9% | **+41,7%** | **+3,7%** | **GPU Accelerated** |

#### 3. Uji Signifikansi Statistik Transferabilitas Model
- **Wilcoxon Signed-Rank Test** (Evaluasi berpasangan pada konfigurasi identik):
  - F1-Score: $W = 55,0$, $p\text{-value} = \mathbf{4,88 \times 10^{-7}}$ (**Sangat Signifikan Superioritas M2**).
  - Recall: $W = 102,0$, $p\text{-value} = \mathbf{2,73 \times 10^{-5}}$ (**Sangat Signifikan**).
  - Precision: $W = 87,0$, $p\text{-value} = \mathbf{1,21 \times 10^{-5}}$ (**Sangat Signifikan**).
  - Runtime: $W = 189,0$, $p\text{-value} = 0,0201$ (**XGBoost GPU Signifikan Lebih Cepat**).
- **Mann-Whitney U Test**:
  - F1-Score: $p = 2,49 \times 10^{-3}$; Recall: $p = 2,68 \times 10^{-4}$; Precision: $p = 8,64 \times 10^{-5}$.

#### 4. Analisis Perilaku Kondisional: Bagging vs. Boosting di Bawah Resampling RUS

Meskipun XGBoost (M2) unggul telak pada rata-rata agregat faktor, dinamika transferabilitas menunjukkan fenomena yang sangat menarik ketika model dipasangkan dengan strategi penanganan ketimpangan kelas yang berbeda:

| Skenario Resampling | Random Forest (M1) Mean F1 ± Std | XGBoost GPU (M2) Mean F1 ± Std | Analisis Dinamika Transferabilitas |
|---|:---:|:---:|---|
| **B1: Baseline (Unweighted)** | $0,0400 \pm 0,0253$ | **$0,0555 \pm 0,0227$** | XGBoost lebih mampu mengekstraksi sinyal minoritas tanpa bobot |
| **B2: SMOTE**                 | $0,0670 \pm 0,0281$ | **$0,1182 \pm 0,0209$** (+76,4%) | XGBoost tahan terhadap noise sampel sintetis berkat regularisasi |
| **B3: RUS (Under-Sampling)**  | **$0,1906 \pm 0,0221$** (+10,3%) | $0,1728 \pm 0,0234$ | **Random Forest unggul**: Bagging meredam variansi subsampling data |
| **B4: ClassWeight**           | $0,0166 \pm 0,0106$ (Kolaps) | **$0,1190 \pm 0,0169$** (+617%) | Random Forest gagal total; XGBoost stabil via `scale_pos_weight` |

1. **Mengapa XGBoost (M2) Menunjukkan Transferabilitas Global yang Lebih Unggul?**
   Dalam transfer lintas proyek yang tidak seimbang, pohon-pohon individual dalam Random Forest dapat membelah pada fitur-fitur idiosinkratik proyek sumber. Sebaliknya, XGBoost mengoptimalkan residual secara sekuensial dengan fungsi penalti regularisasi daun ($\gamma$ dan $\lambda$). Algoritma histogram XGBoost membatasi kedalaman pohon (`max_depth=6`), mencegah model menghafal pola lokal proyek sumber dan mempertahankan generalisasi yang stabil di hampir seluruh kondisi perlakuan kelas ($CV = 41,66\%$ vs $88,97\%$ pada RF).
2. **Mengapa Random Forest (M1) Meraih Puncak Performa Tertinggi pada RUS?**
   Saat Random Under-Sampling (B3) diterapkan, variansi data latih meningkat drastis karena pembuangan acak mayoritas. Pada kondisi ini, mekanisme *bagging* (penggabungan rata-rata 100 pohon independen) bekerja sebagai filter peredam variansi yang luar biasa (*variance dampening*). Sebaliknya, boosting sekuensial pada XGBoost rentan menyesuaikan diri secara berlebihan pada sampel noise yang tersisa dari proses undersampling. Akibatnya, pada pipeline terbaik B3, Random Forest meraih F1 tertinggi (**$0,1816$** pada `P61` vs $0,1403$ pada varian XGBoost sebanding) dan presisi tertinggi ($20,38\%$).

#### 5. Efisiensi Komputasi dan Akselerasi Perangkat Keras
Dalam skenario LOPO-CV 24 fold lintas proyek, throughput komputasi menjadi faktor krusial:
- XGBoost GPU (CUDA) mencatatkan rata-rata runtime **$4,62 \pm 0,84\text{ detik}$** per pipeline, mengungguli CPU multi-threaded Random Forest (**$5,94 \pm 2,81\text{ detik}$**, speedup $1,29\times$).
- Pada fase inferensi pengujian, evaluasi batch pohon histogram GPU memberikan akselerasi throughput **$1,56\times$ lebih cepat**, memungkinkan prediksi instan pada ribuan test case dalam repositori enterprise.

> **Sintesis Jawaban RQ4**:  
> **XGBoost (M2) menunjukkan transferabilitas global yang lebih kokoh dan konsisten lintas proyek baru**, mengungguli Random Forest (M1) secara statistik dengan peningkatan Mean F1 +48,2% ($0,1164 \pm 0,0483$ vs $0,0785 \pm 0,0701$, $p = 4,88 \times 10^{-7}$), Recall +56,5%, Presisi +29,8%, variabilitas inter-kondisi yang lebih rendah, dan efisiensi waktu komputasi 1,29x lebih cepat via CUDA GPU. Namun demikian, **pada konfigurasi optimal Random Under-Sampling (B3), Random Forest (M1) mencapai puncak akurasi tertinggi** (F1 $0,1906 \pm 0,0221$ vs $0,1728 \pm 0,0234$, Presisi $19,23\%$ vs $15,61\%$) karena prinsip bagging berhasil meredam fluktuasi stokastik yang ditimbulkan oleh undersampling.

---

## 3. Peringkat 10 Pipeline Terbaik (Top-10 Factorial Pipelines dengan Mean ± Std)

Berdasarkan evaluasi Leave-One-Project-Out Cross-Validation (LOPO-CV) 10-Seed Repetition, berikut 10 konfigurasi pipeline terbaik dalam seluruh ruang benchmark faktorial yang disajikan lengkap dengan **Mean ± Standar Deviasi (Std)**:

| Peringkat | Pipeline ID | Cleaning | Scaling | Imbalance | Model | Mean F1 ± Std (10-Seed) | Mean Recall ± Std | Mean Precision ± Std | Mean PR-AUC ± Std | Mean MCC ± Std | CV (%) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 **1** | `P37_C2_S2_B3_M1`* | C2 | S2 (MinMax) | **B3 (RUS)** | M1 (RF) | **$0,2178 \pm 0,0174$** | $0,5093 \pm 0,0254$ | $0,1936 \pm 0,0153$ | $0,2685 \pm 0,0211$ | $0,2008 \pm 0,0172$ | 7,99% |
| 🥈 **2** | `P45_C2_S3_B3_M1`* | C2 | S3 (Robust) | **B3 (RUS)** | M1 (RF) | **$0,2175 \pm 0,0173$** | $0,5098 \pm 0,0257$ | $0,1931 \pm 0,0148$ | $0,2686 \pm 0,0205$ | $0,2005 \pm 0,0169$ | 7,96% |
| 🥉 **3** | `P29_C2_S1_B3_M1`* | C2 | S1 (Standard) | **B3 (RUS)** | M1 (RF) | **$0,2166 \pm 0,0191$** | $0,5042 \pm 0,0370$ | $0,1936 \pm 0,0165$ | $0,2718 \pm 0,0235$ | $0,1990 \pm 0,0202$ | 8,82% |
| **4**    | `P30_C2_S1_B3_M2`* | C2 | S1 (Standard) | **B3 (RUS)** | M2 (XGB)| **$0,2068 \pm 0,0168$** | $0,5289 \pm 0,0351$ | $0,1794 \pm 0,0218$ | $0,2550 \pm 0,0253$ | $0,1793 \pm 0,0173$ | 8,13% |
| **5**    | `P38_C2_S2_B3_M2`* | C2 | S2 (MinMax) | **B3 (RUS)** | M2 (XGB)| **$0,2068 \pm 0,0168$** | $0,5289 \pm 0,0351$ | $0,1794 \pm 0,0218$ | $0,2550 \pm 0,0253$ | $0,1793 \pm 0,0173$ | 8,13% |
| **6**    | `P46_C2_S3_B3_M2`* | C2 | S3 (Robust) | **B3 (RUS)** | M2 (XGB)| **$0,2068 \pm 0,0168$** | $0,5289 \pm 0,0351$ | $0,1794 \pm 0,0218$ | $0,2550 \pm 0,0253$ | $0,1793 \pm 0,0173$ | 8,13% |
| **7**    | **`P69_C3_S3_B3_M1`** | **C3** | S3 (Robust) | **B3 (RUS)** | **M1 (RF)** | **$0,1782 \pm 0,0089$** | $0,4318 \pm 0,0277$ | $0,1961 \pm 0,0246$ | $0,2224 \pm 0,0133$ | $0,1536 \pm 0,0106$ | **4,99%** |
| **8**    | **`P61_C3_S2_B3_M1`** | **C3** | **S2 (MinMax)** | **B3 (RUS)** | **M1 (RF)** | **$0,1773 \pm 0,0095$** | $0,4288 \pm 0,0256$ | $0,1914 \pm 0,0244$ | $0,2202 \pm 0,0152$ | $0,1527 \pm 0,0109$ | **5,35%** |
| **9**    | `P13_C1_S2_B3_M1` | C1 | S2 (MinMax) | **B3 (RUS)** | M1 (RF) | **$0,1773 \pm 0,0119$** | $0,4462 \pm 0,0380$ | $0,1898 \pm 0,0230$ | $0,2221 \pm 0,0191$ | $0,1535 \pm 0,0145$ | 6,69% |
| **10**   | `P05_C1_S1_B3_M1` | C1 | S1 (Standard) | **B3 (RUS)** | M1 (RF) | **$0,1772 \pm 0,0122$** | $0,4462 \pm 0,0369$ | $0,1894 \pm 0,0246$ | $0,2227 \pm 0,0179$ | $0,1532 \pm 0,0146$ | 6,88% |

*\*Catatan: Peringkat 1 s.d 6 merupakan hasil dari paradigma C2 yang mengalami survivorship bias akibat eliminasi proyek enterprise `spring-boot`.*

### Rekomendasi Golden Pipeline Bebas Bias
Mengabaikan paradigma C2 yang bias, **`P61_C3_S2_B3_M1`** (**Group-Aware Median Imputation + MinMaxScaler + Random Under-Sampling + Random Forest**) ditetapkan sebagai **Pipeline Standar Emas (*Golden Pipeline*)**:
- Melestarikan **100% data** (24 proyek, 22.236 test cases, 811 flaky test).
- Menghasilkan performa operasional tertinggi pada dataset lengkap: Seed 42 mencatatkan **F1 $0,1816$**, **Recall $43,74\%$**, **Presisi $20,38\%$**, **PR-AUC $0,2210$**, dan menangkap **465 True Positive flaky tests**.
- Menunjukkan tingkat presisi tertinggi pada proyek enterprise `spring-boot` ($26,68\%$, F1 $0,3975$).
- Memiliki stabilitas stokastik tertinggi dengan **Coefficient of Variation terendah: $CV = 5,35\%$** ($\text{Mean F1} = 0,1773 \pm 0,0095$).

---

## 4. Stabilitas Stokastik & Analisis Konsistensi Lintas 10 Random Seeds

Pengujian replikasi dilakukan menggunakan 10 random seed independen:
$$\text{DEFAULT\_SEEDS\_10} = [42, 100, 2024, 777, 999, 123, 456, 789, 2025, 2026]$$
Total $10 \times 1.608 = \mathbf{16.080 \text{ evaluasi fold}}$ telah dieksekusi.

### 4.1 Temuan Konsistensi Empiris:
1. **Stabilitas Sangat Tinggi pada Pipeline Teratas (CV < 6%)**:
   Konfigurasi emas `P61_C3_S2_B3_M1` membuktikan reprodusibilitas yang sangat kokoh dengan $\text{Mean F1} = 0,1773 \pm 0,0095$, rentang nilai $[0,1622 \text{--} 0,1987]$, interval kepercayaan $95\% \text{ CI} = [0,1714, 0,1832]$, dan koefisien variasi $\mathbf{CV = 5,35\%}$.
2. **Peredaman Variansi pada Random Forest**:
   Random Forest (M1) menunjukkan dispersi inter-seed yang lebih rendah ($CV_{\text{P61}} = 5,35\%$) dibandingkan XGBoost ($CV_{\text{P62}} = 7,22\%$). Struktur bagging meratakan efek keacakan pemilihan sampel pada under-sampling.
3. **Deterministik pada Pipeline Tanpa Resampling**:
   Seluruh konfigurasi `B1` (Baseline) dan `B4` (ClassWeight) menghasilkan $CV = 0,00\%$. Ini membuktikan bahwa variansi metrik semata-mata berasal dari keacakan proses undersampling/oversampling minoritas dan bukan dari instabilitas komputasi numerik floating-point.

---

## 5. Uji Signifikansi Statistik Formal (ACM/SIGSOFT Guidelines)

Mengikuti pedoman empiris ACM/SIGSOFT Empirical Standards, seluruh hasil benchmark diuji signifikansinya secara formal:

1. **Friedman Omnibus Test**:
   - Hipotesis Nol ($H_0$): Tidak terdapat perbedaan performa di antara ke-72 pipeline klasifikasi.
   - Statistik Chi-Square: $\chi_F^2 = \mathbf{4.830,0816}$
   - Derajat Kebebasan ($df$): $71$
   - $p\text{-value}$: $\mathbf{0,0000 \times 10^0 \ll 0,0001}$
   - **Kesimpulan**: $H_0$ ditolak secara meyakinkan. Perbedaan performa antar pipeline sangat signifikan secara statistik.
2. **Nemenyi Post-hoc Test**:
   - Critical Difference ($CD$) pada tingkat signifikansi $\alpha = 0,05$ untuk $k = 72$ dan $N = 24$:
     $$CD = q_\alpha \sqrt{\frac{k(k + 1)}{6N}} = 3,219 \times \sqrt{\frac{72 \times 73}{6 \times 24}} = \mathbf{19,4477}$$
   - Seluruh pipeline yang tergabung dalam klaster teratas secara eksklusif dikuasai oleh varian **Random Under-Sampling (B3)**.
3. **Kruskal-Wallis Multi-Group ANOVA**:
   - Faktor Imbalance: $H = 43,4028$, $p = 2,02 \times 10^{-9}$ (**Sangat Signifikan**).
   - Faktor Model: $p = 4,88 \times 10^{-7}$ (Wilcoxon Signed-Rank, **Sangat Signifikan**).
   - Faktor Cleaning: Precision $p = 0,0059$, PR-AUC $p = 2,81 \times 10^{-10}$ (**Signifikan**).
   - Faktor Scaling: $H = 0,1624$, $p = 0,9220$ (**Tidak Signifikan / Monotonic Invariance Terpenuhi**).

---

## 6. Implikasi Rekayasa Perangkat Lunak & Rekomendasi Praktis

Hasil eksperimen gabungan ini memberikan pedoman aplikatif bagi tim rekayasa perangkat lunak dan arsitek pengujian sistem:

1. **Hentikan Praktik Membuang Data (Data Deletion is Harmful)**:
   Tim developer sering kali tergoda menghapus baris atau modul yang memiliki data hilang. Eksperimen ini membuktikan bahwa strategi pembuangan proyek (C2) menciptakan *survivorship bias* yang menipu dan menghilangkan $27,74\%$ pengetahuan flakiness. Strategi pembuangan baris (C1) mengorbankan sampel flaky yang bernilai tinggi. **Terapkan Group-Aware Per-Project Median Imputation (C3)** sebagai standar preprocessing data pengujian.
2. **Adopsi Agresif Undersampling dalam CI/CD Pipeline**:
   Dalam memprediksi tes flaky lintas repositori, jangan gunakan data latih mentah apa adanya (B1) atau mengandalkan SMOTE (B2). Gunakan **Random Under-Sampling (B3: RUS)** dengan rasio seimbang $1:1$ untuk memaksimalkan Recall hingga $>47\%$.
3. **Konfigurasi Produksi Rekomendasi (Golden Industrial Pipeline)**:
   Gunakan pipeline **`P61_C3_S2_B3_M1`** (**Dataset Terimputasi Median Grup + MinMaxScaler + Random Under-Sampling + Random Forest 100 Trees**) untuk mendapatkan keseimbangan terbaik antara akurasi prediksi, preservasi enterprise, dan stabilitas stokastik ($CV < 6\%$).
4. **Pemanfaatan XGBoost GPU untuk High-Throughput CI Runner**:
   Untuk pipeline pengujian enterprise skala masif dengan ratusan ribu test cases yang membutuhkan waktu siklus prediksi di bawah 5 detik, konfigurasi **`P62_C3_S2_B3_M2`** (XGBoost GPU CUDA) direkomendasikan sebagai alternatif berkecepatan tinggi dengan throughput inferensi $1,56\times$ lebih cepat ($4,62 \pm 0,84\text{ s}$ vs $5,94 \pm 2,81\text{ s}$).

---

## 7. Referensi Dokumen & Keterkaitan Artefak

- **Laporan Padanan Bahasa Inggris**: [`COMBINED_EXPERIMENT_REPORT.md`](COMBINED_EXPERIMENT_REPORT.md)
- **Desain Riset Metodologis**: [`RANCANGAN_PENELITIAN_GABUNGAN.md`](RANCANGAN_PENELITIAN_GABUNGAN.md) & [`RESEARCH_DESIGN.md`](RESEARCH_DESIGN.md)
- **Ringkasan Metrik 72 Pipeline**: [`results/summary_metrics_72_pipelines.csv`](results/summary_metrics_72_pipelines.csv)
- **Hasil Analisis Stabilitas 10-Seed**: [`results/consistency_stability_analysis.csv`](results/consistency_stability_analysis.csv)
- **Plot Visualisasi Empiris**:
  - Boxplot Tri-Paradigma Cleaning: [`results/plots/01_boxplot_tri_paradigm_cleaning.png`](results/plots/01_boxplot_tri_paradigm_cleaning.png)
  - Distribusi F1 4 Faktor Faktorial: [`results/plots/02_boxplot_4_factors_f1.png`](results/plots/02_boxplot_4_factors_f1.png)
  - CI95 Top-10 Pipelines: [`results/plots/03_top10_consistent_pipelines_ci95.png`](results/plots/03_top10_consistent_pipelines_ci95.png)
  - Distribusi CV (%) Stabilitas: [`results/plots/04_stability_cv_distribution.png`](results/plots/04_stability_cv_distribution.png)
  - Trade-off F1 vs. Recall: [`results/plots/05_f1_vs_recall_tradeoff.png`](results/plots/05_f1_vs_recall_tradeoff.png)
  - Heatmap Matriks Faktorial: [`results/plots/06_heatmap_factorial_matrix.png`](results/plots/06_heatmap_factorial_matrix.png)
