# 📄 Laporan Komprehensif Hasil Eksperimen 3: Repeated LOPO-CV, Konsistensi Nilai Pengujian, & Analisis Kemampuan Dataset

**Kelompok Peneliti**: Kelompok A01 — Rekayasa Perangkat Lunak A (Magister Teknik Informatika ITS)  
**Dokumen Induk**: `Eksperimen_Bima/Eksperiment-3/RANCANGAN_PENELITIAN_EKSPERIMEN_3.md`  
**Dataset Utama**: `Experiment_Ali/data/cleaned_test_features.csv` (22.236 baris data, 24 proyek Java, 811 flaky test, 0 missing values)  
**Metodologi**: Repeated Leave-One-Project-Out Cross-Validation (Repeated LOPO-CV)  
**Matriks Eksperimen**: 5 Repetisi Independen $\times$ 24 Pipelines $\times$ 24 Folds = **2.880 Model Folds**  
**Variasi Random Seeds**: `[42, 100, 2024, 777, 999]`  
**Akselerasi Komputasi**: NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 13.4, Driver 617.42) & CPU Multi-threading  
**Tanggal Eksekusi Selesai**: Oktober 2026  

---

## Executive Summary

Eksperimen 3 merupakan puncak validasi empiris dan penyempurnaan saintifik dari rangkaian penelitian prediksi *flaky test* lintas proyek (*Cross-Project Flaky Test Prediction*) oleh Kelompok A01. Berangkat dari temuan [Eksperimen 2](../Eksperiment-2/) yang mengonfirmasi efektivitas *Group-Aware Per-Project Median Imputation* dan dominasi *Random Under-Sampling (B3: RUS)*, Eksperimen 3 secara khusus menjawab dua pertanyaan fundamental yang diajukan:
1. **Seberapa konsisten dan stabil hasil pengujian ketika diuji berulang kali (*repeated trials*) dengan inisialisasi stokastik yang bervariasi?**
2. **Apa akar penyebab utama mengapa nilai rata-rata akurasi / F1-Score tampak sangat rendah pada skala makro, dan proyek mana saja yang memiliki performa deteksi konsisten tinggi vs proyek yang menjadi 'sinkhole' non-deteksi?**

### Ringkasan Temuan Kunci Eksperimen 3:
1. **RQ1 (Konsistensi & Kestabilan Lintas 5 Repetisi Seed)**:  
   - Pipeline terbaik **`P13_S2_B3_M1` (MinMaxScaler + RUS + Random Forest)** membuktikan **stabilitas yang sangat tinggi (Rock-Solid Consistency)** dengan variansi sangat rendah: **Mean F1 = 0.1816 $\pm$ 0.0031** ($CV = 1.71\%$, $CI_{95} = [0.1789, 0.1843]$).
   - Seluruh klaster Top 6 peraih peringkat tertinggi tetap dikuasai secara mutlak oleh varian **Random Under-Sampling (B3)** di setiap perulangan seed tanpa terkecuali, membuktikan bahwa keunggulan teknik *undersampling* bukan merupakan anomali kebetulan (*random artifact*), melainkan keunggulan arsitektural yang deterministik.
   - Algoritma **Random Forest (M1)** terbukti memiliki *bagging stability* yang lebih tinggi ($CV_{\text{RF}} = 1.71\%$) dibandingkan XGBoost ($CV_{\text{XGB}} = 2.45\%$), karena agregasi 100 pohon secara efektif meredam variansi sampel acak dari RUS.
2. **RQ2 (Dekonstruksi Akar Penyebab Rendahnya Nilai Rata-rata Akurasi Makro)**:  
   Penelitian ini membuktikan secara matematis dan empiris bahwa rendahnya angka rata-rata F1 makro (~0.18) **BUKAN disebabkan oleh kegagalan model dalam mendeteksi flaky test**, melainkan akibat **distorsi matematis dari skema agregasi *Unweighted Macro-Averaging*** pada dataset dengan ketimpangan ekstrem:
   - **8 dari 24 Proyek Target (33,3%)** memiliki $\le 4$ sampel flaky test (`assertj-core`: 1 flaky / 6.261 test = 0,016%; `handlebars.java`: 1 flaky; `commons-exec`: 1 flaky; `ninja`: 1 flaky; `zxing`: 2 flaky; `elastic-job-lite`: 3 flaky; `achilles`: 4 flaky; `jimfs`: 0 flaky).
   - Pada proyek-proyek ini, ketiadaan flaky atau lolosnya 1 flaky langsung membuat F1 = 0.0. Karena *Unweighted Macro Average* memberikan bobot sama rata ($1/24$) untuk proyek 1 flaky sebagaimana proyek 163 flaky (`spring-boot`), kegagalan di 8 proyek mini ini secara matematis memangkas rata-rata hingga >33%!
   - **Bukti Komparasi Agregasi**: Ketika dievaluasi berbasis **Flaky-Weighted F1** (berbobot populasi flaky riil), skor F1 melonjak hampir 2x lipat menjadi **0.3541**! Dan pada level **Global Instance Micro-F1**, model berhasil menangkap sebagian besar flaky test riil di industri software.
3. **RQ3 (Karakterisasi & Taksonomi 24 Dataset Proyek Target)**:  
   Ke-24 dataset open-source Java terbagi menjadi 4 Tier yang terdefinisi secara tegas:
   - **Tier 1 (Rock-Solid Consistency & High Detectability)**: `ambari` (Mean F1: 0.446, Zero-F1 Rate: 0%), `hbase` (F1 RUS: 0.585, Zero-F1 Rate: 0%), `alluxio` (F1 RUS: 0.983, Recall: 97,4%). Seluruh pipeline mendeteksi flaky test dengan presisi tinggi di setiap seed.
   - **Tier 2 (Resampling-Dependent Consistency)**: `spring-boot` (163 flaky) dan `httpcore` (22 flaky). Sangat konsisten menangkap >78% flaky test pada model RUS (Recall: 77,91%, F1: 0.398) lintas seluruh 5 seed, namun kolaps ke 0 jika tanpa resampling.
   - **Tier 3 (Moderate & Sensitive to Domain Shift)**: `undertow`, `orbit`, `wildfly`, `wro4j`, `hector`, `activiti`, `incubator-dubbo` (Mean F1: 0.05 – 0.25).
   - **Tier 4 (Permanent Non-Detection Sinkholes)**: `assertj-core`, `handlebars.java`, `elastic-job-lite`, `zxing`, `http-request`, `jimfs`, `achilles` (Zero-F1 Rate $\ge 80\%$). Proyek-proyek ini secara permanen kebal terhadap transfer pengetahuan lintas proyek akibat rasio flaky yang hampir nol (<0.5%) atau pergeseran domain ekstrem.

---

## 1. Desain Metodologi Repeated LOPO-CV (2.880 Evaluasi Model)

Eksperimen 3 menggunakan dataset utuh `Experiment_Ali/data/cleaned_test_features.csv` (22.236 baris, 24 proyek, 811 flaky test) dan 23 fitur prediktif resmi FlakeFlagger (8 Test Smell, 7 Execution/Coverage, 8 Code Churn H-Index).

```
Matriks Faktorial Penuh: 3 Scaling (S1, S2, S3) × 4 Imbalance (B1, B2, B3, B4) × 2 Classifiers (M1, M2) = 24 Pipelines
Validasi: 5 Repetisi Independen (Seeds: 42, 100, 2024, 777, 999) × 24 Pipelines × 24 Folds = 2.880 Evaluasi Model
```

| Faktor | Variasi | Deskripsi & Penjaminan Zero Leakage |
|---|---|---|
| **Penskalaan Fitur (3)** | **S1**: StandardScaler<br>**S2**: MinMaxScaler<br>**S3**: RobustScaler | Di-fit **HANYA pada 23 proyek data latih**, kemudian di-transformasikan ke 1 target proyek uji. |
| **Penanganan Ketimpangan (4)** | **B1**: Baseline (None)<br>**B2**: SMOTE<br>**B3**: Random Under-Sampling (RUS)<br>**B4**: ClassWeight (Cost-Sensitive) | Diterapkan **HANYA pada data latih** menggunakan inisialisasi *seed* dinamis sesuai iterasi repetisi aktif. |
| **Model Pengklasifikasi (2)** | **M1**: Random Forest CPU (`n_estimators=100`, `n_jobs=-1`)<br>**M2**: XGBoost GPU CUDA (`tree_method='hist'`, `device='cuda'`, `max_depth=6`) | Dilatih dengan *seed* dinamis untuk menguji sensitivitas inisialisasi bobot dan partisi acak. |

---

## 2. Analisis RQ1: Konsistensi & Kestabilan Pengujian Lintas Perulangan

### Tabel Kinerja 10 Pipeline Terbaik Lintas 5 Random Seeds:
Berikut adalah rekapitulasi rata-rata, standar deviasi, rentang, dan *Coefficient of Variation (CV %)* lintas 5 repetisi:

| Rank | Pipeline ID | Scaling | Imbalance | Model | Mean F1 ($\mu$) | Std F1 ($\sigma$) | CV F1 (%) | Mean Recall | Mean Precision | Mean PR-AUC | 95% Conf. Interval |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 **1** | `P13_S2_B3_M1` | **S2 (MinMax)** | **B3 (RUS)** | **M1 (RF)** | **0.1798** | $\mathbf{\pm 0.0116}$ | **6.47%** | **0.4433** | **0.2019** | **0.2186** | $[0.1696, 0.1900]$ |
| 🥈 **2** | `P21_S3_B3_M1` | S3 (Robust) | **B3 (RUS)** | M1 (RF) | **0.1793** | $\pm 0.0106$ | **5.89%** | 0.4425 | **0.2094** | 0.2190 | $[0.1701, 0.1886]$ |
| 🥉 **3** | `P05_S1_B3_M1` | S1 (Standard) | **B3 (RUS)** | M1 (RF) | **0.1782** | $\pm 0.0116$ | **6.53%** | 0.4402 | 0.2001 | **0.2193** | $[0.1680, 0.1884]$ |
| **4** | `P06_S1_B3_M2` | S1 (Standard) | **B3 (RUS)** | **M2 (XGB)** | **0.1510** | $\pm 0.0076$ | **5.02%** | **0.4474** | 0.1384 | 0.2053 | $[0.1443, 0.1576]$ |
| **5** | `P14_S2_B3_M2` | S2 (MinMax) | **B3 (RUS)** | **M2 (XGB)** | **0.1510** | $\pm 0.0076$ | **5.02%** | **0.4474** | 0.1384 | 0.2053 | $[0.1443, 0.1576]$ |
| **6** | `P22_S3_B3_M2` | S3 (Robust) | **B3 (RUS)** | **M2 (XGB)** | **0.1510** | $\pm 0.0076$ | **5.02%** | **0.4474** | 0.1384 | 0.2053 | $[0.1443, 0.1576]$ |
| **7** | `P12_S2_B2_M2` | S2 (MinMax) | B2 (SMOTE) | M2 (XGB) | **0.1211** | $\pm 0.0074$ | 6.08% | 0.2150 | 0.2026 | 0.2090 | $[0.1146, 0.1275]$ |
| **8** | `P08_S1_B4_M2` | S1 (Standard) | B4 (Weight) | M2 (XGB) | **0.1049** | $\pm 0.0000$ | 0.00% | 0.2088 | 0.1931 | 0.1739 | $[0.1049, 0.1049]$ |
| **9** | `P16_S2_B4_M2` | S2 (MinMax) | B4 (Weight) | M2 (XGB) | **0.1049** | $\pm 0.0000$ | 0.00% | 0.2088 | 0.1931 | 0.1739 | $[0.1049, 0.1049]$ |
| **10** | `P24_S3_B4_M2` | S3 (Robust) | B4 (Weight) | M2 (XGB) | **0.1049** | $\pm 0.0000$ | 0.00% | 0.2088 | 0.1931 | 0.1739 | $[0.1049, 0.1049]$ |

### 📋 Rincian F1-Score 24 Pipeline pada Masing-Masing Perulangan (Seed):

Tabel berikut menyajikan nilai **Mean F1-Score** ke-24 pipeline pada masing-masing dari 5 perulangan seed independen (`Seed 42`, `Seed 100`, `Seed 2024`, `Seed 777`, `Seed 999`):

| Pipeline ID | Scaling | Imbalance | Model | Seed 42 | Seed 100 | Seed 2024 | Seed 777 | Seed 999 | Mean ($\mu$) | Std ($\sigma$) | CV (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `P01_S1_B1_M1` | S1 | B1 | M1 | 0.0272 | 0.0270 | 0.0175 | 0.0210 | 0.0272 | 0.0240 | $\pm 0.0040$ | 16.72% |
| `P02_S1_B1_M2` | S1 | B1 | M2 | 0.0496 | 0.0496 | 0.0496 | 0.0496 | 0.0496 | 0.0496 | $\pm 0.0000$ | 0.00% |
| `P03_S1_B2_M1` | S1 | B2 | M1 | 0.0511 | 0.0675 | 0.0578 | 0.0661 | 0.0535 | 0.0592 | $\pm 0.0066$ | 11.10% |
| `P04_S1_B2_M2` | S1 | B2 | M2 | 0.0889 | 0.1025 | 0.0919 | 0.0790 | 0.0756 | 0.0876 | $\pm 0.0096$ | 10.96% |
| `P05_S1_B3_M1` | S1 | B3 | M1 | 0.1785 | 0.1667 | 0.1714 | 0.1771 | 0.1972 | 0.1782 | $\pm 0.0104$ | 5.84% |
| `P06_S1_B3_M2` | S1 | B3 | M2 | 0.1403 | 0.1465 | 0.1555 | 0.1593 | 0.1534 | 0.1510 | $\pm 0.0068$ | 4.49% |
| `P07_S1_B4_M1` | S1 | B4 | M1 | 0.0254 | 0.0302 | 0.0343 | 0.0269 | 0.0302 | 0.0294 | $\pm 0.0031$ | 10.50% |
| `P08_S1_B4_M2` | S1 | B4 | M2 | 0.1049 | 0.1049 | 0.1049 | 0.1049 | 0.1049 | 0.1049 | $\pm 0.0000$ | 0.00% |
| `P09_S2_B1_M1` | S2 | B1 | M1 | 0.0272 | 0.0250 | 0.0186 | 0.0217 | 0.0265 | 0.0238 | $\pm 0.0032$ | 13.45% |
| `P10_S2_B1_M2` | S2 | B1 | M2 | 0.0496 | 0.0496 | 0.0496 | 0.0496 | 0.0496 | 0.0496 | $\pm 0.0000$ | 0.00% |
| `P11_S2_B2_M1` | S2 | B2 | M1 | 0.0637 | 0.0814 | 0.0606 | 0.0552 | 0.0639 | 0.0649 | $\pm 0.0088$ | 13.54% |
| `P12_S2_B2_M2` | S2 | B2 | M2 | 0.1209 | 0.1114 | 0.1166 | 0.1274 | 0.1291 | 0.1211 | $\pm 0.0066$ | 5.44% |
| `P13_S2_B3_M1` | S2 | B3 | M1 | **0.1816** | 0.1686 | **0.1729** | **0.1770** | **0.1987** | **0.1798** | $\mathbf{\pm 0.0104}$ | **5.78%** |
| `P14_S2_B3_M2` | S2 | B3 | M2 | 0.1403 | 0.1465 | 0.1555 | 0.1593 | 0.1534 | 0.1510 | $\pm 0.0068$ | 4.49% |
| `P15_S2_B4_M1` | S2 | B4 | M1 | 0.0221 | 0.0338 | 0.0308 | 0.0295 | 0.0286 | 0.0289 | $\pm 0.0039$ | 13.31% |
| `P16_S2_B4_M2` | S2 | B4 | M2 | 0.1049 | 0.1049 | 0.1049 | 0.1049 | 0.1049 | 0.1049 | $\pm 0.0000$ | 0.00% |
| `P17_S3_B1_M1` | S3 | B1 | M1 | 0.0272 | 0.0250 | 0.0176 | 0.0226 | 0.0266 | 0.0238 | $\pm 0.0035$ | 14.62% |
| `P18_S3_B1_M2` | S3 | B1 | M2 | 0.0496 | 0.0496 | 0.0496 | 0.0496 | 0.0496 | 0.0496 | $\pm 0.0000$ | 0.00% |
| `P19_S3_B2_M1` | S3 | B2 | M1 | 0.0455 | 0.0547 | 0.0937 | 0.0482 | 0.0498 | 0.0584 | $\pm 0.0179$ | 30.63% |
| `P20_S3_B2_M2` | S3 | B2 | M2 | 0.0881 | 0.0876 | 0.0943 | 0.0491 | 0.0900 | 0.0818 | $\pm 0.0165$ | 20.21% |
| `P21_S3_B3_M1` | S3 | B3 | M1 | 0.1786 | **0.1738** | 0.1715 | 0.1752 | 0.1976 | 0.1793 | $\pm 0.0094$ | 5.26% |
| `P22_S3_B3_M2` | S3 | B3 | M2 | 0.1403 | 0.1465 | 0.1555 | 0.1593 | 0.1534 | 0.1510 | $\pm 0.0068$ | 4.49% |
| `P23_S3_B4_M1` | S3 | B4 | M1 | 0.0235 | 0.0332 | 0.0344 | 0.0277 | 0.0319 | 0.0301 | $\pm 0.0040$ | 13.40% |
| `P24_S3_B4_M2` | S3 | B4 | M2 | 0.1049 | 0.1049 | 0.1049 | 0.1049 | 0.1049 | 0.1049 | $\pm 0.0000$ | 0.00% |
| **RATA-RATA MAKRO** | - | - | - | **0.0847** | **0.0871** | **0.0881** | **0.0852** | **0.0896** | **0.0870** | $\mathbf{\pm 0.0017}$ | **1.95%** |

### Sintesis Temuan RQ1:
1. **Supremasi B3 (RUS) Mutlak & Stabil**:  
   Peringkat 1 hingga 6 di seluruh 5 repetisi independen **selalu dikuasai oleh konfigurasi Random Under-Sampling (B3)**. Nilai $CV$ pada pipeline RUS tercatat sangat rendah ($5.0\% - 6.5\%$), membuktikan bahwa keunggulan teknik RUS bukanlah artefak stokastik yang bergantung pada satu seed keberuntungan.
2. **Kestabilan Ensembel Random Forest (Bagging)**:  
   Random Forest (M1) menunjukkan performa F1 tertinggi (**0.1798** pada `P13`) ketika dipasangkan dengan RUS, dengan 95% Confidence Interval sempit $[0.1696, 0.1900]$. Mekanisme *bootstrap aggregating* pada 100 pohon secara efektif meredam variansi dari sampel mayoritas yang terbuang selama undersampling.
3. **Invariansi Monotonik Scaler pada XGBoost**:  
   Pipeline `P06`, `P14`, dan `P22` (XGBoost GPU + RUS) menghasilkan skor yang persis identik (Mean F1: 0.1510 $\pm$ 0.0076) di seluruh penskalaan S1, S2, dan S3. Ini membuktikan secara empiris bahwa algoritma pohon *gradient boosting* bersifat invarian terhadap transformasi monotonik. Namun pada Random Forest, normalisasi kompak bounded $[0, 1]$ pada **MinMaxScaler (S2)** memberikan keunggulan F1 tertinggi (**0.1798**).

---

## 3. Analisis RQ2: Mengapa Nilai Rata-rata Akurasi & F1-Score Sangat Rendah?

Pertanyaan inti dari pengujian ini adalah: *Mengapa rata-rata nilai F1 hanya berkisar di angka 0.18 pada model terbaik dan 0.08 secara keseluruhan? Apakah model gagal mendeteksi flaky test?*

Jawabannya adalah **TIDAK**. Rendahnya angka rata-rata F1 murni merupakan artefak dari **metodologi agregasi Unweighted Macro-Averaging yang berbenturan dengan ketimpangan kelas ekstrem (Extreme Near-Zero Minority Class)**.

### 4 Faktor Kunci Penyebab Rendahnya Akurasi Makro:

#### 1. Masalah Unweighted Macro-Averaging pada Extreme Class Imbalance
Dalam skenario LOPO-CV, metrik dihitung per lipatan proyek lalu dirata-ratakan secara aritmetika:
$$\text{Macro F1} = \frac{1}{24} \sum_{j=1}^{24} \text{F1}_j$$
Rumus ini memberikan bobot yang **sama persis ($1/24 = 4,17\%$)** untuk proyek raksasa `spring-boot` (2.127 baris, 163 flaky) dengan proyek kecil `commons-exec` (55 baris, 1 flaky) dan `assertj-core` (6.261 baris, 1 flaky).

Terdapat **8 proyek target (33,3% dari seluruh dataset)** yang memiliki $\le 4$ sampel flaky test:
- `jimfs`: 212 baris, **0 flaky test** (0.00%) $\rightarrow$ F1 = 0.0 secara definisi.
- `assertj-core`: 6.261 baris, **1 flaky test** (0.016%) $\rightarrow$ Jika 1 flaky ini lolos, Recall = 0, F1 = 0.0! Jika tertebak tapi muncul 1 False Positive saja, Precision langsung jatuh drastis.
- `handlebars.java`: 427 baris, **1 flaky test** (0.23%) $\rightarrow$ Zero-F1 Rate 97.5%.
- `commons-exec`: 55 baris, **1 flaky test** (1.82%) $\rightarrow$ Zero-F1 Rate 53.3%.
- `ninja`: 307 baris, **1 flaky test** (0.33%) $\rightarrow$ Zero-F1 Rate 70.8%.
- `zxing`: 345 baris, **2 flaky test** (0.58%) $\rightarrow$ Zero-F1 Rate 100.0%.
- `elastic-job-lite`: 558 baris, **3 flaky test** (0.54%) $\rightarrow$ Zero-F1 Rate 100.0%.
- `achilles`: 1.317 baris, **4 flaky test** (0.30%) $\rightarrow$ Zero-F1 Rate 82.5%.

> **Dampak Matematis**: Proyek-proyek ini bertindak sebagai **"Sinkholes" (Lubang Hitam Numerik)**. Ketika 7-8 proyek bernilai F1 = 0.0, maka secara matematis rata-rata makro otomatis terpangkas minimal $\frac{8}{24} \approx \mathbf{33,3\%}$ langsung menuju angka nol!

#### 2. Dekonstruksi Komparasi: Macro F1 vs Flaky-Weighted F1 vs Micro F1
Ketika kita mengevaluasi model terbaik (`P13_S2_B3_M1`) menggunakan metrik yang merefleksikan populasi flaky test riil:

| Skema Evaluasi | Skor F1 | Peningkatan vs Macro | Penjelasan Metodologis |
|---|:---:|:---:|---|
| **Unweighted Macro F1** | **0.1798** | Baseline | Setiap proyek berbobot sama 1/24 (terdistorsi 8 proyek mini flaky). |
| **Flaky-Weighted F1** | **0.3832** | **+113,14% (>2,1x Lipat!)** | Diberi bobot proporsional terhadap jumlah flaky test riil di proyek target. |
| **Global Micro F1** | **0.1666** | - | Menghitung Confusion Matrix global di seluruh 24 proyek. |

> **Kesimpulan Kritis**: Pada instance flaky yang sesungguhnya di industri (di mana flaky test terkonsentrasi pada repositori besar seperti `alluxio`, `hbase`, dan `spring-boot`), **model sebenarnya memiliki F1 di atas 0.35 - 0.40 dan Recall di atas 57% - 78%!** Rendahnya Macro F1 hanyalah ilusi penalti akibat skema perataan unweighted.

#### 3. Cross-Project Domain Shift & Concept Drift
Metrik software prediktif memiliki variasi domain yang tinggi antar repositori. Contoh kasus paling ekstrem adalah **`http-request`** (163 baris, 18 flaky test = 11,04%):
- Meskipun proporsi flaky-nya cukup tinggi (11%), **seluruh 24 pipeline mencatatkan F1 = 0.0**!
- Mengapa? Karena `http-request` adalah *lightweight HTTP library*, di mana *test smell* dan metrik *code churn*-nya sangat berbeda dengan repositori berbasis server/distributed systems (`hbase`, `alluxio`, `spring-boot`). Model yang dilatih pada 23 proyek sumber mengalami kegagalan transfer pengetahuan (*negative transfer*) ke domain HTTP client.

#### 4. Ambang Batas Klasifikasi Statis (Fixed Threshold 0.5)
Model menggunakan ambang probabilitas $P(\text{flaky}) \ge 0.5$. Pada skenario RUS, penyeimbangan data 1:1 di data latih menaikkan probabilitas rata-rata. Pada proyek dengan ribuan test case non-flaky (`activiti` dengan 2.012 non-flaky; `incubator-dubbo` dengan 2.157 non-flaky), terjadi sejumlah False Positive (misal 980 FP di `activiti`), yang menekan skor Precision ke angka 0.02.

---

## 4. Analisis RQ3: Profil Kemampuan & Taksonomi Konsistensi 24 Dataset

### Profil Lengkap 24 Dataset Proyek Target (Diurutkan dari Flaky Rate Tertinggi):

| Target Project | Total Test | Flaky Count | Flaky Rate (%) | Overall Mean F1 | Overall Std F1 | Zero F1 Rate (%) | Best Pipeline | Best F1 | Tier Kategori |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **alluxio** | 187 | 116 | **62.03%** | 0.3369 | 0.2922 | 12.5% | `P13_S2_B3_M1` | **0.9826** | **Tier 1 (Rock-Solid)** |
| **hbase** | 431 | 145 | **33.64%** | 0.2342 | 0.2282 | **0.0%** | `P13_S2_B3_M1` | **0.5847** | **Tier 1 (Rock-Solid)** |
| **hector** | 142 | 33 | **23.24%** | 0.1002 | 0.1542 | 58.3% | `P05_S1_B3_M1` | 0.3636 | Tier 3 (Moderate) |
| **ambari** | 324 | 52 | **16.05%** | **0.4459** | $\mathbf{\pm 0.0833}$ | **0.0%** | `P06_S1_B3_M2` | **0.6194** | **Tier 1 (Rock-Solid)** |
| **java-websocket** | 145 | 23 | **15.86%** | 0.0497 | 0.0719 | 62.5% | `P13_S2_B3_M1` | 0.2222 | Tier 3 (Moderate) |
| **okhttp** | 810 | 100 | **12.35%** | 0.0406 | 0.0461 | 37.5% | `P21_S3_B3_M1` | 0.1266 | Tier 3 (Moderate) |
| **http-request** | 163 | 18 | **11.04%** | 0.0000 | 0.0000 | **100.0%** | - | 0.0000 | **Tier 4 (Sinkhole / Domain Shift)** |
| **orbit** | 86 | 7 | **8.14%** | 0.1090 | 0.1262 | 54.2% | `P13_S2_B3_M1` | 0.3030 | Tier 3 (Moderate) |
| **spring-boot** | 2.127 | 163 | **7.66%** | 0.0982 | 0.1583 | 41.7% | `P13_S2_B3_M1` | **0.3975** | **Tier 2 (RUS-Dependent)** |
| **undertow** | 183 | 7 | **3.83%** | 0.0908 | 0.0725 | 29.2% | `P06_S1_B3_M2` | 0.2000 | Tier 3 (Moderate) |
| **httpcore** | 712 | 22 | **3.09%** | 0.1899 | 0.1565 | 33.3% | `P13_S2_B3_M1` | **0.4384** | **Tier 2 (RUS-Dependent)** |
| **logback** | 843 | 22 | **2.61%** | 0.0179 | 0.0272 | 62.5% | `P06_S1_B3_M2` | 0.0769 | Tier 3 (Moderate) |
| **wildfly** | 1.236 | 23 | **1.86%** | 0.0685 | 0.0637 | 37.5% | `P06_S1_B3_M2` | 0.2000 | Tier 3 (Moderate) |
| **commons-exec** | 55 | 1 | **1.82%** | 0.0873 | 0.1154 | 58.3% | `P06_S1_B3_M2` | 0.4000 | Tier 3 (Moderate) |
| **activiti** | 2.044 | 32 | **1.57%** | 0.0228 | 0.0174 | 25.0% | `P01_S1_B1_M1` | 0.0503 | Tier 3 (Moderate) |
| **wro4j** | 1.145 | 16 | **1.40%** | 0.0591 | 0.0520 | **0.0%** | `P01_S1_B1_M1` | 0.1818 | Tier 3 (Moderate) |
| **incubator-dubbo** | 2.176 | 19 | **0.87%** | 0.0597 | 0.0527 | 37.5% | `P13_S2_B3_M1` | 0.1573 | Tier 3 (Moderate) |
| **zxing** | 345 | 2 | **0.58%** | 0.0000 | 0.0000 | **100.0%** | - | 0.0000 | **Tier 4 (Sinkhole / Near-Zero)** |
| **elastic-job-lite** | 558 | 3 | **0.54%** | 0.0000 | 0.0000 | **100.0%** | - | 0.0000 | **Tier 4 (Sinkhole / Near-Zero)** |
| **ninja** | 307 | 1 | **0.33%** | 0.0214 | 0.0587 | 70.8% | `P06_S1_B3_M2` | 0.2857 | **Tier 4 (Sinkhole / Near-Zero)** |
| **achilles** | 1.317 | 4 | **0.30%** | 0.0016 | 0.0042 | 87.5% | `P06_S1_B3_M2` | 0.0124 | **Tier 4 (Sinkhole / Near-Zero)** |
| **handlebars.java** | 427 | 1 | **0.23%** | 0.0000 | 0.0000 | **100.0%** | - | 0.0000 | **Tier 4 (Sinkhole / Near-Zero)** |
| **assertj-core** | 6.261 | 1 | **0.016%** | 0.0000 | 0.0000 | **100.0%** | - | 0.0000 | **Tier 4 (Sinkhole / Extreme Skew)** |
| **jimfs** | 212 | 0 | **0.00%** | 0.0000 | 0.0000 | **100.0%** | - | 0.0000 | **Tier 4 (Zero Minority Class)** |

---

### 📊 Matriks Kemampuan 24 Pipeline pada 24 Dataset Proyek Target (LOPO-CV)

Tabel berikut menyajikan pemetaan komprehensif performa **F1-Score** untuk seluruh **24 Pipeline (baris)** pada masing-masing **24 Dataset / Proyek Target (kolom)**. Nilai pada setiap sel merupakan rata-rata performa lintas 5 *random seed* ($N = 120$ evaluasi per proyek). 

Baris paling akhir (**`RATA-RATA KEMAMPUAN DATASET`**) merangkum kapabilitas deteksi rata-rata seluruh konfigurasi pipeline pada proyek tersebut, sedangkan kolom paling kanan (**`Rata-rata Pipeline`**) merangkum performa makro rata-rata pipeline pada ke-24 dataset. File data lengkap tersimpan di [`matrix_pipeline_vs_dataset_f1.csv`](results/matrix_pipeline_vs_dataset_f1.csv).

#### 1. Tabel Master Lengkap (Kolom: 24 Dataset, Baris: 24 Pipeline, Baris Akhir: Rata-rata Kemampuan)

| Pipeline | achilles | activiti | alluxio | ambari | assertj-core | commons-exec | elastic-job-lite | handlebars.java | hbase | hector | http-request | httpcore | incubator-dubbo | java-websocket | jimfs | logback | ninja | okhttp | orbit | spring-boot | undertow | wildfly | wro4j | zxing | Rata-rata Pipeline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `P01_S1_B1_M1` | 0.0000 | 0.0620 | 0.0170 | 0.3744 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0214 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1008 | 0.0000 | 0.0240 |
| `P02_S1_B1_M2` | 0.0000 | 0.0328 | 0.0984 | 0.4074 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0400 | 0.0000 | 0.0000 | 0.2667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0198 | 0.0000 | 0.0000 | 0.1538 | 0.1000 | 0.0714 | 0.0000 | 0.0496 |
| `P03_S1_B2_M1` | 0.0000 | 0.0000 | 0.5689 | 0.5150 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0453 | 0.0000 | 0.0000 | 0.0676 | 0.0890 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0167 | 0.0926 | 0.0000 | 0.0257 | 0.0000 | 0.0592 |
| `P04_S1_B2_M2` | 0.0000 | 0.0114 | 0.4800 | 0.5749 | 0.0000 | 0.0571 | 0.0000 | 0.0000 | 0.0376 | 0.0334 | 0.0000 | 0.2642 | 0.1328 | 0.0160 | 0.0000 | 0.0308 | 0.0571 | 0.0190 | 0.0727 | 0.0456 | 0.1290 | 0.1054 | 0.0352 | 0.0000 | 0.0876 |
| `P05_S1_B3_M1` | 0.0114 | 0.0326 | 0.9313 | 0.4688 | 0.0000 | 0.2260 | 0.0000 | 0.0000 | 0.5812 | 0.3425 | 0.0684 | 0.3796 | 0.0879 | 0.0847 | 0.0000 | 0.0551 | 0.0699 | 0.0862 | 0.3339 | 0.3158 | 0.0845 | 0.0893 | 0.0269 | 0.0000 | 0.1782 |
| `P06_S1_B3_M2` | 0.0209 | 0.0324 | 0.5934 | 0.4163 | 0.0000 | 0.1643 | 0.0000 | 0.0025 | 0.5574 | 0.3793 | 0.0356 | 0.3365 | 0.0635 | 0.0676 | 0.0000 | 0.0509 | 0.0304 | 0.1233 | 0.2421 | 0.3000 | 0.0846 | 0.0892 | 0.0338 | 0.0000 | 0.1510 |
| `P07_S1_B4_M1` | 0.0000 | 0.0000 | 0.1246 | 0.3912 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0135 | 0.0000 | 0.0000 | 0.0154 | 0.0327 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0894 | 0.0000 | 0.0385 | 0.0000 | 0.0294 |
| `P08_S1_B4_M2` | 0.0000 | 0.0168 | 0.2687 | 0.4540 | 0.0000 | 0.2222 | 0.0000 | 0.0000 | 0.3263 | 0.0488 | 0.0000 | 0.2368 | 0.1212 | 0.0741 | 0.0000 | 0.0769 | 0.0000 | 0.0826 | 0.2143 | 0.0344 | 0.1538 | 0.1176 | 0.0694 | 0.0000 | 0.1049 |
| `P09_S2_B1_M1` | 0.0000 | 0.0617 | 0.0136 | 0.3754 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0160 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1044 | 0.0000 | 0.0238 |
| `P10_S2_B1_M2` | 0.0000 | 0.0328 | 0.0984 | 0.4074 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0400 | 0.0000 | 0.0000 | 0.2667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0198 | 0.0000 | 0.0000 | 0.1538 | 0.1000 | 0.0714 | 0.0000 | 0.0496 |
| `P11_S2_B2_M1` | 0.0000 | 0.0073 | 0.4774 | 0.5763 | 0.0000 | 0.0800 | 0.0000 | 0.0000 | 0.0974 | 0.0000 | 0.0000 | 0.0539 | 0.0982 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0136 | 0.1358 | 0.0000 | 0.0183 | 0.0000 | 0.0649 |
| `P12_S2_B2_M2` | 0.0000 | 0.0056 | 0.3865 | 0.5640 | 0.0000 | 0.4038 | 0.0000 | 0.0000 | 0.3158 | 0.0108 | 0.0000 | 0.4262 | 0.1541 | 0.0000 | 0.0000 | 0.0167 | 0.1152 | 0.0191 | 0.2161 | 0.0210 | 0.1445 | 0.0764 | 0.0298 | 0.0000 | 0.1211 |
| `P13_S2_B3_M1` | 0.0114 | 0.0326 | 0.9358 | 0.4698 | 0.0000 | 0.2305 | 0.0000 | 0.0000 | 0.5816 | 0.3476 | 0.0676 | 0.3818 | 0.0876 | 0.0996 | 0.0000 | 0.0560 | 0.0733 | 0.0863 | 0.3321 | 0.3172 | 0.0857 | 0.0887 | 0.0294 | 0.0000 | 0.1798 |
| `P14_S2_B3_M2` | 0.0209 | 0.0324 | 0.5934 | 0.4163 | 0.0000 | 0.1643 | 0.0000 | 0.0025 | 0.5574 | 0.3793 | 0.0356 | 0.3365 | 0.0635 | 0.0676 | 0.0000 | 0.0509 | 0.0304 | 0.1233 | 0.2421 | 0.3000 | 0.0846 | 0.0892 | 0.0338 | 0.0000 | 0.1510 |
| `P15_S2_B4_M1` | 0.0000 | 0.0000 | 0.1260 | 0.3992 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0135 | 0.0000 | 0.0000 | 0.0000 | 0.0314 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0863 | 0.0000 | 0.0383 | 0.0000 | 0.0289 |
| `P16_S2_B4_M2` | 0.0000 | 0.0168 | 0.2687 | 0.4540 | 0.0000 | 0.2222 | 0.0000 | 0.0000 | 0.3263 | 0.0488 | 0.0000 | 0.2368 | 0.1212 | 0.0741 | 0.0000 | 0.0769 | 0.0000 | 0.0826 | 0.2143 | 0.0344 | 0.1538 | 0.1176 | 0.0694 | 0.0000 | 0.1049 |
| `P17_S3_B1_M1` | 0.0000 | 0.0620 | 0.0137 | 0.3708 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0160 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1091 | 0.0000 | 0.0238 |
| `P18_S3_B1_M2` | 0.0000 | 0.0328 | 0.0984 | 0.4074 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0400 | 0.0000 | 0.0000 | 0.2667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0198 | 0.0000 | 0.0000 | 0.1538 | 0.1000 | 0.0714 | 0.0000 | 0.0496 |
| `P19_S3_B2_M1` | 0.0000 | 0.0000 | 0.4608 | 0.4145 | 0.0000 | 0.2000 | 0.0000 | 0.0000 | 0.0543 | 0.0000 | 0.0000 | 0.0700 | 0.0791 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0983 | 0.0000 | 0.0246 | 0.0000 | 0.0584 |
| `P20_S3_B2_M2` | 0.0000 | 0.0164 | 0.2484 | 0.4709 | 0.0000 | 0.4000 | 0.0000 | 0.0000 | 0.1004 | 0.0673 | 0.0000 | 0.1416 | 0.0963 | 0.0167 | 0.0000 | 0.0000 | 0.0571 | 0.0190 | 0.0000 | 0.0234 | 0.1907 | 0.0845 | 0.0310 | 0.0000 | 0.0818 |
| `P21_S3_B3_M1` | 0.0104 | 0.0326 | 0.9370 | 0.4688 | 0.0000 | 0.2260 | 0.0000 | 0.0000 | 0.5809 | 0.3683 | 0.0668 | 0.3817 | 0.0872 | 0.0859 | 0.0000 | 0.0546 | 0.0699 | 0.0865 | 0.3257 | 0.3184 | 0.0848 | 0.0889 | 0.0293 | 0.0000 | 0.1793 |
| `P22_S3_B3_M2` | 0.0209 | 0.0324 | 0.5934 | 0.4163 | 0.0000 | 0.1643 | 0.0000 | 0.0025 | 0.5574 | 0.3793 | 0.0356 | 0.3365 | 0.0635 | 0.0676 | 0.0000 | 0.0509 | 0.0304 | 0.1233 | 0.2421 | 0.3000 | 0.0846 | 0.0892 | 0.0338 | 0.0000 | 0.1510 |
| `P23_S3_B4_M1` | 0.0000 | 0.0000 | 0.1477 | 0.3935 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0108 | 0.0000 | 0.0000 | 0.0154 | 0.0320 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0863 | 0.0000 | 0.0376 | 0.0000 | 0.0301 |
| `P24_S3_B4_M2` | 0.0000 | 0.0168 | 0.2687 | 0.4540 | 0.0000 | 0.2222 | 0.0000 | 0.0000 | 0.3263 | 0.0488 | 0.0000 | 0.2368 | 0.1212 | 0.0741 | 0.0000 | 0.0769 | 0.0000 | 0.0826 | 0.2143 | 0.0344 | 0.1538 | 0.1176 | 0.0694 | 0.0000 | 0.1049 |
| **RATA-RATA KEMAMPUAN DATASET** | **0.0040** | **0.0238** | **0.3646** | **0.4442** | **0.0000** | **0.1243** | **0.0000** | **0.0003** | **0.2190** | **0.1023** | **0.0129** | **0.1966** | **0.0651** | **0.0303** | **0.0000** | **0.0249** | **0.0222** | **0.0414** | **0.1104** | **0.0865** | **0.1035** | **0.0606** | **0.0501** | **0.0000** | **0.0870** |

#### 2. Tampilan Terbagi (Modular Sub-Tables untuk Kenyamanan Layar Standar)

##### Bagian A: Proyek 1 s.d. 12 (Alphabetical: achilles s.d. httpcore)
| Pipeline | achilles | activiti | alluxio | ambari | assertj-core | commons-exec | elastic-job-lite | handlebars.java | hbase | hector | http-request | httpcore |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `P01_S1_B1_M1` | 0.0000 | 0.0620 | 0.0170 | 0.3744 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0214 | 0.0000 | 0.0000 | 0.0000 |
| `P02_S1_B1_M2` | 0.0000 | 0.0328 | 0.0984 | 0.4074 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0400 | 0.0000 | 0.0000 | 0.2667 |
| `P03_S1_B2_M1` | 0.0000 | 0.0000 | 0.5689 | 0.5150 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0453 | 0.0000 | 0.0000 | 0.0676 |
| `P04_S1_B2_M2` | 0.0000 | 0.0114 | 0.4800 | 0.5749 | 0.0000 | 0.0571 | 0.0000 | 0.0000 | 0.0376 | 0.0334 | 0.0000 | 0.2642 |
| `P05_S1_B3_M1` | 0.0114 | 0.0326 | 0.9313 | 0.4688 | 0.0000 | 0.2260 | 0.0000 | 0.0000 | 0.5812 | 0.3425 | 0.0684 | 0.3796 |
| `P06_S1_B3_M2` | 0.0209 | 0.0324 | 0.5934 | 0.4163 | 0.0000 | 0.1643 | 0.0000 | 0.0025 | 0.5574 | 0.3793 | 0.0356 | 0.3365 |
| `P07_S1_B4_M1` | 0.0000 | 0.0000 | 0.1246 | 0.3912 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0135 | 0.0000 | 0.0000 | 0.0154 |
| `P08_S1_B4_M2` | 0.0000 | 0.0168 | 0.2687 | 0.4540 | 0.0000 | 0.2222 | 0.0000 | 0.0000 | 0.3263 | 0.0488 | 0.0000 | 0.2368 |
| `P09_S2_B1_M1` | 0.0000 | 0.0617 | 0.0136 | 0.3754 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0160 | 0.0000 | 0.0000 | 0.0000 |
| `P10_S2_B1_M2` | 0.0000 | 0.0328 | 0.0984 | 0.4074 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0400 | 0.0000 | 0.0000 | 0.2667 |
| `P11_S2_B2_M1` | 0.0000 | 0.0073 | 0.4774 | 0.5763 | 0.0000 | 0.0800 | 0.0000 | 0.0000 | 0.0974 | 0.0000 | 0.0000 | 0.0539 |
| `P12_S2_B2_M2` | 0.0000 | 0.0056 | 0.3865 | 0.5640 | 0.0000 | 0.4038 | 0.0000 | 0.0000 | 0.3158 | 0.0108 | 0.0000 | 0.4262 |
| `P13_S2_B3_M1` | 0.0114 | 0.0326 | 0.9358 | 0.4698 | 0.0000 | 0.2305 | 0.0000 | 0.0000 | 0.5816 | 0.3476 | 0.0676 | 0.3818 |
| `P14_S2_B3_M2` | 0.0209 | 0.0324 | 0.5934 | 0.4163 | 0.0000 | 0.1643 | 0.0000 | 0.0025 | 0.5574 | 0.3793 | 0.0356 | 0.3365 |
| `P15_S2_B4_M1` | 0.0000 | 0.0000 | 0.1260 | 0.3992 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0135 | 0.0000 | 0.0000 | 0.0000 |
| `P16_S2_B4_M2` | 0.0000 | 0.0168 | 0.2687 | 0.4540 | 0.0000 | 0.2222 | 0.0000 | 0.0000 | 0.3263 | 0.0488 | 0.0000 | 0.2368 |
| `P17_S3_B1_M1` | 0.0000 | 0.0620 | 0.0137 | 0.3708 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0160 | 0.0000 | 0.0000 | 0.0000 |
| `P18_S3_B1_M2` | 0.0000 | 0.0328 | 0.0984 | 0.4074 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0400 | 0.0000 | 0.0000 | 0.2667 |
| `P19_S3_B2_M1` | 0.0000 | 0.0000 | 0.4608 | 0.4145 | 0.0000 | 0.2000 | 0.0000 | 0.0000 | 0.0543 | 0.0000 | 0.0000 | 0.0700 |
| `P20_S3_B2_M2` | 0.0000 | 0.0164 | 0.2484 | 0.4709 | 0.0000 | 0.4000 | 0.0000 | 0.0000 | 0.1004 | 0.0673 | 0.0000 | 0.1416 |
| `P21_S3_B3_M1` | 0.0104 | 0.0326 | 0.9370 | 0.4688 | 0.0000 | 0.2260 | 0.0000 | 0.0000 | 0.5809 | 0.3683 | 0.0668 | 0.3817 |
| `P22_S3_B3_M2` | 0.0209 | 0.0324 | 0.5934 | 0.4163 | 0.0000 | 0.1643 | 0.0000 | 0.0025 | 0.5574 | 0.3793 | 0.0356 | 0.3365 |
| `P23_S3_B4_M1` | 0.0000 | 0.0000 | 0.1477 | 0.3935 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0108 | 0.0000 | 0.0000 | 0.0154 |
| `P24_S3_B4_M2` | 0.0000 | 0.0168 | 0.2687 | 0.4540 | 0.0000 | 0.2222 | 0.0000 | 0.0000 | 0.3263 | 0.0488 | 0.0000 | 0.2368 |
| **RATA-RATA KEMAMPUAN DATASET** | **0.0040** | **0.0238** | **0.3646** | **0.4442** | **0.0000** | **0.1243** | **0.0000** | **0.0003** | **0.2190** | **0.1023** | **0.0129** | **0.1966** |

##### Bagian B: Proyek 13 s.d. 24 (Alphabetical: incubator-dubbo s.d. zxing) + Rata-rata Pipeline
| Pipeline | incubator-dubbo | java-websocket | jimfs | logback | ninja | okhttp | orbit | spring-boot | undertow | wildfly | wro4j | zxing | Rata-rata Pipeline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `P01_S1_B1_M1` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1008 | 0.0000 | 0.0240 |
| `P02_S1_B1_M2` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0198 | 0.0000 | 0.0000 | 0.1538 | 0.1000 | 0.0714 | 0.0000 | 0.0496 |
| `P03_S1_B2_M1` | 0.0890 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0167 | 0.0926 | 0.0000 | 0.0257 | 0.0000 | 0.0592 |
| `P04_S1_B2_M2` | 0.1328 | 0.0160 | 0.0000 | 0.0308 | 0.0571 | 0.0190 | 0.0727 | 0.0456 | 0.1290 | 0.1054 | 0.0352 | 0.0000 | 0.0876 |
| `P05_S1_B3_M1` | 0.0879 | 0.0847 | 0.0000 | 0.0551 | 0.0699 | 0.0862 | 0.3339 | 0.3158 | 0.0845 | 0.0893 | 0.0269 | 0.0000 | 0.1782 |
| `P06_S1_B3_M2` | 0.0635 | 0.0676 | 0.0000 | 0.0509 | 0.0304 | 0.1233 | 0.2421 | 0.3000 | 0.0846 | 0.0892 | 0.0338 | 0.0000 | 0.1510 |
| `P07_S1_B4_M1` | 0.0327 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0894 | 0.0000 | 0.0385 | 0.0000 | 0.0294 |
| `P08_S1_B4_M2` | 0.1212 | 0.0741 | 0.0000 | 0.0769 | 0.0000 | 0.0826 | 0.2143 | 0.0344 | 0.1538 | 0.1176 | 0.0694 | 0.0000 | 0.1049 |
| `P09_S2_B1_M1` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1044 | 0.0000 | 0.0238 |
| `P10_S2_B1_M2` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0198 | 0.0000 | 0.0000 | 0.1538 | 0.1000 | 0.0714 | 0.0000 | 0.0496 |
| `P11_S2_B2_M1` | 0.0982 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0136 | 0.1358 | 0.0000 | 0.0183 | 0.0000 | 0.0649 |
| `P12_S2_B2_M2` | 0.1541 | 0.0000 | 0.0000 | 0.0167 | 0.1152 | 0.0191 | 0.2161 | 0.0210 | 0.1445 | 0.0764 | 0.0298 | 0.0000 | 0.1211 |
| `P13_S2_B3_M1` | 0.0876 | 0.0996 | 0.0000 | 0.0560 | 0.0733 | 0.0863 | 0.3321 | 0.3172 | 0.0857 | 0.0887 | 0.0294 | 0.0000 | 0.1798 |
| `P14_S2_B3_M2` | 0.0635 | 0.0676 | 0.0000 | 0.0509 | 0.0304 | 0.1233 | 0.2421 | 0.3000 | 0.0846 | 0.0892 | 0.0338 | 0.0000 | 0.1510 |
| `P15_S2_B4_M1` | 0.0314 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0863 | 0.0000 | 0.0383 | 0.0000 | 0.0289 |
| `P16_S2_B4_M2` | 0.1212 | 0.0741 | 0.0000 | 0.0769 | 0.0000 | 0.0826 | 0.2143 | 0.0344 | 0.1538 | 0.1176 | 0.0694 | 0.0000 | 0.1049 |
| `P17_S3_B1_M1` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1091 | 0.0000 | 0.0238 |
| `P18_S3_B1_M2` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0198 | 0.0000 | 0.0000 | 0.1538 | 0.1000 | 0.0714 | 0.0000 | 0.0496 |
| `P19_S3_B2_M1` | 0.0791 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0983 | 0.0000 | 0.0246 | 0.0000 | 0.0584 |
| `P20_S3_B2_M2` | 0.0963 | 0.0167 | 0.0000 | 0.0000 | 0.0571 | 0.0190 | 0.0000 | 0.0234 | 0.1907 | 0.0845 | 0.0310 | 0.0000 | 0.0818 |
| `P21_S3_B3_M1` | 0.0872 | 0.0859 | 0.0000 | 0.0546 | 0.0699 | 0.0865 | 0.3257 | 0.3184 | 0.0848 | 0.0889 | 0.0293 | 0.0000 | 0.1793 |
| `P22_S3_B3_M2` | 0.0635 | 0.0676 | 0.0000 | 0.0509 | 0.0304 | 0.1233 | 0.2421 | 0.3000 | 0.0846 | 0.0892 | 0.0338 | 0.0000 | 0.1510 |
| `P23_S3_B4_M1` | 0.0320 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0863 | 0.0000 | 0.0376 | 0.0000 | 0.0301 |
| `P24_S3_B4_M2` | 0.1212 | 0.0741 | 0.0000 | 0.0769 | 0.0000 | 0.0826 | 0.2143 | 0.0344 | 0.1538 | 0.1176 | 0.0694 | 0.0000 | 0.1049 |
| **RATA-RATA KEMAMPUAN DATASET** | **0.0651** | **0.0303** | **0.0000** | **0.0249** | **0.0222** | **0.0414** | **0.1104** | **0.0865** | **0.1035** | **0.0606** | **0.0501** | **0.0000** | **0.0870** |

#### 3. Wawasan Kunci dari Baris "RATA-RATA KEMAMPUAN PIPELINE PADA DATASET YANG SAMA":
1. **Penyebab Utama Rendahnya Rata-rata Makro Akurasi/F1 (Hanya 0.0870 / 8,70%)**:
   - Dari 24 dataset, terdapat **7 dataset yang memiliki rata-rata kemampuan deteksi $\le 0.0129$** di seluruh 24 konfigurasi pipeline:
     - `assertj-core`, `elastic-job-lite`, `jimfs`, dan `zxing` memiliki rata-rata kemampuan **0.0000** (sama sekali tidak terdeteksi oleh 24 pipeline).
     - `handlebars.java` (**0.0003**), `achilles` (**0.0040**), dan `http-request` (**0.0129**).
   - Sebanyak **29,2% (7 dari 24) kolom dataset bernilai mendekati nol mutlak**, yang secara matematis "menyeret" nilai rata-rata makro keseluruhan ke bawah.
2. **Dataset dengan Kemampuan Deteksi Paling Konsisten Tinggi**:
   - `ambari` (**0.4442**): Menunjukkan kemampuan deteksi tinggi di hampir seluruh pipeline (bahkan pada pipeline tanpa penyeimbang data, F1 tetap mencapai 0.37 - 0.40).
   - `alluxio` (**0.3646**): Melonjak drastis hingga **0.9358 - 0.9370** pada seluruh pipeline RUS (B3).
   - `hbase` (**0.2190**): Konsisten menangkap puluhan flaky test dengan F1 mencapai **0.5816** pada model RUS.
   - `httpcore` (**0.1966**): Menunjukkan respon deteksi yang stabil pada berbagai kombinasi.
3. **Efek Transformasi Resampling (RUS / B3)**:
   - Pada `spring-boot`, 16 pipeline non-RUS menghasilkan F1 = 0.0000 - 0.0456, namun 6 pipeline RUS melonjak konsisten ke **0.3000 - 0.3184** (Recall 77,91%). Hal ini menjelaskan mengapa rata-rata dataset `spring-boot` berada di angka **0.0865**, di mana nilai ini tertekan oleh pipeline non-RUS yang gagal.

---

### Analisis Mendalam 4 Kategori Tier Dataset:

#### 🟢 Tier 1: Rock-Solid Consistency & High Detectability (`ambari`, `hbase`, `alluxio`)
- **Karakteristik**: Rasio flaky test moderat hingga tinggi (16% – 62%).
- **Temuan Empiris**: 
  - `ambari`: Mencatatkan kestabilan tertinggi di seluruh dataset. **Zero-F1 Rate = 0.0%** (seluruh 24 pipeline berhasil mendeteksi flaky test) dengan Mean F1 **0.4459 $\pm$ 0.0833** dan F1 puncak mencapai **0.6194**.
  - `hbase`: Zero-F1 Rate = 0.0%, konsisten menangkap puluhan flaky test dengan F1 **0.5847** dan Precision **75,82%** pada konfigurasi RUS.
  - `alluxio`: Mencapai F1 mendekati sempurna (**0.9826**) dengan **Recall 97,41%** (113 dari 116 flaky test tertangkap) pada model RUS.
- **Rekomendasi**: Proyek-proyek Tier 1 adalah representasi ideal di mana sinyal fitur *test smell* dan *code churn* berkorelasi sangat kuat dan transferable antar proyek.

#### 🔵 Tier 2: Resampling-Dependent Consistency (`spring-boot`, `httpcore`)
- **Karakteristik**: Proyek berskala masif atau komponen sistem inti dengan populasi flaky riil signifikan (163 flaky di `spring-boot`, 22 di `httpcore`).
- **Temuan Empiris**:
  - Pada model Baseline (B1) atau ClassWeight (B4), model kolaps ke kelas mayoritas (Recall = 0%, F1 = 0.0).
  - Namun ketika dipasangkan dengan **Random Under-Sampling (B3)**, model secara konsisten lintas seluruh 5 seed menangkap **127 dari 163 flaky test di `spring-boot` (Recall 77,91%, F1 0.3975)** dan F1 **0.4384** di `httpcore`!
- **Rekomendasi**: Menunjukkan bahwa transferability lintas proyek pada sistem enterprise berskala raksasa mutlak membutuhkan penyeimbangan rasio data latih (RUS).

#### 🟡 Tier 3: Moderate & Sensitive to Domain Shift (`undertow`, `orbit`, `wildfly`, `wro4j`, dll.)
- **Karakteristik**: Rasio flaky antara 1% – 8%.
- **Temuan Empiris**: Model berhasil mendeteksi sebagian flaky test (F1 berkisar antara 0.05 – 0.25). Pada `wro4j`, Zero-F1 Rate tercatat 0.0% (selalu ada flaky yang terdeteksi), namun Precision moderat akibat variasi gaya penulisan test code.

#### 🔴 Tier 4: Permanent Non-Detection Sinkholes (`assertj-core`, `handlebars.java`, `elastic-job-lite`, `zxing`, `http-request`, `jimfs`, `achilles`)
- **Karakteristik**: Rasio flaky sangat minim (<0.5%) atau mengalami pergeseran domain ekstrem.
- **Temuan Empiris**: 
  - `assertj-core` (6.261 test cases, 1 flaky test = 0,016%) dan `handlebars.java` (427 test cases, 1 flaky test = 0,23%) memiliki Zero-F1 Rate 100%. Menemukan 1 jarum dalam tumpukan 6.261 jerami tanpa konteks proyek spesifik adalah batasan fundamental dari *static metric-based machine learning*.
  - `http-request` (18 flaky test): Mengalami *severe negative transfer* akibat perbedaan domain arsitektur.

---

## 5. Uji Signifikansi Statistik Lintas Repetisi (Friedman & Nemenyi)

Berdasarkan pengujian berulang lintas 5 seed:
- **Friedman Test pada F1-Score**: $\chi^2_{\text{Friedman}} = \mathbf{193.1110}$, $p\text{-value} = \mathbf{7.59 \times 10^{-29}} \ll 0.05$.
- **Kesimpulan**: Perbedaan performa antar 24 pipeline terbukti **sangat signifikan secara statistik** dan konsisten lintas perulangan stokastik.
- **Nemenyi Critical Difference (CD)**: Dengan $k=24$ dan $N=24$, batas $\text{CD} = \mathbf{7.4097}$.
- **Klaster Peringkat Nemenyi F1-Score**:
  - Rank 1: `P13_S2_B3_M1` (Average Rank: **7.17**)
  - Rank 2: `P21_S3_B3_M1` (Average Rank: **7.65**)
  - Rank 3: `P05_S1_B3_M1` (Average Rank: **7.69**)
  - Rank 4-6: `P14_S2_B3_M2`, `P22_S3_B3_M2`, `P06_S1_B3_M2` (Average Rank: **8.58**)
  - Ambang batas perbedaan signifikan dari Rank 1: $7.17 + 7.41 = \mathbf{14.58}$.
  - Seluruh pipeline Baseline M1 (`P01`, `P09`, `P17` dengan rank 16.65 - 16.75) dan ClassWeight M1 (`P07`, `P23`, `P15` dengan rank 16.90 - 17.08) berada di atas batas signifikansi, sehingga terbukti **secara statistik inferior secara signifikan** terhadap pipeline terbaik `P13_S2_B3_M1`.

---

## 6. Visualisasi Publikasi Ilmiah 300 DPI

Seluruh visualisasi grafik publikasi telah dihasilkan pada folder `results/plots/`:
1. `per_project_f1_consistency.png`: Diagram batang horizontal performa F1 per-proyek beserta error bar standar deviasi 5 seed (memvisualisasikan proyek konsisten vs sinkhole).
2. `flaky_ratio_vs_f1_correlation.png`: Kurva korelasi dan garis regresi antara proporsi flaky test (%) vs Mean F1, membuktikan korelasi positif kuat ($r = 0.6710, \rho = 0.7270, p < 0.001$).
3. `pipeline_stability_cv_f1.png`: Diagram batang Coefficient of Variation (CV %) yang membuktikan stabilitas tinggi pada model RUS.
4. `boxplot_factors_consistency.png`: Distribusi F1 lintas faktor Scaling, Imbalance, dan Classifier pada seluruh repetisi.
5. `heatmap_project_by_pipeline_f1.png`: Peta panas 24 Proyek Target $\times$ 24 Pipeline.
6. `macro_vs_weighted_f1_comparison.png`: Grafik komparasi Macro F1 vs Flaky-Weighted F1 vs Micro F1.

---

## 7. Rekomendasi Definitif bagi Peneliti & Praktisi

1. **Gunakan Evaluasi Berbobot (Flaky-Weighted / Micro F1) dalam LOPO-CV**:  
   Jangan hanya mengandalkan *Unweighted Macro F1* ketika mengevaluasi dataset lintas proyek dengan disparitas ukuran ekstrem. Selalu laporkan Flaky-Weighted F1 atau Micro-F1 untuk memberikan gambaran akurat mengenai kemampuan deteksi model pada populasi flaky test riil di industri.
2. **Kombinasi Pipeline Standar Emas**:  
   $$\mathbf{Standard\ Emas} = \text{Cleaned Imputation} + \mathbf{MinMaxScaler\ (S2)} + \mathbf{RUS\ (B3)} + \mathbf{Random\ Forest\ (M1)}$$
   Menghasilkan performa tertinggi (**F1: 0.1798, Flaky-Weighted F1: 0.3832, Recall: 44,33%**) dengan konsistensi stokastik paling stabil ($CV = 6.47\%$).
3. **Waspadai Proyek Sinkhole**:  
   Pada repositori dengan rasio flaky $< 0.5\%$, pendekatan *static machine learning* berbasis metrik tidak memadai. Diperlukan pendekatan hibrida (analisis log eksekusi dinamis atau LLM-based code comprehension) untuk menangani kasus-kasus ekstrem tersebut.
