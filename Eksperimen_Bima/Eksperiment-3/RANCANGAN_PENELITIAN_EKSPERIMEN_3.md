# 🔬 Rancangan Penelitian: Eksperimen 3 — Repeated LOPO-CV, Konsistensi Nilai Pengujian, & Analisis Kemampuan Dataset

**Kelompok Peneliti**: Kelompok A01 — Rekayasa Perangkat Lunak A (Magister Teknik Informatika ITS)  
**Dokumen Disimpan Pada**: `Eksperimen_Bima/Eksperiment-3/RANCANGAN_PENELITIAN_EKSPERIMEN_3.md`  
**Dataset Utama**: `Experiment_Ali/data/cleaned_test_features.csv` (Group-Aware Per-Project Median Imputation, 22.236 baris, 24 proyek Java, 811 flaky test)  
**Hubungan dengan Eksperimen 2**: Replikasi penuh, validasi stabilitas stokastik (*repeated runs*), serta investigasi mendalam terhadap variansi performa dan profil per-dataset.  
**Standar Metodologi**: ACM/SIGSOFT Empirical Standards for Software Engineering Research.

---

## 📑 Daftar Isi
1. [Ringkasan Eksekutif & Rasional Eksperimen 3](#1-ringkasan-eksekutif--rasional-eksperimen-3)
2. [Matriks Evolusi Rangkaian Penelitian (Eksperimen 1 $\rightarrow$ 2 $\rightarrow$ 3)](#2-matriks-evolusi-rangkaian-penelitian-eksperimen-1--2--3)
3. [Desain Faktorial & Repeated LOPO-CV (2.880 Evaluasi Model)](#3-desain-faktorial--repeated-lopo-cv-2880-evaluasi-model)
4. [Rumusan Masalah & 3 Research Questions (RQ)](#4-rumusan-masalah--3-research-questions-rq)
5. [Kerangka Analisis Konsistensi & Stabilitas Empiris](#5-kerangka-analisis-konsistensi--stabilitas-empiris)
6. [Kerangka Profiling Kemampuan Dataset & Dekonstruksi Akurasi Rendah](#6-kerangka-profiling-kemampuan-dataset--dekonstruksi-akurasi-rendah)
7. [Taksonomi 4 Tier Konsistensi Dataset](#7-taksonomi-4-tier-konsistensi-dataset)
8. [Protokol Eksekusi & Validasi Bebas Kebocoran Data (Zero Data Leakage)](#8-protokol-eksekusi--validasi-bebas-kebocoran-data-zero-data-leakage)

---

## 1. Ringkasan Eksekutif & Rasional Eksperimen 3

Pada [Eksperimen 2](../Eksperiment-2/), Kelompok A01 berhasil membuktikan bahwa dataset terimputasi bersih (*Group-Aware Median Imputation*) melestarikan 100% data (24 proyek dan 811 flaky test) dan mengungguli pembuangan data C1 (+11,4% F1-Score). Eksperimen 2 juga mengonfirmasi dominasi mutlak strategi *Random Under-Sampling (B3: RUS)* yang mendongkrak Recall hingga 43,03% dan F1-Score hingga 0.1816.

Namun demikian, terdapat dua celah metodologis dan analitis fundamental yang perlu diselesaikan:
1. **Validasi Stabilitas Stokastik (Seed Sensitivity)**:  
   Eksperimen 2 dieksekusi dengan satu *seed* acak tetap (`random_state=42`). Dalam algoritma pembelajaran mesin modern, komponen-komponen seperti *resampling* acak (Random Under-Sampling), interpolasi tetangga terdekat (SMOTE), pembagian fitur per-pohon (Random Forest), dan *subsampling* histogram (XGBoost) memiliki sifat non-deterministik. Untuk memenuhi standar empiris ACM/SIGSOFT, diperlukan pengujian berulang (*repeated experimental trials*) lintas beberapa *random seeds* guna mengukur apakah keunggulan konfigurasi terbaik konsisten atau rentan terhadap variasi inisialisasi acak.
2. **Misteri Rendahnya Nilai Rata-rata Akurasi & F1-Score Lintas Proyek**:  
   Meskipun pipeline terbaik mencapai F1-Score 0.1816, angka ini secara absolut terkesan rendah jika dibandingkan dengan skenario prediksi *within-project* konvensional. Mengapa rata-rata F1 lintas proyek hanya berkisar antara 0.05 – 0.18? Apakah model gagal mengenali pola, ataukah nilai rendah tersebut merupakan konsekuensi matematis dari agregasi *Unweighted Macro-Average* pada dataset yang memiliki *extreme near-zero minority class* (seperti `assertj-core` dengan hanya 1 flaky dari 6.261 test case, atau `jimfs` dengan 0 flaky)?
3. **Pemetaan Kemampuan Dataset (Dataset Profiling)**:  
   Peneliti dan praktisi memerlukan kejelasan: proyek mana saja yang berhasil diprediksi secara konsisten dan akurat, dan proyek mana saja yang menjadi *sinkhole* (selalu menghasilkan deteksi nol).

Eksperimen 3 dirancang secara spesifik untuk menjawab ketiga tantangan tersebut secara tuntas.

---

## 2. Matriks Evolusi Rangkaian Penelitian (Eksperimen 1 $\rightarrow$ 2 $\rightarrow$ 3)

| Parameter Komparasi | Eksperimen 1 | Eksperimen 2 | Eksperimen 3 (Rancangan Ini) |
|---|---|---|---|
| **Kondisi Dataset** | Dataset Mentah (273 NaN) | Dataset Terimputasi Penuh (0 NaN) | Dataset Terimputasi Penuh (0 NaN) |
| **Strategi Missing Values** | C1 (Row Drop) vs C2 (Project Drop) | Imputasi Median Per-Proyek | Imputasi Median Per-Proyek |
| **Cakupan Proyek & Baris** | C1: 24 Proyek, C2: 19 Proyek | 24 Proyek, 22.236 Baris (100% Utuh) | 24 Proyek, 22.236 Baris (100% Utuh) |
| **Populasi Flaky Test** | C1: 808 flaky, C2: 586 flaky | 811 flaky (100% Utuh) | 811 flaky (100% Utuh) |
| **Jumlah Pipeline** | 48 Pipeline Faktorial | 24 Pipeline Faktorial | 24 Pipeline Faktorial |
| **Protokol Validasi** | Single-Run LOPO-CV (Seed 42) | Single-Run LOPO-CV (Seed 42) | **Repeated LOPO-CV (5 Independent Seeds)** |
| **Total Model Folds** | 1.032 Model Folds | 576 Model Folds | **2.880 Model Folds ($5 \times 24 \times 24$)** |
| **Fokus Investigasi** | Dampak Pembuangan Data NaN | Pembuktian Imputasi & Matriks 24 Pipeline | **Konsistensi Pengujian, Analisis Per-Dataset, & Root Cause Akurasi Rendah** |

---

## 3. Desain Faktorial & Repeated LOPO-CV (2.880 Evaluasi Model)

Eksperimen 3 mengevaluasi matriks faktorial penuh $3 \times 4 \times 2 = \mathbf{24\text{ Pipelines}}$ yang diulang sebanyak $\mathbf{5\text{ Repetisi}}$ dengan seed independen:
$$\text{Seeds} = [42, 100, 2024, 777, 999]$$

```
Total Evaluasi = 5 Seeds × 24 Pipelines × 24 Proyek Target = 2.880 Model Folds
```

| Faktor Eksperimen | Notasi & Variasi | Protokol Operasional |
|---|---|---|
| **Feature Scaling (3)** | **S1**: StandardScaler<br>**S2**: MinMaxScaler<br>**S3**: RobustScaler | Fit HANYA pada data latih (23 proyek sumber) dan transform ke target proyek uji (Zero Leakage). |
| **Imbalance Handling (4)** | **B1**: Baseline (None)<br>**B2**: SMOTE<br>**B3**: Random Under-Sampling (RUS)<br>**B4**: ClassWeight (Cost-Sensitive) | Diterapkan HANYA pada data latih dengan seed dinamis sesuai repetisi iterasi aktif. |
| **Classifier Model (2)** | **M1**: Random Forest CPU (`n_estimators=100`, `n_jobs=-1`)<br>**M2**: XGBoost GPU (`tree_method='hist'`, `device='cuda'`, `max_depth=6`) | Dilatih dengan seed dinamis untuk menjamin variabilitas stokastik model. |

---

## 4. Rumusan Masalah & 3 Research Questions (RQ)

### 🔹 RQ1: Konsistensi & Stabilitas Kinerja Lintas Perulangan
*Bagaimanakah tingkat konsistensi dan stabilitas metrik F1-Score, Recall, Precision, dan PR-AUC dari ke-24 pipeline ketika diuji berulang kali pada 5 inisialisasi random seed yang berbeda? Apakah Random Under-Sampling (B3) secara konsisten mempertahankan peringkat teratasnya dengan variansi yang terkendali?*

### 🔹 RQ2: Dekonstruksi Matematis & Empiris Rendahnya Nilai Rata-rata Akurasi Makro
*Mengapa nilai rata-rata F1-Score makro lintas proyek berada pada rentang yang tampak rendah (~0.05 – 0.18)? Sejauh mana fenomena near-zero minority class pada proyek-proyek tertentu mendistorsi Unweighted Macro Average dibandingkan Flaky-Weighted F1 dan Global Micro F1?*

### 🔹 RQ3: Taksonomi Kemampuan & Kestabilan 24 Dataset Proyek Target
*Bagaimanakah profil kemampuan deteksi pada masing-masing 24 proyek target? Dataset mana saja yang terbukti memiliki nilai deteksi yang konsisten tinggi, dataset mana yang sangat sensitif terhadap metode penyeimbangan data, dan dataset mana yang secara konsisten menjadi sinkhole non-deteksi?*

---

## 5. Kerangka Analisis Konsistensi & Stabilitas Empiris

Untuk setiap pipeline $P_i$ ($i = 1, \dots, 24$), metrik evaluasi diukur lintas $R = 5$ pengulangan:
1. **Rata-rata Lintas Repetisi ($\mu_i$)**:
   $$\mu_i = \frac{1}{R} \sum_{r=1}^{R} M_{i, r}$$
2. **Standar Deviasi Lintas Repetisi ($\sigma_i$)**:
   $$\sigma_i = \sqrt{\frac{1}{R-1} \sum_{r=1}^{R} (M_{i, r} - \mu_i)^2}$$
3. **Coefficient of Variation ($CV_i$)**:
   $$CV_i = \left( \frac{\sigma_i}{\mu_i} \right) \times 100\%$$
   *Interpretasi*: Nilai $CV < 10\%$ menunjukkan stabilitas sangat tinggi (kestabilan deterministik), sedangkan $CV > 30\%$ menandakan sensitivitas tinggi terhadap inisialisasi stokastik.
4. **95% Confidence Interval ($CI_{95}$)**:
   $$CI_{95} = \mu_i \pm 1.96 \times \frac{\sigma_i}{\sqrt{R}}$$

---

## 6. Kerangka Profiling Kemampuan Dataset & Dekonstruksi Akurasi Rendah

### 6.1 Karakteristik Heterogenitas 24 Proyek
Ke-24 proyek Java FlakeFlagger memiliki disparitas skala dan distribusi flaky test yang ekstrem:
- **Skala Baris Data**: Berkisar dari 55 test cases (`commons-exec`) hingga 6.261 test cases (`assertj-core`).
- **Populasi Flaky Test**: Berkisar dari 0 flaky (`jimfs`), 1 flaky (`assertj-core`, `handlebars.java`, `commons-exec`, `ninja`), hingga 163 flaky (`spring-boot`) dan 145 flaky (`hbase`).
- **Rasio Ketimpangan Kelas**: Dari 0,016% flaky (`assertj-core`) hingga 62,03% flaky (`alluxio`).

### 6.2 Mengapa Macro Average Menekan Nilai Rata-rata?
Dalam Leave-One-Project-Out Cross-Validation:
- **Unweighted Macro-Average**: Menghitung rata-rata dari ke-24 proyek dengan bobot yang sama persis ($w_j = \frac{1}{24}$):
  $$\text{Macro F1} = \frac{1}{24} \sum_{j=1}^{24} \text{F1}_j$$
  Jika sebuah proyek hanya memiliki 1 flaky test dan model menghasilkan 1 false positive, Precision jatuh menjadi 0 atau sangat kecil, dan jika flaky tersebut terlewat (False Negative = 1), Recall = 0, sehingga F1 = 0. Terdapat 8 proyek dengan $\le 4$ flaky test yang kerap menghasilkan $\text{F1} = 0$. Kegagalan di 8 proyek ini langsung memangkas rata-rata makro sebesar $\frac{8}{24} \approx 33,3\%$ menuju angka nol!
- **Flaky-Weighted Average**: Memberikan bobot proporsional terhadap populasi flaky test riil di setiap proyek:
  $$\text{Weighted F1} = \sum_{j=1}^{24} \left( \frac{N_{\text{flaky}, j}}{N_{\text{flaky, total}}} \right) \text{F1}_j$$
- **Global Micro-Average**: Menggabungkan seluruh Confusion Matrix (TP, FP, FN, TN) dari ke-24 lipatan menjadi satu matriks global:
  $$\text{Micro F1} = \frac{2 \sum \text{TP}}{2 \sum \text{TP} + \sum \text{FP} + \sum \text{FN}}$$

---

## 7. Taksonomi 4 Tier Konsistensi Dataset

Berdasarkan performa dan stabilitas deteksi lintas 5 repetisi dan 24 pipeline, ke-24 proyek diklasifikasikan ke dalam 4 Tier:

| Tier | Kategori | Kriteria Penentuan | Contoh Proyek Representatif |
|:---:|---|---|---|
| **Tier 1** | **Rock-Solid Consistency & High Detectability** | Mean F1 $\ge 0.30$, Zero-F1 Rate $\le 15\%$, CV lintas seed rendah ($<10\%$). | `ambari`, `hbase`, `alluxio` |
| **Tier 2** | **Resampling-Dependent Consistency** | F1 pada model RUS $\ge 0.35$ secara stabil lintas seed, namun kolaps pada baseline B1/B4. | `spring-boot`, `httpcore` |
| **Tier 3** | **Moderate & Fragile Detection** | Mean F1 $0.05 - 0.25$, sensitif terhadap domain shift dan pergeseran ambang sampling. | `undertow`, `orbit`, `wildfly`, `wro4j`, `activiti`, `incubator-dubbo`, `hector` |
| **Tier 4** | **Permanent Non-Detection Sinkholes** | Zero-F1 Rate $\ge 80\%$, proporsi flaky $< 0.6\%$ atau pergeseran domain ekstrem. | `assertj-core`, `handlebars.java`, `elastic-job-lite`, `zxing`, `http-request`, `jimfs`, `achilles` |

---

## 8. Protokol Eksekusi & Validasi Bebas Kebocoran Data (Zero Data Leakage)

Untuk memastikan keabsahan ilmiah:
1. **Pemisahan Partisi Ketat**: Dalam setiap fold $j$, data uji $D_{\text{test}} = \text{Proyek } j$ diisolasi secara total.
2. **Penskalaan Zero Leakage**: Objek Scaler (S1, S2, S3) di-`fit` HANYA pada 23 proyek data latih, kemudian di-`transform` ke data uji.
3. **Resampling Terisolasi**: Resampling (SMOTE atau RUS) diterapkan HANYA pada data latih. Data uji target dibiarkan pada distribusi alaminya.
4. **Reproduksibilitas**: Seluruh eksekusi dicatat ke dalam log CSV individual per pipeline dan per seed di `results/raw_predictions/`.
