# 📑 Rancangan Penelitian Komprehensif: Eksperimen Gabungan 1 & 2
## Cross-Project Flaky Test Prediction (Full Factorial 72 Pipelines & Repeated LOPO-CV 10x)

**Kelompok Peneliti**: Kelompok A01 — Rekayasa Perangkat Lunak A (Magister Teknik Informatika ITS)  
**Dokumen Induk**: `Eksperimen_Bima/Rencana_Penelitian_Flaky_Test.md`  
**Dataset Dasar**: FlakeFlagger Dataset (`test_features.csv` & `cleaned_test_features.csv`, 24 proyek Java, 22.236 baris data)  
**Target Komputasi**: NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM, CUDA 13.4) & CPU Multi-threading  

---

## 1. Latar Belakang & Motivasi Penyatuan Eksperimen 1 & 2

Penelitian prediksi tes flaki lintas proyek (*Cross-Project Flaky Test Prediction*) pada awalnya menghadapi dilema fundamental terkait penanganan nilai hilang (*missing values*) pada 273 baris data FlakeFlagger (Alshammari et al., ICSE 2021):
- **Eksperimen 1** mengeksplorasi dua paradigma pembuangan data:
  - **C1 (Row-Level Drop)**: Membuang 273 baris ber-NaN, mempertahankan 24 proyek (21.963 baris, 808 flaky).
  - **C2 (Project-Level Drop)**: Membuang 5 proyek ber-NaN, mempertahankan 19 proyek (16.458 baris, 586 flaky). Namun, pembuangan proyek besar `spring-boot` memotong ~20% sampel flaky riil industri.
- **Eksperimen 2** mengadopsi teknik prapemrosesan cerdas:
  - **C3 (Group-Aware Per-Project Median Imputation)**: Melestarikan 100% data (24 proyek, 22.236 baris, 811 flaky test) tanpa membuang satu pun sampel.

**Urgensi Eksperimen Gabungan**:  
Agar perbandingan ketiga paradigma data cleaning (**C1 vs C2 vs C3**) bersifat adil (*ceteris paribus*), ketiga strategi tersebut disatukan ke dalam satu rancangan faktorial tunggal yang diuji pada seluruh variasi **Feature Scaling (S1-S3)**, **Class Imbalance Handling (B1-B4)**, dan **Classifier Models (M1-M2)** dengan protokol validasi **Leave-One-Project-Out Cross-Validation (LOPO-CV)** serta pengujian berulang **10x Random Seeds** untuk menguji konsistensi dan stabilitas empiris.

---

## 2. Matriks Desain Faktorial Penuh (72 Pipelines)

Matriks eksperimen dibentuk dari 4 faktor variabel bebas:
$$\text{Total Pipelines} = 3 \text{ Cleaning} \times 3 \text{ Scaling} \times 4 \text{ Imbalance} \times 2 \text{ Models} = \mathbf{72 \text{ Pipelines}}$$

| Faktor | Notasi & Variasi | Deskripsi Operasional & Penjaminan Zero Data Leakage |
|---|---|---|
| **Data Cleaning (3)** | **C1**: Row-Level Drop<br>**C2**: Project-Level Drop<br>**C3**: Group-Aware Median Imputation | **C1**: 24 proyek, 21.963 baris, 808 flaky.<br>**C2**: 19 proyek, 16.458 baris, 586 flaky.<br>**C3**: 24 proyek, 22.236 baris, 811 flaky (100% utuh). |
| **Feature Scaling (3)** | **S1**: StandardScaler<br>**S2**: MinMaxScaler<br>**S3**: RobustScaler | Scaler di-fit **HANYA pada fold latih ($N-1$ proyek)** dan di-transformasikan ke fold target uji unseen. |
| **Imbalance Handling (4)** | **B1**: Baseline (None)<br>**B2**: SMOTE<br>**B3**: Random Under-Sampling (RUS)<br>**B4**: ClassWeight (Cost-Sensitive) | Diterapkan **HANYA pada fold latih**. Data uji dibiarkan pada rasio ketimpangan alaminya. XGBoost B4 menggunakan `scale_pos_weight = neg / pos`. |
| **Classifier Model (2)** | **M1**: Random Forest CPU<br>**M2**: XGBoost GPU CUDA | M1: `n_estimators=100`, CPU parallel (`n_jobs=-1`).<br>M2: `n_estimators=100`, `max_depth=6`, GPU hist (`device="cuda"`, `tree_method="hist"`). |

---

## 3. Protokol Validasi & Framework Uji Konsistensi 10x

### A. Skema Leave-One-Project-Out Cross-Validation (LOPO-CV)
- Pada setiap iterasi, tepat $1$ proyek bertindak sebagai fold uji (*completely unseen target project*), sedangkan $N-1$ proyek lainnya menjadi data latih.
- Untuk C1 dan C3: 24 Folds evaluasi per pipeline.
- Untuk C2: 19 Folds evaluasi per pipeline.
- Total evaluasi single-run: $(24 \times 24) + (24 \times 19) + (24 \times 24) = 576 + 456 + 576 = \mathbf{1.608 \text{ Model Folds}}$.

### B. Framework Repeated Trials (10x Random Seeds)
Untuk menguji variansi stokastik dari proses resampling acak (RUS/SMOTE) dan partisi pohon (RF/XGBoost), eksperimen diulang sebanyak 10 kali menggunakan set random seed standar:
```python
DEFAULT_SEEDS_10 = [42, 100, 2024, 777, 999, 123, 456, 789, 2025, 2026]
```
Total Evaluasi 10x Repetisi: $10 \times 1.608 = \mathbf{16.080 \text{ Model Folds}}$.

---

## 4. Metrik Evaluasi & Analisis Konsistensi

1. **Metrik Performa Operasional**:
   - **F1-Score**: Rerata harmonik presisi dan recall pada kelas minoritas (flaky test).
   - **Recall**: Proporsi flaky test yang berhasil dideteksi dari keseluruhan flaky aktual di proyek target.
   - **Precision**: Proporsi ketepatan prediksi positif flaky.
   - **PR-AUC**: Area di bawah kurva Precision-Recall (metrik ranking probabilitas).
   - **Matthews Correlation Coefficient (MCC)**: Korelasi biner seimbang antara prediksi dan ground truth.
   - **Balanced Accuracy**: Rata-rata sensitivitas dan spesifisitas.
2. **Metrik Konsistensi & Stabilitas**:
   - Mean ($\mu$) dan Standar Deviasi ($\sigma$) lintas 10 repetisi.
   - **Coefficient of Variation (CV %)**: $CV = \frac{\sigma}{\mu} \times 100\%$.
   - **95% Confidence Interval**: $\text{CI}_{95} = \mu \pm 1.96 \frac{\sigma}{\sqrt{10}}$.
   - Rentang Variansi: $\text{Range} = \text{Max} - \text{Min}$.
3. **Uji Signifikansi Statistik (ACM/SIGSOFT Standards)**:
   - **Friedman Test**: Menguji hipotesis nol kesetaraan performa 72 pipeline.
   - **Nemenyi Post-hoc Test**: Mengidentifikasi pasangan pipeline yang berbeda signifikan di atas Critical Difference (CD).
   - **Kruskal-Wallis Test**: Memvalidasi signifikansi perbedaan antara C1, C2, dan C3.

---

## 5. Research Questions (RQ) Eksperimen Gabungan

- **RQ1 (Tri-Paradigma Cleaning)**: Bagaimana dampak relatif dari Imputasi Cerdas (C3) dibandingkan Row-Level Drop (C1) dan Project-Level Drop (C2) terhadap performa deteksi flaky dan pelestarian proyek enterprise?
- **RQ2 (Penskalaan Fitur)**: Manakah penskalaan fitur terbaik (S1, S2, atau S3) yang paling tangguh terhadap domain shift lintas ketiga strategi cleaning?
- **RQ3 (Penanganan Ketimpangan)**: Apakah keunggulan mutlak Random Under-Sampling (B3: RUS) konsisten bertahan di seluruh 3 strategi cleaning dan 10 perulangan seed?
- **RQ4 (Komparasi Model)**: Bagaimana perbandingan efisiensi dan trade-off antara Random Forest (M1) dan XGBoost GPU (M2) lintas 16.080 model folds?
- **RQ5 (Konsistensi Stokastik)**: Seberapa stabil peringkat pipeline terbaik ketika diuji berulang kali lintas 10 random seeds?
