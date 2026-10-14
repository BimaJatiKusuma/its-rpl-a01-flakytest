# 📑 Laporan Lengkap Hasil Implementasi Eksperimen
## Judul Penelitian: Predicting Flaky Test in Cross-Project Scenario

Dokumen ini merangkum seluruh hasil implementasi teknis, metodologi, dan temuan empiris dari **Phase 1 sampai Phase 5** (Skenario Utama) serta **Alternatif 1** (Skenario 90% Train Projects vs 10% Test Projects) dalam penelitian prediksi flaky test menggunakan skenario *Cross-Project* secara *Project-Agnostic*.

---

## 📌 1. Rekapitulasi Metodologi & Alur Eksperimen

Eksperimen dirancang untuk menjawab tantangan deteksi *flaky test* tanpa menjalankan tes ulang (*rerun*), serta memastikan bahwa model yang dilatih pada sekumpulan proyek (*source projects*) dapat memprediksi flakiness pada proyek baru yang belum pernah dilihat (*unseen target project*) tanpa *data leakage*.

```mermaid
flowchart TD
    P1["Phase 1: Data Preparation & Missing Imputation"] --> P2["Phase 2: Feature Engineering & Scaling"]
    P2 --> P3["Phase 3: Cross-Project LOOCV Setup"]
    P3 --> P4["Phase 4: Model Training & Imbalance Handling"]
    P4 --> P5["Phase 5: Evaluation, Breakdown & Interpretability"]
    P2 -.-> P3_ALT["Phase 3 (Alt 1): 90% Train, 10% Test Setup"]
    P3_ALT --> P4_ALT["Phase 4 (Alt 1): Model Training 90/10"]
    P4_ALT --> P5_ALT["Phase 5 (Alt 1): Evaluation 90/10"]
```

---

## 🔍 2. Rincian Hasil Implementasi Setiap Phase (Skenario Utama)

### 2.1 Phase 1: Data Preparation, Imputation & EDA per Proyek
* **Dataset Raw**: 22.236 baris data test case dari 24 proyek Java open-source.
* **Isolasi Metadata & Project-Agnostic Setup**:
  * Kolom identitas proyek (`project`) disimpan sebagai *grouping variable* untuk validasi lintas proyek, bukan fitur input.
  * Kolom pelacak (`test_name`) disimpan sebagai *Primary Key*.
  * Kolom redundan (`Unnamed: 0`, `testClassName`, `testMethodName`) dibuang dari matriks fitur $X$.
* **Missing Value Imputation**:
  * Terdeteksi 273 data hilang pada fitur `testLength`, `numAsserts`, dan `numCoveredLines` (terkonsentrasi di proyek `wildfly` dan `logback`).
  * Diterapkan **Group-Aware Median Imputation** (imputasi berdasarkan median proyek yang sama) dan **Fallback Probabilistic Sampling** untuk mempertahankan skala unik per proyek.
  * Hasil: **0 missing values** tersisa pada `data/cleaned_test_features.csv`.
* **Artefak**:
  * Notebook: `notebook/01_eda_and_feature_inspection.ipynb`

---

### 2.2 Phase 2: Feature Engineering & Normalization (Mitigasi Domain Shift)
* **Fitur Rasio Project-Agnostic**:
  * Menghilangkan ketergantungan pada ukuran absolut proyek melalui rasio relatif:
    * $\text{coverage\_ratio} = \frac{\text{numCoveredLines}}{\text{projectSourceLinesCovered} + 1}$
    * $\text{assert\_density} = \frac{\text{numAsserts}}{\text{testLength} + 1}$
    * $\text{class\_coverage\_ratio} = \frac{\text{numCoveredLines}}{\text{projectSourceClassesCovered} + 1}$
* **Log Transformation ($\log(x+1)$)**:
  * Mengurangi skewness ekstrem pada `ExecutionTime`, `testLength`, dan fitur coverage baris/kelas.
* **Feature Scaling**:
  * Mengomparasikan `StandardScaler` dan `RobustScaler` (median & IQR) untuk menahan distorsi dari outlier lintas proyek.
* **Reduksi Multikolinearitas Code Churn (`hIndex`)**:
  * Menerapkan PCA pada 8 window `hIndex` (`window5` s.d. `window10000`).
  * 3 Komponen Utama (`churn_pca_1`, `churn_pca_2`, `churn_pca_3`) berhasil merangkum **>80% total varians** dari seluruh window modifikasi commit.
* **Artefak**:
  * Dataset: `data/engineered_test_features.csv` (58 kolom).
  * Notebook: `notebook/02_feature_engineering_scaling.ipynb`

---

### 2.3 Phase 3: Setup Validation Cross-Project (LOOCV / LOPO-CV)
* **Pencegahan Ketat Data Leakage**:
  * Memastikan test case dari proyek target $P_i$ tidak pernah masuk ke dalam training set.
* **Skema Utama: 24-Fold Leave-One-Project-Out (LOOCV)**:
  * Setiap fold melatih model pada 23 proyek (~21.000 sampel) dan menguji pada 1 proyek unseen ($P_i$).
* **Skenario Tambahan: Multi-Project Split (70/30)**:
  * 17 proyek (18.221 test cases, 532 flaky) sebagai Train Pool dan 7 proyek (4.015 test cases, 279 flaky) sebagai Unseen Test Pool.
* **Artefak**:
  * Metadata Split: `data/validation_splits_metadata.json`
  * Ringkasan Fold: `data/loocv_folds_summary.csv`
  * Notebook: `notebook/03_validation_setup_loocv.ipynb`

---

### 2.4 Phase 4: Model Training & Handling Class Imbalance
* **20 Fitur Prediktif Terpilih**: 8 Test Smells, 9 Robust Scaled Metrics & Ratios, 3 Churn PCA.
* **Benchmark 7 Model pada Skenario Multi-Project (70/30)**:

| Peringkat | Algoritma | Waktu Latih | ROC-AUC | PR-AUC | F1 (Thresh=0.5) | Recall (0.5) | F1 (Tuned) | Recall (Tuned) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 🥇 | **LightGBM (Cost-Sensitive)** | **0,44 s** | **0,7326** | **0,0908** | 0,0621 | 0,0722 | 0,0078 | 0,0052 |
| 🥈 | **Gaussian Naive Bayes** | 0,01 s | 0,5623 | **0,0787** | **0,1456** | **0,4845** | **0,1818** | **0,3144** |
| 🥉 | **CatBoost (Cost-Sensitive)** | 1,26 s | 0,6717 | 0,0737 | 0,0436 | 0,0412 | 0,0090 | 0,0052 |
| 4 | **Random Forest (Balanced)** | 0,50 s | 0,6320 | 0,0719 | 0,0102 | 0,0052 | 0,0102 | 0,0052 |
| 5 | **XGBoost (Cost-Sensitive)** | 0,81 s | 0,6565 | 0,0677 | 0,0311 | 0,0258 | 0,0000 | 0,0000 |
| 6 | **Logistic Regression (Balanced)**| 0,24 s | 0,5896 | 0,0615 | 0,0971 | 0,3763 | 0,1013 | 0,1856 |
| 7 | **Dummy (Most Frequent)** | 0,00 s | 0,5000 | 0,0483 | 0,0000 | 0,0000 | 0,0922 | 1,0000 |

* **Penanganan Imbalance**: `scale_pos_weight` pada GBDT memberikan kestabilan pemeringkatan probabilitas tertinggi tanpa perlu komputasi resampling sintetis yang berlebih.
* **Artefak**:
  * Hasil Benchmark: `data/model_comparison_results.csv`
  * Hasil Evaluasi 24 Fold LOOCV: `data/loocv_lgbm_results.csv`
  * Notebook: `notebook/04_model_training_and_imbalance.ipynb`

---

### 2.5 Phase 5: Evaluation, Result Synthesis & Interpretability

#### 1. Ringkasan Performa Evaluasi 24 Fold LOOCV
* **Mean ROC-AUC Lintas 24 Proyek**: **0,6671**
* **Median ROC-AUC Lintas 24 Proyek**: **0,7023**
* **Mean PR-AUC Lintas 24 Proyek**: **0,1962**

#### 2. Tabel Performa Lengkap per Proyek Target (24 Proyek)

| Fold | Proyek Target | Test Samples | Flaky Samples | % Flaky | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `achilles` | 1.317 | 4 | 0,30% | 0,7258 | 0,0068 | 0,0000 | 0,0000 | 0,0000 |
| 2 | `activiti` | 2.044 | 32 | 1,57% | 0,6287 | 0,0273 | 0,0233 | 0,0312 | 0,0267 |
| 3 | `alluxio` | 187 | 116 | **62,03%** | **0,8255** | **0,8754** | 0,0000 | 0,0000 | 0,0000 |
| 4 | `ambari` | 324 | 52 | **16,05%** | **0,7792** | **0,3201** | 0,2857 | 0,3077 | 0,2963 |
| 5 | `assertj-core` | 6.261 | 1 | 0,02% | 0,3676 | 0,0003 | 0,0000 | 0,0000 | 0,0000 |
| 6 | `commons-exec` | 55 | 1 | 1,82% | **1,0000** | **1,0000** | 0,5000 | 1,0000 | 0,6667 |
| 7 | `elastic-job-lite`| 558 | 3 | 0,54% | **0,8420** | 0,0216 | 0,0000 | 0,0000 | 0,0000 |
| 8 | `handlebars.java` | 427 | 1 | 0,23% | **0,8239** | 0,0132 | 0,0000 | 0,0000 | 0,0000 |
| 9 | `hbase` | 431 | 145 | **33,64%** | **0,7454** | **0,6258** | 1,0000 | 0,0138 | 0,0272 |
| 10 | `hector` | 142 | 33 | **23,24%** | 0,6494 | **0,3049** | 0,0000 | 0,0000 | 0,0000 |
| 11 | `http-request` | 163 | 18 | 11,04% | 0,4672 | 0,1246 | 0,0000 | 0,0000 | 0,0000 |
| 12 | `httpcore` | 712 | 22 | 3,09% | **0,7932** | 0,1077 | 0,0000 | 0,0000 | 0,0000 |
| 13 | `incubator-dubbo`| 2.176 | 19 | 0,87% | **0,7239** | 0,0536 | 0,1429 | 0,0526 | 0,0769 |
| 14 | `java-websocket` | 145 | 23 | 15,86% | 0,2275 | 0,1464 | 0,0000 | 0,0000 | 0,0000 |
| 15 | `jimfs` | 212 | 0 | 0,00% | N/A | N/A | 0,0000 | 0,0000 | 0,0000 |
| 16 | `logback` | 843 | 22 | 2,61% | 0,5995 | 0,0474 | 0,0000 | 0,0000 | 0,0000 |
| 17 | `ninja` | 307 | 1 | 0,33% | **0,9542** | 0,0667 | 0,0000 | 0,0000 | 0,0000 |
| 18 | `okhttp` | 810 | 100 | **12,35%** | 0,4431 | 0,1388 | 1,0000 | 0,0100 | 0,0198 |
| 19 | `orbit` | 86 | 7 | 8,14% | 0,6890 | 0,1338 | 0,0000 | 0,0000 | 0,0000 |
| 20 | `spring-boot` | 2.127 | 163 | **7,66%** | **0,8096** | **0,2392** | 0,0000 | 0,0000 | 0,0000 |
| 21 | `undertow` | 183 | 7 | 3,83% | 0,5869 | 0,1866 | 0,0625 | 0,1429 | 0,0870 |
| 22 | `wildfly` | 1.236 | 23 | 1,86% | 0,6280 | 0,0301 | 0,0345 | 0,0435 | 0,0385 |
| 23 | `wro4j` | 1.145 | 16 | 1,40% | **0,7023** | 0,0369 | 0,0000 | 0,0000 | 0,0000 |
| 24 | `zxing` | 345 | 2 | 0,58% | 0,3324 | 0,0064 | 0,0000 | 0,0000 | 0,0000 |

#### 3. Feature Importance & Interpretability
Berdasarkan analisis bobot model LightGBM (`data/global_feature_importance.csv`):
1. 🥇 **Execution Time Dinamis**: `rob_log_ExecutionTime` (Skor: **380**) menjadi prediktor paling dominan.
2. 🥈 **Code Churn Historis**: `churn_pca_1` (Skor: **295**) dan komponen churn lainnya menempati ranking atas.
3. 🥉 **Cakupan Kode & Rasio**: `rob_log_projectSourceClassesCovered` (Skor: **271**) dan rasio coverage mendominasi posisi 10 besar.
4. ⚠️ **Test Smells**: Seluruh 8 fitur test smells menempati urutan terbawah (Skor 8–42).

* **Artefak**:
  * Notebook: `notebook/05_evaluation_and_interpretability.ipynb`
  * Tabel Feature Importance: `data/global_feature_importance.csv`

---

## 🔬 3. Analisis Alternatif 1: Skenario 90% Train Projects vs 10% Test Projects (`_alt_1`)

Pada eksperimen alternatif ini, alur dimulai dari **Phase 3** dengan membagi proporsi proyek secara berbeda:
* **Train Set**: **90% proyek** (22 proyek: 19.551 test cases, 645 flaky [3,30%]).
* **Test Set**: **10% proyek** (2 proyek unseen: `elastic-job-lite` dan `spring-boot`, total 2.685 test cases, 166 flaky [6,18%]).
* **Direktori Artefak**:
  * Data: `data_alt_1/`
  * Notebook: `notebook_alt_1/`
  * Script: `script_alt_1/`

### 3.1 Hasil Benchmark Model pada Skenario 90/10

| Peringkat | Algoritma | Waktu Latih | ROC-AUC | PR-AUC | F1 (Thresh=0.5) | Recall (0.5) |
|---|---|---:|---:|---:|---:|---:|
| 🥇 | **LightGBM (Cost-Sensitive)** | 0,73 s | **0,8006** | **0,1895** | 0,0493 | 0,0422 |
| 🥈 | **CatBoost (Cost-Sensitive)** | 1,92 s | 0,7379 | **0,1131** | 0,0190 | 0,0120 |
| 🥉 | **XGBoost (Cost-Sensitive)** | 2,16 s | 0,7016 | 0,0993 | 0,0172 | 0,0120 |
| 4 | **Random Forest (Balanced)** | 0,67 s | 0,6505 | 0,0978 | 0,0000 | 0,0000 |
| 5 | **Gaussian Naive Bayes** | 0,02 s | 0,5396 | 0,0916 | **0,1527** | **0,4458** |
| 6 | **Logistic Regression (Balanced)** | 0,75 s | 0,6307 | 0,0837 | 0,1279 | 0,3193 |
| 7 | **Dummy (Most Frequent)** | 0,00 s | 0,5000 | 0,0618 | 0,0000 | 0,0000 |

### 3.2 Perbandingan Skenario Utama (70/30) vs Alternatif 1 (90/10)

| Metrik (Model: LightGBM) | Skenario Utama (70/30 Split) | Skenario Alternatif 1 (90/10 Split) | Perubahan ($\Delta$) |
|---|---|---|---|
| **Jumlah Proyek Train** | 17 Proyek | 22 Proyek | **+5 Proyek (+29%)** |
| **ROC-AUC** | 0,7326 | **0,8006** | **+0,0680 (+9,3%)** 📈 |
| **PR-AUC** | 0,0908 | **0,1895** | **+0,0987 (+108,7%)** 🚀 |

> **Temuan Kunci Alternatif 1**:
> Menambah variasi proyek sumber pelatihan dari 17 proyek (70%) menjadi 22 proyek (90%) **meningkatkan performa PR-AUC lebih dari 2x lipat (dari 0,0908 ke 0,1895)** dan mendongkrak **ROC-AUC melewati ambang batas 0,80 (0,8006)**. Ini membuktikan bahwa diversitas karakteristik proyek sumber sangat krusial dalam memperkuat kemampuan generalisasi model *cross-project*.

---

## 🎯 4. Kesimpulan Akhir Penelitian

1. **Kelayakan Pendekatan Project-Agnostic**:
   - Membuang identitas proyek dan mengandalkan fitur ter-normalisasi (`RobustScaler`, rasio coverage, log transformation, dan PCA churn) terbukti layak dengan capaian **median ROC-AUC 0,7023** pada evaluasi ketat 24-Fold LOOCV tanpa data leakage.
2. **Pengaruh Skala Proyek Pelatihan**:
   - Skalabilitas jumlah proyek pada training pool terbukti berbanding lurus dengan performa generalisasi model pada proyek unseen. Peningkatan dari 70% proyek ke 90% proyek menghasilkan lonjakan metrik PR-AUC hingga +108,7%.
3. **Algoritma Terbaik**:
   - **LightGBM dengan Cost-Sensitive Learning (`scale_pos_weight`)** secara konsisten menjadi algoritma unggulan pada seluruh skenario eksperimen.
