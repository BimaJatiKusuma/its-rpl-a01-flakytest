# 📑 Rancangan Penelitian: Replikasi Eksperimen 3 pada Dataset IDoFT (Afeltra et al., 2024 Context)
## Repeated LOPO-CV, Analisis Stabilitas Stokastik, dan Dekonstruksi Kemampuan Dataset

**Peneliti**: Bima Jati Kusuma (Kelompok A01 — Magister Teknik Informatika ITS)  
**Mata Kuliah**: Rekayasa Perangkat Lunak Lanjut (RPL A)  
**Konteks Empiris**: Replikasi dan Perluasan Konsep Paper *Afeltra et al. (IEEE Access 2024)* & *Alshammari et al. (ICSE 2021)*  
**Standar Pelaporan**: *ACM/SIGSOFT Empirical Software Engineering Standards (Version 2.0)*

---

## 1. Latar Belakang & Landasan Teoretis

Dalam rekayasa perangkat lunak modern, pengujian perangkat lunak non-deterministik (*flaky tests*) menimbulkan kerugian signifikan pada siklus *Continuous Integration / Continuous Deployment* (CI/CD). Flaky test bergantian menghasilkan status *PASS* dan *FAIL* pada komit kode sumber yang identik tanpa modifikasi apapun.

Pada penelitian seminal **FlakeFlagger** (*Alshammari et al., ICSE 2021*), model pembelajaran mesin dilatih dan diuji menggunakan evaluasi *within-project* atau *pooled cross-validation* pada 24 proyek Java sumber terbuka, menghasilkan nilai F1-Score yang relatif tinggi (~0.66). Namun, investigasi skala besar yang dilakukan oleh **Afeltra et al. (IEEE Access 2024)** (*A Large-Scale Empirical Investigation Into Cross-Project Flaky Test Prediction*) membuktikan bahwa ketika model dievaluasi secara lintas proyek (*Cross-Project* / *Leave-One-Project-Out Cross-Validation* / LOPO-CV), performa model mengalami penurunan drastis hingga mendekati nol ($F_1 \approx 0.03$) akibat fenomena *domain shift*, perbedaan distribusi ruang fitur (*distribution mismatch*), dan ketimpangan kelas minoritas yang sangat tajam (*extreme class imbalance*).

Untuk mengatasi keterbatasan dataset FlakeFlagger yang dihasilkan dari eksekusi ulang 10.000 kali pada subset proyek tertentu, Afeltra et al. mengadopsi **IDoFT (Illinois Dataset of Flaky Tests)**, repositori kurasi global yang memuat ribuan pengujian flaky nyata dari komunitas pengembang Java open-source.

Pada [Eksperimen 3 Eksperimen_Bima](../../Eksperimen_Bima/Eksperiment-3/), telah dirancang metodologi pengujian berulang (*Repeated LOPO-CV*) dengan 5 *independent random seeds* melintasi 24 kombinasi pipeline ML untuk menguji stabilitas stokastik algoritma dan mendekonstruksi akar penyebab rendahnya akurasi makro lintas proyek. Replikasi ini bertujuan menerapkan metodologi tersebut secara penuh pada **Dataset IDoFT**, sehingga validitas temuan empiris dapat diverifikasi silang antara dataset FlakeFlagger dan dataset IDoFT.

---

## 2. Research Questions (RQ)

Replikasi ini merumuskan 3 pertanyaan riset utama:

### 🎯 RQ1: Konsistensi & Stabilitas Kinerja Pipeline Lintas Perulangan Stokastik pada Dataset IDoFT
> *Seberapa konsisten dan stabil metrik evaluasi (F1-Score, PR-AUC, Recall) ke-24 pipeline ketika diuji berulang kali (5 repetisi independent random seeds: 42, 100, 2024, 777, 999) pada dataset IDoFT? Apakah teknik Random Under-Sampling (RUS) tetap mempertahankan dominasi performanya secara stabil, ataukah variansi sampling menyebabkan fluktuasi kinerja yang rentan?*

### 🎯 RQ2: Dekonstruksi Matematis Rendahnya Nilai Rata-rata Akurasi & F1-Score
> *Faktor mendasar apa yang menyebabkan nilai rata-rata F1-Score makro pada skenario cross-project IDoFT bernilai rendah? Sejauh mana disparitas antara Unweighted Macro-Average F1, Active-Only Macro F1, Flaky-Weighted Average F1, dan Global Micro-Average F1 membuktikan pengaruh ketimpangan kelas ekstrem dan proyek non-deteksi permanen (sinkholes)?*

### 🎯 RQ3: Karakterisasi & Taksonomi Kemampuan 17 Dataset Proyek IDoFT
> *Berdasarkan profil kemampuan deteksi model prediktif, bagaimana taksonomi ke-17 proyek IDoFT dapat dikelompokkan ke dalam tier empiris (Tier 1: High Detectability, Tier 2: Resampling-Dependent, Tier 3: Fragile / Moderate, dan Tier 4: Permanent Non-Detection Sinkholes / Zero Minority Class)? Apakah terdapat korelasi statistik signifikan antara rasio flaky test dalam proyek terhadap nilai F1-Score?*

---

## 3. Desain Matriks Eksperimen (Factorial Design)

Eksperimen menerapkan desain faktorial penuh $3 \times 4 \times 2 = 24$ kombinasi pipeline:

```
               ┌───────────────────────────────┐
               │    24 PIPELINE KOMBINASI      │
               └───────────────┬───────────────┘
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
3 Feature Scaling       4 Imbalance Resampling    2 ML Classifiers
 (S1: StandardScaler)    (B1: Baseline / None)    (M1: Random Forest CPU)
 (S2: MinMaxScaler)      (B2: SMOTE Over-sample)  (M2: XGBoost CUDA GPU)
 (S3: RobustScaler)      (B3: Random Under-Sample)
                         (B4: ClassWeight Balanced)
```

### Parameter Eksperimen & Skema Validasi:
| Dimensi | Parameter Konfigurasi |
|---|---|
| **Dataset Dasar** | IDoFT Java Benchmark terimputasi bersih (12.626 test cases, 17 proyek, 264 flaky tests, flakiness rate 2.09%). |
| **Fitur Prediktif** | 23 fitur prediktif resmi FlakeFlagger (8 Test Smells, 7 Execution/Coverage Metrics, 8 Code Churn/H-Index). |
| **Protokol Validasi** | Repeated Leave-One-Project-Out Cross-Validation (Repeated LOPO-CV). |
| **Repetisi Stokastik** | 5 Repetisi Independen: Seeds = $[42, 100, 2024, 777, 999]$. |
| **Lipatan per Repetisi** | 17 Proyek Folds (16 Proyek Train, 1 Proyek Test). |
| **Total Model Folds** | $5 \text{ Seeds} \times 24 \text{ Pipelines} \times 17 \text{ Folds} = \mathbf{2.040 \text{ Model Folds}}$. |
| **Pencegahan Data Leakage**| Fit scaler dan resampler dilakukan STRICTLY HANYA pada data latih (Zero Data Leakage). |
| **Akselerasi Hardware**| NVIDIA GeForce RTX 4050 Laptop GPU (CUDA hist method) & CPU multi-core. |

---

## 4. Variabel Penelitian

### A. Variabel Bebas (Independent Variables):
1. **Faktor Penskalaan Fitur (*Feature Scaling*)**:
   - $S_1$: *StandardScaler* ($z = \frac{x - \mu}{\sigma}$)
   - $S_2$: *MinMaxScaler* ($x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$)
   - $S_3$: *RobustScaler* (berbasis median dan Interquartile Range $IQR = Q_3 - Q_1$, tahan pencilan ekstrem).
2. **Faktor Penanganan Ketimpangan Kelas (*Class Imbalance Handling*)**:
   - $B_1$: *Baseline* (tanpa intervensi sampling).
   - $B_2$: *SMOTE (Synthetic Minority Over-sampling Technique)* ($k=5$, atau $k = \max(1, N_{pos}-1)$ jika $N_{pos} \le 5$).
   - $B_3$: *RUS (Random Under-Sampling)* (menyeimbangkan kelas mayoritas dengan minoritas secara acak).
   - $B_4$: *Cost-Sensitive / ClassWeight* (pemberian bobot penalti loss invers proporsional terhadap frekuensi kelas).
3. **Faktor Algoritma Klasifikasi (*Classifier Algorithm*)**:
   - $M_1$: *Random Forest Classifier* (100 *trees*, *bootstrap sampling*, CPU multi-threading `n_jobs=-1`).
   - $M_2$: *Extreme Gradient Boosting (XGBoost)* (100 *trees*, `max_depth=6`, `learning_rate=0.1`, GPU CUDA `tree_method='hist'`).

### B. Variabel Terikat (Dependent Variables):
1. **F1-Score**: Rata-rata harmonik Precision dan Recall pada kelas positif (flaky).
2. **Precision-Recall AUC (PR-AUC)**: Area di bawah kurva Precision-Recall (metrik paling representatif pada data *imbalanced* ekstrem).
3. **Recall (Sensitivity)**: Proporsi uji flaky riil yang berhasil ditangkap oleh model prediktif.
4. **Precision**: Proporsi alarm uji flaky yang benar-benar merupakan uji flaky (mengukur efisiensi developer).
5. **Matthews Correlation Coefficient (MCC)**: Koefisien korelasi biner seimbang antara prediksi dan observasi riil.
6. **Balanced Accuracy**: Rata-rata aritmatika dari Sensitivity (Recall) dan Specificity.

### C. Ukuran Konsistensi & Stabilitas Empiris:
1. **Mean ($\mu$)**: Nilai rata-rata metrik lintas 5 perulangan seed independen.
2. **Standar Deviasi ($\sigma$)**: Ukuran dispersi performa di sekitar nilai rata-rata.
3. **Coefficient of Variation ($CV$)**: Ukuran variabilitas relatif ternormalisasi:
   $$CV = \frac{\sigma}{\mu} \times 100\%$$
   Kriteria: $CV < 5\%$ (Sangat Konsisten / Stabil), $5\% \le CV \le 15\%$ (Moderat), $CV > 15\%$ (Labil / Fluktuatif).
4. **95% Confidence Interval Margin ($CI_{95}$)**:
   $$CI_{95} = 1.96 \times \frac{\sigma}{\sqrt{R}} \quad (R = 5)$$

---

## 5. Protokol Uji Signifikansi Statistik (Empirical Standards)

Mengikuti panduan ACM/SIGSOFT Empirical Standards untuk evaluasi algoritma pembelajaran mesin:
1. **Friedman Non-Parametric Test**: Menguji hipotesis nol ($H_0$) bahwa seluruh 24 konfigurasi pipeline memiliki performa ekuivalen pada metrik F1-Score dan PR-AUC lintas 17 proyek target.
2. **Nemenyi Post-Hoc Test & Critical Difference (CD)**: Menghitung ambang batas perbedaan peringkat rata-rata minimum agar perbedaan performa antara dua pipeline dinyatakan signifikan secara statistik ($\alpha = 0.05$):
   $$CD = q_\alpha \sqrt{\frac{k(k + 1)}{6n}}$$
   di mana $k = 24$ (jumlah perlakuan pipeline), $n = 17$ (jumlah proyek uji IDoFT), dan $q_{0.05} \approx 3.630$.

---

## 6. Prosedur Penanganan Kasus Khusus (Edge Cases)

Pada dataset IDoFT, terdapat 4 proyek target yang memiliki 0 flaky test terverifikasi (`elastic-job-lite`, `handlebars.java`, `jimfs`, `orbit`). Proyek-proyek ini bertindak sebagai *negative controls* krusial:
- Model dievaluasi untuk mengukur *False Positive Rate (FPR)* dan *Specificity (True Negative Rate)*.
- Metrik yang membutuhkan kelas minoritas positif ($TP + FN > 0$) seperti PR-AUC, ROC-AUC, dan Recall diatur ke `NaN`, sedangkan F1 bernilai `0.0` dan Precision bernilai `1.0` (jika $FP = 0$) atau `0.0` (jika $FP > 0$).
- Pada agregasi makro, analisis menyajikan perbandingan transparan antara:
  1. *Unweighted Macro F1 (17 Proyek)*: Menyertakan seluruh 17 proyek.
  2. *Active-Only Macro F1 (13 Proyek)*: Mengeksklusi 4 proyek zero-flaky.
  3. *Flaky-Weighted F1*: Pembobotan berdasarkan jumlah flaky test riil per proyek.
  4. *Global Instance-Level Micro F1*: Evaluasi berbasis total matriks konfusi global ($TP, FP, TN, FN$).
