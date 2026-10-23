# 📑 Rancangan Penelitian Empiris: Eksperimen 3 One-on-One & Repeated LOPO-CV under Afeltra Paper Concept
## Investigasi Evaluasi Pairwise Transfer, Konsistensi Stokastik, dan Transfer Learning pada Dataset IDoFT

**Peneliti**: Bima Jati Kusuma (Kelompok A01 — Magister Teknik Informatika ITS)  
**Mata Kuliah**: Rekayasa Perangkat Lunak Lanjut (RPL A)  
**Institusi**: Institut Teknologi Sepuluh Nopember (ITS), Surabaya  
**Rujukan Primer**:
1. **Afeltra et al. (IEEE Access 2024)** — *A Large-Scale Empirical Investigation Into Cross-Project Flaky Test Prediction*.
2. **Alshammari et al. (ICSE 2021)** — *FlakeFlagger: Predicting Flakiness Without Rerunning Tests*.
**Standar Pelaporan**: *ACM/SIGSOFT Empirical Software Engineering Standards (Version 2.0)*

---

## 1. Latar Belakang & Landasan Teoretis

Dalam pengujian perangkat lunak berbasis *Continuous Integration / Continuous Deployment* (CI/CD), pengujian non-deterministik (*flaky tests*) memicu ketidakpastian tinggi, pemborosan waktu triase pengembang, dan degradasi keandalan sistem. 

Pada penelitian seminal **FlakeFlagger** (*Alshammari et al., ICSE 2021*), model pembelajaran mesin dilatih dan diuji menggunakan paradigma *within-project* pada 24 proyek Java open-source, menghasilkan skor $F_1 \approx 0.66$ dan Recall $\approx 74\%$. Namun, paradigma ini mensyaratkan ketersediaan data historis *rerun* ribuan kali pada proyek yang sama, sehingga tidak dapat diterapkan pada proyek baru (*cold-start problem*).

Untuk mengatasi keterbatasan tersebut, **Afeltra et al. (IEEE Access 2024)** melakukan investigasi skala besar pada skenario lintas proyek (*Cross-Project Flaky Test Prediction*) menggunakan **Dataset IDoFT (Illinois Dataset of Flaky Tests)**. Temuan kunci Afeltra et al. menunjukkan:
1. **Kegagalan Model Baseline Cross-Project**: Pelatihan model standar pada repositori eksternal menghasilkan performa sangat buruk ($F_1 \approx 0.03$, Precision $< 0.10$) akibat fenomena *domain shift*, perbedaan konvensi penulisan uji (*distribution mismatch*), dan ketimpangan kelas minoritas yang ekstrem.
2. **Superioritas Random Under-Sampling (RUS)**: Di antara teknik penyeimbang data, RUS terbukti secara definitif merupakan teknik terbaik dalam skenario cross-project, mengungguli SMOTE dan Baseline.
3. **Peningkatan Masif melalui TrAdaBoost**: Dengan menerapkan algoritma transfer learning *TrAdaBoost* menggunakan data adaptasi dari proyek target (25%, 50%, 75%), nilai $F_1$-score melonjak hingga $0.70$ (+2.300%).

Eksperimen ini dirancang untuk mereplikasi dan memperluas penelitian di atas dengan menerapkan **perlakuan yang sama persis dengan konsep paper Afeltra et al. (2024)** pada **Dataset IDoFT**, sekaligus menguji dimensi baru: **One-on-One Pairwise Cross-Project Transfer Matrix**.

---

## 2. Research Questions (RQ)

Replikasi ini merumuskan empat (4) Pertanyaan Penelitian utama:

### 🎯 RQ1: Karakteristik & Dinamika Transfer Prediktif One-on-One (Pairwise) Lintas Repositori IDoFT
> *Bagaimana kemampuan generalisasi prediktif model ketika dilatih secara murni pada satu proyek sumber tunggal ($P_{src}$) dan diuji pada satu proyek target tunggal ($P_{tgt}$) untuk seluruh pasangan proyek IDoFT ($13 \times 12 = 156$ kombinasi pasangan)? Proyek mana yang bertindak sebagai donor terbaik (Top Transfer Donors) dan target mana yang paling responsif terhadap transfer ilmu pengetahuan?*

### 🎯 RQ2: Efektivitas Intervensi Transfer Learning & Instance-based Filtering Konsep Afeltra
> *Sejauh mana intervensi konsep Afeltra et al. (2024) — Baseline, Burak Filter ($k=10$), TrAdaBoost (25%, 50%, 75%), dan Within-Project Upper Bound — mampu mengatasi domain shift pada dataset IDoFT? Apakah TrAdaBoost mampu mendongkrak F1 dan Recall secara konsisten pada setiap repositori individual?*

### 🎯 RQ3: Konsistensi Stokastik & Stabilitas Kinerja Lintas Repetisi Random Seeds
> *Seberapa stabil metrik evaluasi ($F_1$-Score, Recall, Precision, PR-AUC) ke-6 perlakuan Afeltra ketika diuji berulang kali (5 repetisi random seeds: 42, 100, 2024, 777, 999) pada 17 proyek IDoFT? Berapa nilai variansi ($\sigma$), koefisien variasi ($CV\%$), dan margin interval kepercayaan ($CI_{95}$)?*

### 🎯 RQ4: Dekonstruksi Matematis Rendahnya F1 Makro & Taksonomi Kemampuan Dataset
> *Faktor mendasar apa yang menyebabkan nilai F1 makro lintas proyek tampak rendah pada dataset IDoFT? Sejauh mana perbandingan 4 skema agregasi (Unweighted Macro 17, Active 13, Flaky-Weighted F1, dan Global Micro F1) membuktikan dampak kelangkaan ekstrem kelas minoritas? Apakah terdapat korelasi linier signifikan antara rasio flaky test per proyek terhadap akurasi deteksi?*

---

## 3. Desain Matriks Eksperimen & Protokol Validasi

### A. Dataset Ground Truth IDoFT
- **Sumber Data**: Illinois Dataset of Flaky Tests (Java projects from `pr-data.csv` and `gr-data.csv`).
- **Penyelarasan Fitur**: 23 fitur prediktif resmi FlakeFlagger (8 Test Smells, 7 Execution/Coverage Metrics, 8 Code Churn).
- **Dimensi**: 12.626 baris pengujian, 17 proyek Java, 264 flaky tests riil (2,09% flakiness rate), 0 missing values.
- **Demografi Proyek**: 13 Proyek Aktif Flaky (`achilles`, `activiti`, `alluxio`, `hbase`, `http-request`, `httpcore`, `java-websocket`, `logback`, `okhttp`, `spring-boot`, `undertow`, `wildfly`, `wro4j`) dan 4 Proyek Zero-Flaky Target (`elastic-job-lite`, `handlebars.java`, `jimfs`, `orbit`).

### B. Skema Validasi Ganda (Dual Validation Protocol):
1. **Skema 1: One-on-One Pairwise Cross-Project Transfer**:
   - $P_{src} \in \{13 \text{ Proyek Aktif}\}$ melatih model.
   - $P_{tgt} \in \{13 \text{ Proyek Aktif}, P_{tgt} \neq P_{src}\}$ menguji model.
   - Total $13 \times 12 = \mathbf{156 \text{ Pasangan Proyek}}$.
   - Intervensi: `Baseline_1on1`, `Burak_Filter_1on1`, dan `TrAdaBoost_50_1on1`.
2. **Skema 2: Repeated Leave-One-Project-Out Cross-Validation (LOPO-CV)**:
   - 5 Repetisi Independen: $\text{Seeds} = [42, 100, 2024, 777, 999]$.
   - 17 Lipatan Proyek per repetisi ($N-1 = 16$ proyek train, 1 proyek test).
   - 6 Intervensi Perlakuan Konsep Afeltra.
   - 2 Algoritma Classifier: Random Forest (M1 CPU) dan XGBoost (M2 CUDA GPU).
   - Total Evaluasi: $5 \text{ Seeds} \times 2 \text{ Model} \times 17 \text{ Folds} \times 6 \text{ Treatments} = \mathbf{1.020 \text{ Evaluasi Intervensi}}$.

---

## 4. Rincian 6 Perlakuan Konsep Paper Afeltra et al. (2024)

| Kode Perlakuan | Nama Perlakuan | Formulasi Teknis & Parameter | Rasional Ilmiah |
|---|---|---|---|
| **T1** | **Baseline Cross-Project** | Penskalaan MinMaxScaler (S2) + Random Under-Sampling (B3) + Random Forest / XGBoost tanpa adaptasi. | Mengukur batas bawah performa cross-project konvensional tanpa transfer learning. |
| **T2** | **Burak Filter** | Instance-based filtering berbasis $k$-NN ($k=10$) pada manifold jarak Euclidean terstandarisasi. | Memilih subset data sumber yang paling mirip dengan karakteristik manifold proyek target. |
| **T3** | **TrAdaBoost_25** | Transfer learning boosting dengan 25% data proyek target sebagai adaptasi terbobot, 75% untuk pengujian. | Menguji ketahanan transfer boosting pada kondisi ketersediaan data target minimal (*few-shot*). |
| **T4** | **TrAdaBoost_50** | Transfer learning boosting dengan 50% data target untuk adaptasi terbobot, 50% untuk pengujian. | Mengevaluasi dinamika transfer boosting pada rasio data seimbang antara adaptasi dan pengujian. |
| **T5** | **TrAdaBoost_75** | Transfer learning boosting dengan 75% data target untuk adaptasi terbobot, 25% untuk pengujian. | Mereplikasi konfigurasi terbaik paper Afeltra et al. (2024) yang mencatatkan $F_1 = 0.70$. |
| **T6** | **Within-Project** | 5-Fold Stratified Cross-Validation pada data proyek target yang bersangkutan secara terisolasi. | Menetapkan batas atas ideal (*theoretical upper bound*) jika proyek memiliki data historis sendiri. |

---

## 5. Metrik Evaluasi & Uji Signifikansi Statistik

1. **Metrik Klasifikasi Positif**:
   - $F_1$-Score, Recall (Sensitivity), Precision, Matthews Correlation Coefficient (MCC).
   - Precision-Recall AUC (PR-AUC) dan ROC-AUC.
2. **Ukuran Konsistensi Stokastik**:
   - Mean ($\mu$), Standar Deviasi ($\sigma$), Koefisien Variasi ($CV\% = \frac{\sigma}{\mu} \times 100\%$), dan Margin $CI_{95}$.
3. **Uji Signifikansi Statistik Non-Parametrik**:
   - **Wilcoxon Signed-Rank Test**: Uji signifikansi beda berpasangan ($\alpha = 0.05$).
   - **Cliff's Delta ($d$)**: Ukuran efek non-parametrik (*Negligible*, *Small*, *Medium*, *Large*).
   - **Friedman Test & Nemenyi Post-Hoc**: Uji peringkat omnibus lintas seluruh perlakuan model.
