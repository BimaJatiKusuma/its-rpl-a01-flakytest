# 🔬 Rancangan Penelitian: Multi-Dataset Repeated LOPO-CV dengan TrAdaBoost Target Adaptation

**Kelompok Peneliti**: Kelompok A01 — Rekayasa Perangkat Lunak A (Magister Teknik Informatika ITS)  
**Dokumen Disimpan Pada**: [`Eksperimen_Kelompok/bima-eksperimen-3-all-dataset-tradaboost/RANCANGAN_PENELITIAN_ALL_DATASET_TRADABOOST.md`](file:///e:/Magister%20ITS/SEMESTER%20I/Rekayasa%20Perangkat%20Lunak%20A/Bell%20Labs%20hehe/Eksperimen_Kelompok/bima-eksperimen-3-all-dataset-tradaboost/RANCANGAN_PENELITIAN_ALL_DATASET_TRADABOOST.md)  
**Dataset Diuji**:
1. **FlakeFlagger Cleaned Benchmark** (*Alshammari et al., ICSE 2021*): 24 proyek Java, 22.236 test cases, 811 flaky tests (3,65% flakiness rate).
2. **Illinois Dataset of Flaky Tests (IDoFT) Cleaned Benchmark** (*Afeltra et al., IEEE Access 2024*): 17 proyek Java, 12.626 test cases, 264 flaky tests (2,09% flakiness rate).

**Hubungan Silsilah Penelitian**:
Replikasi dan perluasan komprehensif dari [`Eksperimen_Bima/Eksperiment-3`](file:///e:/Magister%20ITS/SEMESTER%20I/Rekayasa%20Perangkat%20Lunak%20A/Bell%20Labs%20hehe/Eksperimen_Bima/Eksperiment-3/) dan konsep *instance-based transfer learning* Afeltra et al. (IEEE Access 2024), dengan mengintegrasikan algoritma **TrAdaBoost** (*Dai et al., ICML 2007*) melintasi variasi porsi data berlabel target (Baseline Zero-Shot 0%, TrAdaBoost 25%, 50%, 75%, serta Within-Project Upper Bound) di bawah protokol **Repeated LOPO-CV (5 Independent Random Seeds)**.

---

## 1. Latar Belakang & Rasional Ilmiah

Prediksi tes flaky lintas proyek (*Cross-Project Flaky Test Prediction*) merupakan tantangan besar akibat fenomena *domain shift* dan perbedaan distribusi ruang fitur antar repositori. Pada model konvensional tanpa transfer (*Baseline Zero-Shot*), model yang dilatih pada $N-1$ proyek eksternal menderita lonjakan *false alarms* ekstrem (*Precision* $< 0.10$, *F1-Score* $\approx 0.03 - 0.17$).

Afeltra et al. (IEEE Access 2024) melaporkan bahwa metode pembobotan instans **TrAdaBoost** mampu mendongkrak performa hingga $F_1 \approx 0.70$. Namun, peningkatan spektakuler tersebut dicapai dengan menyuntikkan data berlabel dari repositori target itu sendiri. Muncul pertanyaan penelitian mendasar:
1. *Bagaimana dinamika kenaikan performa model seiring bertambahnya porsi data target (0% $\to$ 25% $\to$ 50% $\to$ 75%)?*
2. *Apakah karakteristik transfer tersebut konsisten ketika diuji pada dua dataset benchmark yang berbeda (FlakeFlagger vs IDoFT)?*
3. *Bagaimana kestabilan stokastiknya ketika dievaluasi berulang melintasi 5 inisialisasi random seed independen?*

---

## 2. Desain Faktorial Eksperimen (1.230 Evaluasi Model)

Eksperimen ini mengevaluasi matriks faktorial penuh lintas dua dataset:
$$\text{Total Evaluasi} = (24\text{ Proyek FF} + 17\text{ Proyek ID}) \times 5\text{ Seeds} \times 6\text{ Treatments} = 1.230\text{ Model Folds}$$

```
Seeds: [42, 100, 2024, 777, 999]
```

| Parameter | Variasi / Perlakuan | Keterangan & Protokol Operasional |
| :--- | :--- | :--- |
| **Dataset (2)** | 1. **FlakeFlagger** (24 Proyek)<br>2. **IDoFT** (17 Proyek) | Kedua dataset telah dibersihkan (Zero Missing Values) dengan 23 fitur FlakeFlagger yang identik. |
| **Porsi Data Target (5)** | **T1: Baseline Zero-Shot (0%)**<br>**T2: TrAdaBoost 25%**<br>**T3: TrAdaBoost 50%**<br>**T4: TrAdaBoost 75%**<br>**T5: Within-Project (100%)** | **0%**: Latih pada $N-1$ proyek sumber, uji pada 100% data target.<br>**25%**: 25% target untuk adaptasi transfer, uji pada 75% sisa target.<br>**50%**: 50% target untuk adaptasi transfer, uji pada 50% sisa target.<br>**75%**: 75% target untuk adaptasi transfer, uji pada 25% sisa target.<br>**Within**: 5-Fold Stratified CV pada proyek target lokal (Upper Bound). |
| **Model Inti (2)** | **M1: Random Forest** (CPU, 100 trees, balanced)<br>**M2: XGBoost** (GPU CUDA, hist, depth 6) | Dievaluasi pada baseline zero-shot untuk menguji keunggulan arsitektur pohon. |
| **Weak Learner TrAdaBoost** | `DecisionTreeClassifier(max_depth=4)` | 20 estimator boosting berbobot logaritmik (Dai et al., 2007). |
| **Preprocessing & Zero Leakage** | `MinMaxScaler` + `RandomUnderSampler (RUS)` | Scaler dan Sampler di-fit **HANYA** pada data training (Zero Data Leakage). |

### Matriks Kombinasi Perlakuan Pipeline & Posisi TrAdaBoost

| ID | Nama Perlakuan | Porsi Data Target (Sampling) | Perlakuan Imbalance (Resampling) | Posisi / Status TrAdaBoost | Model Dasar (Estimator) | Set Pengujian (Test Set) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **T1** | **Baseline Zero-Shot (RF)** | **0%** Target Train<br>(LOPO Murni) | **RUS** pada $N-1$ Proyek Sumber | **Tidak Aktif**<br>(Model Konvensional Non-Transfer) | **Random Forest**<br>(100 trees, `class_weight='balanced'`) | 100% Data Proyek Target |
| **T1b** | **Baseline Zero-Shot (XGB)** | **0%** Target Train<br>(LOPO Murni) | **RUS** pada $N-1$ Proyek Sumber | **Tidak Aktif**<br>(Model Konvensional Non-Transfer) | **XGBoost GPU**<br>(100 trees, `max_depth=6`, `hist`) | 100% Data Proyek Target |
| **T2** | **TrAdaBoost 25%** | **25%** Target Train<br>(Stratified Split) | **RUS** pada $N-1$ Proyek Sumber | **Aktif** (Meta-Learner Pembobot Instans, 20 iterasi) | **Decision Tree**<br>(`max_depth=4`, weak learner) | 75% Sisa Data Proyek Target |
| **T3** | **TrAdaBoost 50%** | **50%** Target Train<br>(Stratified Split) | **RUS** pada $N-1$ Proyek Sumber | **Aktif** (Meta-Learner Pembobot Instans, 20 iterasi) | **Decision Tree**<br>(`max_depth=4`, weak learner) | 50% Sisa Data Proyek Target |
| **T4** | **TrAdaBoost 75%** | **75%** Target Train<br>(Stratified Split) | **RUS** pada $N-1$ Proyek Sumber | **Aktif** (Meta-Learner Pembobot Instans, 20 iterasi) | **Decision Tree**<br>(`max_depth=4`, weak learner) | 25% Sisa Data Proyek Target |
| **T5** | **Within-Project (Upper Bound)** | **100%** Data Target Lokal<br>(Batas Atas Teoritis) | **None**<br>(Distribusi Alami Terstratifikasi) | **Tidak Aktif**<br>(Klasifikasi Terisolasi Lokal) | **Random Forest**<br>(5-Fold Stratified CV, 100 trees) | 20% Data Proyek Target per Fold |

---

## 3. Rumusan 5 Research Questions (RQ)

- **RQ1: Kurva Kinerja TrAdaBoost vs Porsi Data Target**: Sejauh mana F1-Score, Recall, Precision, dan PR-AUC meningkat dari 0% ke 75% porsi data target pada kedua dataset?
- **RQ2: Konsistensi Stokastik & Variabilitas Seeds**: Apakah variabilitas model ($CV\%$) teredam secara deterministik seiring bertambahnya supervisi target?
- **RQ3: Dekonstruksi 4 Skema Agregasi Matematis**: Bagaimana perbandingan antara *Unweighted Macro F1*, *Active Macro F1*, *Flaky-Weighted F1*, dan *Global Micro F1*?
- **RQ4: Uji Signifikansi Statistik & Effect Size**: Apakah lompatan performa TrAdaBoost terbukti signifikan secara statistik (Wilcoxon $p < 0.05$, Cliff's delta $d$, dan Friedman omnibus rank)?
- **RQ5: Taksonomi & Profil Kapabilitas Tiap Proyek**: Proyek mana saja yang responsif terhadap supervisi target (*Tier 1 & Tier 2*) dan mana yang menjadi *Scarcity Sinkhole* (*Tier 4*)?

---

## 4. Struktur Direktori & Artefak

Direktori kerja terletak pada: [`Eksperimen_Kelompok/bima-eksperimen-3-all-dataset-tradaboost/`](file:///e:/Magister%20ITS/SEMESTER%20I/Rekayasa%20Perangkat%20Lunak%20A/Bell%20Labs%20hehe/Eksperimen_Kelompok/bima-eksperimen-3-all-dataset-tradaboost/):
- **`configs/`**: [`configs/experiment_matrix.json`](file:///e:/Magister%20ITS/SEMESTER%20I/Rekayasa%20Perangkat%20Lunak%20A/Bell%20Labs%20hehe/Eksperimen_Kelompok/bima-eksperimen-3-all-dataset-tradaboost/configs/experiment_matrix.json)
- **`data/`**: [`data/dataset_metadata.json`](file:///e:/Magister%20ITS/SEMESTER%20I/Rekayasa%20Perangkat%20Lunak%20A/Bell%20Labs%20hehe/Eksperimen_Kelompok/bima-eksperimen-3-all-dataset-tradaboost/data/dataset_metadata.json), dataset FlakeFlagger & IDoFT.
- **`src/`**: Modul Python lengkap ([`data_loader.py`](file:///e:/Magister%20ITS/SEMESTER%20I/Rekayasa%20Perangkat%20Lunak%20A/Bell%20Labs%20hehe/Eksperimen_Kelompok/bima-eksperimen-3-all-dataset-tradaboost/src/data_loader.py), [`tradaboost.py`](file:///e:/Magister%20ITS/SEMESTER%20I/Rekayasa%20Perangkat%20Lunak%20A/Bell%20Labs%20hehe/Eksperimen_Kelompok/bima-eksperimen-3-all-dataset-tradaboost/src/tradaboost.py), [`dual_dataset_tradaboost_runner.py`](file:///e:/Magister%20ITS/SEMESTER%20I/Rekayasa%20Perangkat%20Lunak%20A/Bell%20Labs%20hehe/Eksperimen_Kelompok/bima-eksperimen-3-all-dataset-tradaboost/src/dual_dataset_tradaboost_runner.py), [`consistency_analyzer.py`](file:///e:/Magister%20ITS/SEMESTER%20I/Rekayasa%20Perangkat%20Lunak%20A/Bell%20Labs%20hehe/Eksperimen_Kelompok/bima-eksperimen-3-all-dataset-tradaboost/src/consistency_analyzer.py), [`dataset_profiler.py`](file:///e:/Magister%20ITS/SEMESTER%20I/Rekayasa%20Perangkat%20Lunak%20A/Bell%20Labs%20hehe/Eksperimen_Kelompok/bima-eksperimen-3-all-dataset-tradaboost/src/dataset_profiler.py), [`statistical_tests.py`](file:///e:/Magister%20ITS/SEMESTER%20I/Rekayasa%20Perangkat%20Lunak%20A/Bell%20Labs%20hehe/Eksperimen_Kelompok/bima-eksperimen-3-all-dataset-tradaboost/src/statistical_tests.py), [`visualizer.py`](file:///e:/Magister%20ITS/SEMESTER%20I/Rekayasa%20Perangkat%20Lunak%20A/Bell%20Labs%20hehe/Eksperimen_Kelompok/bima-eksperimen-3-all-dataset-tradaboost/src/visualizer.py), [`main_executor.py`](file:///e:/Magister%20ITS/SEMESTER%20I/Rekayasa%20Perangkat%20Lunak%20A/Bell%20Labs%20hehe/Eksperimen_Kelompok/bima-eksperimen-3-all-dataset-tradaboost/src/main_executor.py)).
- **`results/`**: Data mentah 1.230 baris, ringkasan CSV, dan 6 plot 300 DPI.
- **`notebooks/`**: [`notebooks/01_run_all_dataset_tradaboost_experiment.ipynb`](file:///e:/Magister%20ITS/SEMESTER%20I/Rekayasa%20Perangkat%20Lunak%20A/Bell%20Labs%20hehe/Eksperimen_Kelompok/bima-eksperimen-3-all-dataset-tradaboost/notebooks/01_run_all_dataset_tradaboost_experiment.ipynb).
