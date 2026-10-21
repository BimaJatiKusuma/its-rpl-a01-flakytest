# 🔬 Eksperimen 3: Repeated LOPO-CV, Konsistensi Nilai Pengujian, & Analisis Kemampuan Dataset

Repositori ini memuat perancangan, implementasi teknis, eksekusi empiris, dan analisis komprehensif dari **Eksperimen 3**, yang mereplikasi dan memperluas temuan dari [Eksperimen 2](../Eksperiment-2/).

---

## 📋 Ikhtisar & Latar Belakang Eksperimen 3

Pada [Eksperimen 2](../Eksperiment-2/), 24 konfigurasi pipeline dievaluasi menggunakan *Leave-One-Project-Out Cross-Validation (LOPO-CV)* pada dataset terimputasi bersih (`Experiment_Ali/data/cleaned_test_features.csv`, 22.236 baris, 24 proyek, 811 flaky test) dengan satu *random seed* tetap (`random_state=42`).

Meskipun Eksperimen 2 membuktikan dominasi *Random Under-Sampling (RUS)* dan keunggulan *MinMaxScaler (S2)*, muncul pertanyaan krusial dalam metodologi rekayasa perangkat lunak empiris (ACM/SIGSOFT standards):
1. **Konsistensi & Stabilitas Stokastik**: Apakah keunggulan pipeline (seperti `P13_S2_B3_M1` dan `P14_S2_B3_M2`) bersifat konsisten dan stabil ketika diuji berulang kali (*repeated runs*) menggunakan inisialisasi *random seed* yang bervariasi? Seberapa besar variansi yang dihasilkan oleh algoritma stokastik seperti *Random Forest bootstrap*, *SMOTE interpolation*, dan *Random Under-Sampling*?
2. **Dekonstruksi Akurasi Rendah**: Mengapa nilai rata-rata F1-Score dan akurasi model terkesan sangat rendah (~0.18 pada pipeline terbaik dan ~0.08 secara agregat)? Faktor apa yang mendasarinya pada level karakteristik dataset individual?
3. **Peta Kemampuan & Konsistensi Dataset**: Di antara ke-24 repositori open-source Java yang diuji, dataset mana saja yang memberikan performa deteksi flaky test yang tinggi dan konsisten, dan dataset mana saja yang selalu gagal dideteksi (*sinkhole projects*)?

---

## ⚙️ Desain Eksperimen: Repeated LOPO-CV (2.880 Evaluasi Model)

Eksperimen 3 melakukan pengujian berulang sebanyak **5 Repetisi Independen** dengan variasi *random seeds*:
$$\text{Seeds} = [42, 100, 2024, 777, 999]$$

| Dimensi Eksperimen | Parameter & Konfigurasi |
|---|---|
| **Dataset Dasar** | `Experiment_Ali/data/cleaned_test_features.csv` (Group-Aware Median Imputed, 22.236 baris, 24 proyek Java, 811 flaky test). |
| **Feature Scaling (3)** | **S1**: StandardScaler \| **S2**: MinMaxScaler \| **S3**: RobustScaler (Fit HANYA pada train). |
| **Imbalance Handling (4)** | **B1**: Baseline (None) \| **B2**: SMOTE \| **B3**: RUS (Random Under-Sampling) \| **B4**: ClassWeight. |
| **Classifier Models (2)** | **M1**: Random Forest CPU \| **M2**: XGBoost GPU CUDA (hist method). |
| **Skema Validasi** | **Repeated LOPO-CV**: 5 Repetisi $\times$ 24 Pipelines $\times$ 24 Folds = **2.880 Model Folds**. |
| **Metrik Evaluasi** | PR-AUC, ROC-AUC, F1-Score, Recall, Precision, MCC, Balanced Accuracy, Specificity, Confusion Matrix. |
| **Ukuran Konsistensi** | Mean, Standard Deviation ($\sigma$), Coefficient of Variation ($CV = \frac{\sigma}{\mu} \times 100\%$), 95% Confidence Interval ($CI_{95}$). |

---

## 🎯 3 Research Questions (RQ) Utama Eksperimen 3

1. **RQ1 (Konsistensi & Kestabilan Hasil Pengujian Lintas Perulangan)**:  
   *Seberapa konsisten dan stabil metrik performa (F1-Score, PR-AUC, Recall) ke-24 pipeline ketika diuji berulang kali dengan inisialisasi random seed yang berbeda? Apakah Random Under-Sampling (B3) mempertahankan peringkat teratasnya secara konsisten tanpa terdistorsi oleh variansi sampling?*
2. **RQ2 (Akar Penyebab Rendahnya Nilai Rata-rata Akurasi & F1-Score)**:  
   *Apa penyebab utama rendahnya nilai rata-rata F1-Score makro lintas proyek? Apakah disebabkan oleh kegagalan representasi model, pergeseran domain (domain shift), atau ketimpangan kelas ekstrem (extreme near-zero minority class) pada dataset-dataset kecil? Bagaimana komparasi antara Unweighted Macro F1, Flaky-Weighted F1, dan Global Micro F1?*
3. **RQ3 (Karakterisasi & Taksonomi Konsistensi 24 Dataset Proyek Target)**:  
   *Berdasarkan profil kemampuan deteksi dan stabilitas lintas perulangan, kelompok proyek mana yang memiliki nilai deteksi konsisten tinggi, kelompok mana yang sangat bergantung pada resampling, dan kelompok mana yang menjadi sinkhole non-deteksi permanen?*

---

## 📊 Matriks Kemampuan 24 Pipeline pada 24 Dataset (Proyek Target)

Tabel berikut menyajikan pemetaan komprehensif performa **F1-Score** ke-24 pipeline (baris) pada masing-masing 24 dataset proyek Java target (kolom) lintas 5 *independent random seeds* (total 2.880 model folds). 

Baris paling akhir (**`RATA-RATA KEMAMPUAN DATASET`**) menunjukkan performa rata-rata seluruh pipeline pada masing-masing dataset, yang mengungkap langsung mengapa rata-rata makro keseluruhan bernilai rendah (akibat 7 proyek sinkhole dengan kemampuan deteksi near-zero) dan dataset mana saja yang memiliki performa konsisten tinggi.

### Tabel Master Kemampuan Pipeline vs Dataset (LOPO-CV):

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

> 📌 **Catatan**: Rincian sub-tabel modular (Bagian A: Proyek 1-12 dan Bagian B: Proyek 13-24) serta analisis mendalam tersedia di dokumen [LAPORAN_EKSPERIMEN_3.md](LAPORAN_EKSPERIMEN_3.md).

---

## 🗂️ Struktur Direktori Eksperimen 3

```
Eksperimen_Bima/Eksperiment-3/
├── README.md                                    # Dokumentasi ikhtisar & petunjuk eksekusi
├── RANCANGAN_PENELITIAN_EKSPERIMEN_3.md         # Dokumen akademik metodologi penelitian lengkap
├── LAPORAN_EKSPERIMEN_3.md                     # Laporan komprehensif temuan empiris & analisa dataset
├── configs/
│   └── experiment_matrix.json                   # Konfigurasi parameter 24 pipelines & 5 random seeds
├── src/
│   ├── __init__.py
│   ├── check_gpu.py                             # Verifikasi GPU RTX 4050 & library ML
│   ├── data_loader.py                           # Loader dataset cleaned_test_features.csv
│   ├── pipeline_builder.py                      # Pembangun pipeline dinamis berbasis random_state
│   ├── repeated_lopo_runner.py                  # Runner 5 repetisi LOPO-CV (2.880 evaluasi model)
│   ├── consistency_analyzer.py                  # Analisis variansi, CV %, dan interval kepercayaan
│   ├── dataset_profiler.py                      # Profiling 24 dataset & root cause analysis akurasi
│   ├── statistical_tests.py                     # Uji Friedman & Nemenyi post-hoc lintas repetisi
│   └── visualizer.py                            # Generator grafik publikasi resolusi tinggi (300 DPI)
├── notebooks/
│   └── 01_run_experiment_3_repeated_lopo.ipynb # Notebook interaktif eksekusi eksperimen
└── results/
    ├── raw_predictions/                         # Log prediksi fold per pipeline & per seed (CSV)
    ├── per_seed_summaries/                      # Ringkasan metrik per repetisi seed
    ├── all_repeated_folds_predictions.csv       # Master CSV 2.880 evaluasi fold
    ├── summary_metrics_repeated_24_pipelines.csv# Rata-rata & std lintas repetisi per pipeline
    ├── consistency_stability_analysis.csv       # Tabel stabilitas & CV % ke-24 pipeline
    ├── dataset_capability_profile.csv           # Profil mendalam ke-24 proyek target
    ├── matrix_pipeline_vs_dataset_f1.csv        # Matriks 24 Pipeline x 24 Dataset F1 & rata-rata
    ├── macro_vs_weighted_f1_explanation.csv     # Komparasi Macro vs Weighted vs Micro F1
    ├── friedman_nemenyi_f1_repeated_results.csv # Hasil uji signifikansi statistik F1
    └── plots/                                   # 6 visualisasi ilmiah publikasi 300 DPI
```

---

## ⚡ Petunjuk Menjalankan Eksperimen

1. **Aktivasi Virtual Environment**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Verifikasi GPU & Pustaka ML**:
   ```powershell
   & ".\.venv\Scripts\python.exe" Eksperimen_Bima/Eksperiment-3/src/check_gpu.py
   ```

3. **Eksekusi Pengujian Berulang (Repeated LOPO-CV)**:
   ```powershell
   & ".\.venv\Scripts\python.exe" Eksperimen_Bima/Eksperiment-3/src/repeated_lopo_runner.py
   ```

4. **Eksekusi Analisis Konsistensi & Profiling Dataset**:
   ```powershell
   & ".\.venv\Scripts\python.exe" Eksperimen_Bima/Eksperiment-3/src/consistency_analyzer.py
   & ".\.venv\Scripts\python.exe" Eksperimen_Bima/Eksperiment-3/src/dataset_profiler.py
   & ".\.venv\Scripts\python.exe" Eksperimen_Bima/Eksperiment-3/src/statistical_tests.py
   & ".\.venv\Scripts\python.exe" Eksperimen_Bima/Eksperiment-3/src/visualizer.py
   ```
