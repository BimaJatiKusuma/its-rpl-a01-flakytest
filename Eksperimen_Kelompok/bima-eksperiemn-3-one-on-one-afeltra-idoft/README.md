# 🔬 Eksperimen 3: One-on-One & Repeated LOPO-CV under Afeltra Paper Concept pada Dataset IDoFT

Repositori ini memuat implementasi teknis lengkap, eksekusi empiris berulang, dan analisis komprehensif dari **Replikasi Eksperimen 3 One-on-One** yang menerapkan perlakuan dan intervensi konsep paper:
> **Afeltra, A. et al. (IEEE Access 2024)** — *A Large-Scale Empirical Investigation Into Cross-Project Flaky Test Prediction*.

Menggunakan **Dataset IDoFT (Illinois Dataset of Flaky Tests)** yang diselaraskan dengan 23 fitur prediktif FlakeFlagger (12.626 baris pengujian, 17 proyek Java, 264 flaky tests ground truth riil).

---

## 📋 Ikhtisar & Landasan Riset

Eksperimen ini mengevaluasi dua paradigma validasi lintas proyek (*cross-project flaky test prediction*):
1. **Skema 1: One-on-One Pairwise Cross-Project Transfer Matrix**:  
   Pelatihan model pada 1 proyek sumber tunggal ($P_{src}$) dan pengujian pada 1 proyek target tunggal ($P_{tgt}$) untuk seluruh **156 kombinasi pasangan proyek aktif** ($13 \times 12$ pasangan) membandingkan:
   - `Baseline_1on1`
   - `Burak_Filter_1on1` ($k=10$)
   - `TrAdaBoost_50_1on1`
2. **Skema 2: Repeated Leave-One-Project-Out Cross-Validation (LOPO-CV)**:  
   Evaluasi berulang lintas **5 *independent random seeds*** ($[42, 100, 2024, 777, 999]$) pada 17 proyek IDoFT (total **1.020 evaluasi intervensi**) membandingkan 6 perlakuan Afeltra:
   - `T1_Baseline` (MinMaxScaler + Random Under-Sampling + RF/XGB)
   - `T2_Burak_Filter` ($k=10$ Euclidean distance nearest neighbors)
   - `T3_TrAdaBoost_25` (25% target adaptation split)
   - `T4_TrAdaBoost_50` (50% target adaptation split)
   - `T5_TrAdaBoost_75` (75% target adaptation split - konfigurasi terbaik paper Afeltra 2024)
   - `T6_Within_Project` (Batas atas ideal 5-Fold Stratified CV)

---

## 🌟 Temuan Kunci Penelitian

1. **Lonjakan Drastis Performa melalui TrAdaBoost**:
   - Pada transfer One-on-One (1-on-1), `TrAdaBoost_50_1on1` melipatgandakan rata-rata $F_1$-score hingga **0.1432** (**+103,1% di atas Baseline 0.0705**).
   - Pada Repeated LOPO-CV, `TrAdaBoost_75` mendongkrak rata-rata F1 makro aktif menjadi **0.2087** (**+272,0% di atas Baseline 0.0561**).
2. **Karakterisasi Proyek Donor Terbaik (Best Transfer Donors)**:
   Analisis One-on-One membuktikan bahwa `hbase` (F1 transfer donor **0.1928**), `undertow` (**0.1767**), `wildfly` (**0.1632**), dan `logback` (**0.1577**) bertindak sebagai *universal donor* yang memiliki ruang fitur paling kaya untuk mentransfer flakiness ke repositori lain.
3. **Stabilitas Stokastik Tinggi ($CV < 10\%$)**:
   Berbeda dengan Baseline yang sangat fluktuatif ($CV = 59,59\%$), `T5_TrAdaBoost_75` dan `T6_Within_Project` mencatatkan stabilitas stokastik superior dengan $CV = 6,84\% - 7,17\%$, membuktikan ketahanan algoritma adaptasi target terhadap variansi acak.
4. **Verifikasi Hipotesis Kelangkaan Minoritas ($r = 0.9309$)**:
   Korelasi linier Pearson mencapai **$r = 0.9309$ ($p = 3.84 \times 10^{-6}$)** dan Spearman mencapai **$\rho = 0.9066$ ($p = 1.93 \times 10^{-5}$)**, mengonfirmasi bahwa rendahnya F1 makro unweighted murni adalah akibat matematis kelangkaan kelas minoritas pada proyek target.

---

## 🗂️ Struktur Direktori

```
Eksperimen_Kelompok/bima-eksperiemn-3-one-on-one-afeltra-idoft/
├── README.md                                         # Dokumentasi ikhtisar & petunjuk eksekusi
├── RANCANGAN_PENELITIAN_AFELTRA_IDOFT.md             # Dokumen akademik metodologi penelitian lengkap
├── LAPORAN_EKSPERIMEN_3_ONE_ON_ONE_AFELTRA_IDOFT.md  # Laporan komprehensif temuan empiris & analisa dataset
├── configs/
│   └── experiment_matrix.json                        # Konfigurasi parameter eksperimen & perlakuan
├── data/
│   ├── cleaned_idoft_test_features.csv               # Dataset master IDoFT (12.626 baris, 17 proyek)
│   └── dataset_metadata.json                         # Demografi proyek & statistik flakiness
├── src/
│   ├── __init__.py
│   ├── data_loader.py                                # Loader dataset IDoFT
│   ├── burak_filter.py                               # Implementasi k-NN Burak Filter (k=10)
│   ├── tradaboost.py                                 # Implementasi algoritma TrAdaBoost transfer learning
│   ├── pipeline_builder.py                           # Pembangun scaler, sampler (RUS), dan classifier
│   ├── pairwise_one_on_one_runner.py                 # Runner Skema 1: Pairwise 1-on-1 (156 pairs)
│   ├── afeltra_lopo_runner.py                        # Runner Skema 2: Repeated LOPO-CV (1.020 folds)
│   ├── consistency_analyzer.py                       # Analisis konsistensi stokastik (Mean, Std, CV%, CI95)
│   ├── dataset_profiler.py                           # Matriks perlakuan vs dataset & dekonstruksi agregasi
│   ├── statistical_tests.py                          # Uji Wilcoxon, Cliff's Delta, dan Friedman
│   ├── visualizer.py                                 # Generator grafik publikasi resolusi tinggi (300 DPI)
│   └── main_executor.py                              # Orkestrator eksekusi end-to-end
├── notebooks/
│   ├── build_notebook.py                             # Skrip pembuat notebook otomatis
│   └── 01_run_afeltra_idoft_experiment.ipynb        # Notebook interaktif eksekusi eksperimen
└── results/
    ├── all_pairwise_one_on_one_predictions.csv       # Log 468 evaluasi One-on-One pairwise
    ├── matrix_pairwise_one_on_one_f1.csv             # Matriks 13x13 One-on-One transfer F1-score
    ├── matrix_pairwise_one_on_one_prauc.csv          # Matriks 13x13 One-on-One transfer PR-AUC
    ├── all_repeated_folds_predictions.csv            # Master CSV 1.020 evaluasi LOPO-CV
    ├── consistency_stability_analysis.csv            # Tabel stabilitas & CV % antar perlakuan
    ├── matrix_project_vs_afeltra_treatments_f1.csv   # Matriks Treatment x 17 Proyek F1 & rata-rata
    ├── macro_vs_weighted_f1_explanation.csv          # Komparasi Macro 17 vs Active 13 vs Weighted vs Micro F1
    ├── dataset_capability_profile.csv                # Profil flakiness rate & korelasi per proyek
    ├── statistical_wilcoxon_cliffs_delta.csv         # Hasil uji beda berpasangan Wilcoxon & Cliff's Delta
    ├── friedman_nemenyi_results.csv                  # Hasil uji Friedman Chi-Square & peringkat
    └── plots/                                        # 6 visualisasi ilmiah publikasi 300 DPI
        ├── heatmap_pairwise_one_on_one_f1.png        # Heatmap Matriks One-on-One Transfer Baseline
        ├── heatmap_pairwise_one_on_one_burak_f1.png  # Heatmap Matriks One-on-One Transfer Burak Filter
        ├── heatmap_afeltra_treatments_f1.png         # Heatmap Perbandingan Treatment vs Proyek IDoFT
        ├── boxplot_afeltra_metrics.png               # Boxplots F1, Recall, Precision, PR-AUC
        ├── scatter_flaky_rate_vs_f1.png              # Scatter Plot Korelasi Flaky Rate vs F1
        └── top_transfer_donors.png                   # Barplot Top Donors pada Transfer One-on-One
```

---

## ⚡ Petunjuk Menjalankan Eksperimen

1. **Menjalankan Seluruh Pipeline via Main Executor**:
   ```powershell
   .\.venv\Scripts\python.exe "Eksperimen_Kelompok/bima-eksperiemn-3-one-on-one-afeltra-idoft/src/main_executor.py"
   ```

2. **Membuka Notebook Interaktif**:
   ```powershell
   .\.venv\Scripts\jupyter.exe notebook "Eksperimen_Kelompok/bima-eksperiemn-3-one-on-one-afeltra-idoft/notebooks/01_run_afeltra_idoft_experiment.ipynb"
   ```
