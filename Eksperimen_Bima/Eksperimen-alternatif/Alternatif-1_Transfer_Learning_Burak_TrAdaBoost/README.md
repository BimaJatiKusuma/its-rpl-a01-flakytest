# 🌟 Alternatif 1: Instance-Based Filtering & Transfer Learning (Burak Filter & TrAdaBoost)

**Kelompok Peneliti**: Kelompok A01 — Rekayasa Perangkat Lunak A (Magister Teknik Informatika ITS)  
**Dokumen Induk**: [`../../Rencana_Penelitian_Flaky_Test.md`](../../Rencana_Penelitian_Flaky_Test.md)  
**Master Prompt**: [`PROMPT_EKSEKUSI_ALTERNATIF_1.md`](./PROMPT_EKSEKUSI_ALTERNATIF_1.md)  
**Laporan Lengkap**: [`LAPORAN_ALTERNATIF_1.md`](./LAPORAN_ALTERNATIF_1.md)  
**Hardware Akselerasi**: Laptop GPU NVIDIA GeForce RTX 4050 (6GB GDDR6 VRAM, CUDA 13.4, Driver 617.42)  
**Dataset**: FlakeFlagger (22.236 test cases dari 24 repositori perangkat lunak Java)  

---

## 📌 1. Latar Belakang & Motivasi Riset

Pada skenario prediksi flaky test lintas proyek (*Cross-Project Flaky Test Prediction* / LOPO-CV), model machine learning konvensional yang dilatih secara naif pada seluruh proyek eksternal ($N-1$ proyek) kerap mengalami kegagalan performa berat ($F_1 \approx 0.03$, Afeltra et al., IEEE Access 2024). Fenomena ini dipicu oleh:
1. **Domain Shift / Distribution Mismatch**: Distribusi fitur arsitektur pengujian, frekuensi commit, dan code churn sangat bervariasi antar repositori perangkat lunak independen.
2. **Negative Transfer**: Memasukkan instance dari repositori sumber yang tidak memiliki kemiripan manifold dengan proyek target justru memperkenalkan noise dan bias struktural.

Untuk mengatasi permasalahan ini, riset ini mengimplementasikan dan menguji dua paradigma transfer learning komplementer:
- **Burak Filter ($k$-NN Instance Filtering)** (Turhan et al., IEEE TSE 2009): Menyeleksi hanya instance data sumber yang memiliki kedekatan jarak Euclidean dengan manifold fitur proyek target ($k=10$).
- **TrAdaBoost (Transfer AdaBoost)** (Dai et al., ICML 2007): Algoritma boosting adaptif yang secara cerdas menurunkan bobot (*down-weighting*) data sumber yang membingungkan/kontradiktif, sembari menaikkan bobot instance target berlabel (*few-shot transfer*).
- **Hybrid Burak Filter + TrAdaBoost**: Kombinasi hibrida dua tahap di mana Burak Filter pertama-tama membuang 70% data sumber yang paling tidak relevan, kemudian TrAdaBoost melakukan adaptasi bobot halus (*fine-grained reweighting*).

---

## 🔬 2. Desain Eksperimen & 4 Konfigurasi Model

Evaluasi dilakukan menggunakan protokol **Leave-One-Project-Out Cross-Validation (LOPO-CV)** pada seluruh 24 proyek perangkat lunak Java dengan strategi pembersihan data **C1 (Row-Level Drop: 21.963 baris valid)**.

| ID | Nama Konfigurasi | Strategi Algoritma | Base Classifier | Skema Transfer |
| :--- | :--- | :--- | :--- | :--- |
| **A** | **Baseline Cross-Project** | Latih pada seluruh data sumber $N-1$ tanpa filter | XGBoost GPU (`device="cuda"`) | Tanpa Transfer Learning |
| **B** | **Burak Filter** | $k$-NN Instance Selection ($k=10$) union manifold | XGBoost GPU (`device="cuda"`) | Instance-Based Selection |
| **C** | **TrAdaBoost** | 30 Iterasi Transfer Boosting adaptif | Ensemble XGBoost GPU | Sample Reweighting (Few-Shot 10%) |
| **D** | **Hybrid (Burak + TrAdaBoost)** | Filter top 30% jarak manifold + TrAdaBoost | Ensemble XGBoost GPU | Hybrid Filtering & Reweighting |

---

## 🗂️ 3. Struktur Direktori Proyek

```
Alternatif-1_Transfer_Learning_Burak_TrAdaBoost/
├── README.md                                 # Panduan eksekusi & ikhtisar eksperimen ini
├── PROMPT_EKSEKUSI_ALTERNATIF_1.md          # Master prompt instruksi eksekusi AI agent
├── LAPORAN_ALTERNATIF_1.md                   # Laporan komprehensif hasil temuan ilmiah
├── configs/
│   └── transfer_config.json                  # Parameter k-NN, boosting iterations, dan GPU settings
├── src/
│   ├── __init__.py                           # Python package init
│   ├── check_gpu.py                          # Pemeriksaan GPU RTX 4050 & CUDA 13.4
│   ├── data_loader.py                        # Pemuatan data FlakeFlagger (C1: 21.963 baris, 24 proyek)
│   ├── burak_filter.py                       # Algoritma Burak Filter k-NN & Manifold Ratio
│   ├── tradaboost.py                         # Algoritma TrAdaBoost terakselerasi GPU XGBoost
│   ├── transfer_lopo_runner.py               # Runner LOPO-CV 4 konfigurasi pada 24 proyek
│   ├── statistical_tests.py                  # Uji Wilcoxon Signed-Rank Test & Cliff's Delta
│   └── visualizer.py                         # Pembuat grafik boxplot, bar chart, dan runtime
├── notebooks/
│   └── 01_run_transfer_learning.ipynb        # Jupyter Notebook eksperimen interaktif
└── results/
    ├── raw_predictions/                      # Log prediksi baris per baris per fold
    ├── transfer_learning_folds.csv           # Metrik evaluasi detail per fold
    ├── transfer_learning_summary.csv         # Ringkasan statistik (Mean ± Std, Median)
    ├── wilcoxon_test_results.csv             # Nilai p-value signifikansi statistik
    └── plots/                                # Grafik publikasi ilmiah (PNG 300 DPI)
        ├── boxplot_comparison_f1_prauc.png
        ├── barchart_metrics_comparison.png
        ├── gpu_runtime_efficiency.png
        └── per_project_f1_gain.png
```

---

## ⚡ 4. Akselerasi Komputasi Hardware (RTX 4050 Laptop GPU)

Eksperimen memanfaatkan akselerasi GPU secara penuh untuk memangkas waktu komputasi LOPO-CV yang melibatkan puluhan ribu sampel:
- **XGBoost GPU**: Menggunakan parameter `tree_method="hist"` dan `device="cuda"`, memungkinkan pembuatan ratusan pohon keputusan dalam fraksi detik.
- **Vektor Jarak Terakselerasi**: Query nearest neighbor memanfaatkan pustaka `NearestNeighbors` teroptimasi multi-threading.
- **Efisiensi Memori (VRAM Safety)**: Seluruh alokasi matriks dipelihara di bawah batas 6GB VRAM GDDR6, bebas dari bahaya Out-of-Memory (OOM).

---

## 🛠️ 5. Cara Menjalankan Eksperimen

Gunakan virtual environment Python proyek (`.\.venv\Scripts\python.exe`):

```powershell
# 1. Verifikasi Akselerasi GPU RTX 4050
.\.venv\Scripts\python.exe src/check_gpu.py

# 2. Jalankan LOPO-CV untuk Seluruh 4 Konfigurasi
.\.venv\Scripts\python.exe src/transfer_lopo_runner.py

# 3. Jalankan Uji Signifikansi Statistik Wilcoxon Signed-Rank Test
.\.venv\Scripts\python.exe src/statistical_tests.py

# 4. Hasilkan Seluruh Visualisasi Publikasi Ilmiah
.\.venv\Scripts\python.exe src/visualizer.py
```

Atau buka notebook interaktif:
```powershell
jupyter notebook notebooks/01_run_transfer_learning.ipynb
```

---

## 📚 6. Referensi Ilmiah

1. **Afeltra, F. et al.** (2024). *A Large-Scale Empirical Investigation Into Cross-Project Flaky Test Prediction*. IEEE Access.
2. **Turhan, B. et al.** (2009). *On the Relative Value of Cross-Company Data for Software Quality Prediction*. IEEE Transactions on Software Engineering (TSE).
3. **Dai, W. et al.** (2007). *Boosting for Transfer Learning (TrAdaBoost)*. Proceedings of the 24th International Conference on Machine Learning (ICML).
4. **Alshammari, A. et al.** (2021). *FlakeFlagger: Predicting Flakiness Without Rerunning Tests*. IEEE/ACM International Conference on Software Engineering (ICSE).
