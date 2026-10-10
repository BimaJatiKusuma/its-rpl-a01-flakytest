# 🌟 Alternatif 3: Project-Agnostic Feature Engineering & Normalisasi Relatif

**Kelompok Peneliti**: Kelompok A01 — Rekayasa Perangkat Lunak A (Magister Teknik Informatika ITS)  
**Dokumen Induk**: [`../../Rencana_Penelitian_Flaky_Test.md`](../../Rencana_Penelitian_Flaky_Test.md)  
**Hardware Akselerasi**: Laptop GPU NVIDIA GeForce RTX 4050 (6GB GDDR6 VRAM, CUDA 13.4, Driver 617.42)  
**Paper Rujukan**:
1. Alshammari et al. (ICSE 2021) — *FlakeFlagger: Predicting Flakiness Without Rerunning Tests*
2. Afeltra et al. (IEEE Access 2024) — *A Large-Scale Empirical Investigation Into Cross-Project Flaky Test Prediction*
3. Pontillo et al. (EMSE 2022) — *Static Test Flakiness Prediction: How Far Can We Go?*

---

## 📌 Deskripsi Singkat Riset

Pada skenario prediksi lintas proyek (Cross-Project LOPO-CV), fitur metrik absolut seperti `testLength` (panjang kode), `numCoveredLines`, dan `projectSourceLinesCovered` sangat **terdistorsi oleh skala ukuran proyek** (misalnya repositori enterprise raksasa seperti `spring-boot` vs pustaka utilitas mikro seperti `jimfs`). Karakteristik ini memicu *domain shift* yang melemahkan transferabilitas model.

Selain itu, terdapat multikolinearitas tinggi pada 8 window fitur *code churn* `hIndex` ($r > 0.90$).

Riset alternatif ini membangun representasi fitur yang **Project-Agnostic (kebal terhadap skala proyek)** melalui:
1. **Rekayasa Fitur Rasio Relatif**:
   - $\text{assert\_density} = \frac{\text{numAsserts}}{\text{testLength} + 1}$
   - $\text{coverage\_ratio} = \frac{\text{numCoveredLines}}{\text{testLength} + 1}$
   - $\text{class\_coverage\_ratio} = \frac{\text{projectSourceClassesCovered}}{\text{num\_third\_party\_libs} + 1}$
   - $\text{execution\_time\_per\_assert} = \frac{\text{ExecutionTime}}{\text{numAsserts} + 1}$
2. **Transformasi Logaritmik ($\log(1+x)$)**: Menormalkan variabel berdistribusi *long-tailed/right-skewed*.
3. **Reduksi Dimensi PCA untuk Churn**: Mengompresi 8 window `hIndex` menjadi 1-2 *Principal Components* yang merangkum >95% variansi tanpa multikolinearitas.

---

## 🗂️ Rencana Struktur Direktori

```
Alternatif-3_Project_Agnostic_Feature_Engineering/
├── README.md                                 # Ringkasan riset & dokumentasi
├── PROMPT_EKSEKUSI_ALTERNATIF_3.md          # Master prompt eksekusi AI agent
├── configs/
│   └── feature_config.json                   # Definisi rumus fitur baru & parameter PCA
├── src/
│   ├── __init__.py
│   ├── check_gpu.py                          # Verifikasi GPU RTX 4050 & CUDA 13
│   ├── data_loader.py                        # Pemuatan data FlakeFlagger C1
│   ├── feature_engineer.py                   # Transformasi rasio, log-transform, dan PCA
│   ├── lopo_feature_runner.py                # Runner LOPO-CV komparasi Original vs Engineered Features
│   ├── statistical_tests.py                  # Uji Wilcoxon Signed-Rank Test
│   └── visualizer.py                         # Korelasi heatmap, distribusi rasio, dan boxplot perbandingan
├── notebooks/
│   └── 01_run_feature_engineering.ipynb      # Jupyter Notebook interaktif
└── results/
    ├── raw_predictions/                      # Prediksi per fold
    ├── feature_comparison_summary.csv        # Komparasi metrik: Fitur Asli (22) vs Fitur Rekayasa Baru
    ├── pca_explained_variance.csv            # Rasio variansi komponen PCA hIndex
    └── plots/                                # Heatmap korelasi & Boxplot komparasi F1/PR-AUC
```

---

## ⚡ Akselerasi Komputasi GPU RTX 4050 & CUDA 13

- **XGBoost GPU**: Pelatihan model pohon bergradien dengan regularisasi L1/L2 pada ruang dimensi fitur baru memanfaatkan GPU RTX 4050 (`tree_method="hist"`, `device="cuda"`).
- **Fast Multiprocessing**: Pipeline transformasi fitur scikit-learn terintegrasi secara cepat tanpa overhead I/O.

---

## 🚀 Cara Memulai

Salin isi dokumen [`PROMPT_EKSEKUSI_ALTERNATIF_3.md`](./PROMPT_EKSEKUSI_ALTERNATIF_3.md) ke AI coding assistant Anda untuk membangun kode, menjalankan eksperimen rekayasa fitur, dan menyusun laporan evaluasi lengkap.
