# 🌟 Alternatif 1: Instance-Based Filtering & Transfer Learning (Burak Filter & TrAdaBoost)

**Kelompok Peneliti**: Kelompok A01 — Rekayasa Perangkat Lunak A (Magister Teknik Informatika ITS)  
**Dokumen Induk**: [`../../Rencana_Penelitian_Flaky_Test.md`](../../Rencana_Penelitian_Flaky_Test.md)  
**Hardware Akselerasi**: Laptop GPU NVIDIA GeForce RTX 4050 (6GB GDDR6 VRAM, CUDA 13.4, Driver 617.42)  
**Paper Rujukan**: 
1. Afeltra et al. (IEEE Access 2024) — *A Large-Scale Empirical Investigation Into Cross-Project Flaky Test Prediction*
2. Turhan et al. (IEEE TSE 2009) — *On the relative value of cross-company data for software quality prediction (Burak Filter)*
3. Dai et al. (ICML 2007) — *Boosting for Transfer Learning (TrAdaBoost)*

---

## 📌 Deskripsi Singkat Riset

Pada skenario prediksi lintas proyek (*Cross-Project Flaky Test Prediction* / LOPO-CV), model standar yang dilatih pada seluruh proyek eksternal ($N-1$ proyek) kerap mengalami degradasi performa drastis ($F1 \approx 0.03$, Afeltra et al., 2024) akibat **perbedaan distribusi fitur (*domain shift / distribution mismatch*)** antara proyek sumber dan proyek target.

Riset alternatif ini menerapkan dan membandingkan dua teknik *transfer learning* mutakhir:
1. **Burak Filter ($k$-NN Instance Selection)**: Memfilter data latih proyek sumber sehingga hanya memilih subset sampel yang memiliki jarak fitur terdekat dengan proyek target ($k=10$, rasio 10%–30%).
2. **TrAdaBoost (Transfer AdaBoost)**: Algoritma boosting adaptif yang secara iteratif menurunkan bobot (*down-weighting*) data sumber yang bertentangan dengan distribusi proyek target, sekaligus meningkatkan bobot data target yang relevan.

---

## 🗂️ Rencana Struktur Direktori

```
Alternatif-1_Transfer_Learning_Burak_TrAdaBoost/
├── README.md                                 # Ringkasan riset & dokumentasi
├── PROMPT_EKSEKUSI_ALTERNATIF_1.md          # Master prompt eksekusi AI agent
├── configs/
│   └── transfer_config.json                  # Konfigurasi parameter Burak (k, ratio) & TrAdaBoost (iterations)
├── src/
│   ├── __init__.py
│   ├── check_gpu.py                          # Verifikasi GPU RTX 4050 & CUDA 13
│   ├── data_loader.py                        # Pemuatan dataset FlakeFlagger (test_features.csv, test_results.csv)
│   ├── burak_filter.py                       # Implementasi k-NN instance filtering
│   ├── tradaboost.py                         # Implementasi TrAdaBoost dengan base classifier XGBoost GPU
│   ├── transfer_lopo_runner.py               # Engine evaluasi LOPO-CV dengan transfer learning
│   └── visualizer.py                         # Plot perbandingan F1, PR-AUC, dan kurva distribusi
├── notebooks/
│   └── 01_run_transfer_learning.ipynb        # Jupyter Notebook eksperimen interaktif
└── results/
    ├── raw_predictions/                      # Prediksi per fold per proyek
    ├── transfer_learning_summary.csv         # Rekapitulasi metrik komparasi (Baseline vs Burak vs TrAdaBoost)
    └── plots/                                # Boxplot perbandingan F1 & Precision-Recall curves
```

---

## ⚡ Akselerasi Komputasi GPU RTX 4050 & CUDA 13

- **XGBoost GPU**: Base learner untuk TrAdaBoost dan evaluasi downstream memanfaatkan `tree_method="hist"` dan `device="cuda"`.
- **Fast Distance Computation**: Perhitungan matriks jarak Euclidean untuk 20.000+ baris data dipercepat menggunakan kalkulasi vektor batch untuk mencegah bottleneck waktu komputasi.
- **VRAM Safety**: Mengatur alokasi VRAM 6GB agar tidak terjadi Out-Of-Memory (OOM) selama iterasi transfer boosting.

---

## 🚀 Cara Memulai

Salin isi dokumen [`PROMPT_EKSEKUSI_ALTERNATIF_1.md`](./PROMPT_EKSEKUSI_ALTERNATIF_1.md) ke AI coding assistant Anda untuk membangun kode, menjalankan eksperimen LOPO-CV transfer learning, dan menyusun laporan evaluasi lengkap.
