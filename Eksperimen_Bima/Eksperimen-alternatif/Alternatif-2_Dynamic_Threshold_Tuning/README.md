# 🌟 Alternatif 2: Validation-Based Dynamic Threshold Tuning vs Default 0.5

**Kelompok Peneliti**: Kelompok A01 — Rekayasa Perangkat Lunak A (Magister Teknik Informatika ITS)  
**Dokumen Induk**: [`../../Rencana_Penelitian_Flaky_Test.md`](../../Rencana_Penelitian_Flaky_Test.md)  
**Hardware Akselerasi**: Laptop GPU NVIDIA GeForce RTX 4050 (6GB GDDR6 VRAM, CUDA 13.4, Driver 617.42)  
**Paper Rujukan**:
1. Alshammari et al. (ICSE 2021) — *FlakeFlagger: Predicting Flakiness Without Rerunning Tests*
2. Afeltra et al. (IEEE Access 2024) — *A Large-Scale Empirical Investigation Into Cross-Project Flaky Test Prediction*
3. Provost (AAAI 2000) — *Machine Learning from Imbalanced Data Sets: Foundational Issues and Threshold Moving*

---

## 📌 Deskripsi Singkat Riset

Pada data dengan ketimpangan kelas ekstrem seperti dataset FlakeFlagger (hanya **3,65% kelas flaky** dan **96,35% non-flaky**), penggunaan ambang batas probabilitas default $0.5$ ($\hat{y} = 1 \iff P(y=1) \ge 0.5$) sering kali menyebabkan kegagalan fatal:
- Model cenderung memprediksi seluruh sampel sebagai kelas mayoritas (non-flaky).
- Menghasilkan nilai **Recall = 0** atau **Precision = 0** dan **F1-Score = 0** pada banyak proyek target.

Riset alternatif ini mengeksplorasi **Dynamic Threshold Moving** berbasis validasi internal (*Inner-Validation Split*) di dalam training set ($N-1$ proyek sumber) untuk mencari ambang batas probabilitas optimal $\tau^*$ tanpa menimbulkan kebocoran data (*zero test-set data leakage*).

Strategi optimasi threshold yang diuji:
1. **F1-Maximizing Threshold**: $\tau^* = \arg\max_\tau F_1(\tau)$
2. **Quality-Guaranteed Recall Target**: $\tau^* = \arg\max_\tau \{\text{Precision}(\tau) \mid \text{Recall}(\tau) \ge 0.70\}$
3. **Youden's J-Statistic (Balanced Sensitivity-Specificity)**: $\tau^* = \arg\max_\tau (\text{TPR} - \text{FPR})$

---

## 🗂️ Rencana Struktur Direktori

```
Alternatif-2_Dynamic_Threshold_Tuning/
├── README.md                                 # Ringkasan riset & dokumentasi
├── PROMPT_EKSEKUSI_ALTERNATIF_2.md          # Master prompt eksekusi AI agent
├── configs/
│   └── threshold_config.json                 # Konfigurasi rentang grid search threshold & rasio inner split
├── src/
│   ├── __init__.py
│   ├── check_gpu.py                          # Verifikasi GPU RTX 4050 & CUDA 13
│   ├── data_loader.py                        # Pemuatan data FlakeFlagger
│   ├── threshold_tuner.py                    # Algoritma tuning threshold berbasis inner validation (Zero Leakage)
│   ├── lopo_threshold_runner.py              # Engine eksekusi LOPO-CV komparasi default 0.5 vs dynamic threshold
│   ├── statistical_tests.py                  # Uji signifikansi Wilcoxon / Paired t-test
│   └── visualizer.py                         # Kurva Precision-Recall vs Threshold, Boxplot F1
├── notebooks/
│   └── 01_run_dynamic_threshold_tuning.ipynb # Jupyter Notebook interaktif
└── results/
    ├── raw_predictions/                      # Log probabilitas dan prediksi per fold
    ├── threshold_comparison_summary.csv      # Komparasi metrik: Default 0.5 vs Max-F1 vs Recall>=70% vs Youden
    ├── learned_thresholds_per_fold.csv       # Nilai threshold optimal yang dipelajari di tiap fold
    └── plots/                                # Kurva PR vs Threshold, Boxplot perbandingan F1
```

---

## ⚡ Akselerasi Komputasi GPU RTX 4050 & CUDA 13

- **Batch Probability Inference**: Ekstraksi probabilitas prediksi `predict_proba` untuk ratusan kombinasi inner fold dan grid threshold $\tau \in [0.01, 0.99]$ diakselerasi penuh pada GPU RTX 4050 menggunakan XGBoost CUDA (`device="cuda"`).
- **Zero VRAM Leak**: Pengelolaan memory cache GPU secara aman untuk proses iterasi grid tanpa risiko crash atau OOM.

---

## 🚀 Cara Memulai

Salin isi dokumen [`PROMPT_EKSEKUSI_ALTERNATIF_2.md`](./PROMPT_EKSEKUSI_ALTERNATIF_2.md) ke AI coding assistant Anda untuk membangun kode, menjalankan eksperimen LOPO-CV threshold tuning, dan menyusun laporan evaluasi lengkap.
