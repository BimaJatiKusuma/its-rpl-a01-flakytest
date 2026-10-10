# 🌟 Alternatif 2: Validation-Based Dynamic Threshold Tuning vs Default 0.5
## Optimalisasi Deteksi Flaky Test pada Skenario Ekstrem Imbalanced Cross-Project

**Kelompok Peneliti**: Kelompok A01 — Rekayasa Perangkat Lunak A (Magister Teknik Informatika ITS)  
**Dokumen Induk**: [`../../Rencana_Penelitian_Flaky_Test.md`](../../Rencana_Penelitian_Flaky_Test.md)  
**Dokumen Laporan Lengkap**: [`./LAPORAN_ALTERNATIF_2.md`](./LAPORAN_ALTERNATIF_2.md)  
**Hardware Akselerasi**: Laptop GPU NVIDIA GeForce RTX 4050 (6GB GDDR6 VRAM, CUDA 13.4, Driver 617.42)  
**Rujukan Paper Utama**:
1. Alshammari et al. (ICSE 2021) — *FlakeFlagger: Predicting Flakiness Without Rerunning Tests*
2. Afeltra et al. (IEEE Access 2024) — *A Large-Scale Empirical Investigation Into Cross-Project Flaky Test Prediction*
3. Provost (AAAI 2000) — *Machine Learning from Imbalanced Data Sets: Foundational Issues and Threshold Moving*

---

## 📌 Deskripsi Riset & Permasalahan

Pada dataset FlakeFlagger, perbandingan kelas sangat tidak seimbang: hanya **808 flaky tests (3,68%)** berbanding **21.155 non-flaky tests (96,32%)**.

Sebagian besar studi terdahulu menggunakan ambang batas probabilitas default $\tau = 0.50$. Akibatnya, pada skenario cross-project Leave-One-Project-Out (LOPO-CV), model sering kali memprediksi probabilitas flaky yang sangat rendah ($< 0.50$), menyebabkan **Precision = 0**, **Recall = 0**, dan **F1-Score runtuh menjadi 0** pada 14 dari 24 fold proyek target.

Mengubah threshold langsung pada data uji target adalah bentuk fatal dari **Data Leakage**. Oleh karena itu, riset ini menerapkan **Inner-Validation Dynamic Threshold Tuning** bebas kebocoran (*Zero Test Leakage*) di dalam data latih ($N-1$ proyek sumber) untuk mencari threshold probabilitas optimal $\tau^*$, kemudian menerapkannya secara *blind* pada proyek target yang diuji secara *unseen*.

---

## 🔬 5 Strategi Threshold yang Diuji

1. **T1 — Default Baseline**: $\tau = 0.50$ (Ambang batas standar tanpa adaptasi).
2. **T2 — Max-F1 Tuning**: $\tau^* = \arg\max_{\tau \in [0.01, 0.99]} F_1(\tau)$ pada inner validation.
3. **T3 — Recall-Constrained Tuning ($\ge 70\%$)**: Cari threshold $\tau^*$ yang memaksimalkan Precision dengan syarat $\text{Recall} \ge 0.70$.
4. **T4 — Youden’s J-Statistic**: $\tau^* = \arg\max_{\tau} (\text{TPR}(\tau) - \text{FPR}(\tau)) = \arg\max_{\tau} (\text{Sensitivity} + \text{Specificity} - 1)$.
5. **T5 — Prior-Shifted Threshold**: $\tau^* = \bar{y}_{\text{source}} \approx 0.0368$ (Penyesuaian berbasis proporsi kelas empiris data latih).

---

## 📊 Ringkasan Hasil Utama (LOPO-CV 24 Fold)

### 1. XGBoost GPU Classifier (`device="cuda"`)

| Strategi | $\tau^*$ (Mean) | F1-Score | Recall | Precision | MCC | Specificity | FPR | Gain F1 vs T1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **T1: Default 0.5** | $0.50$ | $0.0550$ | $7,10\%$ | $0.2252$ | $0.0575$ | $97,11\%$ | $2,89\%$ | Baseline |
| **T2: Max-F1** | $0.34$ | $0.0621$ | $8,79\%$ | $0.2137$ | $0.0544$ | $95,27\%$ | $4,73\%$ | $+12,9\%$ |
| **T3: Recall $\ge 70\%$** | $0.20$ | $0.0860$ | $13,48\%$ | $0.1755$ | $0.0630$ | $93,42\%$ | $6,58\%$ | $+56,3\%$ |
| **T4: Youden’s J** | $\mathbf{0.04}$ | $\mathbf{0.1460}$ | $\mathbf{38,06\%}$ | $0.1857$ | $\mathbf{0.1115}$ | $82,11\%$ | $17,89\%$ | **+165,4% ($p < 0.05$)** |
| **T5: Prior-Shifted** | $0.04$ | $0.1384$ | $\mathbf{38,79\%}$ | $0.1285$ | $0.0950$ | $80,15\%$ | $19,85\%$ | $+151,4\%$ |

### 2. Random Forest Classifier (100 Trees, CPU Multi-Threading)

| Strategi | $\tau^*$ (Mean) | F1-Score | Recall | Precision | MCC | Specificity | FPR | Gain F1 vs T1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **T1: Default 0.5** | $0.50$ | $0.0248$ | $5,44\%$ | $0.1332$ | $0.0192$ | $97,51\%$ | $2,49\%$ | Baseline |
| **T2: Max-F1** | $0.37$ | $0.0624$ | $10,34\%$ | $0.1532$ | $0.0461$ | $94,70\%$ | $5,30\%$ | $+151,3\%$ |
| **T3: Recall $\ge 70\%$** | $0.30$ | $0.0852$ | $14,01\%$ | $0.1683$ | $0.0668$ | $93,61\%$ | $6,39\%$ | $+243,0\%$ |
| **T4: Youden’s J** | $\mathbf{0.06}$ | $0.1639$ | $54,29\%$ | $0.1334$ | $0.1293$ | $74,33\%$ | $25,67\%$ | $+560,0\%$ |
| **T5: Prior-Shifted** | $0.04$ | $\mathbf{0.1948}$ | $\mathbf{67,71\%}$ | $0.1450$ | $\mathbf{0.1640}$ | $65,32\%$ | $34,68\%$ | **+684,7% ($p < 0.01$)** |

---

## 🗂️ Struktur Direktori Deliverables

```
Alternatif-2_Dynamic_Threshold_Tuning/
├── README.md                                 # Panduan eksekusi & ringkasan hasil
├── LAPORAN_ALTERNATIF_2.md                   # Laporan penelitian akademik komprehensif
├── PROMPT_EKSEKUSI_ALTERNATIF_2.md          # Master prompt rujukan
├── configs/
│   └── threshold_config.json                 # Konfigurasi grid search, inner split, dan hyperparameter
├── src/
│   ├── __init__.py                           # Paket modul
│   ├── check_gpu.py                          # Verifikasi GPU RTX 4050 & CUDA 13.4
│   ├── data_loader.py                        # Pemuatan data FlakeFlagger C1 (21.963 baris valid)
│   ├── threshold_tuner.py                    # Engine inner-validation grid search (Zero Leakage)
│   ├── lopo_threshold_runner.py              # Runner LOPO-CV 24 fold lintas 5 strategi
│   ├── statistical_tests.py                  # Uji Wilcoxon Signed-Rank Test & Cliff's Delta
│   └── visualizer.py                         # Pembuat grafik publikasi standar IEEE/ACM (300 DPI)
├── notebooks/
│   └── 01_run_dynamic_threshold_tuning.ipynb # Jupyter Notebook interaktif
└── results/
    ├── raw_predictions/                      # Log probabilitas dan prediksi lengkap per fold
    ├── threshold_comparison_summary.csv      # Rekap metrik (Mean ± Std) untuk seluruh strategi
    ├── fold_thresholds.csv                   # Data metrik per fold proyek (240 baris)
    ├── wilcoxon_test_results.csv             # Hasil uji signifikansi statistik
    └── plots/                                # 10 visualisasi publikasi ilmiah (Boxplot, Bars, Trade-off)
```

---

## 🚀 Panduan Eksekusi

### 1. Verifikasi GPU
```bash
& "..\..\.venv\Scripts\python.exe" src\check_gpu.py
```

### 2. Menjalankan LOPO-CV Eksperimen Penuh (24 Fold)
```bash
& "..\..\.venv\Scripts\python.exe" src\lopo_threshold_runner.py --models xgboost_gpu random_forest
```

### 3. Menjalankan Uji Signifikansi Statistik
```bash
& "..\..\.venv\Scripts\python.exe" src\statistical_tests.py
```

### 4. Membuat Seluruh Plot Visualisasi Publikasi
```bash
& "..\..\.venv\Scripts\python.exe" src\visualizer.py
```

---

## ⚡ Efisiensi Komputasi GPU RTX 4050
- Waktu latih model per fold: **0,165 detik**.
- Waktu grid search 99 titik inner validation per fold: **0,204 detik**.
- Keseluruhan 24 fold LOPO-CV (termasuk validasi bersarang dan inferensi): tuntas hanya dalam **10,50 detik**.
