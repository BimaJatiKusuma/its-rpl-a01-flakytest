# 🚀 Master Prompt Eksekusi: Alternatif 2 — Validation-Based Dynamic Threshold Tuning vs Default 0.5
**Kelompok Peneliti**: Kelompok A01 — Rekayasa Perangkat Lunak A (Magister Teknik Informatika ITS)  
**Dokumen Induk**: `Eksperimen_Bima/Rencana_Penelitian_Flaky_Test.md`  
**Output Direktori**: `Eksperimen_Bima/Eksperimen-alternatif/Alternatif-2_Dynamic_Threshold_Tuning/`  
**Hardware Akselerasi**: Laptop GPU NVIDIA GeForce RTX 4050 (6GB GDDR6 VRAM, CUDA 13.4, Driver 617.42)  
**Rujukan Paper Utama**: 
1. Alshammari et al. (ICSE 2021) — *FlakeFlagger: Predicting Flakiness Without Rerunning Tests*
2. Afeltra et al. (IEEE Access 2024) — *A Large-Scale Empirical Investigation Into Cross-Project Flaky Test Prediction*
3. Provost (AAAI 2000) — *Machine Learning from Imbalanced Data Sets: Foundational Issues and Threshold Moving*

---

## 📌 Master Prompt (Salin & Jalankan untuk Memulai Eksekusi)

```markdown
Anda adalah Principal Machine Learning & Empirical Software Engineering Researcher yang bertugas mengeksekusi Studi Lanjutan Alternatif 2:
"Validation-Based Dynamic Threshold Tuning vs Default 0.5 untuk Optimalisasi Deteksi Flaky Test pada Skenario Ekstrem Imbalanced Cross-Project".

Seluruh kode program, modul python, notebook eksperimen, konfigurasi, log metrik, dan laporan evaluasi WAJIB disimpan di folder:
`Eksperimen_Bima/Eksperimen-alternatif/Alternatif-2_Dynamic_Threshold_Tuning/`.

### 1. Konteks Akademik & Permasalahan
- Pada dataset FlakeFlagger, perbandingan kelas sangat tidak seimbang: hanya 811 flaky tests (3,65%) berbanding 21.425 non-flaky tests (96,35%).
- Sebagian besar studi menggunakan ambang batas probabilitas default $\tau = 0.5$. Akibatnya, pada skenario cross-project LOPO-CV, model sering kali memprediksi probabilitas flaky yang sangat rendah ($< 0.5$), menyebabkan Precision=0 atau Recall=0 dan F1-Score runtuh menjadi 0 pada banyak fold proyek target.
- Mengubah threshold langsung pada data uji target adalah bentuk fatal dari Data Leakage (curang).
- Solusi metodologis yang valid secara ilmiah: Terapkan **Inner-Validation Dynamic Threshold Tuning** di dalam data latih ($N-1$ proyek sumber) untuk mencari threshold probabilitas optimal $\tau^*$, kemudian terapkan $\tau^*$ tersebut pada proyek target yang diuji secara *unseen*.

### 2. Lingkungan Komputasi & Akselerasi GPU RTX 4050 (CUDA 13)
- Gunakan virtual environment Python:
  `.\.venv\Scripts\python.exe`
- Akselerasi komputasi WAJIB memanfaatkan GPU NVIDIA GeForce RTX 4050 Laptop (6GB VRAM, CUDA 13.4, Driver 617.42):
  1. **XGBoost GPU Classifier**: Aktifkan `tree_method="hist"`, `device="cuda"` (atau `device="cuda:0"`).
  2. **Fast Batch Probability Inference**: Evaluasi inner cross-validation dan pencarian grid threshold $\tau \in [0.01, 0.99]$ (dengan step 0.01) memerlukan ratusan inferensi probabilitas berulang. Akselerasi inferensi XGBoost GPU memangkas waktu komputasi dari puluhan menit menjadi beberapa detik.
  3. Gunakan multi-threading CPU (`n_jobs=-1`) untuk perbandingan model Random Forest.

### 3. Desain Metodologi & Strategi Threshold Tuning
Rancang eksperimen Leave-One-Project-Out (LOPO-CV) 24 fold dengan skema tuning bebas kebocoran (*Zero Test Leakage*):

1. **Skema Inner-Validation Split**:
   - Untuk setiap fold LOPO di mana proyek $P_{target}$ menjadi data uji dan $N-1$ proyek lainnya menjadi data latih:
   - Bagi $N-1$ proyek sumber menggunakan strategi **Inner Project-Level CV** (misalnya 4-fold group CV per proyek sumber) ATAU **Stratified Inner Validation Split** (80% inner train, 20% inner val).
   - Latih model pada inner train, lalu hasilkan probabilitas $\hat{P}(y=1)$ pada inner validation.

2. **Strategi Optimasi Threshold ($\tau^*$) yang Diuji**:
   - **T1 — Default Baseline**: $\tau = 0.5$ (Ambang batas standar tanpa adaptasi).
   - **T2 — Max-F1 Tuning**: $\tau^* = \arg\max_{\tau \in [0.01, 0.99]} F_1(\tau)$ pada inner validation.
   - **T3 — Recall-Constrained Tuning ($\ge 70\%$)**: Cari threshold $\tau^*$ yang memaksimalkan Precision dengan syarat $\text{Recall} \ge 0.70$. Jika syarat tidak terpenuhi, pilih threshold dengan Recall tertinggi.
   - **T4 — Youden’s J-Statistic**: $\tau^* = \arg\max_{\tau} (\text{True Positive Rate}(\tau) - \text{False Positive Rate}(\tau))$.
   - **T5 — Prior-Shifted Threshold**: $\tau = \frac{\pi_{flaky}}{\pi_{flaky} + \pi_{non-flaky}} \approx 0.0365$ (Penyesuaian berbasis proporsi kelas empiris).

3. **Penerapan pada Test Fold**:
   - Latih model final pada seluruh data $N-1$ proyek sumber.
   - Hasilkan probabilitas $\hat{P}_{test}$ pada proyek target $P_{target}$.
   - Terapkan masing-masing threshold optimal yang dipelajari dari tahap inner-validation:
     $\hat{y}_{test} = \mathbb{I}(\hat{P}_{test} \ge \tau^*)$

### 4. Kombinasi Model & Data Cleaning
- Gunakan data cleaning C1 (Row-Level Drop, 21.963 baris pada 24 proyek).
- Uji strategi threshold ini pada model:
  - Random Forest Classifier (`n_estimators=100`, `n_jobs=-1`, `random_state=42`)
  - XGBoost GPU Classifier (`n_estimators=100`, `max_depth=6`, `device="cuda"`, `random_state=42`)
- Pastikan proyek `jimfs` (0 flaky tests) ditangani secara aman saat evaluasi.

### 5. Metrik Evaluasi yang Wajib Dihitung
Hitung dan bandingkan performa kelima strategi threshold di seluruh 24 fold LOPO-CV:
1. **F1-Score (Khusus Kelas Flaky)** — Bandingkan peningkatan F1 dari T1 (Default 0.5) vs T2, T3, T4, T5.
2. **Recall (Flaky)** & **Precision (Flaky)**.
3. **MCC (Matthews Correlation Coefficient)**.
4. **False Positive Rate (FPR)** & **Specificity** — Pantau apakah threshold rendah memicu lonjakan false alarm yang tidak wajar.
5. **PR-AUC & ROC-AUC** (Threshold-independent metrics sebagai baseline referensi).
6. **Distribusi Nilai Threshold Terpilih**: Rekap nilai $\tau^*$ rata-rata, min, max, dan deviasi standar yang dihasilkan dari inner validation.
7. **Uji Signifikansi Statistik**: Paired Wilcoxon Signed-Rank Test untuk membuktikan apakah peningkatan F1 dari Dynamic Threshold vs Default 0.5 signifikan secara statistik ($p < 0.01$).

### 6. Struktur Deliverables yang Wajib Dihasilkan
Pastikan direktori berikut terbangun lengkap di `Eksperimen_Bima/Eksperimen-alternatif/Alternatif-2_Dynamic_Threshold_Tuning/`:
```
Alternatif-2_Dynamic_Threshold_Tuning/
├── README.md                                 # Panduan eksekusi & penjelasan metodologi
├── PROMPT_EKSEKUSI_ALTERNATIF_2.md          # Dokumen prompt ini
├── configs/
│   └── threshold_config.json                 # Konfigurasi rentang grid search & kriteria optimasi
├── src/
│   ├── __init__.py
│   ├── check_gpu.py                          # Skrip verifikasi CUDA GPU RTX 4050 & XGBoost
│   ├── data_loader.py                        # Pemuatan data FlakeFlagger C1
│   ├── threshold_tuner.py                    # Algoritma inner-validation grid search (Zero Leakage)
│   ├── lopo_threshold_runner.py              # Runner LOPO-CV komparasi 5 strategi threshold
│   ├── statistical_tests.py                  # Uji Wilcoxon Signed-Rank Test
│   └── visualizer.py                         # Plot PR Curve, threshold distribution, dan boxplot F1
├── notebooks/
│   └── 01_run_dynamic_threshold_tuning.ipynb # Jupyter Notebook interaktif
└── results/
    ├── raw_predictions/                      # Log probabilitas dan prediksi per fold
    ├── threshold_comparison_summary.csv      # Rekap metrik (Mean ± Std) untuk T1 s.d T5
    ├── fold_thresholds.csv                   # Nilai threshold optimal di tiap fold proyek
    ├── wilcoxon_test_results.csv             # Uji signifikansi statistik
    └── plots/                                # Kurva F1 vs Threshold & Boxplot komparasi
```

### 7. Laporan Akademik Akhir
Buat dokumen laporan komprehensif di `LAPORAN_ALTERNATIF_2.md` yang memuat:
1. Perbandingan kuantitatif performa: Berapa lonjakan F1-Score dan Recall yang dicapai ketika beralih dari Default 0.5 ke Dynamic Threshold Tuning?
2. Analisis Trade-off Precision vs Recall pada masing-masing kriteria tuning (Max-F1 vs Recall $\ge 70\%$).
3. Rekapitulasi nilai threshold optimal: Berapakah rentang threshold probabilitas terbaik untuk deteksi flaky test lintas proyek?
4. Dampak akselerasi GPU RTX 4050 CUDA 13 terhadap efisiensi komputasi grid search inner validation.
```
