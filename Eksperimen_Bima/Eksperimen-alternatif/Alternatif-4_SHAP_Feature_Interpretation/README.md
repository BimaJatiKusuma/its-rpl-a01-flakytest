# 🌟 Alternatif 4: Interpretasi Fitur Universal Menggunakan SHAP (SHapley Additive exPlanations)

**Kelompok Peneliti**: Kelompok A01 — Rekayasa Perangkat Lunak A (Magister Teknik Informatika ITS)  
**Dokumen Induk**: [`../../Rencana_Penelitian_Flaky_Test.md`](../../Rencana_Penelitian_Flaky_Test.md)  
**Hardware Akselerasi**: Laptop GPU NVIDIA GeForce RTX 4050 (6GB GDDR6 VRAM, CUDA 13.4, Driver 617.42)  
**Paper Rujukan**:
1. Lundberg & Lee (NeurIPS 2017) — *A Unified Approach to Interpreting Model Predictions (SHAP)*
2. Alshammari et al. (ICSE 2021) — *FlakeFlagger: Predicting Flakiness Without Rerunning Tests*
3. Afeltra et al. (IEEE Access 2024) — *A Large-Scale Empirical Investigation Into Cross-Project Flaky Test Prediction*

---

## 📌 Deskripsi Singkat Riset

Pengembang perangkat lunak (*software engineers*) tidak hanya membutuhkan prediksi biner apakah sebuah unit test *flaky* atau *non-flaky*, melainkan membutuhkan penjelasan sebab-akibat (*explainability*) mengenai **faktor apa yang memicu sifat flakiness tersebut** agar dapat melakukan *refactoring* kode pengujian.

Riset alternatif ini menerapkan teori permainan kooperatif **SHAP (SHapley Additive exPlanations)** melalui **TreeExplainer** untuk:
1. **Global Feature Importance**: Mengidentifikasi fitur penentu flakiness paling dominan secara agregat di seluruh 24 repositori open-source Java.
2. **Komparasi Within-Project vs Cross-Project**: Menjawab hipotesis mendasar:
   *Apakah fitur dominan pada skenario cross-project sama dengan within-project (misal: apakah metrik cakupan baris tetap dominan, ataukah fitur Test Smells seperti `fire-and-forget` dan `sleep` menjadi lebih relevan lintas proyek karena sifatnya yang domain-agnostic)?*
3. **Local Explanations**: Menganalisis *waterfall plot* untuk sampel flaky test spesifik (misalnya pada proyek `spring-boot` dan `wildfly`).
4. **Actionable Developer Guidelines**: Menemukan ambang batas kritis fitur melalui *SHAP Dependence Plots* untuk panduan pencegahan flaky test.

---

## 🗂️ Rencana Struktur Direktori

```
Alternatif-4_SHAP_Feature_Interpretation/
├── README.md                                 # Ringkasan riset & dokumentasi
├── PROMPT_EKSEKUSI_ALTERNATIF_4.md          # Master prompt eksekusi AI agent
├── configs/
│   └── shap_config.json                      # Parameter konfigurasi TreeExplainer & sampling background
├── src/
│   ├── __init__.py
│   ├── check_gpu.py                          # Verifikasi GPU RTX 4050 & CUDA 13
│   ├── data_loader.py                        # Pemuatan data FlakeFlagger C1
│   ├── shap_explainer.py                     # Implementasi SHAP TreeExplainer pada XGBoost GPU & RF
│   ├── cross_vs_within_analyzer.py           # Komparasi peringkat fitur Cross-Project vs Within-Project
│   └── visualizer.py                         # Pembuatan Beeswarm plot, Bar summary, Waterfall, dan Dependence plots
├── notebooks/
│   └── 01_run_shap_interpretation.ipynb      # Jupyter Notebook interaktif
└── results/
    ├── shap_values_summary.csv               # Rata-rata nilai absolut SHAP per fitur
    ├── feature_rankings_cross_vs_within.csv  # Tabel komparasi ranking fitur Cross vs Within
    └── plots/                                # Beeswarm summary plots, waterfall plots, dependence plots
```

---

## ⚡ Akselerasi Komputasi GPU RTX 4050 & CUDA 13

- **XGBoost GPU TreeExplainer**: Model XGBoost dilatih menggunakan akselerasi GPU RTX 4050 (`tree_method="hist"`, `device="cuda"`), sehingga struktur ensemble pohon dapat diekstrak secara cepat untuk perhitungan nilai Shapley.
- **Batched Matrix Processing**: Komputasi nilai SHAP untuk ribuan instance uji dieksekusi secara teroptimasi pada memori VRAM 6GB tanpa kendala memory overflow.

---

## 🚀 Cara Memulai

Salin isi dokumen [`PROMPT_EKSEKUSI_ALTERNATIF_4.md`](./PROMPT_EKSEKUSI_ALTERNATIF_4.md) ke AI coding assistant Anda untuk membangun kode, menjalankan eksperimen interpretasi SHAP, dan menyusun laporan evaluasi lengkap.
