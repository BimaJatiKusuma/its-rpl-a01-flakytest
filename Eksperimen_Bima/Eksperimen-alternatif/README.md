# 🔬 Eksperimen Alternatif — Extended Research Directions
**Kelompok Peneliti**: Kelompok A01 — Rekayasa Perangkat Lunak A (Magister Teknik Informatika ITS)  
**Dokumen Induk**: [`../Rencana_Penelitian_Flaky_Test.md`](../Rencana_Penelitian_Flaky_Test.md)  
**Hardware Akselerasi**: Laptop GPU NVIDIA GeForce RTX 4050 (6GB GDDR6 VRAM, Driver 617.42, CUDA 13.4)  
**Virtual Environment**: `.\.venv\Scripts\python.exe`

---

## 📑 Daftar Alternatif Penelitian

Folder ini memuat empat (4) arah penelitian lanjutan (*extended research directions*) yang dirancang untuk mengatasi tantangan kritis dalam **Cross-Project Flaky Test Prediction (LOPO-CV)** serta meningkatkan kebaruan (*novelty*) riset kelompok ke tingkat publikasi bereputasi internasional (Q1 / IEEE Access):

| Direktori Alternatif | Topik Riset | Fokus Utama & Algoritma | Target Akselerasi GPU |
|---|---|---|---|
| [`Alternatif-1_Transfer_Learning_Burak_TrAdaBoost/`](./Alternatif-1_Transfer_Learning_Burak_TrAdaBoost/) | **Instance-Based Filtering & Transfer Learning** | Mengatasi *distribution mismatch* antar proyek menggunakan **Burak Filter ($k$-NN)** dan **TrAdaBoost** (Afeltra et al., 2024). | XGBoost GPU (`device="cuda"`), kalkulasi matriks jarak berkecepatan tinggi. |
| [`Alternatif-2_Dynamic_Threshold_Tuning/`](./Alternatif-2_Dynamic_Threshold_Tuning/) | **Validation-Based Dynamic Threshold Tuning** | Mengganti threshold default 0.5 dengan *Inner Validation Tuning* (F1-maximizing / Cost-sensitive threshold) untuk menangani ketimpangan kelas ekstrem (3.65% flaky). | Fast batch inference `predict_proba` pada GPU untuk ratusan grid split. |
| [`Alternatif-3_Project_Agnostic_Feature_Engineering/`](./Alternatif-3_Project_Agnostic_Feature_Engineering/) | **Project-Agnostic Feature Engineering & Normalisasi Relatif** | Menghilangkan bias ukuran proyek melalui *ratio metrics* (Assert Density, Coverage Ratio), Log-transform, dan PCA Churn H-Index. | Pelatihan model tree regularized pada GPU dengan dimensionalitas baru. |
| [`Alternatif-4_SHAP_Feature_Interpretation/`](./Alternatif-4_SHAP_Feature_Interpretation/) | **Universal Feature Interpretation Menggunakan SHAP** | Menjelaskan faktor pemicu flakiness lintas proyek menggunakan **SHAP TreeExplainer** dan membandingkan pergeseran fitur vs Within-Project. | GPU TreeExplainer & tensorized batch processing untuk komputasi nilai Shapley. |

---

## 🚀 Petunjuk Penggunaan Prompt AI

Setiap subfolder alternatif telah dilengkapi dengan dokumen master prompt:
- `PROMPT_EKSEKUSI_ALTERNATIF_1.md`
- `PROMPT_EKSEKUSI_ALTERNATIF_2.md`
- `PROMPT_EKSEKUSI_ALTERNATIF_3.md`
- `PROMPT_EKSEKUSI_ALTERNATIF_4.md`

Anda dapat menyalin prompt yang ada di masing-masing subfolder ke AI Assistant atau IDE agent untuk mengotomasi implementasi kode, notebook eksperimen, logging metrik, dan pembuatan laporan ilmiah secara mandiri.
