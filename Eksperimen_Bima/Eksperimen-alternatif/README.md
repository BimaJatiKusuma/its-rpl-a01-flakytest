# 🔬 Eksperimen Alternatif — Extended Research Directions
**Kelompok Peneliti**: Kelompok A01 — Rekayasa Perangkat Lunak A (Magister Teknik Informatika ITS)  
**Dokumen Induk**: [`../Rencana_Penelitian_Flaky_Test.md`](../Rencana_Penelitian_Flaky_Test.md)  
**Hardware Akselerasi**: Laptop GPU NVIDIA GeForce RTX 4050 (6GB GDDR6 VRAM, Driver 617.42, CUDA 13.4)  
**Virtual Environment**: `.\.venv\Scripts\python.exe`

---

## 📑 Status Eksekusi Alternatif Penelitian

Folder ini memuat empat (4) arah penelitian lanjutan (*extended research directions*) yang dirancang untuk mengatasi tantangan kritis dalam **Cross-Project Flaky Test Prediction (LOPO-CV)** serta meningkatkan kebaruan (*novelty*) riset kelompok ke tingkat publikasi bereputasi internasional (Q1 / IEEE Access):

| Direktori Alternatif | Topik Riset | Status Eksekusi | Laporan & Deliverables |
|---|---|:---:|---|
| [`Alternatif-1_Transfer_Learning_Burak_TrAdaBoost/`](./Alternatif-1_Transfer_Learning_Burak_TrAdaBoost/) | **Instance-Based Filtering & Transfer Learning** (Burak Filter & TrAdaBoost) | ✅ **Selesai** | [`LAPORAN_ALTERNATIF_1.md`](./Alternatif-1_Transfer_Learning_Burak_TrAdaBoost/LAPORAN_ALTERNATIF_1.md) — *Recall naik +91,7%, PR-AUC naik +14,7%* |
| [`Alternatif-2_Dynamic_Threshold_Tuning/`](./Alternatif-2_Dynamic_Threshold_Tuning/) | **Validation-Based Dynamic Threshold Tuning vs Default 0.5** (Zero Leakage) | ✅ **Selesai** | [`LAPORAN_ALTERNATIF_2.md`](./Alternatif-2_Dynamic_Threshold_Tuning/LAPORAN_ALTERNATIF_2.md) — *F1 naik +165,4% (Youden), Recall naik +436,3%* |
| [`Alternatif-3_Project_Agnostic_Feature_Engineering/`](./Alternatif-3_Project_Agnostic_Feature_Engineering/) | **Project-Agnostic Feature Engineering & Normalisasi Relatif** | ✅ **Selesai** | [`LAPORAN_ALTERNATIF_3.md`](./Alternatif-3_Project_Agnostic_Feature_Engineering/LAPORAN_ALTERNATIF_3.md) — *PR-AUC naik +16,6% (F2), PCA VIF=1.00* |
| [`Alternatif-4_SHAP_Feature_Interpretation/`](./Alternatif-4_SHAP_Feature_Interpretation/) | **Universal Feature Interpretation Menggunakan SHAP** | ✅ **Selesai** | [`LAPORAN_ALTERNATIF_4.md`](./Alternatif-4_SHAP_Feature_Interpretation/LAPORAN_ALTERNATIF_4.md) — *ExecutionTime 26,8%, Spearman rho=0.9704, GPU 107x speedup* |

---

## 🚀 Petunjuk Penggunaan Prompt AI

Setiap subfolder alternatif telah dilengkapi dengan dokumen master prompt:
- `PROMPT_EKSEKUSI_ALTERNATIF_1.md`
- `PROMPT_EKSEKUSI_ALTERNATIF_2.md`
- `PROMPT_EKSEKUSI_ALTERNATIF_3.md`
- `PROMPT_EKSEKUSI_ALTERNATIF_4.md`

Anda dapat menyalin prompt yang ada di masing-masing subfolder ke AI Assistant atau IDE agent untuk mengotomasi implementasi kode, notebook eksperimen, logging metrik, dan pembuatan laporan ilmiah secara mandiri.
