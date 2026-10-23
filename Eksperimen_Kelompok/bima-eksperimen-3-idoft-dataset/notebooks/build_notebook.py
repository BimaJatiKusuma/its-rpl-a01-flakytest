import json
from pathlib import Path

cells = []

def make_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

def make_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }

# Cell 1: Markdown Header
cells.append(make_markdown_cell("""# 🔬 Replikasi Eksperimen 3: Repeated LOPO-CV & Analisis Kemampuan Dataset IDoFT
**Kelompok Peneliti**: Kelompok A01 — RPL A (Magister Teknik Informatika ITS)  
**Dataset Acuan**: IDoFT (Illinois Dataset of Flaky Tests) Java Projects (`pr-data.csv` & `gr-data.csv`) aligned with FlakeFlagger 23 Features  
**Rujukan Paper**: Afeltra et al. (IEEE Access 2024) — *A Large-Scale Empirical Investigation Into Cross-Project Flaky Test Prediction*  
**Hardware Target**: NVIDIA GeForce RTX 4050 Laptop GPU (CUDA hist) & CPU Multi-threading  
**Metodologi**: Repeated Leave-One-Project-Out Cross-Validation (5 Repetisi x 24 Pipelines x 17 Folds = **2,040 Model Folds**).

---

## 📋 Ikhtisar & Latar Belakang Replikasi
Eksperimen ini mereplikasi metodologi pengujian berulang (*repeated LOPO-CV*) dan analisis kemampuan dataset individual dari **Eksperimen 3**, namun menggunakan **Dataset IDoFT** (sebagaimana digunakan oleh paper Afeltra et al., 2024).

Dataset mencakup **17 proyek Java** (12.626 baris pengujian, 264 flaky test ground truth riil, tingkat ketimpangan ekstrem 2,09%).
Sebanyak 24 konfigurasi pipeline (3 Feature Scaling x 4 Imbalance Techniques x 2 Machine Learning Classifiers) diuji melintasi 5 *random seed* independen ([42, 100, 2024, 777, 999])."""))

# Cell 2: Header GPU
cells.append(make_markdown_cell("## 1. Verifikasi GPU NVIDIA RTX 4050 & Lingkungan ML"))

# Cell 3: Code GPU
cells.append(make_code_cell("""import sys
from pathlib import Path

src_dir = str(Path("../src").resolve())
if src_dir not in sys.path:
    sys.path.append(src_dir)

import check_gpu
check_gpu.check_nvidia_smi()
check_gpu.check_python_environment()
check_gpu.check_xgboost_gpu()"""))

# Cell 4: Header Ingestion
cells.append(make_markdown_cell("## 2. Ingestion & Verifikasi Dataset IDoFT (17 Proyek, 12.626 Baris)"))

# Cell 5: Code Ingestion
cells.append(make_code_cell("""from data_loader import load_idoft_dataset
df, features, projects = load_idoft_dataset()

print(f"Total Baris: {len(df):,} | Proyek: {len(projects)} | Total Fitur Prediktif: {len(features)}")
print(f"Flaky Tests IDoFT: {df['flaky'].sum():,} ({df['flaky'].mean()*100:.2f}%)")
print("Daftar Proyek IDoFT:", projects)"""))

# Cell 6: Header Eksekusi
cells.append(make_markdown_cell("## 3. Eksekusi Repeated LOPO-CV (24 Pipelines x 5 Seeds x 17 Folds = 2,040 Model Evaluasi)"))

# Cell 7: Code Eksekusi
cells.append(make_code_cell("""from repeated_lopo_runner import run_repeated_lopo_experiments

# Menjalankan evaluasi 2.040 folds dengan GPU Acceleration
results = run_repeated_lopo_experiments(use_gpu=True)"""))

# Cell 8: Header Konsistensi
cells.append(make_markdown_cell("## 4. Analisis Konsistensi & Stabilitas Pipeline Lintas Seeds"))

# Cell 9: Code Konsistensi
cells.append(make_code_cell("""from consistency_analyzer import analyze_pipeline_consistency
import pandas as pd

df_cons = analyze_pipeline_consistency()
display_cols = [
    'Pipeline_ID', 'Scaling', 'Imbalance', 'Model',
    'Mean_F1_Mean', 'Mean_F1_Active_13_Mean', 'Mean_F1_Std', 'Mean_F1_CV_%',
    'Mean_Recall_Mean', 'Mean_Precision_Mean', 'Mean_PR_AUC_Mean'
]
df_cons[display_cols].head(10)"""))

# Cell 10: Header Profiling
cells.append(make_markdown_cell("## 5. Profiling Kemampuan 17 Dataset Proyek & Root Cause Analysis"))

# Cell 11: Code Profiling
cells.append(make_code_cell("""from dataset_profiler import profile_all_datasets

df_profile, analysis_dict = profile_all_datasets()
df_profile[[
    'Target_Project', 'Flaky_Count', 'Test_Size', 'Flaky_Rate_%',
    'Overall_Mean_F1', 'Overall_Std_F1', 'Best_Pipeline', 'Best_Pipeline_F1', 'Tier_Code'
]].head(17)"""))

# Cell 12: Header Statistik
cells.append(make_markdown_cell("## 6. Uji Signifikansi Statistik (Friedman & Nemenyi Post-Hoc)"))

# Cell 13: Code Statistik
cells.append(make_code_cell("""from statistical_tests import run_repeated_statistical_analysis

df_stat_f1 = run_repeated_statistical_analysis(metric_col="F1")
df_stat_prauc = run_repeated_statistical_analysis(metric_col="PR_AUC")
print("Top 5 Peringkat Friedman Nemenyi (F1):")
print(df_stat_f1.head(5).to_string(index=False))"""))

# Cell 14: Header Visualisasi
cells.append(make_markdown_cell("## 7. Pembuatan Visualisasi Publikasi Ilmiah 300 DPI"))

# Cell 15: Code Visualisasi
cells.append(make_code_cell("""from visualizer import plot_all_experiment_3_figures
plot_all_experiment_3_figures()"""))

notebook_dict = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_nb = Path("Eksperimen_Kelompok/bima-eksperimen-3-idoft-dataset/notebooks/01_run_experiment_3_idoft_repeated_lopo.ipynb")
out_nb.parent.mkdir(parents=True, exist_ok=True)
with open(out_nb, "w", encoding="utf-8") as f:
    json.dump(notebook_dict, f, indent=1)
print("Jupyter Notebook successfully created at:", out_nb)
