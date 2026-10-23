"""
Skrip Pembangun Jupyter Notebook: 01_run_afeltra_idoft_experiment.ipynb.
Membangun notebook interaktif yang terstruktur secara akademis sesuai standar ACM/SIGSOFT.
Rujukan: Afeltra et al. (IEEE Access 2024), Alshammari et al. (ICSE 2021).
"""

import json
from pathlib import Path

cells = []

def make_markdown_cell(source: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

def make_code_cell(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }

# 1. Header Notebook
cells.append(make_markdown_cell("""# 🔬 Replikasi Eksperimen 3: One-on-One & Repeated LOPO-CV under Afeltra Paper Concept
## Evaluasi Empiris Komparatif Lintas Repositori Java IDoFT Benchmark
**Peneliti**: Bima Jati Kusuma (Kelompok A01 — Magister Teknik Informatika ITS)  
**Mata Kuliah**: Rekayasa Perangkat Lunak Lanjut (RPL A)  
**Rujukan Primer**:
1. **Afeltra et al. (IEEE Access 2024)** — *A Large-Scale Empirical Investigation Into Cross-Project Flaky Test Prediction*.
2. **Alshammari et al. (ICSE 2021)** — *FlakeFlagger: Predicting Flakiness Without Rerunning Tests*.

---

### 📋 Ikhtisar Penelitian:
Notebook ini mereplikasi metodologi **Eksperimen 3 IDoFT Dataset** dengan mengadopsi **seluruh perlakuan dan intervensi konsep paper Afeltra et al. (IEEE Access 2024)**:
1. **Skema 1: One-on-One Pairwise Cross-Project Transfer Matrix**:
   Evaluasi murni transferabilitas 1-on-1 dari 1 proyek sumber tunggal ($P_{src}$) ke 1 proyek target tunggal ($P_{tgt}$) untuk seluruh 156 kombinasi pasangan proyek aktif di IDoFT ($13 \times 12$ pasangan).
2. **Skema 2: Repeated Leave-One-Project-Out Cross-Validation (LOPO-CV)**:
   Evaluasi berulang (5 *independent random seeds*: $[42, 100, 2024, 777, 999]$) pada 17 proyek IDoFT Java membandingkan 6 perlakuan Afeltra:
   - `T1_Baseline`: Random Forest / XGBoost tanpa transfer learning (MinMaxScaler + Random Under-Sampling).
   - `T2_Burak_Filter`: Instance-based filtering berbasis $k$-NN ($k=10$) pada manifold jarak Euclidean terstandarisasi.
   - `T3_TrAdaBoost_25`: Transfer learning TrAdaBoost dengan 25% target adaptation data (75% evaluation).
   - `T4_TrAdaBoost_50`: Transfer learning TrAdaBoost dengan 50% target adaptation data (50% evaluation).
   - `T5_TrAdaBoost_75`: Transfer learning TrAdaBoost dengan 75% target adaptation data (25% evaluation - Konfigurasi Terbaik Afeltra 2024).
   - `T6_Within_Project`: Benchmark batas atas teoritis (5-Fold Stratified Cross-Validation pada data proyek target)."""))

# 2. Verifikasi Lingkungan & GPU
cells.append(make_markdown_cell("## 1. Verifikasi GPU NVIDIA GeForce RTX 4050 & Lingkungan Eksekusi"))
cells.append(make_code_cell("""import sys
from pathlib import Path
src_dir = str(Path("../src").resolve())
if src_dir not in sys.path:
    sys.path.append(src_dir)

import check_gpu
# Verifikasi deteksi driver dan CUDA
print("Environment Python:", sys.executable)"""))

# 3. Ingestion Dataset
cells.append(make_markdown_cell("## 2. Ingestion & Verifikasi Dataset IDoFT (17 Proyek, 12.626 Baris)"))
cells.append(make_code_cell("""from data_loader import load_idoft_dataset, get_active_and_zero_flaky_projects
df, feature_cols, target_col = load_idoft_dataset()
active_projs, zero_projs = get_active_and_zero_flaky_projects(df)

print(f"Total Baris Data        : {len(df):,}")
print(f"Total Proyek Java       : {df['project_clean'].nunique()} (13 Aktif, 4 Zero-Flaky)")
print(f"Fitur Prediktif (23)    : {len(feature_cols)} fitur FlakeFlagger")
print(f"Total Flaky Test Riil   : {df[target_col].sum():,} ({df[target_col].mean()*100:.2f}%)")
print(f"Daftar Proyek Aktif     : {active_projs}")
print(f"Daftar Proyek Zero Flaky: {zero_projs}")"""))

# 4. Skema 1 Pairwise One-on-One
cells.append(make_markdown_cell("""## 3. Hasil Skema 1: One-on-One Pairwise Cross-Project Transfer Matrix
Evaluasi 1-on-1 dari setiap proyek sumber ke setiap proyek target (156 pasangan proyek aktif)."""))
cells.append(make_code_cell("""import pandas as pd
pairwise_matrix_f1 = pd.read_csv("../results/matrix_pairwise_one_on_one_f1.csv", index_col=0)
print("Matriks F1-Score One-on-One Transfer (Source pada Baris -> Target pada Kolom):")
display(pairwise_matrix_f1)"""))

# 5. Skema 2 Repeated LOPO-CV
cells.append(make_markdown_cell("""## 4. Hasil Skema 2: Repeated LOPO-CV under Afeltra Concept Treatments
Tabel Ringkasan Konsistensi Metrik (Mean, Std, CV%, CI95 Margin) lintas 5 seeds."""))
cells.append(make_code_cell("""df_consistency = pd.read_csv("../results/consistency_stability_analysis.csv")
display(df_consistency[[
    'Model', 'Treatment', 'Mean_F1_Active_13', 'Std_F1', 'CV_F1_%',
    'Mean_Recall', 'Mean_Precision', 'Mean_PR_AUC', 'Stability_Tier'
]])"""))

# 6. Matriks Master Treatment vs Dataset
cells.append(make_markdown_cell("""## 5. Matriks Kemampuan Treatment vs 17 Proyek Dataset IDoFT
Pemetaan F1-Score per proyek untuk setiap perlakuan konsep Afeltra."""))
cells.append(make_code_cell("""df_matrix = pd.read_csv("../results/matrix_project_vs_afeltra_treatments_f1.csv", index_col=[0, 1])
display(df_matrix)"""))

# 7. Dekonstruksi Skema Agregasi
cells.append(make_markdown_cell("""## 6. Dekonstruksi Matematis Rendahnya F1 Makro (RQ2)
Komparasi Unweighted Macro 17 vs Active 13 vs Flaky-Weighted vs Global Micro F1."""))
cells.append(make_code_cell("""df_agg = pd.read_csv("../results/macro_vs_weighted_f1_explanation.csv")
display(df_agg)"""))

# 8. Uji Signifikansi Statistik
cells.append(make_markdown_cell("""## 7. Uji Signifikansi Statistik: Wilcoxon Signed-Rank Test & Cliff's Delta"""))
cells.append(make_code_cell("""df_wilcoxon = pd.read_csv("../results/statistical_wilcoxon_cliffs_delta.csv")
display(df_wilcoxon)"""))

# 9. Visualisasi Grafik Publikasi
cells.append(make_markdown_cell("""## 8. Galeri Visualisasi Hasil Penelitian"""))
cells.append(make_code_cell("""from IPython.display import Image, display

plots = [
    ("../results/plots/heatmap_pairwise_one_on_one_f1.png", "Heatmap 1: One-on-One Pairwise Transfer Matrix (F1-Score)"),
    ("../results/plots/heatmap_afeltra_treatments_f1.png", "Heatmap 2: Perbandingan F1-Score per Proyek vs Intervensi Afeltra"),
    ("../results/plots/boxplot_afeltra_metrics.png", "Boxplot: Distribusi Metrik F1, Recall, Precision, PR-AUC Antar Perlakuan"),
    ("../results/plots/scatter_flaky_rate_vs_f1.png", "Scatter: Korelasi Flaky Rate vs Cross-Project F1-Score"),
    ("../results/plots/top_transfer_donors.png", "Barplot: Peringkat Proyek Donor Terbaik pada Transfer One-on-One")
]

for p_path, p_title in plots:
    print(f"\\n=== {p_title} ===")
    display(Image(filename=p_path))"""))

def build_notebook_file():
    current_dir = Path(__file__).resolve().parent
    out_ipynb = current_dir / "01_run_afeltra_idoft_experiment.ipynb"
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.14.8"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(out_ipynb, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"[OK] Notebook berhasil dibuat: {out_ipynb}")

if __name__ == "__main__":
    build_notebook_file()
