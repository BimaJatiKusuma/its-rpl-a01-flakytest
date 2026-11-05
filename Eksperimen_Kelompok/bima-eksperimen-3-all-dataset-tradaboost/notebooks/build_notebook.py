"""
Script untuk membangun Jupyter Notebook 01_run_all_dataset_tradaboost_experiment.ipynb
"""

import json
from pathlib import Path


def build_notebook():
    notebook_path = Path(__file__).resolve().parent / "01_run_all_dataset_tradaboost_experiment.ipynb"

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 Replikasi Eksperimen 3: Multi-Dataset Cross-Project Flaky Test Prediction\n",
                "## Investigasi TrAdaBoost Target Adaptation pada FlakeFlagger (24 Proyek) & IDoFT (17 Proyek)\n",
                "\n",
                "**Peneliti**: Bima Jati Kusuma (Kelompok A01 - Rekayasa Perangkat Lunak A Magister Teknik Informatika ITS)  \n",
                "**Rujukan Paper**:  \n",
                "1. Alshammari et al. (ICSE 2021) — *FlakeFlagger: Predicting Flakiness Without Rerunning Tests*  \n",
                "2. Afeltra et al. (IEEE Access 2024) — *A Large-Scale Empirical Investigation Into Cross-Project Flaky Test Prediction*  \n",
                "3. Dai et al. (ICML 2007) — *Boosting for Transfer Learning (TrAdaBoost)*  \n",
                "\n",
                "---\n",
                "\n",
                "### 🎯 Tujuan Eksperimen\n",
                "Mengevaluasi secara empiris dampak penyuntikan porsi data berlabel target (Baseline Zero-Shot 0%, TrAdaBoost 25%, 50%, dan 75%) terhadap penanganan *domain shift* dan mitigasi *false alarms* pada dua benchmark independen:\n",
                "- **FlakeFlagger**: 24 Proyek Java, 22.236 test cases, 811 flaky tests (3,65% flakiness rate).\n",
                "- **IDoFT**: 17 Proyek Java, 12.626 test cases, 264 flaky tests (2,09% flakiness rate).\n",
                "Evaluasi menggunakan protokol **Repeated LOPO-CV (5 Independent Seeds: [42, 100, 2024, 777, 999])**."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Setup Environment & Import Dependencies\n",
                "import os\n",
                "import sys\n",
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Menambahkan src ke sys.path\n",
                "src_dir = Path.cwd().parent / 'src'\n",
                "if str(src_dir) not in sys.path:\n",
                "    sys.path.insert(0, str(src_dir))\n",
                "\n",
                "from data_loader import load_dataset, get_dataset_profile, PREDICTIVE_FEATURES\n",
                "from consistency_analyzer import compute_consistency_analysis\n",
                "from dataset_profiler import compute_project_capability_profile, compute_four_aggregation_schemes\n",
                "from statistical_tests import run_statistical_tests\n",
                "\n",
                "print('Semua dependensi dan modul berhasil di-load.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Pemuatan & Profiling Demografi Dua Dataset Benchmark"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df_ff = load_dataset('flakeflagger')\n",
                "df_id = load_dataset('idoft')\n",
                "\n",
                "prof_ff = get_dataset_profile(df_ff, 'flakeflagger')\n",
                "prof_id = get_dataset_profile(df_id, 'idoft')\n",
                "\n",
                "print(f\"[FLAKEFLAGGER] Total Baris: {prof_ff['total_instances']:,} | Proyek: {prof_ff['n_projects']} | Flaky: {prof_ff['total_flaky']} ({prof_ff['overall_flakiness_pct']}%)\")\n",
                "print(f\"[IDOFT]        Total Baris: {prof_id['total_instances']:,} | Proyek: {prof_id['n_projects']} | Flaky: {prof_id['total_flaky']} ({prof_id['overall_flakiness_pct']}%)\")\n",
                "print(f\"Fitur Prediktif: {len(PREDICTIVE_FEATURES)} fitur FlakeFlagger (Zero NaN).\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Ringkasan Kinerja Utama: 4 Skema Agregasi Matematis F1-Score\n",
                "Menampilkan komparasi performa antara Baseline Zero-Shot vs Variasi Porsi TrAdaBoost (25%, 50%, 75%) vs Within-Project."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "aggr_csv = Path.cwd().parent / 'results' / 'summary' / 'macro_vs_weighted_f1_summary.csv'\n",
                "df_aggr = pd.read_csv(aggr_csv)\n",
                "\n",
                "display_cols = ['dataset', 'treatment', 'unweighted_macro_f1', 'active_macro_f1', 'flaky_weighted_f1', 'global_micro_f1']\n",
                "df_summary_disp = df_aggr[display_cols]\n",
                "df_summary_disp"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Analisis Konsistensi Stokastik (Variabilitas 5 Random Seeds)\n",
                "Mengukur apakah peningkatan performa TrAdaBoost stabil melintasi berbagai seed acak ($CV\\%$ dan $CI_{95}$)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "const_csv = Path.cwd().parent / 'results' / 'summary' / 'consistency_stability_analysis.csv'\n",
                "df_const = pd.read_csv(const_csv)\n",
                "\n",
                "df_const[['dataset', 'treatment', 'mean_macro_f1', 'std_macro_f1', 'cv_macro_f1_pct', 'ci95_margin_f1', 'stability_tier']]"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Pengujian Signifikansi Statistik (Wilcoxon Signed-Rank, Cliff's Delta, & Friedman)\n",
                "Memverifikasi secara formal apakah lonjakan TrAdaBoost signifikan secara statistik ($p < 0.05$)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "stat_csv = Path.cwd().parent / 'results' / 'summary' / 'statistical_tests_results.csv'\n",
                "df_stat = pd.read_csv(stat_csv)\n",
                "\n",
                "print('=== HASIL UJI SIGNIFIKANSI WILCOXON & CLIFF\\'S DELTA TERHADAP BASELINE ZERO-SHOT ===')\n",
                "df_stat[['dataset', 'comparison', 'mean_baseline', 'mean_comparison', 'mean_difference', 'p_value', 'is_significant_p05', 'cliffs_delta', 'cliffs_magnitude']]"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Visualisasi Publikasi Ilmiah (300 DPI)\n",
                "Menampilkan 6 grafik visualisasi yang merangkum keseluruhan temuan empiris."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from IPython.display import Image, display\n",
                "plots_dir = Path.cwd().parent / 'results' / 'plots'\n",
                "\n",
                "print('1. Progresi Metrik Seiring Bertambahnya Porsi Data Target:')\n",
                "display(Image(str(plots_dir / 'side_by_side_tradaboost_progression.png')))\n",
                "\n",
                "print('2. Distribusi Boxplot Lintas Proyek:')\n",
                "display(Image(str(plots_dir / 'boxplot_tradaboost_portions_both_datasets.png')))\n",
                "\n",
                "print('3. Heatmap F1-Score 24 Proyek FlakeFlagger:')\n",
                "display(Image(str(plots_dir / 'heatmap_flakeflagger_tradaboost_f1.png')))\n",
                "\n",
                "print('4. Heatmap F1-Score 17 Proyek IDoFT:')\n",
                "display(Image(str(plots_dir / 'heatmap_idoft_tradaboost_f1.png')))\n",
                "\n",
                "print('5. Scatter Plot Flakiness Rate vs Delta Peningkatan F1:')\n",
                "display(Image(str(plots_dir / 'flakiness_rate_vs_target_benefit_scatter.png')))\n",
                "\n",
                "print('6. Perbandingan 4 Skema Agregasi Matematis:')\n",
                "display(Image(str(plots_dir / 'radar_or_bar_comparison_treatments.png')))"
            ]
        }
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.14.8"
            },
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Notebook berhasil dibangun: {notebook_path}")


if __name__ == "__main__":
    build_notebook()
