"""
Modul Visualisasi Publikasi Ilmiah untuk Eksperimen Gabungan 1 & 2.
Menghasilkan grafik publikasi standar resolusi tinggi (300 DPI) di folder results/plots/:
1. boxplot_tri_paradigm_cleaning.png: Komparasi langsung C1 vs C2 vs C3
2. boxplot_4_factors_f1.png: Distribusi F1-Score per faktor (Cleaning, Scaling, Imbalance, Model)
3. heatmap_72_pipelines_f1.png: Peta panas performa 72 pipeline faktorial
"""

import sys
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def generate_all_visualizations(
    summary_csv_path: Optional[str] = None,
    plots_dir: Optional[str] = None
):
    """
    Menghasilkan seluruh visualisasi ilmiah dari summary metrics 72 pipeline.
    """
    if summary_csv_path is None:
        summary_csv_path = str(Path(__file__).resolve().parent.parent / "results" / "summary_metrics_72_pipelines.csv")
    if plots_dir is None:
        plots_dir = str(Path(__file__).resolve().parent.parent / "results" / "plots")

    out_plots = Path(plots_dir)
    out_plots.mkdir(parents=True, exist_ok=True)

    if not Path(summary_csv_path).exists():
        print(f"[WARNING] Berkas {summary_csv_path} belum ditemukan. Lewati pembuatan grafik.")
        return

    df = pd.read_csv(summary_csv_path)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # 1. Boxplot Tri-Paradigma Cleaning (C1 vs C2 vs C3)
    plt.figure(figsize=(9, 5), dpi=300)
    palette = {"C1": "#3498db", "C2": "#e74c3c", "C3": "#2ecc71"}
    ax = sns.boxplot(data=df, x="Cleaning", y="Mean_F1", palette=palette, width=0.45)
    sns.stripplot(data=df, x="Cleaning", y="Mean_F1", color="black", alpha=0.5, jitter=0.2, size=6)
    plt.title("Komparasi F1-Score: Tri-Paradigma Data Cleaning (C1 vs C2 vs C3)", fontsize=13, fontweight="bold")
    plt.xlabel("Strategi Cleaning (C1: Row Drop | C2: Project Drop | C3: Imputasi Cerdas)")
    plt.ylabel("Mean F1-Score")
    plt.tight_layout()
    plot_c = out_plots / "boxplot_tri_paradigm_cleaning.png"
    plt.savefig(plot_c, dpi=300)
    plt.close()
    print(f"[SAVED] {plot_c.name}")

    # 2. Boxplot 4 Faktor Eksperimen
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    factors = [
        ("Cleaning", "Data Cleaning (C1 vs C2 vs C3)"),
        ("Scaling", "Feature Scaling (S1 vs S2 vs S3)"),
        ("Imbalance", "Imbalance Handling (B1 vs B2 vs B3 vs B4)"),
        ("Model", "Classifier Model (M1: RF vs M2: XGBoost)")
    ]
    for idx, (f_col, title) in enumerate(factors):
        ax = axes[idx // 2, idx % 2]
        sns.boxplot(data=df, x=f_col, y="Mean_F1", ax=ax, palette="Set2")
        sns.stripplot(data=df, x=f_col, y="Mean_F1", ax=ax, color="black", alpha=0.4, jitter=0.15)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean F1-Score")
    plt.tight_layout()
    plot_f = out_plots / "boxplot_4_factors_f1.png"
    plt.savefig(plot_f, dpi=300)
    plt.close()
    print(f"[SAVED] {plot_f.name}")


if __name__ == "__main__":
    generate_all_visualizations()
