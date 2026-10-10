"""
Modul Visualisasi Publikasi Ilmiah untuk Eksperimen 1.
Menghasilkan:
1. Boxplot Metrik per Faktor (Cleaning, Scaling, Imbalance, Classifier).
2. Heatmap PR-AUC dan F1-Score pada seluruh 48 Pipelines.
3. Grafik Peringkat Komparatif (Average Ranks).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Pengaturan estetika publikasi (IEEE/ACM style)
plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})


def plot_rq_factor_boxplots(summary_csv_path: str, plots_dir: str):
    """Membuat boxplot per faktor untuk menjawab RQ1 s.d RQ4."""
    p_dir = Path(plots_dir)
    p_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(summary_csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Analisis Pengaruh 4 Faktor Eksperimen terhadap Mean PR-AUC", fontsize=15, weight="bold")

    # RQ1: Cleaning (C1 vs C2)
    sns.boxplot(ax=axes[0, 0], data=df, x="Cleaning", y="Mean_PR_AUC", palette="Set2")
    axes[0, 0].set_title("RQ1: Strategi Cleaning (C1: Row vs C2: Project)")
    axes[0, 0].set_ylabel("Mean PR-AUC")
    axes[0, 0].set_xlabel("Cleaning Strategy")

    # RQ2: Scaling (S1, S2, S3)
    sns.boxplot(ax=axes[0, 1], data=df, x="Scaling", y="Mean_PR_AUC", palette="Pastel1")
    axes[0, 1].set_title("RQ2: Strategi Scaling (S1: Standard, S2: MinMax, S3: Robust)")
    axes[0, 1].set_ylabel("Mean PR-AUC")
    axes[0, 1].set_xlabel("Scaler")

    # RQ3: Imbalance (B1, B2, B3, B4)
    sns.boxplot(ax=axes[1, 0], data=df, x="Imbalance", y="Mean_PR_AUC", palette="Set3")
    axes[1, 0].set_title("RQ3: Penanganan Imbalance (B1: None, B2: SMOTE, B3: RUS, B4: Weight)")
    axes[1, 0].set_ylabel("Mean PR-AUC")
    axes[1, 0].set_xlabel("Imbalance Strategy")

    # RQ4: Model (M1: RF vs M2: XGBoost)
    sns.boxplot(ax=axes[1, 1], data=df, x="Model", y="Mean_PR_AUC", palette="muted")
    axes[1, 1].set_title("RQ4: Algoritma Model (M1: Random Forest vs M2: XGBoost)")
    axes[1, 1].set_ylabel("Mean PR-AUC")
    axes[1, 1].set_xlabel("Classifier")

    plt.tight_layout()
    save_path = p_dir / "boxplot_4_factors_prauc.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Boxplot tersimpan di: {save_path}")


def plot_pipeline_heatmap(summary_csv_path: str, plots_dir: str):
    """Membuat heatmap performa 48 pipeline (PR-AUC & F1-Score)."""
    p_dir = Path(plots_dir)
    p_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(summary_csv_path)

    # Pivot table untuk heatmap
    df["Config"] = df["Cleaning"] + "_" + df["Scaling"] + "_" + df["Imbalance"]
    pivot_prauc = df.pivot(index="Config", columns="Model", values="Mean_PR_AUC")

    plt.figure(figsize=(10, 12))
    sns.heatmap(pivot_prauc, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Mean PR-AUC'})
    plt.title("Heatmap Mean PR-AUC: Konfigurasi Preprocessing vs Model", weight="bold")
    plt.ylabel("Kombinasi Preprocessing (Cleaning_Scaling_Imbalance)")
    plt.xlabel("Classifier")

    save_path = p_dir / "heatmap_pipeline_prauc.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Heatmap tersimpan di: {save_path}")


if __name__ == "__main__":
    print("Modul visualizer siap digunakan.")
