"""
Modul Visualisasi Publikasi Ilmiah untuk Eksperimen 2.
Menghasilkan:
1. Boxplot Metrik per Faktor (Scaling, Imbalance, Classifier).
2. Heatmap Mean F1-Score & PR-AUC pada seluruh 24 Pipelines.
3. Diagram Peringkat Rata-rata (Average Ranks Nemenyi).
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


def plot_rq_factor_boxplots(summary_csv_path: str = None, plots_dir: str = None):
    """Membuat boxplot per faktor (Scaling, Imbalance, Model) untuk F1 dan PR-AUC."""
    if summary_csv_path is None:
        summary_csv_path = str(Path(__file__).resolve().parent.parent / "results" / "summary_metrics_24_pipelines.csv")
    if plots_dir is None:
        plots_dir = str(Path(__file__).resolve().parent.parent / "results" / "plots")

    p_dir = Path(plots_dir)
    p_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(summary_csv_path)

    # 1. Boxplot untuk Mean F1-Score
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Eksperimen 2: Pengaruh Faktor Eksperimen terhadap F1-Score Flaky Test", fontsize=14, weight="bold")

    sns.boxplot(ax=axes[0], data=df, x="Scaling", y="Mean_F1", palette="Pastel1")
    axes[0].set_title("Faktor Scaling (S1: Std, S2: MinMax, S3: Robust)")
    axes[0].set_ylabel("Mean F1-Score (Flaky)")
    axes[0].set_xlabel("Scaling Strategy")

    sns.boxplot(ax=axes[1], data=df, x="Imbalance", y="Mean_F1", palette="Set2")
    axes[1].set_title("Faktor Imbalance (B1: None, B2: SMOTE, B3: RUS, B4: Weight)")
    axes[1].set_ylabel("Mean F1-Score (Flaky)")
    axes[1].set_xlabel("Imbalance Strategy")

    sns.boxplot(ax=axes[2], data=df, x="Model", y="Mean_F1", palette="muted")
    axes[2].set_title("Faktor Classifier (M1: RF vs M2: XGBoost GPU)")
    axes[2].set_ylabel("Mean F1-Score (Flaky)")
    axes[2].set_xlabel("Classifier")

    plt.tight_layout()
    save_path_f1 = p_dir / "boxplot_3_factors_f1.png"
    plt.savefig(save_path_f1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Boxplot F1 tersimpan di: {save_path_f1}")

    # 2. Boxplot untuk Mean PR-AUC
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Eksperimen 2: Pengaruh Faktor Eksperimen terhadap PR-AUC Flaky Test", fontsize=14, weight="bold")

    sns.boxplot(ax=axes[0], data=df, x="Scaling", y="Mean_PR_AUC", palette="Pastel1")
    axes[0].set_title("Faktor Scaling (S1: Std, S2: MinMax, S3: Robust)")
    axes[0].set_ylabel("Mean PR-AUC")
    axes[0].set_xlabel("Scaling Strategy")

    sns.boxplot(ax=axes[1], data=df, x="Imbalance", y="Mean_PR_AUC", palette="Set2")
    axes[1].set_title("Faktor Imbalance (B1: None, B2: SMOTE, B3: RUS, B4: Weight)")
    axes[1].set_ylabel("Mean PR-AUC")
    axes[1].set_xlabel("Imbalance Strategy")

    sns.boxplot(ax=axes[2], data=df, x="Model", y="Mean_PR_AUC", palette="muted")
    axes[2].set_title("Faktor Classifier (M1: RF vs M2: XGBoost GPU)")
    axes[2].set_ylabel("Mean PR-AUC")
    axes[2].set_xlabel("Classifier")

    plt.tight_layout()
    save_path_prauc = p_dir / "boxplot_3_factors_prauc.png"
    plt.savefig(save_path_prauc, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Boxplot PR-AUC tersimpan di: {save_path_prauc}")


def plot_pipeline_heatmap(summary_csv_path: str = None, plots_dir: str = None):
    """Membuat heatmap performa 24 pipeline (Mean F1 & Mean PR-AUC)."""
    if summary_csv_path is None:
        summary_csv_path = str(Path(__file__).resolve().parent.parent / "results" / "summary_metrics_24_pipelines.csv")
    if plots_dir is None:
        plots_dir = str(Path(__file__).resolve().parent.parent / "results" / "plots")

    p_dir = Path(plots_dir)
    p_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(summary_csv_path)

    df["Config"] = df["Scaling"] + "_" + df["Imbalance"]

    # Heatmap F1
    pivot_f1 = df.pivot(index="Config", columns="Model", values="Mean_F1")
    plt.figure(figsize=(8, 7))
    sns.heatmap(pivot_f1, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Mean F1-Score'})
    plt.title("Heatmap Mean F1-Score: Preprocessing Config vs Classifier", weight="bold")
    plt.ylabel("Preprocessing (Scaling_Imbalance)")
    plt.xlabel("Classifier Model")
    save_path_f1 = p_dir / "heatmap_pipeline_f1.png"
    plt.savefig(save_path_f1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Heatmap F1 tersimpan di: {save_path_f1}")

    # Heatmap PR-AUC
    pivot_prauc = df.pivot(index="Config", columns="Model", values="Mean_PR_AUC")
    plt.figure(figsize=(8, 7))
    sns.heatmap(pivot_prauc, annot=True, fmt=".3f", cmap="Blues", cbar_kws={'label': 'Mean PR-AUC'})
    plt.title("Heatmap Mean PR-AUC: Preprocessing Config vs Classifier", weight="bold")
    plt.ylabel("Preprocessing (Scaling_Imbalance)")
    plt.xlabel("Classifier Model")
    save_path_prauc = p_dir / "heatmap_pipeline_prauc.png"
    plt.savefig(save_path_prauc, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Heatmap PR-AUC tersimpan di: {save_path_prauc}")


def plot_nemenyi_ranks(results_dir: str = None, plots_dir: str = None):
    """Membuat visualisasi average ranks dari uji Nemenyi jika file hasil tersedia."""
    if results_dir is None:
        results_dir = str(Path(__file__).resolve().parent.parent / "results")
    if plots_dir is None:
        plots_dir = str(Path(__file__).resolve().parent.parent / "results" / "plots")

    r_dir = Path(results_dir)
    p_dir = Path(plots_dir)
    p_dir.mkdir(parents=True, exist_ok=True)

    for metric in ["f1", "prauc"]:
        fpath = r_dir / f"friedman_nemenyi_{metric}_results.csv"
        if fpath.exists():
            df_ranks = pd.read_csv(fpath)
            plt.figure(figsize=(10, 8))
            df_sorted = df_ranks.sort_values(by="Average_Rank", ascending=True)
            y_pos = np.arange(len(df_sorted))
            plt.barh(y_pos, df_sorted["Average_Rank"], color="#4C72B0", edgecolor="black", alpha=0.85)
            plt.yticks(y_pos, df_sorted["Pipeline"])
            plt.gca().invert_yaxis()
            plt.xlabel("Average Rank across 24 Target Projects (Lower is Better)")
            cd_val = df_sorted["CD_Threshold"].iloc[0] if "CD_Threshold" in df_sorted.columns else 0.0
            plt.title(f"Nemenyi Post-Hoc Average Ranks ({metric.upper()})\nCritical Difference (CD) = {cd_val:.2f}", weight="bold")
            plt.axvline(x=df_sorted["Average_Rank"].iloc[0] + cd_val, color="red", linestyle="--", label=f"Rank 1 + CD ({cd_val:.2f})")
            plt.legend(loc="lower right")
            plt.tight_layout()
            out_file = p_dir / f"nemenyi_ranks_{metric}.png"
            plt.savefig(out_file, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[SAVED] Nemenyi Ranks {metric.upper()} tersimpan di: {out_file}")


def generate_all_visualizations(summary_csv_path: str = None, plots_dir: str = None):
    """Menjalankan seluruh rutin pembuatan grafik."""
    plot_rq_factor_boxplots(summary_csv_path, plots_dir)
    plot_pipeline_heatmap(summary_csv_path, plots_dir)
    plot_nemenyi_ranks(plots_dir=plots_dir)
    print("[VISUALIZER] Semua visualisasi berhasil diproduksi!")


if __name__ == "__main__":
    generate_all_visualizations()

