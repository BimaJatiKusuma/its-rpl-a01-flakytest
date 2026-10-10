"""
Modul Visualisasi Publikasi Ilmiah untuk Eksperimen Alternatif 1:
- Boxplot Distribusi F1-Score & PR-AUC (Baseline vs Burak vs TrAdaBoost vs Hybrid)
- Bar Chart Perbandingan Metrik Utama (F1, PR-AUC, Recall, Precision, ROC-AUC, MCC)
- Grafik Komparasi Waktu Eksekusi GPU RTX 4050
- Analisis Gain Performa Per Proyek
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Konfigurasi estetika grafik standar publikasi ACM / IEEE
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


def resolve_default_paths():
    script_dir = Path(__file__).resolve().parent
    res_dir = script_dir.parent / "results"
    folds_path = res_dir / "transfer_learning_folds.csv"
    summary_path = res_dir / "transfer_learning_summary.csv"
    plots_dir = res_dir / "plots"
    return folds_path, summary_path, plots_dir


def plot_comparison_boxplots(folds_csv_path: str = None, plots_dir: str = None):
    """
    Membuat boxplot perbandingan distribusi F1-Score dan PR-AUC across 24 projects.
    """
    def_folds, _, def_plots = resolve_default_paths()
    f_path = Path(folds_csv_path) if folds_csv_path else def_folds
    p_dir = Path(plots_dir) if plots_dir else def_plots
    p_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(f_path)

    name_map = {
        "A_Baseline": "Baseline\n(No Transfer)",
        "B_Burak_Filter": "Burak Filter\n(k-NN Union)",
        "C_TrAdaBoost": "TrAdaBoost\n(Few-Shot)",
        "D_Burak_TrAdaBoost_Hybrid": "Hybrid\n(Burak + TrAda)"
    }
    df["Config_Short"] = df["Config_ID"].map(name_map)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    palette = ["#95a5a6", "#3498db", "#e67e22", "#2ecc71"]

    # 1. Boxplot F1-Score
    sns.boxplot(
        ax=axes[0],
        data=df,
        x="Config_Short",
        y="F1",
        palette=palette,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": 7}
    )
    axes[0].set_title("Distribusi F1-Score (Kelas Flaky)\n[Titik Merah: Nilai Mean]", weight="bold")
    axes[0].set_ylabel("F1-Score")
    axes[0].set_xlabel("Konfigurasi Model Transfer Learning")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # 2. Boxplot PR-AUC
    sns.boxplot(
        ax=axes[1],
        data=df,
        x="Config_Short",
        y="PR_AUC",
        palette=palette,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": 7}
    )
    axes[1].set_title("Distribusi PR-AUC (Average Precision)\n[Titik Merah: Nilai Mean]", weight="bold")
    axes[1].set_ylabel("PR-AUC")
    axes[1].set_xlabel("Konfigurasi Model Transfer Learning")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = p_dir / "boxplot_comparison_f1_prauc.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Boxplot komparasi tersimpan di: {save_path}")


def plot_metric_summary_bars(summary_csv_path: str = None, plots_dir: str = None):
    """
    Membuat diagram batang ringkasan performa 6 metrik utama lintas 4 konfigurasi.
    """
    _, def_sum, def_plots = resolve_default_paths()
    s_path = Path(summary_csv_path) if summary_csv_path else def_sum
    p_dir = Path(plots_dir) if plots_dir else def_plots
    p_dir.mkdir(parents=True, exist_ok=True)

    df_sum = pd.read_csv(s_path)

    metrics = ["F1", "PR_AUC", "Recall", "Precision", "ROC_AUC", "MCC"]
    labels = ["F1-Score", "PR-AUC", "Recall", "Precision", "ROC-AUC", "MCC"]

    configs = df_sum["Config_ID"].tolist()
    names = ["Baseline", "Burak Filter", "TrAdaBoost", "Hybrid (Burak+TrAda)"]

    x = np.arange(len(metrics))
    width = 0.20

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["#7f8c8d", "#2980b9", "#d35400", "#27ae60"]

    for idx, (cid, cname, col) in enumerate(zip(configs, names, colors)):
        row = df_sum[df_sum["Config_ID"] == cid].iloc[0]
        mean_vals = [row[f"Mean_{m}"] for m in metrics]
        std_vals = [row[f"Std_{m}"] for m in metrics]
        ax.bar(x + idx * width, mean_vals, width, yerr=std_vals, capsize=4, label=cname, color=col, alpha=0.9)

    ax.set_title("Perbandingan Metrik Evaluasi Komprehensif (Mean +/- Std Dev)", weight="bold", fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels, fontsize=11, weight="bold")
    ax.set_ylabel("Skor Metrik", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    plt.tight_layout()
    save_path = p_dir / "barchart_metrics_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Bar chart ringkasan metrik tersimpan di: {save_path}")


def plot_execution_time(summary_csv_path: str = None, plots_dir: str = None):
    """
    Membuat grafik efisiensi komputasi akselerasi GPU RTX 4050.
    """
    _, def_sum, def_plots = resolve_default_paths()
    s_path = Path(summary_csv_path) if summary_csv_path else def_sum
    p_dir = Path(plots_dir) if plots_dir else def_plots
    p_dir.mkdir(parents=True, exist_ok=True)

    df_sum = pd.read_csv(s_path)
    names = ["Baseline", "Burak Filter", "TrAdaBoost", "Hybrid (Burak+TrAda)"]

    fig, ax = plt.subplots(figsize=(9, 5))
    times = df_sum["Mean_Train_Time_Sec"].values
    colors = ["#7f8c8d", "#2980b9", "#d35400", "#27ae60"]

    bars = ax.bar(names, times, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}s",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom", weight="bold")

    ax.set_title("Rata-rata Waktu Training per Fold (NVIDIA GeForce RTX 4050)", weight="bold")
    ax.set_ylabel("Waktu Training (Detik)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = p_dir / "gpu_runtime_efficiency.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Grafik waktu training tersimpan di: {save_path}")


def plot_per_project_f1_gain(folds_csv_path: str = None, plots_dir: str = None):
    """
    Membuat visualisasi perbandingan F1-Score per proyek target.
    """
    def_folds, _, def_plots = resolve_default_paths()
    f_path = Path(folds_csv_path) if folds_csv_path else def_folds
    p_dir = Path(plots_dir) if plots_dir else def_plots
    p_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(f_path)

    pivot_f1 = df.pivot(index="Target_Project", columns="Config_ID", values="F1").fillna(0.0)
    projects_with_flaky = df[df["Flaky_in_Test"] > 0]["Target_Project"].unique()
    pivot_filtered = pivot_f1.loc[pivot_f1.index.isin(projects_with_flaky)].copy()

    pivot_filtered["Delta_TrAda_Baseline"] = (
        pivot_filtered["C_TrAdaBoost"] - pivot_filtered["A_Baseline"]
    )
    pivot_filtered = pivot_filtered.sort_values(by="Delta_TrAda_Baseline", ascending=False)

    plt.figure(figsize=(14, 8))
    x = np.arange(len(pivot_filtered))
    width = 0.22

    plt.bar(x - width*1.5, pivot_filtered["A_Baseline"], width, label="Baseline", color="#7f8c8d")
    plt.bar(x - width*0.5, pivot_filtered["B_Burak_Filter"], width, label="Burak Filter", color="#2980b9")
    plt.bar(x + width*0.5, pivot_filtered["C_TrAdaBoost"], width, label="TrAdaBoost", color="#d35400")
    plt.bar(x + width*1.5, pivot_filtered["D_Burak_TrAdaBoost_Hybrid"], width, label="Hybrid", color="#27ae60")

    plt.title("Perbandingan F1-Score Antar Proyek Target (Diurutkan Berdasarkan Gain TrAdaBoost vs Baseline)", weight="bold", fontsize=13)
    plt.xlabel("Proyek Target", weight="bold")
    plt.ylabel("F1-Score (Kelas Flaky)")
    plt.xticks(x, pivot_filtered.index, rotation=45, ha="right", fontsize=9)
    plt.legend(loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = p_dir / "per_project_f1_gain.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Grafik F1 gain per proyek tersimpan di: {save_path}")


def generate_all_plots(folds_csv_path: str = None, summary_csv_path: str = None, plots_dir: str = None):
    """Menjalankan pembuatan seluruh visualisasi secara otomatis."""
    print("\n" + "=" * 80)
    print("[VIZ] MEMBUAT VISUALISASI HASIL PENELITIAN TRANSFER LEARNING")
    print("=" * 80)
    plot_comparison_boxplots(folds_csv_path, plots_dir)
    plot_metric_summary_bars(summary_csv_path, plots_dir)
    plot_execution_time(summary_csv_path, plots_dir)
    plot_per_project_f1_gain(folds_csv_path, plots_dir)
    print("=" * 80)


if __name__ == "__main__":
    generate_all_plots()
