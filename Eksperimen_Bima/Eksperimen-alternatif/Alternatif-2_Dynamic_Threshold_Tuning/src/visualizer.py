"""
Modul Visualisasi Publikasi Ilmiah untuk Eksperimen Alternatif 2:
- Boxplot Distribusi F1-Score (T1 Default vs T2 Max-F1 vs T3 Recall>=70% vs T4 Youden vs T5 Prior)
- Trade-off Precision vs Recall lintas strategi
- Distribusi Nilai Threshold Optimal (tau*) yang Dipelajari
- Kurva Inner-Validation Grid Search (F1, Precision, Recall, FPR vs Threshold)
- Gain Performa F1-Score Per Proyek Target
- Ringkasan Komparasi Metrik (Mean +/- Std Dev)
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

STRATEGY_NAMES = {
    "T1_Default": "T1: Default 0.5",
    "T2_Max_F1": "T2: Max-F1",
    "T3_Recall_Constrained": "T3: Recall>=70%",
    "T4_Youdens_J": "T4: Youden's J",
    "T5_Prior_Shifted": "T5: Prior-Shifted"
}

PALETTE = ["#7f8c8d", "#2980b9", "#27ae60", "#8e44ad", "#e67e22"]


def resolve_default_paths():
    script_dir = Path(__file__).resolve().parent
    res_dir = script_dir.parent / "results"
    folds_path = res_dir / "fold_thresholds.csv"
    summary_path = res_dir / "threshold_comparison_summary.csv"
    plots_dir = res_dir / "plots"
    return folds_path, summary_path, plots_dir


def plot_f1_boxplots(folds_df: pd.DataFrame, plots_dir: Path, model_name: str = "xgboost_gpu"):
    """Boxplot perbandingan distribusi F1-Score lintas 5 strategi threshold."""
    df_mod = folds_df[folds_df["Model"] == model_name].copy()
    df_mod["Strategy_Label"] = df_mod["Strategy_ID"].map(STRATEGY_NAMES)

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        data=df_mod,
        x="Strategy_Label",
        y="F1",
        palette=PALETTE,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": 7}
    )
    plt.title(f"Distribusi F1-Score (Kelas Flaky) Lintas Strategi Threshold\nModel: {model_name} [Titik Merah: Nilai Mean]", weight="bold")
    plt.ylabel("F1-Score", weight="bold")
    plt.xlabel("Strategi Threshold", weight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    save_path = plots_dir / f"boxplot_f1_strategies_{model_name}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Boxplot F1 tersimpan di: {save_path}")


def plot_precision_recall_tradeoff(folds_df: pd.DataFrame, plots_dir: Path, model_name: str = "xgboost_gpu"):
    """Boxplot berdampingan untuk Recall dan Precision guna mengilustrasikan trade-off."""
    df_mod = folds_df[folds_df["Model"] == model_name].copy()
    df_mod["Strategy_Label"] = df_mod["Strategy_ID"].map(STRATEGY_NAMES)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Recall
    sns.boxplot(
        ax=axes[0],
        data=df_mod,
        x="Strategy_Label",
        y="Recall",
        palette=PALETTE,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": 7}
    )
    axes[0].set_title(f"Distribusi Recall (Sensitivity)\n[Titik Merah: Nilai Mean]", weight="bold")
    axes[0].set_ylabel("Recall (Flaky)", weight="bold")
    axes[0].set_xlabel("Strategi Threshold", weight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Precision
    sns.boxplot(
        ax=axes[1],
        data=df_mod,
        x="Strategy_Label",
        y="Precision",
        palette=PALETTE,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": 7}
    )
    axes[1].set_title(f"Distribusi Precision\n[Titik Merah: Nilai Mean]", weight="bold")
    axes[1].set_ylabel("Precision (Flaky)", weight="bold")
    axes[1].set_xlabel("Strategi Threshold", weight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    save_path = plots_dir / f"boxplot_precision_recall_tradeoff_{model_name}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Plot trade-off Recall vs Precision tersimpan di: {save_path}")


def plot_threshold_distribution(folds_df: pd.DataFrame, plots_dir: Path, model_name: str = "xgboost_gpu"):
    """Distribusi nilai ambang batas tau* yang dipelajari pada tiap fold."""
    df_mod = folds_df[folds_df["Model"] == model_name].copy()
    df_mod["Strategy_Label"] = df_mod["Strategy_ID"].map(STRATEGY_NAMES)

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        data=df_mod,
        x="Strategy_Label",
        y="Threshold",
        palette=PALETTE,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": 7}
    )
    # Tambahkan jitter scatter point
    sns.stripplot(data=df_mod, x="Strategy_Label", y="Threshold", color="black", alpha=0.6, jitter=0.2, size=5)

    plt.title(f"Distribusi Nilai Threshold Optimal (tau*) yang Dipelajari per Fold\nModel: {model_name}", weight="bold")
    plt.ylabel("Nilai Ambang Batas Probabilitas (tau*)", weight="bold")
    plt.xlabel("Strategi Threshold", weight="bold")
    plt.axhline(0.5, color="red", linestyle=":", label="Default 0.5 Reference")
    plt.axhline(0.0365, color="orange", linestyle=":", label="Empirical Prior (~0.0365)")
    plt.legend(loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    save_path = plots_dir / f"distribution_learned_thresholds_{model_name}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Plot distribusi threshold tersimpan di: {save_path}")


def plot_metrics_summary_bars(summary_df: pd.DataFrame, plots_dir: Path, model_name: str = "xgboost_gpu"):
    """Diagram batang komparatif metrik (F1, Recall, Precision, MCC, Specificity, FPR)."""
    df_mod = summary_df[summary_df["Model"] == model_name].copy()

    metrics = ["F1", "Recall", "Precision", "MCC", "Specificity", "FPR"]
    labels = ["F1-Score", "Recall", "Precision", "MCC", "Specificity", "FPR"]

    strategies = df_mod["Strategy_ID"].tolist()
    names = [STRATEGY_NAMES.get(s, s) for s in strategies]

    x = np.arange(len(metrics))
    width = 0.16

    fig, ax = plt.subplots(figsize=(15, 7))

    for idx, (sid, sname, col) in enumerate(zip(strategies, names, PALETTE[:len(strategies)])):
        row = df_mod[df_mod["Strategy_ID"] == sid].iloc[0]
        mean_vals = [row[f"Mean_{m}"] for m in metrics]
        std_vals = [row[f"Std_{m}"] for m in metrics]
        ax.bar(x + idx * width, mean_vals, width, yerr=std_vals, capsize=3, label=sname, color=col, alpha=0.9)

    ax.set_title(f"Perbandingan Metrik Evaluasi Komprehensif (Mean +/- Std Dev) — {model_name}", weight="bold", fontsize=14)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(labels, fontsize=11, weight="bold")
    ax.set_ylabel("Skor Metrik", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    save_path = plots_dir / f"barchart_metrics_comparison_{model_name}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Diagram batang metrik tersimpan di: {save_path}")


def plot_per_project_f1_gain(folds_df: pd.DataFrame, plots_dir: Path, model_name: str = "xgboost_gpu"):
    """Perbandingan F1-Score per proyek target antara Default 0.5 vs Max-F1 vs Recall>=70%."""
    df_mod = folds_df[folds_df["Model"] == model_name].copy()

    pivot = df_mod.pivot(index="Target_Project", columns="Strategy_ID", values="F1").fillna(0.0)
    # Filter hanya proyek yang memiliki flaky test di test set
    projects_with_flaky = df_mod[df_mod["Flaky_in_Test"] > 0]["Target_Project"].unique()
    pivot = pivot.loc[pivot.index.isin(projects_with_flaky)].copy()

    pivot["Gain_MaxF1"] = pivot["T2_Max_F1"] - pivot["T1_Default"]
    pivot = pivot.sort_values(by="Gain_MaxF1", ascending=False)

    plt.figure(figsize=(15, 8))
    x = np.arange(len(pivot))
    width = 0.25

    plt.bar(x - width, pivot["T1_Default"], width, label="T1: Default 0.5", color="#7f8c8d")
    plt.bar(x, pivot["T2_Max_F1"], width, label="T2: Max-F1 Tuning", color="#2980b9")
    plt.bar(x + width, pivot["T3_Recall_Constrained"], width, label="T3: Recall>=70%", color="#27ae60")

    plt.title(f"Perbandingan F1-Score per Proyek Target (Diurutkan Berdasarkan Gain Max-F1 vs Default 0.5)\nModel: {model_name}", weight="bold", fontsize=13)
    plt.xlabel("Proyek Target", weight="bold")
    plt.ylabel("F1-Score (Kelas Flaky)")
    plt.xticks(x, pivot.index, rotation=45, ha="right", fontsize=9)
    plt.legend(loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    save_path = plots_dir / f"per_project_f1_gain_{model_name}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Grafik F1 gain per proyek tersimpan di: {save_path}")


def generate_all_plots(folds_csv_path: str = None, summary_csv_path: str = None, plots_dir: str = None):
    """Menjalankan seluruh rutin visualisasi."""
    def_folds, def_sum, def_plots = resolve_default_paths()
    f_path = Path(folds_csv_path) if folds_csv_path else def_folds
    s_path = Path(summary_csv_path) if summary_csv_path else def_sum
    p_dir = Path(plots_dir) if plots_dir else def_plots
    p_dir.mkdir(parents=True, exist_ok=True)

    folds_df = pd.read_csv(f_path)
    summary_df = pd.read_csv(s_path)

    models = folds_df["Model"].unique().tolist() if "Model" in folds_df.columns else ["xgboost_gpu"]

    print("\n" + "=" * 80)
    print("[VIZ] MEMBUAT VISUALISASI HASIL PENELITIAN DYNAMIC THRESHOLD TUNING")
    print("=" * 80)

    for m in models:
        print(f"--> Memproses visualisasi untuk model: {m}")
        plot_f1_boxplots(folds_df, p_dir, model_name=m)
        plot_precision_recall_tradeoff(folds_df, p_dir, model_name=m)
        plot_threshold_distribution(folds_df, p_dir, model_name=m)
        plot_metrics_summary_bars(summary_df, p_dir, model_name=m)
        plot_per_project_f1_gain(folds_df, p_dir, model_name=m)

    print("=" * 80)
    print(f"[SELESAI] Seluruh grafik publikasi tersimpan di: {p_dir}")


if __name__ == "__main__":
    generate_all_plots()
