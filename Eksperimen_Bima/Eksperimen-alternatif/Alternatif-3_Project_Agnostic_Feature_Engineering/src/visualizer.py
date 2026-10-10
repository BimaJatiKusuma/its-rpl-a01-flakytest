"""
Modul Visualisasi Publikasi Ilmiah untuk Eksperimen Alternatif 3:
- Heatmap Korelasi Pearson (Multikolinearitas hIndex vs Ortogonalitas PCA Churn)
- Scree Plot PCA & Cumulative Explained Variance (Ambang 95%)
- Boxplot Komparasi Distribusi Metrik (PR-AUC, F1-Score, Recall, Precision) F1 s.d F4
- Barplot Ringkasan Performa (Mean +/- Std Dev)
- Visualisasi Kontribusi / Feature Importance XGBoost GPU pada Fitur Rekayasa
- Visualisasi Komparasi VIF Sebelum vs Sesudah PCA
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Pengaturan estetika publikasi IEEE / ACM
plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "figure.autolayout": True
})

FEATURE_SET_LABELS = {
    "F1_Original": "F1: Original (23)",
    "F2_Ratios_Only": "F2: Orig+Ratios (28)",
    "F3_Project_Agnostic": "F3: Agnostic (26)",
    "F4_Feature_Selection": "F4: Selected (12)"
}

PALETTE = ["#7f8c8d", "#2980b9", "#27ae60", "#8e44ad"]


def plot_correlation_heatmaps(
    df_raw_hindex: pd.DataFrame,
    df_pca_churn: pd.DataFrame,
    output_path: Path
):
    """
    Heatmap korelasi Pearson berdampingan:
    (Kiri) 8 Window hIndex mentah (multikolinearitas tinggi)
    (Kanan) Komponen PCA Churn (ortogonalitas sempurna)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Korelasi raw hIndex
    short_hindex_names = [f"Win_{c.split('_')[-1]}" for c in df_raw_hindex.columns]
    corr_raw = df_raw_hindex.corr()
    corr_raw.columns = short_hindex_names
    corr_raw.index = short_hindex_names

    sns.heatmap(
        corr_raw,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=axes[0],
        cbar_kws={"label": "Pearson Correlation (r)"}
    )
    axes[0].set_title("(a) Raw Code Churn (hIndex Windows)\nMultikolinearitas Tinggi (r > 0.90)", fontweight="bold")

    # 2. Korelasi PCA Components
    corr_pca = df_pca_churn.corr()
    sns.heatmap(
        corr_pca,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=axes[1],
        cbar_kws={"label": "Pearson Correlation (r)"}
    )
    axes[1].set_title("(b) PCA Churn Components\nOrtogonalitas Sempurna (r = 0.00)", fontweight="bold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Plot correlation heatmaps disimpan di: {output_path}")


def plot_pca_scree(
    explained_variance_ratio: np.ndarray,
    output_path: Path
):
    """
    Scree plot rasio variansi komponen PCA dan kurva kumulatif variansi dengan batas 95%.
    """
    n_comp = len(explained_variance_ratio)
    comp_labels = [f"PC{i+1}" for i in range(n_comp)]
    cum_var = np.cumsum(explained_variance_ratio)

    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    # Bar plot variansi individual
    bars = ax1.bar(comp_labels, explained_variance_ratio * 100, color="#3498db", alpha=0.85, label="Individual Variance (%)")
    ax1.set_xlabel("Principal Component", fontweight="bold")
    ax1.set_ylabel("Individual Explained Variance (%)", color="#2980b9", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#2980b9")
    ax1.set_ylim(0, max(explained_variance_ratio * 100) * 1.25)

    # Label pada bar
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    # Line plot variansi kumulatif
    ax2 = ax1.twinx()
    ax2.plot(comp_labels, cum_var * 100, color="#e74c3c", marker="o", linewidth=2.5, markersize=8, label="Cumulative Variance (%)")
    ax2.axhline(95.0, color="#27ae60", linestyle="--", linewidth=2, label="Ambang Batas 95% Target")
    ax2.set_ylabel("Cumulative Explained Variance (%)", color="#c0392b", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    ax2.set_ylim(40, 105)

    for i, txt in enumerate(cum_var * 100):
        ax2.annotate(f"{txt:.1f}%", (comp_labels[i], txt + 1.5), fontsize=9, fontweight="bold", ha="center")

    plt.title("PCA Scree Plot & Cumulative Explained Variance untuk Code Churn (hIndex)", fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Plot PCA scree disimpan di: {output_path}")


def plot_metric_boxplots(
    folds_df: pd.DataFrame,
    output_path: Path,
    model_name: str = "xgboost_gpu"
):
    """
    Boxplot perbandingan distribusi metrik (PR-AUC, F1-Score, Recall, Precision) lintas 4 set fitur.
    """
    df_mod = folds_df[folds_df["Model"] == model_name].copy()
    df_mod["Set_Label"] = df_mod["Feature_Set"].map(FEATURE_SET_LABELS)

    metrics = [
        ("PR_AUC", "PR-AUC (Precision-Recall AUC) [Metrik Utama]"),
        ("F1", "F1-Score (Kelas Flaky)"),
        ("Recall", "Recall (Flaky)"),
        ("Precision", "Precision (Flaky)")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (m, title) in enumerate(metrics):
        ax = axes[idx]
        sns.boxplot(
            data=df_mod,
            x="Set_Label",
            y=m,
            palette=PALETTE,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": 7},
            ax=ax
        )
        sns.stripplot(
            data=df_mod,
            x="Set_Label",
            y=m,
            color="black",
            alpha=0.35,
            jitter=0.2,
            size=5,
            ax=ax
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(m)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle(f"Distribusi Metrik Prediksi LOPO-CV Lintas Set Fitur (Model: {model_name})", fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Plot boxplot metrik disimpan di: {output_path}")


def plot_mean_summary_bars(
    summary_df: pd.DataFrame,
    output_path: Path
):
    """
    Barplot ringkasan performa Mean +/- Std Dev untuk PR-AUC dan F1 lintas set fitur dan model.
    """
    df = summary_df.copy()
    df["Set_Label"] = df["Feature_Set"].map(FEATURE_SET_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # PR-AUC
    sns.barplot(
        data=df,
        x="Set_Label",
        y="PR_AUC_Mean",
        hue="Model",
        palette=["#e74c3c", "#34495e"],
        ax=axes[0]
    )
    axes[0].set_title("Perbandingan Rata-rata PR-AUC (LOPO-CV 24 Proyek)", fontweight="bold")
    axes[0].set_ylabel("Mean PR-AUC")
    axes[0].set_xlabel("")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # F1-Score
    sns.barplot(
        data=df,
        x="Set_Label",
        y="F1_Mean",
        hue="Model",
        palette=["#e74c3c", "#34495e"],
        ax=axes[1]
    )
    axes[1].set_title("Perbandingan Rata-rata F1-Score (LOPO-CV 24 Proyek)", fontweight="bold")
    axes[1].set_ylabel("Mean F1-Score")
    axes[1].set_xlabel("")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Plot summary bars disimpan di: {output_path}")


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 15
):
    """
    Bar horizontal feature importance model XGBoost GPU pada set fitur F3.
    """
    top_df = importance_df.sort_values(by="Importance", ascending=False).head(top_n).copy()
    top_df = top_df.sort_values(by="Importance", ascending=True)

    plt.figure(figsize=(10, 7))
    colors = ["#27ae60" if ("assert" in f or "ratio" in f or "time" in f) 
              else ("#8e44ad" if "pca" in f 
              else ("#e67e22" if "log" in f else "#3498db")) 
              for f in top_df["Feature"]]

    bars = plt.barh(top_df["Feature"], top_df["Importance"], color=colors, alpha=0.85)
    plt.xlabel("XGBoost Gain / Feature Importance (F3)", fontweight="bold")
    plt.title(f"Top {top_n} Fitur Terpenting pada Model XGBoost GPU (Set F3: Project-Agnostic)", fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    # Tambahkan label nilai
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.002, bar.get_y() + bar.get_height() / 2.0, f"{w:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Plot feature importance disimpan di: {output_path}")


def plot_vif_comparison(
    vif_raw_df: pd.DataFrame,
    vif_pca_df: pd.DataFrame,
    output_path: Path
):
    """
    Perbandingan VIF antara 8 window hIndex asli vs Komponen PCA Churn.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Raw hIndex
    vif_raw_df = vif_raw_df.sort_values(by="VIF", ascending=True)
    axes[0].barh(vif_raw_df["Feature"], vif_raw_df["VIF"], color="#e74c3c", alpha=0.85)
    axes[0].axvline(5.0, color="orange", linestyle="--", label="Ambang Waspada (VIF=5)")
    axes[0].axvline(10.0, color="darkred", linestyle=":", label="Ambang Bahaya (VIF=10)")
    axes[0].set_title("VIF Fitur Asli Code Churn (hIndex)", fontweight="bold")
    axes[0].set_xlabel("Variance Inflation Factor (VIF)")
    axes[0].legend(loc="lower right")
    axes[0].grid(axis="x", linestyle="--", alpha=0.5)

    # PCA Components
    vif_pca_df = vif_pca_df.sort_values(by="VIF", ascending=True)
    axes[1].barh(vif_pca_df["Feature"], vif_pca_df["VIF"], color="#27ae60", alpha=0.85)
    axes[1].set_title("VIF Komponen PCA Churn (Ortogonal)", fontweight="bold")
    axes[1].set_xlabel("Variance Inflation Factor (VIF)")
    axes[1].grid(axis="x", linestyle="--", alpha=0.5)

    plt.suptitle("Analisis Penurunan Multikolinearitas (VIF) Melalui Reduksi Dimensi PCA", fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Plot VIF comparison disimpan di: {output_path}")
