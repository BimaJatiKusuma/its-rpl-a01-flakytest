"""
Modul Visualisasi Grafik Publikasi (Publication-Quality Plots)
untuk Eksperimen One-on-One Afeltra IDoFT Dataset.
Rujukan: Afeltra et al. (IEEE Access 2024), Alshammari et al. (ICSE 2021).
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font="sans-serif")

def plot_pairwise_one_on_one_heatmap(df_pairwise: pd.DataFrame, out_path: Path, treatment: str = "Baseline_1on1"):
    """
    Plot 1: Heatmap Matriks One-on-One Pairwise Transfer (13x13 Proyek).
    Sumbu Y = Proyek Sumber (Source P_src), Sumbu X = Proyek Target (Target P_tgt).
    """
    sub = df_pairwise[df_pairwise["Treatment"] == treatment].copy()
    pivot = sub.pivot_table(index="Source_Project", columns="Target_Project", values="F1", aggfunc="mean").fillna(0.0)

    plt.figure(figsize=(11, 9), dpi=300)
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": "F1-Score (Positive Class)"},
        linewidths=0.5,
        linecolor="white"
    )
    plt.title(f"One-on-One Pairwise Cross-Project Transferability Matrix ({treatment})\nIDoFT Benchmark (13 Active Flaky Projects)", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Proyek Sumber (P_source / Training Domain)", fontsize=11, fontweight="bold")
    plt.xlabel("Proyek Target (P_target / Evaluation Domain)", fontsize=11, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_afeltra_treatments_heatmap(df_runs: pd.DataFrame, out_path: Path, model: str = "M1"):
    """
    Plot 2: Project-by-Project Heatmap (F1-Score) Lintas Intervensi Perlakuan Afeltra 2024.
    """
    sub = df_runs[df_runs["Model"] == model].copy()
    pivot = sub.pivot_table(index="Project", columns="Treatment", values="F1", aggfunc="mean").round(3)

    plt.figure(figsize=(10, 8), dpi=300)
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar_kws={"label": "Mean F1-Score"},
        linewidths=0.5,
        linecolor="white"
    )
    plt.title(f"Perbandingan Metrik F1-Score per Proyek IDoFT vs Intervensi Afeltra (Model {model})\nRepeated LOPO-CV (5 Random Seeds)", fontsize=12, fontweight="bold", pad=15)
    plt.ylabel("Target Project (P_test)", fontsize=11, fontweight="bold")
    plt.xlabel("Perlakuan / Intervensi Model", fontsize=11, fontweight="bold")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_afeltra_metrics_boxplots(df_runs: pd.DataFrame, out_path: Path, model: str = "M1"):
    """
    Plot 3: Boxplots Metrik (F1, Recall, Precision, PR-AUC) Antar Intervensi Perlakuan Afeltra.
    """
    sub = df_runs[(df_runs["Model"] == model) & (df_runs["Flaky_in_Test"] > 0)].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # 1. F1-Score
    sns.boxplot(data=sub, x="Treatment", y="F1", ax=axes[0, 0], palette="Set2")
    axes[0, 0].set_title("Distribusi F1-Score Antar Perlakuan", fontweight="bold")
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=25, ha="right")

    # 2. Recall
    sns.boxplot(data=sub, x="Treatment", y="Recall", ax=axes[0, 1], palette="Set2")
    axes[0, 1].set_title("Distribusi Recall Antar Perlakuan", fontweight="bold")
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=25, ha="right")

    # 3. Precision
    sns.boxplot(data=sub, x="Treatment", y="Precision", ax=axes[1, 0], palette="Set2")
    axes[1, 0].set_title("Distribusi Precision Antar Perlakuan", fontweight="bold")
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=25, ha="right")

    # 4. PR-AUC
    sns.boxplot(data=sub, x="Treatment", y="PR_AUC", ax=axes[1, 1], palette="Set2")
    axes[1, 1].set_title("Distribusi PR-AUC Antar Perlakuan", fontweight="bold")
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=25, ha="right")

    plt.suptitle(f"Evaluasi Komparatif Metrik Prediksi Flaky Test Lintas Perlakuan Paper Afeltra ({model})\nIDoFT Dataset (13 Active Flaky Projects)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_flaky_rate_vs_f1_scatter(corr_data: pd.DataFrame, out_path: Path):
    """
    Plot 4: Scatter Plot Rasio Flaky Test vs F1-Score dengan Regresi Linier.
    """
    plt.figure(figsize=(9, 6), dpi=300)
    sns.regplot(
        data=corr_data[corr_data["Flaky_Tests"] > 0],
        x="Flaky_Rate_%",
        y="F1",
        scatter_kws={"s": 70, "color": "#1f77b4", "alpha": 0.8},
        line_kws={"color": "#d62728", "linewidth": 2}
    )
    for _, row in corr_data[corr_data["Flaky_Tests"] > 0].iterrows():
        plt.annotate(
            row["project_clean"],
            (row["Flaky_Rate_%"], row["F1"]),
            textcoords="offset points",
            xytext=(5, 5),
            ha="left",
            fontsize=8,
            fontweight="bold"
        )
    plt.title("Korelasi Linier Antara Rasio Flaky Test dalam Proyek terhadap F1-Score\n(IDoFT Benchmark - Baseline Cross-Project)", fontsize=12, fontweight="bold")
    plt.xlabel("Flaky Test Ratio dalam Proyek Target (%)", fontsize=10, fontweight="bold")
    plt.ylabel("Cross-Project F1-Score", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_top_transfer_donors(df_pairwise: pd.DataFrame, out_path: Path):
    """
    Plot 5: Barplot Proyek Sumber Terbaik (Top Transfer Donors) pada Evaluasi One-on-One.
    """
    donor_perf = df_pairwise.groupby("Source_Project")["F1"].mean().sort_values(ascending=False).reset_index()

    plt.figure(figsize=(10, 6), dpi=300)
    ax = sns.barplot(data=donor_perf, x="F1", y="Source_Project", palette="mako")
    plt.title("Peringkat Proyek Sumber (Donor) Terbaik pada Transfer One-on-One\nIDoFT Benchmark (Rata-rata F1 saat menjadi training domain)", fontsize=12, fontweight="bold")
    plt.xlabel("Mean Transfer F1-Score", fontsize=10, fontweight="bold")
    plt.ylabel("Source Project", fontsize=10, fontweight="bold")
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f"{width:.3f}", (width + 0.002, p.get_y() + p.get_height() / 2.),
                    va="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
