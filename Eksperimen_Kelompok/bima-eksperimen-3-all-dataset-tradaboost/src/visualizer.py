"""
Modul Visualisasi Publikasi Ilmiah (300 DPI) untuk Eksperimen TrAdaBoost Multi-Dataset.
Menghasilkan 6 grafik analitis berkualitas tinggi.
"""

import os
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14
})


def generate_all_plots(
    df_combined: pd.DataFrame,
    df_prof: pd.DataFrame,
    df_aggr: pd.DataFrame,
    output_dir: str
):
    """
    Menghasilkan seluruh visualisasi komparatif multi-dataset.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Standardisasi urutan treatment
    treatment_order = [
        "Baseline_Zero_Shot_RF",
        "TrAdaBoost_25",
        "TrAdaBoost_50",
        "TrAdaBoost_75",
        "Within_Project"
    ]
    treatment_labels = [
        "Baseline (0%)",
        "TrAdaBoost 25%",
        "TrAdaBoost 50%",
        "TrAdaBoost 75%",
        "Within-Project"
    ]

    # -------------------------------------------------------------
    # 1. SIDE-BY-SIDE TRADABOOST PROGRESSION (LINE PLOT)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300, sharey=True)

    prog_treatments = ["Baseline_Zero_Shot_RF", "TrAdaBoost_25", "TrAdaBoost_50", "TrAdaBoost_75"]
    prog_x = [0, 25, 50, 75]

    for idx, d in enumerate(["flakeflagger", "idoft"]):
        ax = axes[idx]
        sub = df_combined[(df_combined["dataset"] == d) & (df_combined["treatment"].isin(prog_treatments))]

        # Agregasi mean per treatment
        means = sub.groupby("treatment").agg({
            "f1_score": "mean",
            "recall_score": "mean",
            "precision_score": "mean",
            "pr_auc": "mean"
        }).reindex(prog_treatments)

        ax.plot(prog_x, means["f1_score"], marker="o", linewidth=2.5, markersize=8, color="#1f77b4", label="F1-Score")
        ax.plot(prog_x, means["recall_score"], marker="s", linewidth=2.0, markersize=7, color="#2ca02c", linestyle="--", label="Recall")
        ax.plot(prog_x, means["precision_score"], marker="^", linewidth=2.0, markersize=7, color="#d62728", linestyle="-.", label="Precision")
        ax.plot(prog_x, means["pr_auc"], marker="D", linewidth=2.0, markersize=7, color="#9467bd", linestyle=":", label="PR-AUC")

        for px, py in zip(prog_x, means["f1_score"]):
            ax.annotate(f"{py:.3f}", (px, py), textcoords="offset points", xytext=(0, 10), ha="center", fontweight="bold", color="#1f77b4")

        display_name = "FlakeFlagger (24 Proyek)" if d == "flakeflagger" else "IDoFT (17 Proyek)"
        ax.set_title(f"Progresi Transfer: {display_name}", pad=12)
        ax.set_xlabel("Porsi Data Berlabel Target yang Disuntikkan (%)")
        ax.set_xticks(prog_x)
        ax.set_xticklabels(["0% (Zero-Shot)", "25%", "50%", "75%"])
        ax.set_ylim(-0.02, 1.0)
        ax.grid(True, linestyle="--", alpha=0.6)
        if idx == 0:
            ax.set_ylabel("Nilai Metrik Rata-rata")
            ax.legend(loc="upper left", frameon=True)

    plt.suptitle("Dampak Porsi Data Target pada TrAdaBoost Lintas Dua Dataset", fontsize=15, y=0.98)
    plt.tight_layout()
    p1 = os.path.join(output_dir, "side_by_side_tradaboost_progression.png")
    plt.savefig(p1, bbox_inches="tight")
    plt.close()
    print(f" [SAVED] Plot 1 disimpan ke: {p1}")

    # -------------------------------------------------------------
    # 2. BOXPLOT TRADABOOST PORTIONS BOTH DATASETS
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300, sharey=True)

    # Ambil rata-rata per proyek melintasi seeds
    proj_means = df_combined.groupby(["dataset", "project", "treatment"])["f1_score"].mean().reset_index()

    for idx, d in enumerate(["flakeflagger", "idoft"]):
        ax = axes[idx]
        sub = proj_means[proj_means["dataset"] == d]

        palette = ["#aec7e8", "#ffbb78", "#98df8a", "#2ca02c", "#c5b0d5"]
        sns.boxplot(
            data=sub,
            x="treatment",
            y="f1_score",
            order=treatment_order,
            ax=ax,
            palette=palette,
            showmeans=True,
            meanprops={"marker": "D", "markeredgecolor": "black", "markerfacecolor": "yellow", "markersize": 6}
        )
        sns.stripplot(
            data=sub,
            x="treatment",
            y="f1_score",
            order=treatment_order,
            ax=ax,
            color="black",
            alpha=0.5,
            jitter=0.2,
            size=5
        )

        display_name = "FlakeFlagger (24 Proyek)" if d == "flakeflagger" else "IDoFT (17 Proyek)"
        ax.set_title(f"Distribusi F1-Score: {display_name}", pad=12)
        ax.set_xlabel("Perlakuan Model")
        ax.set_xticklabels(treatment_labels, rotation=20, ha="right")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)
        if idx == 0:
            ax.set_ylabel("F1-Score per Proyek")

    plt.suptitle("Distribusi F1-Score Lintas Proyek: Zero-Shot vs Variasi TrAdaBoost vs Upper Bound", fontsize=15, y=0.98)
    plt.tight_layout()
    p2 = os.path.join(output_dir, "boxplot_tradaboost_portions_both_datasets.png")
    plt.savefig(p2, bbox_inches="tight")
    plt.close()
    print(f" [SAVED] Plot 2 disimpan ke: {p2}")

    # -------------------------------------------------------------
    # 3. HEATMAP FLAKEFLAGGER PROJECTS VS TREATMENTS
    # -------------------------------------------------------------
    sub_ff = proj_means[proj_means["dataset"] == "flakeflagger"]
    piv_ff = sub_ff.pivot(index="project", columns="treatment", values="f1_score").reindex(columns=treatment_order)
    piv_ff.columns = treatment_labels

    plt.figure(figsize=(10, 10), dpi=300)
    sns.heatmap(piv_ff, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "F1-Score"}, vmin=0.0, vmax=1.0)
    plt.title("Heatmap F1-Score 24 Proyek FlakeFlagger vs Intervensi TrAdaBoost", fontsize=13, pad=12)
    plt.xlabel("Perlakuan Model", labelpad=8)
    plt.ylabel("Proyek Target Uji", labelpad=8)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    p3 = os.path.join(output_dir, "heatmap_flakeflagger_tradaboost_f1.png")
    plt.savefig(p3, bbox_inches="tight")
    plt.close()
    print(f" [SAVED] Plot 3 disimpan ke: {p3}")

    # -------------------------------------------------------------
    # 4. HEATMAP IDOFT PROJECTS VS TREATMENTS
    # -------------------------------------------------------------
    sub_id = proj_means[proj_means["dataset"] == "idoft"]
    piv_id = sub_id.pivot(index="project", columns="treatment", values="f1_score").reindex(columns=treatment_order)
    piv_id.columns = treatment_labels

    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(piv_id, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "F1-Score"}, vmin=0.0, vmax=1.0)
    plt.title("Heatmap F1-Score 17 Proyek IDoFT vs Intervensi TrAdaBoost", fontsize=13, pad=12)
    plt.xlabel("Perlakuan Model", labelpad=8)
    plt.ylabel("Proyek Target Uji", labelpad=8)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    p4 = os.path.join(output_dir, "heatmap_idoft_tradaboost_f1.png")
    plt.savefig(p4, bbox_inches="tight")
    plt.close()
    print(f" [SAVED] Plot 4 disimpan ke: {p4}")

    # -------------------------------------------------------------
    # 5. SCATTER: FLAKINESS RATE VS DELTA TRADABOOST BENEFIT
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6), dpi=300)

    for d, color, marker, label in [
        ("flakeflagger", "#1f77b4", "o", "FlakeFlagger (24 Proyek)"),
        ("idoft", "#ff7f0e", "s", "IDoFT (17 Proyek)")
    ]:
        sub = df_prof[df_prof["dataset"] == d]
        plt.scatter(
            sub["flakiness_rate_pct"],
            sub["delta_tradaboost75_f1"],
            color=color,
            marker=marker,
            s=80,
            alpha=0.8,
            edgecolor="black",
            label=label
        )
        # Regresi linear garis tren
        if len(sub) > 1:
            z = np.polyfit(sub["flakiness_rate_pct"], sub["delta_tradaboost75_f1"], 1)
            p = np.poly1d(z)
            x_line = np.linspace(sub["flakiness_rate_pct"].min(), sub["flakiness_rate_pct"].max(), 50)
            plt.plot(x_line, p(x_line), color=color, linestyle="--", alpha=0.7)

    plt.title("Korelasi Proporsi Flakiness Proyek (%) terhadap Lonjakan Delta F1 (TrAdaBoost 75%)", fontsize=13, pad=12)
    plt.xlabel("Proporsi Flaky Test Alami dalam Proyek Target (%)", labelpad=8)
    plt.ylabel("Delta Peningkatan F1-Score (TrAdaBoost 75% - Baseline RF)", labelpad=8)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    p5 = os.path.join(output_dir, "flakiness_rate_vs_target_benefit_scatter.png")
    plt.savefig(p5, bbox_inches="tight")
    plt.close()
    print(f" [SAVED] Plot 5 disimpan ke: {p5}")

    # -------------------------------------------------------------
    # 6. GROUPED BAR: COMPARISON OF 4 AGGREGATION SCHEMES
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 6), dpi=300)

    # Format data untuk grouped bar
    aggr_melt = pd.melt(
        df_aggr[df_aggr["treatment"].isin(["Baseline_Zero_Shot_RF", "TrAdaBoost_75"])],
        id_vars=["dataset", "treatment"],
        value_vars=["unweighted_macro_f1", "active_macro_f1", "flaky_weighted_f1", "global_micro_f1"],
        var_name="aggregation_scheme",
        value_name="f1_value"
    )

    scheme_map = {
        "unweighted_macro_f1": "Unweighted Macro",
        "active_macro_f1": "Active Macro",
        "flaky_weighted_f1": "Flaky-Weighted",
        "global_micro_f1": "Global Micro"
    }
    aggr_melt["aggregation_scheme"] = aggr_melt["aggregation_scheme"].map(scheme_map)
    aggr_melt["group"] = aggr_melt["dataset"].str.upper() + " - " + aggr_melt["treatment"].replace({
        "Baseline_Zero_Shot_RF": "Baseline (0%)",
        "TrAdaBoost_75": "TrAdaBoost 75%"
    })

    palette = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78"]
    sns.barplot(
        data=aggr_melt,
        x="aggregation_scheme",
        y="f1_value",
        hue="group",
        palette=palette,
        edgecolor="black"
    )

    for p in plt.gca().patches:
        height = p.get_height()
        if height > 0.005:
            plt.gca().annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height),
                               ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                               textcoords='offset points', rotation=0)

    plt.title("Perbandingan F1-Score Lintas 4 Skema Agregasi: Baseline Zero-Shot vs TrAdaBoost 75%", fontsize=13, pad=12)
    plt.xlabel("Skema Agregasi Matematis", labelpad=8)
    plt.ylabel("F1-Score Agregat", labelpad=8)
    plt.ylim(0.0, 1.0)
    plt.legend(title="Dataset & Perlakuan", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    p6 = os.path.join(output_dir, "radar_or_bar_comparison_treatments.png")
    plt.savefig(p6, bbox_inches="tight")
    plt.close()
    print(f" [SAVED] Plot 6 disimpan ke: {p6}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    raw_csv = base_dir / "results" / "raw" / "all_datasets_combined_predictions.csv"
    prof_csv = base_dir / "results" / "summary" / "project_capability_profile.csv"
    aggr_csv = base_dir / "results" / "summary" / "macro_vs_weighted_f1_summary.csv"
    plots_dir = str(base_dir / "results" / "plots")

    if raw_csv.exists() and prof_csv.exists() and aggr_csv.exists():
        df_c = pd.read_csv(raw_csv)
        df_p = pd.read_csv(prof_csv)
        df_a = pd.read_csv(aggr_csv)
        generate_all_plots(df_c, df_p, df_a, plots_dir)
    else:
        print("Data ringkasan belum lengkap. Jalankan runner dan profiler terlebih dahulu.")
