"""
Modul Visualisasi Publikasi Ilmiah untuk Analisis SHAP (Alternatif 4).
Menghasilkan Beeswarm plots, Waterfall plots, Dependence plots, dan Chart Taksonomi beresolusi tinggi (300 DPI).
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Konfigurasi style publikasi
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "figure.titlesize": 14
})


def plot_beeswarm(
    shap_values: np.ndarray,
    X_df: pd.DataFrame,
    title: str,
    output_path: Path,
    max_display: int = 20
):
    """Membuat dan menyimpan SHAP Beeswarm Summary Plot."""
    explanation = shap.Explanation(
        values=shap_values,
        data=X_df.values,
        feature_names=X_df.columns.tolist()
    )

    fig = plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    plt.title(title, fontsize=13, pad=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close("all")
    print(f"[Visualizer] Beeswarm plot disimpan: {output_path.name}")


def plot_importance_comparison_bar(
    comparison_df: pd.DataFrame,
    output_path: Path
):
    """
    Membuat Bar Chart komparatif Mean Absolute SHAP antara Within-Project vs Cross-Project.
    """
    df_sorted = comparison_df.sort_values(by="cross_mean_abs_shap", ascending=True).copy()

    y_pos = np.arange(len(df_sorted))
    bar_height = 0.38

    fig, ax = plt.subplots(figsize=(11, 8))

    rects1 = ax.barh(y_pos + bar_height/2, df_sorted["within_mean_abs_shap"], bar_height,
                     label="Within-Project (Pooled CV)", color="#4575b4", alpha=0.88, edgecolor="black", linewidth=0.5)
    rects2 = ax.barh(y_pos - bar_height/2, df_sorted["cross_mean_abs_shap"], bar_height,
                     label="Cross-Project (LOPO-CV)", color="#d73027", alpha=0.88, edgecolor="black", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["feature"])
    ax.set_xlabel("Mean Absolute SHAP Value ($E[|SHAP|]$)", fontsize=11, fontweight="bold")
    ax.set_title("Komparasi Global Feature Importance: Within-Project vs Cross-Project (XGBoost GPU)",
                 fontsize=12, pad=14, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Anotasi taksonomi warna
    dim_colors = {"Test Smells": "#2b83ba", "Execution & Coverage": "#fdae61", "Code Churn": "#abdda4"}

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[Visualizer] Bar chart komparasi importance disimpan: {output_path.name}")


def plot_taxonomy_dimension_comparison(
    stats_dict: Dict[str, Any],
    output_path: Path
):
    """
    Membuat Grouped Bar Chart kontribusi 3 Dimensi Taksonomi Fitur
    (Test Smells vs Execution & Coverage vs Code Churn) pada skenario Within vs Cross.
    """
    dim_df = pd.DataFrame(stats_dict["dimension_contributions"])

    dimensions = dim_df["dimension"].tolist()
    within_pct = dim_df["within_importance_pct"].tolist()
    cross_pct = dim_df["cross_importance_pct"].tolist()

    x = np.arange(len(dimensions))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 5.5))

    bars1 = ax.bar(x - width/2, within_pct, width, label="Within-Project (Pooled)",
                   color="#3182bd", edgecolor="black", linewidth=0.7)
    bars2 = ax.bar(x + width/2, cross_pct, width, label="Cross-Project (LOPO-CV)",
                   color="#e6550d", edgecolor="black", linewidth=0.7)

    # Tambahkan angka di atas bar
    for b in bars1:
        h = b.get_height()
        ax.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    for b in bars2:
        h = b.get_height()
        ax.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Total Relatif Kontribusi SHAP (%)", fontsize=11, fontweight="bold")
    ax.set_title("Pergeseran Kontribusi Taksonomi Fitur: Within vs Cross-Project", fontsize=12, pad=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(dimensions, fontsize=10, fontweight="bold")
    ax.legend(frameon=True, framealpha=0.9)
    ax.set_ylim(0, max(max(within_pct), max(cross_pct)) + 12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[Visualizer] Chart taksonomi dimensi disimpan: {output_path.name}")


def plot_waterfall_case(
    case_data: Dict[str, Any],
    features: List[str],
    output_path: Path
):
    """
    Membuat SHAP Waterfall Plot untuk satu sampel pengujian lokal spesifik.
    """
    shap_vals = np.array([case_data["shap_values"][f] for f in features])
    feat_vals = np.array([case_data["feature_values"][f] for f in features])

    explanation = shap.Explanation(
        values=shap_vals,
        base_values=case_data["base_value"],
        data=feat_vals,
        feature_names=features
    )

    fig = plt.figure(figsize=(9, 6))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    status_label = "FLAKY TEST" if case_data["flaky_label"] == 1 else "NON-FLAKY TEST"
    plt.title(f"Local Explanation: {case_data['case_id']} ({status_label})\n"
              f"P(Flaky) = {case_data['pred_proba']:.3f} | Base Log-Odds = {case_data['base_value']:.3f}",
              fontsize=11, pad=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close("all")
    print(f"[Visualizer] Waterfall plot disimpan: {output_path.name}")


def plot_dependence(
    shap_values: np.ndarray,
    X_df: pd.DataFrame,
    feature_name: str,
    output_path: Path,
    interaction_feature: Optional[str] = None,
    inflection_threshold: Optional[float] = None
):
    """
    Membuat SHAP Dependence / Scatter Plot yang memperlihatkan titik infleksi (risk threshold).
    """
    explanation = shap.Explanation(
        values=shap_values,
        data=X_df.values,
        feature_names=X_df.columns.tolist()
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    if interaction_feature and interaction_feature in X_df.columns:
        shap.plots.scatter(explanation[:, feature_name], color=explanation[:, interaction_feature], ax=ax, show=False)
    else:
        shap.plots.scatter(explanation[:, feature_name], ax=ax, show=False)

    if inflection_threshold is not None:
        ax.axvline(x=inflection_threshold, color="crimson", linestyle="--", linewidth=1.8,
                   label=f"Ambang Risiko (x = {inflection_threshold})")
        ax.legend(loc="upper left")

    ax.set_title(f"SHAP Dependence Plot: {feature_name}", fontsize=12, pad=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[Visualizer] Dependence plot disimpan: {output_path.name}")


def plot_rank_shift_slopegraph(
    comparison_df: pd.DataFrame,
    output_path: Path
):
    """
    Membuat Slopegraph visualisasi pergeseran peringkat (Rank Shift)
    dari Within-Project ke Cross-Project.
    """
    fig, ax = plt.subplots(figsize=(7, 9))

    for _, row in comparison_df.iterrows():
        y1 = row["within_rank"]
        y2 = row["cross_rank"]
        dim = row["dimension"]

        if dim == "Test Smells":
            color = "#d95f02" # Oranye (Test Smells)
            lw = 2.2
            alpha = 0.95
        elif dim == "Execution & Coverage":
            color = "#7570b3" # Ungu
            lw = 1.8
            alpha = 0.85
        else:
            color = "#1b9e77" # Hijau (Code Churn)
            lw = 1.2
            alpha = 0.65

        ax.plot([0, 1], [y1, y2], marker="o", color=color, linewidth=lw, alpha=alpha)

        # Label kiri & kanan
        ax.text(-0.04, y1, f"{y1:2d}. {row['feature']}", ha="right", va="center", fontsize=8.5)
        ax.text(1.04, y2, f"{row['feature']} ({y2:2d})", ha="left", va="center", fontsize=8.5)

    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(24, 0) # Invert axis agar rank 1 di paling atas
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Within-Project\n(Pooled CV)", "Cross-Project\n(LOPO-CV)"], fontsize=11, fontweight="bold")
    ax.set_ylabel("Peringkat Kepentingan Fitur (1 = Paling Dominan)", fontsize=11, fontweight="bold")
    ax.set_title("Dinamika Pergeseran Peringkat Fitur (Rank Shift)", fontsize=13, pad=15, fontweight="bold")

    # Legend kustom
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#d95f02", lw=2.5, label="Test Smells (Naik Signifikan)"),
        Line2D([0], [0], color="#7570b3", lw=2, label="Execution & Coverage"),
        Line2D([0], [0], color="#1b9e77", lw=1.5, label="Code Churn (Turun/Multikolinear)")
    ]
    ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[Visualizer] Rank shift slopegraph disimpan: {output_path.name}")


def generate_all_plots(
    results_dir: Path,
    plots_dir: Path
):
    """Menghasilkan seluruh gambar visualisasi secara otomatis."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    cache_file = results_dir / "shap_cache_xgb.pkl"
    comp_csv = results_dir / "feature_rank_comparison.csv"
    stats_json = results_dir / "rank_comparison_stats.json"
    case_json = results_dir / "spring_boot_case_studies.json"

    if not cache_file.exists():
        raise FileNotFoundError(f"Cache {cache_file} tidak ditemukan. Jalankan pipeline SHAP terlebih dahulu.")

    with open(cache_file, "rb") as f:
        cache = pickle.load(f)

    with open(stats_json, "r") as f:
        stats_dict = json.load(f)

    with open(case_json, "r") as f:
        case_studies = json.load(f)

    comp_df = pd.read_csv(comp_csv)
    features = cache["features"]

    # Muat dataset untuk X_df
    from .data_loader import load_dataset
    df, _, _ = load_dataset()
    X_df = df[features]

    # 1. Beeswarm Cross-Project XGBoost GPU
    plot_beeswarm(
        cache["xgb_cross"]["shap_values"],
        X_df,
        "SHAP Summary Beeswarm: Cross-Project LOPO-CV (XGBoost GPU)",
        plots_dir / "shap_beeswarm_cross_project_xgb.png"
    )

    # 2. Beeswarm Within-Project XGBoost GPU
    plot_beeswarm(
        cache["xgb_within"]["shap_values"],
        X_df,
        "SHAP Summary Beeswarm: Within-Project Pooled CV (XGBoost GPU)",
        plots_dir / "shap_beeswarm_within_project_xgb.png"
    )

    # 3. Beeswarm Cross-Project Random Forest
    plot_beeswarm(
        cache["rf_cross"]["shap_values"],
        X_df,
        "SHAP Summary Beeswarm: Cross-Project LOPO-CV (Random Forest)",
        plots_dir / "shap_beeswarm_cross_project_rf.png"
    )

    # 4. Importance Comparison Bar
    plot_importance_comparison_bar(
        comp_df,
        plots_dir / "feature_importance_comparison_bar.png"
    )

    # 5. Taxonomy Dimension Bar
    plot_taxonomy_dimension_comparison(
        stats_dict,
        plots_dir / "taxonomy_dimension_comparison_bar.png"
    )

    # 6. Rank Shift Slopegraph
    plot_rank_shift_slopegraph(
        comp_df,
        plots_dir / "rank_shift_slopegraph.png"
    )

    # 7. Waterfall Plots (3 Flaky, 3 Non-Flaky)
    for i, case in enumerate(case_studies["flaky_cases"], 1):
        plot_waterfall_case(
            case,
            features,
            plots_dir / f"waterfall_flaky_case_{i}.png"
        )

    for i, case in enumerate(case_studies["non_flaky_cases"], 1):
        plot_waterfall_case(
            case,
            features,
            plots_dir / f"waterfall_nonflaky_case_{i}.png"
        )

    # 8. Dependence Plots untuk fitur top
    plot_dependence(
        cache["xgb_cross"]["shap_values"],
        X_df,
        "ExecutionTime",
        plots_dir / "dependence_ExecutionTime.png",
        interaction_feature="testLength",
        inflection_threshold=15.0
    )

    plot_dependence(
        cache["xgb_cross"]["shap_values"],
        X_df,
        "numCoveredLines",
        plots_dir / "dependence_numCoveredLines.png",
        interaction_feature="numAsserts",
        inflection_threshold=50.0
    )

    plot_dependence(
        cache["xgb_cross"]["shap_values"],
        X_df,
        "resource-optimism",
        plots_dir / "dependence_resource_optimism.png",
        interaction_feature="ExecutionTime"
    )

    print(f"\n[SUKSES] Seluruh 15 visualisasi riset berhasil dibuat di: {plots_dir}")


if __name__ == "__main__":
    res_dir = Path(__file__).resolve().parent.parent / "results"
    p_dir = res_dir / "plots"
    generate_all_plots(res_dir, p_dir)
