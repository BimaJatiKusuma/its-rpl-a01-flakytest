"""
Modul Visualisasi Publikasi Ilmiah untuk Replikasi Eksperimen 3 IDoFT.
Menghasilkan 6 grafik publikasi beresolusi 300 DPI:
1. 01_pipeline_overall_f1_stability.png : Stabilitas performa F1 ke-24 pipeline lintas 5 seeds.
2. 02_dataset_capability_heatmap.png    : Heatmap performa 24 Pipeline x 17 Proyek IDoFT.
3. 03_per_project_f1_consistency.png    : Kemampuan deteksi & konsistensi 17 proyek IDoFT (top pipeline).
4. 04_flaky_ratio_vs_f1_correlation.png : Korelasi proporsi flaky test terhadap F1 (bukti empiris root cause).
5. 05_factor_analysis_boxplots.png      : Pengaruh faktor Scaling, Imbalance Handler, dan Model Classifier.
6. 06_macro_vs_weighted_f1_deconstruction.png : Komparasi Macro vs Active vs Weighted vs Micro F1.
"""

from pathlib import Path
from typing import Optional
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

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

def plot_all_experiment_3_figures(results_dir: Optional[str] = None):
    """Menghasilkan seluruh visualisasi komprehensif untuk Replikasi Eksperimen 3 IDoFT."""
    if results_dir is None:
        results_dir = str(Path(__file__).resolve().parent.parent / "results")
    r_path = Path(results_dir)
    plots_dir = r_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_folds = pd.read_csv(r_path / "all_repeated_folds_predictions.csv")
    df_profile = pd.read_csv(r_path / "dataset_capability_profile.csv")
    df_cons = pd.read_csv(r_path / "consistency_stability_analysis.csv")
    df_runs = pd.read_csv(r_path / "all_seed_pipeline_runs.csv")

    # Temukan top pipeline
    top_pipe_id = df_cons.iloc[0]["Pipeline_ID"]
    print(f"[VISUALIZER IDOFT] Top Pipeline teridentifikasi: {top_pipe_id}")

    # -------------------------------------------------------------
    # 1. Pipeline Overall F1 Stability across 5 Seeds
    # -------------------------------------------------------------
    print("  Generating Plot 1: 01_pipeline_overall_f1_stability.png ...")
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Sort by Mean F1
    df_cons_sorted = df_cons.sort_values(by="Mean_F1_Mean", ascending=False)
    x = np.arange(len(df_cons_sorted))
    y = df_cons_sorted["Mean_F1_Mean"].values
    yerr = df_cons_sorted["Mean_F1_Std"].values

    colors = []
    for imb in df_cons_sorted["Imbalance"]:
        if imb == "B3": colors.append("#27ae60")     # Hijau (RUS)
        elif imb == "B2": colors.append("#2980b9")   # Biru (SMOTE)
        elif imb == "B4": colors.append("#8e44ad")   # Ungu (ClassWeight)
        else: colors.append("#7f8c8d")               # Abu-abu (None)

    bars = ax.bar(x, y, yerr=yerr, capsize=4, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df_cons_sorted["Pipeline_ID"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Macro-Average F1-Score (5 Seeds)")
    ax.set_title("Performa & Stabilitas 24 Pipeline Lintas 5 Random Seeds (IDoFT Dataset)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#27ae60", edgecolor="black", label="B3: RUS (Random Under-Sampling)"),
        Patch(facecolor="#2980b9", edgecolor="black", label="B2: SMOTE"),
        Patch(facecolor="#8e44ad", edgecolor="black", label="B4: ClassWeight"),
        Patch(facecolor="#7f8c8d", edgecolor="black", label="B1: Baseline (None)")
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    plt.tight_layout()
    plt.savefig(plots_dir / "01_pipeline_overall_f1_stability.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 2. Dataset Capability Heatmap (24 Pipelines x 17 Projects)
    # -------------------------------------------------------------
    print("  Generating Plot 2: 02_dataset_capability_heatmap.png ...")
    pivot_matrix = df_folds.groupby(["Pipeline_ID", "Target_Project"])["F1"].mean().unstack()
    pipe_sorted = sorted(df_folds["Pipeline_ID"].unique(), key=lambda x: int(x.split("_")[0][1:]))
    proj_sorted = sorted(df_folds["Target_Project"].unique())
    pivot_matrix = pivot_matrix.reindex(index=pipe_sorted, columns=proj_sorted)

    fig, ax = plt.subplots(figsize=(15, 9))
    sns.heatmap(
        pivot_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": "F1-Score"},
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 8}
    )
    ax.set_title("Heatmap Kemampuan Prediksi: 24 Pipeline vs 17 Proyek IDoFT (LOPO-CV)")
    ax.set_xlabel("Target Test Project Fold")
    ax.set_ylabel("Pipeline Configuration")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "02_dataset_capability_heatmap.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 3. Per-Project F1 Consistency on Top Pipeline
    # -------------------------------------------------------------
    print("  Generating Plot 3: 03_per_project_f1_consistency.png ...")
    top_df = df_folds[df_folds["Pipeline_ID"] == top_pipe_id]
    proj_agg = top_df.groupby("Target_Project").agg(
        Mean_F1=("F1", "mean"),
        Std_F1=("F1", "std"),
        Flaky_Count=("Flaky_in_Test", "first"),
        Test_Size=("Test_Size", "first")
    ).reset_index()
    proj_agg["Std_F1"] = proj_agg["Std_F1"].fillna(0.0)
    proj_agg.sort_values(by="Mean_F1", ascending=True, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(proj_agg))
    
    tier_colors = []
    for f1 in proj_agg["Mean_F1"]:
        if f1 >= 0.40: tier_colors.append("#2ecc71")     # Hijau (Tinggi)
        elif f1 >= 0.15: tier_colors.append("#3498db")   # Biru (Moderat)
        elif f1 > 0: tier_colors.append("#f39c12")       # Oranye (Rendah)
        else: tier_colors.append("#e74c3c")              # Merah (Zero / Sinkhole)

    ax.barh(y_pos, proj_agg["Mean_F1"], xerr=proj_agg["Std_F1"], capsize=4, color=tier_colors, alpha=0.85, edgecolor="black")
    ax.set_yticks(y_pos)
    labels = [f"{row['Target_Project']} ({int(row['Flaky_Count'])}/{int(row['Test_Size'])})" for _, row in proj_agg.iterrows()]
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(f"F1-Score pada Top Pipeline ({top_pipe_id})")
    ax.set_title(f"Profil Kemampuan & Stabilitas 17 Proyek IDoFT\n(Label: Proyek (Jumlah Flaky / Total Uji))")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    legend_elements = [
        Patch(facecolor="#2ecc71", edgecolor="black", label="High Detectability (F1 >= 0.40)"),
        Patch(facecolor="#3498db", edgecolor="black", label="Moderate Detection (0.15 <= F1 < 0.40)"),
        Patch(facecolor="#f39c12", edgecolor="black", label="Low / Fragile Detection (0 < F1 < 0.15)"),
        Patch(facecolor="#e74c3c", edgecolor="black", label="Zero Detection Sinkhole (F1 = 0.00)")
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    plt.savefig(plots_dir / "03_per_project_f1_consistency.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 4. Flaky Ratio vs F1 Correlation
    # -------------------------------------------------------------
    print("  Generating Plot 4: 04_flaky_ratio_vs_f1_correlation.png ...")
    valid_p = df_profile[df_profile["Flaky_Count"] > 0]
    
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.regplot(
        data=valid_p,
        x="Flaky_Rate_%",
        y="Overall_Mean_F1",
        color="#2980b9",
        scatter_kws={"s": 60, "alpha": 0.8},
        line_kws={"color": "#e74c3c", "linewidth": 2},
        ax=ax
    )

    for _, row in valid_p.iterrows():
        ax.annotate(
            row["Target_Project"],
            (row["Flaky_Rate_%"], row["Overall_Mean_F1"]),
            textcoords="offset points",
            xytext=(5, 5),
            ha="left",
            fontsize=8,
            alpha=0.85
        )

    ax.set_xlabel("Tingkat Flaky Test dalam Proyek (%)")
    ax.set_ylabel("Rata-rata Keseluruhan F1-Score")
    ax.set_title("Korelasi Proporsi Flaky Test vs Kemampuan Prediksi Model (IDoFT Dataset)")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "04_flaky_ratio_vs_f1_correlation.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 5. Factor Analysis Boxplots (Scaling, Imbalance, Classifier)
    # -------------------------------------------------------------
    print("  Generating Plot 5: 05_factor_analysis_boxplots.png ...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sns.boxplot(data=df_runs, x="Scaling", y="Mean_F1", ax=axes[0], palette="Set2")
    axes[0].set_title("Pengaruh Feature Scaling (S1, S2, S3)")
    axes[0].set_ylabel("Mean F1-Score")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    sns.boxplot(data=df_runs, x="Imbalance", y="Mean_F1", ax=axes[1], palette="Set1")
    axes[1].set_title("Pengaruh Imbalance Handling (B1-B4)")
    axes[1].set_ylabel("")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    sns.boxplot(data=df_runs, x="Model", y="Mean_F1", ax=axes[2], palette="Pastel1")
    axes[2].set_title("Pengaruh Classifier Model (M1 vs M2)")
    axes[2].set_ylabel("")
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Dampak Masing-Masing Faktor Eksperimen terhadap F1-Score Lintas 5 Seeds (IDoFT Dataset)", y=1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / "05_factor_analysis_boxplots.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 6. Macro vs Weighted vs Micro F1 Deconstruction
    # -------------------------------------------------------------
    print("  Generating Plot 6: 06_macro_vs_weighted_f1_deconstruction.png ...")
    df_exp = pd.read_csv(r_path / "macro_vs_weighted_f1_explanation.csv")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        df_exp["Metric_Aggregation"],
        df_exp["Score"],
        color=["#e74c3c", "#e67e22", "#3498db", "#2ecc71"],
        edgecolor="black",
        width=0.55
    )

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold"
        )

    ax.set_ylabel("F1-Score")
    ax.set_title(f"Dekonstruksi Skema Agregasi F1-Score pada Top Pipeline ({top_pipe_id})\n(IDoFT Dataset)")
    ax.set_ylim(0, max(df_exp["Score"]) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "06_macro_vs_weighted_f1_deconstruction.png", dpi=300)
    plt.close()

    print("[OK] Seluruh 6 visualisasi ilmiah berhasil dibuat di:", plots_dir)

if __name__ == "__main__":
    plot_all_experiment_3_figures()
