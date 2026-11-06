"""
Modul Visualisasi Publikasi Ilmiah untuk Eksperimen Gabungan 1 & 2.
Menghasilkan grafik publikasi standar resolusi tinggi (300 DPI) di folder results/plots/:
1. 01_boxplot_tri_paradigm_cleaning.png: Komparasi langsung C1 vs C2 vs C3
2. 02_boxplot_4_factors_f1.png: Distribusi F1-Score per faktor (Cleaning, Scaling, Imbalance, Model)
3. 03_top10_consistent_pipelines_ci95.png: Top 10 Pipeline Terbaik & Konsisten (Mean F1 +- 95% CI)
4. 04_stability_cv_distribution.png: Kestabilan Empiris (Coefficient of Variation CV%) C1 vs C2 vs C3
5. 05_f1_vs_recall_tradeoff.png: Trade-off Mean F1 vs Mean Recall (Warna per Strategi Cleaning)
6. 06_heatmap_factorial_matrix.png: Peta Panas Kinerja Faktorial (Scaling x Imbalance) per Paradigma Cleaning
"""

import sys
import warnings
import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def generate_all_visualizations(
    results_dir: Optional[str] = None,
    plots_dir: Optional[str] = None
):
    """
    Menghasilkan seluruh visualisasi ilmiah dari hasil eksperimen gabungan (single-run atau repeated LOPO-CV).
    """
    base_res = Path(results_dir) if results_dir else Path(__file__).resolve().parent.parent / "results"
    out_plots = Path(plots_dir) if plots_dir else base_res / "plots"
    out_plots.mkdir(parents=True, exist_ok=True)

    summary_file = base_res / "summary_metrics_72_pipelines.csv"
    runs_file = base_res / "all_seed_pipeline_runs.csv"
    cons_file = base_res / "consistency_stability_analysis.csv"

    # 1. Pastikan ketersediaan data (bisa dari repeated runs atau single run)
    df_runs = pd.read_csv(runs_file) if runs_file.exists() else None
    df_cons = pd.read_csv(cons_file) if cons_file.exists() else None

    # Jika summary 72 belum ada tapi runs_file ada, buatkan summary teragregasi secara otomatis
    if not summary_file.exists() and df_runs is not None:
        print("[INFO] 'summary_metrics_72_pipelines.csv' belum ada. Membuat ringkasan teragregasi dari data repetisi...")
        df_summary = df_runs.groupby(["Pipeline_ID", "Cleaning", "Scaling", "Imbalance", "Model"]).agg({
            "Mean_F1": "mean",
            "Std_F1": "mean",
            "Mean_Recall": "mean",
            "Mean_Precision": "mean",
            "Mean_PR_AUC": "mean",
            "Mean_ROC_AUC": "mean",
            "Mean_MCC": "mean",
            "Mean_Balanced_Accuracy": "mean",
            "Execution_Time_Sec": "mean"
        }).reset_index()
        df_summary.sort_values(by="Mean_F1", ascending=False, inplace=True)
        df_summary.to_csv(summary_file, index=False)
        print(f"[OK] Summary berhasil dibuat dan disimpan ke: {summary_file.name}")
    elif summary_file.exists():
        df_summary = pd.read_csv(summary_file)
    else:
        print(f"[ERROR] Tidak ada berkas hasil ditemukan di {base_res}. Harap jalankan eksperimen terlebih dahulu.")
        return

    sns.set_theme(style="whitegrid", font="sans-serif")
    palette_cleaning = {"C1": "#3498db", "C2": "#e74c3c", "C3": "#2ecc71"}
    palette_clean_labels = {
        "C1": "C1: Row Drop (24 Proyek)",
        "C2": "C2: Project Drop (19 Proyek)",
        "C3": "C3: Imputasi Cerdas (24 Proyek)"
    }

    # Data dasar untuk boxplot faktor: gunakan df_runs jika ada (menangkap seluruh variansi seed), atau df_summary
    df_plot_base = df_runs if df_runs is not None else df_summary

    print("\n" + "=" * 75)
    print(f"📊 MEMBUAT GRAFIK PUBLIKASI ILMIAH (300 DPI) -> {out_plots}")
    print("=" * 75)

    # -------------------------------------------------------------------------------------------------
    # PLOT 1: Boxplot Tri-Paradigma Cleaning (C1 vs C2 vs C3)
    # -------------------------------------------------------------------------------------------------
    plt.figure(figsize=(8, 5.5), dpi=300)
    ax = sns.boxplot(data=df_plot_base, x="Cleaning", y="Mean_F1", palette=palette_cleaning, width=0.45, linewidth=1.2)
    sns.stripplot(data=df_plot_base, x="Cleaning", y="Mean_F1", color="black", alpha=0.35, jitter=0.2, size=5)
    plt.title("Komparasi Distribusi F1-Score: Tri-Paradigma Data Cleaning", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Strategi Data Cleaning", fontweight="bold", labelpad=8)
    plt.ylabel("Mean F1-Score (LOPO-CV)", fontweight="bold", labelpad=8)
    ax.set_xticklabels([f"{c}\n({palette_clean_labels[c].split(':')[1].strip()})" for c in ["C1", "C2", "C3"]])
    plt.tight_layout()
    p1 = out_plots / "01_boxplot_tri_paradigm_cleaning.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"  [✓] 1. {p1.name}")

    # -------------------------------------------------------------------------------------------------
    # PLOT 2: Boxplot 4 Faktor Eksperimen (Cleaning, Scaling, Imbalance, Model)
    # -------------------------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=300)
    factors = [
        ("Cleaning", "Data Cleaning Strategy (C1 vs C2 vs C3)", ["#3498db", "#e74c3c", "#2ecc71"]),
        ("Scaling", "Feature Scaling (S1: None | S2: Robust | S3: Standard)", "Blues_r"),
        ("Imbalance", "Imbalance Handling (B1: None | B2: SMOTE | B3: ROS | B4: CW)", "Purples_r"),
        ("Model", "Classifier Model (M1: RF | M2: XGBoost)", ["#e67e22", "#16a085"])
    ]
    for idx, (f_col, title, pal) in enumerate(factors):
        ax_curr = axes[idx // 2, idx % 2]
        sns.boxplot(data=df_plot_base, x=f_col, y="Mean_F1", ax=ax_curr, palette=pal, width=0.45, linewidth=1.1)
        sns.stripplot(data=df_plot_base, x=f_col, y="Mean_F1", ax=ax_curr, color="black", alpha=0.3, jitter=0.15, size=4)
        ax_curr.set_title(title, fontsize=10.5, fontweight="bold")
        ax_curr.set_xlabel(f_col, fontweight="bold")
        ax_curr.set_ylabel("Mean F1-Score", fontweight="bold")
    plt.suptitle("Distribusi F1-Score LOPO-CV Berdasarkan 4 Faktor Faktorial", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    p2 = out_plots / "02_boxplot_4_factors_f1.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"  [✓] 2. {p2.name}")

    # -------------------------------------------------------------------------------------------------
    # PLOT 3: Top 10 Consistent Pipelines (Mean F1 +- 95% Confidence Interval)
    # -------------------------------------------------------------------------------------------------
    if df_cons is not None and "Mean_F1_Mean" in df_cons.columns:
        top10 = df_cons.sort_values(by="Mean_F1_Mean", ascending=False).head(10).copy()
        plt.figure(figsize=(10, 6), dpi=300)
        y_pos = np.arange(len(top10))
        colors = [palette_cleaning.get(c, "#95a5a6") for c in top10["Cleaning"]]

        # Horizontal bar chart with error bars (CI95)
        plt.barh(y_pos, top10["Mean_F1_Mean"], xerr=top10.get("Mean_F1_CI95", 0), align="center",
                 color=colors, alpha=0.85, edgecolor="black", capsize=4, height=0.6)
        plt.yticks(y_pos, top10["Pipeline_ID"], fontsize=9.5)
        plt.gca().invert_yaxis()  # Top pipeline at top
        plt.xlabel("Mean F1-Score (Lintas 10 Seeds ± 95% Confidence Interval)", fontweight="bold")
        plt.title("Top 10 Pipeline Paling Berperforma & Konsisten (10x Repetisi LOPO-CV)", fontsize=12, fontweight="bold", pad=12)

        # Label keterangan nilai F1 dan CV% di ujung bar
        for idx_b, (_, row) in enumerate(top10.iterrows()):
            f1_val = row["Mean_F1_Mean"]
            cv_val = row.get("Mean_F1_CV_%", 0)
            plt.text(f1_val + 0.008, idx_b, f"F1: {f1_val:.4f} (CV: {cv_val:.1f}%)", va="center", fontsize=8.5, fontweight="bold")

        # Custom legend untuk strategi cleaning
        import matplotlib.patches as mpatches
        legend_patches = [mpatches.Patch(color=palette_cleaning[c], label=palette_clean_labels[c]) for c in ["C1", "C2", "C3"]]
        plt.legend(handles=legend_patches, loc="lower right", frameon=True, fontsize=9)
        plt.xlim(0, max(top10["Mean_F1_Mean"]) * 1.35)
        plt.tight_layout()
        p3 = out_plots / "03_top10_consistent_pipelines_ci95.png"
        plt.savefig(p3, dpi=300)
        plt.close()
        print(f"  [✓] 3. {p3.name}")

    # -------------------------------------------------------------------------------------------------
    # PLOT 4: Analisis Kestabilan Empiris (Coefficient of Variation CV%)
    # -------------------------------------------------------------------------------------------------
    if df_cons is not None and "Mean_F1_CV_%" in df_cons.columns:
        plt.figure(figsize=(8, 5), dpi=300)
        ax = sns.boxplot(data=df_cons, x="Cleaning", y="Mean_F1_CV_%", palette=palette_cleaning, width=0.45, linewidth=1.2)
        sns.stripplot(data=df_cons, x="Cleaning", y="Mean_F1_CV_%", color="black", alpha=0.5, jitter=0.2, size=6)
        plt.title("Uji Stabilitas: Coefficient of Variation (CV% F1-Score) Lintas 10 Seeds\n(Semakin Rendah CV% = Model Semakin Konsisten/Stabil)",
                  fontsize=11.5, fontweight="bold", pad=12)
        plt.xlabel("Strategi Data Cleaning", fontweight="bold")
        plt.ylabel("Coefficient of Variation (%)", fontweight="bold")
        ax.set_xticklabels([f"{c}\n({palette_clean_labels[c].split(':')[1].strip()})" for c in ["C1", "C2", "C3"]])
        plt.tight_layout()
        p4 = out_plots / "04_stability_cv_distribution.png"
        plt.savefig(p4, dpi=300)
        plt.close()
        print(f"  [✓] 4. {p4.name}")

    # -------------------------------------------------------------------------------------------------
    # PLOT 5: Trade-off Mean F1 vs Mean Recall (Optimal Flaky Detection)
    # -------------------------------------------------------------------------------------------------
    plt.figure(figsize=(9, 6), dpi=300)
    for c_id, c_group in df_summary.groupby("Cleaning"):
        plt.scatter(
            c_group["Mean_Recall"],
            c_group["Mean_F1"],
            color=palette_cleaning.get(c_id, "#333333"),
            label=palette_clean_labels.get(c_id, c_id),
            s=c_group["Mean_PR_AUC"] * 2500 + 30,  # Ukuran lingkaran proporsional terhadap PR-AUC
            alpha=0.75,
            edgecolors="black",
            linewidth=0.8
        )
    plt.title("Trade-off Kinerja: Mean F1-Score vs Mean Recall (Ukuran Titik = PR-AUC)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Mean Recall (Sensitivitas Menangkap Uji Flaky)", fontweight="bold")
    plt.ylabel("Mean F1-Score (Keseimbangan Harmonis)", fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout()
    p5 = out_plots / "05_f1_vs_recall_tradeoff.png"
    plt.savefig(p5, dpi=300)
    plt.close()
    print(f"  [✓] 5. {p5.name}")

    # -------------------------------------------------------------------------------------------------
    # PLOT 6: Heatmap Faktorial (Scaling x Imbalance) per Paradigma Cleaning
    # -------------------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300, sharey=True)
    c_strategies = ["C1", "C2", "C3"]
    vmax = df_summary["Mean_F1"].max()
    vmin = df_summary["Mean_F1"].min()

    for idx, c_strat in enumerate(c_strategies):
        sub_df = df_summary[df_summary["Cleaning"] == c_strat]
        if not sub_df.empty:
            pivot_t = sub_df.pivot_table(index="Imbalance", columns="Scaling", values="Mean_F1", aggfunc="mean")
            sns.heatmap(pivot_t, ax=axes[idx], cmap="YlGnBu", annot=True, fmt=".3f", vmin=vmin, vmax=vmax,
                        cbar=(idx == 2), cbar_kws={'label': 'Mean F1-Score'} if idx == 2 else None)
            axes[idx].set_title(f"Paradigma {palette_clean_labels[c_strat]}", fontsize=10.5, fontweight="bold")
            axes[idx].set_xlabel("Feature Scaling", fontweight="bold")
            if idx == 0:
                axes[idx].set_ylabel("Imbalance Handling", fontweight="bold")
            else:
                axes[idx].set_ylabel("")

    plt.suptitle("Peta Panas Kinerja Interaksi Faktorial: Imbalance vs Scaling per Paradigma Cleaning",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p6 = out_plots / "06_heatmap_factorial_matrix.png"
    plt.savefig(p6, dpi=300)
    plt.close()
    print(f"  [✓] 6. {p6.name}")

    print("\n[SELESAI] Seluruh 6 plot publikasi resolusi tinggi berhasil disimpan ke:")
    print(f"📁 {out_plots.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizer Publikasi Eksperimen Gabungan 1 & 2")
    parser.add_argument("--results_dir", type=str, default=None, help="Direktori hasil eksperimen")
    parser.add_argument("--plots_dir", type=str, default=None, help="Direktori penyimpanan file gambar")
    args = parser.parse_args()

    generate_all_visualizations(results_dir=args.results_dir, plots_dir=args.plots_dir)
