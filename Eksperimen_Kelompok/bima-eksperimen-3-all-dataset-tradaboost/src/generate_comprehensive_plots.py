"""
Modul Pembuatan Grafik Visualisasi Publikasi Ilmiah untuk Eksplorasi Faktorial Komprehensif
Menghasilkan 4 grafik 300 DPI yang memvisualisasikan seluruh dimensi pipeline.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def generate_comprehensive_plots():
    base_dir = Path(__file__).resolve().parent.parent
    sum_dir = base_dir / "results" / "summary"
    plot_dir = base_dir / "results" / "plots"
    os.makedirs(plot_dir, exist_ok=True)

    summary_csv = sum_dir / "comprehensive_pipeline_benchmark_summary.csv"
    if not summary_csv.exists():
        print("Summary CSV tidak ditemukan!")
        return

    df = pd.read_csv(summary_csv)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # -------------------------------------------------------------
    # PLOT 1: PENGARUH IMBALANCE HANDLING PADA TRADABOOST (Target 50%)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    fig.suptitle("Pengaruh Teknik Imbalance Handling pada TrAdaBoost (50% Target Supervision)", fontsize=14, fontweight="bold")

    imb_map = {
        "TrAdaBoost_50_B1_None": "B1: None (Natural)",
        "TrAdaBoost_50_B2_SMOTE": "B2: SMOTE",
        "TrAdaBoost_50": "B3: RUS (Balanced)",
        "TrAdaBoost_50_B4_ClassWeight": "B4: ClassWeight"
    }

    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for idx, dname in enumerate(["flakeflagger", "idoft"]):
        ax = axes[idx]
        sub = df[(df["dataset"] == dname) & (df["treatment"].isin(imb_map.keys()))].copy()
        sub["label"] = sub["treatment"].map(imb_map)
        sub = sub.sort_values(by="label")

        x = np.arange(len(sub))
        width = 0.35

        rects1 = ax.bar(x - width/2, sub["macro_f1"], width, label="Unweighted Macro F1", color="#2b5c8f")
        rects2 = ax.bar(x + width/2, sub["flaky_weighted_f1"], width, label="Flaky-Weighted F1", color="#e27c3e")

        ax.set_title(f"Dataset {dname.upper()}", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["label"], rotation=15, ha="right", fontsize=10)
        ax.set_ylabel("F1-Score" if idx == 0 else "")
        ax.set_ylim(0, 0.75)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # Nilai di atas bar
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

        if idx == 0:
            ax.legend(loc="upper left")

    plt.tight_layout()
    p1 = plot_dir / "factor_imbalance_handling_tradaboost.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"[PLOT 1/4] Disimpan: {p1}")

    # -------------------------------------------------------------
    # PLOT 2: PENGARUH BASE ESTIMATOR & DEPTH PADA TRADABOOST
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    fig.suptitle("Pengaruh Base Estimator & Kedalaman Pohon pada TrAdaBoost (50% Target)", fontsize=14, fontweight="bold")

    model_map = {
        "TrAdaBoost_50_DT_depth2": "DT (Depth 2)",
        "TrAdaBoost_50": "DT (Depth 4)",
        "TrAdaBoost_50_DT_depth6": "DT (Depth 6)",
        "TrAdaBoost_50_RF_base": "RF Base (10 Trees)"
    }

    for idx, dname in enumerate(["flakeflagger", "idoft"]):
        ax = axes[idx]
        sub = df[(df["dataset"] == dname) & (df["treatment"].isin(model_map.keys()))].copy()
        sub["label"] = sub["treatment"].map(model_map)

        x = np.arange(len(sub))
        width = 0.35

        rects1 = ax.bar(x - width/2, sub["macro_f1"], width, label="Unweighted Macro F1", color="#3b7a57")
        rects2 = ax.bar(x + width/2, sub["flaky_weighted_f1"], width, label="Flaky-Weighted F1", color="#c04000")

        ax.set_title(f"Dataset {dname.upper()}", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["label"], rotation=15, ha="right", fontsize=10)
        ax.set_ylabel("F1-Score" if idx == 0 else "")
        ax.set_ylim(0, 0.75)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

        if idx == 0:
            ax.legend(loc="upper left")

    plt.tight_layout()
    p2 = plot_dir / "factor_base_estimator_depth_tradaboost.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"[PLOT 2/4] Disimpan: {p2}")

    # -------------------------------------------------------------
    # PLOT 3: PENGARUH DATA CLEANING (FlakeFlagger: Imputed vs RowDrop)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle("Komparasi Efektivitas Data Cleaning: Imputasi Cerdas vs Row-Drop (FlakeFlagger)", fontsize=13, fontweight="bold")

    stages = ["Baseline (0%)", "TrAdaBoost (50%)", "Within-Project (100%)"]
    imputed_f1 = [
        df[(df["dataset"] == "flakeflagger") & (df["treatment"] == "Baseline_Zero_Shot_RF")]["macro_f1"].values[0],
        df[(df["dataset"] == "flakeflagger") & (df["treatment"] == "TrAdaBoost_50")]["macro_f1"].values[0],
        df[(df["dataset"] == "flakeflagger") & (df["treatment"] == "Within_Project")]["macro_f1"].values[0]
    ]
    rowdrop_f1 = [
        df[(df["dataset"] == "flakeflagger") & (df["treatment"] == "RowDrop_Baseline_RF")]["macro_f1"].values[0],
        df[(df["dataset"] == "flakeflagger") & (df["treatment"] == "RowDrop_TrAdaBoost_50")]["macro_f1"].values[0],
        df[(df["dataset"] == "flakeflagger") & (df["treatment"] == "RowDrop_Within_Project")]["macro_f1"].values[0]
    ]

    x = np.arange(len(stages))
    width = 0.35

    rects1 = ax.bar(x - width/2, imputed_f1, width, label="C2: Cleaned Imputed (22.236 baris, Zero Loss)", color="#1f77b4")
    rects2 = ax.bar(x + width/2, rowdrop_f1, width, label="C1: Row-Drop (21.963 baris, -273 baris)", color="#aec7e8")

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=11, fontweight="bold")
    ax.set_ylabel("Unweighted Macro F1-Score", fontsize=11)
    ax.set_ylim(0, 0.55)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(loc="upper left")

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    p3 = plot_dir / "factor_data_cleaning_tradaboost.png"
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"[PLOT 3/4] Disimpan: {p3}")

    # -------------------------------------------------------------
    # PLOT 4: PENGARUH ITERASI BOOSTING & SKEMA VOTING
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    fig.suptitle("Pengaruh Jumlah Iterasi Boosting & Skema Voting pada TrAdaBoost", fontsize=14, fontweight="bold")

    iter_map = {
        "TrAdaBoost_50_iter10": "N = 10 (Second-Half)",
        "TrAdaBoost_50": "N = 20 (Second-Half)",
        "TrAdaBoost_50_iter30": "N = 30 (Second-Half)",
        "TrAdaBoost_50_all_estimators": "N = 20 (All Estimators)"
    }

    for idx, dname in enumerate(["flakeflagger", "idoft"]):
        ax = axes[idx]
        sub = df[(df["dataset"] == dname) & (df["treatment"].isin(iter_map.keys()))].copy()
        sub["label"] = sub["treatment"].map(iter_map)

        x = np.arange(len(sub))
        width = 0.35

        rects1 = ax.bar(x - width/2, sub["macro_f1"], width, label="Unweighted Macro F1", color="#6a3d9a")
        rects2 = ax.bar(x + width/2, sub["flaky_weighted_f1"], width, label="Flaky-Weighted F1", color="#cab2d6")

        ax.set_title(f"Dataset {dname.upper()}", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["label"], rotation=15, ha="right", fontsize=10)
        ax.set_ylabel("F1-Score" if idx == 0 else "")
        ax.set_ylim(0, 0.75)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

        if idx == 0:
            ax.legend(loc="upper left")

    plt.tight_layout()
    p4 = plot_dir / "factor_scaling_and_voting_tradaboost.png"
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"[PLOT 4/4] Disimpan: {p4}")

if __name__ == "__main__":
    generate_comprehensive_plots()
