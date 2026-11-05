"""
Modul Analisis Konsistensi Stokastik dan Stabilitas Empiris Lintas 5 Random Seeds.
Menghitung Mean (mu), Std (sigma), Coefficient of Variation (CV%), dan Margin CI95.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def compute_consistency_analysis(
    df_combined: pd.DataFrame,
    output_csv: Optional[str] = None
) -> pd.DataFrame:
    """
    Menghitung konsistensi performa antar seed per perlakuan dan per dataset.
    """
    records = []

    metrics_to_analyze = [
        ("f1_score", "F1"),
        ("recall_score", "Recall"),
        ("precision_score", "Precision"),
        ("pr_auc", "PR-AUC")
    ]

    datasets = df_combined["dataset"].unique().tolist()
    treatments = df_combined["treatment"].unique().tolist()

    for d in datasets:
        df_d = df_combined[df_combined["dataset"] == d]
        n_proj = df_d["project"].nunique()

        for t in treatments:
            df_dt = df_d[df_d["treatment"] == t]
            if len(df_dt) == 0:
                continue

            # Hitung macro average per seed
            seed_macro = df_dt.groupby("seed").agg({
                "f1_score": "mean",
                "recall_score": "mean",
                "precision_score": "mean",
                "pr_auc": "mean"
            }).reset_index()

            # Juga hitung macro average hanya pada proyek dengan flaky test aktif
            df_active = df_dt[df_dt["total_project_flaky"] > 0]
            seed_active_f1 = df_active.groupby("seed")["f1_score"].mean().values if len(df_active) > 0 else np.zeros(len(seed_macro))

            n_seeds = len(seed_macro)

            f1_vals = seed_macro["f1_score"].values
            rec_vals = seed_macro["recall_score"].values
            prec_vals = seed_macro["precision_score"].values
            prauc_vals = seed_macro["pr_auc"].values

            mean_f1 = float(np.mean(f1_vals))
            std_f1 = float(np.std(f1_vals, ddof=1)) if n_seeds > 1 else 0.0
            cv_f1 = (std_f1 / mean_f1 * 100.0) if mean_f1 > 0 else 0.0
            ci95_f1 = 1.96 * (std_f1 / np.sqrt(n_seeds)) if n_seeds > 1 else 0.0

            mean_act_f1 = float(np.mean(seed_active_f1))
            std_act_f1 = float(np.std(seed_active_f1, ddof=1)) if n_seeds > 1 else 0.0
            cv_act_f1 = (std_act_f1 / mean_act_f1 * 100.0) if mean_act_f1 > 0 else 0.0

            mean_rec = float(np.mean(rec_vals))
            std_rec = float(np.std(rec_vals, ddof=1)) if n_seeds > 1 else 0.0
            cv_rec = (std_rec / mean_rec * 100.0) if mean_rec > 0 else 0.0

            mean_prec = float(np.mean(prec_vals))
            std_prec = float(np.std(prec_vals, ddof=1)) if n_seeds > 1 else 0.0

            mean_prauc = float(np.mean(prauc_vals))
            std_prauc = float(np.std(prauc_vals, ddof=1)) if n_seeds > 1 else 0.0

            # Kategori stabilitas
            if cv_f1 < 10.0:
                stability_tier = "Sangat Tinggi (CV < 10%)"
            elif cv_f1 < 25.0:
                stability_tier = "Moderat (10% <= CV < 25%)"
            else:
                stability_tier = "Sensitif Stokastik (CV >= 25%)"

            records.append({
                "dataset": d,
                "n_projects": n_proj,
                "treatment": t,
                "n_seeds": n_seeds,
                "mean_macro_f1": round(mean_f1, 4),
                "std_macro_f1": round(std_f1, 4),
                "cv_macro_f1_pct": round(cv_f1, 2),
                "ci95_margin_f1": round(ci95_f1, 4),
                "mean_active_f1": round(mean_act_f1, 4),
                "std_active_f1": round(std_act_f1, 4),
                "cv_active_f1_pct": round(cv_act_f1, 2),
                "mean_recall": round(mean_rec, 4),
                "std_recall": round(std_rec, 4),
                "cv_recall_pct": round(cv_rec, 2),
                "mean_precision": round(mean_prec, 4),
                "std_precision": round(std_prec, 4),
                "mean_pr_auc": round(mean_prauc, 4),
                "std_pr_auc": round(std_prauc, 4),
                "stability_tier": stability_tier
            })

    res_df = pd.DataFrame(records)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        res_df.to_csv(output_csv, index=False)
        print(f" [SAVED] Ringkasan konsistensi disimpan ke: {output_csv}")

    return res_df


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    comb_csv = base_dir / "results" / "raw" / "all_datasets_combined_predictions.csv"
    if comb_csv.exists():
        df_c = pd.read_csv(comb_csv)
        out_csv = str(base_dir / "results" / "summary" / "consistency_stability_analysis.csv")
        compute_consistency_analysis(df_c, out_csv)
    else:
        print(f"File {comb_csv} belum ada. Jalankan runner terlebih dahulu.")
