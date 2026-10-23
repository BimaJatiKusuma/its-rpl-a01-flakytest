"""
Modul Analisis Konsistensi Stokastik & Stabilitas Metrik Lintas Repetisi Seed.
Rujukan: ACM/SIGSOFT Empirical Standards, Afeltra et al. (IEEE Access 2024).
"""

from typing import Tuple
import numpy as np
import pandas as pd
from scipy import stats

def analyze_stochastic_consistency(df_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung metrik konsistensi stokastik (Mean, Std, CV%, CI95 Margin)
    lintas 5 random seeds untuk setiap kombinasi Model dan Treatment.
    """
    metrics = ["F1", "Recall", "Precision", "PR_AUC", "ROC_AUC", "MCC", "Balanced_Accuracy"]
    group_cols = ["Model", "Treatment"]

    # Filter proyek aktif (13 proyek) untuk F1/Recall/Precision agar tidak terdistorsi oleh 0-flaky projects
    df_active = df_runs[df_runs["Flaky_in_Test"] > 0].copy()

    records = []
    for (model, treatment), sub in df_runs.groupby(group_cols):
        sub_act = df_active[(df_active["Model"] == model) & (df_active["Treatment"] == treatment)]

        # Mean F1 pada seluruh 17 proyek (per seed)
        seed_f1_all = sub.groupby("Seed")["F1"].mean()
        # Mean F1 pada 13 proyek aktif (per seed)
        seed_f1_act = sub_act.groupby("Seed")["F1"].mean()
        # Mean Recall pada 13 proyek aktif
        seed_rec = sub_act.groupby("Seed")["Recall"].mean()
        # Mean Precision pada 13 proyek aktif
        seed_prec = sub_act.groupby("Seed")["Precision"].mean()
        # Mean PR_AUC pada 13 proyek aktif
        seed_prauc = sub_act.groupby("Seed")["PR_AUC"].mean()

        mean_f1_17 = float(seed_f1_all.mean())
        mean_f1_13 = float(seed_f1_act.mean())
        std_f1 = float(seed_f1_act.std(ddof=1)) if len(seed_f1_act) > 1 else 0.0
        cv_f1 = (std_f1 / mean_f1_13 * 100.0) if mean_f1_13 > 0 else 0.0

        n_seeds = len(seed_f1_act)
        ci95_margin = float(stats.t.ppf(0.975, df=n_seeds - 1) * (std_f1 / np.sqrt(n_seeds))) if (n_seeds > 1 and std_f1 > 0) else 0.0

        mean_rec = float(seed_rec.mean())
        std_rec = float(seed_rec.std(ddof=1)) if len(seed_rec) > 1 else 0.0
        mean_prec = float(seed_prec.mean())
        std_prec = float(seed_prec.std(ddof=1)) if len(seed_prec) > 1 else 0.0
        mean_prauc = float(seed_prauc.mean())
        std_prauc = float(seed_prauc.std(ddof=1)) if len(seed_prauc) > 1 else 0.0

        # Kategori stabilitas
        if cv_f1 < 10.0:
            stability = "Sangat Stabil (CV < 10%)"
        elif cv_f1 <= 25.0:
            stability = "Stabil Moderat (10% - 25%)"
        else:
            stability = "Fluktuatif (CV > 25%)"

        records.append({
            "Model": model,
            "Treatment": treatment,
            "Mean_F1_17_Projects": round(mean_f1_17, 4),
            "Mean_F1_Active_13": round(mean_f1_13, 4),
            "Std_F1": round(std_f1, 4),
            "CV_F1_%": round(cv_f1, 2),
            "CI95_Margin": round(ci95_margin, 4),
            "Mean_Recall": round(mean_rec, 4),
            "Std_Recall": round(std_rec, 4),
            "Mean_Precision": round(mean_prec, 4),
            "Std_Precision": round(std_prec, 4),
            "Mean_PR_AUC": round(mean_prauc, 4),
            "Std_PR_AUC": round(std_prauc, 4),
            "Stability_Tier": stability
        })

    df_summary = pd.DataFrame(records)
    df_summary = df_summary.sort_values(by="Mean_F1_Active_13", ascending=False).reset_index(drop=True)
    return df_summary
