"""
Modul Profiling Kemampuan Dataset dan Dekonstruksi 4 Skema Agregasi Matematis.
Menganalisis perbedaan performa antar proyek dan dampak porsi data target pada FlakeFlagger & IDoFT.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def compute_project_capability_profile(
    df_combined: pd.DataFrame,
    output_csv: Optional[str] = None
) -> pd.DataFrame:
    """
    Menghasilkan matriks kapabilitas proyek vs perlakuan TrAdaBoost per dataset.
    """
    records = []

    datasets = df_combined["dataset"].unique().tolist()

    for d in datasets:
        df_d = df_combined[df_combined["dataset"] == d]
        projects = sorted(df_d["project"].unique().tolist())

        for p in projects:
            sub = df_d[df_d["project"] == p]
            n_total = int(sub["total_project_tests"].iloc[0])
            n_flaky = int(sub["total_project_flaky"].iloc[0])
            rate = float(sub["flakiness_rate_pct"].iloc[0])

            p_rec: Dict[str, Any] = {
                "dataset": d,
                "project": p,
                "total_tests": n_total,
                "flaky_tests": n_flaky,
                "flakiness_rate_pct": rate
            }

            for t in df_d["treatment"].unique():
                sub_t = sub[sub["treatment"] == t]
                if len(sub_t) > 0:
                    mean_f1 = float(sub_t["f1_score"].mean())
                    mean_rec = float(sub_t["recall_score"].mean())
                    mean_prec = float(sub_t["precision_score"].mean())
                else:
                    mean_f1, mean_rec, mean_prec = 0.0, 0.0, 0.0

                p_rec[f"{t}_f1"] = round(mean_f1, 4)
                p_rec[f"{t}_rec"] = round(mean_rec, 4)
                p_rec[f"{t}_prec"] = round(mean_prec, 4)

            # Hitung delta peningkatan TrAdaBoost 75% terhadap Baseline RF
            base_f1 = p_rec.get("Baseline_Zero_Shot_RF_f1", 0.0)
            trada75_f1 = p_rec.get("TrAdaBoost_75_f1", 0.0)
            delta_f1 = trada75_f1 - base_f1
            p_rec["delta_tradaboost75_f1"] = round(delta_f1, 4)

            # Klasifikasi Tier
            if n_flaky == 0 or (rate < 0.2 and trada75_f1 < 0.05):
                tier = "Tier 4: Scarcity Sinkhole (Near-Zero Flaky)"
            elif trada75_f1 >= 0.50:
                tier = "Tier 1: High Responsiveness / Transfer (F1 >= 0.50)"
            elif delta_f1 >= 0.25:
                tier = "Tier 2: Target-Supervision Boosted (Delta >= +0.25)"
            else:
                tier = "Tier 3: Moderate Capability (F1 < 0.50)"

            p_rec["capability_tier"] = tier
            records.append(p_rec)

    res_df = pd.DataFrame(records)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        res_df.to_csv(output_csv, index=False)
        print(f" [SAVED] Profil kapabilitas proyek disimpan ke: {output_csv}")

    return res_df


def compute_four_aggregation_schemes(
    df_combined: pd.DataFrame,
    output_csv: Optional[str] = None
) -> pd.DataFrame:
    """
    Menghitung 4 skema agregasi matematis F1-Score:
    1. Unweighted Macro F1
    2. Active Macro F1 (proyek dengan flaky >= 1)
    3. Flaky-Weighted F1
    4. Global Micro F1 (dari akumulasi TP, FP, FN)
    """
    records = []

    datasets = df_combined["dataset"].unique().tolist()
    treatments = df_combined["treatment"].unique().tolist()

    for d in datasets:
        df_d = df_combined[df_combined["dataset"] == d]

        for t in treatments:
            df_dt = df_d[df_d["treatment"] == t]
            if len(df_dt) == 0:
                continue

            # Rata-rata per proyek melintasi seeds
            proj_summary = df_dt.groupby("project").agg({
                "f1_score": "mean",
                "recall_score": "mean",
                "precision_score": "mean",
                "total_project_tests": "first",
                "total_project_flaky": "first",
                "tp": "sum",
                "fp": "sum",
                "fn": "sum",
                "tn": "sum"
            }).reset_index()

            # 1. Unweighted Macro F1
            macro_f1 = float(proj_summary["f1_score"].mean())

            # 2. Active Macro F1
            active_sub = proj_summary[proj_summary["total_project_flaky"] > 0]
            active_f1 = float(active_sub["f1_score"].mean()) if len(active_sub) > 0 else 0.0

            # 3. Flaky-Weighted F1
            total_flaky = proj_summary["total_project_flaky"].sum()
            if total_flaky > 0:
                weighted_f1 = float(np.sum(proj_summary["f1_score"] * proj_summary["total_project_flaky"]) / total_flaky)
            else:
                weighted_f1 = 0.0

            # 4. Global Micro F1
            total_tp = proj_summary["tp"].sum()
            total_fp = proj_summary["fp"].sum()
            total_fn = proj_summary["fn"].sum()

            denom = (2 * total_tp + total_fp + total_fn)
            micro_f1 = float((2 * total_tp) / denom) if denom > 0 else 0.0
            micro_rec = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
            micro_prec = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0

            records.append({
                "dataset": d,
                "treatment": t,
                "unweighted_macro_f1": round(macro_f1, 4),
                "active_macro_f1": round(active_f1, 4),
                "flaky_weighted_f1": round(weighted_f1, 4),
                "global_micro_f1": round(micro_f1, 4),
                "global_micro_recall": round(micro_rec, 4),
                "global_micro_precision": round(micro_prec, 4),
                "n_projects_total": len(proj_summary),
                "n_projects_active": len(active_sub),
                "total_flaky_evaluated": int(total_flaky)
            })

    res_df = pd.DataFrame(records)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        res_df.to_csv(output_csv, index=False)
        print(f" [SAVED] 4 Skema Agregasi disimpan ke: {output_csv}")

    return res_df


def analyze_flakiness_rate_correlation(df_prof: pd.DataFrame) -> Dict[str, Any]:
    """
    Mengukur korelasi linear Pearson antara flakiness rate proyek (%) dengan performa model.
    """
    res = {}
    for d in df_prof["dataset"].unique():
        sub = df_prof[df_prof["dataset"] == d]
        rates = sub["flakiness_rate_pct"].values

        base_f1 = sub["Baseline_Zero_Shot_RF_f1"].values
        trada75_f1 = sub["TrAdaBoost_75_f1"].values

        r_base, p_base = pearsonr(rates, base_f1)
        r_trada, p_trada = pearsonr(rates, trada75_f1)

        res[d] = {
            "baseline_r": round(float(r_base), 4),
            "baseline_p": float(p_base),
            "tradaboost75_r": round(float(r_trada), 4),
            "tradaboost75_p": float(p_trada)
        }
    return res
