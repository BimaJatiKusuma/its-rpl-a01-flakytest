"""
Modul Uji Signifikansi Statistik (Wilcoxon Signed-Rank Test & Effect Size).
Mengikuti ACM/SIGSOFT Empirical Software Engineering Standards untuk evaluasi komparasi model.
Rujukan:
- Kitchenham et al. (2017): Robust statistical methods in empirical software engineering.
- Wilcoxon (1945): Individual comparisons by ranking methods.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def compute_cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Menghitung ukuran efek non-parametrik Cliff's Delta antara dua sampel berpasangan.
    Delta = (#(x > y) - #(x < y)) / (n_x * n_y)
    """
    n_x = len(x)
    n_y = len(y)
    if n_x == 0 or n_y == 0:
        return 0.0

    greater = 0
    lesser = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                greater += 1
            elif xi < yj:
                lesser += 1

    delta = (greater - lesser) / (n_x * n_y)
    return float(delta)


def run_wilcoxon_analysis(
    folds_csv_path: str = None,
    output_file: str = None
) -> pd.DataFrame:
    """
    Menjalankan uji Wilcoxon Signed-Rank Test berpasangan (paired)
    pada metrik F1, PR-AUC, Recall, Precision, dan ROC-AUC antar konfigurasi transfer learning.
    """
    script_dir = Path(__file__).resolve().parent
    default_results_dir = script_dir.parent / "results"

    if folds_csv_path is None:
        folds_path = default_results_dir / "transfer_learning_folds.csv"
    else:
        folds_path = Path(folds_csv_path)

    if output_file is None:
        out_p = default_results_dir / "wilcoxon_test_results.csv"
    else:
        out_p = Path(output_file)

    df_folds = pd.read_csv(folds_path)

    comparisons = [
        ("A_Baseline", "B_Burak_Filter", "Baseline vs Burak Filter"),
        ("A_Baseline", "C_TrAdaBoost", "Baseline vs TrAdaBoost"),
        ("A_Baseline", "D_Burak_TrAdaBoost_Hybrid", "Baseline vs Hybrid (Burak+TrAdaBoost)"),
        ("B_Burak_Filter", "C_TrAdaBoost", "Burak Filter vs TrAdaBoost"),
        ("B_Burak_Filter", "D_Burak_TrAdaBoost_Hybrid", "Burak Filter vs Hybrid"),
        ("C_TrAdaBoost", "D_Burak_TrAdaBoost_Hybrid", "TrAdaBoost vs Hybrid")
    ]

    metrics = ["F1", "PR_AUC", "Recall", "Precision", "ROC_AUC", "MCC"]
    results = []

    print("\n" + "=" * 80)
    print("[STAT] MELAKUKAN UJI SIGNIFIKANSI STATISTIK (WILCOXON SIGNED-RANK TEST)")
    print("=" * 80)

    for base_id, treat_id, comp_name in comparisons:
        df_base = df_folds[df_folds["Config_ID"] == base_id].set_index("Target_Project")
        df_treat = df_folds[df_folds["Config_ID"] == treat_id].set_index("Target_Project")

        common_projects = df_base.index.intersection(df_treat.index)

        for metric in metrics:
            val_base = df_base.loc[common_projects, metric]
            val_treat = df_treat.loc[common_projects, metric]

            valid_mask = (~val_base.isna()) & (~val_treat.isna())
            x = val_base[valid_mask].values.astype(float)
            y = val_treat[valid_mask].values.astype(float)

            diff = y - x
            n_pairs = len(diff)

            if np.all(diff == 0) or n_pairs < 5:
                stat = np.nan
                p_val = 1.0
            else:
                try:
                    res = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
                    stat = float(res.statistic)
                    p_val = float(res.pvalue)
                except Exception:
                    stat = np.nan
                    p_val = 1.0

            mean_diff = float(np.mean(diff)) if len(diff) > 0 else 0.0
            median_diff = float(np.median(diff)) if len(diff) > 0 else 0.0
            cliffs_d = compute_cliffs_delta(y, x)

            abs_d = abs(cliffs_d)
            if abs_d < 0.147:
                effect_label = "Negligible"
            elif abs_d < 0.33:
                effect_label = "Small"
            elif abs_d < 0.474:
                effect_label = "Medium"
            else:
                effect_label = "Large"

            is_sig = bool(p_val < 0.05)
            star = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))

            results.append({
                "Comparison": comp_name,
                "Base_Config": base_id,
                "Treatment_Config": treat_id,
                "Metric": metric,
                "N_Pairs": n_pairs,
                "Base_Mean": float(np.mean(x)),
                "Treatment_Mean": float(np.mean(y)),
                "Mean_Diff": mean_diff,
                "Median_Diff": median_diff,
                "Wilcoxon_Stat": stat,
                "P_Value": p_val,
                "Significance": star,
                "Is_Significant": is_sig,
                "Cliffs_Delta": cliffs_d,
                "Effect_Size": effect_label
            })

            if metric in ["F1", "PR_AUC", "Recall"]:
                print(f"[{comp_name:<32}] {metric:<7} | Diff: {mean_diff:+.4f} | p: {p_val:.4e} ({star}) | Cliff's d: {cliffs_d:+.3f} ({effect_label})")

    df_res = pd.DataFrame(results)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_p, index=False)
    print(f"\n[SAVED] Hasil lengkap Wilcoxon Signed-Rank Test tersimpan di: {out_p}")
    return df_res


if __name__ == "__main__":
    run_wilcoxon_analysis()
