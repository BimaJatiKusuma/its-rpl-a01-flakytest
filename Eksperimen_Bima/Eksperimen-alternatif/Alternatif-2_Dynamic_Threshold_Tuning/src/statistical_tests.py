"""
Modul Uji Signifikansi Statistik (Paired Wilcoxon Signed-Rank Test & Cliff's Delta)
untuk Eksperimen Alternatif 2: Validation-Based Dynamic Threshold Tuning vs Default 0.5.
Mengikuti ACM/SIGSOFT Empirical Software Engineering Standards.
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
    Menghitung ukuran efek non-parametrik Cliff's Delta antara dua perlakuan:
    Delta = (#(y > x) - #(y < x)) / (n_x * n_y)
    Positif menandakan y (treatment) lebih unggul daripada x (baseline).
    """
    n_x = len(x)
    n_y = len(y)
    if n_x == 0 or n_y == 0:
        return 0.0

    greater = 0
    lesser = 0
    for yj in y:
        for xi in x:
            if yj > xi:
                greater += 1
            elif yj < xi:
                lesser += 1

    delta = (greater - lesser) / (n_x * n_y)
    return float(delta)


def run_wilcoxon_analysis(
    folds_csv_path: str = None,
    output_file: str = None,
    model_name: str = "xgboost_gpu"
) -> pd.DataFrame:
    """
    Menjalankan uji Wilcoxon Signed-Rank Test berpasangan (paired)
    antara strategi threshold T1_Default vs T2, T3, T4, T5, serta perbandingan antar strategi.
    """
    script_dir = Path(__file__).resolve().parent
    default_results_dir = script_dir.parent / "results"

    if folds_csv_path is None:
        folds_path = default_results_dir / "fold_thresholds.csv"
    else:
        folds_path = Path(folds_csv_path)

    if output_file is None:
        out_p = default_results_dir / "wilcoxon_test_results.csv"
    else:
        out_p = Path(output_file)

    df_folds = pd.read_csv(folds_path)

    if "Model" in df_folds.columns:
        df_target = df_folds[df_folds["Model"] == model_name].copy()
    else:
        df_target = df_folds.copy()

    comparisons = [
        ("T1_Default", "T2_Max_F1", "Default 0.5 vs Max-F1 Tuning"),
        ("T1_Default", "T3_Recall_Constrained", "Default 0.5 vs Recall>=70%"),
        ("T1_Default", "T4_Youdens_J", "Default 0.5 vs Youden's J"),
        ("T1_Default", "T5_Prior_Shifted", "Default 0.5 vs Prior-Shifted"),
        ("T2_Max_F1", "T3_Recall_Constrained", "Max-F1 vs Recall>=70%"),
        ("T2_Max_F1", "T4_Youdens_J", "Max-F1 vs Youden's J"),
        ("T2_Max_F1", "T5_Prior_Shifted", "Max-F1 vs Prior-Shifted"),
        ("T4_Youdens_J", "T5_Prior_Shifted", "Youden's J vs Prior-Shifted")
    ]

    metrics = ["F1", "Recall", "Precision", "MCC", "Specificity", "FPR"]
    results = []

    print("\n" + "=" * 80)
    print(f"[STAT] UJI SIGNIFIKANSI STATISTIK WILCOXON SIGNED-RANK TEST (Model: {model_name})")
    print("=" * 80)

    for base_id, treat_id, comp_name in comparisons:
        df_base = df_target[df_target["Strategy_ID"] == base_id].set_index("Target_Project")
        df_treat = df_target[df_target["Strategy_ID"] == treat_id].set_index("Target_Project")

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
            cliffs_d = compute_cliffs_delta(x, y)

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
                "Model": model_name,
                "Comparison": comp_name,
                "Base_Strategy": base_id,
                "Treatment_Strategy": treat_id,
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

            if metric in ["F1", "Recall", "Precision"]:
                print(f"[{comp_name:<30}] {metric:<10} | Diff: {mean_diff:+.4f} | p: {p_val:.4e} ({star}) | Cliff's d: {cliffs_d:+.3f} ({effect_label})")

    df_res = pd.DataFrame(results)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_p, index=False)
    print(f"\n[SAVED] Hasil lengkap Wilcoxon Signed-Rank Test tersimpan di: {out_p}")
    return df_res


if __name__ == "__main__":
    run_wilcoxon_analysis()
