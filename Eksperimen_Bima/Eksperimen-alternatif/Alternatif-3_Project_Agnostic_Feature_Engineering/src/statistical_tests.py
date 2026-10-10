"""
Modul Uji Signifikansi Statistik (Wilcoxon Signed-Rank Test & Cliff's Delta)
serta Analisis Multikolinearitas (VIF / Variance Inflation Factor)
untuk Eksperimen Alternatif 3: Project-Agnostic Feature Engineering & Normalisasi Relatif.
Mengikuti standar empiris ACM/SIGSOFT Empirical Standards.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def compute_cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Menghitung effect size non-parametrik Cliff's Delta:
    Delta = (#(y > x) - #(y < x)) / (n_x * n_y)
    Positif menunjukkan y (treatment) lebih unggul daripada x (baseline).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
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

    return float((greater - lesser) / (n_x * n_y))


def interpret_cliffs_delta(delta: float) -> str:
    """Interpretasi besaran efek Cliff's Delta menurut Romano et al. (2006)."""
    abs_d = abs(delta)
    if abs_d < 0.147:
        magnitude = "Negligible"
    elif abs_d < 0.33:
        magnitude = "Small"
    elif abs_d < 0.474:
        magnitude = "Medium"
    else:
        magnitude = "Large"
    sign = "+" if delta > 0 else ("-" if delta < 0 else "0")
    return f"{magnitude} ({sign})"


def run_wilcoxon_analysis(
    folds_df: pd.DataFrame,
    output_csv_path: Optional[Path] = None,
    model_name: str = "xgboost_gpu"
) -> pd.DataFrame:
    """
    Menjalankan uji berpasangan Wilcoxon Signed-Rank Test dan Cliff's Delta
    pada performa LOPO-CV lintas fold 24 proyek.
    """
    if "Model" in folds_df.columns:
        df_target = folds_df[folds_df["Model"] == model_name].copy()
    else:
        df_target = folds_df.copy()

    comparisons = [
        ("F1_Original", "F2_Ratios_Only", "Set F1 (Original) vs Set F2 (Ratios)"),
        ("F1_Original", "F3_Project_Agnostic", "Set F1 (Original) vs Set F3 (Project-Agnostic)"),
        ("F1_Original", "F4_Feature_Selection", "Set F1 (Original) vs Set F4 (Selected F3)"),
        ("F2_Ratios_Only", "F3_Project_Agnostic", "Set F2 (Ratios) vs Set F3 (Project-Agnostic)"),
        ("F3_Project_Agnostic", "F4_Feature_Selection", "Set F3 (Project-Agnostic) vs Set F4 (Selected F3)")
    ]

    metrics = ["PR_AUC", "F1", "Recall", "Precision", "ROC_AUC", "MCC"]
    results = []

    print("\n" + "=" * 80)
    print(f"[STAT] UJI SIGNIFIKANSI STATISTIK WILCOXON & CLIFF'S DELTA (Model: {model_name})")
    print("=" * 80)

    for base_id, treat_id, comp_name in comparisons:
        df_base = df_target[df_target["Feature_Set"] == base_id].set_index("Target_Project")
        df_treat = df_target[df_target["Feature_Set"] == treat_id].set_index("Target_Project")

        common_projects = df_base.index.intersection(df_treat.index)

        for metric in metrics:
            if metric not in df_base.columns or metric not in df_treat.columns:
                continue

            s_base = df_base.loc[common_projects, metric].dropna()
            s_treat = df_treat.loc[common_projects, metric].dropna()
            valid_idx = s_base.index.intersection(s_treat.index)

            val_base = s_base.loc[valid_idx].values.astype(float)
            val_treat = s_treat.loc[valid_idx].values.astype(float)

            n_samples = len(valid_idx)
            if n_samples < 5:
                continue

            diff = val_treat - val_base
            mean_base = float(np.mean(val_base))
            mean_treat = float(np.mean(val_treat))
            mean_diff = float(np.mean(diff))

            # Uji Wilcoxon
            if np.all(diff == 0):
                stat, p_val = 0.0, 1.0
            else:
                try:
                    res = wilcoxon(val_treat, val_base, alternative="two-sided", zero_method="wilcox")
                    stat, p_val = float(res.statistic), float(res.pvalue)
                except Exception:
                    stat, p_val = np.nan, np.nan

            delta = compute_cliffs_delta(val_base, val_treat)
            delta_interp = interpret_cliffs_delta(delta)
            is_significant = (p_val < 0.05) if not np.isnan(p_val) else False

            results.append({
                "Model": model_name,
                "Comparison": comp_name,
                "Base_Set": base_id,
                "Treat_Set": treat_id,
                "Metric": metric,
                "N_Projects": n_samples,
                "Mean_Baseline": round(mean_base, 4),
                "Mean_Treatment": round(mean_treat, 4),
                "Mean_Difference": round(mean_diff, 4),
                "Wilcoxon_Stat": round(stat, 2) if not np.isnan(stat) else np.nan,
                "P_Value": round(p_val, 5) if not np.isnan(p_val) else np.nan,
                "Significant_p_005": is_significant,
                "Cliffs_Delta": round(delta, 4),
                "Effect_Size": delta_interp
            })

            sig_mark = "***" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns")
            print(f"[{comp_name[:35]:<35}] {metric:<10}: Base={mean_base:.4f} -> Treat={mean_treat:.4f} "
                  f"(Diff={mean_diff:+.4f}) | p={p_val:.4f} ({sig_mark}) | Delta={delta:+.3f} [{delta_interp}]")

    df_results = pd.DataFrame(results)
    if output_csv_path:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(output_csv_path, index=False)
        print(f"[STAT] Hasil uji statistik berhasil disimpan ke: {output_csv_path}")

    return df_results
