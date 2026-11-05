"""
Modul Uji Signifikansi Statistik dan Effect Size untuk Eksperimen TrAdaBoost Multi-Dataset.
Mencakup:
- Wilcoxon Signed-Rank Test
- Cliff's Delta Effect Size
- Friedman Omnibus Chi-Square Test
- Rata-rata Peringkat Nemenyi
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
    """
    Menghitung ukuran efek non-parametrik Cliff's Delta (d) dan mengklasifikasikan magnitudonya:
    |d| < 0.147: Negligible
    0.147 <= |d| < 0.33: Small
    0.33 <= |d| < 0.474: Medium
    |d| >= 0.474: Large (Romano et al., 2006)
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 0.0, "Negligible"

    # Perhitungan matriks perbandingan
    greater = 0
    less = 0
    for val1 in x:
        greater += np.sum(val1 > y)
        less += np.sum(val1 < y)

    delta = (greater - less) / (n1 * n2)
    abs_d = abs(delta)

    if abs_d < 0.147:
        magnitude = "Negligible"
    elif abs_d < 0.330:
        magnitude = "Small"
    elif abs_d < 0.474:
        magnitude = "Medium"
    else:
        magnitude = "Large"

    return float(delta), magnitude


def run_statistical_tests(
    df_combined: pd.DataFrame,
    metric: str = "f1_score",
    output_csv: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Menjalankan uji komparasi statistik antar perlakuan untuk masing-masing dataset.
    """
    pairwise_records = []
    friedman_records = []

    datasets = df_combined["dataset"].unique().tolist()

    for d in datasets:
        df_d = df_combined[df_combined["dataset"] == d]

        # Rata-rata metrik per proyek per treatment melintasi 5 seeds
        piv = df_d.groupby(["project", "treatment"])[metric].mean().unstack()

        treatments = [t for t in [
            "Baseline_Zero_Shot_RF",
            "Baseline_Zero_Shot_XGB",
            "TrAdaBoost_25",
            "TrAdaBoost_50",
            "TrAdaBoost_75",
            "Within_Project"
        ] if t in piv.columns]

        # 1. Friedman Omnibus Test
        matrix_data = [piv[t].values for t in treatments]
        try:
            stat, p_val = friedmanchisquare(*matrix_data)
            ranks = piv[treatments].rank(axis=1, ascending=False).mean()

            friedman_records.append({
                "dataset": d,
                "n_projects": len(piv),
                "metric": metric,
                "friedman_stat": round(float(stat), 4),
                "p_value": float(p_val),
                "is_significant": bool(p_val < 0.05),
                **{f"rank_{t}": round(float(ranks[t]), 2) for t in treatments}
            })
        except Exception as e:
            pass

        # 2. Pairwise Comparisons terhadap Baseline_Zero_Shot_RF
        baseline_col = "Baseline_Zero_Shot_RF"
        if baseline_col in piv.columns:
            base_vals = piv[baseline_col].values

            for comp_col in treatments:
                if comp_col == baseline_col:
                    continue

                comp_vals = piv[comp_col].values
                diff = comp_vals - base_vals

                # Wilcoxon signed-rank test
                if np.all(diff == 0):
                    stat_w, p_w = 0.0, 1.0
                else:
                    try:
                        stat_w, p_w = wilcoxon(comp_vals, base_vals, zero_method="wilcox", alternative="two-sided")
                    except Exception:
                        stat_w, p_w = 0.0, 1.0

                # Cliff's Delta
                d_val, mag = cliffs_delta(comp_vals, base_vals)

                pairwise_records.append({
                    "dataset": d,
                    "metric": metric,
                    "baseline": baseline_col,
                    "comparison": comp_col,
                    "mean_baseline": round(float(np.mean(base_vals)), 4),
                    "mean_comparison": round(float(np.mean(comp_vals)), 4),
                    "mean_difference": round(float(np.mean(diff)), 4),
                    "wilcoxon_w": round(float(stat_w), 2),
                    "p_value": float(p_w),
                    "is_significant_p05": bool(p_w < 0.05),
                    "cliffs_delta": round(float(d_val), 4),
                    "cliffs_magnitude": mag
                })

    df_pairwise = pd.DataFrame(pairwise_records)
    df_friedman = pd.DataFrame(friedman_records)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_pairwise.to_csv(output_csv, index=False)
        f_csv = output_csv.replace(".csv", "_friedman.csv")
        df_friedman.to_csv(f_csv, index=False)
        print(f" [SAVED] Hasil uji statistik disimpan ke: {output_csv} dan {f_csv}")

    return df_pairwise, df_friedman


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    comb_csv = base_dir / "results" / "raw" / "all_datasets_combined_predictions.csv"
    if comb_csv.exists():
        df_c = pd.read_csv(comb_csv)
        out_csv = str(base_dir / "results" / "summary" / "statistical_tests_results.csv")
        run_statistical_tests(df_c, metric="f1_score", output_csv=out_csv)
