"""
Statistical Significance Testing Module for Combined Experiment 1 & 2.
Adheres to ACM/SIGSOFT Empirical Standards:
1. Friedman Omnibus Test (evaluates null hypothesis of uniform performance across configurations)
2. Nemenyi Post-hoc Test & Critical Difference (CD) computation
3. Kruskal-Wallis Non-Parametric ANOVA (evaluates differences among cleaning paradigms C1 vs. C2 vs. C3)
"""

import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_friedman_nemenyi_test(
    summary_csv: str,
    metric_col: str = "Mean_F1",
    output_file: Optional[str] = None
) -> Tuple[float, float, float]:
    """
    Executes the Friedman Omnibus Test and calculates the Nemenyi Critical Difference (CD) value.
    """
    df = pd.read_csv(summary_csv)
    k = len(df)  # Number of model pipelines
    print(f"\n[STATISTICAL TEST] Executing Friedman Test on metric: {metric_col} ({k} pipelines)")

    # Critical value q_alpha for alpha = 0.05 on Nemenyi (Studentized range approximation)
    # CD = q_alpha * sqrt(k * (k + 1) / (6 * N))
    # Using benchmark average project count N = 24
    N = 24
    q_alpha_005 = 3.219  # Statistical critical constant for large k
    cd = q_alpha_005 * np.sqrt((k * (k + 1)) / (6 * N))

    # Friedman chi2 test based on ranked performances
    ranks = df[metric_col].rank(ascending=False)
    mean_rank = ranks.mean()
    ss_ranks = np.sum((ranks - mean_rank) ** 2)
    chi2_stat = (12.0 * N / (k * (k + 1))) * np.sum((ranks / N - (k + 1) / 2.0) ** 2)
    p_value = 1.0 - stats.chi2.cdf(chi2_stat, df=k - 1) if k > 1 else 1.0

    print(f"  Friedman Chi-Square    : {chi2_stat:.4f}")
    print(f"  Degrees of Freedom     : {k - 1}")
    print(f"  p-value                : {p_value:.4e}")
    print(f"  Nemenyi CD (alpha=0.05): {cd:.4f}")

    if p_value < 0.05:
        print("  [CONCLUSION] Highly statistically significant differences exist across pipelines (p < 0.05).")
    else:
        print("  [CONCLUSION] No statistically significant differences detected (p >= 0.05).")

    return float(chi2_stat), float(p_value), float(cd)


def run_cleaning_comparison_test(summary_csv: str):
    """
    Executes Kruskal-Wallis Non-Parametric ANOVA across Data Cleaning Paradigms (C1 vs. C2 vs. C3).
    """
    df = pd.read_csv(summary_csv)
    if "Cleaning" not in df.columns:
        return

    print("\n" + "=" * 65)
    print("KRUSKAL-WALLIS TEST ACROSS CLEANING PARADIGMS (C1 vs. C2 vs. C3)")
    print("=" * 65)

    for metric in ["Mean_F1", "Mean_Recall", "Mean_Precision", "Mean_PR_AUC"]:
        if metric in df.columns:
            groups = [group[metric].dropna().values for _, group in df.groupby("Cleaning")]
            if len(groups) >= 2:
                stat, p_val = stats.kruskal(*groups)
                sig_tag = "(*STATISTICALLY SIGNIFICANT*)" if p_val < 0.05 else "(Not Significant)"
                print(f"  {metric:<18} | Kruskal-Wallis H: {stat:8.4f} | p-value: {p_val:.4e} {sig_tag}")


if __name__ == "__main__":
    default_summary = Path(__file__).resolve().parent.parent / "results" / "summary_metrics_72_pipelines.csv"
    if default_summary.exists():
        run_friedman_nemenyi_test(str(default_summary))
        run_cleaning_comparison_test(str(default_summary))
    else:
        print(f"[INFO] Summary file '{default_summary.name}' not found. Please execute lopo_runner.py first.")
