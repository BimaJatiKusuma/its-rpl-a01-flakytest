"""
Modul Uji Signifikansi Statistik: Wilcoxon Signed-Rank Test, Cliff's Delta, dan Friedman / Nemenyi Test.
Rujukan: ACM/SIGSOFT Empirical Standards, Kitchenham et al. (2017), Afeltra et al. (IEEE Access 2024).
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from scipy import stats

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Menghitung ukuran efek non-parametrik Cliff's Delta (d).
    |d| < 0.147 (Negligible), < 0.33 (Small), < 0.474 (Medium), >= 0.474 (Large).
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 0.0

    more = sum(i > j for i in x for j in y)
    less = sum(i < j for i in x for j in y)
    d = (more - less) / (n1 * n2)
    return float(d)

def interpret_cliffs_delta(d: float) -> str:
    """Menginterpretasikan nilai Cliff's Delta."""
    abs_d = abs(d)
    if abs_d < 0.147:
        return "Negligible"
    elif abs_d < 0.330:
        return "Small"
    elif abs_d < 0.474:
        return "Medium"
    else:
        return "Large"

def run_wilcoxon_and_cliffs_tests(df_runs: pd.DataFrame, base_treatment: str = "T1_Baseline", model: str = "M1") -> pd.DataFrame:
    """
    Menjalankan uji non-parametrik berpasangan Wilcoxon Signed-Rank Test
    dan Cliff's Delta antara Baseline vs seluruh perlakuan lainnya pada model yang ditentukan.
    """
    sub = df_runs[df_runs["Model"] == model].copy()
    treatments = [t for t in sub["Treatment"].unique() if t != base_treatment]

    records = []
    # Ambil rata-rata F1 per proyek (13 proyek aktif)
    base_data = sub[sub["Treatment"] == base_treatment].groupby("Project")["F1"].mean()

    for tr in sorted(treatments):
        tr_data = sub[sub["Treatment"] == tr].groupby("Project")["F1"].mean()

        # Selaraskan proyek
        common_projs = [p for p in base_data.index if p in tr_data.index]
        x_base = base_data.loc[common_projs].values
        x_tr = tr_data.loc[common_projs].values

        diff = x_tr - x_base
        mean_base = float(np.mean(x_base))
        mean_tr = float(np.mean(x_tr))
        delta_mean = mean_tr - mean_base

        if np.all(diff == 0):
            p_val = 1.0
            stat = 0.0
        else:
            try:
                stat, p_val = stats.wilcoxon(x_base, x_tr, zero_method="pratt")
            except Exception:
                p_val = 1.0
                stat = 0.0

        d = cliffs_delta(x_tr, x_base)
        d_interp = interpret_cliffs_delta(d)
        sig = "Signifikan (*)" if p_val < 0.05 else "ns (tidak signifikan)"

        records.append({
            "Comparison": f"{base_treatment} vs {tr}",
            "Model": model,
            "Mean_Baseline": round(mean_base, 4),
            "Mean_Treatment": round(mean_tr, 4),
            "Delta_Gain": round(delta_mean, 4),
            "Wilcoxon_W": round(float(stat), 2),
            "p_value": float(p_val),
            "Significance_alpha_0.05": sig,
            "Cliffs_Delta_d": round(d, 4),
            "Effect_Size": d_interp
        })

    return pd.DataFrame(records)

def run_friedman_test(df_runs: pd.DataFrame, model: str = "M1") -> Dict[str, Any]:
    """
    Menjalankan uji omnibus Friedman Test lintas seluruh intervensi perlakuan.
    """
    sub = df_runs[df_runs["Model"] == model].copy()
    pivot = sub.pivot_table(index="Project", columns="Treatment", values="F1", aggfunc="mean")

    # Siapkan matriks data untuk scipy.stats.friedmanchisquare
    samples = [pivot[col].values for col in pivot.columns]
    try:
        stat, p_val = stats.friedmanchisquare(*samples)
    except Exception:
        stat, p_val = 0.0, 1.0

    ranks = pivot.rank(axis=1, ascending=False).mean().sort_values()

    return {
        "friedman_stat": round(float(stat), 4),
        "friedman_p": float(p_val),
        "average_ranks": ranks.to_dict(),
        "pivot_table": pivot
    }
