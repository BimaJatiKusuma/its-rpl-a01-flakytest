"""
Modul Uji Signifikansi Statistik (Friedman Test & Nemenyi Post-hoc Test) untuk Eksperimen 3.
Mengevaluasi signifikansi perbedaan performa 24 pipelines pada data hasil perulangan LOPO-CV (5 Repetisi).
Mengikuti ACM/SIGSOFT Empirical Standards.
"""

from pathlib import Path
from typing import Optional
import sys
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def compute_critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    """
    Menghitung nilai Critical Difference (CD) untuk Nemenyi test.
    k: Jumlah perlakuan (24 pipeline)
    n: Jumlah fold proyek uji (24 proyek target)
    """
    q_alpha_dict = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        10: 3.164, 15: 3.410, 20: 3.540, 24: 3.630, 30: 3.738, 48: 3.935
    }
    q_alpha = q_alpha_dict.get(k, 3.630)
    cd = q_alpha * np.sqrt((k * (k + 1)) / (6.0 * n))
    return float(cd)


def run_repeated_statistical_analysis(
    all_folds_csv: Optional[str] = None,
    metric_col: str = "F1",
    output_dir: Optional[str] = None
):
    """
    Melakukan Friedman Test dan Nemenyi Ranking pada metrik yang dirata-ratakan lintas 5 random seeds.
    """
    if all_folds_csv is None:
        all_folds_csv = str(Path(__file__).resolve().parent.parent / "results" / "all_repeated_folds_predictions.csv")
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent.parent / "results")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_all = pd.read_csv(all_folds_csv)

    # Rata-ratakan metrik lintas 5 random seeds untuk setiap (Pipeline_ID, Target_Project)
    df_mean_folds = df_all.groupby(["Target_Project", "Pipeline_ID"])[metric_col].mean().reset_index()

    # Pivot: Index = Target_Project, Columns = Pipeline_ID
    df_matrix = df_mean_folds.pivot(index="Target_Project", columns="Pipeline_ID", values=metric_col).fillna(0.0)

    k = df_matrix.shape[1]
    n = df_matrix.shape[0]

    print(f"\n[STAT REPEATED] Uji Statistik: {k} Pipelines pada {n} Proyek Target (Dirata-ratakan lintas Seeds)...")

    # 1. Friedman Test
    stat, p_val = friedmanchisquare(*[df_matrix[col] for col in df_matrix.columns])
    print(f"Friedman Test ({metric_col}): Chi-squared = {stat:.4f}, p-value = {p_val:.4e}")

    # 2. Ranking per proyek (Ranking 1 = Performa Tertinggi)
    ranks = df_matrix.apply(lambda row: rankdata(-row, method="average"), axis=1, result_type="expand")
    ranks.columns = df_matrix.columns
    avg_ranks = ranks.mean().sort_values()

    # 3. Critical Difference (CD)
    cd = compute_critical_difference(k, n, alpha=0.05)
    print(f"Nemenyi Critical Difference (CD at alpha=0.05): {cd:.4f}")

    df_ranking = pd.DataFrame({
        "Pipeline": avg_ranks.index,
        "Average_Rank": avg_ranks.values,
        "CD_Threshold": cd,
        "Friedman_Stat": stat,
        "Friedman_PValue": p_val,
        "Significant": (p_val < 0.05)
    })

    metric_tag = "prauc" if metric_col.lower().replace("_", "") == "prauc" else metric_col.lower()
    output_file = out_path / f"friedman_nemenyi_{metric_tag}_repeated_results.csv"
    df_ranking.to_csv(output_file, index=False)
    print(f"[STAT REPEATED] Hasil peringkat Nemenyi disimpan ke: {output_file}")
    return df_ranking


if __name__ == "__main__":
    run_repeated_statistical_analysis(metric_col="F1")
    run_repeated_statistical_analysis(metric_col="PR_AUC")
