"""
Modul Uji Signifikansi Statistik (Friedman Test & Nemenyi Post-hoc Test) untuk Eksperimen 2.
Mengikuti ACM/SIGSOFT Empirical Standards untuk analisis komparatif 24 konfigurasi pipeline.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata


def compute_critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    """
    Menghitung nilai Critical Difference (CD) untuk Nemenyi test.
    k: Jumlah perlakuan (24 pipeline)
    n: Jumlah fold proyek uji (24 proyek target)
    """
    # Nilai q_alpha studentized range statistic dibagi sqrt(2) untuk alpha=0.05
    q_alpha_dict = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        10: 3.164, 15: 3.410, 20: 3.540, 24: 3.630, 30: 3.738, 48: 3.935
    }
    q_alpha = q_alpha_dict.get(k, 3.630)
    cd = q_alpha * np.sqrt((k * (k + 1)) / (6.0 * n))
    return float(cd)


def run_statistical_analysis(
    raw_results_dir: str = None,
    metric_col: str = "F1",
    output_file: str = None
):
    """
    Melakukan Friedman Test dan Nemenyi Ranking pada metrik yang ditentukan (F1 / PR_AUC).
    """
    if raw_results_dir is None:
        raw_results_dir = str(Path(__file__).resolve().parent.parent / "results" / "raw_predictions")
    raw_path = Path(raw_results_dir)
    csv_files = sorted(list(raw_path.glob("*.csv")))

    if not csv_files:
        print(f"[STAT ERROR] Tidak ada file raw prediction ditemukan di {raw_path}.")
        return None

    matrix_dict = {}
    for f in csv_files:
        df = pd.read_csv(f)
        pipe_name = f.stem.replace("_folds", "")
        # Filter nilai NaN pada metrik (misal proyek jimfs) dengan fill 0 agar test valid
        df_metric = df.set_index("Target_Project")[metric_col].fillna(0.0)
        matrix_dict[pipe_name] = df_metric

    df_matrix = pd.DataFrame(matrix_dict).dropna()

    k = df_matrix.shape[1]
    n = df_matrix.shape[0]

    print(f"\n[STAT] Melakukan Uji Statistik: {k} Pipelines pada {n} Proyek Target...")

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

    if output_file is None:
        metric_tag = "prauc" if metric_col.lower().replace("_", "") == "prauc" else metric_col.lower()
        output_file = str(Path(__file__).resolve().parent.parent / "results" / f"friedman_nemenyi_{metric_tag}_results.csv")
    out_p = Path(output_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df_ranking.to_csv(out_p, index=False)
    print(f"[STAT] Hasil uji statistik disimpan ke: {out_p}")
    return df_ranking


if __name__ == "__main__":
    run_statistical_analysis(metric_col="F1")
    run_statistical_analysis(metric_col="PR_AUC")

