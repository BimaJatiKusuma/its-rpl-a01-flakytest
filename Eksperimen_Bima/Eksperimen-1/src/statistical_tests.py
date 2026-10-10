"""
Modul Uji Signifikansi Statistik (Friedman Test & Nemenyi Post-hoc Test).
Mengikuti ACM/SIGSOFT Empirical Standards untuk analisis komparatif performa model.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata


def compute_critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    """
    Menghitung nilai Critical Difference (CD) untuk Nemenyi test.
    k: Jumlah perlakuan (jumlah pipeline)
    n: Jumlah dataset / fold uji (jumlah target project)
    """
    # Nilai q_alpha aproksimasi untuk Nemenyi pada alpha=0.05
    # Untuk k besar, q_alpha berada di kisaran 3.0 - 4.0
    # Berdasarkan tabel Studentized Range Statistic / sqrt(2)
    q_alpha_dict = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        10: 3.164, 20: 3.540, 30: 3.738, 40: 3.865, 48: 3.935
    }
    q_alpha = q_alpha_dict.get(k, 3.935)
    cd = q_alpha * np.sqrt((k * (k + 1)) / (6.0 * n))
    return float(cd)


def run_statistical_analysis(
    raw_results_dir: str = "../results/raw_predictions",
    metric_col: str = "PR_AUC",
    output_file: str = "../results/friedman_nemenyi_results.csv"
):
    """
    Melakukan Friedman Test dan Nemenyi Ranking pada metrik yang ditentukan.
    """
    raw_path = Path(raw_results_dir)
    csv_files = sorted(list(raw_path.glob("*.csv")))

    if not csv_files:
        print(f"Tidak ada file raw prediction ditemukan di {raw_path}.")
        return None

    matrix_dict = {}
    projects = None

    for f in csv_files:
        df = pd.read_csv(f)
        pipe_name = f.stem
        # Filter nilai NaN pada metrik (misal proyek jimfs) dengan fill 0 agar test valid
        df_metric = df.set_index("Target_Project")[metric_col].fillna(0.0)
        matrix_dict[pipe_name] = df_metric
        if projects is None:
            projects = df_metric.index.tolist()

    df_matrix = pd.DataFrame(matrix_dict)
    # Hapus proyek yang tidak beririsan jika ada (C1 vs C2)
    df_matrix = df_matrix.dropna()

    k = df_matrix.shape[1]
    n = df_matrix.shape[0]

    print(f"\n[STAT] Melakukan Uji Statistik: {k} Pipelines pada {n} Proyek Bersama...")

    # 1. Friedman Test
    stat, p_val = friedmanchisquare(*[df_matrix[col] for col in df_matrix.columns])
    print(f"Friedman Test ({metric_col}): Chi-squared = {stat:.4f}, p-value = {p_val:.4e}")

    # 2. Ranking per proyek (Ranking 1 = Performa Terbaik)
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

    out_p = Path(output_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df_ranking.to_csv(out_p, index=False)
    print(f"[SAVED] Hasil analisis statistik disimpan ke: {out_p}")
    return df_ranking


if __name__ == "__main__":
    print("Modul statistical_tests siap digunakan.")
