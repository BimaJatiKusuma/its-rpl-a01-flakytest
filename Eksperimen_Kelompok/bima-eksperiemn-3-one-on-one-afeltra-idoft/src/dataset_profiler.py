"""
Modul Profiling Karakteristik Dataset, Korelasi Empiris, dan Dekonstruksi Skema Agregasi.
Rujukan: Afeltra et al. (IEEE Access 2024), Alshammari et al. (ICSE 2021).
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats

def build_treatment_dataset_matrix(df_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Membangun Matriks Kemampuan Treatment vs Dataset:
    Baris = Treatment (Baseline, Burak, TrAdaBoost_25, TrAdaBoost_50, TrAdaBoost_75, Within)
    Kolom = 17 Proyek IDoFT
    Nilai = Rata-rata F1-Score lintas repetisi seed.
    """
    pivot = df_runs.pivot_table(
        index=["Treatment", "Model"],
        columns="Project",
        values="F1",
        aggfunc="mean"
    ).round(4)

    # Tambahkan rata-rata per treatment
    pivot["Rata-rata Treatment"] = pivot.mean(axis=1).round(4)

    # Tambahkan baris rata-rata kemampuan dataset
    avg_dataset = pivot.mean(axis=0).round(4)
    avg_dataset.name = ("RATA-RATA KEMAMPUAN DATASET", "-")
    pivot = pd.concat([pivot, pd.DataFrame([avg_dataset])])

    return pivot

def deconstruct_aggregation_schemes(df_runs: pd.DataFrame, top_treatment: str = "T5_TrAdaBoost_75", top_model: str = "M1") -> pd.DataFrame:
    """
    Mendekonstruksi 4 skema agregasi matematis F1-Score pada konfigurasi unggulan:
    1. Unweighted Macro F1 (17 Proyek)
    2. Active-Only Macro F1 (13 Proyek)
    3. Flaky-Weighted Average F1
    4. Global Instance-Level Micro F1
    """
    sub = df_runs[(df_runs["Treatment"] == top_treatment) & (df_runs["Model"] == top_model)].copy()

    # Hitung rata-rata per proyek lintas seeds
    proj_perf = sub.groupby("Project").agg({
        "F1": "mean",
        "Flaky_in_Test": "first",
        "Test_Size": "first",
        "TP": "sum",
        "FP": "sum",
        "FN": "sum",
        "TN": "sum"
    }).reset_index()

    # 1. Unweighted Macro F1 (17 Proyek)
    macro_17 = float(proj_perf["F1"].mean())

    # 2. Active-Only Macro F1 (13 Proyek)
    act_proj = proj_perf[proj_perf["Flaky_in_Test"] > 0]
    macro_13 = float(act_proj["F1"].mean())

    # 3. Flaky-Weighted Average F1
    total_flaky = act_proj["Flaky_in_Test"].sum()
    if total_flaky > 0:
        flaky_weighted = float(np.sum(act_proj["F1"] * act_proj["Flaky_in_Test"]) / total_flaky)
    else:
        flaky_weighted = 0.0

    # 4. Global Instance-Level Micro F1
    tot_tp = float(proj_perf["TP"].sum())
    tot_fp = float(proj_perf["FP"].sum())
    tot_fn = float(proj_perf["FN"].sum())
    if (2 * tot_tp + tot_fp + tot_fn) > 0:
        micro_f1 = float((2 * tot_tp) / (2 * tot_tp + tot_fp + tot_fn))
    else:
        micro_f1 = 0.0

    delta_macro_13 = ((macro_13 - macro_17) / macro_17 * 100.0) if macro_17 > 0 else 0.0
    delta_flaky_w = ((flaky_weighted - macro_17) / macro_17 * 100.0) if macro_17 > 0 else 0.0
    delta_micro = ((micro_f1 - macro_17) / macro_17 * 100.0) if macro_17 > 0 else 0.0

    df_decomp = pd.DataFrame([
        {
            "Skema_Agregasi": "1. Unweighted Macro F1 (17 Proyek)",
            "Nilai_F1": round(macro_17, 4),
            "Delta_vs_Macro_17_%": "Baseline (0.0%)",
            "Karakteristik": "Rata-rata aritmatika unweighted murni; proyek 0 flaky berbobot sama dengan proyek kaya flaky."
        },
        {
            "Skema_Agregasi": "2. Active-Only Macro F1 (13 Proyek)",
            "Nilai_F1": round(macro_13, 4),
            "Delta_vs_Macro_17_%": f"+{delta_macro_13:.1f}%",
            "Karakteristik": "Mengeksklusi 4 proyek tanpa flaky test riil di IDoFT; mengukur kemampuan pada target aktif."
        },
        {
            "Skema_Agregasi": "3. Flaky-Weighted Average F1",
            "Nilai_F1": round(flaky_weighted, 4),
            "Delta_vs_Macro_17_%": f"+{delta_flaky_w:.1f}%",
            "Karakteristik": "Pembobotan proporsional terhadap populasi flaky riil; mencerminkan efektivitas pada repositori skala besar."
        },
        {
            "Skema_Agregasi": "4. Global Instance-Level Micro F1",
            "Nilai_F1": round(micro_f1, 4),
            "Delta_vs_Macro_17_%": f"{delta_micro:.1f}%",
            "Karakteristik": "Akumulasi langsung TP, FP, FN global; presisi ditekan oleh false positive pada proyek masif."
        }
    ])

    return df_decomp

def analyze_flaky_rate_correlation(df_runs: pd.DataFrame, df_raw: pd.DataFrame) -> Dict[str, Any]:
    """
    Menganalisis korelasi statistik Pearson dan Spearman antara rasio flaky test per proyek terhadap F1-Score.
    """
    proj_summary = df_raw.groupby("project_clean").agg(
        Total_Tests=("flaky", "count"),
        Flaky_Tests=("flaky", "sum")
    ).reset_index()
    proj_summary["Flaky_Rate_%"] = (proj_summary["Flaky_Tests"] / proj_summary["Total_Tests"]) * 100.0

    # Rata-rata F1 per proyek pada Baseline
    sub_base = df_runs[df_runs["Treatment"] == "T1_Baseline"].groupby("Project")["F1"].mean().reset_index()
    merged = pd.merge(proj_summary, sub_base, left_on="project_clean", right_on="Project")

    # Filter proyek aktif untuk korelasi
    act = merged[merged["Flaky_Tests"] > 0]

    pearson_r, pearson_p = stats.pearsonr(act["Flaky_Rate_%"], act["F1"])
    spearman_rho, spearman_p = stats.spearmanr(act["Flaky_Rate_%"], act["F1"])

    return {
        "pearson_r": round(float(pearson_r), 4),
        "pearson_p": float(pearson_p),
        "spearman_rho": round(float(spearman_rho), 4),
        "spearman_p": float(spearman_p),
        "data_table": merged
    }
