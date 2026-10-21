"""
Modul Profiling & Analisis Kemampuan pada Masing-Masing Dataset (24 Proyek) untuk Eksperimen 3.
Menjawab secara mendalam:
1. Mengapa rata-rata nilai akurasi / F1 sangat rendah pada level makro lintas proyek?
2. Dataset / Proyek mana saja yang memiliki nilai konsisten dan stabil lintas perulangan dan konfigurasi?
3. Dataset / Proyek mana saja yang menjadi 'sinkhole' non-deteksi akibat ketimpangan ekstrem atau domain shift?
"""

from pathlib import Path
from typing import Optional, Dict, Tuple
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def profile_all_datasets(
    all_folds_csv: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Menganalisis performa dan stabilitas setiap proyek dari ke-24 proyek target.
    Data dievaluasi dari 5 Repetisi x 24 Pipelines = 120 evaluasi per proyek.
    """
    if all_folds_csv is None:
        all_folds_csv = str(Path(__file__).resolve().parent.parent / "results" / "all_repeated_folds_predictions.csv")
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent.parent / "results")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_folds = pd.read_csv(all_folds_csv)

    print("\n" + "=" * 90)
    print("🔬 ANALISIS KEMAMPUAN, KONSISTENSI, & ROOT-CAUSE AKURASI PADA SETIAP DATASET (24 PROYEK)")
    print("=" * 90)

    dataset_records = []
    projects = sorted(df_folds["Target_Project"].unique().tolist())

    for proj in projects:
        sub = df_folds[df_folds["Target_Project"] == proj]
        test_size = int(sub["Test_Size"].iloc[0])
        flaky_count = int(sub["Flaky_in_Test"].iloc[0])
        non_flaky = int(sub["NonFlaky_in_Test"].iloc[0])
        flaky_rate = (flaky_count / test_size * 100.0) if test_size > 0 else 0.0

        # Metrik agregat seluruh 120 evaluasi (24 pipeline x 5 seeds)
        f1_vals = sub["F1"].fillna(0.0).values
        rec_vals = sub["Recall"].fillna(0.0).values
        prec_vals = sub["Precision"].fillna(0.0).values
        prauc_vals = sub["PR_AUC"].dropna().values
        roc_vals = sub["ROC_AUC"].dropna().values

        mean_f1 = float(np.mean(f1_vals))
        std_f1 = float(np.std(f1_vals, ddof=1))
        median_f1 = float(np.median(f1_vals))
        max_f1 = float(np.max(f1_vals))
        min_f1 = float(np.min(f1_vals))
        zero_f1_count = int(np.sum(f1_vals == 0.0))
        zero_f1_rate = (zero_f1_count / len(f1_vals)) * 100.0

        mean_rec = float(np.mean(rec_vals))
        mean_prec = float(np.mean(prec_vals))
        mean_prauc = float(np.mean(prauc_vals)) if len(prauc_vals) > 0 else np.nan
        mean_roc = float(np.mean(roc_vals)) if len(roc_vals) > 0 else np.nan

        # Temukan Pipeline Terbaik untuk Proyek ini (rata-rata F1 tertinggi lintas 5 seed)
        pipe_group = sub.groupby("Pipeline_ID")["F1"].mean()
        best_pipe = pipe_group.idxmax()
        best_pipe_f1 = float(pipe_group.max())

        # Kestabilan seed pada pipeline terbaik tersebut
        sub_best = sub[sub["Pipeline_ID"] == best_pipe]
        best_pipe_seed_std = float(sub_best["F1"].std(ddof=1)) if len(sub_best) > 1 else 0.0
        best_pipe_cv = (best_pipe_seed_std / best_pipe_f1 * 100.0) if best_pipe_f1 > 0 else 0.0

        # Klasifikasi Tier Konsistensi & Kemampuan
        if flaky_count == 0:
            tier = "Tier 4: Zero Minority Class (Baseline Jimfs)"
            tier_code = "T4"
        elif zero_f1_rate >= 80.0:
            tier = "Tier 4: Permanent Non-Detection Sinkhole (Extreme Skew / Shift)"
            tier_code = "T4"
        elif mean_f1 >= 0.30 and zero_f1_rate <= 15.0:
            tier = "Tier 1: High Detectability & Rock-Solid Consistency"
            tier_code = "T1"
        elif best_pipe_f1 >= 0.30 and zero_f1_rate > 15.0:
            tier = "Tier 2: Resampling-Dependent Consistency (High on RUS)"
            tier_code = "T2"
        else:
            tier = "Tier 3: Moderate & Fragile Detection (High Sensitivity)"
            tier_code = "T3"

        dataset_records.append({
            "Target_Project": proj,
            "Test_Size": test_size,
            "Flaky_Count": flaky_count,
            "NonFlaky_Count": non_flaky,
            "Flaky_Rate_%": flaky_rate,
            "Total_Evaluations": len(sub),
            "Overall_Mean_F1": mean_f1,
            "Overall_Std_F1": std_f1,
            "Overall_Median_F1": median_f1,
            "Overall_Max_F1": max_f1,
            "Overall_Min_F1": min_f1,
            "Overall_Mean_Recall": mean_rec,
            "Overall_Mean_Precision": mean_prec,
            "Overall_Mean_PR_AUC": mean_prauc,
            "Overall_Mean_ROC_AUC": mean_roc,
            "Zero_F1_Count": zero_f1_count,
            "Zero_F1_Rate_%": zero_f1_rate,
            "Best_Pipeline": best_pipe,
            "Best_Pipeline_F1": best_pipe_f1,
            "Best_Pipe_Seed_Std": best_pipe_seed_std,
            "Best_Pipe_Seed_CV_%": best_pipe_cv,
            "Tier_Code": tier_code,
            "Tier_Description": tier
        })

    df_profile = pd.DataFrame(dataset_records)
    # Urutkan berdasarkan Flaky_Rate_% menurun
    df_profile.sort_values(by="Flaky_Rate_%", ascending=False, inplace=True)
    
    out_profile_csv = out_path / "dataset_capability_profile.csv"
    df_profile.to_csv(out_profile_csv, index=False)
    print(f"[OK] Profil kemampuan dataset disimpan ke: {out_profile_csv}")

    # Analisis Korelasi Statistik: Flaky Rate vs Mean F1
    valid_sub = df_profile[df_profile["Flaky_Count"] > 0]
    p_corr, p_pval = pearsonr(valid_sub["Flaky_Rate_%"], valid_sub["Overall_Mean_F1"])
    s_corr, s_pval = spearmanr(valid_sub["Flaky_Rate_%"], valid_sub["Overall_Mean_F1"])

    print("\n--- ANALISIS KORELASI TINGKAT FLAKY TEST vs KINERJA PREDIKSI ---")
    print(f"Pearson Correlation (Flaky Rate % vs Overall Mean F1): r = {p_corr:.4f} (p = {p_pval:.4e})")
    print(f"Spearman Rank Correlation: rho = {s_corr:.4f} (p = {s_pval:.4e})")

    # Analisis Dampak Macro-Average vs Weighted-Average vs Micro-Average
    # Pada Top Pipeline P13_S2_B3_M1 across 5 seeds
    top_p13 = df_folds[df_folds["Pipeline_ID"] == "P13_S2_B3_M1"]
    macro_f1 = top_p13.groupby("Seed")["F1"].mean().mean()
    
    # Weighted Average (berbobot Flaky_Count)
    def calc_weighted_f1(seed_group):
        weights = seed_group["Flaky_in_Test"]
        if weights.sum() == 0:
            return 0.0
        return np.sum(seed_group["F1"].fillna(0.0) * weights) / weights.sum()

    weighted_f1 = top_p13.groupby("Seed").apply(calc_weighted_f1).mean()

    # Micro-Average: Total TP, FP, FN global across all 24 projects
    def calc_micro_f1(seed_group):
        tot_tp = seed_group["TP"].sum()
        tot_fp = seed_group["FP"].sum()
        tot_fn = seed_group["FN"].sum()
        prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
        rec = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        return f1

    micro_f1 = top_p13.groupby("Seed").apply(calc_micro_f1).mean()

    print("\n--- DEKONSTRUKSI MATEMATIS NILAI AKURASI/F1 YANG RENDAH (Top Pipeline P13) ---")
    print(f"1. Unweighted Macro-Average F1 : {macro_f1:.4f} (Setiap proyek berbobot sama 1/24)")
    print(f"2. Flaky-Weighted Average F1   : {weighted_f1:.4f} (Berbobot jumlah flaky test per proyek)")
    print(f"3. Global Instance-Level Micro F1: {micro_f1:.4f} (Menghitung total TP, FP, FN keseluruhan)")
    print(f"   Delta Peningkatan (Weighted vs Macro): +{(weighted_f1 - macro_f1)/macro_f1 * 100:.2f}%!")

    # Identifikasi Proyek Sinkhole
    sinkholes = df_profile[df_profile["Zero_F1_Rate_%"] >= 75.0]
    print(f"\nProyek Sinkhole (Zero F1 Rate >= 75%): {len(sinkholes)} Proyek")
    for _, s in sinkholes.iterrows():
        print(f"  - {s['Target_Project']:<18}: {s['Flaky_Count']:>3} flaky / {s['Test_Size']:>5} tests ({s['Flaky_Rate_%']:.2f}%) | Zero-F1: {s['Zero_F1_Rate_%']:.1f}%")

    analysis_dict = {
        "pearson_r": p_corr,
        "pearson_p": p_pval,
        "spearman_rho": s_corr,
        "spearman_p": s_pval,
        "macro_f1_p13": macro_f1,
        "weighted_f1_p13": weighted_f1,
        "micro_f1_p13": micro_f1,
        "num_sinkholes": len(sinkholes)
    }

    # Simpan ringkasan analisis metrik
    df_comp = pd.DataFrame([{
        "Metric_Aggregation": "Unweighted Macro F1",
        "Score": macro_f1,
        "Rationale": "Rata-rata aritmatika murni: proyek 1 flaky (assertj) berbobot sama dengan 163 flaky (spring-boot)"
    }, {
        "Metric_Aggregation": "Flaky-Weighted F1",
        "Score": weighted_f1,
        "Rationale": "Berbobot proporsi flaky test riil: proyek enterprise kaya flaky mendominasi evaluasi"
    }, {
        "Metric_Aggregation": "Global Micro F1",
        "Score": micro_f1,
        "Rationale": "Berbasis confusion matrix global: mengukur ketepatan total test case flaky yang berhasil ditangkap"
    }])
    df_comp.to_csv(out_path / "macro_vs_weighted_f1_explanation.csv", index=False)

    # Matriks Kemampuan 24 Pipeline vs 24 Dataset
    pivot_matrix = df_folds.groupby(["Pipeline_ID", "Target_Project"])["F1"].mean().unstack()
    pipe_sorted = sorted(df_folds["Pipeline_ID"].unique(), key=lambda x: int(x.split("_")[0][1:]))
    proj_sorted = sorted(df_folds["Target_Project"].unique())
    pivot_matrix = pivot_matrix.reindex(index=pipe_sorted, columns=proj_sorted)

    ds_mean = pivot_matrix.mean(axis=0)
    pipe_mean = pivot_matrix.mean(axis=1)

    pivot_export = pivot_matrix.copy()
    pivot_export["Rata-rata Pipeline"] = pipe_mean
    pivot_export.loc["RATA-RATA KEMAMPUAN DATASET"] = ds_mean.to_dict()
    pivot_export.loc["RATA-RATA KEMAMPUAN DATASET", "Rata-rata Pipeline"] = ds_mean.mean()

    matrix_csv_path = out_path / "matrix_pipeline_vs_dataset_f1.csv"
    pivot_export.round(4).to_csv(matrix_csv_path, index_label="Pipeline")
    print(f"\n[Matrix Exported] Matriks 24 Pipeline vs 24 Dataset disimpan di: {matrix_csv_path}")

    return df_profile, analysis_dict


if __name__ == "__main__":
    profile_all_datasets()
