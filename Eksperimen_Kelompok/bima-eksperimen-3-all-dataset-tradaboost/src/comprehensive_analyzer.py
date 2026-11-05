"""
Modul Analisis Komprehensif Seluruh Kombinasi Pipeline
Menganalisis hasil dari comprehensive_all_pipelines_predictions.csv melintasi 6 faktor:
1. Data Cleaning (RowDrop vs Imputed)
2. Feature Scaling (S1, S2, S3)
3. Imbalance Handling (B1, B2, B3, B4)
4. Classifier Models (RF, XGB, DT)
5. Sampling & Partitioning (0% -> 100%)
6. TrAdaBoost Configs (Depth, Iterations, Voting, Base Model)
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

def analyze_all_pipelines():
    base_dir = Path(__file__).resolve().parent.parent
    raw_csv = base_dir / "results" / "raw" / "comprehensive_all_pipelines_predictions.csv"
    sum_dir = base_dir / "results" / "summary"
    os.makedirs(sum_dir, exist_ok=True)

    if not raw_csv.exists():
        print(f"File {raw_csv} tidak ditemukan!")
        return

    df = pd.read_csv(raw_csv)
    print(f"Total baris data evaluasi: {len(df)}")
    print(f"Dataset unik: {df['dataset'].unique().tolist()}")
    print(f"Perlakuan unik: {len(df['treatment'].unique())}")

    # 1. Rangkuman Global per Dataset dan Perlakuan
    metrics = ["f1_score", "recall_score", "precision_score", "pr_auc", "roc_auc", "mcc", "balanced_accuracy"]
    
    summary_list = []
    for (dname, trt), grp in df.groupby(["dataset", "treatment"]):
        rec = {
            "dataset": dname,
            "treatment": trt,
            "pipeline_group": grp["pipeline_group"].iloc[0] if "pipeline_group" in grp.columns else "General",
            "n_folds": len(grp),
            "macro_f1": round(grp["f1_score"].mean(), 4),
            "std_f1": round(grp["f1_score"].std(), 4),
            "cv_pct_f1": round((grp["f1_score"].std() / grp["f1_score"].mean()) * 100, 2) if grp["f1_score"].mean() > 0 else 0.0,
            "macro_recall": round(grp["recall_score"].mean(), 4),
            "macro_precision": round(grp["precision_score"].mean(), 4),
            "macro_pr_auc": round(grp["pr_auc"].mean(), 4),
            "macro_roc_auc": round(grp["roc_auc"].mean(), 4),
            "macro_mcc": round(grp["mcc"].mean(), 4),
            "total_flaky_evaluated": grp["n_eval_flaky"].sum(),
            "flaky_weighted_f1": round(
                (grp["f1_score"] * grp["n_eval_flaky"]).sum() / max(1, grp["n_eval_flaky"].sum()), 4
            )
        }
        summary_list.append(rec)

    df_summary = pd.DataFrame(summary_list)
    sum_csv = sum_dir / "comprehensive_pipeline_benchmark_summary.csv"
    df_summary.to_csv(sum_csv, index=False)
    print(f"Rangkuman komprehensif disimpan ke: {sum_csv}")

    # 2. Faktor Imbalance Handling pada TrAdaBoost (Target 50%)
    imbalance_trts = [
        "TrAdaBoost_50_B1_None",
        "TrAdaBoost_50_B2_SMOTE",
        "TrAdaBoost_50", # B3 RUS
        "TrAdaBoost_50_B4_ClassWeight"
    ]
    df_imb = df_summary[df_summary["treatment"].isin(imbalance_trts)].copy()
    imb_csv = sum_dir / "factor_analysis_imbalance.csv"
    df_imb.to_csv(imb_csv, index=False)

    # 3. Faktor Feature Scaling pada TrAdaBoost (Target 50%)
    scaling_trts = [
        "TrAdaBoost_50_S1_Standard",
        "TrAdaBoost_50", # S2 MinMax
        "TrAdaBoost_50_S3_Robust"
    ]
    df_scale = df_summary[df_summary["treatment"].isin(scaling_trts)].copy()
    scale_csv = sum_dir / "factor_analysis_scaling.csv"
    df_scale.to_csv(scale_csv, index=False)

    # 4. Faktor Base Estimator & Tree Depth pada TrAdaBoost (Target 50%)
    model_trts = [
        "TrAdaBoost_50_DT_depth2",
        "TrAdaBoost_50", # DT_depth4
        "TrAdaBoost_50_DT_depth6",
        "TrAdaBoost_50_RF_base"
    ]
    df_model = df_summary[df_summary["treatment"].isin(model_trts)].copy()
    model_csv = sum_dir / "factor_analysis_base_estimators.csv"
    df_model.to_csv(model_csv, index=False)

    # 5. Faktor Iterasi Boosting & Skema Voting pada TrAdaBoost
    tb_cfg_trts = [
        "TrAdaBoost_50_iter10",
        "TrAdaBoost_50", # iter20
        "TrAdaBoost_50_iter30",
        "TrAdaBoost_50_all_estimators"
    ]
    df_tb_cfg = df_summary[df_summary["treatment"].isin(tb_cfg_trts)].copy()
    tb_cfg_csv = sum_dir / "factor_analysis_tradaboost_configs.csv"
    df_tb_cfg.to_csv(tb_cfg_csv, index=False)

    # 6. Faktor Data Cleaning (FlakeFlagger: Row-Drop vs Imputed)
    cleaning_trts = [
        "Baseline_Zero_Shot_RF",
        "RowDrop_Baseline_RF",
        "TrAdaBoost_50",
        "RowDrop_TrAdaBoost_50",
        "Within_Project",
        "RowDrop_Within_Project"
    ]
    df_clean = df_summary[(df_summary["dataset"] == "flakeflagger") & (df_summary["treatment"].isin(cleaning_trts))].copy()
    clean_csv = sum_dir / "factor_analysis_cleaning.csv"
    df_clean.to_csv(clean_csv, index=False)

    # 7. Print Hasil Analisis Utama
    print("\n" + "=" * 80)
    print("HASIL ANALISIS FAKTORIAL KOMPREHENSIF:")
    print("=" * 80)

    print("\n--- A. FAKTOR IMBALANCE HANDLING PADA TRADABOOST (Target 50%) ---")
    print(df_imb[["dataset", "treatment", "macro_f1", "flaky_weighted_f1", "macro_precision", "macro_recall"]].to_string(index=False))

    print("\n--- B. FAKTOR FEATURE SCALING PADA TRADABOOST (Target 50%) ---")
    print(df_scale[["dataset", "treatment", "macro_f1", "flaky_weighted_f1", "macro_precision", "macro_recall"]].to_string(index=False))

    print("\n--- C. FAKTOR BASE ESTIMATOR & DEPTH PADA TRADABOOST (Target 50%) ---")
    print(df_model[["dataset", "treatment", "macro_f1", "flaky_weighted_f1", "macro_precision", "macro_recall"]].to_string(index=False))

    print("\n--- D. FAKTOR ITERASI & VOTING PADA TRADABOOST (Target 50%) ---")
    print(df_tb_cfg[["dataset", "treatment", "macro_f1", "flaky_weighted_f1", "macro_precision", "macro_recall"]].to_string(index=False))

    print("\n--- E. FAKTOR DATA CLEANING (FlakeFlagger: Imputed vs Row-Drop) ---")
    print(df_clean[["treatment", "macro_f1", "flaky_weighted_f1", "macro_precision", "macro_recall"]].to_string(index=False))

if __name__ == "__main__":
    analyze_all_pipelines()
