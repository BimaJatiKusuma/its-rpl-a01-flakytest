"""
Main Orchestrator & Executor untuk Eksperimen 3 One-on-One Afeltra IDoFT Dataset.
Menjalankan seluruh siklus eksperimen Skema 1 (Pairwise One-on-One) dan Skema 2 (Repeated LOPO-CV Afeltra Concepts),
menghitung analisis statistik, dekonstruksi agregasi, dan menghasilkan seluruh visualisasi publikasi.

Rujukan:
- Afeltra et al. (IEEE Access 2024): A Large-Scale Empirical Investigation Into Cross-Project Flaky Test Prediction.
- Alshammari et al. (ICSE 2021): FlakeFlagger.
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_idoft_dataset, get_active_and_zero_flaky_projects
from pairwise_one_on_one_runner import run_pairwise_one_on_one_experiment
from afeltra_lopo_runner import run_afeltra_repeated_lopo_experiment
from consistency_analyzer import analyze_stochastic_consistency
from dataset_profiler import build_treatment_dataset_matrix, deconstruct_aggregation_schemes, analyze_flaky_rate_correlation
from statistical_tests import run_wilcoxon_and_cliffs_tests, run_friedman_test
from visualizer import (
    plot_pairwise_one_on_one_heatmap,
    plot_afeltra_treatments_heatmap,
    plot_afeltra_metrics_boxplots,
    plot_flaky_rate_vs_f1_scatter,
    plot_top_transfer_donors
)

def run_full_experiment_pipeline():
    start_time = time.time()
    current_dir = Path(__file__).resolve().parent
    results_dir = current_dir.parent / "results"
    plots_dir = results_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("🌟 ORKESTRASI PENUH EKSPERIMEN 3 ONE-ON-ONE AFELTRA IDOFT DATASET")
    print("   Author    : Bima Jati Kusuma (Magister Teknik Informatika ITS)")
    print("   Context   : Replikasi Afeltra et al. (IEEE Access 2024) pada Dataset IDoFT")
    print("   OutputDir :", results_dir)
    print("=" * 85)

    # 1. Load Dataset
    print("\n[FASE 1] Memuat Dataset IDoFT...")
    df, feature_cols, target_col = load_idoft_dataset()
    active_projs, zero_projs = get_active_and_zero_flaky_projects(df)
    print(f" -> Total Data     : {len(df):,} baris")
    print(f" -> Total Proyek   : {df['project_clean'].nunique()} (13 Aktif, 4 Zero-Flaky)")
    print(f" -> Total Fitur    : {len(feature_cols)} fitur prediktif FlakeFlagger")
    print(f" -> Flaky Tests    : {df[target_col].sum():,} ({df[target_col].mean()*100:.2f}%)")

    # 2. Eksekusi Skema 1: Pairwise One-on-One Transfer Matrix (156 pairs)
    pairwise_csv = results_dir / "all_pairwise_one_on_one_predictions.csv"
    if pairwise_csv.exists():
        print("\n[FASE 2] Memuat hasil Skema 1 yang sudah tersimpan...")
        df_pairwise = pd.read_csv(pairwise_csv)
    else:
        print("\n[FASE 2] Mengeksekusi Skema 1: One-on-One Pairwise Cross-Project Transfer...")
        df_pairwise = run_pairwise_one_on_one_experiment(
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            active_projects=active_projs,
            model_type="M1",
            random_state=42
        )
        df_pairwise.to_csv(pairwise_csv, index=False)
        print(f" -> Disimpan ke: {pairwise_csv}")

    # Matriks Pivot F1 dan PR-AUC untuk Pairwise 1-on-1
    pivot_pw_f1 = df_pairwise[df_pairwise["Treatment"] == "Baseline_1on1"].pivot_table(
        index="Source_Project", columns="Target_Project", values="F1", aggfunc="mean"
    ).round(4)
    matrix_pw_f1_csv = results_dir / "matrix_pairwise_one_on_one_f1.csv"
    pivot_pw_f1.to_csv(matrix_pw_f1_csv)

    pivot_pw_prauc = df_pairwise[df_pairwise["Treatment"] == "Baseline_1on1"].pivot_table(
        index="Source_Project", columns="Target_Project", values="PR_AUC", aggfunc="mean"
    ).round(4)
    matrix_pw_prauc_csv = results_dir / "matrix_pairwise_one_on_one_prauc.csv"
    pivot_pw_prauc.to_csv(matrix_pw_prauc_csv)

    # 3. Eksekusi Skema 2: Repeated LOPO-CV Afeltra Concepts (5 Seeds x 2 Models x 17 Projects)
    lopo_csv = results_dir / "all_repeated_folds_predictions.csv"
    if lopo_csv.exists() and len(pd.read_csv(lopo_csv)) >= 1020:
        print("\n[FASE 3] Memuat hasil Skema 2 LOPO-CV yang sudah tersimpan...")
        df_lopo = pd.read_csv(lopo_csv)
    else:
        print("\n[FASE 3] Mengeksekusi Skema 2: Repeated LOPO-CV Afeltra Concept Treatments...")
        seeds = [42, 100, 2024, 777, 999]
        models = ["M1", "M2"]  # M1: Random Forest, M2: XGBoost CUDA GPU
        df_lopo = run_afeltra_repeated_lopo_experiment(
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            seeds=seeds,
            models=models
        )
        df_lopo.to_csv(lopo_csv, index=False)
        print(f" -> Disimpan ke: {lopo_csv}")

    # 4. Analisis Konsistensi Stokastik (RQ1)
    print("\n[FASE 4] Menghitung Analisis Konsistensi Stokastik & Stabilitas...")
    df_consistency = analyze_stochastic_consistency(df_lopo)
    consistency_csv = results_dir / "consistency_stability_analysis.csv"
    df_consistency.to_csv(consistency_csv, index=False)
    print(f" -> Disimpan ke: {consistency_csv}")

    # 5. Membangun Matriks Master Treatment vs Dataset & Dekonstruksi Skema Agregasi (RQ2)
    print("\n[FASE 5] Membangun Matriks Kemampuan Treatment vs Dataset IDoFT...")
    matrix_dataset = build_treatment_dataset_matrix(df_lopo)
    matrix_dataset_csv = results_dir / "matrix_project_vs_afeltra_treatments_f1.csv"
    matrix_dataset.to_csv(matrix_dataset_csv)
    print(f" -> Disimpan ke: {matrix_dataset_csv}")

    print("\n[FASE 6] Mendekonstruksi Skema Agregasi Matematis F1...")
    df_agg = deconstruct_aggregation_schemes(df_lopo, top_treatment="T5_TrAdaBoost_75", top_model="M1")
    agg_csv = results_dir / "macro_vs_weighted_f1_explanation.csv"
    df_agg.to_csv(agg_csv, index=False)
    print(f" -> Disimpan ke: {agg_csv}")

    # 6. Analisis Korelasi Rasio Flaky vs F1 (RQ3)
    print("\n[FASE 7] Menghitung Uji Korelasi Rasio Flaky vs Performa Deteksi...")
    corr_res = analyze_flaky_rate_correlation(df_lopo, df)
    print(f" -> Pearson Correlation : r = {corr_res['pearson_r']} (p = {corr_res['pearson_p']:.4e})")
    print(f" -> Spearman Correlation: rho = {corr_res['spearman_rho']} (p = {corr_res['spearman_p']:.4e})")
    corr_csv = results_dir / "dataset_capability_profile.csv"
    corr_res["data_table"].to_csv(corr_csv, index=False)

    # 7. Uji Signifikansi Statistik (Wilcoxon, Cliff's Delta, Friedman)
    print("\n[FASE 8] Menjalankan Uji Signifikansi Statistik...")
    df_wilcoxon = run_wilcoxon_and_cliffs_tests(df_lopo, base_treatment="T1_Baseline", model="M1")
    wilcoxon_csv = results_dir / "statistical_wilcoxon_cliffs_delta.csv"
    df_wilcoxon.to_csv(wilcoxon_csv, index=False)
    print(f" -> Disimpan ke: {wilcoxon_csv}")

    friedman_res = run_friedman_test(df_lopo, model="M1")
    friedman_df = pd.DataFrame([
        {"Metric": "Friedman Chi-Square Stat", "Value": friedman_res["friedman_stat"]},
        {"Metric": "p-value", "Value": friedman_res["friedman_p"]}
    ])
    ranks_df = pd.DataFrame(list(friedman_res["average_ranks"].items()), columns=["Treatment", "Average_Rank"])
    friedman_combined = pd.concat([friedman_df, ranks_df], axis=1)
    friedman_csv = results_dir / "friedman_nemenyi_results.csv"
    friedman_combined.to_csv(friedman_csv, index=False)
    print(f" -> Disimpan ke: {friedman_csv}")

    # 8. Menghasilkan Grafik & Plot Publikasi
    print("\n[FASE 9] Menghasilkan Visualisasi Publikasi...")
    plot_pairwise_one_on_one_heatmap(df_pairwise, plots_dir / "heatmap_pairwise_one_on_one_f1.png", treatment="Baseline_1on1")
    print(" -> [Plot 1] Heatmap Pairwise One-on-One Baseline tuntas.")

    plot_pairwise_one_on_one_heatmap(df_pairwise, plots_dir / "heatmap_pairwise_one_on_one_burak_f1.png", treatment="Burak_Filter_1on1")
    print(" -> [Plot 1B] Heatmap Pairwise One-on-One Burak Filter tuntas.")

    plot_afeltra_treatments_heatmap(df_lopo, plots_dir / "heatmap_afeltra_treatments_f1.png", model="M1")
    print(" -> [Plot 2] Heatmap Afeltra Treatments vs Project tuntas.")

    plot_afeltra_metrics_boxplots(df_lopo, plots_dir / "boxplot_afeltra_metrics.png", model="M1")
    print(" -> [Plot 3] Boxplots Metrik Afeltra tuntas.")

    plot_flaky_rate_vs_f1_scatter(corr_res["data_table"], plots_dir / "scatter_flaky_rate_vs_f1.png")
    print(" -> [Plot 4] Scatter Plot Korelasi Flaky Rate vs F1 tuntas.")

    plot_top_transfer_donors(df_pairwise, plots_dir / "top_transfer_donors.png")
    print(" -> [Plot 5] Barplot Top Transfer Donors tuntas.")

    # Ringkasan Eksekutif Hasil
    total_elapsed = time.time() - start_time
    print("\n" + "=" * 85)
    print("🎉 SELURUH EKSPERIMEN BERHASIL DISELESAIKAN DENGAN SEMPURNA!")
    print(f"   Total Waktu Komputasi : {total_elapsed:.2f} detik ({total_elapsed/60:.2f} menit)")
    print(f"   Seluruh Artefak CSV   : {results_dir}")
    print(f"   Seluruh Grafik Gambar : {plots_dir}")
    print("=" * 85)

if __name__ == "__main__":
    run_full_experiment_pipeline()
