"""
Master Runner Eksekusi Eksperimen Alternatif 4:
Interpretasi Fitur Universal Berbasis SHAP (SHapley Additive exPlanations).

Menjalankan pipeline:
1. Pengecekan Akselerasi GPU RTX 4050 & CUDA 13.4
2. Ekstraksi SHAP TreeExplainer (XGBoost GPU & Random Forest, Cross vs Within)
3. Komparasi Peringkat Fitur & Uji Korelasi Spearman
4. Pembuatan Plot Publikasi Ilmiah (300 DPI)
"""

import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.check_gpu import check_nvidia_smi, check_python_environment, check_xgboost_and_shap_gpu
from src.shap_explainer import run_full_shap_pipeline
from src.rank_comparator import compare_feature_rankings
from src.visualizer import generate_all_plots


def main():
    print("=" * 70)
    print("[START] EKSEKUSI PENELITIAN ALTERNATIF 4:")
    print("        INTERPRETASI FITUR UNIVERSAL BERBASIS SHAP (CROSS VS WITHIN)")
    print("        Kelompok A01 - Magister Teknik Informatika ITS")
    print("=" * 70)

    total_start = time.time()

    # 1. Pengecekan GPU
    print("\n>>> TAHAP 1: Verifikasi GPU RTX 4050 & Environment...")
    check_python_environment()
    check_xgboost_and_shap_gpu()

    # 2. Pipeline Ekstraksi SHAP
    print("\n>>> TAHAP 2: Ekstraksi Nilai Shapley (XGBoost GPU & Random Forest)...")
    base_dir = Path(__file__).resolve().parent.parent
    results_dir = base_dir / "results"
    plots_dir = results_dir / "plots"

    pipeline_res = run_full_shap_pipeline(output_dir=str(results_dir))

    # 3. Analisis Komparasi Peringkat Fitur & Statistik
    print("\n>>> TAHAP 3: Analisis Komparasi Peringkat Fitur & Uji Spearman...")
    within_csv = results_dir / "shap_summary_within_project.csv"
    cross_csv = results_dir / "shap_summary_cross_project.csv"
    comp_csv = results_dir / "feature_rank_comparison.csv"
    stats_json = results_dir / "rank_comparison_stats.json"

    merged_df, stats_res = compare_feature_rankings(
        within_summary_path=within_csv,
        cross_summary_path=cross_csv,
        output_csv_path=comp_csv,
        output_json_path=stats_json
    )

    print("\n--- HASIL ANALISIS STATISTIK PERINGKAT FITUR ---")
    print(f"Koefisien Korelasi Spearman (rho) : {stats_res['spearman_rho']:.4f}")
    print(f"Spearman p-value                   : {stats_res['spearman_pval']:.4e}")
    print(f"Koefisien Korelasi Pearson (r)    : {stats_res['pearson_r']:.4f}")
    print(f"Rata-rata Rank Shift Test Smells  : +{stats_res['test_smells_shift']['mean_rank_shift']:.2f} peringkat")
    print(f"Smells Naik Peringkat             : {stats_res['test_smells_shift']['smells_promoted']} dari 8 fitur")

    # 4. Pembuatan Plot Visualisasi
    print("\n>>> TAHAP 4: Pembuatan Seluruh Visualisasi Riset (300 DPI)...")
    generate_all_plots(results_dir, plots_dir)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"[SELESAI] SELURUH PIPELINE ALTERNATIF 4 SELESAI DALAM {total_elapsed:.2f} DETIK ({total_elapsed/60:.2f} MENIT)!")
    print(f"[OUTPUT] Direktori hasil: {results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
