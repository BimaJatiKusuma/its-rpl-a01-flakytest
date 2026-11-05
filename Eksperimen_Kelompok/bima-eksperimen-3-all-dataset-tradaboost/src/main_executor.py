"""
Eksekutor Utama Pipeline Eksperimen TrAdaBoost Multi-Dataset (FlakeFlagger & IDoFT).
Mengorkestrasi seluruh tahapan secara otomatis dari eksekusi model hingga visualisasi dan pelaporan.
"""

import os
import sys
import time
from pathlib import Path

# Konfigurasi UTF-8 console output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dual_dataset_tradaboost_runner import run_all_datasets, DEFAULT_SEEDS
from consistency_analyzer import compute_consistency_analysis
from dataset_profiler import compute_project_capability_profile, compute_four_aggregation_schemes, analyze_flakiness_rate_correlation
from statistical_tests import run_statistical_tests
from visualizer import generate_all_plots


def main():
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "results" / "raw"
    sum_dir = base_dir / "results" / "summary"
    plot_dir = base_dir / "results" / "plots"

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(sum_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    t_global_start = time.time()
    print("=" * 78)
    print("  MEMULAI PIPELINE EKSEKUSI TRADABOOST MULTI-DATASET (FLAKEFLAGGER & IDOFT)")
    print("=" * 78)

    # ---------------------------------------------------------
    # TAHAP 1: EKSEKUSI REPEATED LOPO-CV PADA DUA DATASET
    # ---------------------------------------------------------
    print("\n[TAHAP 1/6] Menjalankan Evaluasi Repeated LOPO-CV (5 Seeds x Proyek x 6 Treatments)...")
    df_ff, df_id, df_combined = run_all_datasets(seeds=DEFAULT_SEEDS, base_dir=base_dir)

    # ---------------------------------------------------------
    # TAHAP 2: ANALISIS KONSISTENSI STOKASTIK (CV%, CI95)
    # ---------------------------------------------------------
    print("\n[TAHAP 2/6] Menghitung Analisis Konsistensi Stokastik & Variabilitas Seeds...")
    consistency_csv = str(sum_dir / "consistency_stability_analysis.csv")
    df_consistency = compute_consistency_analysis(df_combined, output_csv=consistency_csv)

    # ---------------------------------------------------------
    # TAHAP 3: PROFILING KAPABILITAS PROYEK & KORELASI FLAKINESS
    # ---------------------------------------------------------
    print("\n[TAHAP 3/6] Memetakan Kapabilitas Tiap Proyek & Korelasi Flakiness Rate...")
    profile_csv = str(sum_dir / "project_capability_profile.csv")
    df_prof = compute_project_capability_profile(df_combined, output_csv=profile_csv)

    flaky_corr = analyze_flakiness_rate_correlation(df_prof)
    print(" -> Korelasi Flakiness Rate vs Performa:")
    for d, c in flaky_corr.items():
        print(f"    * {d.upper()}: Baseline r={c['baseline_r']} (p={c['baseline_p']:.4e}) | TrAdaBoost75 r={c['tradaboost75_r']} (p={c['tradaboost75_p']:.4e})")

    # ---------------------------------------------------------
    # TAHAP 4: DEKONSTRUKSI 4 SKEMA AGREGASI MATEMATIS
    # ---------------------------------------------------------
    print("\n[TAHAP 4/6] Menghitung Dekonstruksi 4 Skema Agregasi Matematis...")
    aggr_csv = str(sum_dir / "macro_vs_weighted_f1_summary.csv")
    df_aggr = compute_four_aggregation_schemes(df_combined, output_csv=aggr_csv)

    # ---------------------------------------------------------
    # TAHAP 5: PENGUJIAN SIGNIFIKANSI STATISTIK (WILCOXON & CLIFF'S D)
    # ---------------------------------------------------------
    print("\n[TAHAP 5/6] Menjalankan Uji Signifikansi Statistik (Wilcoxon, Cliff's d, Friedman)...")
    stat_csv = str(sum_dir / "statistical_tests_results.csv")
    df_pw, df_fr = run_statistical_tests(df_combined, metric="f1_score", output_csv=stat_csv)

    # ---------------------------------------------------------
    # TAHAP 6: GENERASI VISUALISASI PUBLIKASI (300 DPI)
    # ---------------------------------------------------------
    print("\n[TAHAP 6/6] Menghasilkan 6 Grafik Visualisasi Publikasi Ilmiah 300 DPI...")
    generate_all_plots(df_combined, df_prof, df_aggr, str(plot_dir))

    total_time = time.time() - t_global_start
    print("\n" + "=" * 78)
    print(f"  SELURUH PIPELINE SELESAI DENGAN SUKSES DALAM {total_time:.2f} DETIK!")
    print("=" * 78)

    print("\nRINGKASAN TEMUAN KUNCI TRADABOOST LINTAS DUA DATASET:")
    summary_pivot = df_aggr[df_aggr["treatment"].isin(["Baseline_Zero_Shot_RF", "TrAdaBoost_25", "TrAdaBoost_50", "TrAdaBoost_75", "Within_Project"])][
        ["dataset", "treatment", "unweighted_macro_f1", "active_macro_f1", "flaky_weighted_f1", "global_micro_f1"]
    ]
    print(summary_pivot.to_string(index=False))


if __name__ == "__main__":
    main()
