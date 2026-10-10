"""
Modul Komparasi Peringkat Fitur Within-Project vs Cross-Project Berbasis SHAP.
Menghitung koefisien korelasi rank Spearman, rank shift, dan proporsi kontribusi taksonomi.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from scipy import stats
import json

from .data_loader import TAXONOMY_MAP


def compare_feature_rankings(
    within_summary_path: Path,
    cross_summary_path: Path,
    output_csv_path: Path = None,
    output_json_path: Path = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Membandingkan peringkat fitur antara Within-Project dan Cross-Project.

    Args:
        within_summary_path: Path ke shap_summary_within_project.csv
        cross_summary_path: Path ke shap_summary_cross_project.csv
        output_csv_path: Path tujuan output CSV komparasi
        output_json_path: Path tujuan output JSON metrik statistik

    Returns:
        comparison_df: DataFrame komparasi lengkap per fitur
        stats_results: Dictionary hasil analisis statistik dan kontribusi dimensi
    """
    df_within = pd.read_csv(within_summary_path)
    df_cross = pd.read_csv(cross_summary_path)

    # Rename kolom
    df_within = df_within.rename(columns={
        "mean_abs_shap": "within_mean_abs_shap",
        "rank": "within_rank",
        "relative_importance_pct": "within_importance_pct"
    })
    df_cross = df_cross.rename(columns={
        "mean_abs_shap": "cross_mean_abs_shap",
        "rank": "cross_rank",
        "relative_importance_pct": "cross_importance_pct"
    })

    # Gabungkan berdasarkan fitur
    merged = pd.merge(
        df_within[["feature", "dimension", "within_mean_abs_shap", "within_rank", "within_importance_pct"]],
        df_cross[["feature", "cross_mean_abs_shap", "cross_rank", "cross_importance_pct"]],
        on="feature"
    )

    # Rank shift: nilai positif artinya naik peringkat (semakin penting di Cross-Project)
    merged["rank_shift"] = merged["within_rank"] - merged["cross_rank"]

    # Urutkan berdasarkan kepentingan Cross-Project (rank 1 ke 23)
    merged = merged.sort_values(by="cross_rank").reset_index(drop=True)

    # 1. Korelasi Spearman & Pearson
    spearman_rho, spearman_pval = stats.spearmanr(merged["within_rank"], merged["cross_rank"])
    pearson_r, pearson_pval = stats.pearsonr(merged["within_mean_abs_shap"], merged["cross_mean_abs_shap"])

    # 2. Kontribusi per Dimensi Taksonomi
    dim_group = merged.groupby("dimension").agg({
        "within_mean_abs_shap": "sum",
        "cross_mean_abs_shap": "sum",
        "within_importance_pct": "sum",
        "cross_importance_pct": "sum"
    }).reset_index()

    dim_group["delta_importance_pct"] = dim_group["cross_importance_pct"] - dim_group["within_importance_pct"]

    # 3. Analisis Test Smells Shift
    smells_df = merged[merged["dimension"] == "Test Smells"]
    mean_smell_rank_shift = float(smells_df["rank_shift"].mean())
    smells_promoted = int((smells_df["rank_shift"] > 0).sum())
    smells_demoted = int((smells_df["rank_shift"] < 0).sum())
    smells_unchanged = int((smells_df["rank_shift"] == 0).sum())

    stats_results = {
        "spearman_rho": float(spearman_rho),
        "spearman_pval": float(spearman_pval),
        "pearson_r": float(pearson_r),
        "pearson_pval": float(pearson_pval),
        "dimension_contributions": dim_group.to_dict(orient="records"),
        "test_smells_shift": {
            "mean_rank_shift": mean_smell_rank_shift,
            "smells_promoted": smells_promoted,
            "smells_demoted": smells_demoted,
            "smells_unchanged": smells_unchanged
        },
        "top_universal_cross_features": merged.head(5)[["feature", "cross_rank", "cross_importance_pct", "rank_shift"]].to_dict(orient="records")
    }

    if output_csv_path:
        merged.to_csv(output_csv_path, index=False)
        print(f"[RankComparator] Tabel komparasi disimpan ke: {output_csv_path}")

    if output_json_path:
        with open(output_json_path, "w") as f:
            json.dump(stats_results, f, indent=2)
        print(f"[RankComparator] Statistik disimpan ke: {output_json_path}")

    return merged, stats_results


if __name__ == "__main__":
    results_dir = Path(__file__).resolve().parent.parent / "results"
    within_csv = results_dir / "shap_summary_within_project.csv"
    cross_csv = results_dir / "shap_summary_cross_project.csv"
    out_csv = results_dir / "feature_rank_comparison.csv"
    out_json = results_dir / "rank_comparison_stats.json"

    if within_csv.exists() and cross_csv.exists():
        merged, stats_res = compare_feature_rankings(within_csv, cross_csv, out_csv, out_json)
        print("\n=== TOP 5 FITUR CROSS-PROJECT ===")
        print(merged.head(5)[["feature", "cross_rank", "within_rank", "rank_shift", "cross_importance_pct"]])
        print(f"\nSpearman's Rho: {stats_res['spearman_rho']:.4f} (p-value: {stats_res['spearman_pval']:.4e})")
    else:
        print("[PERINGATAN] File CSV ringkasan SHAP belum ditemukan. Jalankan shap_explainer.py terlebih dahulu.")
