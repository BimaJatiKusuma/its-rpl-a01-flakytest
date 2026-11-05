"""
Modul Analisis Konsistensi & Stabilitas Empiris untuk Eksperimen Gabungan 1 & 2.
Mengevaluasi variansi performa model lintas repetisi (hingga 10 random seeds).
Menghitung:
- Mean, Standar Deviasi, Coefficient of Variation (CV = Std / Mean * 100)
- 95% Confidence Interval (CI)
- Kestabilan per faktor eksperimen (Cleaning C1/C2/C3, Scaling S1-S3, Imbalance B1-B4, Model M1-M2)
- Komparasi Langsung Tri-Paradigma Data Cleaning (C1 vs C2 vs C3)
"""

import sys
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def analyze_pipeline_consistency(
    all_runs_csv: Optional[str] = None,
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Menganalisis konsistensi dan stabilitas dari seluruh 72 pipeline lintas perulangan seed.
    """
    if all_runs_csv is None:
        all_runs_csv = str(Path(__file__).resolve().parent.parent / "results" / "all_seed_pipeline_runs.csv")
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent.parent / "results")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not Path(all_runs_csv).exists():
        raise FileNotFoundError(f"File log perulangan tidak ditemukan: {all_runs_csv}")

    df_runs = pd.read_csv(all_runs_csv)
    num_seeds_total = df_runs["Seed"].nunique()

    print("\n" + "=" * 85)
    print(f"📊 ANALISIS KONSISTENSI & STABILITAS EKSPERIMEN GABUNGAN LINTAS {num_seeds_total}x PERULANGAN SEED")
    print("=" * 85)

    metrics_to_analyze = [
        "Mean_F1", "Mean_Recall", "Mean_Precision", "Mean_PR_AUC",
        "Mean_ROC_AUC", "Mean_MCC", "Execution_Time_Sec"
    ]

    records = []
    for pid, group in df_runs.groupby("Pipeline_ID"):
        cleaning = group["Cleaning"].iloc[0]
        scaling = group["Scaling"].iloc[0]
        imbalance = group["Imbalance"].iloc[0]
        model = group["Model"].iloc[0]

        row = {
            "Pipeline_ID": pid,
            "Cleaning": cleaning,
            "Scaling": scaling,
            "Imbalance": imbalance,
            "Model": model,
            "Num_Seeds": len(group),
        }

        for m in metrics_to_analyze:
            if m in group.columns:
                vals = group[m].dropna().values
                if len(vals) > 0:
                    mean_v = float(np.mean(vals))
                    std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    min_v = float(np.min(vals))
                    max_v = float(np.max(vals))
                    cv_v = (std_v / mean_v * 100.0) if mean_v > 0 else 0.0
                    ci95 = 1.96 * (std_v / np.sqrt(len(vals))) if len(vals) > 1 else 0.0

                    row[f"{m}_Mean"] = mean_v
                    row[f"{m}_Std"] = std_v
                    row[f"{m}_CV_%"] = cv_v
                    row[f"{m}_Min"] = min_v
                    row[f"{m}_Max"] = max_v
                    row[f"{m}_Range"] = max_v - min_v
                    row[f"{m}_CI95"] = ci95

        records.append(row)

    df_cons = pd.DataFrame(records)
    if "Mean_F1_Mean" in df_cons.columns:
        df_cons.sort_values(by="Mean_F1_Mean", ascending=False, inplace=True)

    out_file = out_path / "consistency_stability_analysis.csv"
    df_cons.to_csv(out_file, index=False)
    print(f"[OK] Analisis konsistensi pipeline tersimpan di: {out_file}")

    # Analisis Kestabilan per Faktor (Cleaning, Scaling, Imbalance, Model)
    factors = ["Cleaning", "Scaling", "Imbalance", "Model"]
    for f in factors:
        if f in df_runs.columns:
            f_agg = df_runs.groupby(f).agg({
                "Mean_F1": ["mean", "std"],
                "Mean_Recall": ["mean", "std"],
                "Mean_Precision": ["mean", "std"],
                "Mean_PR_AUC": ["mean", "std"],
            })
            f_agg.columns = [f"{c[0]}_{c[1].upper()}" for c in f_agg.columns]
            f_agg["CV_F1_%"] = (f_agg["Mean_F1_STD"] / f_agg["Mean_F1_MEAN"]) * 100.0
            factor_file = out_path / f"consistency_factor_{f.lower()}.csv"
            f_agg.reset_index().to_csv(factor_file, index=False)
            print(f"  📁 Ringkasan faktor '{f}' tersimpan di: {factor_file.name}")

    # Komparasi Tri-Paradigma Data Cleaning (C1 vs C2 vs C3)
    print("\n" + "-" * 85)
    print("🔬 KOMPARASI TRI-PARADIGMA DATA CLEANING (C1 vs C2 vs C3):")
    print("-" * 85)
    if "Cleaning" in df_runs.columns:
        c_comp = df_runs.groupby("Cleaning").agg({
            "Mean_F1": ["mean", "std"],
            "Mean_Recall": ["mean", "std"],
            "Mean_Precision": ["mean", "std"],
            "Mean_PR_AUC": ["mean", "std"],
        })
        c_comp.columns = [f"{c[0]}_{c[1].upper()}" for c in c_comp.columns]
        c_comp["CV_F1_%"] = (c_comp["Mean_F1_STD"] / c_comp["Mean_F1_MEAN"]) * 100.0
        print(c_comp.reset_index().to_string(index=False))

    print("\n" + "-" * 85)
    print("🏆 TOP 10 PIPELINE PALING KONSISTEN & BERPERFORMA TINGGI:")
    print("-" * 85)
    top_cols = ["Pipeline_ID", "Cleaning", "Scaling", "Imbalance", "Model",
                "Mean_F1_Mean", "Mean_F1_Std", "Mean_F1_CV_%", "Mean_Recall_Mean", "Mean_PR_AUC_Mean"]
    existing_cols = [c for c in top_cols if c in df_cons.columns]
    print(df_cons[existing_cols].head(10).to_string(index=False))
    print("=" * 85)

    return df_cons


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analisis Konsistensi Eksperimen Gabungan 1 & 2")
    parser.add_argument("--runs_csv", type=str, default=None, help="Path ke all_seed_pipeline_runs.csv")
    parser.add_argument("--output_dir", type=str, default=None, help="Direktori output")
    args = parser.parse_args()

    analyze_pipeline_consistency(all_runs_csv=args.runs_csv, output_dir=args.output_dir)
