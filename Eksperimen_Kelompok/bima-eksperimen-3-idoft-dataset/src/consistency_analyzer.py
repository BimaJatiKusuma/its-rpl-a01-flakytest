"""
Modul Analisis Konsistensi & Stabilitas Empiris untuk Replikasi Eksperimen 3 IDoFT.
Mengevaluasi variansi performa model lintas 5 random seeds (42, 100, 2024, 777, 999).
Menghitung:
- Mean, Standar Deviasi, Coefficient of Variation (CV = Std / Mean * 100)
- 95% Confidence Interval (CI)
- Kestabilan per faktor eksperimen (Scaling, Imbalance, Classifier)
"""

from pathlib import Path
from typing import Optional, Dict
import sys
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
    Menganalisis konsistensi dan stabilitas dari 24 pipeline di seluruh 5 seed perulangan pada dataset IDoFT.
    """
    if all_runs_csv is None:
        all_runs_csv = str(Path(__file__).resolve().parent.parent / "results" / "all_seed_pipeline_runs.csv")
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent.parent / "results")
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_runs = pd.read_csv(all_runs_csv)

    print("\n" + "=" * 80)
    print("📊 ANALISIS KONSISTENSI & STABILITAS PIPELINE LINTAS REPETISI (DATASET IDOFT 5 SEEDS)")
    print("=" * 80)

    # Agregasi per pipeline
    metrics_to_analyze = [
        "Mean_F1",
        "Mean_F1_Active_13",
        "Mean_Recall",
        "Mean_Precision",
        "Mean_PR_AUC",
        "Mean_MCC",
        "Execution_Time_Sec"
    ]
    
    records = []
    for pid, group in df_runs.groupby("Pipeline_ID"):
        scaling = group["Scaling"].iloc[0]
        imbalance = group["Imbalance"].iloc[0]
        model = group["Model"].iloc[0]
        
        row = {
            "Pipeline_ID": pid,
            "Scaling": scaling,
            "Imbalance": imbalance,
            "Model": model,
            "Num_Seeds": len(group),
        }
        
        for m in metrics_to_analyze:
            vals = group[m].dropna().values
            if len(vals) == 0:
                mean_v, std_v, min_v, max_v, cv_v, ci95 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            else:
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
    df_cons.sort_values(by="Mean_F1_Mean", ascending=False, inplace=True)
    
    out_file = out_path / "consistency_stability_analysis.csv"
    df_cons.to_csv(out_file, index=False)
    print(f"[OK] Analisis konsistensi pipeline tersimpan di: {out_file}")

    # Analisis Kestabilan per Faktor (Scaling, Imbalance, Model)
    factor_summaries = {}
    factors = ["Scaling", "Imbalance", "Model"]
    for f in factors:
        f_agg = df_runs.groupby(f).agg({
            "Mean_F1": ["mean", "std"],
            "Mean_F1_Active_13": ["mean", "std"],
            "Mean_Recall": ["mean", "std"],
            "Mean_Precision": ["mean", "std"],
            "Mean_PR_AUC": ["mean", "std"],
        })
        f_agg.columns = [f"{c[0]}_{c[1].upper()}" for c in f_agg.columns]
        f_agg[f"CV_F1_%"] = (f_agg["Mean_F1_STD"] / f_agg["Mean_F1_MEAN"]) * 100.0
        factor_summaries[f] = f_agg.reset_index()
        f_agg.to_csv(out_path / f"consistency_factor_{f.lower()}.csv")
        print(f"[OK] Analisis faktor {f} tersimpan.")

    print("=" * 80)
    return df_cons

if __name__ == "__main__":
    analyze_pipeline_consistency()
