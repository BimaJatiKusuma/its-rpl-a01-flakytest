"""
Skrip Ingestion & Pembentukan Dataset IDoFT (Illinois Dataset of Flaky Tests)
untuk Replikasi Eksperimen 3 pada Skenario Cross-Project (Afeltra et al., IEEE Access 2024).

Dataset IDoFT bersumber dari:
- pr-data.csv (Pull Request data)
- gr-data.csv (Git Re-run data)
Fitur diambil dari FlakeFlagger yang telah diimputasi bersih (cleaned_test_features.csv).
Proses matching dilakukan dengan mencocokkan lowercase(project_name) dan lowercase(packageName.ClassName.methodName).
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def generate_idoft_dataset():
    current_dir = Path(__file__).resolve().parent
    base_dir = current_dir.parent.parent.parent  # Workspace root

    # 1. Path ke sumber data
    pr_path = base_dir / "Eksperimen_Kelompok" / "Eksperimen_Cross_Project_Afeltra_paper_concept" / "using_Idoft_set" / "idoft-main" / "idoft-main" / "pr-data.csv"
    gr_path = base_dir / "Eksperimen_Kelompok" / "Eksperimen_Cross_Project_Afeltra_paper_concept" / "using_Idoft_set" / "idoft-main" / "idoft-main" / "gr-data.csv"
    feat_path = base_dir / "Experiment_Ali" / "data" / "cleaned_test_features.csv"

    print("=" * 80)
    print("🚀 MEMULAI PEMBENTUKAN DATASET IDOFT (AFELTRA ET AL., 2024 CONTEXT)")
    print("=" * 80)
    print(f"Loading PR Data: {pr_path}")
    print(f"Loading GR Data: {gr_path}")
    print(f"Loading Features: {feat_path}")

    pr_data = pd.read_csv(pr_path)
    gr_data = pd.read_csv(gr_path)
    idoft_java = pd.concat([pr_data, gr_data], ignore_index=True)
    print(f"[OK] Total flaky test entries di IDoFT (Java): {len(idoft_java):,}")

    # 2. Standarisasi nama project dan fully-qualified test name
    idoft_java["project_clean"] = idoft_java["Project URL"].apply(
        lambda x: x.split("/")[-1].lower() if isinstance(x, str) else ""
    )
    idoft_java["test_name_clean"] = idoft_java[
        "Fully-Qualified Test Name (packageName.ClassName.methodName)"
    ].apply(lambda x: str(x).lower().strip())

    idoft_flaky_set = set(zip(idoft_java["project_clean"], idoft_java["test_name_clean"]))
    print(f"[OK] Total unique project-test pairs flaky di IDoFT: {len(idoft_flaky_set):,}")

    # 3. Load feature matrix (terimputasi bersih, 23 fitur FlakeFlagger, 0 nulls)
    df_clean = pd.read_csv(feat_path)
    df_clean["project_clean"] = df_clean["project"].str.lower()
    df_clean["test_name_clean"] = df_clean["test_name"].apply(lambda x: str(x).lower().strip())

    # 4. Terapkan label flakiness berdasarkan IDoFT
    df_clean["flaky_original_flakeflagger"] = df_clean["flaky"]
    df_clean["flaky"] = df_clean.apply(
        lambda r: 1 if (r["project_clean"], r["test_name_clean"]) in idoft_flaky_set else 0,
        axis=1
    )

    # 5. Filter subset hanya untuk proyek yang tercakup dalam IDoFT Java
    idoft_projects = sorted(idoft_java["project_clean"].unique().tolist())
    df_idoft = df_clean[df_clean["project_clean"].isin(idoft_projects)].copy().reset_index(drop=True)

    # Validasi 0 missing values
    null_count = df_idoft.isnull().sum().sum()
    if null_count > 0:
        raise ValueError(f"Ditemukan {null_count} nilai null di dataset hasil!")

    # 6. Simpan CSV
    out_csv = current_dir / "cleaned_idoft_test_features.csv"
    df_idoft.to_csv(out_csv, index=False)
    print(f"\n[OK] Dataset IDoFT berhasil disimpan ke: {out_csv}")
    print(f"     Total baris data    : {len(df_idoft):,}")
    print(f"     Jumlah proyek Java  : {df_idoft['project_clean'].nunique()}")
    print(f"     Total flaky test    : {df_idoft['flaky'].sum():,} ({df_idoft['flaky'].mean()*100:.2f}%)")
    print(f"     Total non-flaky test: {len(df_idoft) - df_idoft['flaky'].sum():,}")

    # 7. Ekstrak metadata proyek
    proj_breakdown = []
    for p in sorted(df_idoft["project_clean"].unique()):
        sub = df_idoft[df_idoft["project_clean"] == p]
        tot = len(sub)
        flk = int(sub["flaky"].sum())
        non_flk = tot - flk
        rate = (flk / tot * 100.0) if tot > 0 else 0.0
        proj_breakdown.append({
            "project": p,
            "total_tests": tot,
            "flaky_tests": flk,
            "non_flaky_tests": non_flk,
            "flaky_rate_%": round(rate, 2),
            "status": "Active Flaky" if flk > 0 else "Zero Flaky Target"
        })

    metadata = {
        "dataset_name": "IDoFT_Java_Flaky_Benchmark_Cleaned",
        "reference_paper": "Afeltra et al. (IEEE Access 2024) - A Large-Scale Empirical Investigation Into Cross-Project Flaky Test Prediction",
        "total_instances": len(df_idoft),
        "total_projects": df_idoft["project_clean"].nunique(),
        "total_flaky_count": int(df_idoft["flaky"].sum()),
        "total_non_flaky_count": int(len(df_idoft) - df_idoft["flaky"].sum()),
        "overall_flakiness_rate_%": round(float(df_idoft["flaky"].mean() * 100), 2),
        "predictive_features_count": 23,
        "imputation_strategy": "Group-Aware Per-Project Median Imputation (Inherited from Cleaned Features)",
        "project_demographics": proj_breakdown
    }

    out_meta = current_dir / "dataset_metadata.json"
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Metadata dataset tersimpan di: {out_meta}")

    print("\nRingkasan Demografi Proyek IDoFT:")
    print(pd.DataFrame(proj_breakdown).to_string(index=False))
    print("=" * 80)

if __name__ == "__main__":
    generate_idoft_dataset()
