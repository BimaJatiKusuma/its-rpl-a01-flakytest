"""
Modul Ingestion Data untuk Replikasi Eksperimen 3 menggunakan Dataset IDoFT.
Memuat dataset: data/cleaned_idoft_test_features.csv
Dataset ini mencakup:
- 12.626 baris pengujian perangkat lunak
- 17 proyek open-source Java dari IDoFT
- 264 flaky test riil terlabeli (IDoFT ground truth)
- 0 missing values (group-aware median imputed)
- 23 fitur prediktif FlakeFlagger (test smells, coverage/execution, churn/h-index)
"""

import sys
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 23 Fitur Prediktif Resmi FlakeFlagger
TEST_SMELLS_FEATURES: List[str] = [
    "assertion-roulette",
    "conditional-test-logic",
    "eager-test",
    "fire-and-forget",
    "indirect-testing",
    "mystery-guest",
    "resource-optimism",
    "test-run-war",
]

EXECUTION_COVERAGE_FEATURES: List[str] = [
    "testLength",
    "numAsserts",
    "numCoveredLines",
    "ExecutionTime",
    "projectSourceLinesCovered",
    "projectSourceClassesCovered",
    "num_third_party_libs",
]

CHURN_HINDEX_FEATURES: List[str] = [
    "hIndexModificationsPerCoveredLine_window5",
    "hIndexModificationsPerCoveredLine_window10",
    "hIndexModificationsPerCoveredLine_window25",
    "hIndexModificationsPerCoveredLine_window50",
    "hIndexModificationsPerCoveredLine_window75",
    "hIndexModificationsPerCoveredLine_window100",
    "hIndexModificationsPerCoveredLine_window500",
    "hIndexModificationsPerCoveredLine_window10000",
]

ALL_PREDICTIVE_FEATURES: List[str] = (
    TEST_SMELLS_FEATURES + EXECUTION_COVERAGE_FEATURES + CHURN_HINDEX_FEATURES
)

METADATA_COLUMNS: List[str] = [
    "Unnamed: 0",
    "test_name",
    "project",
    "testClassName",
    "testMethodName",
    "project_clean",
    "test_name_clean",
    "flaky_original_flakeflagger"
]

def resolve_idoft_data_path(custom_path: str = None) -> Path:
    """Mencari lokasi cleaned_idoft_test_features.csv secara otomatis."""
    if custom_path and Path(custom_path).exists():
        return Path(custom_path)

    candidates = [
        Path("data/cleaned_idoft_test_features.csv"),
        Path("../data/cleaned_idoft_test_features.csv"),
        Path(__file__).resolve().parent.parent / "data" / "cleaned_idoft_test_features.csv",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(
        "File cleaned_idoft_test_features.csv tidak ditemukan di jalur: "
        f"{[str(p) for p in candidates]}"
    )

def load_idoft_dataset(data_path: str = None) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Memuat dataset cleaned_idoft_test_features.csv untuk Replikasi Eksperimen 3.
    
    Returns:
        df: DataFrame lengkap (12.626 baris, 17 proyek)
        feature_cols: Daftar nama kolom fitur prediktif (23 fitur)
        project_list: Daftar 17 proyek unik berurutan
    """
    filepath = resolve_idoft_data_path(data_path)
    print(f"[DATA LOADER IDOFT] Membaca dataset: {filepath}")
    df = pd.read_csv(filepath)

    # Validasi integritas
    null_count = df[ALL_PREDICTIVE_FEATURES].isnull().sum().sum()
    if null_count > 0:
        raise ValueError(f"Ditemukan {null_count} nilai null di kolom prediktif dataset IDoFT!")

    project_list = sorted(df["project_clean"].unique().tolist())
    print(f"[DATA LOADER IDOFT] Total baris: {len(df):,} | Proyek: {len(project_list)} | Missing Values: 0")
    print(f"[DATA LOADER IDOFT] Distribusi Flaky: {df['flaky'].sum():,} ({df['flaky'].mean()*100:.2f}%)")

    return df, ALL_PREDICTIVE_FEATURES, project_list

if __name__ == "__main__":
    df, feats, projs = load_idoft_dataset()
    print("Daftar Proyek:", projs)
