"""
Modul Ingestion Data untuk Eksperimen 2.
Memuat dataset bersih terimputasi: Experiment_Ali/data/cleaned_test_features.csv
Dataset ini telah melalui imputasi group-aware per-project median sehingga:
- 0 missing values
- Seluruh 24 proyek dipertahankan
- 22.236 baris data lengkap (tidak memerlukan pembuangan C1/C2)
"""

from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np

# Daftar 23 Fitur Prediktif Resmi FlakeFlagger
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
]


def resolve_cleaned_data_path(custom_path: str = None) -> Path:
    """Mencari lokasi cleaned_test_features.csv secara otomatis."""
    if custom_path and Path(custom_path).exists():
        return Path(custom_path)

    candidates = [
        Path("Experiment_Ali/data/cleaned_test_features.csv"),
        Path("../../Experiment_Ali/data/cleaned_test_features.csv"),
        Path("../Experiment_Ali/data/cleaned_test_features.csv"),
        Path(__file__).resolve().parent.parent.parent.parent / "Experiment_Ali" / "data" / "cleaned_test_features.csv",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(
        "File cleaned_test_features.csv tidak ditemukan di jalur yang diantisipasi: "
        f"{[str(p) for p in candidates]}"
    )


def load_cleaned_dataset(data_path: str = None) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Memuat dataset cleaned_test_features.csv untuk Eksperimen 2.
    
    Returns:
        df: DataFrame lengkap (22.236 baris, 24 proyek)
        feature_cols: Daftar nama kolom fitur prediktif (23 fitur)
        project_list: Daftar 24 proyek unik berurutan
    """
    filepath = resolve_cleaned_data_path(data_path)
    print(f"[DATA LOADER] Membaca dataset terimputasi: {filepath}")
    df = pd.read_csv(filepath)

    # Validasi integritas
    null_count = df[ALL_PREDICTIVE_FEATURES].isnull().sum().sum()
    if null_count > 0:
        raise ValueError(f"Ditemukan {null_count} nilai null di kolom prediktif pada dataset terimputasi!")

    project_list = sorted(df["project"].unique().tolist())
    print(f"[DATA LOADER] Total baris: {len(df):,} | Proyek: {len(project_list)} | Missing Values: {null_count}")
    print(f"[DATA LOADER] Distribusi Flaky: {df['flaky'].sum()} ({df['flaky'].mean()*100:.2f}%)")

    return df, ALL_PREDICTIVE_FEATURES, project_list


if __name__ == "__main__":
    df, features, projects = load_cleaned_dataset()
    print(f"Fitur prediktif ({len(features)}): {features}")
    print(f"Proyek ({len(projects)}): {projects}")
