"""
Modul Pemuatan Dataset FlakeFlagger untuk Eksperimen Alternatif 4:
Interpretasi Fitur Universal Berbasis SHAP (SHapley Additive exPlanations).

Menerapkan pembersihan data strategi C1 (Row-Level Drop: 21.963 baris pada 24 proyek).
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np

# Daftar 23 Fitur Prediktif Resmi FlakeFlagger (Alshammari et al., ICSE 2021)
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

TAXONOMY_MAP: Dict[str, str] = {}
for f in TEST_SMELLS_FEATURES:
    TAXONOMY_MAP[f] = "Test Smells"
for f in EXECUTION_COVERAGE_FEATURES:
    TAXONOMY_MAP[f] = "Execution & Coverage"
for f in CHURN_HINDEX_FEATURES:
    TAXONOMY_MAP[f] = "Code Churn"


def resolve_data_path(custom_path: str = None) -> Path:
    """Mencari lokasi test_features.csv secara otomatis di berbagai root path."""
    if custom_path and Path(custom_path).exists():
        return Path(custom_path)

    candidates = [
        Path("test_features.csv"),
        Path("../../../test_features.csv"),
        Path("../../test_features.csv"),
        Path("../test_features.csv"),
        Path(__file__).resolve().parent.parent.parent.parent / "test_features.csv",
        Path(__file__).resolve().parent.parent.parent / "test_features.csv"
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("File test_features.csv tidak ditemukan di direktori yang diperiksa.")


def load_dataset(
    cleaning_strategy: str = "C1",
    data_path: str = None
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Memuat dataset FlakeFlagger dengan strategi pembersihan C1 (Row-Level Drop).

    Args:
        cleaning_strategy: 'C1' (Default dan wajib untuk Alternatif 4).
        data_path: Lokasi file kustom test_features.csv jika ada.

    Returns:
        df: DataFrame yang telah dibersihkan (21.963 baris).
        features: List 23 nama fitur prediktif FlakeFlagger asli.
        projects: List 24 proyek unik berurutan.
    """
    file_path = resolve_data_path(data_path)
    df = pd.read_csv(file_path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    features = [f for f in ALL_PREDICTIVE_FEATURES if f in df.columns]

    if cleaning_strategy.upper() == "C1":
        initial_len = len(df)
        df = df.dropna(subset=features).copy()
        print(f"[DataLoader C1] Row-Level Drop: dari {initial_len} menjadi {len(df)} baris valid ({df['project'].nunique()} proyek).")
    else:
        raise ValueError(f"Hanya strategi C1 yang didukung untuk Alternatif 4 (diberikan: {cleaning_strategy}).")

    projects = sorted(df["project"].unique().tolist())
    return df, features, projects


if __name__ == "__main__":
    df, features, projects = load_dataset()
    print(f"Total Baris Data : {len(df)}")
    print(f"Total Fitur Asli : {len(features)}")
    print(f"Daftar Proyek    : {len(projects)} proyek")
    print(f"Distribusi Flaky : {df['flaky'].sum()} flaky ({df['flaky'].mean()*100:.2f}%)")
