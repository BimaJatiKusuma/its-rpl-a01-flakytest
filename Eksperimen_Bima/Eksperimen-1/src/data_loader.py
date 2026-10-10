"""
Modul Ingestion & Preprocessing Data FlakeFlagger untuk Eksperimen 1.
Mendukung dua strategi pembersihan data:
- C1: Row-Level Drop (mempertahankan 24 proyek, membuang 273 baris NaN)
- C2: Project-Level Drop (membuang 5 proyek dengan missing values, mempertahankan 19 proyek bersih)
"""

from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np

# Daftar fitur prediktif resmi FlakeFlagger (Total: 23 fitur)
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

PROJECTS_WITH_MISSING_VALUES: List[str] = [
    "wildfly",
    "logback",
    "spring-boot",
    "wro4j",
    "handlebars.java",
]


def resolve_data_path(custom_path: str = None) -> Path:
    """Mencari lokasi test_features.csv secara otomatis."""
    if custom_path and Path(custom_path).exists():
        return Path(custom_path)
    
    candidates = [
        Path("test_features.csv"),
        Path("../../test_features.csv"),
        Path("../test_features.csv"),
        Path(__file__).resolve().parent.parent.parent.parent / "test_features.csv"
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("test_features.csv tidak ditemukan di direktori yang diperiksa.")


def load_dataset(
    cleaning_strategy: str = "C1",
    data_path: str = None
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Memuat dataset dengan menerapkan strategi pembersihan C1 atau C2.
    
    Args:
        cleaning_strategy: 'C1' (Row-Level Drop) atau 'C2' (Project-Level Drop).
        data_path: Path opsional ke file test_features.csv.
        
    Returns:
        df: DataFrame yang telah dibersihkan.
        features: List nama fitur prediktif (23 fitur).
        projects: List nama proyek unik yang tersisa.
    """
    file_path = resolve_data_path(data_path)
    df = pd.read_csv(file_path)

    # Pastikan Unnamed: 0 dihapus jika ada
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    features = [f for f in ALL_PREDICTIVE_FEATURES if f in df.columns]

    if cleaning_strategy.upper() == "C1":
        # C1: Row-Level Drop — hapus baris yang memiliki NaN pada kolom fitur
        initial_len = len(df)
        df = df.dropna(subset=features).copy()
        print(f"[DataLoader C1] Row-Level Drop: dari {initial_len} baris menjadi {len(df)} baris. Proyek aktif: {df['project'].nunique()} proyek.")
    elif cleaning_strategy.upper() == "C2":
        # C2: Project-Level Drop — hapus seluruh 5 proyek yang memiliki missing values
        initial_proj = df['project'].nunique()
        df = df[~df['project'].isin(PROJECTS_WITH_MISSING_VALUES)].copy()
        # Bersihkan juga sisa baris NaN jika ada
        df = df.dropna(subset=features).copy()
        print(f"[DataLoader C2] Project-Level Drop: dari {initial_proj} proyek menjadi {df['project'].nunique()} proyek ({len(df)} baris).")
    else:
        raise ValueError(f"Strategi cleaning tidak dikenal: {cleaning_strategy}. Pilih 'C1' atau 'C2'.")

    projects = sorted(df["project"].unique().tolist())
    return df, features, projects


if __name__ == "__main__":
    df_c1, feat_c1, proj_c1 = load_dataset("C1")
    print(f"C1 Features: {len(feat_c1)}, Projects: {len(proj_c1)}, Flaky count: {df_c1['flaky'].sum()} ({df_c1['flaky'].mean()*100:.2f}%)")

    df_c2, feat_c2, proj_c2 = load_dataset("C2")
    print(f"C2 Features: {len(feat_c2)}, Projects: {len(proj_c2)}, Flaky count: {df_c2['flaky'].sum()} ({df_c2['flaky'].mean()*100:.2f}%)")
