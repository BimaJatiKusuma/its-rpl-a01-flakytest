"""
Modul Ingestion & Preprocessing Data Terpadu untuk Eksperimen Gabungan 1 & 2.
Mendukung tiga (3) strategi data cleaning/preprocessing:
- C1: Row-Level Drop (membuang baris NaN, mempertahankan 24 proyek, 21.963 baris)
- C2: Project-Level Drop (membuang 5 proyek ber-NaN, mempertahankan 19 proyek, 16.458 baris)
- C3: Group-Aware Median Imputation (imputasi cerdas, mempertahankan 24 proyek, 22.236 baris utuh)
"""

from pathlib import Path
from typing import List, Tuple, Optional
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

PROJECTS_WITH_MISSING_VALUES: List[str] = [
    "wildfly",
    "logback",
    "spring-boot",
    "wro4j",
    "handlebars.java",
]


def resolve_file(filename: str, custom_dir: Optional[str] = None) -> Path:
    """Mencari lokasi berkas data secara otomatis."""
    if custom_dir and (Path(custom_dir) / filename).exists():
        return (Path(custom_dir) / filename).resolve()

    candidates = [
        Path(__file__).resolve().parent.parent / "data" / filename,
        Path("data") / filename,
        Path("../data") / filename,
        Path(__file__).resolve().parent.parent.parent / "Eksperimen-1" / "data" / filename,
        Path(__file__).resolve().parent.parent.parent / "Eksperiment-2" / "data" / filename,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(f"Berkas '{filename}' tidak ditemukan di jalur data mana pun.")


def load_dataset(
    cleaning_strategy: str = "C3",
    data_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Memuat dataset sesuai strategi penanganan data:
    - C1: Row-Level Drop (dari test_features.csv)
    - C2: Project-Level Drop (dari test_features.csv)
    - C3: Group-Aware Median Imputation (dari cleaned_test_features.csv)

    Returns:
        df: DataFrame yang telah dibersihkan
        features: List nama fitur prediktif (23 fitur)
        projects: List nama proyek unik
    """
    strat = cleaning_strategy.upper()

    if strat == "C1":
        file_path = resolve_file("test_features.csv", data_dir)
        df = pd.read_csv(file_path)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        # C1: Drop rows with NaN
        df = df.dropna(subset=ALL_PREDICTIVE_FEATURES).copy()
        projects = sorted(df["project"].unique().tolist())
        print(f"[DATA LOADER C1] Row-Level Drop: {len(df):,} baris | {len(projects)} proyek | Flaky: {df['flaky'].sum():,}")

    elif strat == "C2":
        file_path = resolve_file("test_features.csv", data_dir)
        df = pd.read_csv(file_path)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        # C2: Drop entire projects with missing values
        df = df[~df["project"].isin(PROJECTS_WITH_MISSING_VALUES)].copy()
        df = df.dropna(subset=ALL_PREDICTIVE_FEATURES).copy()
        projects = sorted(df["project"].unique().tolist())
        print(f"[DATA LOADER C2] Project-Level Drop: {len(df):,} baris | {len(projects)} proyek | Flaky: {df['flaky'].sum():,}")

    elif strat == "C3":
        file_path = resolve_file("cleaned_test_features.csv", data_dir)
        df = pd.read_csv(file_path)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        projects = sorted(df["project"].unique().tolist())
        print(f"[DATA LOADER C3] Group-Aware Median Imputation: {len(df):,} baris | {len(projects)} proyek (100% Utuh) | Flaky: {df['flaky'].sum():,}")

    else:
        raise ValueError(f"Cleaning strategy '{cleaning_strategy}' tidak dikenali. Pilih C1, C2, atau C3.")

    # Validasi bahwa tidak ada NaN pada fitur
    null_count = df[ALL_PREDICTIVE_FEATURES].isnull().sum().sum()
    if null_count > 0:
        raise ValueError(f"Terdeteksi {null_count} nilai NaN yang tidak terselesaikan pada strategi {strat}!")

    return df, ALL_PREDICTIVE_FEATURES, projects
