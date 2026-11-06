"""
Unified Data Ingestion & Preprocessing Module for Combined Experiment 1 & 2.
Supports three (3) data cleaning / preprocessing strategies:
- C1: Row-Level Discarding (omits NaN rows, preserves 24 projects, 21,963 instances)
- C2: Project-Level Discarding (omits 5 NaN projects, preserves 19 projects, 16,458 instances)
- C3: Group-Aware Median Imputation (contextual imputation, preserves 24 projects, 22,236 instances intact)
"""

from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

# Canonical 23 FlakeFlagger Predictive Features
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
    """Locates data files across standard directory candidate paths."""
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

    raise FileNotFoundError(f"Data file '{filename}' was not found in any candidate path.")


def load_dataset(
    cleaning_strategy: str = "C3",
    data_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Loads dataset adhering to the designated data cleaning paradigm:
    - C1: Row-Level Discarding (from test_features.csv)
    - C2: Project-Level Discarding (from test_features.csv)
    - C3: Group-Aware Median Imputation (from cleaned_test_features.csv)

    Returns:
        df: Preprocessed pandas DataFrame
        features: List of 23 predictive feature names
        projects: Sorted list of unique target project names
    """
    strat = cleaning_strategy.upper()

    if strat == "C1":
        file_path = resolve_file("test_features.csv", data_dir)
        df = pd.read_csv(file_path)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        # C1: Drop rows with NaN in predictive features
        df = df.dropna(subset=ALL_PREDICTIVE_FEATURES).copy()
        projects = sorted(df["project"].unique().tolist())
        print(f"[DATA LOADER C1] Row-Level Discarding: {len(df):,} rows | {len(projects)} projects | Flaky: {df['flaky'].sum():,}")

    elif strat == "C2":
        file_path = resolve_file("test_features.csv", data_dir)
        df = pd.read_csv(file_path)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        # C2: Drop entire projects containing missing values
        df = df[~df["project"].isin(PROJECTS_WITH_MISSING_VALUES)].copy()
        df = df.dropna(subset=ALL_PREDICTIVE_FEATURES).copy()
        projects = sorted(df["project"].unique().tolist())
        print(f"[DATA LOADER C2] Project-Level Discarding: {len(df):,} rows | {len(projects)} projects | Flaky: {df['flaky'].sum():,}")

    elif strat == "C3":
        file_path = resolve_file("cleaned_test_features.csv", data_dir)
        df = pd.read_csv(file_path)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        projects = sorted(df["project"].unique().tolist())
        print(f"[DATA LOADER C3] Group-Aware Median Imputation: {len(df):,} rows | {len(projects)} projects (100% Intact) | Flaky: {df['flaky'].sum():,}")

    else:
        raise ValueError(f"Cleaning strategy '{cleaning_strategy}' is unrecognized. Select C1, C2, or C3.")

    # Validate that no unhandled NaNs remain in predictive features
    null_count = df[ALL_PREDICTIVE_FEATURES].isnull().sum().sum()
    if null_count > 0:
        raise ValueError(f"Detected {null_count} unresolved NaN values under cleaning strategy {strat}!")

    return df, ALL_PREDICTIVE_FEATURES, projects
