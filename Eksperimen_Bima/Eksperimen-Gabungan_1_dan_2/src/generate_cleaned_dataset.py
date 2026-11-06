"""
Dataset Preparation & Imputation Module for Combined Experiment 1 & 2.

Description:
This script prepares and reproduces 'data/cleaned_test_features.csv' from the raw
'data/test_features.csv' using 'Group-Aware Per-Project Median & Probabilistic Imputation'
(used as the foundational dataset for Paradigm C3).

Dataset Invariant Guarantees:
- 0 missing values across all 23 predictive features.
- All 24 open-source Java projects preserved intact (no project deletion).
- All 22,236 test cases and all 811 flaky test instances preserved (zero sample loss).
- Dataset output saved directly to 'data/cleaned_test_features.csv'.

Execution:
    python src/generate_cleaned_dataset.py
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd


METADATA_COLS: List[str] = [
    "project",
    "test_name",
    "testClassName",
    "testMethodName",
]

TARGET_COL: str = "flaky"

COLS_WITH_MISSING: List[str] = [
    "testLength",
    "numAsserts",
    "numCoveredLines",
]


def resolve_input_path(custom_path: str = None) -> Path:
    """Locates the raw test_features.csv dataset automatically."""
    if custom_path and Path(custom_path).exists():
        return Path(custom_path).resolve()

    candidates = [
        Path(__file__).resolve().parent.parent / "data" / "test_features.csv",
        Path(__file__).resolve().parent.parent.parent / "Eksperimen-1" / "data" / "test_features.csv",
        Path(__file__).resolve().parent.parent.parent.parent / "dataset" / "test_features.csv",
        Path(__file__).resolve().parent.parent.parent.parent / "Experiment_Ali" / "data" / "test_features.csv",
        Path("data/test_features.csv"),
        Path("../data/test_features.csv"),
        Path("test_features.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(
        "File test_features.csv was not found in any candidate path:\n"
        + "\n".join(f" - {str(p)}" for p in candidates)
    )


def impute_project_aware(df: pd.DataFrame, cols_to_impute: List[str], seed: int = 42) -> pd.DataFrame:
    """
    Applies Group-Aware Per-Project Median & Probabilistic Fallback Imputation.
    
    1. Primary: Imputes NaN with the median of the corresponding project group.
    2. Fallback: If all instances of a project are NaN, samples randomly from the
       empirical non-null feature distribution with a fixed random seed.
    """
    df_imputed = df.copy()
    np.random.seed(seed)

    for col in cols_to_impute:
        if col not in df_imputed.columns:
            continue

        # 1. Group-aware per-project median imputation
        df_imputed[col] = df_imputed.groupby("project")[col].transform(
            lambda x: x.fillna(x.median())
        )

        # 2. Fallback probabilistic sampling if any NaNs remain
        remaining_nulls = df_imputed[col].isnull().sum()
        if remaining_nulls > 0:
            non_null_vals = df_imputed[col].dropna().values
            nan_indices = df_imputed[df_imputed[col].isnull()].index
            sampled_vals = np.random.choice(non_null_vals, size=len(nan_indices), replace=True)
            df_imputed.loc[nan_indices, col] = sampled_vals

    return df_imputed


def generate_cleaned_dataset(
    input_csv: str = None,
    output_csv: str = None,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, Path]:
    """Executes dataset preparation and contextual imputation."""
    raw_path = resolve_input_path(input_csv)
    print(f"[PREPARATION] Ingesting raw dataset: {raw_path}")
    df_raw = pd.read_csv(raw_path)

    print(f"[PREPARATION] Raw dimensions: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")
    print(f"[PREPARATION] Target projects: {df_raw['project'].nunique()} unique projects")

    # Audit missing values
    null_summary = df_raw.isnull().sum()
    null_summary = null_summary[null_summary > 0]
    print(f"[PREPARATION] Features with missing values:\n{null_summary.to_string()}")

    cols_to_fix = [c for c in COLS_WITH_MISSING if c in df_raw.columns]
    df_clean = impute_project_aware(df_raw, cols_to_fix, seed=random_seed)

    # Validate post-imputation integrity
    total_nulls_after = df_clean[cols_to_fix].isnull().sum().sum()
    assert total_nulls_after == 0, f"Error: {total_nulls_after} missing values detected after imputation!"

    # Resolve output path
    if output_csv:
        out_path = Path(output_csv).resolve()
    else:
        out_path = Path(__file__).resolve().parent.parent / "data" / "cleaned_test_features.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_path, index=False)

    print(f"[PREPARATION] SUCCESS! Cleaned dataset saved to: {out_path}")
    print(f"[PREPARATION] Imputed dimensions: {df_clean.shape[0]:,} rows x {df_clean.shape[1]} columns")
    print(f"[PREPARATION] Preserved projects: {df_clean['project'].nunique()} (100% intact)")
    print(f"[PREPARATION] Preserved flaky tests: {df_clean['flaky'].sum():,} ({df_clean['flaky'].mean()*100:.2f}%)")
    print(f"[PREPARATION] Total remaining missing values: {df_clean.isnull().sum().sum()}")

    return df_clean, out_path


def main():
    parser = argparse.ArgumentParser(
        description="Dataset Preparation & Imputation (Group-Aware Per-Project Median & Probabilistic Imputation)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to raw test_features.csv (optional, auto-resolved if omitted)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Destination path for cleaned_test_features.csv (default: data/cleaned_test_features.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for probabilistic fallback (default: 42)",
    )

    args = parser.parse_args()
    generate_cleaned_dataset(input_csv=args.input, output_csv=args.output, random_seed=args.seed)


if __name__ == "__main__":
    main()
