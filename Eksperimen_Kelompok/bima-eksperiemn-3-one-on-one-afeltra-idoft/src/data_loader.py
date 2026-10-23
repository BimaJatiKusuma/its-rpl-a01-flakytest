"""
Modul Loader Dataset IDoFT (Illinois Dataset of Flaky Tests).
Memuat dataset cleaned_idoft_test_features.csv (12.626 baris, 17 proyek Java, 264 flaky tests, 23 fitur prediktif).
Rujukan: Afeltra et al. (IEEE Access 2024), Alshammari et al. (ICSE 2021).
"""

from pathlib import Path
from typing import Tuple, List
import pandas as pd
import numpy as np

# Metadata columns yang harus dieksklusi dari fitur prediktif
METADATA_COLS = {
    "Unnamed: 0",
    "project",
    "project_clean",
    "test_name",
    "test_name_clean",
    "flaky",
    "flaky_original_flakeflagger",
    "testClassName",
    "testMethodName",
    "TestName",
    "Test",
    "Project",
    "IsFlaky",
    "NumFailingRuns"
}

def get_data_path() -> Path:
    """Mengembalikan path absolut ke cleaned_idoft_test_features.csv."""
    current_dir = Path(__file__).resolve().parent
    csv_path = current_dir.parent / "data" / "cleaned_idoft_test_features.csv"
    if not csv_path.exists():
        # Fallback path ke bima-eksperimen-3-idoft-dataset
        fallback = current_dir.parent.parent / "bima-eksperimen-3-idoft-dataset" / "data" / "cleaned_idoft_test_features.csv"
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"Dataset IDoFT tidak ditemukan di {csv_path} maupun {fallback}")
    return csv_path

def load_idoft_dataset() -> Tuple[pd.DataFrame, List[str], str]:
    """
    Memuat dataset IDoFT dan mengembalikan dataframe, daftar kolom fitur, dan kolom target.
    """
    csv_path = get_data_path()
    df = pd.read_csv(csv_path)

    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    target_col = "flaky"

    # Validasi integritas dataset
    assert len(feature_cols) == 23, f"Jumlah fitur harus 23, terdeteksi {len(feature_cols)}: {feature_cols}"
    assert target_col in df.columns, f"Kolom target {target_col} tidak ditemukan!"
    assert df[feature_cols].isnull().sum().sum() == 0, "Ditemukan nilai missing/null pada kolom fitur!"

    return df, feature_cols, target_col

def get_active_and_zero_flaky_projects(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Mengidentifikasi proyek dengan flaky test aktif (>0) dan proyek zero-flaky.
    """
    counts = df.groupby("project_clean")["flaky"].sum()
    active_projects = sorted(counts[counts > 0].index.tolist())
    zero_projects = sorted(counts[counts == 0].index.tolist())
    return active_projects, zero_projects

if __name__ == "__main__":
    df, feat_cols, tgt = load_idoft_dataset()
    active, zero = get_active_and_zero_flaky_projects(df)
    print(f"[OK] IDoFT Dataset Loaded Successfully!")
    print(f"     Total baris data       : {len(df):,}")
    print(f"     Total proyek           : {df['project_clean'].nunique()} (17)")
    print(f"     Active Flaky Projects  : {len(active)} ({active})")
    print(f"     Zero Flaky Projects    : {len(zero)} ({zero})")
    print(f"     Total Flaky Tests      : {df['flaky'].sum()} ({df['flaky'].mean()*100:.2f}%)")
    print(f"     Predictive Features (23): {feat_cols}")
