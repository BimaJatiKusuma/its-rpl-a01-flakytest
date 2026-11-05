"""
Modul dan Script Penghasil Dataset Bersih (Cleaned Dataset Generator)
untuk Eksperimen 2: Cross-Project Flaky Test Prediction.

Deskripsi:
Script ini mereproduksi dataset 'cleaned_test_features.csv' dari dataset mentah
'test_features.csv' menggunakan strategi 'Group-Aware Per-Project Median & Probabilistic Imputation'
(sebagaimana dikembangkan oleh Ali pada Phase 1 penelitian).

Hasil:
- 0 missing values pada seluruh fitur prediktif.
- Seluruh 24 proyek Java open-source dipertahankan (tanpa pembuangan proyek C2).
- Seluruh 22.236 test cases dan 811 flaky test dilestarikan secara utuh.
- Dataset disimpan langsung ke 'data/cleaned_test_features.csv'.

Cara Menjalankan:
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
    """Mencari path dataset mentah test_features.csv secara otomatis."""
    if custom_path and Path(custom_path).exists():
        return Path(custom_path).resolve()

    candidates = [
        Path(__file__).resolve().parent.parent / "data" / "test_features.csv",
        Path(__file__).resolve().parent.parent.parent / "Eksperimen-1" / "data" / "test_features.csv",
        Path(__file__).resolve().parent.parent.parent.parent / "test_features.csv",
        Path(__file__).resolve().parent.parent.parent.parent / "Experiment_Ali" / "data" / "test_features.csv",
        Path("data/test_features.csv"),
        Path("../data/test_features.csv"),
        Path("test_features.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(
        "File test_features.csv tidak ditemukan di jalur pencarian:\n"
        + "\n".join(f" - {str(p)}" for p in candidates)
    )


def impute_project_aware(df: pd.DataFrame, cols_to_impute: List[str], seed: int = 42) -> pd.DataFrame:
    """
    Menerapkan Group-Aware Per-Project Median & Probabilistic Fallback Imputation.
    
    1. Primary: Mengisi NaN dengan median dari proyek yang sama.
    2. Fallback: Jika seluruh test case pada suatu proyek bernilai NaN, ambil sampel
       secara acak dari distribusi empiris non-null dengan random seed tetap.
    """
    df_imputed = df.copy()
    np.random.seed(seed)

    for col in cols_to_impute:
        if col not in df_imputed.columns:
            continue

        # 1. Imputasi berbasis median per kelompok proyek
        df_imputed[col] = df_imputed.groupby("project")[col].transform(
            lambda x: x.fillna(x.median())
        )

        # 2. Fallback probabilistic sampling jika masih tersisa NaN
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
    """Eksekusi proses pembersihan dan imputasi data."""
    raw_path = resolve_input_path(input_csv)
    print(f"[IMPUTASI] Membaca dataset mentah: {raw_path}")
    df_raw = pd.read_csv(raw_path)

    print(f"[IMPUTASI] Dimensi awal: {df_raw.shape[0]} baris, {df_raw.shape[1]} kolom")
    print(f"[IMPUTASI] Total proyek: {df_raw['project'].nunique()} proyek")

    # Identifikasi missing value
    null_summary = df_raw.isnull().sum()
    null_summary = null_summary[null_summary > 0]
    print(f"[IMPUTASI] Kolom dengan missing values:\n{null_summary.to_string()}")

    cols_to_fix = [c for c in COLS_WITH_MISSING if c in df_raw.columns]
    df_clean = impute_project_aware(df_raw, cols_to_fix, seed=random_seed)

    # Validasi pasca-imputasi
    total_nulls_after = df_clean[cols_to_fix].isnull().sum().sum()
    assert total_nulls_after == 0, f"Masih terdapat {total_nulls_after} missing values setelah imputasi!"

    # Tentukan output path
    if output_csv:
        out_path = Path(output_csv).resolve()
    else:
        out_path = Path(__file__).resolve().parent.parent / "data" / "cleaned_test_features.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_path, index=False)

    print(f"[IMPUTASI] Sukses! Dataset bersih disimpan ke: {out_path}")
    print(f"[IMPUTASI] Dimensi akhir: {df_clean.shape[0]} baris, {df_clean.shape[1]} kolom")
    print(f"[IMPUTASI] Total Proyek: {df_clean['project'].nunique()} (100% lengkap)")
    print(f"[IMPUTASI] Total Flaky Test: {df_clean['flaky'].sum()} baris ({df_clean['flaky'].mean()*100:.2f}%)")
    print(f"[IMPUTASI] Total Missing Values: {df_clean.isnull().sum().sum()}")

    return df_clean, out_path


def main():
    parser = argparse.ArgumentParser(
        description="Script penghasil cleaned_test_features.csv (Group-Aware Median Imputation)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path ke file test_features.csv mentah (opsional, dicari otomatis jika kosong)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path tujuan file cleaned_test_features.csv (default: data/cleaned_test_features.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed untuk probabilistic fallback (default: 42)",
    )

    args = parser.parse_args()
    generate_cleaned_dataset(input_csv=args.input, output_csv=args.output, random_seed=args.seed)


if __name__ == "__main__":
    main()

