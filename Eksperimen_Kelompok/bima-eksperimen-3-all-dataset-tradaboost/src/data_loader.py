"""
Modul Pemuatan dan Penyelarasan Dataset Multi-Benchmark (FlakeFlagger & IDoFT).
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd

# 23 Fitur prediktif standar FlakeFlagger
PREDICTIVE_FEATURES = [
    'assertion-roulette', 'conditional-test-logic', 'eager-test', 'fire-and-forget',
    'indirect-testing', 'mystery-guest', 'resource-optimism', 'test-run-war',
    'testLength', 'numAsserts', 'numCoveredLines', 'ExecutionTime',
    'projectSourceLinesCovered', 'projectSourceClassesCovered',
    'hIndexModificationsPerCoveredLine_window5', 'hIndexModificationsPerCoveredLine_window10',
    'hIndexModificationsPerCoveredLine_window25', 'hIndexModificationsPerCoveredLine_window50',
    'hIndexModificationsPerCoveredLine_window75', 'hIndexModificationsPerCoveredLine_window100',
    'hIndexModificationsPerCoveredLine_window500', 'hIndexModificationsPerCoveredLine_window10000',
    'num_third_party_libs'
]


def get_base_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def load_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Memuat dataset dengan standardisasi kolom dan penanganan duplikasi/tipe data.

    Args:
        dataset_name: 'flakeflagger' atau 'idoft'

    Returns:
        pd.DataFrame yang siap digunakan untuk evaluasi model.
    """
    base_dir = get_base_dir()
    dname = dataset_name.lower().strip()

    if dname in ['flakeflagger', 'ff']:
        file_path = base_dir / 'data' / 'cleaned_flakeflagger_features.csv'
        if not file_path.exists():
            file_path = base_dir.parent.parent / 'Experiment_Ali' / 'data' / 'cleaned_test_features.csv'
    elif dname in ['flakeflagger_rowdrop', 'ff_rowdrop', 'rowdrop']:
        raw_path = base_dir.parent.parent / 'test_features.csv'
        if not raw_path.exists():
            file_path = base_dir / 'data' / 'cleaned_flakeflagger_features.csv'
        else:
            df_raw = pd.read_csv(raw_path)
            df_drop = df_raw.dropna(subset=['testLength', 'numAsserts', 'numCoveredLines']).copy()
            if 'project_clean' not in df_drop.columns and 'project' in df_drop.columns:
                df_drop['project_clean'] = df_drop['project'].astype(str).str.lower().str.strip()
            df_drop['flaky'] = df_drop['flaky'].astype(int)
            for col in PREDICTIVE_FEATURES:
                df_drop[col] = pd.to_numeric(df_drop[col], errors='coerce').fillna(0.0)
            return df_drop
    elif dname in ['idoft', 'id']:
        file_path = base_dir / 'data' / 'cleaned_idoft_features.csv'
        if not file_path.exists():
            file_path = base_dir.parent / 'bima-eksperimen-3-idoft-dataset' / 'data' / 'cleaned_idoft_test_features.csv'
    else:
        raise ValueError(f"Dataset '{dataset_name}' tidak dikenal. Pilih 'flakeflagger', 'flakeflagger_rowdrop', atau 'idoft'.")

    df = pd.read_csv(file_path)

    # Standardisasi nama kolom proyek
    if 'project_clean' not in df.columns:
        if 'project' in df.columns:
            df['project_clean'] = df['project'].astype(str).str.lower().str.strip()
        else:
            raise KeyError(f"Kolom project tidak ditemukan di {file_path}")
    else:
        df['project_clean'] = df['project_clean'].astype(str).str.lower().str.strip()

    # Pastikan label flaky bertipe integer 0/1
    df['flaky'] = df['flaky'].astype(int)

    # Validasi keberadaan 23 fitur prediktif
    missing_feats = [f for f in PREDICTIVE_FEATURES if f not in df.columns]
    if missing_feats:
        raise KeyError(f"Fitur berikut hilang dari {dataset_name}: {missing_feats}")

    # Konversi fitur numerik dan tangani jika ada null
    for col in PREDICTIVE_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    return df


def get_project_splits(
    df: pd.DataFrame,
    target_project: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Membagi dataset menjadi Source ($N-1$ proyek) dan Target (1 proyek) untuk LOPO-CV.
    """
    target_proj_clean = target_project.lower().strip()
    mask_target = (df['project_clean'] == target_proj_clean)

    df_source = df[~mask_target]
    df_target = df[mask_target]

    X_source = df_source[PREDICTIVE_FEATURES].values.astype(np.float64)
    y_source = df_source['flaky'].values.astype(int)

    X_target = df_target[PREDICTIVE_FEATURES].values.astype(np.float64)
    y_target = df_target['flaky'].values.astype(int)

    return X_source, y_source, X_target, y_target


def get_dataset_profile(df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """
    Menganalisis profil demografi dan statistik dasar dataset.
    """
    projects = sorted(df['project_clean'].unique().tolist())
    proj_stats = []

    for p in projects:
        sub = df[df['project_clean'] == p]
        n_flaky = int(sub['flaky'].sum())
        n_total = len(sub)
        rate = (n_flaky / n_total) * 100.0 if n_total > 0 else 0.0
        proj_stats.append({
            'project': p,
            'total_tests': n_total,
            'flaky_tests': n_flaky,
            'flakiness_rate_pct': round(rate, 2),
            'has_flaky': (n_flaky > 0)
        })

    return {
        'dataset_name': dataset_name,
        'total_instances': len(df),
        'n_projects': len(projects),
        'total_flaky': int(df['flaky'].sum()),
        'overall_flakiness_pct': round(float(df['flaky'].mean() * 100.0), 2),
        'project_breakdown': proj_stats
    }


if __name__ == '__main__':
    for d in ['flakeflagger', 'idoft']:
        df = load_dataset(d)
        prof = get_dataset_profile(df, d)
        print(f"[{d.upper()}] Rows: {prof['total_instances']}, Projects: {prof['n_projects']}, Flaky: {prof['total_flaky']} ({prof['overall_flakiness_pct']}%)")
