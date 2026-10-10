"""
Modul Mesin Ekstraksi SHAP TreeExplainer untuk Model XGBoost GPU dan Random Forest.
Mendukung evaluasi Within-Project (Pooled CV) dan Cross-Project (LOPO-CV).
"""

import sys
import time
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import shap

from .data_loader import load_dataset, ALL_PREDICTIVE_FEATURES, TAXONOMY_MAP


def create_xgb_model(random_state: int = 42, n_estimators: int = 100, max_depth: int = 6) -> xgb.XGBClassifier:
    """Membuat instance model XGBoost dengan akselerasi GPU (CUDA)."""
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        tree_method="hist",
        device="cuda",
        eval_metric="logloss",
        random_state=random_state
    )


def create_rf_model(random_state: int = 42, n_estimators: int = 100) -> RandomForestClassifier:
    """Membuat instance model Random Forest multi-core CPU."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state
    )


def compute_shap_within_project(
    df: pd.DataFrame,
    features: List[str],
    model_type: str = "xgboost",
    n_splits: int = 10,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Menghitung nilai SHAP pada skenario Within-Project (Pooled Stratified K-Fold CV).
    Seluruh 21.963 data dievaluasi secara out-of-fold (zero test leakage).
    """
    print(f"\n[Within-Project SHAP] Memulai {n_splits}-Fold Stratified CV (Model: {model_type.upper()})...")
    start_time = time.time()

    X = df[features].copy()
    y = df["flaky"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Matriks penampung SHAP values sesuai urutan baris asli df
    shap_values_full = np.zeros(X.shape, dtype=np.float32)
    base_values_full = np.zeros(len(df), dtype=np.float32)
    preds_proba_full = np.zeros(len(df), dtype=np.float32)

    fold_idx = 1
    for train_idx, test_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]

        if model_type.lower() == "xgboost":
            clf = create_xgb_model(random_state=random_state + fold_idx)
            clf.fit(X_train, y_train)
            explainer = shap.TreeExplainer(clf)
            sv = explainer(X_test, check_additivity=False)
            shap_values_full[test_idx] = sv.values
            base_values_full[test_idx] = sv.base_values
            preds_proba_full[test_idx] = clf.predict_proba(X_test)[:, 1]
        else:
            clf = create_rf_model(random_state=random_state + fold_idx)
            clf.fit(X_train, y_train)
            explainer = shap.TreeExplainer(clf)
            sv = explainer(X_test, check_additivity=False)
            # Untuk RF binary classification, ambil kelas 1 (flaky)
            if sv.values.ndim == 3:
                shap_values_full[test_idx] = sv.values[:, :, 1]
                base_values_full[test_idx] = sv.base_values[:, 1]
            else:
                shap_values_full[test_idx] = sv.values
                base_values_full[test_idx] = sv.base_values
            preds_proba_full[test_idx] = clf.predict_proba(X_test)[:, 1]

        fold_idx += 1

    elapsed = time.time() - start_time
    print(f"[Within-Project SHAP] Selesai dalam {elapsed:.2f} detik.")

    mean_abs_shap = np.mean(np.abs(shap_values_full), axis=0)
    shap_summary_df = pd.DataFrame({
        "feature": features,
        "mean_abs_shap": mean_abs_shap,
        "dimension": [TAXONOMY_MAP.get(f, "Unknown") for f in features]
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    shap_summary_df["rank"] = shap_summary_df.index + 1
    total_shap = shap_summary_df["mean_abs_shap"].sum()
    shap_summary_df["relative_importance_pct"] = (shap_summary_df["mean_abs_shap"] / total_shap) * 100.0

    return {
        "scenario": "within_project",
        "model_type": model_type,
        "elapsed_seconds": elapsed,
        "summary_df": shap_summary_df,
        "shap_values": shap_values_full,
        "base_values": base_values_full,
        "preds_proba": preds_proba_full,
        "features": features,
        "X": X
    }


def compute_shap_cross_project(
    df: pd.DataFrame,
    features: List[str],
    projects: List[str],
    model_type: str = "xgboost",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Menghitung nilai SHAP pada skenario Cross-Project (Leave-One-Project-Out / LOPO-CV).
    Setiap proyek target diuji menggunakan model yang dilatih pada 23 proyek lainnya.
    """
    print(f"\n[Cross-Project SHAP] Memulai 24-Fold LOPO-CV (Model: {model_type.upper()})...")
    start_time = time.time()

    X = df[features].copy()
    y = df["flaky"].values

    shap_values_full = np.zeros(X.shape, dtype=np.float32)
    base_values_full = np.zeros(len(df), dtype=np.float32)
    preds_proba_full = np.zeros(len(df), dtype=np.float32)

    fold_details = []

    for idx, target_proj in enumerate(projects, 1):
        target_mask = (df["project"] == target_proj).values
        train_mask = ~target_mask

        X_train, y_train = X.iloc[train_mask], y[train_mask]
        X_test, y_test = X.iloc[target_mask], y[target_mask]

        t_fold_start = time.time()
        if model_type.lower() == "xgboost":
            clf = create_xgb_model(random_state=random_state + idx)
            clf.fit(X_train, y_train)
            explainer = shap.TreeExplainer(clf)
            sv = explainer(X_test, check_additivity=False)
            shap_values_full[target_mask] = sv.values
            base_values_full[target_mask] = sv.base_values
            preds_proba_full[target_mask] = clf.predict_proba(X_test)[:, 1]
        else:
            clf = create_rf_model(random_state=random_state + idx)
            clf.fit(X_train, y_train)
            explainer = shap.TreeExplainer(clf)
            sv = explainer(X_test, check_additivity=False)
            if sv.values.ndim == 3:
                shap_values_full[target_mask] = sv.values[:, :, 1]
                base_values_full[target_mask] = sv.base_values[:, 1]
            else:
                shap_values_full[target_mask] = sv.values
                base_values_full[target_mask] = sv.base_values
            preds_proba_full[target_mask] = clf.predict_proba(X_test)[:, 1]

        t_fold_elapsed = time.time() - t_fold_start
        fold_details.append({
            "fold": idx,
            "project": target_proj,
            "test_samples": int(np.sum(target_mask)),
            "flaky_samples": int(np.sum(y_test)),
            "fold_time_sec": t_fold_elapsed
        })
        if idx % 6 == 0 or idx == len(projects):
            print(f"  > Fold {idx:2d}/{len(projects)} ({target_proj:<16}): {np.sum(target_mask):4d} tests ({t_fold_elapsed:.2f}s)")

    elapsed = time.time() - start_time
    print(f"[Cross-Project SHAP] LOPO-CV selesai dalam {elapsed:.2f} detik.")

    mean_abs_shap = np.mean(np.abs(shap_values_full), axis=0)
    shap_summary_df = pd.DataFrame({
        "feature": features,
        "mean_abs_shap": mean_abs_shap,
        "dimension": [TAXONOMY_MAP.get(f, "Unknown") for f in features]
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    shap_summary_df["rank"] = shap_summary_df.index + 1
    total_shap = shap_summary_df["mean_abs_shap"].sum()
    shap_summary_df["relative_importance_pct"] = (shap_summary_df["mean_abs_shap"] / total_shap) * 100.0

    return {
        "scenario": "cross_project",
        "model_type": model_type,
        "elapsed_seconds": elapsed,
        "summary_df": shap_summary_df,
        "shap_values": shap_values_full,
        "base_values": base_values_full,
        "preds_proba": preds_proba_full,
        "features": features,
        "X": X,
        "fold_details": fold_details
    }


def extract_case_studies(
    df: pd.DataFrame,
    features: List[str],
    shap_results: Dict[str, Any],
    target_project: str = "spring-boot",
    num_samples: int = 3
) -> Dict[str, Any]:
    """
    Mengekstrak sampel kasus lokal (local explanations) untuk proyek spesifik (misal spring-boot).
    Mengambil test case flaky dengan probabilitas tertinggi dan test case non-flaky representatif.
    """
    proj_mask = (df["project"] == target_project).values
    proj_indices = np.where(proj_mask)[0]

    flaky_indices = [i for i in proj_indices if df.iloc[i]["flaky"] == 1]
    nonflaky_indices = [i for i in proj_indices if df.iloc[i]["flaky"] == 0]

    # Urutkan flaky berdasarkan probabilitas tertinggi
    flaky_sorted = sorted(flaky_indices, key=lambda i: shap_results["preds_proba"][i], reverse=True)
    # Urutkan non-flaky berdasarkan probabilitas terendah
    nonflaky_sorted = sorted(nonflaky_indices, key=lambda i: shap_results["preds_proba"][i])

    selected_flaky = flaky_sorted[:num_samples]
    selected_nonflaky = nonflaky_sorted[:num_samples]

    cases = {
        "project": target_project,
        "flaky_cases": [],
        "non_flaky_cases": []
    }

    for rank, idx in enumerate(selected_flaky, 1):
        row = df.iloc[idx]
        cases["flaky_cases"].append({
            "case_id": f"Flaky_Case_{rank}",
            "global_index": int(idx),
            "test_name": row.get("test_name", f"Test_{idx}"),
            "flaky_label": int(row["flaky"]),
            "pred_proba": float(shap_results["preds_proba"][idx]),
            "base_value": float(shap_results["base_values"][idx]),
            "feature_values": {f: float(row[f]) for f in features},
            "shap_values": {f: float(shap_results["shap_values"][idx, f_idx]) for f_idx, f in enumerate(features)}
        })

    for rank, idx in enumerate(selected_nonflaky, 1):
        row = df.iloc[idx]
        cases["non_flaky_cases"].append({
            "case_id": f"NonFlaky_Case_{rank}",
            "global_index": int(idx),
            "test_name": row.get("test_name", f"Test_{idx}"),
            "flaky_label": int(row["flaky"]),
            "pred_proba": float(shap_results["preds_proba"][idx]),
            "base_value": float(shap_results["base_values"][idx]),
            "feature_values": {f: float(row[f]) for f in features},
            "shap_values": {f: float(shap_results["shap_values"][idx, f_idx]) for f_idx, f in enumerate(features)}
        })

    return cases


def run_full_shap_pipeline(
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Menjalankan seluruh pipeline ekstraksi SHAP untuk XGBoost GPU dan Random Forest,
    baik Within-Project maupun Cross-Project.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "results"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df, features, projects = load_dataset(cleaning_strategy="C1", data_path=data_path)

    # 1. Cross-Project XGBoost GPU
    print("\n" + "=" * 60)
    print("STEP 1: Ekstraksi SHAP Cross-Project LOPO-CV (XGBoost GPU)")
    print("=" * 60)
    xgb_cross = compute_shap_cross_project(df, features, projects, model_type="xgboost")
    xgb_cross["summary_df"].to_csv(output_dir / "shap_summary_cross_project.csv", index=False)

    # 2. Within-Project XGBoost GPU
    print("\n" + "=" * 60)
    print("STEP 2: Ekstraksi SHAP Within-Project Pooled CV (XGBoost GPU)")
    print("=" * 60)
    xgb_within = compute_shap_within_project(df, features, model_type="xgboost")
    xgb_within["summary_df"].to_csv(output_dir / "shap_summary_within_project.csv", index=False)

    # 3. Cross-Project Random Forest (sebagai model pembanding Model B)
    print("\n" + "=" * 60)
    print("STEP 3: Ekstraksi SHAP Cross-Project LOPO-CV (Random Forest)")
    print("=" * 60)
    rf_cross = compute_shap_cross_project(df, features, projects, model_type="random_forest")
    rf_cross["summary_df"].to_csv(output_dir / "shap_summary_cross_project_rf.csv", index=False)

    # 4. Within-Project Random Forest
    print("\n" + "=" * 60)
    print("STEP 4: Ekstraksi SHAP Within-Project Pooled CV (Random Forest)")
    print("=" * 60)
    rf_within = compute_shap_within_project(df, features, model_type="random_forest")
    rf_within["summary_df"].to_csv(output_dir / "shap_summary_within_project_rf.csv", index=False)

    # 5. Ekstraksi Case Study spring-boot
    print("\n" + "=" * 60)
    print("STEP 5: Ekstraksi Studi Kasus Lokal (spring-boot)")
    print("=" * 60)
    case_studies = extract_case_studies(df, features, xgb_cross, target_project="spring-boot", num_samples=3)
    with open(output_dir / "spring_boot_case_studies.json", "w") as f:
        json.dump(case_studies, f, indent=2)

    # Simpan binary cache matriks SHAP untuk visualisasi instan
    with open(output_dir / "shap_cache_xgb.pkl", "wb") as f:
        pickle.dump({
            "xgb_cross": {
                "shap_values": xgb_cross["shap_values"],
                "base_values": xgb_cross["base_values"],
                "preds_proba": xgb_cross["preds_proba"],
                "summary_df": xgb_cross["summary_df"],
                "elapsed_seconds": xgb_cross["elapsed_seconds"],
                "fold_details": xgb_cross["fold_details"]
            },
            "xgb_within": {
                "shap_values": xgb_within["shap_values"],
                "base_values": xgb_within["base_values"],
                "preds_proba": xgb_within["preds_proba"],
                "summary_df": xgb_within["summary_df"],
                "elapsed_seconds": xgb_within["elapsed_seconds"]
            },
            "rf_cross": {
                "shap_values": rf_cross["shap_values"],
                "base_values": rf_cross["base_values"],
                "preds_proba": rf_cross["preds_proba"],
                "summary_df": rf_cross["summary_df"],
                "elapsed_seconds": rf_cross["elapsed_seconds"]
            },
            "rf_within": {
                "shap_values": rf_within["shap_values"],
                "base_values": rf_within["base_values"],
                "preds_proba": rf_within["preds_proba"],
                "summary_df": rf_within["summary_df"],
                "elapsed_seconds": rf_within["elapsed_seconds"]
            },
            "features": features,
            "projects": projects,
            "case_studies": case_studies
        }, f)

    print(f"\n[SUKSES] Seluruh ekstraksi SHAP selesai dan tersimpan di: {output_dir}")
    return {
        "xgb_cross": xgb_cross,
        "xgb_within": xgb_within,
        "rf_cross": rf_cross,
        "rf_within": rf_within,
        "case_studies": case_studies,
        "df": df,
        "features": features,
        "projects": projects
    }


if __name__ == "__main__":
    run_full_shap_pipeline()
