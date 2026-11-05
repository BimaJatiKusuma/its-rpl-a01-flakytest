"""
Runner Eksperimen Komprehensif: Eksplorasi Seluruh Kombinasi Pipeline
(Data Cleaning x Feature Scaling x Imbalance Handling x Classifier Model x Sampling & Partitioning x TrAdaBoost Configs).
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_dataset, get_project_splits, PREDICTIVE_FEATURES
from pipeline_builder import get_scaler, apply_resampling, build_classifier, evaluate_predictions
from tradaboost import TrAdaBoostClassifier, split_target_data

DEFAULT_SEEDS = [42, 100, 2024, 777, 999]

COMPREHENSIVE_TREATMENTS = {
    # --- Group A: Sampling & Partitioning (Progression 0% -> 100%) ---
    "Baseline_Zero_Shot_RF": {
        "type": "baseline", "model": "M1", "scaler": "S2", "resampler": "B3", "target_ratio": 0.0,
        "group": "Sampling_Progression"
    },
    "Baseline_Zero_Shot_XGB": {
        "type": "baseline", "model": "M2", "scaler": "S2", "resampler": "B3", "target_ratio": 0.0,
        "group": "Sampling_Progression"
    },
    "TrAdaBoost_25": {
        "type": "tradaboost", "base_est": "DT_depth4", "scaler": "S2", "resampler": "B3", "target_ratio": 0.25,
        "n_est": 20, "second_half": True, "group": "Sampling_Progression"
    },
    "TrAdaBoost_50": {
        "type": "tradaboost", "base_est": "DT_depth4", "scaler": "S2", "resampler": "B3", "target_ratio": 0.50,
        "n_est": 20, "second_half": True, "group": "Sampling_Progression"
    },
    "TrAdaBoost_75": {
        "type": "tradaboost", "base_est": "DT_depth4", "scaler": "S2", "resampler": "B3", "target_ratio": 0.75,
        "n_est": 20, "second_half": True, "group": "Sampling_Progression"
    },
    "Within_Project": {
        "type": "within", "model": "M1", "scaler": "S2", "resampler": "B1", "target_ratio": 1.0,
        "group": "Sampling_Progression"
    },

    # --- Group B: Imbalance Handling on TrAdaBoost (Target 50%) ---
    "TrAdaBoost_50_B1_None": {
        "type": "tradaboost", "base_est": "DT_depth4", "scaler": "S2", "resampler": "B1", "target_ratio": 0.50,
        "n_est": 20, "second_half": True, "group": "Imbalance_Handling"
    },
    "TrAdaBoost_50_B2_SMOTE": {
        "type": "tradaboost", "base_est": "DT_depth4", "scaler": "S2", "resampler": "B2", "target_ratio": 0.50,
        "n_est": 20, "second_half": True, "group": "Imbalance_Handling"
    },
    "TrAdaBoost_50_B4_ClassWeight": {
        "type": "tradaboost", "base_est": "DT_depth4_CW", "scaler": "S2", "resampler": "B1", "target_ratio": 0.50,
        "n_est": 20, "second_half": True, "group": "Imbalance_Handling"
    },

    # --- Group C: Feature Scaling on TrAdaBoost (Target 50%) ---
    "TrAdaBoost_50_S1_Standard": {
        "type": "tradaboost", "base_est": "DT_depth4", "scaler": "S1", "resampler": "B3", "target_ratio": 0.50,
        "n_est": 20, "second_half": True, "group": "Feature_Scaling"
    },
    "TrAdaBoost_50_S3_Robust": {
        "type": "tradaboost", "base_est": "DT_depth4", "scaler": "S3", "resampler": "B3", "target_ratio": 0.50,
        "n_est": 20, "second_half": True, "group": "Feature_Scaling"
    },

    # --- Group D: Base Estimator & Tree Depth in TrAdaBoost (Target 50%) ---
    "TrAdaBoost_50_DT_depth2": {
        "type": "tradaboost", "base_est": "DT_depth2", "scaler": "S2", "resampler": "B3", "target_ratio": 0.50,
        "n_est": 20, "second_half": True, "group": "Base_Estimator_Depth"
    },
    "TrAdaBoost_50_DT_depth6": {
        "type": "tradaboost", "base_est": "DT_depth6", "scaler": "S2", "resampler": "B3", "target_ratio": 0.50,
        "n_est": 20, "second_half": True, "group": "Base_Estimator_Depth"
    },
    "TrAdaBoost_50_RF_base": {
        "type": "tradaboost", "base_est": "RF_shallow", "scaler": "S2", "resampler": "B3", "target_ratio": 0.50,
        "n_est": 10, "second_half": True, "group": "Base_Estimator_Depth"
    },

    # --- Group E: TrAdaBoost Iterations & Ensemble Voting (Target 50%) ---
    "TrAdaBoost_50_iter10": {
        "type": "tradaboost", "base_est": "DT_depth4", "scaler": "S2", "resampler": "B3", "target_ratio": 0.50,
        "n_est": 10, "second_half": True, "group": "TrAdaBoost_Hyperparameters"
    },
    "TrAdaBoost_50_iter30": {
        "type": "tradaboost", "base_est": "DT_depth4", "scaler": "S2", "resampler": "B3", "target_ratio": 0.50,
        "n_est": 30, "second_half": True, "group": "TrAdaBoost_Hyperparameters"
    },
    "TrAdaBoost_50_all_estimators": {
        "type": "tradaboost", "base_est": "DT_depth4", "scaler": "S2", "resampler": "B3", "target_ratio": 0.50,
        "n_est": 20, "second_half": False, "group": "TrAdaBoost_Hyperparameters"
    },

    # --- Group F: Zero-Shot Baseline Model & Preprocessing Variations ---
    "Baseline_Zero_Shot_DT": {
        "type": "baseline", "model": "M3", "scaler": "S2", "resampler": "B3", "target_ratio": 0.0,
        "group": "Baseline_Variations"
    },
    "Baseline_Zero_Shot_S1_RF": {
        "type": "baseline", "model": "M1", "scaler": "S1", "resampler": "B3", "target_ratio": 0.0,
        "group": "Baseline_Variations"
    },
    "Baseline_Zero_Shot_B1_RF": {
        "type": "baseline", "model": "M1", "scaler": "S2", "resampler": "B1", "target_ratio": 0.0,
        "group": "Baseline_Variations"
    },
    "Baseline_Zero_Shot_B2_RF": {
        "type": "baseline", "model": "M1", "scaler": "S2", "resampler": "B2", "target_ratio": 0.0,
        "group": "Baseline_Variations"
    },

    # --- Group G: Data Cleaning on FlakeFlagger (Row-Drop vs Imputed) ---
    "RowDrop_Baseline_RF": {
        "type": "baseline", "model": "M1", "scaler": "S2", "resampler": "B3", "target_ratio": 0.0,
        "dataset_override": "flakeflagger_rowdrop", "group": "Data_Cleaning"
    },
    "RowDrop_TrAdaBoost_50": {
        "type": "tradaboost", "base_est": "DT_depth4", "scaler": "S2", "resampler": "B3", "target_ratio": 0.50,
        "n_est": 20, "second_half": True, "dataset_override": "flakeflagger_rowdrop", "group": "Data_Cleaning"
    },
    "RowDrop_Within_Project": {
        "type": "within", "model": "M1", "scaler": "S2", "resampler": "B1", "target_ratio": 1.0,
        "dataset_override": "flakeflagger_rowdrop", "group": "Data_Cleaning"
    }
}


def get_tradaboost_base_estimator(base_est_code: str, random_state: int = 42):
    """
    Membuat estimator dasar untuk TrAdaBoost sesuai kode konfigurasi.
    """
    if base_est_code == "DT_depth2":
        return DecisionTreeClassifier(max_depth=2, random_state=random_state)
    elif base_est_code == "DT_depth4":
        return DecisionTreeClassifier(max_depth=4, random_state=random_state)
    elif base_est_code == "DT_depth4_CW":
        return DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=random_state)
    elif base_est_code == "DT_depth6":
        return DecisionTreeClassifier(max_depth=6, random_state=random_state)
    elif base_est_code == "RF_shallow":
        return RandomForestClassifier(n_estimators=10, max_depth=4, class_weight="balanced", random_state=random_state, n_jobs=1)
    return DecisionTreeClassifier(max_depth=4, random_state=random_state)


def run_comprehensive_experiment(
    seeds: List[int] = DEFAULT_SEEDS,
    output_csv: Optional[str] = None
) -> pd.DataFrame:
    """
    Menjalankan eksplorasi seluruh kombinasi pipeline across FlakeFlagger dan IDoFT.
    """
    base_dir = Path(__file__).resolve().parent.parent
    if output_csv is None:
        output_csv = str(base_dir / "results" / "raw" / "comprehensive_all_pipelines_predictions.csv")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    results: List[Dict[str, Any]] = []

    # Load existing cached results if available
    existing_keys = set()
    if os.path.exists(output_csv):
        try:
            prev_df = pd.read_csv(output_csv)
            results = prev_df.to_dict('records')
            existing_keys = set(f"{r['dataset']}_{r['seed']}_{r['project']}_{r['treatment']}" for r in results)
            print(f" Memuat {len(results)} evaluasi dari cache: {output_csv}")
        except Exception:
            pass

    # Load base 6 treatments from previous raw file if not in cache
    prev_comb_csv = str(base_dir / "results" / "raw" / "all_datasets_combined_predictions.csv")
    if os.path.exists(prev_comb_csv) and len(results) == 0:
        try:
            df_old = pd.read_csv(prev_comb_csv)
            for rec in df_old.to_dict('records'):
                t_name = rec['treatment']
                grp = COMPREHENSIVE_TREATMENTS.get(t_name, {}).get("group", "Sampling_Progression")
                rec['pipeline_group'] = grp
                results.append(rec)
            existing_keys = set(f"{r['dataset']}_{r['seed']}_{r['project']}_{r['treatment']}" for r in results)
            print(f" Berhasil menyerap {len(results)} evaluasi dari baseline cache sebelumnya.")
        except Exception as e:
            print(f" Warning: Gagal memuat baseline cache: {e}")

    # Datasets to evaluate
    datasets_to_run = ["flakeflagger", "idoft"]

    total_tasks = 0
    for dname in datasets_to_run:
        df_demo = load_dataset(dname)
        n_proj = df_demo['project_clean'].nunique()
        for t_name, cfg in COMPREHENSIVE_TREATMENTS.items():
            if cfg.get("dataset_override") and cfg["dataset_override"] != dname:
                continue
            total_tasks += len(seeds) * n_proj

    print("\n" + "=" * 78)
    print("  MEMULAI EVALUASI SELURUH KOMBINASI PIPELINE TRADABOOST & BASELINE")
    print(f"  Total Perlakuan: {len(COMPREHENSIVE_TREATMENTS)}")
    print(f"  Estimasi Total Evaluasi: {total_tasks} folds")
    print("=" * 78)

    t_start = time.time()
    counter = len(results)

    for dname in datasets_to_run:
        df_default = load_dataset(dname)
        df_rowdrop = load_dataset("flakeflagger_rowdrop") if dname == "flakeflagger" else None

        projects = sorted(df_default['project_clean'].unique().tolist())

        for seed in seeds:
            for p_idx, target_project in enumerate(projects):
                for treatment_name, cfg in COMPREHENSIVE_TREATMENTS.items():
                    # Handle dataset override (e.g. Row-Drop)
                    ds_override = cfg.get("dataset_override")
                    if ds_override and ds_override.startswith("flakeflagger") and dname != "flakeflagger":
                        continue

                    key = f"{dname}_{seed}_{target_project}_{treatment_name}"
                    if key in existing_keys:
                        continue

                    # Choose dataframe
                    curr_df = df_rowdrop if ds_override == "flakeflagger_rowdrop" else df_default

                    # Check if project exists in curr_df (e.g. RowDrop)
                    if target_project not in curr_df['project_clean'].values:
                        continue

                    t0 = time.time()
                    X_src_raw, y_src, X_tgt_raw, y_tgt = get_project_splits(curr_df, target_project)

                    n_tgt_flaky = int(np.sum(y_tgt == 1))
                    n_tgt_total = len(y_tgt)
                    t_type = cfg["type"]
                    ratio = cfg["target_ratio"]
                    scaler_code = cfg["scaler"]
                    resampler_code = cfg["resampler"]

                    y_eval_true = y_tgt
                    y_eval_probs = np.zeros(len(y_tgt), dtype=float)
                    y_eval_preds = np.zeros(len(y_tgt), dtype=int)

                    # ----------------------------------------------------
                    # 1. BASELINE ZERO-SHOT (0% Target)
                    # ----------------------------------------------------
                    if t_type == "baseline":
                        scaler = get_scaler(scaler_code)
                        X_src_sc = scaler.fit_transform(X_src_raw)
                        X_tgt_sc = scaler.transform(X_tgt_raw)

                        X_src_res, y_src_res = apply_resampling(X_src_sc, y_src, resampler_code, random_state=seed)

                        clf = build_classifier(cfg["model"], random_state=seed)
                        try:
                            clf.fit(X_src_res, y_src_res)
                            if hasattr(clf, "predict_proba"):
                                probs = clf.predict_proba(X_tgt_sc)
                                y_eval_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
                            else:
                                y_eval_probs = clf.predict(X_tgt_sc).astype(float)
                            y_eval_preds = (y_eval_probs >= 0.5).astype(int)
                        except Exception:
                            y_eval_preds = np.zeros(len(y_tgt), dtype=int)
                            y_eval_probs = np.zeros(len(y_tgt), dtype=float)

                    # ----------------------------------------------------
                    # 2. TRADABOOST TRANSFER LEARNING
                    # ----------------------------------------------------
                    elif t_type == "tradaboost":
                        X_t_train, y_t_train, X_t_test, y_t_test = split_target_data(
                            X_tgt_raw, y_tgt, ratio, random_state=seed
                        )

                        # Zero Leakage: fit scaler pada source + target train saja
                        scaler = get_scaler(scaler_code)
                        X_src_sc = scaler.fit_transform(X_src_raw)
                        X_t_train_sc = scaler.transform(X_t_train)
                        X_t_test_sc = scaler.transform(X_t_test)

                        # Resampling pada source
                        X_src_res, y_src_res = apply_resampling(X_src_sc, y_src, resampler_code, random_state=seed)

                        base_est = get_tradaboost_base_estimator(cfg["base_est"], random_state=seed)
                        trada = TrAdaBoostClassifier(
                            base_estimator=base_est,
                            n_estimators=cfg.get("n_est", 20),
                            random_state=seed,
                            use_second_half=cfg.get("second_half", True)
                        )

                        try:
                            trada.fit(X_src_res, y_src_res, X_t_train_sc, y_t_train)
                            probs = trada.predict_proba(X_t_test_sc)
                            y_eval_probs = probs[:, 1]
                            y_eval_preds = (y_eval_probs >= 0.5).astype(int)
                            y_eval_true = y_t_test
                        except Exception:
                            y_eval_true = y_t_test
                            y_eval_preds = np.zeros(len(y_t_test), dtype=int)
                            y_eval_probs = np.zeros(len(y_t_test), dtype=float)

                    # ----------------------------------------------------
                    # 3. WITHIN-PROJECT (Upper Bound)
                    # ----------------------------------------------------
                    elif t_type == "within":
                        y_eval_true = y_tgt
                        y_eval_preds = np.zeros(len(y_tgt), dtype=int)
                        y_eval_probs = np.zeros(len(y_tgt), dtype=float)

                        if len(np.unique(y_tgt)) > 1 and n_tgt_flaky >= 2 and n_tgt_total >= 5:
                            n_splits = min(5, n_tgt_flaky)
                            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                            for tr_idx, te_idx in skf.split(X_tgt_raw, y_tgt):
                                scaler_w = get_scaler(scaler_code)
                                X_tr_sc = scaler_w.fit_transform(X_tgt_raw[tr_idx])
                                X_te_sc = scaler_w.transform(X_tgt_raw[te_idx])

                                clf_w = build_classifier(cfg["model"], random_state=seed)
                                clf_w.fit(X_tr_sc, y_tgt[tr_idx])
                                probs_w = clf_w.predict_proba(X_te_sc)
                                y_eval_probs[te_idx] = probs_w[:, 1]
                                y_eval_preds[te_idx] = (probs_w[:, 1] >= 0.5).astype(int)

                    # Hitung metrik
                    metrics = evaluate_predictions(y_eval_true, y_eval_probs, y_eval_preds)
                    elapsed = time.time() - t0

                    rec = {
                        "dataset": dname,
                        "seed": seed,
                        "project": target_project,
                        "treatment": treatment_name,
                        "pipeline_group": cfg.get("group", "General"),
                        "target_train_ratio": ratio,
                        "n_eval_samples": len(y_eval_true),
                        "n_eval_flaky": int(np.sum(y_eval_true == 1)),
                        "total_project_tests": n_tgt_total,
                        "total_project_flaky": n_tgt_flaky,
                        "flakiness_rate_pct": round((n_tgt_flaky / n_tgt_total) * 100.0, 2) if n_tgt_total > 0 else 0.0,
                        **metrics,
                        "elapsed_sec": round(elapsed, 3)
                    }

                    results.append(rec)
                    existing_keys.add(key)
                    counter += 1

                    if counter % 50 == 0:
                        print(f" Progress: {counter} evaluasi selesai | Proj: {target_project:<14} | {treatment_name:<30} | F1: {metrics['f1_score']:.4f}")

    df_all = pd.DataFrame(results)
    df_all.to_csv(output_csv, index=False)
    print(f"\n [SUCCESS] Seluruh hasil komprehensif ({len(df_all)} evaluasi) berhasil disimpan ke: {output_csv}")
    print(f" Waktu eksekusi total: {time.time() - t_start:.2f} detik.")
    return df_all


if __name__ == "__main__":
    run_comprehensive_experiment()
