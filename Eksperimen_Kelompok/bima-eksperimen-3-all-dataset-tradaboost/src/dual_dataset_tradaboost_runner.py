"""
Runner Eksekusi Multi-Dataset Repeated LOPO-CV dengan TrAdaBoost Target Adaptation.
Mengevaluasi FlakeFlagger (24 proyek) dan IDoFT (17 proyek) lintas 5 seeds:
- T1: Baseline Zero-Shot (0% Target)
- T2: TrAdaBoost 25% Target Data
- T3: TrAdaBoost 50% Target Data
- T4: TrAdaBoost 75% Target Data
- T5: Within-Project 5-Fold Stratified CV (Upper Bound)
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_dataset, get_project_splits, PREDICTIVE_FEATURES
from pipeline_builder import get_scaler, apply_resampling, build_classifier, evaluate_predictions
from tradaboost import TrAdaBoostClassifier, split_target_data

DEFAULT_SEEDS = [42, 100, 2024, 777, 999]
TREATMENT_NAMES = [
    "Baseline_Zero_Shot_RF",
    "Baseline_Zero_Shot_XGB",
    "TrAdaBoost_25",
    "TrAdaBoost_50",
    "TrAdaBoost_75",
    "Within_Project"
]


def run_single_dataset(
    dataset_name: str,
    seeds: List[int] = DEFAULT_SEEDS,
    output_csv: Optional[str] = None
) -> pd.DataFrame:
    """
    Menjalankan eksperimen Repeated LOPO-CV dengan variasi TrAdaBoost pada satu dataset.
    """
    df = load_dataset(dataset_name)
    projects = sorted(df['project_clean'].unique().tolist())
    total_evals = len(seeds) * len(projects) * len(TREATMENT_NAMES)

    print(f"\n=======================================================")
    print(f" Memulai Eksekusi: Dataset {dataset_name.upper()}")
    print(f" Proyek: {len(projects)}, Seeds: {len(seeds)}, Treatments: {len(TREATMENT_NAMES)}")
    print(f" Total Evaluasi: {total_evals}")
    print(f"=======================================================")

    results: List[Dict[str, Any]] = []

    # Cek apakah sudah ada hasil sebagian (resume mode)
    if output_csv and os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            if len(existing_df) > 0:
                results = existing_df.to_dict('records')
                print(f" Berhasil me-load {len(results)} evaluasi dari cache sebelumnya.")
        except Exception:
            pass

    completed_keys = set(
        f"{r['dataset']}_{r['seed']}_{r['project']}_{r['treatment']}"
        for r in results
    )

    t_start = time.time()
    counter = len(results)

    for seed in seeds:
        for p_idx, target_project in enumerate(projects):
            X_source_raw, y_source, X_target_raw, y_target = get_project_splits(df, target_project)

            n_target_flaky = int(np.sum(y_target == 1))
            n_target_total = len(y_target)

            for treatment in TREATMENT_NAMES:
                key = f"{dataset_name}_{seed}_{target_project}_{treatment}"
                if key in completed_keys:
                    continue

                t0 = time.time()

                # Inisialisasi variabel prediksi
                y_eval_true = y_target
                y_eval_probs = np.zeros(len(y_target), dtype=np.float64)
                y_eval_preds = np.zeros(len(y_target), dtype=int)
                target_train_ratio = 0.0

                # ----------------------------------------------------
                # 1. BASELINE ZERO-SHOT (0% Target Data)
                # ----------------------------------------------------
                if treatment.startswith("Baseline_Zero_Shot"):
                    # Zero Data Leakage: fit scaler hanya pada source
                    scaler = get_scaler("MinMaxScaler")
                    X_src_scaled = scaler.fit_transform(X_source_raw)
                    X_tgt_scaled = scaler.transform(X_target_raw)

                    # Resampling pada source
                    X_src_res, y_src_res = apply_resampling(X_src_scaled, y_source, "RUS", random_state=seed)

                    clf_type = "M2" if "XGB" in treatment else "M1"
                    clf = build_classifier(clf_type, random_state=seed)

                    try:
                        clf.fit(X_src_res, y_src_res)
                        if hasattr(clf, "predict_proba"):
                            probs = clf.predict_proba(X_tgt_scaled)
                            y_eval_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
                        else:
                            y_eval_probs = clf.predict(X_tgt_scaled).astype(float)
                        y_eval_preds = (y_eval_probs >= 0.5).astype(int)
                    except Exception as e:
                        y_eval_preds = np.zeros(len(y_target), dtype=int)
                        y_eval_probs = np.zeros(len(y_target), dtype=float)

                # ----------------------------------------------------
                # 2. TRADABOOST (25%, 50%, 75% Target Adaptation)
                # ----------------------------------------------------
                elif treatment.startswith("TrAdaBoost"):
                    if "25" in treatment:
                        target_train_ratio = 0.25
                    elif "50" in treatment:
                        target_train_ratio = 0.50
                    elif "75" in treatment:
                        target_train_ratio = 0.75

                    X_t_train, y_t_train, X_t_test, y_t_test = split_target_data(
                        X_target_raw, y_target, target_train_ratio, random_state=seed
                    )

                    # Zero Leakage: Fit scaler pada source + target train saja
                    scaler = get_scaler("MinMaxScaler")
                    X_src_scaled = scaler.fit_transform(X_source_raw)
                    X_t_train_scaled = scaler.transform(X_t_train)
                    X_t_test_scaled = scaler.transform(X_t_test)

                    # Resampling RUS pada source untuk menyeimbangkan source
                    X_src_res, y_src_res = apply_resampling(X_src_scaled, y_source, "RUS", random_state=seed)

                    base_weak = DecisionTreeClassifier(max_depth=4, random_state=seed)
                    trada = TrAdaBoostClassifier(
                        base_estimator=base_weak,
                        n_estimators=20,
                        random_state=seed,
                        use_second_half=True
                    )

                    try:
                        trada.fit(X_src_res, y_src_res, X_t_train_scaled, y_t_train)
                        probs = trada.predict_proba(X_t_test_scaled)
                        y_eval_probs = probs[:, 1]
                        y_eval_preds = (y_eval_probs >= 0.5).astype(int)
                        y_eval_true = y_t_test
                    except Exception as e:
                        y_eval_true = y_t_test
                        y_eval_preds = np.zeros(len(y_t_test), dtype=int)
                        y_eval_probs = np.zeros(len(y_t_test), dtype=float)

                # ----------------------------------------------------
                # 3. WITHIN-PROJECT (Upper Bound 5-Fold Stratified CV)
                # ----------------------------------------------------
                elif treatment == "Within_Project":
                    target_train_ratio = 1.0
                    y_eval_true = y_target
                    y_eval_preds = np.zeros(len(y_target), dtype=int)
                    y_eval_probs = np.zeros(len(y_target), dtype=float)

                    if len(np.unique(y_target)) > 1 and n_target_flaky >= 2 and n_target_total >= 5:
                        n_splits = min(5, n_target_flaky)
                        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                        for tr_idx, te_idx in skf.split(X_target_raw, y_target):
                            scaler_w = get_scaler("MinMaxScaler")
                            X_tr_sc = scaler_w.fit_transform(X_target_raw[tr_idx])
                            X_te_sc = scaler_w.transform(X_target_raw[te_idx])

                            clf_w = build_classifier("M1", random_state=seed)
                            clf_w.fit(X_tr_sc, y_target[tr_idx])
                            probs_w = clf_w.predict_proba(X_te_sc)
                            y_eval_probs[te_idx] = probs_w[:, 1]
                            y_eval_preds[te_idx] = (probs_w[:, 1] >= 0.5).astype(int)
                    else:
                        # Fallback jika flaky < 2
                        y_eval_preds = np.zeros(len(y_target), dtype=int)
                        y_eval_probs = np.zeros(len(y_target), dtype=float)

                # Evaluasi metrik fold
                metrics = evaluate_predictions(y_eval_true, y_eval_probs, y_eval_preds)
                elapsed = time.time() - t0

                res_record = {
                    "dataset": dataset_name,
                    "seed": seed,
                    "project": target_project,
                    "treatment": treatment,
                    "target_train_ratio": target_train_ratio,
                    "n_eval_samples": len(y_eval_true),
                    "n_eval_flaky": int(np.sum(y_eval_true == 1)),
                    "total_project_tests": n_target_total,
                    "total_project_flaky": n_target_flaky,
                    "flakiness_rate_pct": round((n_target_flaky / n_target_total) * 100.0, 2) if n_target_total > 0 else 0.0,
                    **metrics,
                    "elapsed_sec": round(elapsed, 3)
                }

                results.append(res_record)
                counter += 1

                if counter % 50 == 0 or counter == total_evals:
                    print(f" [{dataset_name.upper()}] Selesai {counter}/{total_evals} evaluasi ({(counter/total_evals)*100:.1f}%) | "
                          f"Proj: {target_project:<16} | {treatment:<24} | F1: {metrics['f1_score']:.4f} | Rec: {metrics['recall_score']:.4f}")

    df_res = pd.DataFrame(results)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_res.to_csv(output_csv, index=False)
        print(f" [SAVED] Hasil {dataset_name} disimpan ke: {output_csv}")

    total_time = time.time() - t_start
    print(f" Selesai {dataset_name.upper()} dalam {total_time:.2f} detik.")
    return df_res


def run_all_datasets(
    seeds: List[int] = DEFAULT_SEEDS,
    base_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Menjalankan eksekusi lengkap pada FlakeFlagger dan IDoFT secara berurutan.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent

    raw_dir = base_dir / "results" / "raw"
    os.makedirs(raw_dir, exist_ok=True)

    ff_csv = str(raw_dir / "flakeflagger_repeated_folds_predictions.csv")
    id_csv = str(raw_dir / "idoft_repeated_folds_predictions.csv")
    comb_csv = str(raw_dir / "all_datasets_combined_predictions.csv")

    df_ff = run_single_dataset("flakeflagger", seeds=seeds, output_csv=ff_csv)
    df_id = run_single_dataset("idoft", seeds=seeds, output_csv=id_csv)

    df_combined = pd.concat([df_ff, df_id], ignore_index=True)
    df_combined.to_csv(comb_csv, index=False)
    print(f"\n [SUCCESS] Seluruh hasil gabungan ({len(df_combined)} baris) disimpan ke: {comb_csv}")

    return df_ff, df_id, df_combined


if __name__ == "__main__":
    run_all_datasets()
