"""
Runner Eksperimen Skema 2: Repeated Leave-One-Project-Out Cross-Validation (Repeated LOPO-CV)
menggunakan Konsep Perlakuan Paper Afeltra et al. (IEEE Access 2024).

Perlakuan yang dievaluasi lintas 5 random seeds ([42, 100, 2024, 777, 999]) pada 17 proyek:
1. Baseline: Model Cross-Project Standar (MinMaxScaler + RUS + RF/XGB)
2. Burak_Filter: k-NN Instance Selection (k=10)
3. TrAdaBoost_25: Transfer Learning 25% Target Adaptation
4. TrAdaBoost_50: Transfer Learning 50% Target Adaptation
5. TrAdaBoost_75: Transfer Learning 75% Target Adaptation (Best Afeltra 2024)
6. Within_Project: Benchmark Batas Atas Ideal (5-Fold Stratified CV)
"""

import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_idoft_dataset, get_active_and_zero_flaky_projects
from burak_filter import apply_burak_filter
from tradaboost import TrAdaBoostClassifier
from pipeline_builder import get_scaler, apply_resampling, build_classifier

DEFAULT_SEEDS = [42, 100, 2024, 777, 999]

def calculate_metrics(y_true: np.ndarray, y_probs: np.ndarray, y_preds: np.ndarray) -> Dict[str, float]:
    """Menghitung metrik performa klasifikasi lengkap dengan pengaman zero-flaky."""
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    tn, fp, fn, tp = 0, 0, 0, 0

    if len(y_true) > 0:
        cm = confusion_matrix(y_true, y_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

    if n_pos == 0:
        return {
            "Test_Size": len(y_true),
            "Flaky_in_Test": 0,
            "NonFlaky_in_Test": n_neg,
            "PR_AUC": np.nan,
            "ROC_AUC": np.nan,
            "F1": 0.0,
            "Recall": np.nan,
            "Precision": 0.0 if fp > 0 else 1.0,
            "MCC": 0.0,
            "Balanced_Accuracy": tn / (tn + fp) if (tn + fp) > 0 else 1.0,
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)
        }

    try:
        pr_auc = float(average_precision_score(y_true, y_probs))
    except Exception:
        pr_auc = np.nan

    try:
        roc_auc = float(roc_auc_score(y_true, y_probs))
    except Exception:
        roc_auc = np.nan

    prec = float(precision_score(y_true, y_preds, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_preds, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_preds, pos_label=1, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_preds))
    bal_acc = float(balanced_accuracy_score(y_true, y_preds))

    return {
        "Test_Size": len(y_true),
        "Flaky_in_Test": n_pos,
        "NonFlaky_in_Test": n_neg,
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc,
        "F1": f1,
        "Recall": rec,
        "Precision": prec,
        "MCC": mcc,
        "Balanced_Accuracy": bal_acc,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)
    }

def run_afeltra_repeated_lopo_experiment(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seeds: List[int] = DEFAULT_SEEDS,
    models: List[str] = ["M1", "M2"]
) -> pd.DataFrame:
    """
    Mengeksekusi Repeated LOPO-CV untuk seluruh perlakuan paper Afeltra et al. (2024).
    """
    projects = sorted(df["project_clean"].unique().tolist())
    print("=" * 80)
    print("🚀 MEMULAI EKSEKUSI SKEMA 2: REPEATED LOPO-CV AFELTRA CONCEPT TREATMENTS")
    print(f"   Jumlah Proyek Target  : {len(projects)} proyek (17)")
    print(f"   Random Seeds          : {seeds} ({len(seeds)} repetisi)")
    print(f"   Model Classifiers     : {models} (M1: RF, M2: XGBoost GPU)")
    print(f"   Intervensi Perlakuan  : Baseline, Burak_Filter, TrAdaBoost_25, TrAdaBoost_50, TrAdaBoost_75, Within_Project")
    print("=" * 80)

    all_results = []
    t0 = time.time()
    total_evals = len(seeds) * len(models) * len(projects)
    counter = 0

    for seed in seeds:
        print(f"\n>>> MENJALANKAN REPETISI SEED: {seed} <<<")
        for model_type in models:
            for p_test in projects:
                counter += 1
                df_train = df[df["project_clean"] != p_test]
                df_test = df[df["project_clean"] == p_test]

                X_train_raw = df_train[feature_cols].values
                y_train = df_train[target_col].values
                X_test_raw = df_test[feature_cols].values
                y_test = df_test[target_col].values

                # 1. Penskalaan Fitur (Fit HANYA pada data latih)
                scaler = get_scaler("S2")  # MinMaxScaler
                X_train_sc = scaler.fit_transform(X_train_raw)
                X_test_sc = scaler.transform(X_test_raw)

                # --- 1. BASELINE CROSS-PROJECT (RUS + Model) ---
                X_tr_base, y_tr_base = apply_resampling(X_train_sc, y_train, strategy="RUS", random_state=seed)
                clf_base = build_classifier(model_type, random_state=seed)
                clf_base.fit(X_tr_base, y_tr_base)

                probs_base = clf_base.predict_proba(X_test_sc)[:, 1] if hasattr(clf_base, "predict_proba") else clf_base.predict(X_test_sc)
                preds_base = clf_base.predict(X_test_sc)
                m_base = calculate_metrics(y_test, probs_base, preds_base)

                all_results.append({
                    "Seed": seed,
                    "Project": p_test,
                    "Model": model_type,
                    "Treatment": "T1_Baseline",
                    **m_base
                })

                # --- 2. BURAK FILTER (k=10 Nearest Neighbors) ---
                X_tr_burak_sub, y_tr_burak_sub, _ = apply_burak_filter(X_train_sc, y_train, X_test_sc, k=10)
                X_tr_burak, y_tr_burak = apply_resampling(X_tr_burak_sub, y_tr_burak_sub, strategy="RUS", random_state=seed)
                clf_burak = build_classifier(model_type, random_state=seed)

                if len(np.unique(y_tr_burak)) > 1:
                    clf_burak.fit(X_tr_burak, y_tr_burak)
                    probs_burak = clf_burak.predict_proba(X_test_sc)[:, 1] if hasattr(clf_burak, "predict_proba") else clf_burak.predict(X_test_sc)
                    preds_burak = clf_burak.predict(X_test_sc)
                else:
                    probs_burak = probs_base
                    preds_burak = preds_base

                m_burak = calculate_metrics(y_test, probs_burak, preds_burak)
                all_results.append({
                    "Seed": seed,
                    "Project": p_test,
                    "Model": model_type,
                    "Treatment": "T2_Burak_Filter",
                    **m_burak
                })

                # --- 3, 4, 5. TRADABOOST VARIANT (25%, 50%, 75% Adaptation Splits) ---
                for tr_ratio, t_name in [(0.25, "T3_TrAdaBoost_25"), (0.50, "T4_TrAdaBoost_50"), (0.75, "T5_TrAdaBoost_75")]:
                    split_idx = int(len(X_test_sc) * tr_ratio)
                    # Subsample data sumber agar efisiensi terjaga (max 3000 instances)
                    if len(X_train_sc) > 3000:
                        np.random.seed(seed)
                        sub_idx = np.random.choice(len(X_train_sc), 3000, replace=False)
                        X_s_sub, y_s_sub = X_train_sc[sub_idx], y_train[sub_idx]
                    else:
                        X_s_sub, y_s_sub = X_train_sc, y_train

                    if split_idx > 0 and len(np.unique(y_test[:split_idx])) > 1:
                        X_t_train, y_t_train = X_test_sc[:split_idx], y_test[:split_idx]
                        X_t_test, y_t_test = X_test_sc[split_idx:], y_test[split_idx:]

                        trada = TrAdaBoostClassifier(n_estimators=20, random_state=seed)
                        trada.fit(X_s_sub, y_s_sub, X_t_train, y_t_train)
                        probs_trada = trada.predict_proba(X_t_test)[:, 1]
                        preds_trada = trada.predict(X_t_test)
                        m_trada = calculate_metrics(y_t_test, probs_trada, preds_trada)
                    else:
                        # Fallback jika target split tidak memiliki 2 kelas
                        probs_trada = probs_base
                        preds_trada = preds_base
                        m_trada = m_base

                    all_results.append({
                        "Seed": seed,
                        "Project": p_test,
                        "Model": model_type,
                        "Treatment": t_name,
                        **m_trada
                    })

                # --- 6. WITHIN-PROJECT UPPER BOUND (5-Fold Stratified CV pada Target) ---
                y_pred_within = np.zeros_like(y_test)
                y_prob_within = np.zeros(len(y_test), dtype=np.float64)

                n_pos_test = int(np.sum(y_test == 1))
                if n_pos_test >= 2 and len(y_test) >= 5:
                    n_splits = min(5, n_pos_test)
                    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                    for tr_idx, te_idx in skf.split(X_test_raw, y_test):
                        X_w_tr_raw, X_w_te_raw = X_test_raw[tr_idx], X_test_raw[te_idx]
                        y_w_tr = y_test[tr_idx]

                        sc_w = get_scaler("S2")
                        X_w_tr_sc = sc_w.fit_transform(X_w_tr_raw)
                        X_w_te_sc = sc_w.transform(X_w_te_raw)

                        X_w_tr_res, y_w_tr_res = apply_resampling(X_w_tr_sc, y_w_tr, strategy="RUS", random_state=seed)
                        clf_w = build_classifier(model_type, random_state=seed)

                        if len(np.unique(y_w_tr_res)) > 1:
                            clf_w.fit(X_w_tr_res, y_w_tr_res)
                            y_prob_within[te_idx] = clf_w.predict_proba(X_w_te_sc)[:, 1] if hasattr(clf_w, "predict_proba") else clf_w.predict(X_w_te_sc)
                            y_pred_within[te_idx] = clf_w.predict(X_w_te_sc)
                        else:
                            y_prob_within[te_idx] = float(y_w_tr[0])
                            y_pred_within[te_idx] = y_w_tr[0]
                else:
                    y_prob_within = np.zeros(len(y_test))
                    y_pred_within = np.zeros_like(y_test)

                m_within = calculate_metrics(y_test, y_prob_within, y_pred_within)
                all_results.append({
                    "Seed": seed,
                    "Project": p_test,
                    "Model": model_type,
                    "Treatment": "T6_Within_Project",
                    **m_within
                })

            print(f"   [OK] Model {model_type} pada Seed {seed} tuntas. ({counter}/{total_evals} folds)")

    elapsed = time.time() - t0
    print(f"\n[SELESAI] Total {len(all_results)} evaluasi LOPO-CV tuntas dalam {elapsed:.2f} detik ({elapsed/60:.2f} menit).")
    return pd.DataFrame(all_results)

if __name__ == "__main__":
    df, feat_cols, tgt = load_idoft_dataset()
    df_res = run_afeltra_repeated_lopo_experiment(df, feat_cols, tgt, seeds=[42], models=["M1"])
    print(df_res.head())
