"""
Runner Eksperimen Skema 1: One-on-One Pairwise Cross-Project Transfer Matrix.
Mengevaluasi transferabilitas prediktif dari 1 proyek sumber tunggal (P_source) ke 1 proyek target tunggal (P_target)
pada seluruh 156 kombinasi pasangan proyek aktif di IDoFT (13 proyek).

Rujukan: Afeltra et al. (IEEE Access 2024).
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

def get_metrics_dict(y_true: np.ndarray, y_probs: np.ndarray, y_preds: np.ndarray) -> Dict[str, float]:
    """Menghitung metrik klasifikasi lengkap."""
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
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc,
        "F1": f1,
        "Recall": rec,
        "Precision": prec,
        "MCC": mcc,
        "Balanced_Accuracy": bal_acc,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)
    }

def run_pairwise_one_on_one_experiment(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    active_projects: List[str],
    model_type: str = "M1",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Mengeksekusi One-on-One Pairwise Cross-Project Transfer Matrix.
    Untuk setiap source project P_src dan target project P_tgt (P_src != P_tgt):
    - Melatih Baseline (Direct 1-on-1 Transfer)
    - Melatih Burak Filter (1-on-1 k-NN Filtered Transfer)
    - Melatih TrAdaBoost (1-on-1 Transfer Boosting dengan adaptasi target)
    """
    print("=" * 80)
    print("🚀 MEMULAI EKSEKUSI SKEMA 1: ONE-ON-ONE PAIRWISE CROSS-PROJECT TRANSFER")
    print(f"   Model Classifier      : {model_type} (RandomForest/XGBoost)")
    print(f"   Jumlah Proyek Aktif   : {len(active_projects)} proyek")
    print(f"   Total Pasangan (1-on-1): {len(active_projects) * (len(active_projects) - 1)} pasangan")
    print("=" * 80)

    results = []
    total_pairs = len(active_projects) * (len(active_projects) - 1)
    pair_count = 0
    t0 = time.time()

    for p_src in active_projects:
        df_src = df[df["project_clean"] == p_src]
        X_src_raw = df_src[feature_cols].values
        y_src = df_src[target_col].values

        if np.sum(y_src == 1) == 0:
            continue

        for p_tgt in active_projects:
            if p_src == p_tgt:
                continue

            pair_count += 1
            df_tgt = df[df["project_clean"] == p_tgt]
            X_tgt_raw = df_tgt[feature_cols].values
            y_tgt = df_tgt[target_col].values

            # 1. Penskalaan Fitur (Fit STRICTLY pada Source)
            scaler = get_scaler("S2")  # MinMaxScaler
            X_src_sc = scaler.fit_transform(X_src_raw)
            X_tgt_sc = scaler.transform(X_tgt_raw)

            # --- A. BASELINE ONE-ON-ONE ---
            X_tr_base, y_tr_base = apply_resampling(X_src_sc, y_src, strategy="RUS", random_state=random_state)
            clf_base = build_classifier(model_type, random_state=random_state)
            clf_base.fit(X_tr_base, y_tr_base)

            probs_base = clf_base.predict_proba(X_tgt_sc)[:, 1] if hasattr(clf_base, "predict_proba") else clf_base.predict(X_tgt_sc)
            preds_base = clf_base.predict(X_tgt_sc)
            m_base = get_metrics_dict(y_tgt, probs_base, preds_base)

            results.append({
                "Source_Project": p_src,
                "Target_Project": p_tgt,
                "Pair_Name": f"{p_src}->{p_tgt}",
                "Treatment": "Baseline_1on1",
                "Model": model_type,
                **m_base
            })

            # --- B. BURAK FILTER ONE-ON-ONE ---
            X_src_burak, y_src_burak, _ = apply_burak_filter(X_src_sc, y_src, X_tgt_sc, k=10)
            X_tr_burak, y_tr_burak = apply_resampling(X_src_burak, y_src_burak, strategy="RUS", random_state=random_state)
            clf_burak = build_classifier(model_type, random_state=random_state)

            if len(np.unique(y_tr_burak)) > 1:
                clf_burak.fit(X_tr_burak, y_tr_burak)
                probs_burak = clf_burak.predict_proba(X_tgt_sc)[:, 1] if hasattr(clf_burak, "predict_proba") else clf_burak.predict(X_tgt_sc)
                preds_burak = clf_burak.predict(X_tgt_sc)
            else:
                probs_burak = probs_base
                preds_burak = preds_base

            m_burak = get_metrics_dict(y_tgt, probs_burak, preds_burak)
            results.append({
                "Source_Project": p_src,
                "Target_Project": p_tgt,
                "Pair_Name": f"{p_src}->{p_tgt}",
                "Treatment": "Burak_Filter_1on1",
                "Model": model_type,
                **m_burak
            })

            # --- C. TRADABOOST ONE-ON-ONE (50% target adaptation, 50% test) ---
            split_idx = int(len(X_tgt_sc) * 0.5)
            if split_idx > 0 and len(np.unique(y_tgt[:split_idx])) > 1:
                X_tgt_train = X_tgt_sc[:split_idx]
                y_tgt_train = y_tgt[:split_idx]
                X_tgt_test = X_tgt_sc[split_idx:]
                y_tgt_test = y_tgt[split_idx:]

                trada = TrAdaBoostClassifier(n_estimators=20, random_state=random_state)
                trada.fit(X_src_sc, y_src, X_tgt_train, y_tgt_train)
                probs_trada = trada.predict_proba(X_tgt_test)[:, 1]
                preds_trada = trada.predict(X_tgt_test)
                m_trada = get_metrics_dict(y_tgt_test, probs_trada, preds_trada)
            else:
                probs_trada = probs_base
                preds_trada = preds_base
                m_trada = m_base

            results.append({
                "Source_Project": p_src,
                "Target_Project": p_tgt,
                "Pair_Name": f"{p_src}->{p_tgt}",
                "Treatment": "TrAdaBoost_50_1on1",
                "Model": model_type,
                **m_trada
            })

            if pair_count % 25 == 0 or pair_count == total_pairs:
                print(f"   [Progress] {pair_count}/{total_pairs} pasangan tuntas ({time.time() - t0:.2f}s)...")

    elapsed = time.time() - t0
    print(f"[OK] Selesai! {len(results)} evaluasi One-on-One tuntas dalam {elapsed:.2f} detik.")
    return pd.DataFrame(results)

if __name__ == "__main__":
    df, feat_cols, tgt = load_idoft_dataset()
    active, _ = get_active_and_zero_flaky_projects(df)
    df_res = run_pairwise_one_on_one_experiment(df, feat_cols, tgt, active, model_type="M1", random_state=42)
    print("Contoh hasil 5 baris pertama:")
    print(df_res[["Source_Project", "Target_Project", "Treatment", "F1", "Recall", "Precision", "PR_AUC"]].head())
