"""
Engine Eksekusi LOPO-CV untuk Eksperimen Alternatif 1:
Membandingkan 4 Konfigurasi Transfer Learning:
- Konfigurasi A: Baseline Cross-Project (XGBoost GPU)
- Konfigurasi B: Burak Filter (k-NN Instance Selection k=10)
- Konfigurasi C: TrAdaBoost (Adaptive Sample Reweighting)
- Konfigurasi D: Hybrid Burak Filter + TrAdaBoost (Top 30% Relevant + TrAdaBoost)

Hardware: GPU NVIDIA RTX 4050 Laptop (CUDA 13.4, Driver 617.42).
"""

import os
import sys
import time
import json
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    confusion_matrix,
)
import xgboost as xgb

from data_loader import load_dataset
from burak_filter import burak_filter_knn, burak_filter_ratio
from tradaboost import TrAdaBoostGPU


def evaluate_metrics(y_true: np.ndarray, y_probs: np.ndarray, y_preds: np.ndarray) -> Dict[str, Any]:
    """
    Menghitung metrik evaluasi lengkap dengan proteksi aman untuk kasus zero-positive (seperti jimfs).
    """
    n_flaky = int(np.sum(y_true == 1))
    n_non_flaky = int(np.sum(y_true == 0))
    tn, fp, fn, tp = 0, 0, 0, 0

    if len(y_true) > 0:
        cm = confusion_matrix(y_true, y_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

    if n_flaky == 0:
        pr_auc = np.nan
        roc_auc = np.nan
        rec = np.nan
        prec = 0.0 if fp > 0 else 1.0
        f1 = 0.0
        mcc = 0.0
        bal_acc = float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0
    else:
        pr_auc = float(average_precision_score(y_true, y_probs))
        try:
            roc_auc = float(roc_auc_score(y_true, y_probs))
        except ValueError:
            roc_auc = np.nan

        prec = float(precision_score(y_true, y_preds, pos_label=1, zero_division=0))
        rec = float(recall_score(y_true, y_preds, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_true, y_preds, pos_label=1, zero_division=0))
        mcc = float(matthews_corrcoef(y_true, y_preds))
        bal_acc = float(balanced_accuracy_score(y_true, y_preds))

    return {
        "Test_Size": len(y_true),
        "Flaky_in_Test": n_flaky,
        "NonFlaky_in_Test": n_non_flaky,
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "MCC": mcc,
        "Balanced_Accuracy": bal_acc,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def split_target_project(
    df_target: pd.DataFrame,
    target_train_ratio: float = 0.10,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Membagi data proyek target menjadi:
    - 10% data target training (few-shot labeled target data untuk TrAdaBoost)
    - 90% data target testing (unseen evaluation set untuk semua konfigurasi)
    Menjamin ketersediaan minimal 1 sampel flaky pada test set jika proyek memiliki flaky test.
    """
    n_flaky = int(df_target["flaky"].sum())

    if n_flaky == 0:
        df_tr, df_te = train_test_split(
            df_target,
            test_size=(1.0 - target_train_ratio),
            random_state=random_state,
            shuffle=True
        )
    elif n_flaky == 1:
        flaky_row = df_target[df_target["flaky"] == 1]
        non_flaky_rows = df_target[df_target["flaky"] == 0]
        nf_tr, nf_te = train_test_split(
            non_flaky_rows,
            test_size=(1.0 - target_train_ratio),
            random_state=random_state,
            shuffle=True
        )
        df_tr = nf_tr.copy()
        df_te = pd.concat([nf_te, flaky_row]).sample(frac=1.0, random_state=random_state).copy()
    else:
        df_tr, df_te = train_test_split(
            df_target,
            test_size=(1.0 - target_train_ratio),
            stratify=df_target["flaky"],
            random_state=random_state,
            shuffle=True
        )
        if df_te["flaky"].sum() == 0:
            pos_in_tr = df_tr[df_tr["flaky"] == 1].iloc[0:1]
            neg_in_te = df_te[df_te["flaky"] == 0].iloc[0:1]
            df_tr = pd.concat([df_tr.drop(pos_in_tr.index), neg_in_te])
            df_te = pd.concat([df_te.drop(neg_in_te.index), pos_in_tr])

    return df_tr, df_te


def resolve_default_output_dir() -> Path:
    """Mengembalikan direktori output standar pada folder Alternatif 1."""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent / "results"


def run_transfer_lopo_experiment(
    config_path: str = None,
    output_dir: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Menjalankan eksperimen LOPO-CV untuk 4 konfigurasi pada 24 proyek.
    """
    if output_dir is None:
        out_path = resolve_default_output_dir()
    else:
        out_path = Path(output_dir)

    raw_dir = out_path / "raw_predictions"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pemuatan Konfigurasi
    if config_path is None:
        script_dir = Path(__file__).resolve().parent
        cfg_file = script_dir.parent / "configs" / "transfer_config.json"
    else:
        cfg_file = Path(config_path)
    
    with open(cfg_file, "r") as f:
        cfg = json.load(f)

    k_neighbors = cfg["burak_filter_settings"].get("k_neighbors", 10)
    hybrid_ratio = cfg["burak_filter_settings"].get("hybrid_selection_ratio", 0.30)
    tradaboost_iters = cfg["tradaboost_settings"].get("n_estimators", 30)
    target_train_ratio = cfg["data_settings"].get("target_train_ratio", 0.10)
    seed = cfg["data_settings"].get("random_state", 42)
    use_gpu = cfg["hardware_settings"].get("use_gpu", True)

    # 2. Pemuatan Dataset C1
    df, features, projects = load_dataset(cfg["data_settings"]["cleaning_strategy"])

    print("=" * 80)
    print(f"🚀 MEMULAI EKSPERIMEN ALTERNATIF 1: TRANSFER LEARNING LOPO-CV (24 PROYEK)")
    print(f"Hardware: GPU RTX 4050 Laptop (CUDA 13.4, device='cuda')")
    print(f"Output Directory: {out_path.resolve()}")
    print(f"Konfigurasi Diuji:")
    print(f"  [A] Baseline Cross-Project (All N-1 Sources)")
    print(f"  [B] Burak Filter (k-NN Instance Selection, k={k_neighbors})")
    print(f"  [C] TrAdaBoost (N_iter={tradaboost_iters}, Few-Shot Target={int(target_train_ratio*100)}%)")
    print(f"  [D] Hybrid Burak Filter + TrAdaBoost (Top {int(hybrid_ratio*100)}% Sources + TrAdaBoost)")
    print("=" * 80)

    all_fold_records = []
    raw_pred_dfs = {
        "A_Baseline": [],
        "B_Burak_Filter": [],
        "C_TrAdaBoost": [],
        "D_Burak_TrAdaBoost_Hybrid": []
    }

    total_start = time.time()

    for idx, target_proj in enumerate(projects, 1):
        print(f"\n[{idx:02d}/24] Mengevaluasi Proyek Target: '{target_proj}' ...")

        # Partisi Data Sumber (23 Proyek) dan Data Target (1 Proyek)
        df_source = df[df["project"] != target_proj].copy()
        df_target = df[df["project"] == target_proj].copy()

        # Partisi Few-Shot Target: 10% Train (Few-Shot), 90% Test (Evaluasi Bersama)
        df_t_train, df_t_test = split_target_project(
            df_target,
            target_train_ratio=target_train_ratio,
            random_state=seed
        )

        X_s_raw = df_source[features].values.astype(float)
        y_s = df_source["flaky"].values.astype(int)

        X_tt_raw = df_t_train[features].values.astype(float)
        y_tt = df_t_train["flaky"].values.astype(int)

        X_te_raw = df_t_test[features].values.astype(float)
        y_te = df_t_test["flaky"].values.astype(int)

        # Standardisasi Fitur: Fit HANYA pada X_source (Zero Leakage!)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_s_raw)
        X_tt = scaler.transform(X_tt_raw)
        X_te = scaler.transform(X_te_raw)

        # Bobot kelas untuk menangani ketimpangan ekstrem
        pos_weight_s = float((len(y_s) - np.sum(y_s)) / max(1, np.sum(y_s)))

        # -------------------------------------------------------------
        # [A] Konfigurasi A: Baseline Cross-Project (XGBoost GPU)
        # -------------------------------------------------------------
        t0 = time.time()
        clf_base = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=pos_weight_s,
            tree_method="hist",
            device="cuda" if use_gpu else "cpu",
            eval_metric="logloss",
            random_state=seed
        )
        clf_base.fit(X_s, y_s)
        train_time_a = time.time() - t0

        t0_inf = time.time()
        probs_a = clf_base.predict_proba(X_te)[:, 1]
        inf_time_a = time.time() - t0_inf
        preds_a = (probs_a >= 0.5).astype(int)

        m_a = evaluate_metrics(y_te, probs_a, preds_a)
        m_a.update({
            "Config_ID": "A_Baseline",
            "Config_Name": "Baseline Cross-Project",
            "Target_Project": target_proj,
            "Train_Samples": len(X_s),
            "Train_Time_Sec": train_time_a,
            "Inference_Time_Sec": inf_time_a
        })
        all_fold_records.append(m_a)

        raw_pred_dfs["A_Baseline"].append(pd.DataFrame({
            "Target_Project": target_proj,
            "Test_Index": df_t_test.index,
            "True_Label": y_te,
            "Predicted_Prob": probs_a,
            "Predicted_Label": preds_a
        }))

        # -------------------------------------------------------------
        # [B] Konfigurasi B: Burak Filter (k-NN Instance Selection)
        # -------------------------------------------------------------
        t0 = time.time()
        X_s_burak, y_s_burak, burak_idx = burak_filter_knn(
            X_source=X_s,
            y_source=y_s,
            X_target=X_te,
            k=k_neighbors
        )
        pos_weight_burak = float((len(y_s_burak) - np.sum(y_s_burak)) / max(1, np.sum(y_s_burak)))

        clf_burak = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=pos_weight_burak,
            tree_method="hist",
            device="cuda" if use_gpu else "cpu",
            eval_metric="logloss",
            random_state=seed
        )
        clf_burak.fit(X_s_burak, y_s_burak)
        train_time_b = time.time() - t0

        t0_inf = time.time()
        probs_b = clf_burak.predict_proba(X_te)[:, 1]
        inf_time_b = time.time() - t0_inf
        preds_b = (probs_b >= 0.5).astype(int)

        m_b = evaluate_metrics(y_te, probs_b, preds_b)
        m_b.update({
            "Config_ID": "B_Burak_Filter",
            "Config_Name": "Burak Filter (k-NN)",
            "Target_Project": target_proj,
            "Train_Samples": len(X_s_burak),
            "Train_Time_Sec": train_time_b,
            "Inference_Time_Sec": inf_time_b
        })
        all_fold_records.append(m_b)

        raw_pred_dfs["B_Burak_Filter"].append(pd.DataFrame({
            "Target_Project": target_proj,
            "Test_Index": df_t_test.index,
            "True_Label": y_te,
            "Predicted_Prob": probs_b,
            "Predicted_Label": preds_b
        }))

        # -------------------------------------------------------------
        # [C] Konfigurasi C: TrAdaBoost (Adaptive Sample Reweighting)
        # -------------------------------------------------------------
        t0 = time.time()
        tb = TrAdaBoostGPU(
            n_estimators=tradaboost_iters,
            random_state=seed,
            use_gpu=use_gpu
        )
        tb.fit(X_source=X_s, y_source=y_s, X_target=X_tt, y_target=y_tt)
        train_time_c = time.time() - t0

        t0_inf = time.time()
        probs_c = tb.predict_proba(X_te)[:, 1]
        inf_time_c = time.time() - t0_inf
        preds_c = (probs_c >= 0.5).astype(int)

        m_c = evaluate_metrics(y_te, probs_c, preds_c)
        m_c.update({
            "Config_ID": "C_TrAdaBoost",
            "Config_Name": "TrAdaBoost",
            "Target_Project": target_proj,
            "Train_Samples": len(X_s) + len(X_tt),
            "Train_Time_Sec": train_time_c,
            "Inference_Time_Sec": inf_time_c
        })
        all_fold_records.append(m_c)

        raw_pred_dfs["C_TrAdaBoost"].append(pd.DataFrame({
            "Target_Project": target_proj,
            "Test_Index": df_t_test.index,
            "True_Label": y_te,
            "Predicted_Prob": probs_c,
            "Predicted_Label": preds_c
        }))

        # -------------------------------------------------------------
        # [D] Konfigurasi D: Hybrid Burak Filter + TrAdaBoost
        # -------------------------------------------------------------
        t0 = time.time()
        X_s_hyb, y_s_hyb, _ = burak_filter_ratio(
            X_source=X_s,
            y_source=y_s,
            X_target=X_te,
            ratio=hybrid_ratio
        )
        tb_hyb = TrAdaBoostGPU(
            n_estimators=tradaboost_iters,
            random_state=seed,
            use_gpu=use_gpu
        )
        tb_hyb.fit(X_source=X_s_hyb, y_source=y_s_hyb, X_target=X_tt, y_target=y_tt)
        train_time_d = time.time() - t0

        t0_inf = time.time()
        probs_d = tb_hyb.predict_proba(X_te)[:, 1]
        inf_time_d = time.time() - t0_inf
        preds_d = (probs_d >= 0.5).astype(int)

        m_d = evaluate_metrics(y_te, probs_d, preds_d)
        m_d.update({
            "Config_ID": "D_Burak_TrAdaBoost_Hybrid",
            "Config_Name": "Hybrid Burak + TrAdaBoost",
            "Target_Project": target_proj,
            "Train_Samples": len(X_s_hyb) + len(X_tt),
            "Train_Time_Sec": train_time_d,
            "Inference_Time_Sec": inf_time_d
        })
        all_fold_records.append(m_d)

        raw_pred_dfs["D_Burak_TrAdaBoost_Hybrid"].append(pd.DataFrame({
            "Target_Project": target_proj,
            "Test_Index": df_t_test.index,
            "True_Label": y_te,
            "Predicted_Prob": probs_d,
            "Predicted_Label": preds_d
        }))

        print(f"   [A Baseline] PR-AUC: {m_a['PR_AUC']:.4f} | F1: {m_a['F1']:.4f} | Recall: {m_a['Recall']:.4f} | Train: {train_time_a:.2f}s")
        print(f"   [B Burak   ] PR-AUC: {m_b['PR_AUC']:.4f} | F1: {m_b['F1']:.4f} | Recall: {m_b['Recall']:.4f} | Train: {train_time_b:.2f}s")
        print(f"   [C TrAda   ] PR-AUC: {m_c['PR_AUC']:.4f} | F1: {m_c['F1']:.4f} | Recall: {m_c['Recall']:.4f} | Train: {train_time_c:.2f}s")
        print(f"   [D Hybrid  ] PR-AUC: {m_d['PR_AUC']:.4f} | F1: {m_d['F1']:.4f} | Recall: {m_d['Recall']:.4f} | Train: {train_time_d:.2f}s")

    elapsed_total = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"✅ EKSPERIMEN LOPO-CV SELESAI DALAM {elapsed_total:.2f} DETIK ({elapsed_total/60:.2f} MENIT)!")
    print("=" * 80)

    # 3. Simpan Raw Predictions per Konfigurasi
    for cid, dfs in raw_pred_dfs.items():
        comb_df = pd.concat(dfs, ignore_index=True)
        raw_file = raw_dir / f"{cid}_raw_predictions.csv"
        comb_df.to_csv(raw_file, index=False)
        print(f"[SAVED] Raw predictions untuk {cid}: {raw_file}")

    # 4. Simpan Log Per-Fold
    df_folds = pd.DataFrame(all_fold_records)
    folds_file = out_path / "transfer_learning_folds.csv"
    df_folds.to_csv(folds_file, index=False)
    print(f"[SAVED] Log metrik per fold disimpan ke: {folds_file}")

    # 5. Hitung Ringkasan Statistik Agregat (Mean ± Std Dev, Median)
    summary_rows = []
    configs_order = [
        ("A_Baseline", "Baseline Cross-Project (No Transfer)"),
        ("B_Burak_Filter", "Burak Filter (k-NN Instance Selection)"),
        ("C_TrAdaBoost", "TrAdaBoost (Adaptive Sample Reweighting)"),
        ("D_Burak_TrAdaBoost_Hybrid", "Hybrid Burak Filter + TrAdaBoost")
    ]

    metric_cols = ["PR_AUC", "F1", "Recall", "Precision", "ROC_AUC", "MCC", "Balanced_Accuracy", "Train_Time_Sec"]

    for cid, cname in configs_order:
        sub = df_folds[df_folds["Config_ID"] == cid]
        row = {
            "Config_ID": cid,
            "Config_Name": cname,
            "N_Folds": len(sub),
        }
        for col in metric_cols:
            vals = sub[col].dropna()
            row[f"Mean_{col}"] = float(vals.mean())
            row[f"Std_{col}"] = float(vals.std())
            row[f"Median_{col}"] = float(vals.median())

        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    summary_file = out_path / "transfer_learning_summary.csv"
    df_summary.to_csv(summary_file, index=False)
    print(f"[SAVED] Ringkasan komparasi metrik disimpan ke: {summary_file}")

    return df_folds, df_summary


if __name__ == "__main__":
    run_transfer_lopo_experiment()
