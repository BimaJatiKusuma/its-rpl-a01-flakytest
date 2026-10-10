"""
Engine Eksekusi LOPO-CV untuk Eksperimen Alternatif 2:
Validation-Based Dynamic Threshold Tuning vs Default 0.5.

Membandingkan 5 Strategi Threshold:
- T1: Default Baseline (tau = 0.5)
- T2: Max-F1 Dynamic Tuning (tau* = argmax F1)
- T3: Recall-Constrained Tuning (Recall >= 70%)
- T4: Youden's J-Statistic (tau* = argmax [TPR - FPR])
- T5: Prior-Shifted Threshold (tau = prior flaky empiris)

Model yang diuji:
- XGBoost GPU (tree_method='hist', device='cuda')
- Random Forest Classifier (n_estimators=100, n_jobs=-1)

Hardware Akselerasi: Laptop GPU NVIDIA GeForce RTX 4050 (6GB VRAM, CUDA 13.4).
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
from sklearn.ensemble import RandomForestClassifier
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
from threshold_tuner import DynamicThresholdTuner
from statistical_tests import run_wilcoxon_analysis
from visualizer import generate_all_plots


def evaluate_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    y_preds: np.ndarray,
    threshold: float
) -> Dict[str, Any]:
    """
    Menghitung metrik evaluasi lengkap dengan proteksi aman untuk kasus zero-positive (seperti jimfs).
    """
    n_flaky = int(np.sum(y_true == 1))
    n_non_flaky = int(np.sum(y_true == 0))
    tn, fp, fn, tp = 0, 0, 0, 0

    if len(y_true) > 0:
        cm = confusion_matrix(y_true, y_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

    fpr_denom = fp + tn
    fpr = float(fp / fpr_denom) if fpr_denom > 0 else 0.0
    specificity = float(tn / fpr_denom) if fpr_denom > 0 else 1.0

    if n_flaky == 0:
        pr_auc = np.nan
        roc_auc = np.nan
        rec = np.nan
        prec = 0.0 if fp > 0 else 1.0
        f1 = 0.0
        mcc = 0.0
        bal_acc = float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0
    else:
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
        "Flaky_in_Test": n_flaky,
        "NonFlaky_in_Test": n_non_flaky,
        "Threshold": float(threshold),
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "MCC": mcc,
        "Specificity": specificity,
        "FPR": fpr,
        "Balanced_Accuracy": bal_acc,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def create_model(model_name: str, config: Dict[str, Any]):
    """Instansiasi classifier sesuai spesifikasi konfigurasi."""
    if model_name == "xgboost_gpu":
        cfg = config.get("models", {}).get("xgboost_gpu", {})
        return xgb.XGBClassifier(
            n_estimators=cfg.get("n_estimators", 100),
            max_depth=cfg.get("max_depth", 6),
            learning_rate=cfg.get("learning_rate", 0.1),
            tree_method=cfg.get("tree_method", "hist"),
            device=cfg.get("device", "cuda"),
            eval_metric=cfg.get("eval_metric", "logloss"),
            random_state=cfg.get("random_state", 42),
        )
    elif model_name == "random_forest":
        cfg = config.get("models", {}).get("random_forest", {})
        return RandomForestClassifier(
            n_estimators=cfg.get("n_estimators", 100),
            n_jobs=cfg.get("n_jobs", -1),
            random_state=cfg.get("random_state", 42),
        )
    else:
        raise ValueError(f"Model {model_name} tidak dikenali.")


def run_lopo_dynamic_threshold(
    config_path: str = None,
    models_to_run: List[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Menjalankan seluruh 24 fold Leave-One-Project-Out Cross-Validation (LOPO-CV)
    untuk menguji 5 strategi ambang batas probabilitas pada model terpilih.
    """
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent

    if config_path is None:
        cfg_file = base_dir / "configs" / "threshold_config.json"
    else:
        cfg_file = Path(config_path)

    with open(cfg_file, "r") as f:
        config = json.load(f)

    if models_to_run is None:
        models_to_run = ["xgboost_gpu", "random_forest"]

    results_dir = base_dir / "results"
    raw_preds_dir = results_dir / "raw_predictions"
    plots_dir = results_dir / "plots"

    results_dir.mkdir(parents=True, exist_ok=True)
    raw_preds_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🚀 MEMULAI EKSEKUSI ALTERNATIF 2: VALIDATION-BASED DYNAMIC THRESHOLD TUNING")
    print("=" * 80)
    print(f"Konfigurasi        : {cfg_file}")
    print(f"Model yang diuji   : {models_to_run}")
    print(f"Hardware Akselerasi: {config.get('hardware_settings', {}).get('gpu_name', 'GPU')}")
    print("=" * 80)

    # 1. Pemuatan Dataset C1
    df, features, projects = load_dataset(cleaning_strategy="C1")
    print(f"\n[INFO] Dataset memuat {len(df)} baris valid dari {len(projects)} proyek software.")
    print(f"[INFO] Total sampel flaky: {df['flaky'].sum()} ({df['flaky'].mean()*100:.2f}%)\n")

    # Inisialisasi Threshold Tuner
    grid_cfg = config.get("grid_search", {})
    tuner = DynamicThresholdTuner(
        min_threshold=grid_cfg.get("min_threshold", 0.01),
        max_threshold=grid_cfg.get("max_threshold", 0.99),
        step=grid_cfg.get("step", 0.01),
        recall_constraint=grid_cfg.get("recall_constraint", 0.70),
        inner_val_ratio=config.get("inner_validation", {}).get("val_ratio", 0.20),
        random_state=config.get("inner_validation", {}).get("random_state", 42)
    )

    strategies_meta = {s["id"]: s["name"] for s in config.get("strategies", [])}
    all_fold_records: List[Dict[str, Any]] = []

    # Iterasi Model
    for model_name in models_to_run:
        print("\n" + "#" * 80)
        print(f"### EVALUASI MODEL: {model_name.upper()} ###")
        print("#" * 80)

        total_model_start = time.time()

        for fold_idx, target_project in enumerate(projects, 1):
            fold_start = time.time()

            # Split Source (N-1) dan Target (1)
            train_mask = (df["project"] != target_project)
            test_mask = (df["project"] == target_project)

            df_source = df[train_mask]
            df_target = df[test_mask]

            X_source = df_source[features].values.astype(float)
            y_source = df_source["flaky"].values.astype(int)

            X_target = df_target[features].values.astype(float)
            y_target = df_target["flaky"].values.astype(int)

            # Standardisasi (Fit HANYA pada sumber, transform target) -> Zero Leakage
            scaler = StandardScaler()
            X_source_scaled = scaler.fit_transform(X_source)
            X_target_scaled = scaler.transform(X_target)

            # ----------------------------------------------------
            # TAHAP 1: INNER-VALIDATION TUNING (Zero Test Leakage)
            # ----------------------------------------------------
            inner_start = time.time()
            X_in_train, X_in_val, y_in_train, y_in_val = tuner.split_source_data(
                X_source_scaled, y_source
            )

            # Latih model inner
            inner_model = create_model(model_name, config)
            inner_model.fit(X_in_train, y_in_train)

            # Inferensi probabilitas inner validation
            p_in_val = inner_model.predict_proba(X_in_val)[:, 1]

            # Optimasi threshold
            optimal_thresholds, _ = tuner.tune_thresholds(y_in_val, p_in_val, y_source)
            inner_duration = time.time() - inner_start

            # ----------------------------------------------------
            # TAHAP 2: TRAINING MODEL FINAL PADA SELURUH DATA SUMBER
            # ----------------------------------------------------
            train_start = time.time()
            final_model = create_model(model_name, config)
            final_model.fit(X_source_scaled, y_source)
            train_duration = time.time() - train_start

            # ----------------------------------------------------
            # TAHAP 3: INFERENSI PADA PROYEK TARGET (TEST FOLD)
            # ----------------------------------------------------
            p_test = final_model.predict_proba(X_target_scaled)[:, 1]

            # Log raw prediction untuk audit
            raw_pred_dict = {
                "project": target_project,
                "y_true": y_target,
                "prob_flaky": p_test
            }

            # Evaluasi ke-5 strategi threshold
            for strat_id, tau_val in optimal_thresholds.items():
                y_pred = (p_test >= tau_val).astype(int)
                raw_pred_dict[f"pred_{strat_id}"] = y_pred

                metrics = evaluate_metrics(y_target, p_test, y_pred, threshold=tau_val)

                record = {
                    "Model": model_name,
                    "Fold_Idx": fold_idx,
                    "Target_Project": target_project,
                    "Strategy_ID": strat_id,
                    "Strategy_Name": strategies_meta.get(strat_id, strat_id),
                    "Train_Time_Sec": float(train_duration),
                    "Inner_Val_Time_Sec": float(inner_duration),
                    **metrics
                }
                all_fold_records.append(record)

            # Simpan raw predictions per proyek
            df_raw_pred = pd.DataFrame(raw_pred_dict)
            raw_save_path = raw_preds_dir / f"{model_name}_{target_project}_preds.csv"
            df_raw_pred.to_csv(raw_save_path, index=False)

            # Cetak ringkasan fold
            fold_duration = time.time() - fold_start
            t1_rec = next(r for r in all_fold_records if r["Model"] == model_name and r["Target_Project"] == target_project and r["Strategy_ID"] == "T1_Default")
            t2_rec = next(r for r in all_fold_records if r["Model"] == model_name and r["Target_Project"] == target_project and r["Strategy_ID"] == "T2_Max_F1")
            t3_rec = next(r for r in all_fold_records if r["Model"] == model_name and r["Target_Project"] == target_project and r["Strategy_ID"] == "T3_Recall_Constrained")

            print(
                f"[Fold {fold_idx:02d}/24] {target_project:<20} | Test: {len(y_target):<4} (Flaky: {np.sum(y_target==1):<2}) | "
                f"T1 F1: {t1_rec['F1']:.4f} (tau={t1_rec['Threshold']:.2f}) -> "
                f"T2 F1: {t2_rec['F1']:.4f} (tau={t2_rec['Threshold']:.2f}) | "
                f"T3 Rec: {t3_rec['Recall']:.4f} (tau={t3_rec['Threshold']:.2f}) | {fold_duration:.2f}s"
            )

        model_duration = time.time() - total_model_start
        print(f"\n[SELESAI MODEL] Evaluasi LOPO-CV 24 fold untuk {model_name} tuntas dalam {model_duration:.2f} detik.")

    # ----------------------------------------------------
    # TAHAP 4: MENYUSUN REKAPITULASI HASIL & METRIK
    # ----------------------------------------------------
    df_folds = pd.DataFrame(all_fold_records)
    folds_save_path = results_dir / "fold_thresholds.csv"
    df_folds.to_csv(folds_save_path, index=False)
    print(f"\n[SAVED] Hasil lengkap per fold tersimpan di: {folds_save_path}")

    # Menghitung Summary (Mean ± Std Dev) lintas fold
    metric_cols = [
        "Threshold", "F1", "Recall", "Precision", "MCC", "Specificity", "FPR",
        "PR_AUC", "ROC_AUC", "Balanced_Accuracy", "Train_Time_Sec", "Inner_Val_Time_Sec"
    ]

    summary_records = []
    for model_name in models_to_run:
        df_mod = df_folds[df_folds["Model"] == model_name]
        for strat_id in config.get("strategies", []):
            sid = strat_id["id"]
            df_strat = df_mod[df_mod["Strategy_ID"] == sid]

            rec = {
                "Model": model_name,
                "Strategy_ID": sid,
                "Strategy_Name": strat_id["name"],
                "Total_Folds": len(df_strat),
                "Min_Threshold": float(df_strat["Threshold"].min()),
                "Max_Threshold": float(df_strat["Threshold"].max()),
            }

            for col in metric_cols:
                vals = df_strat[col].dropna()
                rec[f"Mean_{col}"] = float(vals.mean()) if len(vals) > 0 else np.nan
                rec[f"Std_{col}"] = float(vals.std()) if len(vals) > 0 else np.nan

            summary_records.append(rec)

    df_summary = pd.DataFrame(summary_records)
    summary_save_path = results_dir / "threshold_comparison_summary.csv"
    df_summary.to_csv(summary_save_path, index=False)
    print(f"[SAVED] Ringkasan perbandingan metrik tersimpan di: {summary_save_path}")

    # Cetak tabel summary di terminal
    print("\n" + "=" * 105)
    print("📊 TABEL PERBANDINGAN METRIK EVALUASI UTAMA (MEAN ± STD DEV)")
    print("=" * 105)
    print(f"{'Model':<14} | {'Strategi':<25} | {'Threshold':<14} | {'F1-Score':<14} | {'Recall':<14} | {'Precision':<14} | {'MCC':<10}")
    print("-" * 105)
    for _, row in df_summary.iterrows():
        th_str = f"{row['Mean_Threshold']:.2f} ± {row['Std_Threshold']:.2f}"
        f1_str = f"{row['Mean_F1']:.4f} ± {row['Std_F1']:.4f}"
        rec_str = f"{row['Mean_Recall']:.4f} ± {row['Std_Recall']:.4f}"
        prec_str = f"{row['Mean_Precision']:.4f} ± {row['Std_Precision']:.4f}"
        mcc_str = f"{row['Mean_MCC']:.4f}"
        print(f"{row['Model']:<14} | {row['Strategy_Name']:<25} | {th_str:<14} | {f1_str:<14} | {rec_str:<14} | {prec_str:<14} | {mcc_str:<10}")
    print("=" * 105)

    # ----------------------------------------------------
    # TAHAP 5: UJI SIGNIFIKANSI STATISTIK & VISUALISASI
    # ----------------------------------------------------
    for m in models_to_run:
        stat_out = results_dir / f"wilcoxon_test_results_{m}.csv"
        run_wilcoxon_analysis(
            folds_csv_path=str(folds_save_path),
            output_file=str(stat_out),
            model_name=m
        )

    # Uji Wilcoxon standar di root results
    run_wilcoxon_analysis(
        folds_csv_path=str(folds_save_path),
        output_file=str(results_dir / "wilcoxon_test_results.csv"),
        model_name="xgboost_gpu"
    )

    # Hasilkan seluruh grafik publikasi
    generate_all_plots(
        folds_csv_path=str(folds_save_path),
        summary_csv_path=str(summary_save_path),
        plots_dir=str(plots_dir)
    )

    print("\n" + "=" * 80)
    print("🎉 SELURUH TAHAPAN EKSEKUSI ALTERNATIF 2 TELAH SELESAI DENGAN SUKSES!")
    print("=" * 80)

    return df_folds, df_summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Runner LOPO-CV Dynamic Threshold Tuning")
    parser.add_argument("--models", nargs="+", default=["xgboost_gpu", "random_forest"], help="Model yang diuji")
    args = parser.parse_args()

    run_lopo_dynamic_threshold(models_to_run=args.models)
