"""
Engine Eksekusi Leave-One-Project-Out Cross-Validation (LOPO-CV) untuk 24 Pipelines pada Eksperimen 2.
Mengevaluasi model pada dataset terimputasi: Experiment_Ali/data/cleaned_test_features.csv
Menghasilkan metrik evaluasi per fold (24 Folds) dan ringkasan agregat (Mean, Std, Median).
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Konfigurasi UTF-8 encoding untuk Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_cleaned_dataset
from pipeline_builder import get_scaler, apply_resampling, build_classifier, get_pipeline_specs


def evaluate_single_fold(
    y_test: np.ndarray,
    y_probs: np.ndarray,
    y_preds: np.ndarray
) -> Dict[str, float]:
    """
    Menghitung metrik performa dengan penanganan kasus khusus proyek tanpa flaky test (seperti jimfs).
    """
    n_flaky = int(np.sum(y_test == 1))
    n_non_flaky = int(np.sum(y_test == 0))
    tn, fp, fn, tp = 0, 0, 0, 0

    if len(y_test) > 0:
        cm = confusion_matrix(y_test, y_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

    # Kasus khusus jimfs: jika tidak ada flaky test di target test fold (n_flaky == 0)
    if n_flaky == 0:
        pr_auc = np.nan
        roc_auc = np.nan
        rec = np.nan
        prec = 0.0 if fp > 0 else 1.0
        f1 = 0.0
        mcc = 0.0
        bal_acc = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    else:
        pr_auc = float(average_precision_score(y_test, y_probs))
        try:
            roc_auc = float(roc_auc_score(y_test, y_probs))
        except ValueError:
            roc_auc = np.nan

        prec = float(precision_score(y_test, y_preds, pos_label=1, zero_division=0))
        rec = float(recall_score(y_test, y_preds, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_test, y_preds, pos_label=1, zero_division=0))
        mcc = float(matthews_corrcoef(y_test, y_preds))
        bal_acc = float(balanced_accuracy_score(y_test, y_preds))

    return {
        "Test_Size": len(y_test),
        "Flaky_in_Test": n_flaky,
        "NonFlaky_in_Test": n_non_flaky,
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc,
        "F1": f1,
        "Recall": rec,
        "Precision": prec,
        "MCC": mcc,
        "Balanced_Accuracy": bal_acc,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def run_pipeline_lopo(
    df: pd.DataFrame,
    features: List[str],
    projects: List[str],
    pipe_spec: Dict[str, Any],
    use_gpu: bool = True
) -> pd.DataFrame:
    """
    Mengeksekusi 1 konfigurasi pipeline pada seluruh lipatan 24 proyek (LOPO-CV).
    """
    pipe_id = pipe_spec["pipeline_id"]
    s_type = pipe_spec["scaling"]
    b_type = pipe_spec["imbalance"]
    m_type = pipe_spec["model"]

    fold_results = []

    for target_proj in projects:
        # 1. Isolasi Train ($N-1$ proyek) dan Test ($1$ target proyek)
        train_mask = df["project"] != target_proj
        test_mask = df["project"] == target_proj

        df_train = df[train_mask]
        df_test = df[test_mask]

        X_train_raw = df_train[features].values.astype(np.float32)
        y_train_raw = df_train["flaky"].values.astype(int)

        X_test_raw = df_test[features].values.astype(np.float32)
        y_test = df_test["flaky"].values.astype(int)

        # 2. Data Scaling (Zero Leakage: fit HANYA pada train)
        scaler = get_scaler(s_type)
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # 3. Resampling Ketimpangan Kelas (HANYA pada train)
        X_train_res, y_train_res = apply_resampling(X_train_scaled, y_train_raw, b_type)

        # 4. Inisialisasi Model
        clf = build_classifier(m_type, b_type, use_gpu=use_gpu)

        # Penyesuaian khusus XGBoost jika B4 (ClassWeight / Cost-Sensitive)
        if (m_type == "M2" or "XGB" in m_type) and (b_type == "B4"):
            neg_c = np.sum(y_train_res == 0)
            pos_c = np.sum(y_train_res == 1)
            scale_pos = (neg_c / pos_c) if pos_c > 0 else 1.0
            clf.set_params(scale_pos_weight=scale_pos)

        # 5. Pelatihan Model
        clf.fit(X_train_res, y_train_res)

        # 6. Prediksi
        if hasattr(clf, "predict_proba"):
            y_probs = clf.predict_proba(X_test_scaled)[:, 1]
        else:
            y_probs = clf.predict(X_test_scaled).astype(float)

        y_preds = (y_probs >= 0.5).astype(int)

        # 7. Evaluasi Metrik
        metrics = evaluate_single_fold(y_test, y_probs, y_preds)
        metrics["Pipeline_ID"] = pipe_id
        metrics["Scaling"] = s_type
        metrics["Imbalance"] = b_type
        metrics["Model"] = m_type
        metrics["Target_Project"] = target_proj
        fold_results.append(metrics)

    return pd.DataFrame(fold_results)


def run_all_24_pipelines(
    data_path: str = None,
    output_dir: str = None,
    use_gpu: bool = True
) -> pd.DataFrame:
    """
    Eksekusi penuh 24 pipeline eksperimen secara Leave-One-Project-Out Cross-Validation.
    """
    start_total_time = time.time()
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent.parent / "results")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    raw_preds_dir = out_path / "raw_predictions"
    raw_preds_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("🚀 MEMULAI EKSEKUSI EKSPERIMEN 2: LOPO-CV PADA DATASET TERIMPUTASI (24 PIPELINES)")
    print("=" * 80)

    # Muat dataset bersih terimputasi
    df, features, projects = load_cleaned_dataset(data_path)
    pipelines = get_pipeline_specs()

    print(f"Total Proyek   : {len(projects)} Folds")
    print(f"Total Pipeline : {len(pipelines)} Konfigurasi")
    print(f"Total Evaluasi : {len(projects) * len(pipelines)} Model Folds")
    print(f"Output Folder  : {out_path.resolve()}\n")

    summary_rows = []
    all_folds_list = []

    for idx, pipe in enumerate(pipelines, 1):
        p_id = pipe["pipeline_id"]
        t_start = time.time()
        print(f"[{idx:02d}/{len(pipelines):02d}] Menjalankan {p_id} ... ", end="", flush=True)

        try:
            df_folds = run_pipeline_lopo(df, features, projects, pipe, use_gpu=use_gpu)
            all_folds_list.append(df_folds)

            # Simpan log prediksi mentah per fold
            df_folds.to_csv(raw_preds_dir / f"{p_id}_folds.csv", index=False)

            # Rekapitulasi metrik
            elapsed = time.time() - t_start
            print(f"Selesai dalam {elapsed:.2f}s | Mean PR-AUC: {df_folds['PR_AUC'].mean():.4f} | Mean F1: {df_folds['F1'].mean():.4f}")

            summary_rows.append({
                "Pipeline_ID": p_id,
                "Scaling": pipe["scaling"],
                "Imbalance": pipe["imbalance"],
                "Model": pipe["model"],
                "Mean_PR_AUC": df_folds["PR_AUC"].mean(),
                "Std_PR_AUC": df_folds["PR_AUC"].std(),
                "Median_PR_AUC": df_folds["PR_AUC"].median(),
                "Mean_F1": df_folds["F1"].mean(),
                "Std_F1": df_folds["F1"].std(),
                "Median_F1": df_folds["F1"].median(),
                "Mean_Recall": df_folds["Recall"].mean(),
                "Std_Recall": df_folds["Recall"].std(),
                "Mean_Precision": df_folds["Precision"].mean(),
                "Std_Precision": df_folds["Precision"].std(),
                "Mean_ROC_AUC": df_folds["ROC_AUC"].mean(),
                "Mean_MCC": df_folds["MCC"].mean(),
                "Mean_Balanced_Accuracy": df_folds["Balanced_Accuracy"].mean(),
                "Total_TP": df_folds["TP"].sum(),
                "Total_FP": df_folds["FP"].sum(),
                "Total_TN": df_folds["TN"].sum(),
                "Total_FN": df_folds["FN"].sum(),
                "Execution_Time_Sec": elapsed,
            })
        except Exception as e:
            print(f"GAGAL: {e}")

    df_summary = pd.DataFrame(summary_rows)
    df_summary.sort_values(by="Mean_F1", ascending=False, inplace=True)
    summary_csv = out_path / "summary_metrics_24_pipelines.csv"
    df_summary.to_csv(summary_csv, index=False)

    # Gabung semua fold ke satu file untuk analisis statistik
    if all_folds_list:
        df_all_folds = pd.concat(all_folds_list, ignore_index=True)
        df_all_folds.to_csv(out_path / "all_folds_predictions.csv", index=False)

    total_time = time.time() - start_total_time
    print("\n" + "=" * 80)
    print(f"✅ EKSPERIMEN 2 SELESAI SEMPURNA DALAM {total_time:.2f} DETIK ({total_time/60:.2f} MENIT)!")
    print(f"📊 Ringkasan metrik disimpan ke: {summary_csv}")
    print("=" * 80)

    return df_summary


if __name__ == "__main__":
    df_res = run_all_24_pipelines(use_gpu=True)
    print("\nTOP 5 PIPELINES (BERDASARKAN MEAN F1-SCORE):")
    print(df_res[["Pipeline_ID", "Scaling", "Imbalance", "Model", "Mean_F1", "Mean_Recall", "Mean_Precision", "Mean_PR_AUC", "Mean_MCC"]].head(5).to_string())
