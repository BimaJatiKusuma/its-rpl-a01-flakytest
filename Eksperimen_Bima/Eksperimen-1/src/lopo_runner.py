"""
Engine Eksekusi Leave-One-Project-Out Cross-Validation (LOPO-CV) untuk 48 Pipelines.
Menghasilkan metrik evaluasi lengkap per fold dan ringkasan agregat (Mean, Std, Median).
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

from data_loader import load_dataset
from pipeline_builder import get_scaler, apply_resampling, build_classifier


def evaluate_single_fold(
    y_test: np.ndarray,
    y_probs: np.ndarray,
    y_preds: np.ndarray
) -> Dict[str, float]:
    """
    Menghitung metrik performa dengan penanganan khusus proyek tanpa flaky test (seperti jimfs).
    """
    n_flaky = int(np.sum(y_test == 1))
    n_non_flaky = int(np.sum(y_test == 0))
    tn, fp, fn, tp = 0, 0, 0, 0

    if len(y_test) > 0:
        cm = confusion_matrix(y_test, y_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

    # Kasus khusus: jika tidak ada flaky test di fold uji (n_flaky == 0)
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
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "MCC": mcc,
        "Balanced_Accuracy": bal_acc,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn)
    }


def run_pipeline_lopo(
    pipeline_id: str,
    cleaning_type: str,
    scaling_type: str,
    imbalance_type: str,
    model_type: str,
    output_dir: Path,
    use_gpu: bool = True
) -> pd.DataFrame:
    """
    Mengeksekusi 1 pipeline melewati seluruh iterasi LOPO-CV (Leave-One-Project-Out).
    """
    df, features, projects = load_dataset(cleaning_type)
    fold_results = []

    print(f"\n[START] [{pipeline_id}] {cleaning_type}_{scaling_type}_{imbalance_type}_{model_type} ({len(projects)} proyek)...")
    start_time = time.time()

    for proj in projects:
        train_mask = df["project"] != proj
        test_mask = df["project"] == proj

        df_train = df[train_mask]
        df_test = df[test_mask]

        X_train_raw = df_train[features].values
        y_train = df_train["flaky"].values.astype(int)

        X_test_raw = df_test[features].values
        y_test = df_test["flaky"].values.astype(int)

        # 1. Feature Scaling (Fit HANYA pada X_train, Zero Leakage!)
        scaler = get_scaler(scaling_type)
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # 2. Imbalance Handling (HANYA pada X_train)
        X_train_final, y_train_final = apply_resampling(
            X_train_scaled, y_train, imbalance_type
        )

        # 3. Hitung bobot kelas untuk B4
        pos_weight = 1.0
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        if n_pos > 0:
            pos_weight = float(n_neg / n_pos)

        # 4. Inisialisasi & Training Model
        clf = build_classifier(
            model_type=model_type,
            imbalance_type=imbalance_type,
            use_gpu=use_gpu,
            scale_pos_weight_val=pos_weight
        )

        clf.fit(X_train_final, y_train_final)

        # 5. Prediksi pada data uji (Unseen target project)
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_test_scaled)
            y_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        else:
            y_probs = clf.predict(X_test_scaled).astype(float)

        y_preds = (y_probs >= 0.5).astype(int)

        metrics = evaluate_single_fold(y_test, y_probs, y_preds)
        metrics["Pipeline_ID"] = pipeline_id
        metrics["Cleaning"] = cleaning_type
        metrics["Scaling"] = scaling_type
        metrics["Imbalance"] = imbalance_type
        metrics["Model"] = model_type
        metrics["Target_Project"] = proj

        fold_results.append(metrics)

    elapsed = time.time() - start_time
    print(f"[DONE] Selesai [{pipeline_id}] dalam {elapsed:.2f} detik.")

    df_res = pd.DataFrame(fold_results)

    # Simpan hasil per fold langsung ke results/raw_predictions/
    raw_dir = Path(output_dir) / "raw_predictions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_file = raw_dir / f"{pipeline_id}.csv"
    df_res.to_csv(out_file, index=False)
    print(f"[SAVED] Log prediksi per-fold disimpan ke: {out_file}")

    return df_res


def run_all_48_pipelines(output_dir: str = "../results", use_gpu: bool = True):
    """
    Eksekutor terpadu seluruh 48 matriks pipeline.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cleaning_options = ["C1", "C2"]
    scaling_options = ["S1", "S2", "S3"]
    imbalance_options = ["B1", "B2", "B3", "B4"]
    model_options = ["M1", "M2"]

    all_fold_dfs = []
    summary_rows = []

    pipe_count = 0
    total_pipes = len(cleaning_options) * len(scaling_options) * len(imbalance_options) * len(model_options)

    for c in cleaning_options:
        for s in scaling_options:
            for b in imbalance_options:
                for m in model_options:
                    pipe_count += 1
                    pipe_id = f"P{pipe_count:02d}_{c}_{s}_{b}_{m}"
                    
                    df_pipe = run_pipeline_lopo(
                        pipeline_id=pipe_id,
                        cleaning_type=c,
                        scaling_type=s,
                        imbalance_type=b,
                        model_type=m,
                        output_dir=out_path,
                        use_gpu=use_gpu
                    )
                    all_fold_dfs.append(df_pipe)

                    summary_rows.append({
                        "Pipeline_ID": pipe_id,
                        "Cleaning": c,
                        "Scaling": s,
                        "Imbalance": b,
                        "Model": m,
                        "Mean_PR_AUC": df_pipe["PR_AUC"].mean(),
                        "Std_PR_AUC": df_pipe["PR_AUC"].std(),
                        "Mean_ROC_AUC": df_pipe["ROC_AUC"].mean(),
                        "Mean_F1": df_pipe["F1"].mean(),
                        "Std_F1": df_pipe["F1"].std(),
                        "Mean_Precision": df_pipe["Precision"].mean(),
                        "Mean_Recall": df_pipe["Recall"].mean(),
                        "Mean_MCC": df_pipe["MCC"].mean(),
                        "Mean_Balanced_Accuracy": df_pipe["Balanced_Accuracy"].mean()
                    })

    df_summary = pd.DataFrame(summary_rows)
    df_summary = df_summary.sort_values(by="Mean_PR_AUC", ascending=False)
    summary_file = out_path / "summary_metrics_48_pipelines.csv"
    df_summary.to_csv(summary_file, index=False)
    print(f"\n[ALL DONE] Seluruh 48 Pipeline berhasil dieksekusi! Ringkasan disimpan ke: {summary_file}")
    return df_summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LOPO-CV Runner Eksperimen 1")
    parser.add_argument("--pipeline_id", type=str, default="P01_C1_S1_B1_M1")
    parser.add_argument("--cleaning", type=str, default="C1", choices=["C1", "C2"])
    parser.add_argument("--scaling", type=str, default="S1", choices=["S1", "S2", "S3"])
    parser.add_argument("--imbalance", type=str, default="B1", choices=["B1", "B2", "B3", "B4"])
    parser.add_argument("--model", type=str, default="M1", choices=["M1", "M2"])
    parser.add_argument("--no_gpu", action="store_true", help="Nonaktifkan GPU (gunakan CPU saja)")
    parser.add_argument("--all", action="store_true", help="Jalankan seluruh 48 pipelines")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    default_out = Path(__file__).resolve().parent.parent / "results"
    target_out = Path(args.output_dir) if args.output_dir else default_out
    use_gpu_flag = not args.no_gpu

    if args.all:
        print("[MODE] Menjalankan seluruh 48 pipelines eksperimen...")
        df_summary = run_all_48_pipelines(output_dir=target_out, use_gpu=use_gpu_flag)
    else:
        print(f"[MODE] Menjalankan single pipeline: {args.pipeline_id}...")
        df_pipe = run_pipeline_lopo(
            pipeline_id=args.pipeline_id,
            cleaning_type=args.cleaning,
            scaling_type=args.scaling,
            imbalance_type=args.imbalance,
            model_type=args.model,
            output_dir=target_out,
            use_gpu=use_gpu_flag
        )
        print("\n" + "=" * 80)
        print(f"HASIL LENGKAP LOPO-CV PIPELINE: {args.pipeline_id}")
        print("=" * 80)
        cols_show = ["Target_Project", "Test_Size", "Flaky_in_Test", "PR_AUC", "ROC_AUC", "Precision", "Recall", "F1", "MCC"]
        print(df_pipe[cols_show].to_string(index=False))
        print("-" * 80)
        print(f"RATA-RATA (MEAN +/- STD) DARI {len(df_pipe)} FOLD PROYEK:")
        print(f"  Mean PR-AUC           : {df_pipe['PR_AUC'].mean():.4f} +/- {df_pipe['PR_AUC'].std():.4f}")
        print(f"  Mean ROC-AUC          : {df_pipe['ROC_AUC'].mean():.4f} +/- {df_pipe['ROC_AUC'].std():.4f}")
        print(f"  Mean F1 (Flaky)       : {df_pipe['F1'].mean():.4f} +/- {df_pipe['F1'].std():.4f}")
        print(f"  Mean Recall (Flaky)   : {df_pipe['Recall'].mean():.4f} +/- {df_pipe['Recall'].std():.4f}")
        print(f"  Mean Precision (Flaky): {df_pipe['Precision'].mean():.4f} +/- {df_pipe['Precision'].std():.4f}")
        print(f"  Mean MCC              : {df_pipe['MCC'].mean():.4f} +/- {df_pipe['MCC'].std():.4f}")
        print(f"  Mean Balanced Accuracy: {df_pipe['Balanced_Accuracy'].mean():.4f} +/- {df_pipe['Balanced_Accuracy'].std():.4f}")
        print("=" * 80)
