"""
Engine Eksekusi Leave-One-Project-Out Cross-Validation (LOPO-CV) untuk 72 Pipelines.
Mengevaluasi secara terpadu seluruh kombinasi faktorial:
3 Data Cleaning (C1/C2/C3) x 3 Scaling (S1/S2/S3) x 4 Imbalance (B1/B2/B3/B4) x 2 Models (M1/M2).

Total Evaluasi (Single Run):
- C1: 24 Pipelines x 24 Folds = 576 Folds
- C2: 24 Pipelines x 19 Folds = 456 Folds
- C3: 24 Pipelines x 24 Folds = 576 Folds
Total: 1.608 Model Folds.
"""

import os
import sys
import time
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
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
from data_loader import load_dataset
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


def run_pipeline_lopo(
    df: pd.DataFrame,
    features: List[str],
    projects: List[str],
    pipe_spec: Dict[str, Any],
    output_dir: Optional[Path] = None,
    use_gpu: bool = True,
    random_state: int = 42,
    pipe_idx: int = 1,
    total_pipes: int = 72
) -> pd.DataFrame:
    """
    Mengeksekusi 1 konfigurasi pipeline melewati seluruh iterasi LOPO-CV (Leave-One-Project-Out).
    """
    pipe_id = pipe_spec["pipeline_id"]
    c_type = pipe_spec["cleaning"]
    s_type = pipe_spec["scaling"]
    b_type = pipe_spec["imbalance"]
    m_type = pipe_spec["model"]

    fold_results = []
    start_time = time.time()

    for fold_idx, proj in enumerate(projects, start=1):
        fold_start = time.time()
        # 1. Isolasi Train ($N-1$ proyek) dan Test ($1$ target proyek)
        train_mask = df["project"] != proj
        test_mask = df["project"] == proj

        df_train = df[train_mask]
        df_test = df[test_mask]

        X_train_raw = df_train[features].values.astype(np.float32)
        y_train_raw = df_train["flaky"].values.astype(int)

        X_test_raw = df_test[features].values.astype(np.float32)
        y_test = df_test["flaky"].values.astype(int)

        # 2. Data Scaling (Zero Leakage: fit HANYA pada train fold)
        scaler = get_scaler(s_type)
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # 3. Resampling Ketimpangan Kelas (HANYA pada train fold)
        X_train_res, y_train_res = apply_resampling(
            X_train_scaled, y_train_raw, b_type, random_state=random_state
        )

        # 4. Hitung bobot kelas untuk B4
        pos_weight = 1.0
        n_pos = np.sum(y_train_res == 1)
        n_neg = np.sum(y_train_res == 0)
        if n_pos > 0:
            pos_weight = float(n_neg / n_pos)

        # 5. Inisialisasi Model
        clf = build_classifier(
            model_type=m_type,
            imbalance_type=b_type,
            use_gpu=use_gpu,
            scale_pos_weight_val=pos_weight,
            random_state=random_state
        )

        # 6. Pelatihan Model
        clf.fit(X_train_res, y_train_res)

        # 7. Prediksi pada fold uji unseen
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_test_scaled)
            y_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        else:
            y_probs = clf.predict(X_test_scaled).astype(float)

        y_preds = (y_probs >= 0.5).astype(int)

        # 8. Evaluasi Metrik
        metrics = evaluate_single_fold(y_test, y_probs, y_preds)
        metrics["Pipeline_ID"] = pipe_id
        metrics["Cleaning"] = c_type
        metrics["Scaling"] = s_type
        metrics["Imbalance"] = b_type
        metrics["Model"] = m_type
        metrics["Target_Project"] = proj

        fold_results.append(metrics)
        fold_elapsed = time.time() - fold_start

        # Cetak progres real-time per fold ke terminal (seperti fold1 P1/P72)
        print(
            f"  ↳ [{pipe_id} ({pipe_idx:02d}/{total_pipes:02d})] [Fold {fold_idx:02d}/{len(projects):02d}: {proj:<16}] "
            f"Flaky: {metrics['Flaky_in_Test']:3d} | F1: {metrics['F1']:.4f} | Rec: {metrics['Recall']:.4f} | Prec: {metrics['Precision']:.4f} ({fold_elapsed:.2f}s)"
        )

    df_res = pd.DataFrame(fold_results)
    elapsed = time.time() - start_time
    m_f1 = df_res["F1"].mean()
    m_prauc = df_res["PR_AUC"].mean()
    print(f"[DONE] Selesai [{pipe_id} ({pipe_idx:02d}/{total_pipes:02d})] {len(projects)} Folds ({elapsed:.2f}s) | Mean F1: {m_f1:.4f} | Mean PR-AUC: {m_prauc:.4f}")

    if output_dir:
        raw_dir = Path(output_dir) / "raw_predictions"
        raw_dir.mkdir(parents=True, exist_ok=True)
        out_file = raw_dir / f"{pipe_id}.csv"
        df_res.to_csv(out_file, index=False)

    return df_res


def run_all_72_pipelines(
    output_dir: Optional[str] = None,
    use_gpu: bool = True,
    quick_test: bool = False
) -> pd.DataFrame:
    """
    Eksekutor terpadu seluruh 72 matriks pipeline eksperimen gabungan.
    """
    start_total_time = time.time()
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent.parent / "results")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 85)
    print("🚀 MEMULAI EKSEKUSI EKSPERIMEN GABUNGAN 1 & 2 (72 FACTORIAL PIPELINES)")
    print("=" * 85)

    pipelines = get_pipeline_specs()

    # Cache dataset C1, C2, dan C3
    data_cache = {}
    for c_strat in ["C1", "C2", "C3"]:
        df_c, feat_c, proj_c = load_dataset(cleaning_strategy=c_strat)
        if quick_test:
            proj_c = proj_c[:2]
        data_cache[c_strat] = (df_c, feat_c, proj_c)

    summary_rows = []

    for p_idx, pipe_spec in enumerate(pipelines, start=1):
        pid = pipe_spec["pipeline_id"]
        c_type = pipe_spec["cleaning"]
        df_curr, feat_curr, proj_curr = data_cache[c_type]

        p_start = time.time()
        df_pipe = run_pipeline_lopo(
            df=df_curr,
            features=feat_curr,
            projects=proj_curr,
            pipe_spec=pipe_spec,
            output_dir=out_path,
            use_gpu=use_gpu,
            pipe_idx=p_idx,
            total_pipes=len(pipelines)
        )
        p_elapsed = time.time() - p_start

        m_f1 = float(df_pipe["F1"].mean())
        m_rec = float(df_pipe["Recall"].mean())
        m_prec = float(df_pipe["Precision"].mean())
        m_prauc = float(df_pipe["PR_AUC"].mean())
        m_rocauc = float(df_pipe["ROC_AUC"].mean())
        m_mcc = float(df_pipe["MCC"].mean())
        m_balacc = float(df_pipe["Balanced_Accuracy"].mean())

        summary_rows.append({
            "Pipeline_ID": pid,
            "Cleaning": c_type,
            "Scaling": pipe_spec["scaling"],
            "Imbalance": pipe_spec["imbalance"],
            "Model": pipe_spec["model"],
            "Mean_F1": m_f1,
            "Std_F1": float(df_pipe["F1"].std()),
            "Mean_Recall": m_rec,
            "Mean_Precision": m_prec,
            "Mean_PR_AUC": m_prauc,
            "Mean_ROC_AUC": m_rocauc,
            "Mean_MCC": m_mcc,
            "Mean_Balanced_Accuracy": m_balacc,
            "Total_TP": int(df_pipe["TP"].sum()),
            "Total_FP": int(df_pipe["FP"].sum()),
            "Total_FN": int(df_pipe["FN"].sum()),
            "Total_TN": int(df_pipe["TN"].sum()),
            "Execution_Time_Sec": p_elapsed,
        })

        print(
            f"  [{p_idx:02d}/{len(pipelines):02d}] {pid:<20} | "
            f"F1: {m_f1:.4f} | Rec: {m_rec:.4f} | Prec: {m_prec:.4f} | "
            f"PR-AUC: {m_prauc:.4f} | ({p_elapsed:.2f}s)"
        )

    df_summary = pd.DataFrame(summary_rows)
    df_summary.sort_values(by="Mean_F1", ascending=False, inplace=True)
    summary_file = out_path / "summary_metrics_72_pipelines.csv"
    df_summary.to_csv(summary_file, index=False)

    print(f"\n[ALL DONE] Eksekusi selesai dalam {(time.time() - start_total_time) / 60:.2f} menit!")
    print(f"Ringkasan tersimpan ke: {summary_file}")
    return df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LOPO-CV Runner Eksperimen Gabungan 1 & 2")
    parser.add_argument("--pipeline_id", type=str, default=None, help="Jalankan 1 pipeline tertentu")
    parser.add_argument("--cleaning", type=str, default=None, choices=["C1", "C2", "C3"])
    parser.add_argument("--all", action="store_true", help="Jalankan seluruh 72 pipelines")
    parser.add_argument("--quick_test", action="store_true", help="Smoke test cepat (2 folds)")
    parser.add_argument("--no_gpu", action="store_true", help="Nonaktifkan GPU")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    default_out = Path(__file__).resolve().parent.parent / "results"
    target_out = Path(args.output_dir) if args.output_dir else default_out
    use_gpu_flag = not args.no_gpu

    if args.quick_test:
        print("[MODE] QUICK SMOKE TEST (2 folds)...")
        run_all_72_pipelines(output_dir=str(target_out), use_gpu=use_gpu_flag, quick_test=True)
    elif args.all or (args.pipeline_id is None and args.cleaning is None):
        print("[MODE] Menjalankan seluruh 72 pipelines eksperimen gabungan...")
        run_all_72_pipelines(output_dir=str(target_out), use_gpu=use_gpu_flag)
    elif args.pipeline_id:
        all_specs = get_pipeline_specs()
        matched = [p for p in all_specs if p["pipeline_id"].upper() == args.pipeline_id.upper()]
        if not matched:
            print(f"[ERROR] Pipeline '{args.pipeline_id}' tidak ditemukan.")
            sys.exit(1)
        spec = matched[0]
        df_d, feat_d, proj_d = load_dataset(spec["cleaning"])
        df_p = run_pipeline_lopo(df_d, feat_d, proj_d, spec, output_dir=target_out, use_gpu=use_gpu_flag)
        print("\nHASIL LOPO-CV:")
        print(df_p[["Target_Project", "F1", "Recall", "Precision", "PR_AUC", "MCC"]].to_string(index=False))
