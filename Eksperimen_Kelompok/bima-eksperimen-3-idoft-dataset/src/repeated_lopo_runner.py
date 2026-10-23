"""
Engine Eksekusi Repeated Leave-One-Project-Out Cross-Validation (Repeated LOPO-CV)
untuk Replikasi Eksperimen 3 pada Dataset IDoFT (Afeltra et al., 2024 Context).

Konfigurasi Eksekusi:
- 5 Repetisi Independen (Random Seeds: [42, 100, 2024, 777, 999])
- 24 Pipelines (3 Scalers x 4 Imbalance Handlers x 2 Classifiers)
- 17 Proyek IDoFT Java
Total Evaluasi: 5 Repetisi x 24 Pipelines x 17 Folds = 2.040 Model Folds.
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

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
from data_loader import load_idoft_dataset
from pipeline_builder import get_scaler, apply_resampling, build_classifier, get_pipeline_specs

DEFAULT_SEEDS = [42, 100, 2024, 777, 999]

def evaluate_single_fold(
    y_test: np.ndarray,
    y_probs: np.ndarray,
    y_preds: np.ndarray
) -> Dict[str, float]:
    """
    Menghitung metrik performa dengan penanganan kasus target fold tanpa flaky test.
    (Pada IDoFT terdapat 4 proyek zero flaky: elastic-job-lite, handlebars.java, jimfs, orbit).
    """
    n_flaky = int(np.sum(y_test == 1))
    n_non_flaky = int(np.sum(y_test == 0))
    tn, fp, fn, tp = 0, 0, 0, 0

    if len(y_test) > 0:
        cm = confusion_matrix(y_test, y_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

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

def run_single_pipeline_fold(
    df: pd.DataFrame,
    features: List[str],
    projects: List[str],
    pipe_spec: Dict[str, Any],
    seed: int,
    rep_idx: int,
    use_gpu: bool = True
) -> pd.DataFrame:
    """
    Mengeksekusi 1 pipeline pada seluruh 17 lipatan proyek IDoFT untuk 1 random seed tertentu.
    """
    pipe_id = pipe_spec["pipeline_id"]
    s_type = pipe_spec["scaling"]
    b_type = pipe_spec["imbalance"]
    m_type = pipe_spec["model"]

    fold_results = []

    for target_proj in projects:
        # 1. Train / Test Split
        train_mask = df["project_clean"] != target_proj
        test_mask = df["project_clean"] == target_proj

        df_train = df[train_mask]
        df_test = df[test_mask]

        X_train_raw = df_train[features].values.astype(np.float32)
        y_train_raw = df_train["flaky"].values.astype(int)

        X_test_raw = df_test[features].values.astype(np.float32)
        y_test = df_test["flaky"].values.astype(int)

        # 2. Scaling (Zero Data Leakage)
        scaler = get_scaler(s_type)
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # 3. Resampling (dengan seed dinamis)
        X_train_res, y_train_res = apply_resampling(
            X_train_scaled, y_train_raw, b_type, random_state=seed
        )

        # 4. Inisialisasi Model (dengan seed dinamis)
        clf = build_classifier(m_type, b_type, use_gpu=use_gpu, random_state=seed)

        # Penyesuaian scale_pos_weight untuk XGBoost jika B4 (ClassWeight)
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
        metrics["Seed"] = seed
        metrics["Repetition"] = rep_idx
        fold_results.append(metrics)

    return pd.DataFrame(fold_results)

def run_repeated_lopo_experiments(
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    seeds: Optional[List[int]] = None,
    use_gpu: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Eksekusi penuh Repeated LOPO-CV lintas 5 random seeds pada dataset IDoFT.
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS

    start_total_time = time.time()
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent.parent / "results")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    raw_preds_dir = out_path / "raw_predictions"
    raw_preds_dir.mkdir(parents=True, exist_ok=True)

    per_seed_dir = out_path / "per_seed_summaries"
    per_seed_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print("🚀 MEMULAI EKSEKUSI REPLIKASI EKSPERIMEN 3: REPEATED LOPO-CV (DATASET IDOFT AFELTRA CONTEXT)")
    print("=" * 90)

    df, features, projects = load_idoft_dataset(data_path)
    pipelines = get_pipeline_specs()

    total_evals = len(seeds) * len(pipelines) * len(projects)
    print(f"Jumlah Repetisi : {len(seeds)} Seeds ({seeds})")
    print(f"Jumlah Pipeline : {len(pipelines)} Konfigurasi")
    print(f"Jumlah Proyek   : {len(projects)} Folds")
    print(f"Total Evaluasi  : {total_evals:,} Model Folds")
    print(f"Output Direktori: {out_path.resolve()}\n")

    all_fold_records = []
    per_seed_summary_records = []

    for rep_idx, seed in enumerate(seeds, 1):
        print("\n" + "-" * 90)
        print(f"🔄 [REPETISI {rep_idx}/{len(seeds)}] Menjalankan Evaluasi dengan Random Seed: {seed}")
        print("-" * 90)

        seed_summary_rows = []

        for p_idx, pipe in enumerate(pipelines, 1):
            p_id = pipe["pipeline_id"]
            t_pipe_start = time.time()
            print(f"  [{p_idx:02d}/24] {p_id} (Seed {seed}) ... ", end="", flush=True)

            try:
                df_fold = run_single_pipeline_fold(
                    df, features, projects, pipe, seed=seed, rep_idx=rep_idx, use_gpu=use_gpu
                )
                all_fold_records.append(df_fold)

                # Simpan per pipeline x seed
                df_fold.to_csv(raw_preds_dir / f"{p_id}_seed_{seed}_folds.csv", index=False)

                t_pipe_elapsed = time.time() - t_pipe_start
                m_f1 = df_fold["F1"].mean()
                
                # Active flaky projects (excluding zero-flaky projects)
                df_active = df_fold[df_fold["Flaky_in_Test"] > 0]
                m_f1_active = df_active["F1"].mean() if len(df_active) > 0 else 0.0

                m_prauc = df_fold["PR_AUC"].dropna().mean()
                m_rec = df_fold["Recall"].dropna().mean()
                m_prec = df_fold["Precision"].mean()

                print(f"Selesai {t_pipe_elapsed:.2f}s | Mean F1 (17): {m_f1:.4f} | F1 Active (13): {m_f1_active:.4f} | Recall: {m_rec:.4f} | PR-AUC: {m_prauc:.4f}")

                row_stat = {
                    "Seed": seed,
                    "Repetition": rep_idx,
                    "Pipeline_ID": p_id,
                    "Scaling": pipe["scaling"],
                    "Imbalance": pipe["imbalance"],
                    "Model": pipe["model"],
                    "Mean_F1": m_f1,
                    "Mean_F1_Active_13": m_f1_active,
                    "Std_F1": df_fold["F1"].std(),
                    "Median_F1": df_fold["F1"].median(),
                    "Mean_Recall": m_rec,
                    "Std_Recall": df_fold["Recall"].std(),
                    "Mean_Precision": m_prec,
                    "Std_Precision": df_fold["Precision"].std(),
                    "Mean_PR_AUC": m_prauc,
                    "Std_PR_AUC": df_fold["PR_AUC"].std(),
                    "Mean_ROC_AUC": df_fold["ROC_AUC"].dropna().mean(),
                    "Mean_MCC": df_fold["MCC"].mean(),
                    "Mean_Balanced_Accuracy": df_fold["Balanced_Accuracy"].mean(),
                    "Total_TP": int(df_fold["TP"].sum()),
                    "Total_FP": int(df_fold["FP"].sum()),
                    "Total_TN": int(df_fold["TN"].sum()),
                    "Total_FN": int(df_fold["FN"].sum()),
                    "Execution_Time_Sec": t_pipe_elapsed,
                }
                seed_summary_rows.append(row_stat)
                per_seed_summary_records.append(row_stat)

            except Exception as e:
                print(f"GAGAL: {e}")

        # Simpan ringkasan seed ini
        df_seed_summary = pd.DataFrame(seed_summary_rows)
        df_seed_summary.sort_values(by="Mean_F1", ascending=False, inplace=True)
        df_seed_summary.to_csv(per_seed_dir / f"summary_seed_{seed}.csv", index=False)

    # Gabung semua fold ke satu CSV master
    df_all_folds = pd.concat(all_fold_records, ignore_index=True)
    all_folds_csv = out_path / "all_repeated_folds_predictions.csv"
    df_all_folds.to_csv(all_folds_csv, index=False)
    print(f"\n[OK] Seluruh {len(df_all_folds):,} prediksi fold tersimpan di: {all_folds_csv}")

    # Rekapitulasi lintas repetisi (Across-Seeds Summary)
    df_per_seed_all = pd.DataFrame(per_seed_summary_records)
    df_per_seed_all.to_csv(out_path / "all_seed_pipeline_runs.csv", index=False)

    # Hitung mean & std agregat lintas seed untuk setiap pipeline
    agg_funcs = {
        "Mean_F1": ["mean", "std", "min", "max"],
        "Mean_F1_Active_13": ["mean", "std", "min", "max"],
        "Mean_Recall": ["mean", "std"],
        "Mean_Precision": ["mean", "std"],
        "Mean_PR_AUC": ["mean", "std"],
        "Mean_ROC_AUC": ["mean", "std"],
        "Mean_MCC": ["mean", "std"],
        "Total_TP": ["mean"],
        "Total_FP": ["mean"],
        "Execution_Time_Sec": ["mean"]
    }

    df_agg = df_per_seed_all.groupby(["Pipeline_ID", "Scaling", "Imbalance", "Model"]).agg(agg_funcs)
    df_agg.columns = [f"{col[0]}_{col[1].upper()}" if col[1] != "" else col[0] for col in df_agg.columns]
    df_agg = df_agg.reset_index()

    # Hitung Coefficient of Variation (CV) untuk F1 (%): Std / Mean * 100
    df_agg["CV_F1_%"] = (df_agg["Mean_F1_STD"] / df_agg["Mean_F1_MEAN"].replace(0, np.nan)) * 100.0

    # 95% Confidence Interval for F1: 1.96 * (std / sqrt(R))
    r_count = len(seeds)
    df_agg["CI95_F1_Margin"] = 1.96 * (df_agg["Mean_F1_STD"] / np.sqrt(r_count))

    df_agg.sort_values(by="Mean_F1_MEAN", ascending=False, inplace=True)
    repeated_summary_csv = out_path / "summary_metrics_repeated_24_pipelines.csv"
    df_agg.to_csv(repeated_summary_csv, index=False)

    total_time = time.time() - start_total_time
    print("\n" + "=" * 90)
    print(f"✅ REPLIKASI EKSPERIMEN 3 IDOFT SELESAI SEMPURNA DALAM {total_time:.2f} DETIK ({total_time/60:.2f} MENIT)!")
    print(f"📊 Ringkasan metrik lintas seed disimpan ke: {repeated_summary_csv}")
    print("=" * 90)

    return {
        "all_folds": df_all_folds,
        "seed_runs": df_per_seed_all,
        "repeated_summary": df_agg
    }

if __name__ == "__main__":
    results = run_repeated_lopo_experiments(use_gpu=True)
    print("\nTOP 5 PIPELINES (BERDASARKAN MEAN F1 LINTAS 5 SEEDS - DATASET IDOFT):")
    cols = ["Pipeline_ID", "Scaling", "Imbalance", "Model", "Mean_F1_MEAN", "Mean_F1_Active_13_MEAN", "Mean_F1_STD", "CV_F1_%", "Mean_Recall_MEAN", "Mean_PR_AUC_MEAN"]
    print(results["repeated_summary"][cols].head(5).to_string())
