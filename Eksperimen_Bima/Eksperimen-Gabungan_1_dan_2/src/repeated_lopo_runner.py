"""
Engine Eksekusi Repeated Leave-One-Project-Out Cross-Validation (Repeated LOPO-CV) untuk Eksperimen Gabungan 1 & 2.
Melakukan pengujian berulang hingga 10x (10 Independent Random Seeds) untuk uji konsistensi & stabilitas empiris
pada 72 Pipelines Faktorial (C1/C2/C3 x S1/S2/S3 x B1/B2/B3/B4 x M1/M2).

Total Evaluasi (Full Matrix):
10 Repetisi x [24 Folds C1 + 19 Folds C2 + 24 Folds C3] x 24 Pipelines = 16.080 Model Folds.
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

# 10 Random Seeds Baku untuk Uji Konsistensi
DEFAULT_SEEDS_10: List[int] = [42, 100, 2024, 777, 999, 123, 456, 789, 2025, 2026]


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


def run_single_pipeline_fold(
    df: pd.DataFrame,
    features: List[str],
    projects: List[str],
    pipe_spec: Dict[str, Any],
    seed: int,
    rep_idx: int,
    total_reps: int = 10,
    pipe_idx: int = 1,
    total_pipes: int = 72,
    use_gpu: bool = True
) -> pd.DataFrame:
    """
    Mengeksekusi 1 konfigurasi pipeline pada seluruh lipatan proyek untuk 1 random seed tertentu.
    Zero Data Leakage: fit scaler HANYA pada train fold, resampling & model menggunakan seed dinamis.
    """
    pipe_id = pipe_spec["pipeline_id"]
    c_type = pipe_spec["cleaning"]
    s_type = pipe_spec["scaling"]
    b_type = pipe_spec["imbalance"]
    m_type = pipe_spec["model"]

    fold_results = []

    for fold_idx, target_proj in enumerate(projects, start=1):
        fold_start = time.time()
        # 1. Isolasi Train ($N-1$ proyek) dan Test ($1$ target proyek)
        train_mask = df["project"] != target_proj
        test_mask = df["project"] == target_proj

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

        # 3. Resampling Ketimpangan Kelas dengan seed dinamis
        X_train_res, y_train_res = apply_resampling(
            X_train_scaled, y_train_raw, b_type, random_state=seed
        )

        # 4. Hitung bobot kelas untuk B4
        pos_weight = 1.0
        n_pos = np.sum(y_train_res == 1)
        n_neg = np.sum(y_train_res == 0)
        if n_pos > 0:
            pos_weight = float(n_neg / n_pos)

        # 5. Inisialisasi Model dengan seed dinamis
        clf = build_classifier(
            model_type=m_type,
            imbalance_type=b_type,
            use_gpu=use_gpu,
            scale_pos_weight_val=pos_weight,
            random_state=seed
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
        metrics["Target_Project"] = target_proj
        metrics["Seed"] = seed
        metrics["Repetition"] = rep_idx
        fold_results.append(metrics)
        fold_elapsed = time.time() - fold_start

        # Cetak status live per-fold ke terminal (seperti fold1 P1/P72)
        print(
            f"    ↳ [Rep {rep_idx:02d}/{total_reps:02d}] [{pipe_id} ({pipe_idx:02d}/{total_pipes:02d})] "
            f"[Fold {fold_idx:02d}/{len(projects):02d}: {target_proj:<16}] "
            f"Flaky: {metrics['Flaky_in_Test']:3d} | F1: {metrics['F1']:.4f} | Rec: {metrics['Recall']:.4f} | Prec: {metrics['Precision']:.4f} ({fold_elapsed:.2f}s)"
        )

    return pd.DataFrame(fold_results)


def run_repeated_lopo_experiments(
    pipelines: Optional[List[Dict[str, Any]]] = None,
    seeds: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
    use_gpu: bool = True,
    quick_test: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Eksekutor Repeated LOPO-CV Eksperimen Gabungan lintas R perulangan random seeds (default 10x).
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS_10
    if pipelines is None:
        pipelines = get_pipeline_specs()
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent.parent / "results")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    raw_preds_dir = out_path / "raw_predictions"
    raw_preds_dir.mkdir(parents=True, exist_ok=True)

    per_seed_dir = out_path / "per_seed_summaries"
    per_seed_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 85)
    print(f"🚀 REPEATED LOPO-CV EKSPERIMEN GABUNGAN 1 & 2 — UJI KONSISTENSI {len(seeds)}x PERULANGAN")
    print("=" * 85)
    print(f"Daftar Seeds     : {seeds}")
    print(f"Jumlah Pipelines : {len(pipelines)} Konfigurasi")
    print(f"Output Directory : {out_path.resolve()}\n")

    # Cache dataset C1, C2, dan C3
    data_cache = {}
    for c_strat in ["C1", "C2", "C3"]:
        df_c, feat_c, proj_c = load_dataset(cleaning_strategy=c_strat)
        if quick_test:
            proj_c = proj_c[:2]
        data_cache[c_strat] = (df_c, feat_c, proj_c)

    all_seed_runs = []
    total_start = time.time()

    for rep_idx, seed in enumerate(seeds, start=1):
        print("\n" + "#" * 80)
        print(f"🔁 [REPETISI {rep_idx}/{len(seeds)}] Menguji dengan Random Seed: {seed}")
        print("#" * 80)

        seed_start = time.time()
        seed_summary_rows = []

        for p_idx, pipe_spec in enumerate(pipelines, start=1):
            pid = pipe_spec["pipeline_id"]
            c_type = pipe_spec["cleaning"]
            df_curr, feat_curr, proj_curr = data_cache[c_type]

            p_start = time.time()
            df_pipe_folds = run_single_pipeline_fold(
                df=df_curr,
                features=feat_curr,
                projects=proj_curr,
                pipe_spec=pipe_spec,
                seed=seed,
                rep_idx=rep_idx,
                total_reps=len(seeds),
                pipe_idx=p_idx,
                total_pipes=len(pipelines),
                use_gpu=use_gpu
            )
            p_elapsed = time.time() - p_start

            # Simpan fold predictions per seed
            fold_csv = raw_preds_dir / f"{pid}_seed_{seed}_folds.csv"
            df_pipe_folds.to_csv(fold_csv, index=False)

            m_f1 = float(df_pipe_folds["F1"].mean())
            m_rec = float(df_pipe_folds["Recall"].mean())
            m_prec = float(df_pipe_folds["Precision"].mean())
            m_prauc = float(df_pipe_folds["PR_AUC"].mean())
            m_rocauc = float(df_pipe_folds["ROC_AUC"].mean())
            m_mcc = float(df_pipe_folds["MCC"].mean())
            m_balacc = float(df_pipe_folds["Balanced_Accuracy"].mean())

            summary_item = {
                "Seed": seed,
                "Repetition": rep_idx,
                "Pipeline_ID": pid,
                "Cleaning": pipe_spec["cleaning"],
                "Scaling": pipe_spec["scaling"],
                "Imbalance": pipe_spec["imbalance"],
                "Model": pipe_spec["model"],
                "Mean_F1": m_f1,
                "Std_F1": float(df_pipe_folds["F1"].std()),
                "Mean_Recall": m_rec,
                "Mean_Precision": m_prec,
                "Mean_PR_AUC": m_prauc,
                "Mean_ROC_AUC": m_rocauc,
                "Mean_MCC": m_mcc,
                "Mean_Balanced_Accuracy": m_balacc,
                "Total_TP": int(df_pipe_folds["TP"].sum()),
                "Total_FP": int(df_pipe_folds["FP"].sum()),
                "Total_FN": int(df_pipe_folds["FN"].sum()),
                "Total_TN": int(df_pipe_folds["TN"].sum()),
                "Execution_Time_Sec": p_elapsed,
            }
            seed_summary_rows.append(summary_item)
            all_seed_runs.append(summary_item)

            print(
                f"  [{p_idx:02d}/{len(pipelines):02d}] {pid:<20} | "
                f"F1: {m_f1:.4f} | Rec: {m_rec:.4f} | Prec: {m_prec:.4f} | "
                f"PR-AUC: {m_prauc:.4f} | ({p_elapsed:.2f}s)"
            )

        # Simpan ringkasan per-seed
        df_seed_summary = pd.DataFrame(seed_summary_rows)
        df_seed_summary.sort_values(by="Mean_F1", ascending=False, inplace=True)
        seed_summary_file = per_seed_dir / f"summary_seed_{seed}.csv"
        df_seed_summary.to_csv(seed_summary_file, index=False)
        print(f"  💾 Ringkasan Seed {seed} disimpan ke: {seed_summary_file.name} ({time.time() - seed_start:.2f}s)")

    # Simpan master seluruh runs
    df_all_runs = pd.DataFrame(all_seed_runs)
    all_runs_file = out_path / "all_seed_pipeline_runs.csv"
    df_all_runs.to_csv(all_runs_file, index=False)
    print(f"\n[OK] Master seluruh perulangan ({len(df_all_runs)} baris evaluasi) disimpan ke: {all_runs_file}")
    print(f"Total Waktu Keseluruhan: {(time.time() - total_start) / 60:.2f} menit")

    # Jalankan analisis konsistensi otomatis
    try:
        from consistency_analyzer import analyze_pipeline_consistency
        df_consistency = analyze_pipeline_consistency(
            all_runs_csv=str(all_runs_file),
            output_dir=output_dir
        )
    except Exception as e:
        print(f"[WARNING] Gagal menjalankan konsistensi otomatis: {e}")
        df_consistency = pd.DataFrame()

    return {
        "all_runs": df_all_runs,
        "consistency": df_consistency
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repeated LOPO-CV Engine Eksperimen Gabungan (10x Seeds)")
    parser.add_argument("--n_repeats", type=int, default=10, help="Jumlah repetisi seed (default 10)")
    parser.add_argument("--pipeline_id", type=str, default=None, help="Eksekusi 1 pipeline spesifik")
    parser.add_argument("--cleaning", type=str, default=None, choices=["C1", "C2", "C3"])
    parser.add_argument("--top_pipelines", type=int, default=None, help="Jalankan Top N pipelines saja")
    parser.add_argument("--all", action="store_true", help="Jalankan seluruh 72 pipelines")
    parser.add_argument("--quick_test", action="store_true", help="Smoke test cepat (2 folds, 2 seeds)")
    parser.add_argument("--no_gpu", action="store_true", help="Paksa gunakan CPU (tanpa CUDA)")
    parser.add_argument("--output_dir", type=str, default=None, help="Direktori output")
    args = parser.parse_args()

    all_specs = get_pipeline_specs()

    if args.quick_test:
        test_seeds = DEFAULT_SEEDS_10[:2]
        test_pipes = [all_specs[0], all_specs[24], all_specs[48]]  # P01 (C1), P25 (C2), P49 (C3)
        print("[MODE] QUICK SMOKE TEST (3 pipelines C1/C2/C3, 2 seeds, 2 folds)...")
        run_repeated_lopo_experiments(
            pipelines=test_pipes,
            seeds=test_seeds,
            output_dir=args.output_dir,
            use_gpu=not args.no_gpu,
            quick_test=True
        )
    elif args.cleaning:
        filtered = [p for p in all_specs if p["cleaning"] == args.cleaning]
        active_seeds = DEFAULT_SEEDS_10[:args.n_repeats]
        print(f"[MODE] Mengeksekusi Strategi Cleaning {args.cleaning} ({len(filtered)} pipelines) pada {len(active_seeds)} seeds...")
        run_repeated_lopo_experiments(
            pipelines=filtered,
            seeds=active_seeds,
            output_dir=args.output_dir,
            use_gpu=not args.no_gpu
        )
    elif args.top_pipelines:
        # Top 12 pipelines gabungan (Top 4 dari C1, Top 4 dari C2, Top 4 dari C3)
        top_ids = [
            "P13_C1_S2_B3_M1", "P14_C1_S2_B3_M2", "P05_C1_S1_B3_M1", "P06_C1_S1_B3_M2",
            "P37_C2_S2_B3_M1", "P38_C2_S2_B3_M2", "P45_C2_S3_B3_M1", "P46_C2_S3_B3_M2",
            "P61_C3_S2_B3_M1", "P62_C3_S2_B3_M2", "P53_C3_S1_B3_M1", "P54_C3_S1_B3_M2"
        ]
        selected_top = [p for p in all_specs if p["pipeline_id"] in top_ids[:args.top_pipelines]]
        active_seeds = DEFAULT_SEEDS_10[:args.n_repeats]
        print(f"[MODE] Mengeksekusi Top {len(selected_top)} pipelines pada {len(active_seeds)} seeds...")
        run_repeated_lopo_experiments(
            pipelines=selected_top,
            seeds=active_seeds,
            output_dir=args.output_dir,
            use_gpu=not args.no_gpu
        )
    else:
        active_seeds = DEFAULT_SEEDS_10[:args.n_repeats]
        print(f"[MODE] Mengeksekusi seluruh {len(all_specs)} pipelines pada {len(active_seeds)} seeds...")
        run_repeated_lopo_experiments(
            pipelines=all_specs,
            seeds=active_seeds,
            output_dir=args.output_dir,
            use_gpu=not args.no_gpu
        )
