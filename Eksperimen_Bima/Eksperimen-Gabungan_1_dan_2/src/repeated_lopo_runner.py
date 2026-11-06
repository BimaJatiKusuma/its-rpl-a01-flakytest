"""
Repeated Leave-One-Project-Out Cross-Validation (Repeated LOPO-CV) Execution Engine for Combined Experiment 1 & 2.
Performs repeated evaluations across up to 10 independent random seeds for empirical consistency and stability testing
on 72 Full Factorial Pipelines (C1/C2/C3 x S1/S2/S3 x B1/B2/B3/B4 x M1/M2).

Full Evaluation Scope:
10 Repetitions x [24 Folds C1 + 19 Folds C2 + 24 Folds C3] x 24 Pipelines = 16,080 Total Evaluated Model Folds.
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

# Configure UTF-8 encoding for Windows console environments
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

# Standard 10 Random Seeds for Empirical Consistency Benchmarking
DEFAULT_SEEDS_10: List[int] = [42, 100, 2024, 777, 999, 123, 456, 789, 2025, 2026]


def evaluate_single_fold(
    y_test: np.ndarray,
    y_probs: np.ndarray,
    y_preds: np.ndarray
) -> Dict[str, float]:
    """
    Computes performance metrics with zero-division handling for projects with zero flaky tests (e.g., jimfs).
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
    total_reps: int,
    pipe_idx: int,
    total_pipes: int,
    use_gpu: bool = True
) -> pd.DataFrame:
    """
    Executes a single pipeline configuration across all LOPO folds under a specific random seed.
    """
    pipe_id = pipe_spec["pipeline_id"]
    c_type = pipe_spec["cleaning"]
    s_type = pipe_spec["scaling"]
    b_type = pipe_spec["imbalance"]
    m_type = pipe_spec["model"]

    fold_results = []
    pipe_start = time.time()

    for fold_idx, proj in enumerate(projects, start=1):
        fold_start = time.time()
        train_mask = df["project"] != proj
        test_mask = df["project"] == proj

        df_train = df[train_mask]
        df_test = df[test_mask]

        X_train_raw = df_train[features].values.astype(np.float32)
        y_train_raw = df_train["flaky"].values.astype(int)

        X_test_raw = df_test[features].values.astype(np.float32)
        y_test = df_test["flaky"].values.astype(int)

        # 1. Fit scaler ONLY on training fold (Zero Data Leakage guarantee)
        scaler = get_scaler(s_type)
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # 2. Resampling ONLY on training fold using the specified stochastic seed
        X_train_res, y_train_res = apply_resampling(
            X_train_scaled, y_train_raw, b_type, random_state=seed
        )

        # 3. Class imbalance ratio for cost-sensitive loss weighting
        n_pos = int(np.sum(y_train_res == 1))
        n_neg = int(np.sum(y_train_res == 0))
        spw = (n_neg / n_pos) if (n_pos > 0 and n_neg > 0) else 1.0

        # 4. Build and train classifier with seed
        clf = build_classifier(
            model_type=m_type,
            imbalance_type=b_type,
            use_gpu=use_gpu,
            scale_pos_weight_val=spw,
            random_state=seed
        )
        clf.fit(X_train_res, y_train_res)

        # 5. Predict on unseen test fold
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_test_scaled)[:, 1]
        else:
            probs = clf.predict(X_test_scaled)

        preds = (probs >= 0.5).astype(int)
        metrics = evaluate_single_fold(y_test, probs, preds)
        fold_elapsed = time.time() - fold_start

        fold_record = {
            "Seed": seed,
            "Repetition": rep_idx,
            "Pipeline_ID": pipe_id,
            "Fold": fold_idx,
            "Target_Project": proj,
            "Cleaning": c_type,
            "Scaling": s_type,
            "Imbalance": b_type,
            "Model": m_type,
            "Train_Size": len(y_train_res),
            "Train_Flaky": n_pos,
            "Train_NonFlaky": n_neg,
            "Fold_Time_Sec": fold_elapsed,
            **metrics
        }
        fold_results.append(fold_record)

    return pd.DataFrame(fold_results)


def run_repeated_lopo_experiments(
    pipelines: Optional[List[Dict[str, Any]]] = None,
    seeds: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
    use_gpu: bool = True,
    quick_test: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Executes Repeated LOPO-CV across R stochastic seed iterations (default: 10 seeds).
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
    print(f"🚀 REPEATED LOPO-CV BENCHMARK — EMPIRICAL CONSISTENCY ACROSS {len(seeds)} STOCHASTIC SEEDS")
    print("=" * 85)
    print(f"Random Seeds List   : {seeds}")
    print(f"Number of Pipelines : {len(pipelines)} Configurations")
    print(f"Target Directory    : {out_path.resolve()}\n")

    # Cache dataset C1, C2, and C3
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
        print(f"🔁 [REPETITION {rep_idx}/{len(seeds)}] Evaluating with Random Seed: {seed}")
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

            # Save fold predictions per seed
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

        # Save per-seed summary
        df_seed_summary = pd.DataFrame(seed_summary_rows)
        df_seed_summary.sort_values(by="Mean_F1", ascending=False, inplace=True)
        seed_summary_file = per_seed_dir / f"summary_seed_{seed}.csv"
        df_seed_summary.to_csv(seed_summary_file, index=False)
        print(f"  💾 Seed {seed} summary saved to: {seed_summary_file.name} ({time.time() - seed_start:.2f}s)")

    # Save master execution table across all seeds
    df_all_runs = pd.DataFrame(all_seed_runs)
    all_runs_file = out_path / "all_seed_pipeline_runs.csv"
    df_all_runs.to_csv(all_runs_file, index=False)
    print(f"\n[OK] Master repeated runs table ({len(df_all_runs)} evaluated pipeline runs) saved to: {all_runs_file}")
    print(f"Total Cumulative Execution Time: {(time.time() - total_start) / 60:.2f} minutes")

    # Automated consistency analysis invocation
    try:
        from consistency_analyzer import analyze_pipeline_consistency
        df_consistency = analyze_pipeline_consistency(
            all_runs_csv=str(all_runs_file),
            output_dir=output_dir
        )
    except Exception as e:
        print(f"[WARNING] Automated consistency analysis encountered an error: {e}")
        df_consistency = pd.DataFrame()

    return {
        "all_runs": df_all_runs,
        "consistency": df_consistency
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repeated LOPO-CV Engine for Combined Experiment 1 & 2 (10x Seeds)")
    parser.add_argument("--n_repeats", type=int, default=10, help="Number of random seed repetitions (default: 10)")
    parser.add_argument("--pipeline_id", type=str, default=None, help="Execute a specific pipeline ID")
    parser.add_argument("--cleaning", type=str, default=None, choices=["C1", "C2", "C3"], help="Execute only pipelines under a specific cleaning strategy")
    parser.add_argument("--top_pipelines", type=int, default=None, help="Execute only the Top N pipelines")
    parser.add_argument("--all", action="store_true", help="Execute all 72 full factorial pipelines")
    parser.add_argument("--quick_test", action="store_true", help="Quick smoke test verification (2 folds, 2 seeds)")
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU acceleration and run strictly on CPU")
    parser.add_argument("--output_dir", type=str, default=None, help="Target results directory")
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
        print(f"[MODE] Executing Cleaning Strategy {args.cleaning} ({len(filtered)} pipelines) across {len(active_seeds)} seeds...")
        run_repeated_lopo_experiments(
            pipelines=filtered,
            seeds=active_seeds,
            output_dir=args.output_dir,
            use_gpu=not args.no_gpu
        )
    elif args.top_pipelines:
        top_ids = [
            "P13_C1_S2_B3_M1", "P14_C1_S2_B3_M2", "P05_C1_S1_B3_M1", "P06_C1_S1_B3_M2",
            "P37_C2_S2_B3_M1", "P38_C2_S2_B3_M2", "P45_C2_S3_B3_M1", "P46_C2_S3_B3_M2",
            "P61_C3_S2_B3_M1", "P62_C3_S2_B3_M2", "P53_C3_S1_B3_M1", "P54_C3_S1_B3_M2"
        ]
        selected_top = [p for p in all_specs if p["pipeline_id"] in top_ids[:args.top_pipelines]]
        active_seeds = DEFAULT_SEEDS_10[:args.n_repeats]
        print(f"[MODE] Executing Top {len(selected_top)} pipelines across {len(active_seeds)} seeds...")
        run_repeated_lopo_experiments(
            pipelines=selected_top,
            seeds=active_seeds,
            output_dir=args.output_dir,
            use_gpu=not args.no_gpu
        )
    else:
        active_seeds = DEFAULT_SEEDS_10[:args.n_repeats]
        print(f"[MODE] Executing all {len(all_specs)} pipelines across {len(active_seeds)} seeds...")
        run_repeated_lopo_experiments(
            pipelines=all_specs,
            seeds=active_seeds,
            output_dir=args.output_dir,
            use_gpu=not args.no_gpu
        )
