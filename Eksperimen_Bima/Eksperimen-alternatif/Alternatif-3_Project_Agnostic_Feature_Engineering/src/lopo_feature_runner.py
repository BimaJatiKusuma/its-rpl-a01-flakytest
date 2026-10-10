"""
Engine Eksekusi LOPO-CV untuk Eksperimen Alternatif 3:
Project-Agnostic Feature Engineering & Normalisasi Relatif vs 23 Fitur Asli FlakeFlagger.

Membandingkan 4 Set Fitur:
- Set F1: Original Baseline (23 fitur FlakeFlagger)
- Set F2: Ratios Only (23 fitur asli + 5 rasio relatif baru = 28 fitur)
- Set F3: Project-Agnostic Reformed (5 rasio + 8 test smells + log-transformed size + PCA churn components = 26 fitur)
- Set F4: Feature Selection on F3 (Top-12 fitur dari F3 via ANOVA F-test)

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

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    matthews_corrcoef,
    confusion_matrix,
)
import xgboost as xgb

from data_loader import load_dataset
from feature_engineer import (
    FeatureSetPipeline,
    compute_vif_dataframe,
    CHURN_HINDEX_RAW,
    TEST_SMELLS,
    EXECUTION_COVERAGE_RAW
)
from statistical_tests import run_wilcoxon_analysis
from visualizer import (
    plot_correlation_heatmaps,
    plot_pca_scree,
    plot_metric_boxplots,
    plot_mean_summary_bars,
    plot_feature_importance,
    plot_vif_comparison
)


def evaluate_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    y_preds: np.ndarray
) -> Dict[str, Any]:
    """
    Menghitung metrik evaluasi lengkap dengan penanganan aman untuk proyek tanpa flaky tests (jimfs).
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
        "Specificity": specificity,
        "FPR": fpr,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def create_model(model_name: str, config: Dict[str, Any]):
    """Instansiasi model classifier sesuai konfigurasi."""
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


def run_lopo_feature_engineering(
    config_path: str = None,
    models_to_run: List[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Menjalankan eksperimen LOPO-CV komparasi 4 set fitur pada 24 proyek.
    """
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent

    if config_path is None:
        config_path = base_dir / "configs" / "feature_config.json"
    else:
        config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    results_dir = base_dir / "results"
    raw_preds_dir = results_dir / "raw_predictions"
    plots_dir = results_dir / "plots"

    results_dir.mkdir(parents=True, exist_ok=True)
    raw_preds_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MEMULAI LOPO-CV EXPERIMENT: ALTERNATIF 3 (PROJECT-AGNOSTIC FEATURE ENGINEERING)")
    print("=" * 80)
    print(f"Hardware        : {config.get('hardware_settings', {}).get('gpu_name')}")
    print(f"Feature Sets    : F1_Original, F2_Ratios_Only, F3_Project_Agnostic, F4_Feature_Selection")
    print(f"Data Cleaning   : C1 (Row-Level Drop)")

    # 1. Muat Dataset C1
    df, features, projects = load_dataset(cleaning_strategy="C1")
    n_projects = len(projects)
    total_samples = len(df)
    total_flaky = int(df["flaky"].sum())
    print(f"Total Dataset   : {total_samples:,} baris | {total_flaky:,} flaky ({total_flaky/total_samples*100:.2f}%) | {n_projects} proyek")

    if models_to_run is None:
        models_to_run = ["xgboost_gpu", "random_forest"]

    feature_sets = ["F1_Original", "F2_Ratios_Only", "F3_Project_Agnostic", "F4_Feature_Selection"]

    fold_records = []
    raw_pred_records = []
    f3_feature_importances = []
    pca_loading_records = []
    explained_variance_last = None

    start_total_time = time.time()

    # LOPO-CV Loop
    for fold_idx, target_project in enumerate(projects, 1):
        print(f"\n[Fold {fold_idx:02d}/{n_projects:02d}] Target Project: {target_project:<20}", end="")
        fold_start = time.time()

        train_mask = df["project"] != target_project
        test_mask = df["project"] == target_project

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        y_train = train_df["flaky"].values
        y_test = test_df["flaky"].values

        n_test_flaky = int(np.sum(y_test == 1))
        print(f"| Test Size: {len(test_df):<5} | Flaky: {n_test_flaky:<3} ", end="")

        # Fit Feature Pipeline STRICTLY on train_df
        pipeline = FeatureSetPipeline(
            k_features=config.get("feature_selection", {}).get("k", 12),
            pca_variance_threshold=config.get("pca_settings", {}).get("target_variance", 0.95)
        )
        pipeline.fit(train_df, train_df["flaky"])

        train_features = pipeline.transform(train_df)
        test_features = pipeline.transform(test_df)

        # Simpan informasi PCA dari fold pertama untuk analisis
        if fold_idx == 1:
            explained_variance_last = pipeline.pca_transformer.explained_variance_ratio_
            for comp_idx, comp_name in enumerate(pipeline.pca_transformer.component_names_):
                for f_idx, feat_name in enumerate(pipeline.pca_transformer.feature_names_in_):
                    pca_loading_records.append({
                        "Component": comp_name,
                        "Feature": feat_name,
                        "Loading": round(float(pipeline.pca_transformer.components_[comp_idx, f_idx]), 4),
                        "Explained_Variance_Ratio": round(float(pipeline.pca_transformer.explained_variance_ratio_[comp_idx]), 4)
                    })

        # Loop Model & Feature Sets
        for model_name in models_to_run:
            for f_set in feature_sets:
                X_tr = train_features[f_set].values
                X_te = test_features[f_set].values
                feature_names = train_features[f_set].columns.tolist()

                clf = create_model(model_name, config)

                t0_fit = time.time()
                clf.fit(X_tr, y_train)
                train_time_sec = time.time() - t0_fit

                t0_inf = time.time()
                y_probs = clf.predict_proba(X_te)[:, 1]
                infer_time_sec = time.time() - t0_inf

                # Ekstraksi feature importance XGBoost pada F3
                if model_name == "xgboost_gpu" and f_set == "F3_Project_Agnostic":
                    importances = clf.feature_importances_
                    for fn, imp in zip(feature_names, importances):
                        f3_feature_importances.append({
                            "Target_Project": target_project,
                            "Feature": fn,
                            "Importance": float(imp)
                        })

                # Prediksi biner standar (ambang 0.5)
                y_preds = (y_probs >= 0.5).astype(int)

                # Evaluasi metrik
                metrics = evaluate_metrics(y_test, y_probs, y_preds)

                rec = {
                    "Fold": fold_idx,
                    "Target_Project": target_project,
                    "Model": model_name,
                    "Feature_Set": f_set,
                    "N_Features": len(feature_names),
                    "Train_Time_Sec": round(train_time_sec, 4),
                    "Infer_Time_Sec": round(infer_time_sec, 4),
                    **metrics
                }
                fold_records.append(rec)

                # Simpan log prediksi raw untuk fold pertama & sampel proyek penting
                if target_project in ["spring-boot", "okhttp", "achilles", "jimfs"]:
                    for i in range(len(test_df)):
                        raw_pred_records.append({
                            "Target_Project": target_project,
                            "Model": model_name,
                            "Feature_Set": f_set,
                            "Test_Index": i,
                            "True_Label": int(y_test[i]),
                            "Pred_Prob": round(float(y_probs[i]), 5),
                            "Pred_Label": int(y_preds[i])
                        })

        fold_elapsed = time.time() - fold_start
        print(f"| Selesai ({fold_elapsed:.2f}s)")

    total_time = time.time() - start_total_time
    print("\n" + "=" * 80)
    print(f"EKSEKUSI LOPO-CV SELESAI DALAM {total_time:.2f} DETIK ({total_time/60:.2f} MENIT)!")
    print("=" * 80)

    # 1. Simpan fold_records
    df_folds = pd.DataFrame(fold_records)
    folds_csv_path = results_dir / "lopo_feature_folds.csv"
    df_folds.to_csv(folds_csv_path, index=False)
    print(f"[Hasil] Detail metrik per fold disimpan ke: {folds_csv_path}")

    # 2. Simpan raw predictions sampel
    if raw_pred_records:
        df_raw = pd.DataFrame(raw_pred_records)
        raw_csv_path = raw_preds_dir / "sample_predictions.csv"
        df_raw.to_csv(raw_csv_path, index=False)
        print(f"[Hasil] Log prediksi raw sampel disimpan ke: {raw_csv_path}")

    # 3. Hitung Rekap Mean ± Std Dev
    summary_list = []
    for model_name in models_to_run:
        for f_set in feature_sets:
            sub = df_folds[(df_folds["Model"] == model_name) & (df_folds["Feature_Set"] == f_set)]
            summary_list.append({
                "Model": model_name,
                "Feature_Set": f_set,
                "N_Features": int(sub["N_Features"].iloc[0]),
                "PR_AUC_Mean": round(float(sub["PR_AUC"].dropna().mean()), 4),
                "PR_AUC_Std": round(float(sub["PR_AUC"].dropna().std()), 4),
                "F1_Mean": round(float(sub["F1"].mean()), 4),
                "F1_Std": round(float(sub["F1"].std()), 4),
                "Recall_Mean": round(float(sub["Recall"].dropna().mean()), 4),
                "Recall_Std": round(float(sub["Recall"].dropna().std()), 4),
                "Precision_Mean": round(float(sub["Precision"].mean()), 4),
                "Precision_Std": round(float(sub["Precision"].std()), 4),
                "ROC_AUC_Mean": round(float(sub["ROC_AUC"].dropna().mean()), 4),
                "ROC_AUC_Std": round(float(sub["ROC_AUC"].dropna().std()), 4),
                "MCC_Mean": round(float(sub["MCC"].mean()), 4),
                "MCC_Std": round(float(sub["MCC"].std()), 4),
                "Specificity_Mean": round(float(sub["Specificity"].mean()), 4),
                "FPR_Mean": round(float(sub["FPR"].mean()), 4),
                "Train_Time_Avg_Sec": round(float(sub["Train_Time_Sec"].mean()), 4)
            })

    summary_df = pd.DataFrame(summary_list)
    summary_csv_path = results_dir / "feature_comparison_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"[Hasil] Rekap metrik (Mean +/- Std) disimpan ke: {summary_csv_path}")

    # 4. Simpan Bobot PCA Loadings
    if pca_loading_records:
        df_pca_load = pd.DataFrame(pca_loading_records)
        pca_csv_path = results_dir / "pca_components.csv"
        df_pca_load.to_csv(pca_csv_path, index=False)
        print(f"[Hasil] Bobot komponen PCA disimpan ke: {pca_csv_path}")

    # 5. Analisis VIF Multikolinearitas
    print("\n[VIF] Menghitung Variance Inflation Factor (Sebelum vs Sesudah PCA)...")
    # Pipeline utuh pada full dataset untuk analisis korelasi & VIF
    full_pipeline = FeatureSetPipeline(
        k_features=config.get("feature_selection", {}).get("k", 12),
        pca_variance_threshold=config.get("pca_settings", {}).get("target_variance", 0.95)
    )
    full_pipeline.fit(df, df["flaky"])
    transformed_all = full_pipeline.transform(df)

    vif_raw_hindex = compute_vif_dataframe(df[CHURN_HINDEX_RAW])
    pca_churn_cols = full_pipeline.pca_transformer.component_names_
    vif_pca_churn = compute_vif_dataframe(transformed_all["F3_Project_Agnostic"][pca_churn_cols])

    vif_f1 = compute_vif_dataframe(transformed_all["F1_Original"])
    vif_f3 = compute_vif_dataframe(transformed_all["F3_Project_Agnostic"])

    vif_raw_hindex["Category"] = "Raw_hIndex"
    vif_pca_churn["Category"] = "PCA_Churn"
    vif_f1["Category"] = "F1_Original_All"
    vif_f3["Category"] = "F3_Project_Agnostic_All"

    vif_merged = pd.concat([vif_raw_hindex, vif_pca_churn, vif_f1, vif_f3], ignore_index=True)
    vif_csv_path = results_dir / "vif_comparison.csv"
    vif_merged.to_csv(vif_csv_path, index=False)
    print(f"[Hasil] Komparasi VIF disimpan ke: {vif_csv_path}")

    # 6. Jalankan Uji Statistik Wilcoxon & Cliff's Delta
    wilcoxon_csv_path = results_dir / "wilcoxon_test_results.csv"
    wilcox_dfs = []
    for m_name in models_to_run:
        w_df = run_wilcoxon_analysis(df_folds, None, model_name=m_name)
        wilcox_dfs.append(w_df)
    if wilcox_dfs:
        df_wilcox_all = pd.concat(wilcox_dfs, ignore_index=True)
        df_wilcox_all.to_csv(wilcoxon_csv_path, index=False)
        print(f"[STAT] Hasil uji statistik gabungan berhasil disimpan ke: {wilcoxon_csv_path}")

    # 7. Generate Visualisasi Lengkap
    print("\n[Visualizer] Membangun visualisasi grafik publikasi...")
    # (a) Heatmaps Korelasi
    plot_correlation_heatmaps(
        df[CHURN_HINDEX_RAW],
        transformed_all["F3_Project_Agnostic"][pca_churn_cols],
        plots_dir / "correlation_heatmap_pca.png"
    )
    # (b) PCA Scree Plot
    if explained_variance_last is not None:
        plot_pca_scree(
            explained_variance_last,
            plots_dir / "pca_scree_variance.png"
        )
    # (c) Boxplots Metrik
    plot_metric_boxplots(
        df_folds,
        plots_dir / "metric_boxplots_xgboost.png",
        model_name="xgboost_gpu"
    )
    plot_metric_boxplots(
        df_folds,
        plots_dir / "metric_boxplots_random_forest.png",
        model_name="random_forest"
    )
    # (d) Summary Barplot
    plot_mean_summary_bars(
        summary_df,
        plots_dir / "summary_performance_bars.png"
    )
    # (e) Feature Importance XGBoost F3
    if f3_feature_importances:
        df_imp = pd.DataFrame(f3_feature_importances)
        df_imp_mean = df_imp.groupby("Feature")["Importance"].mean().reset_index()
        plot_feature_importance(
            df_imp_mean,
            plots_dir / "feature_importance_f3.png",
            top_n=15
        )
    # (f) VIF Comparison
    plot_vif_comparison(
        vif_raw_hindex[["Feature", "VIF"]],
        vif_pca_churn[["Feature", "VIF"]],
        plots_dir / "vif_comparison_bars.png"
    )

    # Print Tabel Ringkasan
    print("\n" + "=" * 90)
    print("RINGKASAN HASIL EVALUASI KOMPARASI SET FITUR (LOPO-CV 24 PROYEK)")
    print("=" * 90)
    header = f"{'Model':<15} | {'Feature Set':<22} | {'Features':<8} | {'PR-AUC (Mean±Std)':<18} | {'F1 (Mean±Std)':<18} | {'Recall':<8} | {'Precision':<9} | {'Train (s)':<9}"
    print(header)
    print("-" * len(header))
    for _, r in summary_df.iterrows():
        print(f"{r['Model']:<15} | {r['Feature_Set']:<22} | {r['N_Features']:<8} | "
              f"{r['PR_AUC_Mean']:.4f}±{r['PR_AUC_Std']:.4f}  | {r['F1_Mean']:.4f}±{r['F1_Std']:.4f}  | "
              f"{r['Recall_Mean']:.4f}   | {r['Precision_Mean']:.4f}    | {r['Train_Time_Avg_Sec']:.4f}s")
    print("=" * 90)

    return df_folds, summary_df


if __name__ == "__main__":
    run_lopo_feature_engineering()
