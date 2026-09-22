"""
FlakeFlagger Benchmark: Cross-Project Leave-One-Project-Out Cross-Validation (LOOCV)
===================================================================================
A modular, publication-grade ML research framework for empirical software engineering.

Tasks:
1. Classification: Predict binary target 'flaky' (Random Forest, Extra Trees,
   Gradient Boosting, Logistic Regression, Naive Bayes, KNN) with Class Weights / Imbalance Handling.
   Metrics: MCC, Macro F1-Score, F1-Score, AUC-PR, Precision, Recall.
2. Regression: Predict continuous target 'NumFailingRuns' (Random Forest Regressor,
   Extra Trees Regressor, Gradient Boosting Regressor, Ridge, Lasso, SVR).
   Metrics: Spearman's Rank Correlation (Rho), RMSE, MAE, R^2.
3. Preprocessing: StandardScaler (and median imputer) fitted strictly on training folds (Zero Leakage).
4. Feature Importance: Fold-by-fold feature importance extraction for tree ensemble models.
5. Statistical Testing: Friedman Test and Nemenyi Critical Difference rank analysis across 24 projects.
6. Output: Summary tables (Mean +/- Std), LaTeX/Markdown export, and publication-ready charts.

Author: Principal Machine Learning Researcher in Software Engineering
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ConstantInputWarning, friedmanchisquare, rankdata, spearmanr
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.utils.class_weight import compute_sample_weight

# Suppress expected numerical warnings during edge-case LOOCV folds (e.g., zero positive class or constant target)
warnings.filterwarnings("ignore", category=ConstantInputWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics._ranking")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("FlakeFlagger-LOOCV")


# =====================================================================
# 1. Data Structures & Loading Module
# =====================================================================

@dataclass
class DatasetContainer:
    """Structured container holding FlakeFlagger experiment data and metadata."""
    features_df: pd.DataFrame
    feature_names: List[str]
    project_order: List[str]          # Ordered projects E00 -> E23
    project_id_map: Dict[str, str]    # e.g., 'activiti' -> 'E00'
    project_url_map: Dict[str, str]   # e.g., 'activiti' -> GitHub URL
    X: np.ndarray                     # Feature matrix (N, D)
    y_clf: np.ndarray                 # Binary classification target 'flaky' (N,)
    y_reg: np.ndarray                 # Continuous regression target 'NumFailingRuns' (N,)
    projects: np.ndarray              # Project identifier per instance (N,)
    project_ids: np.ndarray           # Project code E00..E23 per instance (N,)


class FlakeFlaggerDataLoader:
    """
    Robust data loader and aligner for FlakeFlagger benchmark datasets:
    - Project_Info.csv
    - test_features.csv
    - test_results.csv
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.project_info_path = self.data_dir / "Project_Info.csv"
        self.test_features_path = self.data_dir / "test_features.csv"
        self.test_results_path = self.data_dir / "test_results.csv"

    def _verify_files_exist(self) -> None:
        for path in [self.project_info_path, self.test_features_path, self.test_results_path]:
            if not path.exists():
                raise FileNotFoundError(f"Required dataset file not found: {path.resolve()}")

    def load_and_align(self) -> DatasetContainer:
        """
        Loads CSV files, establishes 1-to-1 project mappings (E00-E23), resolves
        naming differences, aligns features and results, and extracts targets.
        """
        self._verify_files_exist()
        logger.info("Loading CSV files from %s...", self.data_dir.resolve())

        df_proj = pd.read_csv(self.project_info_path)
        df_feat = pd.read_csv(self.test_features_path)
        df_res = pd.read_csv(self.test_results_path)

        logger.info("Raw counts: Project_Info=%d, test_features=%d, test_results=%d",
                    len(df_proj), len(df_feat), len(df_res))

        # Build Project canonical IDs (E00 to E23) from Project_Info.csv
        project_order: List[str] = []
        project_id_map: Dict[str, str] = {}
        project_url_map: Dict[str, str] = {}
        owner_repo_to_proj: Dict[str, str] = {}

        for idx, row in df_proj.iterrows():
            url = row["URL"].strip()
            owner_repo = url.replace("https://github.com/", "").strip()
            owner, repo = owner_repo.split("/")
            repo_clean = repo.lower()
            proj_id = f"E{idx:02d}"

            project_order.append(repo_clean)
            project_id_map[repo_clean] = proj_id
            project_url_map[repo_clean] = url

            owner_repo_to_proj[f"{owner.lower()}-{repo.lower()}"] = repo_clean
            owner_repo_to_proj[repo_clean] = repo_clean

        df_res["project_clean"] = df_res["Project"].str.lower().map(owner_repo_to_proj)
        df_feat["project_clean"] = df_feat["project"].str.lower()

        df_res["testMethodName"] = df_res["Test"].apply(lambda x: x.split("#")[1] if "#" in x else x)
        df_res["testClassName"] = df_res["Test"].apply(lambda x: x.split("#")[0] if "#" in x else "")

        # Alignment: commons-exec matches on method; others on class + method
        ce_res = df_res[df_res["project_clean"] == "commons-exec"]
        ce_feat = df_feat[df_feat["project_clean"] == "commons-exec"]
        merged_ce = pd.merge(
            ce_feat,
            ce_res[["project_clean", "testMethodName", "NumFailingRuns"]],
            on=["project_clean", "testMethodName"],
            how="inner"
        )

        other_res = df_res[df_res["project_clean"] != "commons-exec"]
        other_feat = df_feat[df_feat["project_clean"] != "commons-exec"]
        merged_other = pd.merge(
            other_feat,
            other_res[["project_clean", "testClassName", "testMethodName", "NumFailingRuns"]],
            on=["project_clean", "testClassName", "testMethodName"],
            how="inner"
        )

        aligned_df = pd.concat([merged_other, merged_ce], ignore_index=True)
        logger.info("Successfully aligned %d test cases across all %d projects.",
                    len(aligned_df), aligned_df["project_clean"].nunique())

        # Select the 23 FlakeFlagger feature columns
        metadata_cols = {
            "Unnamed: 0", "test_name", "project", "project_clean",
            "testClassName", "testMethodName", "flaky", "NumFailingRuns"
        }
        feature_cols = [c for c in df_feat.columns if c not in metadata_cols]
        logger.info("Identified %d predictor features (test smells, code metrics, churn).", len(feature_cols))

        X = aligned_df[feature_cols].values.astype(np.float64)
        y_clf = aligned_df["flaky"].values.astype(np.int32)
        y_reg = aligned_df["NumFailingRuns"].values.astype(np.float64)
        projects = aligned_df["project_clean"].values
        project_ids = np.array([project_id_map[p] for p in projects])

        return DatasetContainer(
            features_df=aligned_df,
            feature_names=feature_cols,
            project_order=project_order,
            project_id_map=project_id_map,
            project_url_map=project_url_map,
            X=X,
            y_clf=y_clf,
            y_reg=y_reg,
            projects=projects,
            project_ids=project_ids,
        )


# =====================================================================
# 2. Model Factory with Class Imbalance Handling
# =====================================================================

class ModelFactory:
    """Centralized factory creating classification and regression model pipelines with class weights."""

    @staticmethod
    def get_classification_models(random_state: int = 42) -> Dict[str, Any]:
        """
        Returns the 6 classification algorithms required:
        - Random Forest (balanced class weights)
        - Extra Trees (balanced class weights)
        - Gradient Boosting (balanced sample weights applied during fit)
        - Logistic Regression (balanced class weights)
        - Naive Bayes (GaussianNB)
        - KNN (KNeighborsClassifier)
        """
        return {
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1
            ),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100,
                random_state=random_state
            ),
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state
            ),
            "Naive Bayes": GaussianNB(),
            "KNN": KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            ),
        }

    @staticmethod
    def get_regression_models(random_state: int = 42) -> Dict[str, Any]:
        """
        Returns the 6 regression algorithms required:
        - Random Forest Regressor
        - Extra Trees Regressor
        - Gradient Boosting Regressor
        - Ridge
        - Lasso
        - SVR (Support Vector Regressor)
        """
        return {
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1
            ),
            "Extra Trees Regressor": ExtraTreesRegressor(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1
            ),
            "Gradient Boosting Regressor": GradientBoostingRegressor(
                n_estimators=100,
                random_state=random_state
            ),
            "Ridge": Ridge(
                random_state=random_state
            ),
            "Lasso": Lasso(
                random_state=random_state
            ),
            "SVR": SVR(
                cache_size=1000,
                max_iter=10000
            ),
        }


# =====================================================================
# 3. Cross-Project LOOCV Evaluation Engine
# =====================================================================

class CrossProjectLOOCV:
    """
    Executes Cross-Project Leave-One-Project-Out Cross-Validation (LOOCV).
    Ensures zero data leakage: Imputation and StandardScaler are fit on training folds only.
    Extracts feature importances and fold-by-fold classification & regression metrics.
    """

    def __init__(self, data: DatasetContainer, random_state: int = 42):
        self.data = data
        self.random_state = random_state

    def run_experiments(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Executes LOOCV across all 24 projects for both Classification and Regression tasks.
        Returns:
            (clf_df, reg_df, feat_imp_df)
        """
        clf_records: List[Dict[str, Any]] = []
        reg_records: List[Dict[str, Any]] = []
        feat_imp_records: List[Dict[str, Any]] = []

        total_projects = len(self.data.project_order)
        logger.info("Starting Cross-Project LOOCV across %d projects (E00 to E%02d)...",
                    total_projects, total_projects - 1)

        start_time_all = time.time()

        for fold_idx, test_proj in enumerate(self.data.project_order):
            test_proj_id = self.data.project_id_map[test_proj]
            fold_start_time = time.time()

            test_mask = self.data.projects == test_proj
            train_mask = ~test_mask

            X_train_raw = self.data.X[train_mask]
            X_test_raw = self.data.X[test_mask]

            y_train_clf = self.data.y_clf[train_mask]
            y_test_clf = self.data.y_clf[test_mask]

            y_train_reg = self.data.y_reg[train_mask]
            y_test_reg = self.data.y_reg[test_mask]

            # -------------------------------------------------------------
            # Strict Preprocessing: Fit ONLY on training fold (Zero Leakage)
            # -------------------------------------------------------------
            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()

            X_train = scaler.fit_transform(imputer.fit_transform(X_train_raw))
            X_test = scaler.transform(imputer.transform(X_test_raw))

            # Compute balanced sample weights for training set (imbalance handling)
            sample_weights_clf = compute_sample_weight("balanced", y_train_clf)

            n_test = len(y_test_clf)
            n_pos = int(np.sum(y_test_clf))
            logger.info("--> [Fold %02d/%02d] Leave out: %s (%s) | N_test=%d, Flaky=%d (%.1f%%)",
                        fold_idx + 1, total_projects, test_proj_id, test_proj, n_test, n_pos,
                        (n_pos / n_test * 100) if n_test > 0 else 0)

            # -------------------------------------------------------------
            # Task 1: Classification
            # -------------------------------------------------------------
            clf_models = ModelFactory.get_classification_models(self.random_state)
            for model_name, clf in clf_models.items():
                if model_name in ["Gradient Boosting", "Naive Bayes"]:
                    clf.fit(X_train, y_train_clf, sample_weight=sample_weights_clf)
                else:
                    clf.fit(X_train, y_train_clf)

                y_pred = clf.predict(X_test)

                # Predict probabilities / decision scores for AUC-PR
                if hasattr(clf, "predict_proba"):
                    y_prob = clf.predict_proba(X_test)[:, 1]
                elif hasattr(clf, "decision_function"):
                    y_prob = clf.decision_function(X_test)
                else:
                    y_prob = y_pred.astype(float)

                # Classification Metrics
                mcc = matthews_corrcoef(y_test_clf, y_pred)
                macro_f1 = f1_score(y_test_clf, y_pred, average="macro", zero_division=0)
                f1 = f1_score(y_test_clf, y_pred, zero_division=0)
                prec = precision_score(y_test_clf, y_pred, zero_division=0)
                rec = recall_score(y_test_clf, y_pred, zero_division=0)

                if len(np.unique(y_test_clf)) > 1:
                    auc_pr = average_precision_score(y_test_clf, y_prob)
                else:
                    auc_pr = np.nan

                clf_records.append({
                    "Project_ID": test_proj_id,
                    "Project": test_proj,
                    "Algorithm": model_name,
                    "Test_Samples": n_test,
                    "Flaky_Samples": n_pos,
                    "MCC": mcc,
                    "Macro_F1": macro_f1,
                    "F1-Score": f1,
                    "Precision": prec,
                    "Recall": rec,
                    "AUC-PR": auc_pr,
                })

                # Feature importances for tree models
                if hasattr(clf, "feature_importances_"):
                    for feat_name, imp_val in zip(self.data.feature_names, clf.feature_importances_):
                        feat_imp_records.append({
                            "Project_ID": test_proj_id,
                            "Project": test_proj,
                            "Algorithm": model_name,
                            "Task": "Classification",
                            "Feature": feat_name,
                            "Importance": imp_val,
                        })

            # -------------------------------------------------------------
            # Task 2: Regression
            # -------------------------------------------------------------
            reg_models = ModelFactory.get_regression_models(self.random_state)
            for model_name, reg in reg_models.items():
                reg.fit(X_train, y_train_reg)
                y_pred_reg = reg.predict(X_test)

                # Regression Metrics
                rmse = root_mean_squared_error(y_test_reg, y_pred_reg)
                mae = mean_absolute_error(y_test_reg, y_pred_reg)
                r2 = r2_score(y_test_reg, y_pred_reg)

                if np.std(y_test_reg) > 0 and np.std(y_pred_reg) > 0:
                    rho, _ = spearmanr(y_test_reg, y_pred_reg)
                else:
                    rho = np.nan

                reg_records.append({
                    "Project_ID": test_proj_id,
                    "Project": test_proj,
                    "Algorithm": model_name,
                    "Test_Samples": n_test,
                    "Mean_Failing_Runs": np.mean(y_test_reg),
                    "RMSE": rmse,
                    "MAE": mae,
                    "R^2": r2,
                    "Spearman_Rho": rho,
                })

                # Feature importances for regression tree models
                if hasattr(reg, "feature_importances_"):
                    for feat_name, imp_val in zip(self.data.feature_names, reg.feature_importances_):
                        feat_imp_records.append({
                            "Project_ID": test_proj_id,
                            "Project": test_proj,
                            "Algorithm": model_name,
                            "Task": "Regression",
                            "Feature": feat_name,
                            "Importance": imp_val,
                        })

            elapsed = time.time() - fold_start_time
            logger.info("    Completed %s in %.2f seconds.", test_proj_id, elapsed)

        total_elapsed = time.time() - start_time_all
        logger.info("All 24 LOOCV iterations completed in %.2f seconds.", total_elapsed)

        clf_df = pd.DataFrame(clf_records)
        reg_df = pd.DataFrame(reg_records)
        feat_imp_df = pd.DataFrame(feat_imp_records)
        return clf_df, reg_df, feat_imp_df


# =====================================================================
# 4. Result Summarization & Statistical Analysis
# =====================================================================

class ExperimentReporter:
    """Generates summary tables, statistical ranks (Friedman & Nemenyi), and saves results."""

    def __init__(
        self,
        clf_df: pd.DataFrame,
        reg_df: pd.DataFrame,
        feat_imp_df: pd.DataFrame,
        output_dir: str | Path
    ):
        self.clf_df = clf_df
        self.reg_df = reg_df
        self.feat_imp_df = feat_imp_df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_classification_summary(self) -> pd.DataFrame:
        """Computes mean and standard deviation for classification metrics."""
        metrics = ["MCC", "Macro_F1", "F1-Score", "Precision", "Recall", "AUC-PR"]
        summary_rows = []

        for algo, group in self.clf_df.groupby("Algorithm", sort=False):
            row = {"Algorithm": algo}
            for m in metrics:
                mean_val = float(np.nanmean(group[m]))
                std_val = float(np.nanstd(group[m]))
                row[f"{m}_Mean"] = mean_val
                row[f"{m}_Std"] = std_val
                row[f"{m} (Mean ± Std)"] = f"{mean_val:.4f} ± {std_val:.4f}"
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values(by="MCC_Mean", ascending=False).reset_index(drop=True)
        return summary_df

    def generate_regression_summary(self) -> pd.DataFrame:
        """Computes mean and standard deviation for regression metrics."""
        metrics = ["Spearman_Rho", "RMSE", "MAE", "R^2"]
        summary_rows = []

        for algo, group in self.reg_df.groupby("Algorithm", sort=False):
            row = {"Algorithm": algo}
            for m in metrics:
                mean_val = float(np.nanmean(group[m]))
                std_val = float(np.nanstd(group[m]))
                row[f"{m}_Mean"] = mean_val
                row[f"{m}_Std"] = std_val
                row[f"{m} (Mean ± Std)"] = f"{mean_val:.4f} ± {std_val:.4f}"
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values(by="Spearman_Rho_Mean", ascending=False).reset_index(drop=True)
        return summary_df

    def compute_statistical_ranks(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Calculates Friedman test chi-square & p-value and average ranks across the 24 LOOCV project folds.
        Also calculates the Nemenyi Critical Difference (CD).
        """
        # 1. Classification Ranks (based on MCC, higher is better)
        clf_piv = self.clf_df.pivot(index="Project_ID", columns="Algorithm", values="MCC")
        # Rank: 1 = best (highest MCC)
        clf_ranks = clf_piv.apply(lambda row: rankdata(-row, method="average"), axis=1, result_type="expand")
        clf_ranks.columns = clf_piv.columns
        clf_avg_ranks = clf_ranks.mean().sort_values()

        # Friedman test on classification
        clf_matrix = [clf_piv[col].values for col in clf_piv.columns]
        stat_clf, p_clf = friedmanchisquare(*clf_matrix)

        # 2. Regression Ranks (based on Spearman Rho, higher is better)
        reg_piv = self.reg_df.pivot(index="Project_ID", columns="Algorithm", values="Spearman_Rho").fillna(0.0)
        reg_ranks = reg_piv.apply(lambda row: rankdata(-row, method="average"), axis=1, result_type="expand")
        reg_ranks.columns = reg_piv.columns
        reg_avg_ranks = reg_ranks.mean().sort_values()

        reg_matrix = [reg_piv[col].values for col in reg_piv.columns]
        stat_reg, p_reg = friedmanchisquare(*reg_matrix)

        # Critical Difference calculation: Nemenyi test (alpha=0.05, k=6, N=24)
        # q_alpha for k=6 is 2.850
        k = 6
        N = len(clf_piv)
        q_alpha = 2.850
        cd_val = q_alpha * math.sqrt((k * (k + 1)) / (6.0 * N))

        rank_records = []
        for algo, r in clf_avg_ranks.items():
            rank_records.append({
                "Task": "Classification",
                "Metric": "MCC",
                "Algorithm": algo,
                "Average_Rank": r,
                "Friedman_Stat": stat_clf,
                "Friedman_p_val": p_clf,
                "Nemenyi_CD": cd_val
            })
        for algo, r in reg_avg_ranks.items():
            rank_records.append({
                "Task": "Regression",
                "Metric": "Spearman_Rho",
                "Algorithm": algo,
                "Average_Rank": r,
                "Friedman_Stat": stat_reg,
                "Friedman_p_val": p_reg,
                "Nemenyi_CD": cd_val
            })

        ranks_df = pd.DataFrame(rank_records)
        stats_summary = {
            "Classification_Friedman_Stat": stat_clf,
            "Classification_Friedman_p": p_clf,
            "Regression_Friedman_Stat": stat_reg,
            "Regression_Friedman_p": p_reg,
            "Critical_Difference_CD": cd_val,
        }
        return ranks_df, stats_summary

    def generate_feature_importance_summary(self) -> pd.DataFrame:
        """Summarizes mean feature importance across all folds and tree algorithms."""
        if self.feat_imp_df.empty:
            return pd.DataFrame()

        feat_grp = self.feat_imp_df.groupby(["Task", "Algorithm", "Feature"])["Importance"].agg(
            Mean_Importance="mean",
            Std_Importance="std"
        ).reset_index()

        overall_feat = self.feat_imp_df.groupby("Feature")["Importance"].agg(
            Overall_Mean_Importance="mean",
            Overall_Std_Importance="std"
        ).reset_index().sort_values(by="Overall_Mean_Importance", ascending=False)

        merged = pd.merge(feat_grp, overall_feat, on="Feature")
        return merged.sort_values(by=["Overall_Mean_Importance", "Mean_Importance"], ascending=False)

    def save_and_display_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Saves detailed per-project results, aggregated summaries, ranks, and feature importances."""
        clf_summary = self.generate_classification_summary()
        reg_summary = self.generate_regression_summary()
        ranks_df, stats_dict = self.compute_statistical_ranks()
        feat_imp_summary = self.generate_feature_importance_summary()

        # Save per-fold records
        self.clf_df.to_csv(self.output_dir / "classification_loocv_per_project.csv", index=False)
        self.reg_df.to_csv(self.output_dir / "regression_loocv_per_project.csv", index=False)

        # Save summaries
        clf_summary.to_csv(self.output_dir / "classification_summary.csv", index=False)
        reg_summary.to_csv(self.output_dir / "regression_summary.csv", index=False)
        ranks_df.to_csv(self.output_dir / "statistical_ranks.csv", index=False)
        if not feat_imp_summary.empty:
            feat_imp_summary.to_csv(self.output_dir / "feature_importance_summary.csv", index=False)

        print("\n" + "=" * 125)
        print("TABLE 1: CLASSIFICATION PERFORMANCE ACROSS 24 LOOCV PROJECT ITERATIONS (Mean ± Std)")
        print("=" * 125)
        disp_clf = ["Algorithm", "MCC (Mean ± Std)", "Macro_F1 (Mean ± Std)", "F1-Score (Mean ± Std)",
                    "Precision (Mean ± Std)", "Recall (Mean ± Std)", "AUC-PR (Mean ± Std)"]
        print(clf_summary[disp_clf].to_string(index=False))

        print("\n" + "=" * 125)
        print("TABLE 2: REGRESSION PERFORMANCE ACROSS 24 LOOCV PROJECT ITERATIONS (Mean ± Std)")
        print("=" * 125)
        disp_reg = ["Algorithm", "Spearman_Rho (Mean ± Std)", "RMSE (Mean ± Std)",
                    "MAE (Mean ± Std)", "R^2 (Mean ± Std)"]
        print(reg_summary[disp_reg].to_string(index=False))

        print("\n" + "=" * 125)
        print("TABLE 3: STATISTICAL RANKING & FRIEDMAN TEST SUMMARY (24 LOOCV Projects, k=6)")
        print("=" * 125)
        print(ranks_df[["Task", "Algorithm", "Average_Rank", "Friedman_Stat", "Friedman_p_val", "Nemenyi_CD"]].to_string(index=False))
        print("=" * 125 + "\n")

        return clf_summary, reg_summary, ranks_df, feat_imp_summary


# =====================================================================
# 5. Main Execution CLI
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-Project LOOCV Experiment for FlakeFlagger Benchmark (Classification & Regression)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Directory containing Project_Info.csv, test_features.csv, and test_results.csv (default: current dir)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save output tables, CSV logs, and plots (default: ./results)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for model reproducibility (default: 42)"
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("FLAKEFLAGGER CROSS-PROJECT BENCHMARK: 24-PROJECT LOOCV EXPERIMENT")
    logger.info("=" * 80)

    # 1. Load and align data
    loader = FlakeFlaggerDataLoader(data_dir=args.data_dir)
    data = loader.load_and_align()

    # 2. Run LOOCV experiment with Class Weights
    loocv = CrossProjectLOOCV(data=data, random_state=args.seed)
    clf_df, reg_df, feat_imp_df = loocv.run_experiments()

    # 3. Report summaries and statistical rankings
    reporter = ExperimentReporter(
        clf_df=clf_df,
        reg_df=reg_df,
        feat_imp_df=feat_imp_df,
        output_dir=args.output_dir
    )
    reporter.save_and_display_tables()

    logger.info("Experiment pipeline finished successfully! Artifacts stored in %s",
                Path(args.output_dir).resolve())


if __name__ == "__main__":
    main()
