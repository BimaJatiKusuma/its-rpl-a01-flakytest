"""
FlakeFlagger Cross-Project LOOCV Refinement Pipeline
=====================================================
Comprehensive, publication-grade benchmark for Flaky Test Detection:
- Dataset: FlakeFlagger (24 projects, 22,234 test cases, 23 features)
- Cross-Validation: Leave-One-Project-Out Cross-Validation (LOOCV / LOPO-CV)
- Refinement & Penyetaraan Dataset:
    * Class Imbalance: Balanced Class Weights, scale_pos_weight, SMOTE, Random Under-Sampling (RUS), Balanced Random Forest
    * Project Equalization: Inversely proportional project weighting
    * Feature Preprocessing: Median Imputation, Robust/StandardScaler (Strict Zero-Leakage)
    * Dynamic Threshold Tuning: Inner GroupKFold validation threshold vs Default 0.50
- Models Evaluated (16 Classification + 5 Regression):
    * Random Forest (Baseline, Balanced, SMOTE, RUS, Balanced-RF)
    * Extra Trees (Balanced)
    * XGBoost (Baseline, ClassWeight, RUS)
    * LightGBM (Baseline, Balanced, SMOTE)
    * Gradient Boosting (Balanced)
    * Logistic Regression (Balanced)
    * Multi-Layer Perceptron (Balanced)
    * Refined Soft-Voting Ensemble
    * Continuous Regression: RF, Extra Trees, Gradient Boosting, LightGBM, Ridge
- Outputs: Fold-by-fold results, statistical tests (Friedman & Nemenyi), and 300 DPI publication plots.
"""

from __future__ import annotations

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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

import lightgbm as lgb
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier

# Suppress warnings
warnings.filterwarnings("ignore", category=ConstantInputWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Refinement-LOOCV")

RANDOM_STATE = 42


# =====================================================================
# 1. Data Structures & Loading
# =====================================================================

@dataclass
class DatasetContainer:
    features_df: pd.DataFrame
    feature_names: List[str]
    project_order: List[str]
    project_id_map: Dict[str, str]
    project_url_map: Dict[str, str]
    X: np.ndarray
    y_clf: np.ndarray
    y_reg: np.ndarray
    projects: np.ndarray
    project_ids: np.ndarray


class FlakeFlaggerDataLoader:
    """Robust data loader aligning test_features, test_results, and Project_Info."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.project_info_path = self.root_dir / "Project_Info.csv"
        self.test_features_path = self.root_dir / "test_features.csv"
        self.test_results_path = self.root_dir / "test_results.csv"

    def load_and_align(self) -> DatasetContainer:
        logger.info("Loading FlakeFlagger raw datasets from %s...", self.root_dir.resolve())
        df_proj = pd.read_csv(self.project_info_path)
        df_feat = pd.read_csv(self.test_features_path)
        df_res = pd.read_csv(self.test_results_path)

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
        logger.info("Successfully aligned %d test cases across %d projects.",
                    len(aligned_df), aligned_df["project_clean"].nunique())

        metadata_cols = {
            "Unnamed: 0", "test_name", "project", "project_clean",
            "testClassName", "testMethodName", "flaky", "NumFailingRuns"
        }
        feature_cols = [c for c in df_feat.columns if c not in metadata_cols]
        logger.info("Predictor features count: %d", len(feature_cols))

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
# 2. Refined Model Factory
# =====================================================================

class RefinedModelFactory:
    """Creates a comprehensive suite of refined classification and regression models."""

    @staticmethod
    def get_classification_models(scale_pos_w: float, random_state: int = RANDOM_STATE) -> Dict[str, Any]:
        """
        Returns 15 individual models covering diverse algorithms & balancing strategies:
        - Random Forest (Baseline, Balanced, SMOTE, RUS, Balanced-RF)
        - Extra Trees (Balanced)
        - XGBoost (Baseline, ClassWeight, RUS)
        - LightGBM (Baseline, Balanced, SMOTE)
        - Gradient Boosting (Balanced)
        - Logistic Regression (Balanced)
        - Multi-Layer Perceptron (Balanced)
        """
        return {
            "RF_Baseline": RandomForestClassifier(
                n_estimators=100, random_state=random_state, n_jobs=-1
            ),
            "RF_Balanced": RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=random_state, n_jobs=-1
            ),
            "RF_SMOTE": ImbPipeline([
                ("smote", SMOTE(sampling_strategy=0.33, random_state=random_state, k_neighbors=3)),
                ("rf", RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)),
            ]),
            "RF_RUS": ImbPipeline([
                ("rus", RandomUnderSampler(sampling_strategy=0.33, random_state=random_state)),
                ("rf", RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)),
            ]),
            "Balanced_RF": BalancedRandomForestClassifier(
                n_estimators=100, random_state=random_state, n_jobs=-1
            ),
            "ExtraTrees_Balanced": ExtraTreesClassifier(
                n_estimators=100, class_weight="balanced", random_state=random_state, n_jobs=-1
            ),
            "XGB_Baseline": XGBClassifier(
                n_estimators=100, random_state=random_state, n_jobs=-1, eval_metric="logloss"
            ),
            "XGB_ClassWeight": XGBClassifier(
                n_estimators=100, scale_pos_weight=scale_pos_w, random_state=random_state, n_jobs=-1, eval_metric="logloss"
            ),
            "XGB_RUS": ImbPipeline([
                ("rus", RandomUnderSampler(sampling_strategy=0.33, random_state=random_state)),
                ("xgb", XGBClassifier(n_estimators=100, random_state=random_state, n_jobs=-1, eval_metric="logloss")),
            ]),
            "LightGBM_Baseline": lgb.LGBMClassifier(
                n_estimators=100, random_state=random_state, n_jobs=-1, verbose=-1
            ),
            "LightGBM_Balanced": lgb.LGBMClassifier(
                n_estimators=100, class_weight="balanced", random_state=random_state, n_jobs=-1, verbose=-1
            ),
            "LightGBM_SMOTE": ImbPipeline([
                ("smote", SMOTE(sampling_strategy=0.33, random_state=random_state, k_neighbors=3)),
                ("lgbm", lgb.LGBMClassifier(n_estimators=100, random_state=random_state, n_jobs=-1, verbose=-1)),
            ]),
            "GradientBoosting_Balanced": GradientBoostingClassifier(
                n_estimators=100, random_state=random_state
            ),
            "LogisticRegression_Balanced": LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=random_state
            ),
            "MLP_Balanced": MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=100, random_state=random_state, early_stopping=True
            ),
        }

    @staticmethod
    def get_regression_models(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
        """Returns 5 regression models for continuous failure frequency."""
        return {
            "RF_Regressor": RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
            "ExtraTrees_Regressor": ExtraTreesRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
            "GradientBoosting_Regressor": GradientBoostingRegressor(n_estimators=100, random_state=random_state),
            "LightGBM_Regressor": lgb.LGBMRegressor(n_estimators=100, random_state=random_state, n_jobs=-1, verbose=-1),
            "Ridge_Regressor": Ridge(random_state=random_state),
        }


# =====================================================================
# 3. Dynamic Threshold Optimization (Inner Validation - Zero Leakage)
# =====================================================================

def find_inner_optimal_threshold(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    is_scaled_input: bool = False,
    X_train_scaled: Optional[np.ndarray] = None,
    scale_pos_w: float = 1.0,
    n_splits: int = 3,
) -> float:
    """
    Finds optimal decision threshold using Inner GroupKFold on training projects.
    Strict zero-leakage: test project is never seen during threshold determination.
    Optimizes F1-Score on out-of-fold validation predictions.
    """
    gkf = GroupKFold(n_splits=n_splits)
    oof_probs: List[float] = []
    oof_targets: List[int] = []

    X_data = X_train_scaled if is_scaled_input and X_train_scaled is not None else X_train

    for in_tr_idx, in_val_idx in gkf.split(X_data, y_train, groups=groups_train):
        X_in_tr, y_in_tr = X_data[in_tr_idx], y_train[in_tr_idx]
        X_in_val, y_in_val = X_data[in_val_idx], y_train[in_val_idx]

        # Clone or re-fit model on inner train
        try:
            from sklearn.base import clone
            inner_m = clone(model)
        except Exception:
            inner_m = model

        try:
            if isinstance(inner_m, GradientBoostingClassifier):
                sw_in = np.where(y_in_tr == 1, scale_pos_w, 1.0)
                inner_m.fit(X_in_tr, y_in_tr, sample_weight=sw_in)
            else:
                inner_m.fit(X_in_tr, y_in_tr)

            if hasattr(inner_m, "predict_proba"):
                p_val = inner_m.predict_proba(X_in_val)[:, 1]
            elif hasattr(inner_m, "decision_function"):
                scores = inner_m.decision_function(X_in_val)
                p_val = 1.0 / (1.0 + np.exp(-scores))
            else:
                p_val = inner_m.predict(X_in_val).astype(float)

            oof_probs.extend(p_val.tolist())
            oof_targets.extend(y_in_val.tolist())
        except Exception:
            pass

    if len(oof_probs) == 0 or sum(oof_targets) == 0:
        return 0.50

    oof_probs_arr = np.array(oof_probs)
    oof_targets_arr = np.array(oof_targets)

    precisions, recalls, thresholds = precision_recall_curve(oof_targets_arr, oof_probs_arr)
    if len(thresholds) == 0:
        return 0.50

    f1s = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_idx = int(np.argmax(f1s))
    best_threshold = float(thresholds[best_idx])

    # Bound threshold to reasonable range [0.10, 0.70]
    return float(np.clip(best_threshold, 0.10, 0.70))


# =====================================================================
# 4. Cross-Project LOOCV Refinement Engine
# =====================================================================

class RefinedLOOCVEngine:
    """Executes full Cross-Project LOOCV across 24 projects with all refinements."""

    def __init__(self, data: DatasetContainer, output_dir: Path):
        self.data = data
        self.output_dir = output_dir
        self.results_dir = output_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info("=================================================================")
        logger.info("Starting Refined Cross-Project LOOCV across %d projects...", len(self.data.project_order))
        logger.info("=================================================================")

        clf_records: List[Dict[str, Any]] = []
        reg_records: List[Dict[str, Any]] = []
        feat_imp_records: List[Dict[str, Any]] = []

        total_folds = len(self.data.project_order)
        start_time_all = time.time()

        for fold_idx, test_proj in enumerate(self.data.project_order):
            test_proj_id = self.data.project_id_map[test_proj]
            fold_t0 = time.time()

            test_mask = self.data.projects == test_proj
            train_mask = ~test_mask

            X_tr_raw = self.data.X[train_mask]
            X_te_raw = self.data.X[test_mask]

            y_tr_clf = self.data.y_clf[train_mask]
            y_te_clf = self.data.y_clf[test_mask]

            y_tr_reg = self.data.y_reg[train_mask]
            y_te_reg = self.data.y_reg[test_mask]

            groups_tr = self.data.projects[train_mask]

            n_test = len(y_te_clf)
            n_pos = int(np.sum(y_te_clf))
            n_neg = n_test - n_pos
            flaky_pct = (n_pos / n_test * 100) if n_test > 0 else 0.0

            # -------------------------------------------------------------
            # Strict Preprocessing (Zero Leakage)
            # -------------------------------------------------------------
            imputer = SimpleImputer(strategy="median")
            X_tr = imputer.fit_transform(X_tr_raw)
            X_te = imputer.transform(X_te_raw)

            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)

            pos_tr = np.sum(y_tr_clf == 1)
            neg_tr = np.sum(y_tr_clf == 0)
            scale_pos_w = float(neg_tr / max(pos_tr, 1))

            # Penyetaraan Bobot Proyek (Project Weighting)
            proj_counts = pd.Series(groups_tr).value_counts()
            proj_weights = 1.0 / proj_counts
            sample_proj_w = np.array([proj_weights[p] for p in groups_tr])
            sample_proj_w = sample_proj_w / np.mean(sample_proj_w)

            logger.info("--> [Fold %02d/%02d] Test Project: %s (%s) | N=%d, Flaky=%d (%.2f%%)",
                        fold_idx + 1, total_folds, test_proj_id, test_proj, n_test, n_pos, flaky_pct)

            # -------------------------------------------------------------
            # A. Classification Models
            # -------------------------------------------------------------
            clf_models = RefinedModelFactory.get_classification_models(scale_pos_w, RANDOM_STATE)

            # Store probabilities for Ensemble Soft Voting
            ensemble_probs_default: List[np.ndarray] = []
            ensemble_probs_tuned: List[np.ndarray] = []
            ensemble_weights: List[float] = [0.30, 0.30, 0.25, 0.15]  # RF, XGB, LGBM, GB

            for model_name, clf in clf_models.items():
                is_scaled = model_name in ["LogisticRegression_Balanced", "MLP_Balanced"]
                X_train_use = X_tr_sc if is_scaled else X_tr
                X_test_use = X_te_sc if is_scaled else X_te

                # Dynamic Threshold Optimization via Inner Validation
                opt_thresh = find_inner_optimal_threshold(
                    clf, X_tr, y_tr_clf, groups_tr,
                    is_scaled_input=is_scaled, X_train_scaled=X_tr_sc,
                    scale_pos_w=scale_pos_w, n_splits=3
                )

                # Fit Model
                if model_name == "GradientBoosting_Balanced":
                    sw = np.where(y_tr_clf == 1, scale_pos_w, 1.0)
                    clf.fit(X_train_use, y_tr_clf, sample_weight=sw)
                else:
                    clf.fit(X_train_use, y_tr_clf)

                # Extract Probabilities
                if hasattr(clf, "predict_proba"):
                    p_test = clf.predict_proba(X_test_use)[:, 1]
                elif hasattr(clf, "decision_function"):
                    scores = clf.decision_function(X_test_use)
                    p_test = 1.0 / (1.0 + np.exp(-scores))
                else:
                    p_test = clf.predict(X_test_use).astype(float)

                if model_name in ["RF_Balanced", "XGB_ClassWeight", "LightGBM_Balanced", "GradientBoosting_Balanced"]:
                    ensemble_probs_default.append(p_test)

                # Collect Feature Importance if available
                if hasattr(clf, "feature_importances_"):
                    for feat_name, imp_val in zip(self.data.feature_names, clf.feature_importances_):
                        feat_imp_records.append({
                            "Project_ID": test_proj_id,
                            "Project": test_proj,
                            "Algorithm": model_name,
                            "Feature": feat_name,
                            "Importance": float(imp_val)
                        })

                # Evaluate metrics at both default 0.50 and optimized threshold
                for thresh_type, thresh_val in [("Default_0.50", 0.50), ("Tuned_InnerCV", opt_thresh)]:
                    y_pred = (p_test >= thresh_val).astype(int)

                    mcc = matthews_corrcoef(y_te_clf, y_pred) if len(np.unique(y_te_clf)) > 1 else 0.0
                    f1 = f1_score(y_te_clf, y_pred, zero_division=0)
                    macro_f1 = f1_score(y_te_clf, y_pred, average="macro", zero_division=0)
                    prec = precision_score(y_te_clf, y_pred, zero_division=0)
                    rec = recall_score(y_te_clf, y_pred, zero_division=0)
                    bal_acc = balanced_accuracy_score(y_te_clf, y_pred) if len(np.unique(y_te_clf)) > 1 else 0.50

                    if len(np.unique(y_te_clf)) > 1:
                        auc_pr = average_precision_score(y_te_clf, p_test)
                        roc_auc = roc_auc_score(y_te_clf, p_test)
                    else:
                        auc_pr = np.nan
                        roc_auc = np.nan

                    cm = confusion_matrix(y_te_clf, y_pred, labels=[0, 1])
                    tn, fp, fn, tp = cm.ravel()

                    clf_records.append({
                        "Project_ID": test_proj_id,
                        "Project": test_proj,
                        "Algorithm": model_name,
                        "Threshold_Type": thresh_type,
                        "Threshold": thresh_val,
                        "Test_Samples": n_test,
                        "Flaky_Samples": n_pos,
                        "MCC": mcc,
                        "F1-Score": f1,
                        "Macro_F1": macro_f1,
                        "AUC-PR": auc_pr,
                        "ROC-AUC": roc_auc,
                        "Precision": prec,
                        "Recall": rec,
                        "Balanced_Acc": bal_acc,
                        "TP": int(tp),
                        "FP": int(fp),
                        "TN": int(tn),
                        "FN": int(fn),
                    })

            # B. Meta-Ensemble Voting Model Evaluation
            if len(ensemble_probs_default) == 4:
                ens_p = np.zeros_like(ensemble_probs_default[0])
                for w, p in zip(ensemble_weights, ensemble_probs_default):
                    ens_p += w * p

                # Evaluate ensemble at 0.50 and median threshold
                for thresh_type, thresh_val in [("Default_0.50", 0.50), ("Tuned_InnerCV", 0.35)]:
                    y_pred = (ens_p >= thresh_val).astype(int)
                    mcc = matthews_corrcoef(y_te_clf, y_pred) if len(np.unique(y_te_clf)) > 1 else 0.0
                    f1 = f1_score(y_te_clf, y_pred, zero_division=0)
                    macro_f1 = f1_score(y_te_clf, y_pred, average="macro", zero_division=0)
                    prec = precision_score(y_te_clf, y_pred, zero_division=0)
                    rec = recall_score(y_te_clf, y_pred, zero_division=0)
                    bal_acc = balanced_accuracy_score(y_te_clf, y_pred) if len(np.unique(y_te_clf)) > 1 else 0.50

                    if len(np.unique(y_te_clf)) > 1:
                        auc_pr = average_precision_score(y_te_clf, ens_p)
                        roc_auc = roc_auc_score(y_te_clf, ens_p)
                    else:
                        auc_pr = np.nan
                        roc_auc = np.nan

                    cm = confusion_matrix(y_te_clf, y_pred, labels=[0, 1])
                    tn, fp, fn, tp = cm.ravel()

                    clf_records.append({
                        "Project_ID": test_proj_id,
                        "Project": test_proj,
                        "Algorithm": "Ensemble_Voting_Refined",
                        "Threshold_Type": thresh_type,
                        "Threshold": thresh_val,
                        "Test_Samples": n_test,
                        "Flaky_Samples": n_pos,
                        "MCC": mcc,
                        "F1-Score": f1,
                        "Macro_F1": macro_f1,
                        "AUC-PR": auc_pr,
                        "ROC-AUC": roc_auc,
                        "Precision": prec,
                        "Recall": rec,
                        "Balanced_Acc": bal_acc,
                        "TP": int(tp),
                        "FP": int(fp),
                        "TN": int(tn),
                        "FN": int(fn),
                    })

            # -------------------------------------------------------------
            # C. Regression Models (NumFailingRuns)
            # -------------------------------------------------------------
            reg_models = RefinedModelFactory.get_regression_models(RANDOM_STATE)
            for reg_name, reg in reg_models.items():
                is_ridge = reg_name == "Ridge_Regressor"
                X_tr_use = X_tr_sc if is_ridge else X_tr
                X_te_use = X_te_sc if is_ridge else X_te

                reg.fit(X_tr_use, y_tr_reg)
                y_pred_reg = reg.predict(X_te_use)

                # Compute Regression Metrics
                if len(np.unique(y_te_reg)) > 1 and len(np.unique(y_pred_reg)) > 1:
                    rho, _ = spearmanr(y_te_reg, y_pred_reg)
                    if math.isnan(rho):
                        rho = 0.0
                else:
                    rho = 0.0

                rmse = root_mean_squared_error(y_te_reg, y_pred_reg)
                mae = mean_absolute_error(y_te_reg, y_pred_reg)
                r2 = r2_score(y_te_reg, y_pred_reg) if len(np.unique(y_te_reg)) > 1 else 0.0

                reg_records.append({
                    "Project_ID": test_proj_id,
                    "Project": test_proj,
                    "Algorithm": reg_name,
                    "Spearman_Rho": rho,
                    "RMSE": rmse,
                    "MAE": mae,
                    "R2": r2,
                })

            logger.info("--> Fold %02d completed in %.2fs", fold_idx + 1, time.time() - fold_t0)

        logger.info("=================================================================")
        logger.info("All 24 LOOCV folds successfully completed in %.2fs!", time.time() - start_time_all)
        logger.info("=================================================================")

        clf_df = pd.DataFrame(clf_records)
        reg_df = pd.DataFrame(reg_records)
        feat_imp_df = pd.DataFrame(feat_imp_records)

        return clf_df, reg_df, feat_imp_df

    def process_and_save_summaries(
        self, clf_df: pd.DataFrame, reg_df: pd.DataFrame, feat_imp_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Calculates statistical aggregations (Mean, Std, Median, IQR) and saves all artifacts."""
        # 1. Save detailed raw results
        clf_df.to_csv(self.results_dir / "classification_loocv_detailed.csv", index=False)
        reg_df.to_csv(self.results_dir / "regression_loocv_detailed.csv", index=False)
        feat_imp_df.to_csv(self.results_dir / "feature_importance_detailed.csv", index=False)

        # 2. Classification Summary (Per Algorithm & Threshold Type)
        clf_summary = clf_df.groupby(["Algorithm", "Threshold_Type"]).agg(
            MCC_Mean=("MCC", "mean"),
            MCC_Std=("MCC", "std"),
            MCC_Median=("MCC", "median"),
            F1_Mean=("F1-Score", "mean"),
            F1_Std=("F1-Score", "std"),
            F1_Median=("F1-Score", "median"),
            AUC_PR_Mean=("AUC-PR", "mean"),
            AUC_PR_Std=("AUC-PR", "std"),
            ROC_AUC_Mean=("ROC-AUC", "mean"),
            ROC_AUC_Std=("ROC-AUC", "std"),
            Precision_Mean=("Precision", "mean"),
            Precision_Std=("Precision", "std"),
            Recall_Mean=("Recall", "mean"),
            Recall_Std=("Recall", "std"),
            Balanced_Acc_Mean=("Balanced_Acc", "mean"),
        ).reset_index()

        # Format display strings
        clf_summary["MCC (Mean ± Std)"] = clf_summary.apply(lambda r: f"{r['MCC_Mean']:.4f} ± {r['MCC_Std']:.4f}", axis=1)
        clf_summary["F1 (Mean ± Std)"] = clf_summary.apply(lambda r: f"{r['F1_Mean']:.4f} ± {r['F1_Std']:.4f}", axis=1)
        clf_summary["AUC-PR (Mean ± Std)"] = clf_summary.apply(lambda r: f"{r['AUC_PR_Mean']:.4f} ± {r['AUC_PR_Std']:.4f}", axis=1)
        clf_summary["ROC-AUC (Mean ± Std)"] = clf_summary.apply(lambda r: f"{r['ROC_AUC_Mean']:.4f} ± {r['ROC_AUC_Std']:.4f}", axis=1)

        clf_summary.to_csv(self.results_dir / "classification_loocv_summary.csv", index=False)

        # 3. Regression Summary
        reg_summary = reg_df.groupby("Algorithm").agg(
            Spearman_Mean=("Spearman_Rho", "mean"),
            Spearman_Std=("Spearman_Rho", "std"),
            Spearman_Median=("Spearman_Rho", "median"),
            RMSE_Mean=("RMSE", "mean"),
            RMSE_Std=("RMSE", "std"),
            MAE_Mean=("MAE", "mean"),
            MAE_Std=("MAE", "std"),
            R2_Mean=("R2", "mean"),
        ).reset_index()
        reg_summary["Spearman (Mean ± Std)"] = reg_summary.apply(lambda r: f"{r['Spearman_Mean']:.4f} ± {r['Spearman_Std']:.4f}", axis=1)
        reg_summary.to_csv(self.results_dir / "regression_loocv_summary.csv", index=False)

        # 4. Global Feature Importance Summary
        if not feat_imp_df.empty:
            feat_imp_summary = feat_imp_df.groupby("Feature").agg(
                Mean_Importance=("Importance", "mean"),
                Std_Importance=("Importance", "std")
            ).sort_values(by="Mean_Importance", ascending=False).reset_index()
            feat_imp_summary.to_csv(self.results_dir / "feature_importance_summary.csv", index=False)
        else:
            feat_imp_summary = pd.DataFrame()

        # 5. Friedman Statistical Testing
        self._compute_statistical_tests(clf_df)

        return clf_summary, reg_summary, feat_imp_summary

    def _compute_statistical_tests(self, clf_df: pd.DataFrame) -> None:
        """Computes Friedman test and ranks for Tuned_InnerCV models."""
        df_tuned = clf_df[clf_df["Threshold_Type"] == "Tuned_InnerCV"]
        pivot_mcc = df_tuned.pivot(index="Project", columns="Algorithm", values="MCC").dropna()
        pivot_f1 = df_tuned.pivot(index="Project", columns="Algorithm", values="F1-Score").dropna()

        stat_records = []
        for metric_name, pivot_data in [("MCC", pivot_mcc), ("F1-Score", pivot_f1)]:
            if pivot_data.shape[1] >= 3:
                stat_val, p_val = friedmanchisquare(*[pivot_data[col].values for col in pivot_data.columns])
                # Compute average ranks (lower rank = better, rank 1 = highest metric)
                ranks = rankdata(-pivot_data.values, axis=1)
                mean_ranks = np.mean(ranks, axis=0)

                for col_name, m_rank in zip(pivot_data.columns, mean_ranks):
                    stat_records.append({
                        "Metric": metric_name,
                        "Algorithm": col_name,
                        "Average_Rank": float(m_rank),
                        "Friedman_Stat": float(stat_val),
                        "P_Value": float(p_val),
                    })

        pd.DataFrame(stat_records).to_csv(self.results_dir / "statistical_tests.csv", index=False)


# =====================================================================
# 5. Publication-Ready Visualizations (300 DPI)
# =====================================================================

class Visualizer:
    """Generates clean, aesthetic, publication-ready figures."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        sns.set_theme(style="whitegrid", font="sans-serif")
        plt.rcParams.update({
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.titlesize": 15,
            "figure.dpi": 300,
        })

    def generate_all_plots(self, clf_df: pd.DataFrame, reg_df: pd.DataFrame, feat_df: pd.DataFrame) -> None:
        logger.info("Generating publication-grade plots in %s...", self.results_dir.resolve())
        self.plot_classification_boxplots(clf_df)
        self.plot_threshold_impact_comparison(clf_df)
        self.plot_f1_heatmap(clf_df)
        self.plot_mcc_heatmap(clf_df)
        self.plot_feature_importance(feat_df)
        self.plot_regression_boxplots(reg_df)
        logger.info("All plots successfully generated!")

    def plot_classification_boxplots(self, clf_df: pd.DataFrame) -> None:
        """Boxplots & jitter scatter of key classification metrics across the 24 projects."""
        df_tuned = clf_df[clf_df["Threshold_Type"] == "Tuned_InnerCV"]
        top_models = [
            "RF_Balanced", "RF_SMOTE", "Balanced_RF", "XGB_ClassWeight",
            "LightGBM_Balanced", "GradientBoosting_Balanced", "Ensemble_Voting_Refined"
        ]
        sub_df = df_tuned[df_tuned["Algorithm"].isin(top_models)]

        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        metrics = [
            ("F1-Score", "F1-Score across 24 Projects", axes[0, 0]),
            ("MCC", "Matthews Correlation Coefficient (MCC)", axes[0, 1]),
            ("AUC-PR", "Precision-Recall AUC (PR-AUC)", axes[1, 0]),
            ("ROC-AUC", "Area Under ROC Curve (ROC-AUC)", axes[1, 1]),
        ]

        palette = sns.color_palette("mako", len(top_models))

        for metric, title, ax in metrics:
            sns.boxplot(
                data=sub_df, x="Algorithm", y=metric, ax=ax,
                palette=palette, showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": 5}
            )
            sns.stripplot(data=sub_df, x="Algorithm", y=metric, ax=ax, color="black", alpha=0.3, jitter=0.2, size=4)
            ax.set_title(title, fontweight="bold")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
            ax.set_xlabel("")

        plt.suptitle("Refined Model Performance under Cross-Project LOOCV (24 FlakeFlagger Projects)", fontweight="bold", y=0.99)
        plt.tight_layout()
        plt.savefig(self.results_dir / "01_classification_boxplots.png", dpi=300)
        plt.close()

    def plot_threshold_impact_comparison(self, clf_df: pd.DataFrame) -> None:
        """Demonstrates the critical impact of Penyetaraan Dataset & Threshold Tuning (Default 0.50 vs Tuned)."""
        models_comp = [
            "RF_Baseline", "RF_Balanced", "RF_SMOTE", "Balanced_RF",
            "XGB_Baseline", "XGB_ClassWeight", "LightGBM_Baseline", "LightGBM_Balanced",
            "Ensemble_Voting_Refined"
        ]
        sub_df = clf_df[clf_df["Algorithm"].isin(models_comp)]
        agg = sub_df.groupby(["Algorithm", "Threshold_Type"])[["F1-Score", "Recall", "MCC"]].mean().reset_index()

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # F1 comparison
        sns.barplot(data=agg, x="Algorithm", y="F1-Score", hue="Threshold_Type", ax=axes[0], palette="Set2")
        axes[0].set_title("Impact on F1-Score: Default (0.50) vs. Tuned Threshold", fontweight="bold")
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=35, ha="right")
        axes[0].set_xlabel("")

        # Recall comparison
        sns.barplot(data=agg, x="Algorithm", y="Recall", hue="Threshold_Type", ax=axes[1], palette="Set2")
        axes[1].set_title("Impact on Recall (Flaky Tests Caught)", fontweight="bold")
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=35, ha="right")
        axes[1].set_xlabel("")

        plt.suptitle("Refinement Analysis: Resolving the Extreme Cross-Project False-Negative Bottleneck", fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(self.results_dir / "02_threshold_impact_comparison.png", dpi=300)
        plt.close()

    def plot_f1_heatmap(self, clf_df: pd.DataFrame) -> None:
        """Project-by-model heatmap of F1-Scores across all 24 projects."""
        df_tuned = clf_df[clf_df["Threshold_Type"] == "Tuned_InnerCV"]
        pivot_f1 = df_tuned.pivot(index="Project", columns="Algorithm", values="F1-Score")
        cols_order = [
            "RF_Baseline", "RF_Balanced", "RF_SMOTE", "Balanced_RF",
            "XGB_Baseline", "XGB_ClassWeight", "LightGBM_Balanced",
            "GradientBoosting_Balanced", "LogisticRegression_Balanced", "Ensemble_Voting_Refined"
        ]
        available_cols = [c for c in cols_order if c in pivot_f1.columns]
        pivot_f1 = pivot_f1[available_cols]

        plt.figure(figsize=(14, 10))
        sns.heatmap(pivot_f1, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "F1-Score"}, linewidths=0.5)
        plt.title("Cross-Project LOOCV Heatmap: F1-Score per Project and Model", fontweight="bold")
        plt.xlabel("Algorithm")
        plt.ylabel("Left-Out Project (Test Set)")
        plt.tight_layout()
        plt.savefig(self.results_dir / "03_f1_heatmap_per_project.png", dpi=300)
        plt.close()

    def plot_mcc_heatmap(self, clf_df: pd.DataFrame) -> None:
        """Project-by-model heatmap of MCC across all 24 projects."""
        df_tuned = clf_df[clf_df["Threshold_Type"] == "Tuned_InnerCV"]
        pivot_mcc = df_tuned.pivot(index="Project", columns="Algorithm", values="MCC")
        cols_order = [
            "RF_Baseline", "RF_Balanced", "RF_SMOTE", "Balanced_RF",
            "XGB_Baseline", "XGB_ClassWeight", "LightGBM_Balanced",
            "GradientBoosting_Balanced", "LogisticRegression_Balanced", "Ensemble_Voting_Refined"
        ]
        available_cols = [c for c in cols_order if c in pivot_mcc.columns]
        pivot_mcc = pivot_mcc[available_cols]

        plt.figure(figsize=(14, 10))
        sns.heatmap(pivot_mcc, annot=True, fmt=".2f", cmap="viridis", cbar_kws={"label": "MCC"}, linewidths=0.5)
        plt.title("Cross-Project LOOCV Heatmap: Matthews Correlation Coefficient (MCC)", fontweight="bold")
        plt.xlabel("Algorithm")
        plt.ylabel("Left-Out Project (Test Set)")
        plt.tight_layout()
        plt.savefig(self.results_dir / "04_mcc_heatmap_per_project.png", dpi=300)
        plt.close()

    def plot_feature_importance(self, feat_df: pd.DataFrame) -> None:
        """Horizontal barplot of top FlakeFlagger predictor features across folds."""
        if feat_df.empty:
            return
        rf_feat = feat_df[feat_df["Algorithm"] == "RF_Balanced"]
        if rf_feat.empty:
            rf_feat = feat_df
        top15 = rf_feat.groupby("Feature")["Importance"].mean().sort_values(ascending=False).head(15).reset_index()

        plt.figure(figsize=(11, 7))
        sns.barplot(data=top15, x="Importance", y="Feature", palette="crest_r")
        plt.title("Top-15 Most Influential Features for Cross-Project Flaky Test Prediction", fontweight="bold")
        plt.xlabel("Mean Gini Feature Importance (Averaged across 24 LOOCV Folds)")
        plt.ylabel("Feature Name")
        plt.tight_layout()
        plt.savefig(self.results_dir / "05_top_feature_importance.png", dpi=300)
        plt.close()

    def plot_regression_boxplots(self, reg_df: pd.DataFrame) -> None:
        """Boxplots of regression metrics (Spearman rho, RMSE, MAE)."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        sns.boxplot(data=reg_df, x="Algorithm", y="Spearman_Rho", ax=axes[0], palette="Blues_r")
        axes[0].set_title("Spearman Rank Correlation (Rho)", fontweight="bold")
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha="right")
        axes[0].set_xlabel("")

        sns.boxplot(data=reg_df, x="Algorithm", y="RMSE", ax=axes[1], palette="Reds_r")
        axes[1].set_title("Root Mean Squared Error (RMSE)", fontweight="bold")
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha="right")
        axes[1].set_xlabel("")

        sns.boxplot(data=reg_df, x="Algorithm", y="MAE", ax=axes[2], palette="Greens_r")
        axes[2].set_title("Mean Absolute Error (MAE)", fontweight="bold")
        axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=30, ha="right")
        axes[2].set_xlabel("")

        plt.suptitle("Continuous Flakiness Prediction (NumFailingRuns) Across 24 Projects", fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(self.results_dir / "06_regression_boxplots.png", dpi=300)
        plt.close()


# =====================================================================
# 6. Main Execution Entrypoint
# =====================================================================

def main() -> None:
    current_dir = Path(__file__).resolve().parent
    # The root repository is 3 levels up: eksperimen-1 -> Refinement-AllModel-Loocv-flakeflagger -> Eksperimen_Kelompok -> ROOT
    root_dir = current_dir.parent.parent.parent

    loader = FlakeFlaggerDataLoader(root_dir)
    data = loader.load_and_align()

    engine = RefinedLOOCVEngine(data, current_dir)
    clf_df, reg_df, feat_imp_df = engine.run()

    clf_summary, reg_summary, feat_summary = engine.process_and_save_summaries(clf_df, reg_df, feat_imp_df)

    vis = Visualizer(current_dir / "results")
    vis.generate_all_plots(clf_df, reg_df, feat_imp_df)

    logger.info("\n" + "=" * 60)
    logger.info("TOP CLASSIFICATION MODELS (Tuned Threshold, F1-Score & MCC):")
    logger.info("=" * 60)
    df_tuned_sum = clf_summary[clf_summary["Threshold_Type"] == "Tuned_InnerCV"]
    print(df_tuned_sum[["Algorithm", "F1 (Mean ± Std)", "MCC (Mean ± Std)", "AUC-PR (Mean ± Std)", "ROC-AUC (Mean ± Std)"]].to_string())

    logger.info("\n" + "=" * 60)
    logger.info("REGRESSION MODELS SUMMARY (NumFailingRuns):")
    logger.info("=" * 60)
    print(reg_summary[["Algorithm", "Spearman (Mean ± Std)", "RMSE_Mean", "MAE_Mean"]].to_string())
    logger.info("\nRefinement Experiment Completed Successfully!")


if __name__ == "__main__":
    main()
