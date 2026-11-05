"""
Modul Pembangun Pipeline dan Evaluasi Metrik Prediksi Flaky Test.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    average_precision_score,
    roc_auc_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    XGBClassifier = None
    HAS_XGB = False


def get_scaler(scaler_type: str = "MinMaxScaler"):
    """
    Mengembalikan scaler yang harus di-fit HANYA pada data latih (Zero Leakage).
    """
    st = scaler_type.lower()
    if st in ["s1", "standard", "standardscaler"]:
        return StandardScaler()
    elif st in ["s2", "minmax", "minmaxscaler"]:
        return MinMaxScaler()
    elif st in ["s3", "robust", "robustscaler"]:
        return RobustScaler()
    return MinMaxScaler()


def apply_resampling(
    X: np.ndarray,
    y: np.ndarray,
    resampler_type: str = "RUS",
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Menerapkan penyeimbangan kelas (RUS, SMOTE, atau None/Natural) hanya pada data latih.
    """
    rt = resampler_type.upper()
    n_classes = len(np.unique(y))
    n_flaky = int(np.sum(y == 1))

    if rt in ["B3", "RUS", "RANDOMUNDERSAMPLER"]:
        if n_classes > 1 and n_flaky >= 1:
            rus = RandomUnderSampler(random_state=random_state)
            try:
                return rus.fit_resample(X, y)
            except Exception:
                return X, y
    elif rt in ["B2", "SMOTE"]:
        if n_classes > 1 and n_flaky >= 2:
            k_neigh = max(1, min(5, n_flaky - 1))
            smote = SMOTE(k_neighbors=k_neigh, random_state=random_state)
            try:
                return smote.fit_resample(X, y)
            except Exception:
                return X, y
    return X, y


def build_classifier(
    classifier_id: str = "M1",
    random_state: int = 42
) -> Any:
    """
    Membangun model klasifikasi: M1 (Random Forest), M2 (XGBoost), atau M3 (Decision Tree).
    """
    cid = classifier_id.upper()
    if cid in ["M2", "XGBOOST", "XGB"]:
        if HAS_XGB:
            try:
                clf = XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    tree_method="hist",
                    device="cuda",
                    random_state=random_state,
                    eval_metric="logloss"
                )
                return clf
            except Exception:
                pass
        # Fallback CPU XGBoost jika CUDA error
        return XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric="logloss"
        )
    elif cid in ["M3", "DT", "DECISIONTREE"]:
        return DecisionTreeClassifier(
            max_depth=4,
            class_weight="balanced",
            random_state=random_state
        )
    else:
        return RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )


def evaluate_predictions(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    y_preds: np.ndarray
) -> Dict[str, float]:
    """
    Menghitung seluruh metrik performa secara terstandarisasi.
    Menangani kasus ekstrem kelas tunggal (misal: 0 flaky test pada jimfs).
    """
    n_flaky = int(np.sum(y_true == 1))
    n_non_flaky = int(np.sum(y_true == 0))
    tn, fp, fn, tp = 0, 0, 0, 0

    if len(y_true) > 0:
        cm = confusion_matrix(y_true, y_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

    if n_flaky == 0:
        pr_auc = 0.0
        roc_auc = 0.5
        rec = 0.0
        prec = 0.0 if fp > 0 else 1.0
        f1 = 0.0
        mcc = 0.0
        bal_acc = float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0
    else:
        # PR-AUC
        try:
            pr_auc = float(average_precision_score(y_true, y_probs))
        except Exception:
            pr_auc = 0.0

        # ROC-AUC
        try:
            roc_auc = float(roc_auc_score(y_true, y_probs))
        except Exception:
            roc_auc = 0.5

        prec = float(precision_score(y_true, y_preds, pos_label=1, zero_division=0))
        rec = float(recall_score(y_true, y_preds, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_true, y_preds, pos_label=1, zero_division=0))
        mcc = float(matthews_corrcoef(y_true, y_preds))
        bal_acc = float(balanced_accuracy_score(y_true, y_preds))

    return {
        "f1_score": round(f1, 4),
        "recall_score": round(rec, 4),
        "precision_score": round(prec, 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "mcc": round(mcc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "n_test_samples": len(y_true),
        "n_flaky": n_flaky,
        "n_non_flaky": n_non_flaky
    }
