"""
Pipeline Builder Module (Feature Scaler, Imbalance Sampler, and Classifier) for Combined Experiment 1 & 2.
Supports:
- Data Cleaning: C1 (Row Discarding), C2 (Project Discarding), C3 (Group-Aware Median Imputation)
- Feature Scaler: StandardScaler (S1), MinMaxScaler (S2), RobustScaler (S3)
- Imbalance Resampling: None/Baseline (B1), SMOTE (B2), RUS (B3), ClassWeight (B4)
- Classifiers: Random Forest CPU (M1), XGBoost GPU (M2)

Total Full Factorial Combinations: 3 x 3 x 4 x 2 = 72 Machine Learning Pipelines.
"""

from typing import Any, Tuple, Optional, List, Dict
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier

# Safe import for imblearn
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:
    SMOTE = None
    RandomUnderSampler = None

# Safe import for xgboost
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except ImportError:
    xgb = None
    XGBClassifier = None


def get_scaler(scaling_type: str):
    """Instantiates and returns the configured scikit-learn feature scaler object."""
    st = scaling_type.upper()
    if st == "S1" or "STANDARD" in st:
        return StandardScaler()
    elif st == "S2" or "MINMAX" in st:
        return MinMaxScaler()
    elif st == "S3" or "ROBUST" in st:
        return RobustScaler()
    else:
        raise ValueError(f"Scaling type '{scaling_type}' is unrecognized. Select S1, S2, or S3.")


def apply_resampling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    imbalance_type: str,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies class imbalance resampling EXCLUSIVELY to training folds (Guarantees Zero Data Leakage).
    """
    it = imbalance_type.upper()
    if it == "B1" or "NONE" in it or "BASELINE" in it:
        return X_train, y_train

    elif it == "B2" or "SMOTE" in it:
        if SMOTE is None:
            raise ImportError("Package 'imbalanced-learn' is required for SMOTE oversampling.")
        min_samples = np.sum(y_train == 1)
        if min_samples <= 5:
            k = max(1, min_samples - 1)
            sampler = SMOTE(k_neighbors=k, random_state=random_state)
        else:
            sampler = SMOTE(random_state=random_state)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        return X_res, y_res

    elif it == "B3" or "RUS" in it or "UNDER" in it:
        if RandomUnderSampler is None:
            raise ImportError("Package 'imbalanced-learn' is required for RandomUnderSampler.")
        sampler = RandomUnderSampler(random_state=random_state)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        return X_res, y_res

    elif it == "B4" or "WEIGHT" in it or "COST" in it:
        # B4 is handled via loss function sample weighting in the classifier model
        return X_train, y_train

    else:
        raise ValueError(f"Imbalance type '{imbalance_type}' is unrecognized. Select B1, B2, B3, or B4.")


def build_classifier(
    model_type: str,
    imbalance_type: str,
    use_gpu: bool = True,
    scale_pos_weight_val: float = 1.0,
    random_state: int = 42
) -> Any:
    """
    Constructs the classifier model (Random Forest or GPU-accelerated XGBoost) based on specifications.
    """
    mt = model_type.upper()
    it = imbalance_type.upper()
    is_cost_sensitive = (it == "B4" or "WEIGHT" in it or "COST" in it)

    if mt == "M1" or "RF" in mt or "FOREST" in mt:
        cw = "balanced" if is_cost_sensitive else None
        return RandomForestClassifier(
            n_estimators=100,
            class_weight=cw,
            random_state=random_state,
            n_jobs=-1
        )

    elif mt == "M2" or "XGB" in mt or "XGBOOST" in mt:
        if XGBClassifier is None:
            raise ImportError("Package 'xgboost' is required.")

        weight = scale_pos_weight_val if is_cost_sensitive else 1.0
        device = "cuda" if use_gpu else "cpu"

        try:
            model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=weight,
                tree_method="hist",
                device=device,
                eval_metric="logloss",
                random_state=random_state
            )
            return model
        except Exception:
            # Fallback to CPU tree_method='hist' if CUDA device is uninitialized at runtime
            return XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=weight,
                tree_method="hist",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1
            )
    else:
        raise ValueError(f"Model type '{model_type}' is unrecognized. Select M1 or M2.")


def get_pipeline_specs() -> List[Dict[str, Any]]:
    """
    Generates definitions for all 72 full factorial pipeline combinations:
    3 Cleaning (C1, C2, C3) x 3 Scaling (S1, S2, S3) x 4 Imbalance (B1, B2, B3, B4) x 2 Models (M1, M2) = 72 Pipelines.
    """
    cleanings = ["C1", "C2", "C3"]
    scalers = ["S1", "S2", "S3"]
    imbalances = ["B1", "B2", "B3", "B4"]
    models = ["M1", "M2"]

    pipelines = []
    pid = 1
    for c in cleanings:
        for s in scalers:
            for b in imbalances:
                for m in models:
                    p_code = f"P{pid:02d}_{c}_{s}_{b}_{m}"
                    pipelines.append({
                        "pipeline_id": p_code,
                        "pipeline_number": pid,
                        "cleaning": c,
                        "scaling": s,
                        "imbalance": b,
                        "model": m,
                    })
                    pid += 1
    return pipelines
