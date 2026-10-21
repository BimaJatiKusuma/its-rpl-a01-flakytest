"""
Modul pembangun pipeline (Scaler, Imbalance Sampler, dan Classifier) untuk Eksperimen 3.
Mendukung variasi random_state untuk pengujian konsistensi dan stabilitas empiris:
- Scaler: StandardScaler (S1), MinMaxScaler (S2), RobustScaler (S3)
- Imbalance: None (B1), SMOTE (B2), RUS (B3), ClassWeight (B4)
- Classifier: Random Forest (M1), XGBoost GPU (M2)
"""

from typing import Any, Tuple, List, Dict
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier

# Import imblearn
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:
    SMOTE = None
    RandomUnderSampler = None

# Import xgboost
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except ImportError:
    xgb = None
    XGBClassifier = None


def get_scaler(scaling_type: str):
    """Mengembalikan objek scaler scikit-learn."""
    st = scaling_type.upper()
    if st == "S1" or "STANDARD" in st:
        return StandardScaler()
    elif st == "S2" or "MINMAX" in st:
        return MinMaxScaler()
    elif st == "S3" or "ROBUST" in st:
        return RobustScaler()
    else:
        raise ValueError(f"Scaling type '{scaling_type}' tidak dikenali. Pilih S1, S2, atau S3.")


def apply_resampling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    imbalance_type: str,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Menerapkan teknik penanganan ketimpangan kelas HANYA pada data latih (Zero Leakage).
    """
    it = imbalance_type.upper()
    if it == "B1" or "NONE" in it or "BASELINE" in it:
        return X_train, y_train

    elif it == "B2" or "SMOTE" in it:
        if SMOTE is None:
            raise ImportError("Pustaka imbalanced-learn belum terpasang untuk SMOTE.")
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
            raise ImportError("Pustaka imbalanced-learn belum terpasang untuk RandomUnderSampler.")
        sampler = RandomUnderSampler(random_state=random_state)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        return X_res, y_res

    elif it == "B4" or "WEIGHT" in it or "COST" in it:
        # B4 ditangani pada level bobot loss di classifier
        return X_train, y_train

    else:
        raise ValueError(f"Imbalance type '{imbalance_type}' tidak dikenali. Pilih B1, B2, B3, atau B4.")


def build_classifier(
    model_type: str,
    imbalance_type: str,
    use_gpu: bool = True,
    random_state: int = 42
) -> Any:
    """
    Membangun model estimator classifier (Random Forest atau XGBoost) dengan random_state spesifik.
    """
    mt = model_type.upper()
    it = imbalance_type.upper()
    is_cost_sensitive = (it == "B4" or "WEIGHT" in it or "COST" in it)

    if mt == "M1" or "RF" in mt or "FOREST" in mt:
        class_weight = "balanced" if is_cost_sensitive else None
        return RandomForestClassifier(
            n_estimators=100,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1
        )

    elif mt == "M2" or "XGB" in mt:
        if XGBClassifier is None:
            raise ImportError("Pustaka xgboost belum terpasang.")

        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": random_state,
            "eval_metric": "logloss"
        }

        if use_gpu:
            params["tree_method"] = "hist"
            params["device"] = "cuda"
        else:
            params["tree_method"] = "hist"
            params["n_jobs"] = -1

        clf = XGBClassifier(**params)
        return clf

    else:
        raise ValueError(f"Model type '{model_type}' tidak dikenali. Pilih M1 (RF) atau M2 (XGBoost).")


def get_pipeline_specs() -> List[Dict[str, Any]]:
    """
    Menghasilkan definisi 24 kombinasi pipeline:
    3 Scaling (S1, S2, S3) x 4 Imbalance (B1, B2, B3, B4) x 2 Models (M1, M2) = 24 Pipelines.
    """
    scalers = ["S1", "S2", "S3"]
    imbalances = ["B1", "B2", "B3", "B4"]
    models = ["M1", "M2"]

    pipelines = []
    pid = 1
    for s in scalers:
        for b in imbalances:
            for m in models:
                p_code = f"P{pid:02d}_{s}_{b}_{m}"
                pipelines.append({
                    "pipeline_id": p_code,
                    "pipeline_number": pid,
                    "scaling": s,
                    "imbalance": b,
                    "model": m,
                })
                pid += 1
    return pipelines
