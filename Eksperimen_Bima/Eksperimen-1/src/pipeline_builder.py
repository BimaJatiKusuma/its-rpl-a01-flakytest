"""
Modul pembangun pipeline (Scaler, Imbalance Sampler, dan Classifier).
Mendukung:
- Scaler: StandardScaler (S1), MinMaxScaler (S2), RobustScaler (S3)
- Imbalance: None (B1), SMOTE (B2), RUS (B3), ClassWeight (B4)
- Classifier: Random Forest (M1), XGBoost GPU (M2)
"""

from typing import Any, Tuple, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier

# Coba import imblearn
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:
    SMOTE = None
    RandomUnderSampler = None

# Coba import xgboost
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except ImportError:
    xgb = None
    XGBClassifier = None


def get_scaler(scaling_type: str):
    """Mengembalikan objek scaler yang sesuai."""
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
        # Cek ketersediaan sampel minoritas minimum untuk k_neighbors
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

    elif it == "B4" or "WEIGHT" in it:
        # B4 ditangani pada level model classifier (class_weight / scale_pos_weight)
        return X_train, y_train

    else:
        raise ValueError(f"Imbalance type '{imbalance_type}' tidak dikenali. Pilih B1, B2, B3, atau B4.")


def build_classifier(
    model_type: str,
    imbalance_type: str,
    use_gpu: bool = True,
    scale_pos_weight_val: float = 1.0,
    random_state: int = 42
):
    """
    Membangun instance model classifier (RF atau XGBoost) sesuai konfigurasi.
    """
    mt = model_type.upper()
    is_class_weighted = (imbalance_type.upper() == "B4" or "WEIGHT" in imbalance_type.upper())

    if mt == "M1" or "RF" in mt or "RANDOM_FOREST" in mt:
        cw = "balanced" if is_class_weighted else None
        return RandomForestClassifier(
            n_estimators=100,
            class_weight=cw,
            random_state=random_state,
            n_jobs=-1
        )

    elif mt == "M2" or "XGB" in mt or "XGBOOST" in mt:
        if XGBClassifier is None:
            raise ImportError("Pustaka xgboost belum terpasang.")
        
        weight = scale_pos_weight_val if is_class_weighted else 1.0
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
            # Fallback ke CPU jika CUDA device belum siap di runtime
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
        raise ValueError(f"Model type '{model_type}' tidak dikenali. Pilih M1 atau M2.")
