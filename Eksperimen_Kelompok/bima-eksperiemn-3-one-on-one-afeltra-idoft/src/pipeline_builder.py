"""
Modul Pipeline Builder untuk Eksperimen Afeltra IDoFT Dataset.
Menyediakan konfigurasi Feature Scaler, Imbalance Resampler, dan Classifiers (RF & XGBoost CUDA GPU).
Rujukan: Afeltra et al. (IEEE Access 2024).
"""

from typing import Tuple, Any, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

def get_scaler(scaling_type: str = "S2") -> Any:
    """
    Mengembalikan scaler yang diminta:
    - S1: StandardScaler
    - S2: MinMaxScaler (default Afeltra)
    - S3: RobustScaler
    """
    if scaling_type == "S1":
        return StandardScaler()
    elif scaling_type == "S2":
        return MinMaxScaler()
    elif scaling_type == "S3":
        return RobustScaler()
    else:
        return MinMaxScaler()

def apply_resampling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str = "RUS",
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Menerapkan teknik penanganan ketimpangan kelas pada lipatan pelatihan:
    - RUS: Random Under-Sampling (Teknik terbaik menurut temuan Afeltra 2024 Table 3)
    - SMOTE: Synthetic Minority Over-sampling
    - None: Tanpa penyeimbangan (asli)
    """
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))

    if n_pos == 0 or n_neg == 0:
        return X_train, y_train

    if strategy == "RUS":
        rus = RandomUnderSampler(random_state=random_state)
        X_res, y_res = rus.fit_resample(X_train, y_train)
        return X_res, y_res

    elif strategy == "SMOTE":
        k_neighbors = min(5, max(1, n_pos - 1))
        if k_neighbors < 1:
            return X_train, y_train
        smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
        try:
            X_res, y_res = smote.fit_resample(X_train, y_train)
            return X_res, y_res
        except Exception:
            return X_train, y_train

    else:
        return X_train, y_train

def build_classifier(
    model_type: str = "M1",
    random_state: int = 42,
    use_gpu: bool = True
) -> Any:
    """
    Membangun classifier:
    - M1: Random Forest Classifier (Model utama Afeltra et al., 2024)
    - M2: XGBoost Classifier dengan akselerasi GPU CUDA hist method
    """
    if model_type == "M1":
        return RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == "M2":
        if XGBClassifier is not None:
            device = "cuda" if use_gpu else "cpu"
            try:
                clf = XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    tree_method="hist",
                    device=device,
                    eval_metric="logloss",
                    random_state=random_state
                )
                return clf
            except Exception:
                # Fallback ke CPU jika CUDA gagal
                return XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    tree_method="hist",
                    device="cpu",
                    eval_metric="logloss",
                    random_state=random_state
                )
        else:
            return RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1
            )
    else:
        return RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
