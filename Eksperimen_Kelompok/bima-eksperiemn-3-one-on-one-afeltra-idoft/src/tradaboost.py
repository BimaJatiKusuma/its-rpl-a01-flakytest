"""
Modul Implementasi Algoritma TrAdaBoost (Transfer AdaBoost) untuk Cross-Project Flaky Test Prediction.
Rujukan:
- Dai et al. (ICML 2007): Boosting for Transfer Learning.
- Afeltra et al. (IEEE Access 2024): Section IV.B & Section V.B (evaluating 25%, 50%, and 75% target adaptation splits).
"""

import warnings
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


class TrAdaBoostClassifier(BaseEstimator, ClassifierMixin):
    """
    TrAdaBoost Classifier sesuai formulasi Dai et al. (2007) dan Afeltra et al. (2024).
    Mendukung Base Learner DecisionTreeClassifier, RandomForestClassifier, dan XGBoost.
    """

    def __init__(
        self,
        base_estimator: Optional[Any] = None,
        n_estimators: int = 20,
        random_state: int = 42,
        use_second_half: bool = True
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.use_second_half = use_second_half

        self.estimators_: List[Any] = []
        self.beta_all_: List[float] = []
        self.beta_source_: float = 1.0

    def fit(self, X_source: np.ndarray, y_source: np.ndarray, X_target: np.ndarray, y_target: np.ndarray):
        """
        Melatih TrAdaBoost menggunakan kombinasi data sumber dan data target adaptasi.

        Args:
            X_source: Fitur data sumber (N_source, n_features)
            y_source: Label biner sumber (N_source,)
            X_target: Fitur data target adaptasi (N_target, n_features)
            y_target: Label biner target adaptasi (N_target,)
        """
        n_source = len(X_source)
        n_target = len(X_target)
        n_total = n_source + n_target

        if n_source == 0 or n_target == 0:
            raise ValueError(f"Ukuran source ({n_source}) atau target ({n_target}) tidak boleh 0.")

        X_all = np.vstack([X_source, X_target])
        y_all = np.concatenate([y_source, y_target]).astype(int)

        # Inisialisasi bobot sampel: w = 1/n_total
        weights = np.ones(n_total, dtype=np.float64) / n_total

        # Faktor penalti sumber konstan (Dai et al., 2007)
        self.beta_source_ = 1.0 / (1.0 + np.sqrt(2.0 * np.log(max(2.0, float(n_source)) / float(self.n_estimators))))

        self.estimators_ = []
        self.beta_all_ = []

        base = self.base_estimator
        if base is None:
            base = DecisionTreeClassifier(max_depth=4, random_state=self.random_state)

        for i in range(self.n_estimators):
            # Normalisasi bobot
            sum_w = np.sum(weights)
            if sum_w <= 0:
                weights = np.ones(n_total, dtype=np.float64) / n_total
            else:
                weights = weights / sum_w

            # Clone dan latih base estimator dengan sample_weight
            est = clone(base)
            try:
                est.fit(X_all, y_all, sample_weight=weights)
            except Exception:
                # Fallback tanpa sample_weight jika estimator tidak mendukung
                est.fit(X_all, y_all)

            pred_all = est.predict(X_all)
            pred_target = pred_all[n_source:]
            weights_target = weights[n_source:]

            # Evaluasi error pada target
            denom = np.sum(weights_target)
            if denom <= 0:
                error_target = 0.5
            else:
                error_target = float(np.sum(weights_target * (pred_target != y_target)) / denom)

            # Batasan stabilitas numerik
            if error_target > 0.499:
                error_target = 0.499
            if error_target < 1e-6:
                beta_target = 1e-4
            else:
                beta_target = error_target / (1.0 - error_target)

            self.estimators_.append(est)
            self.beta_all_.append(beta_target)

            # Update bobot data sumber (menurunkan bobot sampel yang membingungkan)
            source_errors = (pred_all[:n_source] != y_source)
            weights[:n_source][source_errors] *= self.beta_source_

            # Update bobot data target (menaikkan bobot sampel target yang salah diklasifikasikan)
            target_errors = (pred_target != y_target)
            weights[n_source:][target_errors] *= (1.0 / max(1e-6, beta_target))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Menghasilkan probabilitas positif menggunakan voting terbobot TrAdaBoost.
        """
        n_samples = len(X)
        if len(self.estimators_) == 0:
            return np.zeros((n_samples, 2))

        start_idx = int(np.ceil(self.n_estimators / 2.0)) if self.use_second_half else 0
        estimators_to_use = self.estimators_[start_idx:]
        betas_to_use = self.beta_all_[start_idx:]

        if len(estimators_to_use) == 0:
            estimators_to_use = self.estimators_
            betas_to_use = self.beta_all_

        weights_0 = np.zeros(n_samples, dtype=np.float64)
        weights_1 = np.zeros(n_samples, dtype=np.float64)

        for est, beta_t in zip(estimators_to_use, betas_to_use):
            alpha = float(np.log(1.0 / max(1e-6, beta_t)))
            preds = est.predict(X)
            weights_0[preds == 0] += alpha
            weights_1[preds == 1] += alpha

        # Konversi voting ke rentang probabilitas [0, 1] via softmax / normalisasi
        total = weights_0 + weights_1
        total[total == 0] = 1.0
        prob_1 = weights_1 / total
        prob_0 = weights_0 / total

        return np.vstack([prob_0, prob_1]).T

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Prediksi kelas biner [0, 1].
        """
        probs = self.predict_proba(X)
        return (probs[:, 1] >= threshold).astype(int)
