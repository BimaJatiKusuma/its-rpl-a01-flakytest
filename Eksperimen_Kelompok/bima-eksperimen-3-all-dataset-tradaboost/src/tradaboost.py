"""
Modul Implementasi Algoritma TrAdaBoost (Transfer AdaBoost) untuk Cross-Project Flaky Test Prediction.
Rujukan:
- W. Dai, Q. Yang, G.-R. Xue, and Y. Yu (ICML 2007): "Boosting for transfer learning".
- A. Afeltra, A. Cannavale, F. Pecorelli, V. Pontillo, and F. Palomba (IEEE Access 2024):
  "A Large-Scale Empirical Investigation Into Cross-Project Flaky Test Prediction" (Section IV-B & V-B).
"""

import warnings
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")


class TrAdaBoostClassifier(BaseEstimator, ClassifierMixin):
    """
    TrAdaBoost instance-based transfer learning classifier.
    Menyelaraskan bobot instans dari domain sumber dengan domain target.
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

    def fit(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        X_target: np.ndarray,
        y_target: np.ndarray
    ) -> "TrAdaBoostClassifier":
        """
        Melatih TrAdaBoost menggunakan kombinasi data sumber dan data target adaptasi.

        Args:
            X_source: Fitur data proyek sumber (N_source, n_features)
            y_source: Label biner sumber (N_source,)
            X_target: Fitur data proyek target adaptasi (N_target, n_features)
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
        weights = np.ones(n_total, dtype=np.float64) / float(n_total)

        # Faktor penalti sumber konstan (Dai et al., ICML 2007)
        self.beta_source_ = 1.0 / (1.0 + np.sqrt(2.0 * np.log(max(2.0, float(n_source)) / float(self.n_estimators))))

        self.estimators_ = []
        self.beta_all_ = []

        base = self.base_estimator
        if base is None:
            base = DecisionTreeClassifier(max_depth=4, random_state=self.random_state)

        for i in range(self.n_estimators):
            # Normalisasi bobot sampel
            sum_w = np.sum(weights)
            if sum_w <= 0:
                weights = np.ones(n_total, dtype=np.float64) / float(n_total)
            else:
                weights = weights / sum_w

            # Clone dan latih base estimator dengan sample_weight
            est = clone(base)
            if hasattr(est, "random_state"):
                est.random_state = self.random_state + i

            try:
                est.fit(X_all, y_all, sample_weight=weights)
            except Exception:
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

            # Update bobot data sumber: menurunkan bobot sampel sumber yang membingungkan target
            source_errors = (pred_all[:n_source] != y_source)
            weights[:n_source][source_errors] *= self.beta_source_

            # Update bobot data target: menaikkan bobot sampel target yang salah diklasifikasikan
            target_errors = (pred_target != y_target)
            weights[n_source:][target_errors] *= (1.0 / max(1e-6, beta_target))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Menghasilkan probabilitas positif menggunakan voting log-terbobot TrAdaBoost.
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

        # Konversi voting ke rentang probabilitas [0, 1] via normalisasi
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


def split_target_data(
    X_target: np.ndarray,
    y_target: np.ndarray,
    target_train_ratio: float,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Membagi data proyek target menjadi train (adaptasi) dan test (evaluasi).
    Mempertahankan rasio kelas secara terstratifikasi jika memungkinkan.
    """
    n_target = len(X_target)
    if target_train_ratio <= 0.0:
        # Zero-Shot: 0% train, 100% test
        return np.empty((0, X_target.shape[1])), np.empty((0,), dtype=int), X_target, y_target

    n_train = int(np.round(n_target * target_train_ratio))
    n_train = max(1, min(n_train, n_target - 1))

    # Cek apakah kedua kelas ada di target
    has_both_classes = (len(np.unique(y_target)) > 1) and (np.sum(y_target == 1) >= 2) and (np.sum(y_target == 0) >= 2)

    if has_both_classes:
        try:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=n_train, random_state=random_state)
            train_idx, test_idx = next(sss.split(X_target, y_target))
            return X_target[train_idx], y_target[train_idx], X_target[test_idx], y_target[test_idx]
        except Exception:
            pass

    # Fallback jika Stratified gagal atau hanya 1 kelas
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(n_target)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    return X_target[train_idx], y_target[train_idx], X_target[test_idx], y_target[test_idx]
