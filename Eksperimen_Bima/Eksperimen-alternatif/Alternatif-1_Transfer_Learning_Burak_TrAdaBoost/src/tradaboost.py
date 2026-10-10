"""
Modul Implementasi Algoritma TrAdaBoost (Transfer AdaBoost) dengan Akselerasi GPU XGBoost.
Rujukan:
- Dai et al. (ICML 2007): Boosting for Transfer Learning.
- Afeltra et al. (IEEE Access 2024): Mitigating cross-project domain shift.

Mekanisme Utama:
- Menggabungkan data sumber (Ds) dan subset berlabel data target (Dt).
- Base learner XGBoost GPU dengan scale_pos_weight untuk menangani ketimpangan kelas flaky (3.65%).
- Menurunkan bobot sampel sumber yang salah diklasifikasikan secara iteratif:
  beta_s = 1 / (1 + sqrt(2 * ln(n / N)))
- Meningkatkan bobot sampel target yang salah diklasifikasikan:
  beta_t = epsilon_t / (1 - epsilon_t)
- Mengombinasikan model pada separuh iterasi terakhir (t = N/2 s.d N) untuk inferensi probabilitas kontinu.
"""

import warnings
from typing import Dict, Any, List, Optional
import numpy as np

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except ImportError:
    xgb = None
    XGBClassifier = None


class TrAdaBoostGPU:
    """
    TrAdaBoost Classifier yang mengintegrasikan XGBoost berakselerasi GPU (NVIDIA RTX 4050)
    sebagai Base Learner dalam setiap iterasi boosting transfer learning.
    """

    def __init__(
        self,
        n_estimators: int = 30,
        base_learner_params: Optional[Dict[str, Any]] = None,
        use_second_half_ensemble: bool = True,
        clip_error_min: float = 1e-4,
        clip_error_max: float = 0.499,
        random_state: int = 42,
        use_gpu: bool = True
    ):
        self.n_estimators = n_estimators
        self.use_second_half_ensemble = use_second_half_ensemble
        self.clip_error_min = clip_error_min
        self.clip_error_max = clip_error_max
        self.random_state = random_state
        self.use_gpu = use_gpu

        default_base_params = {
            "n_estimators": 25,
            "max_depth": 4,
            "learning_rate": 0.1,
            "tree_method": "hist",
            "device": "cuda" if use_gpu else "cpu",
            "eval_metric": "logloss",
            "random_state": random_state
        }
        if base_learner_params:
            default_base_params.update(base_learner_params)
        self.base_params = default_base_params

        self.models: List[Any] = []
        self.beta_ts: List[float] = []
        self.alphas: List[float] = []
        self.beta_source: float = 1.0

    def fit(self, X_source: np.ndarray, y_source: np.ndarray, X_target: np.ndarray, y_target: np.ndarray):
        """
        Melatih TrAdaBoost menggunakan kombinasi data sumber dan data target berlabel.
        """
        n = len(X_source)
        m = len(X_target)
        N = self.n_estimators

        if n == 0 or m == 0:
            raise ValueError(f"Ukuran data sumber ({n}) atau target ({m}) tidak boleh kosong.")

        # Inisialisasi bobot sampel sesuai kaidah Dai et al. (2007)
        w = np.zeros(n + m, dtype=np.float64)
        w[:n] = 1.0 / n
        w[n:] = 1.0 / m
        w = w / np.sum(w)

        # Penggabungan dataset
        X_all = np.vstack([X_source, X_target])
        y_all = np.concatenate([y_source, y_target]).astype(int)

        # Hitung scale_pos_weight untuk menangani ketimpangan kelas ekstrem
        n_pos = int(np.sum(y_all == 1))
        n_neg = int(np.sum(y_all == 0))
        pos_weight = float(n_neg / max(1, n_pos))

        # Faktor penalti sumber konstan
        self.beta_source = 1.0 / (1.0 + np.sqrt(2.0 * np.log(max(2.0, float(n)) / max(1, N))))

        self.models = []
        self.beta_ts = []
        self.alphas = []

        for t in range(N):
            # 1. Normalisasi distribusi probabilitas bobot
            sum_w = np.sum(w)
            p = w / sum_w if sum_w > 0 else np.ones_like(w) / len(w)

            # 2. Latih Base Classifier h_t dengan scale_pos_weight
            seed = self.random_state + t
            clf_params = dict(self.base_params)
            clf_params["random_state"] = seed
            clf_params["scale_pos_weight"] = pos_weight

            try:
                clf = XGBClassifier(**clf_params)
                clf.fit(X_all, y_all, sample_weight=p)
            except Exception:
                # Fallback ke CPU jika terjadi kendala alokasi CUDA
                fallback_params = dict(clf_params)
                fallback_params["device"] = "cpu"
                fallback_params["n_jobs"] = -1
                clf = XGBClassifier(**fallback_params)
                clf.fit(X_all, y_all, sample_weight=p)

            # 3. Hitung error epsilon_t pada data target berlabel
            probs_target = clf.predict_proba(X_target)
            p_pos_t = probs_target[:, 1] if probs_target.shape[1] > 1 else probs_target[:, 0]
            loss_t = np.abs(p_pos_t - y_target)

            # Error target seimbang
            p_target = p[n:]
            sum_p_target = np.sum(p_target)
            if sum_p_target > 0:
                p_target_norm = p_target / sum_p_target
            else:
                p_target_norm = np.ones(m) / m

            # Jika terdapat kelas positif pada data target, gunakan balanced loss
            m_pos = (y_target == 1)
            m_neg = (y_target == 0)
            if np.sum(m_pos) > 0 and np.sum(m_neg) > 0:
                err_pos = np.average(loss_t[m_pos], weights=p_target_norm[m_pos])
                err_neg = np.average(loss_t[m_neg], weights=p_target_norm[m_neg])
                err_t = 0.5 * (err_pos + err_neg)
            else:
                err_t = float(np.sum(p_target_norm * loss_t))

            # Penanganan error edge cases
            if err_t >= 0.5:
                err_t = self.clip_error_max
            elif err_t < self.clip_error_min:
                err_t = self.clip_error_min

            # 4. Hitung beta_t dan alpha_t
            beta_t = float(err_t / (1.0 - err_t))
            alpha_t = 0.5 * np.log(1.0 / max(beta_t, 1e-10))

            self.models.append(clf)
            self.beta_ts.append(beta_t)
            self.alphas.append(alpha_t)

            # 5. Pembaruan Bobot Sampel
            probs_all = clf.predict_proba(X_all)
            p_pos_all = probs_all[:, 1] if probs_all.shape[1] > 1 else probs_all[:, 0]
            diff = np.abs(p_pos_all - y_all)

            # Penalti instance sumber
            loss_source = diff[:n]
            w[:n] = w[:n] * np.power(self.beta_source, loss_source)

            # Peningkatan instance target
            loss_target = diff[n:]
            w[n:] = w[n:] * np.power(beta_t, -loss_target)

            # Proteksi numerik
            w = np.nan_to_num(w, nan=1e-10, posinf=1e10, neginf=1e-10)
            w = np.clip(w, 1e-15, 1e15)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Menghasilkan estimasi probabilitas kontinu kelas [0, 1] melalui ensemble terbobot.
        """
        if not self.models:
            raise RuntimeError("Model TrAdaBoost belum dilatih. Panggil fit() terlebih dahulu.")

        N = len(self.models)
        if self.use_second_half_ensemble and N > 1:
            start_idx = int(np.ceil(N / 2.0))
        else:
            start_idx = 0

        active_models = self.models[start_idx:]
        active_alphas = self.alphas[start_idx:]

        sum_alpha = sum(active_alphas)
        if sum_alpha <= 0 or np.isnan(sum_alpha):
            weights = [1.0 / len(active_models)] * len(active_models)
        else:
            weights = [a / sum_alpha for a in active_alphas]

        ensemble_probs_pos = np.zeros(len(X), dtype=np.float64)
        for clf, w in zip(active_models, weights):
            probs = clf.predict_proba(X)
            p_pos = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
            ensemble_probs_pos += w * p_pos

        ensemble_probs_pos = np.clip(ensemble_probs_pos, 0.0, 1.0)
        ensemble_probs_neg = 1.0 - ensemble_probs_pos

        return np.column_stack([ensemble_probs_neg, ensemble_probs_pos])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Menghasilkan prediksi kelas biner [0, 1] berdasarkan threshold probabilitas.
        """
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


if __name__ == "__main__":
    np.random.seed(42)
    Xs = np.random.randn(500, 10)
    ys = np.random.choice([0, 1], size=500, p=[0.95, 0.05])
    Xt = np.random.randn(50, 10)
    yt = np.random.choice([0, 1], size=50, p=[0.9, 0.1])
    X_test = np.random.randn(100, 10)

    tb = TrAdaBoostGPU(n_estimators=10, use_gpu=True)
    tb.fit(Xs, ys, Xt, yt)
    probs = tb.predict_proba(X_test)
    preds = tb.predict(X_test)
    print(f"TrAdaBoost berhasil dieksekusi! Prediksi shape: {probs.shape}, Mean Prob: {probs[:, 1].mean():.4f}, Pred Positif: {np.sum(preds)}")
