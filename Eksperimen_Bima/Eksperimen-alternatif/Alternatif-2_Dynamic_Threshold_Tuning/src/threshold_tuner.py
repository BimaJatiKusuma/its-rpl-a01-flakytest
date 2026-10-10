"""
Modul Dynamic Threshold Tuner untuk Eksperimen Alternatif 2:
Validation-Based Dynamic Threshold Tuning vs Default 0.5 (Zero Test Leakage).

Rujukan:
- Provost (AAAI 2000): Machine Learning from Imbalanced Data Sets: Foundational Issues and Threshold Moving.
- Alshammari et al. (ICSE 2021): FlakeFlagger Benchmark.
- Afeltra et al. (IEEE Access 2024): Cross-Project Flaky Test Prediction.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DynamicThresholdTuner:
    """
    Kelas untuk melakukan optimasi ambang batas probabilitas (threshold moving)
    berbasis validasi internal (Inner-Validation Split) pada data latih (N-1 proyek sumber).
    Menjamin ZERO DATA LEAKAGE terhadap proyek uji target.
    """

    def __init__(
        self,
        min_threshold: float = 0.01,
        max_threshold: float = 0.99,
        step: float = 0.01,
        recall_constraint: float = 0.70,
        inner_val_ratio: float = 0.20,
        random_state: int = 42
    ):
        self.threshold_grid = np.round(np.arange(min_threshold, max_threshold + step / 2, step), 4)
        self.recall_constraint = recall_constraint
        self.inner_val_ratio = inner_val_ratio
        self.random_state = random_state

    def evaluate_grid(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray
    ) -> pd.DataFrame:
        """
        Mengevaluasi seluruh metrik performa (Precision, Recall, F1, TPR, FPR, Youden's J)
        secara tervektorisasi untuk setiap nilai threshold dalam grid.

        Args:
            y_true: Ground truth label biner (0/1) pada inner validation set.
            y_probs: Probabilitas kelas positif P(y=1) pada inner validation set.

        Returns:
            df_grid: DataFrame yang berisi metrik lengkap untuk setiap threshold.
        """
        y_true = np.asarray(y_true).astype(int)
        y_probs = np.asarray(y_probs).astype(float)
        thresholds = self.threshold_grid

        # Vectorized broadcasting: (N_samples, N_thresholds)
        preds = y_probs[:, None] >= thresholds[None, :]

        y_pos = (y_true == 1)[:, None]
        y_neg = (y_true == 0)[:, None]

        tp = np.sum(preds & y_pos, axis=0)
        fp = np.sum(preds & y_neg, axis=0)
        fn = np.sum((~preds) & y_pos, axis=0)
        tn = np.sum((~preds) & y_neg, axis=0)

        # Precision & Recall
        prec_denom = tp + fp
        rec_denom = tp + fn
        fpr_denom = fp + tn

        precision = np.divide(tp, prec_denom, out=np.zeros_like(tp, dtype=float), where=prec_denom > 0)
        recall = np.divide(tp, rec_denom, out=np.zeros_like(tp, dtype=float), where=rec_denom > 0)

        # F1-Score
        f1_denom = precision + recall
        f1 = np.divide(2 * precision * recall, f1_denom, out=np.zeros_like(precision), where=f1_denom > 0)

        # TPR and FPR (Youden's J)
        tpr = recall
        fpr = np.divide(fp, fpr_denom, out=np.zeros_like(fp, dtype=float), where=fpr_denom > 0)
        specificity = 1.0 - fpr
        youdens_j = tpr - fpr

        df_grid = pd.DataFrame({
            "threshold": thresholds,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tpr": tpr,
            "fpr": fpr,
            "specificity": specificity,
            "youdens_j": youdens_j
        })
        return df_grid

    def tune_thresholds(
        self,
        y_inner_val: np.ndarray,
        p_inner_val: np.ndarray,
        y_source_full: np.ndarray
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Mencari 5 strategi threshold optimal:
        1. T1: Default 0.5
        2. T2: Max-F1
        3. T3: Recall-Constrained (Recall >= 70%)
        4. T4: Youden's J-Statistic
        5. T5: Prior-Shifted (berdasarkan proporsi kelas flaky pada data latih)

        Returns:
            optimal_thresholds: Dict pemetaan ID strategi ke nilai threshold terpilih.
            grid_metrics: DataFrame evaluasi grid search lengkap.
        """
        grid_df = self.evaluate_grid(y_inner_val, p_inner_val)

        # 1. T1 — Default 0.5
        tau_t1 = 0.50

        # 2. T2 — Max-F1
        # Cari threshold dengan F1 tertinggi; jika tie, pilih yang precision lebih tinggi
        max_f1 = grid_df["f1"].max()
        if max_f1 > 0:
            candidates_t2 = grid_df[grid_df["f1"] == max_f1]
            best_idx_t2 = candidates_t2["precision"].idxmax()
            tau_t2 = float(grid_df.loc[best_idx_t2, "threshold"])
        else:
            # Fallback jika F1 seluruhnya 0 (kasus sangat langka)
            tau_t2 = 0.50

        # 3. T3 — Recall-Constrained (Recall >= 70%)
        # Cari threshold dengan Precision tertinggi yang memenuhi Recall >= 70%
        valid_recall = grid_df[grid_df["recall"] >= self.recall_constraint]
        if len(valid_recall) > 0:
            best_idx_t3 = valid_recall["precision"].idxmax()
            tau_t3 = float(valid_recall.loc[best_idx_t3, "threshold"])
        else:
            # Fallback jika tidak ada threshold dengan Recall >= 70%: pilih recall tertinggi
            max_rec = grid_df["recall"].max()
            candidates_rec = grid_df[grid_df["recall"] == max_rec]
            best_idx_t3 = candidates_rec["precision"].idxmax()
            tau_t3 = float(grid_df.loc[best_idx_t3, "threshold"])

        # 4. T4 — Youden's J-Statistic (TPR - FPR)
        max_j = grid_df["youdens_j"].max()
        candidates_t4 = grid_df[grid_df["youdens_j"] == max_j]
        best_idx_t4 = candidates_t4["f1"].idxmax()
        tau_t4 = float(grid_df.loc[best_idx_t4, "threshold"])

        # 5. T5 — Prior-Shifted Threshold
        # Dihitung dari proporsi kelas positif empiris pada data latih sumber
        pi_flaky = float(np.mean(y_source_full))
        # Pembulatan ke 4 desimal terdekat
        tau_t5 = float(np.round(pi_flaky, 4))
        # Pastikan berada di dalam batas yang aman [0.01, 0.99]
        tau_t5 = max(0.01, min(0.99, tau_t5))

        optimal_thresholds = {
            "T1_Default": tau_t1,
            "T2_Max_F1": tau_t2,
            "T3_Recall_Constrained": tau_t3,
            "T4_Youdens_J": tau_t4,
            "T5_Prior_Shifted": tau_t5
        }

        return optimal_thresholds, grid_df

    def split_source_data(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Membagi data sumber menjadi inner train dan inner validation dengan stratifikasi kelas.
        """
        n_flaky = int(np.sum(y_source == 1))
        # Jika ada cukup flaky test untuk stratifikasi
        if n_flaky >= 5:
            X_in_train, X_in_val, y_in_train, y_in_val = train_test_split(
                X_source,
                y_source,
                test_size=self.inner_val_ratio,
                random_state=self.random_state,
                stratify=y_source
            )
        else:
            X_in_train, X_in_val, y_in_train, y_in_val = train_test_split(
                X_source,
                y_source,
                test_size=self.inner_val_ratio,
                random_state=self.random_state,
                shuffle=True
            )
        return X_in_train, X_in_val, y_in_train, y_in_val
