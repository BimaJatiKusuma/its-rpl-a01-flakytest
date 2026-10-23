"""
Modul Implementasi Algoritma Burak Filter (k-NN Instance Selection).
Rujukan:
- Turhan et al. (IEEE TSE 2009): On the relative value of cross-company data for software quality prediction.
- Afeltra et al. (IEEE Access 2024): A Large-Scale Empirical Investigation Into Cross-Project Flaky Test Prediction (Section IV.B, k=10).
"""

from typing import Tuple
import numpy as np
from sklearn.neighbors import NearestNeighbors

def apply_burak_filter(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    k: int = 10,
    metric: str = "euclidean",
    ensure_minority: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Menerapkan Burak Filter standar (Turhan et al., 2009; Afeltra et al., 2024).
    Untuk setiap instance pada X_target, cari k tetangga terdekat pada X_source.
    Ambil union seluruh indeks tetangga terdekat tersebut.

    Args:
        X_source: Matriks fitur data sumber (scaled).
        y_source: Label biner data sumber.
        X_target: Matriks fitur data target (scaled).
        k: Jumlah tetangga terdekat (default: 10, mengacu paper Afeltra 2024).
        metric: Metrik jarak ('euclidean').
        ensure_minority: Jika True, pastikan ada sampel minoritas (flaky=1) jika tersedia di source.

    Returns:
        X_filtered: Subset matriks fitur sumber yang terpilih.
        y_filtered: Subset label sumber yang terpilih.
        selected_indices: Array indeks sumber yang lolos filter.
    """
    n_source = len(X_source)
    if n_source == 0:
        return X_source, y_source, np.array([], dtype=int)

    effective_k = min(k, n_source)

    # Inisialisasi NearestNeighbors
    nn = NearestNeighbors(n_neighbors=effective_k, metric=metric, algorithm="auto", n_jobs=-1)
    nn.fit(X_source)

    # Query tetangga terdekat untuk seluruh instance target secara vectorized
    _, indices = nn.kneighbors(X_target)
    selected_indices = np.unique(indices.ravel())

    # Fallback pengaman: pastikan terdapat kelas minoritas jika tersedia di source
    if ensure_minority and np.sum(y_source[selected_indices] == 1) == 0:
        pos_indices = np.where(y_source == 1)[0]
        if len(pos_indices) > 0:
            nn_pos = NearestNeighbors(n_neighbors=min(effective_k, len(pos_indices)), metric=metric, n_jobs=-1)
            nn_pos.fit(X_source[pos_indices])
            _, pos_k_idx = nn_pos.kneighbors(X_target)
            extra_pos = np.unique(pos_indices[pos_k_idx.ravel()][:min(10, len(pos_indices))])
            selected_indices = np.unique(np.concatenate([selected_indices, extra_pos]))

    X_filtered = X_source[selected_indices]
    y_filtered = y_source[selected_indices]

    return X_filtered, y_filtered, selected_indices
