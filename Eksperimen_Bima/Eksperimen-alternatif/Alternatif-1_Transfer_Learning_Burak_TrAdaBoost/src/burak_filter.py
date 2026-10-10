"""
Modul Implementasi Algoritma Burak Filter (k-NN Instance Selection).
Rujukan:
- Turhan et al. (IEEE TSE 2009): On the relative value of cross-company data for software quality prediction.
- Afeltra et al. (IEEE Access 2024): Mitigating cross-project distribution mismatch in flaky test prediction.

Fungsi utama:
1. burak_filter_knn: Memilih union k-NN sumber untuk setiap instance target (k=10).
2. burak_filter_ratio: Memilih top R% sampel sumber yang paling dekat dengan manifold target (misal top 30% / buang 70%).
"""

from typing import Tuple, Optional
import numpy as np
from sklearn.neighbors import NearestNeighbors


def burak_filter_knn(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    k: int = 10,
    metric: str = "euclidean",
    ensure_positive: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Menerapkan Burak Filter standar (Turhan et al., 2009).
    Untuk setiap instance pada X_target, cari k tetangga terdekat pada X_source.
    Ambil union seluruh indeks tetangga terdekat tersebut.

    Args:
        X_source: Matriks fitur data sumber (N-1 proyek eksternal).
        y_source: Label biner data sumber.
        X_target: Matriks fitur data target (unseen target project).
        k: Jumlah tetangga terdekat (default: 10).
        metric: Metrik jarak (default: 'euclidean').
        ensure_positive: Jika True, pastikan setidaknya ada sampel positif (flaky) di subset.

    Returns:
        X_filtered: Subset matriks fitur sumber yang terpilih.
        y_filtered: Subset label sumber yang terpilih.
        selected_indices: Array indeks sumber yang lolos filter.
    """
    n_source = len(X_source)
    effective_k = min(k, n_source)

    # Inisialisasi NearestNeighbors
    nn = NearestNeighbors(n_neighbors=effective_k, metric=metric, algorithm="auto", n_jobs=-1)
    nn.fit(X_source)

    # Query tetangga terdekat untuk seluruh instance target secara vectorized
    _, indices = nn.kneighbors(X_target)
    selected_indices = np.unique(indices.ravel())

    # Fallback pengaman: pastikan terdapat kelas minoritas (flaky=1) jika tersedia di source
    if ensure_positive and np.sum(y_source[selected_indices] == 1) == 0:
        pos_indices = np.where(y_source == 1)[0]
        if len(pos_indices) > 0:
            # Ambil tetangga positif terdekat ke target
            nn_pos = NearestNeighbors(n_neighbors=min(5, len(pos_indices)), metric=metric, n_jobs=-1)
            nn_pos.fit(X_source[pos_indices])
            _, pos_k_idx = nn_pos.kneighbors(X_target)
            extra_pos = np.unique(pos_indices[pos_k_idx.ravel()][:5])
            selected_indices = np.unique(np.concatenate([selected_indices, extra_pos]))

    X_filtered = X_source[selected_indices]
    y_filtered = y_source[selected_indices]

    return X_filtered, y_filtered, selected_indices


def burak_filter_ratio(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    ratio: float = 0.30,
    metric: str = "euclidean",
    ensure_positive: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Menerapkan Burak Manifold Distance Ranking:
    Menghitung jarak minimum setiap sampel sumber ke titik terdekat pada manifold target.
    Memilih top (ratio * 100)% sampel sumber dengan jarak terpendek (membuang (1-ratio)*100% sampel terjauh).

    Args:
        X_source: Matriks fitur data sumber.
        y_source: Label biner data sumber.
        X_target: Matriks fitur data target.
        ratio: Rasio sampel sumber yang dipertahankan (e.g. 0.30 untuk membuang 70% data tidak relevan).
        metric: Metrik jarak.
        ensure_positive: Memastikan sampel positif terpelihara.

    Returns:
        X_filtered, y_filtered, selected_indices.
    """
    n_source = len(X_source)
    n_keep = max(1, int(round(n_source * ratio)))

    # Fit k-NN pada manifold target (1 tetangga terdekat)
    nn_target = NearestNeighbors(n_neighbors=1, metric=metric, algorithm="auto", n_jobs=-1)
    nn_target.fit(X_target)

    # Hitung jarak dari setiap sampel sumber ke manifold target
    distances, _ = nn_target.kneighbors(X_source)
    min_distances = distances.ravel()

    # Urutkan berdasarkan jarak terkecil
    sorted_source_indices = np.argsort(min_distances)
    selected_indices = sorted_source_indices[:n_keep]

    # Pastikan representasi positif
    if ensure_positive and np.sum(y_source[selected_indices] == 1) == 0:
        pos_indices = np.where(y_source == 1)[0]
        if len(pos_indices) > 0:
            # Ambil sampel positif dengan jarak terpendek ke target
            pos_sorted = pos_indices[np.argsort(min_distances[pos_indices])]
            extra_pos = pos_sorted[:min(5, len(pos_sorted))]
            selected_indices = np.unique(np.concatenate([selected_indices, extra_pos]))

    X_filtered = X_source[selected_indices]
    y_filtered = y_source[selected_indices]

    return X_filtered, y_filtered, selected_indices


if __name__ == "__main__":
    np.random.seed(42)
    Xs = np.random.randn(1000, 10)
    ys = np.random.choice([0, 1], size=1000, p=[0.95, 0.05])
    Xt = np.random.randn(50, 10)

    X_knn, y_knn, idx_knn = burak_filter_knn(Xs, ys, Xt, k=10)
    print(f"Burak k-NN (k=10): {len(idx_knn)} sampel terpilih ({len(idx_knn)/len(Xs)*100:.1f}%), Flaky: {np.sum(y_knn==1)}")

    X_rat, y_rat, idx_rat = burak_filter_ratio(Xs, ys, Xt, ratio=0.30)
    print(f"Burak Ratio (30%): {len(idx_rat)} sampel terpilih ({len(idx_rat)/len(Xs)*100:.1f}%), Flaky: {np.sum(y_rat==1)}")
