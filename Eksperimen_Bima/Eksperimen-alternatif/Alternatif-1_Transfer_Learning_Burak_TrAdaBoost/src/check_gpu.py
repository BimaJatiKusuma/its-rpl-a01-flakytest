"""
Script untuk memverifikasi ketersediaan GPU NVIDIA RTX 4050 dan akselerasi CUDA
untuk Alternatif 1: Burak Filter & TrAdaBoost.
"""

import sys
import subprocess
import os

def check_nvidia_smi():
    print("=" * 60)
    print("1. MEMERIKSA STATUS GPU (NVIDIA-SMI)")
    print("=" * 60)
    try:
        res = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            print(res.stdout)
            return True
        else:
            print("nvidia-smi error:", res.stderr)
            return False
    except FileNotFoundError:
        print("nvidia-smi tidak ditemukan di PATH.")
        return False

def check_python_environment():
    print("=" * 60)
    print("2. MEMERIKSA LINGKUNGAN RUNTIME PYTHON & PUSTAKA")
    print("=" * 60)
    print(f"Python Executable : {sys.executable}")
    print(f"Python Version    : {sys.version}")

    packages = ["numpy", "pandas", "scipy", "sklearn", "xgboost", "imblearn", "matplotlib", "seaborn"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "terpasang")
            print(f"[OK] {pkg:<15} versi: {version}")
        except ImportError:
            print(f"[BELUM TERPASANG] {pkg:<15}")

def check_xgboost_gpu():
    print("=" * 60)
    print("3. MEMERIKSA AKSELERASI GPU XGBOOST (RTX 4050 CUDA 13.4)")
    print("=" * 60)
    try:
        import xgboost as xgb
        import numpy as np

        print(f"XGBoost version: {xgb.__version__}")
        X = np.random.randn(200, 15)
        y = np.random.randint(0, 2, size=200)
        sample_weights = np.random.uniform(0.1, 1.0, size=200)

        # Uji training dengan tree_method='hist' dan device='cuda'
        clf = xgb.XGBClassifier(
            n_estimators=15,
            max_depth=4,
            tree_method="hist",
            device="cuda",
            random_state=42
        )
        clf.fit(X, y, sample_weight=sample_weights)
        preds = clf.predict_proba(X)
        print("[BERHASIL] XGBoost GPU berhasil melatih model dan inferensi dengan device='cuda' & sample_weight!")
        return True
    except Exception as e:
        print(f"[CATATAN/FALLBACK] Uji XGBoost GPU menghasilkan pesan: {e}")
        return False

if __name__ == "__main__":
    check_nvidia_smi()
    check_python_environment()
    check_xgboost_gpu()
