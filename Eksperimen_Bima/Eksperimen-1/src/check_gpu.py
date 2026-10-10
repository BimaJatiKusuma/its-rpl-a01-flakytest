"""
Script untuk memverifikasi ketersediaan GPU NVIDIA RTX 4050 dan akselerasi CUDA.
"""

import sys
import subprocess
import os

def check_nvidia_smi():
    print("=" * 60)
    print("1. MEMERIKSA NVIDIA-SMI")
    print("=" * 60)
    try:
        res = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            print(res.stdout)
            return True
        else:
            print("nvidia-smi gagal dijalankan:", res.stderr)
            return False
    except FileNotFoundError:
        print("nvidia-smi tidak ditemukan di PATH sistem.")
        return False

def check_python_environment():
    print("=" * 60)
    print("2. MEMERIKSA LINGKUNGAN PYTHON & PUSTAKA ML")
    print("=" * 60)
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version   : {sys.version}")

    packages = ["numpy", "pandas", "scipy", "sklearn", "xgboost", "imblearn", "matplotlib", "seaborn"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "installed")
            print(f"[OK] {pkg:<15} versi: {version}")
        except ImportError:
            print(f"[NOT INSTALLED] {pkg:<15}")

def check_xgboost_gpu():
    print("=" * 60)
    print("3. MEMERIKSA AKSELERASI GPU XGBOOST (RTX 4050)")
    print("=" * 60)
    try:
        import xgboost as xgb
        import numpy as np

        print(f"XGBoost version: {xgb.__version__}")
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, size=100)

        # Test training with device='cuda'
        clf = xgb.XGBClassifier(
            n_estimators=10,
            tree_method="hist",
            device="cuda",
            random_state=42
        )
        clf.fit(X, y)
        preds = clf.predict_proba(X)
        print("[SUCCESS] XGBoost berhasil melatih model menggunakan GPU (device='cuda')!")
    except Exception as e:
        print(f"[NOTE/FALLBACK] XGBoost GPU test menghasilkan pesan: {e}")
        print("Jika CUDA belum terikat ke XGBoost, XGBoost dapat dijalankan dengan tree_method='hist' pada CPU.")

if __name__ == "__main__":
    check_nvidia_smi()
    check_python_environment()
    check_xgboost_gpu()
