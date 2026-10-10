"""
Script untuk memverifikasi ketersediaan GPU NVIDIA RTX 4050 dan akselerasi CUDA
untuk Alternatif 2: Validation-Based Dynamic Threshold Tuning.
"""

import sys
import subprocess
import os

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def check_nvidia_smi() -> bool:
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

    packages = ["numpy", "pandas", "scipy", "sklearn", "xgboost", "matplotlib", "seaborn"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "terpasang")
            print(f"[OK] {pkg:<15} versi: {version}")
        except ImportError:
            print(f"[BELUM TERPASANG] {pkg:<15}")


def check_xgboost_gpu() -> bool:
    print("=" * 60)
    print("3. MEMERIKSA AKSELERASI GPU XGBOOST (RTX 4050 CUDA 13.4)")
    print("=" * 60)
    try:
        import xgboost as xgb
        import numpy as np

        print(f"XGBoost version: {xgb.__version__}")
        X = np.random.randn(500, 23)
        y = np.random.randint(0, 2, size=500)

        clf = xgb.XGBClassifier(
            n_estimators=20,
            max_depth=6,
            tree_method="hist",
            device="cuda",
            random_state=42
        )
        clf.fit(X, y)
        probs = clf.predict_proba(X)
        print(f"[BERHASIL] XGBoost GPU berhasil melatih model dan inferensi {len(probs)} sampel dengan device='cuda'!")
        return True
    except Exception as e:
        print(f"[CATATAN/FALLBACK] Uji XGBoost GPU menghasilkan pesan: {e}")
        return False


if __name__ == "__main__":
    check_nvidia_smi()
    check_python_environment()
    check_xgboost_gpu()
