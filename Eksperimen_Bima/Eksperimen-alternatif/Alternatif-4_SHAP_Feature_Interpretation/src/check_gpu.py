"""
Script untuk memverifikasi ketersediaan GPU NVIDIA RTX 4050, CUDA 13.4,
dan kesiapan TreeExplainer SHAP untuk Alternatif 4: Interpretasi Fitur Universal.
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

    packages = ["numpy", "pandas", "scipy", "sklearn", "xgboost", "shap", "matplotlib", "seaborn"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "terpasang")
            print(f"[OK] {pkg:<15} versi: {version}")
        except ImportError:
            print(f"[BELUM TERPASANG] {pkg:<15}")

def check_xgboost_and_shap_gpu():
    print("=" * 60)
    print("3. MEMERIKSA AKSELERASI GPU XGBOOST & TREEEXPLAINER SHAP")
    print("=" * 60)
    try:
        import xgboost as xgb
        import shap
        import numpy as np

        print(f"XGBoost version : {xgb.__version__}")
        print(f"SHAP version    : {shap.__version__}")

        # Dummy data
        X = np.random.randn(200, 10)
        y = np.random.randint(0, 2, size=200)

        # Latih XGBoost dengan CUDA
        clf = xgb.XGBClassifier(
            n_estimators=15,
            max_depth=4,
            tree_method="hist",
            device="cuda",
            random_state=42
        )
        clf.fit(X, y)
        print("[BERHASIL] XGBoost GPU berhasil dilatih dengan device='cuda'.")

        # TreeExplainer SHAP
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer(X, check_additivity=False)
        print(f"[BERHASIL] SHAP TreeExplainer sukses mengekstrak nilai Shapley! Shape: {shap_values.values.shape}")
        return True
    except Exception as e:
        print(f"[ERROR/FALLBACK] Terjadi kendala saat pengujian: {e}")
        return False

if __name__ == "__main__":
    check_nvidia_smi()
    check_python_environment()
    check_xgboost_and_shap_gpu()
