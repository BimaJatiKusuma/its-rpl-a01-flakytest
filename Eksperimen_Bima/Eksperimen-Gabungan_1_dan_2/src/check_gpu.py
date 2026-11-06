"""
Hardware diagnostic module for NVIDIA GeForce RTX GPU acceleration and Python ML runtime environment.
"""

import sys
import subprocess
import shutil

def check_nvidia_smi():
    print("=" * 60)
    print("1. HARDWARE & DRIVER DIAGNOSTICS VIA NVIDIA-SMI")
    print("=" * 60)
    if shutil.which("nvidia-smi") is None:
        print("[WARNING] 'nvidia-smi' executable was not found in the system PATH.")
        return
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"[OK] Detected GPU: {res.stdout.strip()}")
    except Exception as e:
        print(f"[ERROR] Failed to execute nvidia-smi: {e}")

def check_python_environment():
    print("\n" + "=" * 60)
    print("2. PYTHON RUNTIME & DEPENDENCY VERIFICATION")
    print("=" * 60)
    print(f"Python Executable : {sys.executable}")
    print(f"Python Version    : {sys.version.split()[0]}")

    modules = ["numpy", "pandas", "scipy", "sklearn", "imblearn", "xgboost", "matplotlib", "seaborn"]
    for mod in modules:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "Installed")
            print(f"[OK] {mod:<15}: v{ver}")
        except ImportError:
            print(f"[MISSING] {mod:<11}: Not installed!")

def check_xgboost_gpu():
    print("\n" + "=" * 60)
    print("3. XGBOOST CUDA GPU ACCELERATION CHECK")
    print("=" * 60)
    try:
        import xgboost as xgb
        import numpy as np

        X_dummy = np.random.rand(100, 10).astype(np.float32)
        y_dummy = np.random.randint(0, 2, size=100)

        clf = xgb.XGBClassifier(
            n_estimators=10,
            tree_method="hist",
            device="cuda",
            random_state=42
        )
        clf.fit(X_dummy, y_dummy)
        preds = clf.predict_proba(X_dummy)
        print("[OK] XGBoost GPU Acceleration (tree_method='hist', device='cuda') is fully functional!")
        print(f"     Dummy prediction probability output shape: {preds.shape}")
    except Exception as e:
        print(f"[WARNING] XGBoost GPU acceleration check failed: {e}")
        print("          Falling back to CPU tree_method='hist' if needed.")

if __name__ == "__main__":
    check_nvidia_smi()
    check_python_environment()
    check_xgboost_gpu()
