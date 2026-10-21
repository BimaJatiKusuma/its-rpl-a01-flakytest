"""
Modul diagnostik GPU NVIDIA GeForce RTX 4050 & lingkungan ML untuk Eksperimen 3.
"""

import sys
import subprocess
import shutil

def check_nvidia_smi():
    print("=" * 60)
    print("1. PEMERIKSAAN DRIVER & GPU VIA NVIDIA-SMI")
    print("=" * 60)
    if shutil.which("nvidia-smi") is None:
        print("[WARNING] nvidia-smi tidak ditemukan di PATH sistem.")
        return
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"[OK] GPU Terdeteksi: {res.stdout.strip()}")
    except Exception as e:
        print(f"[ERROR] Gagal menjalankan nvidia-smi: {e}")

def check_python_environment():
    print("\n" + "=" * 60)
    print("2. PEMERIKSAAN PYTHON INTERPRETER & PUSTAKA UTAMA")
    print("=" * 60)
    print(f"Python Executable : {sys.executable}")
    print(f"Python Version    : {sys.version.split()[0]}")

    modules = ["numpy", "pandas", "scipy", "sklearn", "imblearn", "xgboost", "matplotlib", "seaborn"]
    for mod in modules:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "Terpasang")
            print(f"[OK] {mod:<15}: v{ver}")
        except ImportError:
            print(f"[MISSING] {mod:<11}: Belum terpasang!")

def check_xgboost_gpu():
    print("\n" + "=" * 60)
    print("3. PEMERIKSAAN DUKUNGAN GPU DI XGBOOST")
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
        print("[OK] XGBoost GPU Acceleration (tree_method='hist', device='cuda') BERFUNGSI SEMPURNA!")
        print(f"     Output dummy prediction shape: {preds.shape}")
    except Exception as e:
        print(f"[WARNING] XGBoost GPU gagal: {e}")
        print("          Dapat beralih ke tree_method='hist' pada CPU jika diperlukan.")

if __name__ == "__main__":
    check_nvidia_smi()
    check_python_environment()
    check_xgboost_gpu()
