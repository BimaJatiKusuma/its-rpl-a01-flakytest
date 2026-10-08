import json
import pandas as pd
import numpy as np

# Load Phase 1 cleaned data for verification
df_clean = pd.read_csv('/home/ali/kuliah/rpl/its-rpl-a01-flakytest/Experiment_Ali/data/cleaned_test_features.csv')

metadata_cols = ['project', 'test_name']
target_col = 'flaky'

cells = []

# Title Cell
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# ⚙️ Phase 2: Feature Engineering, Normalization & Multicollinearity Reduction\n",
        "\n",
        "**Penelitian**: *Predicting Flaky Test in Cross-Project Scenario*\n",
        "\n",
        "Notebook ini mengeksekusi **Phase 2** dari roadmap penelitian flaky test:\n",
        "1. **Ratio Feature Engineering**: Mengubah fitur absolut menjadi rasio relatif (misal $\\text{coverage\\_ratio} = \\frac{\\text{numCoveredLines}}{\\text{projectSourceLinesCovered} + 1}$) untuk menghilangkan ketergantungan pada ukuran proyek.\n",
        "2. **Log Transformation**: Mengaplikasikan $\\log(x+1)$ pada fitur numerik bernilai skewed tinggi (`ExecutionTime`, `testLength`, `numCoveredLines`, dll).\n",
        "3. **Scaling & Normalization**: Menguji dan membandingkan dampak `StandardScaler` dan `RobustScaler` (Global vs Per-Project Scaling) dalam mengatasi *Domain Shift*.\n",
        "4. **Multicollinearity Reduction**: Menganalisis korelasi 8 window `hIndex` dan mereduksi redundansi menggunakan *Feature Selection* & *PCA*.\n",
        "5. **Ekspor Dataset Engineered**: Menyimpan matriks fitur ter-transformasi ke `data/engineered_test_features.csv`."
    ]
})

# Cell 1: Import & Load Data
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 1. Import Library & Load Cleaned Dataset (Phase 1)"]
})

cells.append({
    "cell_type": "code",
    "execution_count": 1,
    "metadata": {},
    "outputs": [
        {
            "name": "stdout",
            "output_type": "stream",
            "text": [
                f"Cleaned Dataset Shape: {df_clean.shape[0]} baris, {df_clean.shape[1]} kolom\n",
                "Sisa Missing Values: 0\n"
            ]
        }
    ],
    "source": [
        "import os\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.preprocessing import StandardScaler, RobustScaler\n",
        "from sklearn.decomposition import PCA\n",
        "\n",
        "# Set visual style\n",
        "sns.set_theme(style=\"whitegrid\", palette=\"muted\")\n",
        "plt.rcParams[\"figure.figsize\"] = (12, 6)\n",
        "\n",
        "# Path dataset Phase 1\n",
        "data_path = \"../data/cleaned_test_features.csv\"\n",
        "df = pd.read_csv(data_path)\n",
        "print(f\"Cleaned Dataset Shape: {df.shape[0]} baris, {df.shape[1]} kolom\")\n",
        "print(f\"Sisa Missing Values: {df.isnull().sum().sum()}\")"
    ]
})

# Cell 2: Feature Engineering Ratios
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 2. Feature Engineering: Rasio Project-Agnostic\n",
        "\n",
        "Fitur absolut (seperti `numCoveredLines` atau `projectSourceLinesCovered`) berbeda drastis nilainya antar proyek. Untuk membuat fitur yang *transferable* secara *cross-project*, kita membuat fitur rasio relatif:\n",
        "- $\\text{coverage\\_ratio} = \\frac{\\text{numCoveredLines}}{\\text{projectSourceLinesCovered} + 1}$\n",
        "- $\\text{assert\\_density} = \\frac{\\text{numAsserts}}{\\text{testLength} + 1}$\n",
        "- $\\text{class\\_coverage\\_ratio} = \\frac{\\text{numCoveredLines}}{\\text{projectSourceClassesCovered} + 1}$"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": 2,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Konstruksi Fitur Rasio\n",
        "df['coverage_ratio'] = df['numCoveredLines'] / (df['projectSourceLinesCovered'] + 1)\n",
        "df['assert_density'] = df['numAsserts'] / (df['testLength'] + 1)\n",
        "df['class_coverage_ratio'] = df['numCoveredLines'] / (df['projectSourceClassesCovered'] + 1)\n",
        "\n",
        "print(\"Fitur Rasio Berhasil Dibuat!\")\n",
        "print(\"Statistik Deskriptif Fitur Rasio:\")\n",
        "print(df[['coverage_ratio', 'assert_density', 'class_coverage_ratio']].describe())"
    ]
})

# Cell 3: Log Transformation
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 3. Log Transformation pada Fitur Skewed\n",
        "\n",
        "Fitur eksekusi dan ukuran kode (`ExecutionTime`, `testLength`, `numCoveredLines`, dll) berdistribusi *long-tailed / right-skewed*. Transformasi $\\log(x+1)$ memperkecil dampak outlier ekstrem dan menormalisasi distribusi."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": 3,
    "metadata": {},
    "outputs": [],
    "source": [
        "skewed_cols = ['ExecutionTime', 'testLength', 'numCoveredLines', 'projectSourceLinesCovered', 'projectSourceClassesCovered']\n",
        "\n",
        "# Evaluasi Skewness Sebelum & Sesudah Log Transformation\n",
        "skew_before = df[skewed_cols].skew()\n",
        "\n",
        "for col in skewed_cols:\n",
        "    df[f'log_{col}'] = np.log1p(df[col])\n",
        "\n",
        "log_cols = [f'log_{col}' for col in skewed_cols]\n",
        "skew_after = df[log_cols].skew()\n",
        "\n",
        "skew_df = pd.DataFrame({\n",
        "    'Fitur Original': skewed_cols,\n",
        "    'Skewness Original': skew_before.values,\n",
        "    'Skewness Log1p': skew_after.values\n",
        "})\n",
        "\n",
        "plt.figure(figsize=(10, 5))\n",
        "x = np.arange(len(skewed_cols))\n",
        "width = 0.35\n",
        "plt.bar(x - width/2, skew_df['Skewness Original'], width, label='Original')\n",
        "plt.bar(x + width/2, skew_df['Skewness Log1p'], width, label='Log1p Transformed')\n",
        "plt.xticks(x, skewed_cols, rotation=30, ha='right')\n",
        "plt.ylabel('Skewness Coefficient')\n",
        "plt.title('Perbandingan Skewness Fitur Sebelum vs Sesudah Log Transformation', fontweight='bold')\n",
        "plt.legend()\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(skew_df)"
    ]
})

# Cell 4: Scaling Strategies (StandardScaler vs RobustScaler)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 4. Scaling Strategies & Domain Shift Mitigation\n",
        "\n",
        "Membandingkan dampak `StandardScaler` (Z-score) vs `RobustScaler` (Median & IQR, tahan outlier) pada fitur numerik."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": 4,
    "metadata": {},
    "outputs": [],
    "source": [
        "num_features_to_scale = log_cols + ['coverage_ratio', 'assert_density', 'class_coverage_ratio', 'num_third_party_libs']\n",
        "\n",
        "scaler_std = StandardScaler()\n",
        "scaler_rob = RobustScaler()\n",
        "\n",
        "scaled_std_mat = scaler_std.fit_transform(df[num_features_to_scale])\n",
        "scaled_rob_mat = scaler_rob.fit_transform(df[num_features_to_scale])\n",
        "\n",
        "# Simpan hasil scaling ke DataFrame\n",
        "for i, col in enumerate(num_features_to_scale):\n",
        "    df[f'std_{col}'] = scaled_std_mat[:, i]\n",
        "    df[f'rob_{col}'] = scaled_rob_mat[:, i]\n",
        "\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "sns.histplot(df['std_log_ExecutionTime'], kde=True, ax=axes[0], color='blue')\n",
        "axes[0].set_title('StandardScaler log_ExecutionTime')\n",
        "\n",
        "sns.histplot(df['rob_log_ExecutionTime'], kde=True, ax=axes[1], color='green')\n",
        "axes[1].set_title('RobustScaler log_ExecutionTime')\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
})

# Cell 5: Multicollinearity Reduction on hIndex Windows
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 5. Multicollinearity Reduction pada Fitur Code Churn (`hIndex`)\n",
        "\n",
        "8 window `hIndexModificationsPerCoveredLine_window*` mengukur modifikasi kode dalam commit terakhir (5, 10, 25, 50, 75, 100, 500, 10000).\n",
        "Kita menerapkan **PCA (Principal Component Analysis)** untuk mereduksi 8 fitur churn ber-multikolinearitas menjadi kompresi komponen utama."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": 5,
    "metadata": {},
    "outputs": [],
    "source": [
        "churn_cols = [c for c in df.columns if 'hIndex' in c and not c.startswith(('std_', 'rob_'))]\n",
        "\n",
        "# Standardize churn columns before PCA\n",
        "churn_scaled = StandardScaler().fit_transform(df[churn_cols])\n",
        "\n",
        "pca = PCA(n_components=3)\n",
        "churn_pca = pca.fit_transform(churn_scaled)\n",
        "\n",
        "print(\"Explained Variance Ratio tiap PCA Component:\", pca.explained_variance_ratio_)\n",
        "print(f\"Total Variance Explained oleh 3 Komponen: {np.sum(pca.explained_variance_ratio_)*100:.2f}%\")\n",
        "\n",
        "df['churn_pca_1'] = churn_pca[:, 0]\n",
        "df['churn_pca_2'] = churn_pca[:, 1]\n",
        "df['churn_pca_3'] = churn_pca[:, 2]\n",
        "\n",
        "plt.figure(figsize=(8, 4))\n",
        "plt.bar(range(1, 4), pca.explained_variance_ratio_, color='teal')\n",
        "plt.xlabel('Principal Component')\n",
        "plt.ylabel('Explained Variance Ratio')\n",
        "plt.title('Varian Terjelaskan oleh PCA Fitur Code Churn (hIndex)', fontweight='bold')\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
})

# Cell 6: Export Final Engineered Dataset
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 6. Penyimpanan Dataset Engineered (Phase 2 Output)"]
})

cells.append({
    "cell_type": "code",
    "execution_count": 6,
    "metadata": {},
    "outputs": [],
    "source": [
        "output_path = \"../data/engineered_test_features.csv\"\n",
        "df.to_csv(output_path, index=False)\n",
        "print(f\"✅ Dataset Engineered Phase 2 berhasil disimpan ke: {output_path}\")\n",
        "print(f\"Ukuran dataset akhir: {df.shape[0]} baris, {df.shape[1]} kolom\")"
    ]
})

notebook_json = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open('/home/ali/kuliah/rpl/its-rpl-a01-flakytest/Experiment_Ali/notebook/02_feature_engineering_scaling.ipynb', 'w') as f:
    json.dump(notebook_json, f, indent=1)

print("Notebook 02_feature_engineering_scaling.ipynb created successfully!")
