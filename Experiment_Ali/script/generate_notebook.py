import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🔍 Phase 1: Data Preparation, Missing Value Imputation & EDA per Project\n",
    "\n",
    "**Penelitian**: *Predicting Flaky Test in Cross-Project Scenario*\n",
    "\n",
    "Notebook ini mengeksekusi **Phase 1** dari roadmap penelitian, yang meliputi:\n",
    "1. **Load Data**: Membaca dataset `data/test_features.csv`.\n",
    "2. **Pemisahan Metadata vs Fitur**:\n",
    "   - Mengisolasi `project` sebagai variabel grup/stratifikasi evaluasi.\n",
    "   - Mengisolasi `test_name` sebagai *primary key* pelacakan.\n",
    "   - Mengeliminasi fitur identitas berlebih (`Unnamed: 0`, `testClassName`, `testMethodName`).\n",
    "3. **Analisis & Imputasi Missing Values**:\n",
    "   - Deteksi data hilang/kosong pada fitur.\n",
    "   - Evaluasi dan penerapan **Imputasi Probabilitas / Per-Project Median Imputation** untuk mempertahankan distribusi asli per proyek.\n",
    "4. **Exploratory Data Analysis (EDA) per Proyek**:\n",
    "   - Distribusi target `flaky` per proyek.\n",
    "   - Statistik deskriptif dan deteksi *domain shift* (perbedaan skala & outlier) antar proyek.\n",
    "   - Analisis multikolinearitas (terutama fitur `hIndex` dan *Coverage*).\n",
    "5. **Ekspor Dataset Bersih** (`data/cleaned_test_features.csv`)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Import Library & Load Raw Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Set style untuk visualisasi\n",
    "sns.set_theme(style=\"whitegrid\", palette=\"muted\")\n",
    "plt.rcParams[\"figure.figsize\"] = (12, 6)\n",
    "plt.rcParams[\"font.size\"] = 10\n",
    "\n",
    "# Path dataset\n",
    "data_path = \"../data/test_features.csv\"\n",
    "df_raw = pd.read_csv(data_path)\n",
    "\n",
    "print(f\"Dataset Shape: {df_raw.shape[0]} baris, {df_raw.shape[1]} kolom\")\n",
    "df_raw.head(3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Pemisahan Metadata, Primary Key, dan Matriks Fitur ($X$)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Kolom Identitas & Metadata\n",
    "metadata_cols = ['project', 'test_name']\n",
    "dropped_cols = ['Unnamed: 0', 'testClassName', 'testMethodName']\n",
    "\n",
    "# Kolom Target\n",
    "target_col = 'flaky'\n",
    "\n",
    "# Fitur-fitur Prediktif (22 Fitur)\n",
    "feature_cols = [c for c in df_raw.columns if c not in metadata_cols + dropped_cols + [target_col]]\n",
    "\n",
    "print(f\"Total Fitur Prediktif: {len(feature_cols)} fitur\")\n",
    "print(\"\\nPengelompokan Fitur:\")\n",
    "test_smells = [c for c in feature_cols if c in [\n",
    "    'assertion-roulette', 'conditional-test-logic', 'eager-test', 'fire-and-forget',\n",
    "    'indirect-testing', 'mystery-guest', 'resource-optimism', 'test-run-war'\n",
    "]]\n",
    "test_metrics = ['testLength', 'numAsserts', 'numCoveredLines', 'ExecutionTime']\n",
    "coverage_features = ['projectSourceLinesCovered', 'projectSourceClassesCovered']\n",
    "churn_features = [c for c in feature_cols if 'hIndex' in c]\n",
    "dep_features = ['num_third_party_libs']\n",
    "\n",
    "print(f\"- Test Smells ({len(test_smells)}): {test_smells}\")\n",
    "print(f\"- Test Metrics ({len(test_metrics)}): {test_metrics}\")\n",
    "print(f\"- Coverage Features ({len(coverage_features)}): {coverage_features}\")\n",
    "print(f\"- Code Churn Features ({len(churn_features)}): {churn_features}\")\n",
    "print(f\"- Dependency ({len(dep_features)}): {dep_features}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Analisis Missing Values & Strategy Imputasi Probabilitas / Per-Project"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Pengecekan Missing Value\n",
    "null_counts = df_raw[feature_cols].isnull().sum()\n",
    "null_summary = null_counts[null_counts > 0]\n",
    "print(\"=== Ringkasan Missing Values ===\")\n",
    "print(null_summary if len(null_summary) > 0 else \"Tidak ada missing values!\")\n",
    "\n",
    "# Detail sebaran missing values per proyek & target\n",
    "if len(null_summary) > 0:\n",
    "    missing_df = df_raw[df_raw[null_summary.index[0]].isnull()]\n",
    "    print(\"\\nMissing values per proyek:\")\n",
    "    print(missing_df['project'].value_counts())\n",
    "    print(\"\\nMissing values per status target (flaky):\")\n",
    "    print(missing_df['flaky'].value_counts())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Imputasi Fitur yang Hilang (`testLength`, `numAsserts`, `numCoveredLines`)\n",
    "\n",
    "Karena data missing bersifat spesifik per proyek (misal mayoritas di `wildfly` dan `logback`), imputasi global akan merusak distribusi spesifik proyek.\n",
    "Kita menerapkan **Group Median Imputation (Per-Project Median)** yang dikombinasikan dengan **Empirical Distribution Sampling (Imputasi Probabilitas)** jika median bernilai NaN."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def impute_project_aware(df, cols_to_impute):\n",
    "    df_imputed = df.copy()\n",
    "    for col in cols_to_impute:\n",
    "        # 1. Imputasi dengan Median per Proyek\n",
    "        df_imputed[col] = df_imputed.groupby('project')[col].transform(lambda x: x.fillna(x.median()))\n",
    "        \n",
    "        # 2. Fallback jika masih ada NaN (Imputasi Probabilitas / Random Sampling dari nilai non-null)\n",
    "        if df_imputed[col].isnull().sum() > 0:\n",
    "            non_null_vals = df_imputed[col].dropna().values\n",
    "            nan_indices = df_imputed[df_imputed[col].isnull()].index\n",
    "            sampled_vals = np.random.choice(non_null_vals, size=len(nan_indices), replace=True)\n",
    "            df_imputed.loc[nan_indices, col] = sampled_vals\n",
    "            \n",
    "    return df_imputed\n",
    "\n",
    "cols_missing = null_summary.index.tolist()\n",
    "df_clean = impute_project_aware(df_raw, cols_missing)\n",
    "\n",
    "print(\"=== Verifikasi Setelah Imputasi ===\")\n",
    "print(f\"Sisa Missing Values: {df_clean[feature_cols].isnull().sum().sum()}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Exploratory Data Analysis (EDA) per Proyek"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4.1 Distribusi Flaky Test per Proyek"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "project_stats = df_clean.groupby('project')['flaky'].agg(\n",
    "    total_tests='count',\n",
    "    flaky_count='sum',\n",
    "    flaky_ratio=lambda x: x.mean() * 100\n",
    ").reset_index().sort_values(by='flaky_ratio', ascending=False)\n",
    "\n",
    "plt.figure(figsize=(14, 6))\n",
    "ax = sns.barplot(data=project_stats, x='project', y='flaky_ratio', palette='viridis')\n",
    "plt.xticks(rotation=60, ha='right')\n",
    "plt.ylabel('Flaky Test Ratio (%)')\n",
    "plt.title('Rasio Flaky Test per Proyek (24 Open-Source Projects)', fontsize=14, fontweight='bold')\n",
    "for p in ax.patches:\n",
    "    if p.get_height() > 0:\n",
    "        ax.annotate(f\"{p.get_height():.1f}%\", (p.get_x() + p.get_width() / 2., p.get_height()),\n",
    "                    ha='center', va='bottom', fontsize=8, rotation=0, xytext=(0, 2), textcoords='offset points')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "project_stats"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4.2 Inspeksi Skala & Domain Shift Fitur Absolut Lintas Proyek"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Boxplot Distribusi Fitur Absolut Utama Lintas Proyek\n",
    "fig, axes = plt.subplots(2, 2, figsize=(16, 10))\n",
    "\n",
    "sns.boxplot(data=df_clean, x='project', y='ExecutionTime', ax=axes[0,0])\n",
    "axes[0,0].set_yscale('log')\n",
    "axes[0,0].set_title('ExecutionTime (Log Scale)')\n",
    "axes[0,0].tick_params(axis='x', rotation=90)\n",
    "\n",
    "sns.boxplot(data=df_clean, x='project', y='numCoveredLines', ax=axes[0,1])\n",
    "axes[0,1].set_yscale('log')\n",
    "axes[0,1].set_title('numCoveredLines (Log Scale)')\n",
    "axes[0,1].tick_params(axis='x', rotation=90)\n",
    "\n",
    "sns.boxplot(data=df_clean, x='project', y='projectSourceLinesCovered', ax=axes[1,0])\n",
    "axes[1,0].set_yscale('log')\n",
    "axes[1,0].set_title('projectSourceLinesCovered (Log Scale)')\n",
    "axes[1,0].tick_params(axis='x', rotation=90)\n",
    "\n",
    "sns.boxplot(data=df_clean, x='project', y='testLength', ax=axes[1,1])\n",
    "axes[1,1].set_yscale('log')\n",
    "axes[1,1].set_title('testLength (Log Scale)')\n",
    "axes[1,1].tick_params(axis='x', rotation=90)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4.3 Analisis Multikolinearitas (Korelasi Antar Fitur)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(14, 10))\n",
    "corr = df_clean[feature_cols].corr()\n",
    "sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1, annot=False, linewidths=0.5)\n",
    "plt.title('Heatmap Korelasi 22 Fitur Prediktif', fontsize=14, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Simpan Dataset Hasil Preprocessing Phase 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "output_path = \"../data/cleaned_test_features.csv\"\n",
    "df_clean.to_csv(output_path, index=False)\n",
    "print(f\"✅ Dataset bersih berhasil disimpan ke: {output_path}\")\n",
    "print(f\"Ukuran akhir: {df_clean.shape[0]} baris, {df_clean.shape[1]} kolom\")"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open('/home/ali/kuliah/rpl/its-rpl-a01-flakytest/Experiment_Ali/script/generate_notebook.py', 'w') as f:
    f.write('import json\nwith open("../notebook/01_eda_and_feature_inspection.ipynb", "w") as out:\n    json.dump(' + repr(notebook) + ', out, indent=1)\nprint("Notebook template generated successfully!")\n')
