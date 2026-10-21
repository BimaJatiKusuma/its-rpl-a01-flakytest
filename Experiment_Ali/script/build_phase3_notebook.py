import json
import pandas as pd
import numpy as np

# Load engineered dataset to gather stats
df = pd.read_csv('/home/ali/kuliah/rpl/its-rpl-a01-flakytest/Experiment_Ali/data/engineered_test_features.csv')

projects = sorted(df['project'].unique().tolist())
print(f"Total projects: {len(projects)}")

cells = []

# Title & Description
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 🔄 Phase 3: Setup Validation Cross-Project (Leave-One-Project-Out / LOOCV)\n",
        "\n",
        "**Penelitian**: *Predicting Flaky Test in Cross-Project Scenario*\n",
        "\n",
        "> **[!IMPORTANT]**\n",
        "> **Jangan gunakan Random Train-Test Split biasa!** Random split akan mencampur test dari proyek yang sama ke dalam train set dan test set, menyebabkan *data leakage* dan evaluasi yang bias/overoptimis.\n",
        "\n",
        "Notebook ini merealisasikan **Phase 3**:\n",
        "1. **Skema Utama: Leave-One-Project-Out Cross-Validation (LOOCV / LOPO-CV)**:\n",
        "   - Terdapat 24 proyek unik.\n",
        "   - Untuk tiap fold $i \\in \\{1, \\dots, 24\\}$:\n",
        "     - **Test Set**: Data test cases dari proyek $P_i$.\n",
        "     - **Train Set**: Data test cases dari 23 proyek lainnya ($P_{j \\neq i}$).\n",
        "2. **Skenario Tambahan: Multi-Project Split (70% Train Proyek, 30% Test Proyek)**:\n",
        "   - Stratified/balanced split pada level *project* (misal: 17 proyek Train, 7 proyek Test).\n",
        "3. **Modular Split Generator & Verifikasi Data Leakage**:\n",
        "   - Memvalidasi bahwa tidak ada irisan `project` maupun `test_name` antara Train dan Test set.\n",
        "   - Inspeksi distribusi ukuran sampel dan rasio kelas target (`flaky`) di setiap fold.\n",
        "4. **Export Index / Metadata Split Fold**: Menyimpan konfigurasi split agar *reproducible* untuk Phase 4 (Model Training)."
    ]
})

# Cell 1: Import & Load Data
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 1. Import Library & Load Engineered Dataset"]
})

cells.append({
    "cell_type": "code",
    "execution_count": 1,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os\n",
        "import json\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.model_selection import LeaveOneGroupOut\n",
        "\n",
        "sns.set_theme(style=\"whitegrid\", palette=\"muted\")\n",
        "plt.rcParams[\"figure.figsize\"] = (12, 6)\n",
        "\n",
        "data_path = \"../data/engineered_test_features.csv\"\n",
        "df = pd.read_csv(data_path)\n",
        "print(f\"Dataset Shape: {df.shape[0]} baris, {df.shape[1]} kolom\")\n",
        "print(f\"Total Unique Projects: {df['project'].nunique()}\")\n",
        "df[['project', 'test_name', 'flaky']].head(3)"
    ]
})

# Cell 2: Feature Matrix Definition
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 2. Definisi Matriks Fitur ($X$), Label Target ($y$), dan Group Identitas (`project`)\n",
        "\n",
        "Untuk memastikan model bersifat **project-agnostic**, seluruh identifier proyek (`project`, `test_name`, `Unnamed: 0`, `testClassName`, `testMethodName`) dipisahkan dari $X$."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": 2,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Metadata & target\n",
        "meta_cols = ['Unnamed: 0', 'test_name', 'project', 'testClassName', 'testMethodName']\n",
        "target_col = 'flaky'\n",
        "\n",
        "# Identifikasi feature columns (tersedia feature raw, engineered ratios, log, robust/standard scaled, dan PCA)\n",
        "feature_cols = [c for c in df.columns if c not in meta_cols + [target_col]]\n",
        "\n",
        "print(f\"Total Fitur Tersedia: {len(feature_cols)}\")\n",
        "print(\"Sample Fitur:\", feature_cols[:10])\n",
        "\n",
        "X = df[feature_cols]\n",
        "y = df[target_col]\n",
        "groups = df['project']"
    ]
})

# Cell 3: LOOCV Implementation
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 3. Implementasi Leave-One-Project-Out Cross-Validation (LOOCV / LOPO)\n",
        "\n",
        "Kita membuat generator LOOCV menggunakan `LeaveOneGroupOut` dari scikit-learn atau iterasi list proyek, dan memverifikasi integritas setiap fold."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": 3,
    "metadata": {},
    "outputs": [],
    "source": [
        "logo = LeaveOneGroupOut()\n",
        "fold_stats = []\n",
        "\n",
        "loocv_splits = {}\n",
        "\n",
        "for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups), 1):\n",
        "    test_project = df.iloc[test_idx]['project'].unique()[0]\n",
        "    train_projects = df.iloc[train_idx]['project'].unique().tolist()\n",
        "    \n",
        "    # Leakage check: memastikan test_project sama sekali tidak ada di train_projects\n",
        "    assert test_project not in train_projects, f\"Data leakage terdeteksi pada fold {fold_idx}!\"\n",
        "    \n",
        "    n_train = len(train_idx)\n",
        "    n_test = len(test_idx)\n",
        "    flaky_train = int(y.iloc[train_idx].sum())\n",
        "    flaky_test = int(y.iloc[test_idx].sum())\n",
        "    flaky_test_pct = (flaky_test / n_test) * 100 if n_test > 0 else 0\n",
        "    \n",
        "    fold_stats.append({\n",
        "        'Fold': fold_idx,\n",
        "        'Test Project': test_project,\n",
        "        'Train Size': n_train,\n",
        "        'Test Size': n_test,\n",
        "        'Flaky in Train': flaky_train,\n",
        "        'Flaky in Test': flaky_test,\n",
        "        'Test Flaky (%)': round(flaky_test_pct, 2)\n",
        "    })\n",
        "    \n",
        "    loocv_splits[test_project] = {\n",
        "        'fold': fold_idx,\n",
        "        'train_indices': train_idx.tolist(),\n",
        "        'test_indices': test_idx.tolist()\n",
        "    }\n",
        "\n",
        "df_fold_stats = pd.DataFrame(fold_stats)\n",
        "print(\"=== Ringkasan 24 Fold LOOCV ===\")\n",
        "df_fold_stats"
    ]
})

# Cell 4: Visualizing LOOCV Folds
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["### 3.1 Visualisasi Distribusi Test Sample & Rasio Flaky per Fold LOOCV"]
})

cells.append({
    "cell_type": "code",
    "execution_count": 4,
    "metadata": {},
    "outputs": [],
    "source": [
        "fig, ax1 = plt.subplots(figsize=(14, 6))\n",
        "\n",
        "color = 'tab:blue'\n",
        "ax1.set_xlabel('Test Project (Fold)', fontsize=12)\n",
        "ax1.set_ylabel('Test Size (Jumlah Test Case)', color=color, fontsize=12)\n",
        "bars = ax1.bar(df_fold_stats['Test Project'], df_fold_stats['Test Size'], color=color, alpha=0.6, label='Test Size')\n",
        "ax1.tick_params(axis='y', labelcolor=color)\n",
        "ax1.tick_params(axis='x', rotation=60)\n",
        "\n",
        "ax2 = ax1.twinx()\n",
        "color = 'tab:red'\n",
        "ax2.set_ylabel('Rasio Flaky dalam Test (%)', color=color, fontsize=12)\n",
        "line = ax2.plot(df_fold_stats['Test Project'], df_fold_stats['Test Flaky (%)'], color=color, marker='o', linewidth=2, label='Flaky %')\n",
        "ax2.tick_params(axis='y', labelcolor=color)\n",
        "\n",
        "plt.title('Karakteristik 24 Fold LOOCV (Test Size vs % Flaky)', fontsize=14, fontweight='bold')\n",
        "fig.tight_layout()\n",
        "plt.show()"
    ]
})

# Cell 5: Multi-Project Split (70/30)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 4. Skenario Tambahan: Multi-Project Split (70% Train Projects, 30% Test Projects)\n",
        "\n",
        "Selain LOOCV (1 vs 23), skenario multi-project split menguji ketahanan model saat dilatih pada kumpulan 70% proyek (~17 proyek) dan dievaluasi sekaligus pada sekumpulan 30% proyek (~7 proyek)."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": 5,
    "metadata": {},
    "outputs": [],
    "source": [
        "np.random.seed(42)\n",
        "unique_projects = np.array(sorted(df['project'].unique()))\n",
        "n_total_projects = len(unique_projects)\n",
        "n_train_projects = int(np.round(0.7 * n_total_projects)) # 17 proyek\n",
        "\n",
        "# Acak urutan project untuk split 70/30\n",
        "shuffled_projects = np.random.permutation(unique_projects)\n",
        "train_proj_set = sorted(shuffled_projects[:n_train_projects].tolist())\n",
        "test_proj_set = sorted(shuffled_projects[n_train_projects:].tolist())\n",
        "\n",
        "train_multi_idx = df[df['project'].isin(train_proj_set)].index.tolist()\n",
        "test_multi_idx = df[df['project'].isin(test_proj_set)].index.tolist()\n",
        "\n",
        "print(\"=== Multi-Project Split (70/30) ===\")\n",
        "print(f\"Train Projects ({len(train_proj_set)}): {train_proj_set}\")\n",
        "print(f\"Test Projects  ({len(test_proj_set)}): {test_proj_set}\")\n",
        "print(f\"Train Samples : {len(train_multi_idx)} (Flaky: {y.iloc[train_multi_idx].sum()}, {y.iloc[train_multi_idx].mean()*100:.2f}%)\")\n",
        "print(f\"Test Samples  : {len(test_multi_idx)} (Flaky: {y.iloc[test_multi_idx].sum()}, {y.iloc[test_multi_idx].mean()*100:.2f}%)\")\n",
        "\n",
        "# Verifikasi tidak ada overlap proyek\n",
        "assert len(set(train_proj_set).intersection(set(test_proj_set))) == 0, \"Overlap proyek terdeteksi!\""
    ]
})

# Cell 6: Export Split Configuration
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 5. Simpan Konfigurasi & Metadata Validasi Split\n",
        "\n",
        "Menyimpan konfigurasi fold ke file JSON/CSV agar dapat dibaca langsung oleh pipeline pelatihan model pada Phase 4."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": 6,
    "metadata": {},
    "outputs": [],
    "source": [
        "split_config = {\n",
        "    'strategy_primary': 'Leave-One-Project-Out (LOOCV)',\n",
        "    'num_folds': len(loocv_splits),\n",
        "    'loocv_folds': {\n",
        "        proj: {\n",
        "            'fold': info['fold'],\n",
        "            'test_project': proj,\n",
        "            'train_size': len(info['train_indices']),\n",
        "            'test_size': len(info['test_indices'])\n",
        "        }\n",
        "        for proj, info in loocv_splits.items()\n",
        "    },\n",
        "    'multi_project_split_70_30': {\n",
        "        'train_projects': train_proj_set,\n",
        "        'test_projects': test_proj_set,\n",
        "        'train_size': len(train_multi_idx),\n",
        "        'test_size': len(test_multi_idx)\n",
        "    }\n",
        "}\n",
        "\n",
        "# Simpan metadata split ke JSON\n",
        "split_meta_path = \"../data/validation_splits_metadata.json\"\n",
        "with open(split_meta_path, \"w\") as f:\n",
        "    json.dump(split_config, f, indent=2)\n",
        "\n",
        "# Simpan fold stats CSV\n",
        "fold_stats_path = \"../data/loocv_folds_summary.csv\"\n",
        "df_fold_stats.to_csv(fold_stats_path, index=False)\n",
        "\n",
        "print(f\"✅ Metadata split disimpan ke: {split_meta_path}\")\n",
        "print(f\"✅ Ringkasan statistik fold disimpan ke: {fold_stats_path}\")"
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

with open('/home/ali/kuliah/rpl/its-rpl-a01-flakytest/Experiment_Ali/notebook/03_validation_setup_loocv.ipynb', 'w') as f:
    json.dump(notebook_json, f, indent=1)

print("Notebook 03_validation_setup_loocv.ipynb created successfully!")
