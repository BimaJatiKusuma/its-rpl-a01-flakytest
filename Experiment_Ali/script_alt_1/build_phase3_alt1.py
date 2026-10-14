import json
import numpy as np
import pandas as pd

# Load engineered dataset
df = pd.read_csv('/home/ali/kuliah/rpl/its-rpl-a01-flakytest/Experiment_Ali/data_alt_1/engineered_test_features.csv')
projects = np.array(sorted(df['project'].unique()))
n_projects = len(projects)

# 90% train (22 projects), 10% test (2 projects)
np.random.seed(42)
shuffled = np.random.permutation(projects)
n_train = 22
n_test = 2

train_projects = sorted(shuffled[:n_train].tolist())
test_projects = sorted(shuffled[n_train:].tolist())

train_mask = df['project'].isin(train_projects)
test_mask = df['project'].isin(test_projects)

train_indices = df[train_mask].index.tolist()
test_indices = df[test_mask].index.tolist()

# Save metadata
split_meta = {
    'strategy': 'Multi-Project Split (90% Train, 10% Test)',
    'total_projects': n_projects,
    'n_train_projects': n_train,
    'n_test_projects': n_test,
    'train_projects': train_projects,
    'test_projects': test_projects,
    'train_samples': len(train_indices),
    'test_samples': len(test_indices),
    'train_flaky': int(df.loc[train_indices, 'flaky'].sum()),
    'test_flaky': int(df.loc[test_indices, 'flaky'].sum()),
    'train_flaky_pct': round((df.loc[train_indices, 'flaky'].mean()) * 100, 2),
    'test_flaky_pct': round((df.loc[test_indices, 'flaky'].mean()) * 100, 2)
}

with open('/home/ali/kuliah/rpl/its-rpl-a01-flakytest/Experiment_Ali/data_alt_1/validation_splits_metadata_90_10.json', 'w') as f:
    json.dump(split_meta, f, indent=2)

print("Split metadata 90/10 saved successfully!")

# Build Notebook 03_validation_setup_90_10.ipynb
cells = []

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 🔄 Phase 3 (Alt 1): Setup Validation Cross-Project (90% Train Projects, 10% Test Projects)\n",
        "\n",
        "**Penelitian**: *Predicting Flaky Test in Cross-Project Scenario (Alternative 1)*\n",
        "\n",
        "> **[!IMPORTANT]**\n",
        "> Pemisahan train-test dilakukan secara ketat pada level **proyek (project-level split)**, bukan random test split, untuk memastikan model diuji pada proyek *unseen* tanpa *data leakage*.\n",
        "\n",
        "Notebook ini merealisasikan alternatif validasi:\n",
        "- **Rasio Proyek**: **90% Proyek Pelatihan** (22 proyek) dan **10% Proyek Pengujian** (2 proyek).\n",
        "- **Tujuan**: Menguji pengaruh memperbesar jumlah variasi proyek sumber (*source projects pool*) terhadap performa generalisasi model pada proyek target baru."
    ]
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
        "\n",
        "sns.set_theme(style=\"whitegrid\", palette=\"muted\")\n",
        "plt.rcParams[\"figure.figsize\"] = (12, 6)\n",
        "\n",
        "data_path = \"../data_alt_1/engineered_test_features.csv\"\n",
        "df = pd.read_csv(data_path)\n",
        "print(f\"Dataset Shape: {df.shape[0]} baris, {df.shape[1]} kolom\")\n",
        "print(f\"Total Unique Projects: {df['project'].nunique()}\")\n",
        "\n",
        "with open(\"../data_alt_1/validation_splits_metadata_90_10.json\") as f:\n",
        "    split_info = json.load(f)\n",
        "\n",
        "print(\"=== Konfigurasi Split 90/10 ===\")\n",
        "print(f\"Train Projects ({split_info['n_train_projects']}): {split_info['train_projects']}\")\n",
        "print(f\"Test Projects  ({split_info['n_test_projects']}): {split_info['test_projects']}\")\n",
        "print(f\"Train Samples : {split_info['train_samples']} (Flaky: {split_info['train_flaky']}, {split_info['train_flaky_pct']}%)\")\n",
        "print(f\"Test Samples  : {split_info['test_samples']} (Flaky: {split_info['test_flaky']}, {split_info['test_flaky_pct']}%)\")"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": 2,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Visualisasi Komposisi Train vs Test Projects\n",
        "split_summary = pd.DataFrame([\n",
        "    {'Set': 'Train (22 Proyek)', 'Samples': split_info['train_samples'], 'Flaky': split_info['train_flaky']},\n",
        "    {'Set': 'Test (2 Proyek)', 'Samples': split_info['test_samples'], 'Flaky': split_info['test_flaky']}\n",
        "])\n",
        "\n",
        "plt.figure(figsize=(8, 5))\n",
        "sns.barplot(data=split_summary, x='Set', y='Samples', palette='Blues_d')\n",
        "plt.title('Distribusi Sampel pada Split 90% Train vs 10% Test Projects', fontweight='bold')\n",
        "for i, row in split_summary.iterrows():\n",
        "    plt.text(i, row['Samples'] + 200, f\"{row['Samples']:,} samples\\n(Flaky: {row['Flaky']})\", ha='center', fontweight='bold')\n",
        "plt.ylim(0, 22000)\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
})

notebook_json = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open('/home/ali/kuliah/rpl/its-rpl-a01-flakytest/Experiment_Ali/notebook_alt_1/03_validation_setup_90_10.ipynb', 'w') as f:
    json.dump(notebook_json, f, indent=1)

print("Notebook 03_validation_setup_90_10.ipynb created successfully!")
