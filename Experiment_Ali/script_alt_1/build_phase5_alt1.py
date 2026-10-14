import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier

df = pd.read_csv('/home/ali/kuliah/rpl/its-rpl-a01-flakytest/Experiment_Ali/data_alt_1/engineered_test_features.csv')
with open('/home/ali/kuliah/rpl/its-rpl-a01-flakytest/Experiment_Ali/data_alt_1/validation_splits_metadata_90_10.json') as f:
    split_info = json.load(f)

smell_features = [
    'assertion-roulette', 'conditional-test-logic', 'eager-test', 'fire-and-forget',
    'indirect-testing', 'mystery-guest', 'resource-optimism', 'test-run-war'
]
scaled_features = [
    'rob_log_ExecutionTime', 'rob_log_testLength', 'rob_log_numCoveredLines',
    'rob_log_projectSourceLinesCovered', 'rob_log_projectSourceClassesCovered',
    'rob_coverage_ratio', 'rob_assert_density', 'rob_class_coverage_ratio',
    'rob_num_third_party_libs'
]
pca_churn_features = ['churn_pca_1', 'churn_pca_2', 'churn_pca_3']
selected_features = smell_features + scaled_features + pca_churn_features

train_mask = df['project'].isin(split_info['train_projects'])
test_mask = df['project'].isin(split_info['test_projects'])

X_train, y_train = df.loc[train_mask, selected_features], df.loc[train_mask, 'flaky']
X_test, y_test = df.loc[test_mask, selected_features], df.loc[test_mask, 'flaky']

scale_pos = (len(y_train) - y_train.sum()) / y_train.sum()
clf = LGBMClassifier(scale_pos_weight=scale_pos, random_state=42, n_jobs=-1, verbose=-1)
clf.fit(X_train, y_train)

# Calculate feature importance
imp_df = pd.DataFrame({
    'Feature': selected_features,
    'Importance': clf.feature_importances_
}).sort_values(by='Importance', ascending=False)

out_imp = '/home/ali/kuliah/rpl/its-rpl-a01-flakytest/Experiment_Ali/data_alt_1/feature_importance_90_10.csv'
imp_df.to_csv(out_imp, index=False)
print("Feature importance saved to:", out_imp)

# Build Notebook 05_evaluation_and_interpretability_90_10.ipynb
cells = []

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 📈 Phase 5 (Alt 1): Evaluation, Result Synthesis & Interpretability (90/10 Split)\n",
        "\n",
        "**Penelitian**: *Predicting Flaky Test in Cross-Project Scenario (Alternative 1)*\n",
        "\n",
        "Notebook ini merangkum evaluasi mendalam pada skenario alternatif 90% Train Projects vs 10% Test Projects:\n",
        "1. **Komparasi Hasil**: Membandingkan performa model antara baseline 70/30 dengan alternatif 90/10.\n",
        "2. **Analisis Proyek Test Target (`spring-boot` & `elastic-job-lite`)**.\n",
        "3. **Interpretasi Fitur (Feature Importance)**."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": 1,
    "metadata": {},
    "outputs": [],
    "source": [
        "import json\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "# Load perbandingan model dan feature importance\n",
        "df_res = pd.read_csv(\"../data_alt_1/model_comparison_90_10.csv\")\n",
        "df_imp = pd.read_csv(\"../data_alt_1/feature_importance_90_10.csv\")\n",
        "\n",
        "print(\"=== Performa Model pada Split 90% Train vs 10% Test Projects ===\")\n",
        "df_res"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": 2,
    "metadata": {},
    "outputs": [],
    "source": [
        "plt.figure(figsize=(10, 6))\n",
        "sns.barplot(data=df_imp, x='Importance', y='Feature', palette='mako')\n",
        "plt.title('Global Feature Importance pada Model 90% Train Projects', fontweight='bold')\n",
        "plt.xlabel('Split-Gain Importance')\n",
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

with open('/home/ali/kuliah/rpl/its-rpl-a01-flakytest/Experiment_Ali/notebook_alt_1/05_evaluation_and_interpretability_90_10.ipynb', 'w') as f:
    json.dump(notebook_json, f, indent=1)

print("Notebook 05_evaluation_and_interpretability_90_10.ipynb created successfully!")
