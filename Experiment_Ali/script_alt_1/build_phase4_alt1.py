import json
import time
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, precision_recall_curve
)

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

def find_optimal_threshold(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores[:-1]) if len(thresholds) > 0 else 0
    optimal_threshold = thresholds[optimal_idx] if len(thresholds) > 0 else 0.5
    return optimal_threshold, f1_scores[optimal_idx]

def evaluate_predictions(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)
    if len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_probs)
        pr_auc = average_precision_score(y_true, y_probs)
    else:
        roc_auc = 0.5
        pr_auc = 0.0
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        'Precision': round(p, 4),
        'Recall': round(r, 4),
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(roc_auc, 4),
        'PR-AUC': round(pr_auc, 4)
    }

train_projects = split_info['train_projects']
test_projects = split_info['test_projects']

train_mask = df['project'].isin(train_projects)
test_mask = df['project'].isin(test_projects)

X_train = df.loc[train_mask, selected_features].values
y_train = df.loc[train_mask, 'flaky'].values
X_test = df.loc[test_mask, selected_features].values
y_test = df.loc[test_mask, 'flaky'].values

scale_pos = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-5)

models = {
    'Dummy (Most Frequent)': DummyClassifier(strategy='most_frequent'),
    'Logistic Regression (Balanced)': LogisticRegression(class_weight='balanced', max_iter=500, random_state=42),
    'Gaussian Naive Bayes': GaussianNB(),
    'Random Forest (Balanced)': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
    'XGBoost (Cost-Sensitive)': XGBClassifier(scale_pos_weight=scale_pos, eval_metric='logloss', random_state=42, n_jobs=-1),
    'LightGBM (Cost-Sensitive)': LGBMClassifier(scale_pos_weight=scale_pos, random_state=42, n_jobs=-1, verbose=-1),
    'CatBoost (Cost-Sensitive)': CatBoostClassifier(scale_pos_weight=scale_pos, iterations=200, verbose=0, random_state=42)
}

benchmark_results = []
for model_name, model in models.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    
    train_probs = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_train).astype(float)
    test_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test).astype(float)
    
    opt_thresh, _ = find_optimal_threshold(y_train, train_probs)
    eval_default = evaluate_predictions(y_test, test_probs, threshold=0.5)
    eval_opt = evaluate_predictions(y_test, test_probs, threshold=opt_thresh)
    
    benchmark_results.append({
        'Model': model_name,
        'Train Time (s)': round(fit_time, 2),
        'ROC-AUC': eval_default['ROC-AUC'],
        'PR-AUC': eval_default['PR-AUC'],
        'F1 (Thresh=0.5)': eval_default['F1-Score'],
        'Recall (Thresh=0.5)': eval_default['Recall'],
        'Opt Thresh': round(opt_thresh, 4),
        'F1 (Tuned)': eval_opt['F1-Score'],
        'Recall (Tuned)': eval_opt['Recall']
    })

df_benchmark = pd.DataFrame(benchmark_results).sort_values(by='PR-AUC', ascending=False)
out_csv = '/home/ali/kuliah/rpl/its-rpl-a01-flakytest/Experiment_Ali/data_alt_1/model_comparison_90_10.csv'
df_benchmark.to_csv(out_csv, index=False)
print(f"Results saved to {out_csv}:")
print(df_benchmark.to_string())

# Build Notebook 04_model_training_90_10.ipynb
cells = []

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 🤖 Phase 4 (Alt 1): Model Training & Handling Imbalance (90/10 Split)\n",
        "\n",
        "**Penelitian**: *Predicting Flaky Test in Cross-Project Scenario (Alternative 1)*\n",
        "\n",
        "Notebook ini mengevaluasi 7 model klasifikasi (Baseline vs Tree Ensembles) dengan skenario:\n",
        "- **Train Set**: 22 Proyek (~90% proyek) dengan Cost-Sensitive Learning (`scale_pos_weight`).\n",
        "- **Test Set**: 2 Proyek Unseen (~10% proyek).\n",
        "- **Threshold Tuning**: Mengoptimasi cut-off probabilitas berdasarkan kurva Precision-Recall."
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
        "df_res = pd.read_csv(\"../data_alt_1/model_comparison_90_10.csv\")\n",
        "df_res"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": 2,
    "metadata": {},
    "outputs": [],
    "source": [
        "plt.figure(figsize=(10, 5))\n",
        "sns.barplot(data=df_res, x='PR-AUC', y='Model', palette='viridis')\n",
        "plt.title('Performa PR-AUC Model pada Skenario 90% Train vs 10% Test Projects', fontweight='bold')\n",
        "plt.xlabel('PR-AUC')\n",
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

with open('/home/ali/kuliah/rpl/its-rpl-a01-flakytest/Experiment_Ali/notebook_alt_1/04_model_training_90_10.ipynb', 'w') as f:
    json.dump(notebook_json, f, indent=1)

print("Notebook 04_model_training_90_10.ipynb created successfully!")
