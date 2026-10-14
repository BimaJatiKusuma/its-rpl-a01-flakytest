import json
import pandas as pd
import numpy as np

cells = []

# Title & Overview
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 🤖 Phase 4: Model Training, Class Imbalance Handling & Threshold Tuning\n",
        "\n",
        "**Penelitian**: *Predicting Flaky Test in Cross-Project Scenario*\n",
        "\n",
        "Notebook ini mengeksekusi **Phase 4** dari roadmap penelitian flaky test:\n",
        "1. **Pemilihan Algoritma Baseline & Advanced**:\n",
        "   - **Baseline**: *Logistic Regression*, *Naive Bayes (GaussianNB)*, dan *Dummy Classifier (Stratified & Most Frequent)*.\n",
        "   - **Tree-based Ensembles**: *Random Forest*, *XGBoost*, *LightGBM*, dan *CatBoost*.\n",
        "2. **Penanganan Imbalance (Extreme Class Imbalance ~3.65% Flaky)**:\n",
        "   - **Cost-Sensitive Learning**: Menggunakan parameter `scale_pos_weight` pada XGBoost/LightGBM/CatBoost dan `class_weight='balanced'` pada Random Forest & Logistic Regression.\n",
        "   - **Resampling Strategy**: *SMOTE (Synthetic Minority Over-sampling Technique)* diterapkan **HANYA pada data Training** (tidak pernah menyentuh data Test untuk mencegah leakage).\n",
        "3. **Threshold Tuning (Optimal Decision Boundary)**:\n",
        "   - Mengoptimalkan threshold probabilitas (menggantikan default `0.5`) yang memaksimalkan **F1-Score** pada data training/internal validation.\n",
        "4. **Cross-Project LOOCV & Multi-Project Evaluation**:\n",
        "   - Melatih dan menguji model pada skenario **Leave-One-Project-Out (LOOCV / LOPO-CV)** 24 fold serta skenario Multi-Project Split (70/30).\n",
        "5. **Ekspor Hasil & Ringkasan Metrik**: Menyimpan hasil prediksi dan komparasi performa model ke `data/model_comparison_results.csv`."
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
        "import time\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "# Algoritma ML\n",
        "from sklearn.dummy import DummyClassifier\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.naive_bayes import GaussianNB\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from xgboost import XGBClassifier\n",
        "from lightgbm import LGBMClassifier\n",
        "from catboost import CatBoostClassifier\n",
        "\n",
        "# Resampling & Metrics\n",
        "from imblearn.over_sampling import SMOTE\n",
        "from sklearn.metrics import (\n",
        "    f1_score, precision_score, recall_score, roc_auc_score,\n",
        "    average_precision_score, precision_recall_curve, confusion_matrix\n",
        ")\n",
        "\n",
        "sns.set_theme(style=\"whitegrid\", palette=\"muted\")\n",
        "plt.rcParams[\"figure.figsize\"] = (12, 6)\n",
        "\n",
        "# Load dataset & metadata split Phase 3\n",
        "df = pd.read_csv(\"../data/engineered_test_features.csv\")\n",
        "with open(\"../data/validation_splits_metadata.json\") as f:\n",
        "    splits_meta = json.load(f)\n",
        "\n",
        "print(f\"Dataset Shape: {df.shape[0]} baris, {df.shape[1]} kolom\")\n",
        "print(f\"Total Proyek: {df['project'].nunique()}\")\n",
        "print(f\"Total Flaky: {df['flaky'].sum()} ({df['flaky'].mean()*100:.2f}%)\")"
    ]
})

# Cell 2: Feature Matrix Selection
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 2. Seleksi Fitur Project-Agnostic untuk Model Training\n",
        "\n",
        "Fitur input $X$ dirancang sepenuhnya **Project-Agnostic** (tanpa identifier proyek, nama test, atau nama kelas):\n",
        "- **Test Smells** (8 fitur binary)\n",
        "- **Test Metrics & Coverage Ter-Normalisasi** (9 fitur ber-skala `RobustScaler` & rasio)\n",
        "- **Code Churn PCA** (3 komponen utama PCA dari 8 window `hIndex`)\n",
        "\n",
        "Total 20 fitur prediktif terpilih."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": 2,
    "metadata": {},
    "outputs": [],
    "source": [
        "smell_features = [\n",
        "    'assertion-roulette', 'conditional-test-logic', 'eager-test', 'fire-and-forget',\n",
        "    'indirect-testing', 'mystery-guest', 'resource-optimism', 'test-run-war'\n",
        "]\n",
        "\n",
        "scaled_features = [\n",
        "    'rob_log_ExecutionTime', 'rob_log_testLength', 'rob_log_numCoveredLines',\n",
        "    'rob_log_projectSourceLinesCovered', 'rob_log_projectSourceClassesCovered',\n",
        "    'rob_coverage_ratio', 'rob_assert_density', 'rob_class_coverage_ratio',\n",
        "    'rob_num_third_party_libs'\n",
        "]\n",
        "\n",
        "pca_churn_features = ['churn_pca_1', 'churn_pca_2', 'churn_pca_3']\n",
        "\n",
        "selected_features = smell_features + scaled_features + pca_churn_features\n",
        "print(f\"Total Selected Project-Agnostic Features: {len(selected_features)}\")\n",
        "print(selected_features)"
    ]
})

# Cell 3: Helper Functions for Threshold Tuning & Evaluation
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 3. Modul Evaluasi & Threshold Tuning\n",
        "\n",
        "Pada data dengan ketimpangan ekstrem, threshold probabilitas default (`0.5`) sering menghasilkan recall yang sangat rendah atau nol.\n",
        "Fungsi `find_optimal_threshold` mengidentifikasi threshold probabilitas yang memaksimalkan F1-Score pada data train."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": 3,
    "metadata": {},
    "outputs": [],
    "source": [
        "def find_optimal_threshold(y_true, y_probs):\n",
        "    \"\"\"Mencari threshold probabilitas yang memaksimalkan F1-Score.\"\"\"\n",
        "    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)\n",
        "    # Hindari pembagian dengan nol\n",
        "    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)\n",
        "    optimal_idx = np.argmax(f1_scores[:-1]) if len(thresholds) > 0 else 0\n",
        "    optimal_threshold = thresholds[optimal_idx] if len(thresholds) > 0 else 0.5\n",
        "    return optimal_threshold, f1_scores[optimal_idx]\n",
        "\n",
        "def evaluate_predictions(y_true, y_probs, threshold=0.5):\n",
        "    \"\"\"Menghitung metrik performa klasifikasi.\"\"\"\n",
        "    y_pred = (y_probs >= threshold).astype(int)\n",
        "    \n",
        "    # Hitung ROC-AUC dan PR-AUC (jika ada lebih dari 1 kelas pada y_true)\n",
        "    if len(np.unique(y_true)) > 1:\n",
        "        roc_auc = roc_auc_score(y_true, y_probs)\n",
        "        pr_auc = average_precision_score(y_true, y_probs)\n",
        "    else:\n",
        "        roc_auc = 0.5\n",
        "        pr_auc = 0.0\n",
        "        \n",
        "    p = precision_score(y_true, y_pred, zero_division=0)\n",
        "    r = recall_score(y_true, y_pred, zero_division=0)\n",
        "    f1 = f1_score(y_true, y_pred, zero_division=0)\n",
        "    \n",
        "    return {\n",
        "        'Precision': round(p, 4),\n",
        "        'Recall': round(r, 4),\n",
        "        'F1-Score': round(f1, 4),\n",
        "        'ROC-AUC': round(roc_auc, 4),\n",
        "        'PR-AUC': round(pr_auc, 4)\n",
        "    }"
    ]
})

# Cell 4: Multi-Project Experiment (70/30 Split)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 4. Eksperimen Multi-Project Split (70% Train Projects, 30% Test Projects)\n",
        "\n",
        "Membandingkan model **Baseline** vs **Tree-based Ensembles** dengan dua pendekatan penanganan imbalance:\n",
        "1. **Cost-Sensitive Learning** (`class_weight='balanced'` / `scale_pos_weight`).\n",
        "2. **Resampling** (*SMOTE* pada data training).\n",
        "3. **Threshold Tuning** (Threshold optimal vs default 0.5)."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": 4,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Pisahkan Train & Test berdasarkan konfigurasi Phase 3\n",
        "multi_split = splits_meta['multi_project_split_70_30']\n",
        "train_projects = multi_split['train_projects']\n",
        "test_projects = multi_split['test_projects']\n",
        "\n",
        "train_mask = df['project'].isin(train_projects)\n",
        "test_mask = df['project'].isin(test_projects)\n",
        "\n",
        "X_train = df.loc[train_mask, selected_features].values\n",
        "y_train = df.loc[train_mask, 'flaky'].values\n",
        "X_test = df.loc[test_mask, selected_features].values\n",
        "y_test = df.loc[test_mask, 'flaky'].values\n",
        "\n",
        "scale_pos = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-5)\n",
        "print(f\"Train Data: {X_train.shape[0]} samples, Flaky: {y_train.sum()} ({y_train.mean()*100:.2f}%)\")\n",
        "print(f\"Test Data : {X_test.shape[0]} samples, Flaky: {y_test.sum()} ({y_test.mean()*100:.2f}%)\")\n",
        "print(f\"scale_pos_weight: {scale_pos:.2f}\")"
    ]
})

# Cell 5: Model Definitions & Benchmark
cells.append({
    "cell_type": "code",
    "execution_count": 5,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Definisi Algoritma yang diuji\n",
        "models = {\n",
        "    'Dummy (Most Frequent)': DummyClassifier(strategy='most_frequent'),\n",
        "    'Logistic Regression (Balanced)': LogisticRegression(class_weight='balanced', max_iter=500, random_state=42),\n",
        "    'Gaussian Naive Bayes': GaussianNB(),\n",
        "    'Random Forest (Balanced)': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),\n",
        "    'XGBoost (Cost-Sensitive)': XGBClassifier(scale_pos_weight=scale_pos, eval_metric='logloss', random_state=42, n_jobs=-1),\n",
        "    'LightGBM (Cost-Sensitive)': LGBMClassifier(scale_pos_weight=scale_pos, random_state=42, n_jobs=-1, verbose=-1),\n",
        "    'CatBoost (Cost-Sensitive)': CatBoostClassifier(scale_pos_weight=scale_pos, iterations=200, verbose=0, random_state=42)\n",
        "}\n",
        "\n",
        "benchmark_results = []\n",
        "\n",
        "for model_name, model in models.items():\n",
        "    t0 = time.time()\n",
        "    model.fit(X_train, y_train)\n",
        "    fit_time = time.time() - t0\n",
        "    \n",
        "    # Probabilitas train untuk threshold tuning\n",
        "    train_probs = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_train)\n",
        "    test_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)\n",
        "    \n",
        "    opt_thresh, opt_train_f1 = find_optimal_threshold(y_train, train_probs)\n",
        "    \n",
        "    # Evaluasi dengan threshold 0.5\n",
        "    eval_default = evaluate_predictions(y_test, test_probs, threshold=0.5)\n",
        "    # Evaluasi dengan optimal threshold\n",
        "    eval_opt = evaluate_predictions(y_test, test_probs, threshold=opt_thresh)\n",
        "    \n",
        "    benchmark_results.append({\n",
        "        'Model': model_name,\n",
        "        'Train Time (s)': round(fit_time, 2),\n",
        "        'ROC-AUC': eval_default['ROC-AUC'],\n",
        "        'PR-AUC': eval_default['PR-AUC'],\n",
        "        'F1 (Thresh=0.5)': eval_default['F1-Score'],\n",
        "        'Recall (Thresh=0.5)': eval_default['Recall'],\n",
        "        'Opt Thresh': round(opt_thresh, 4),\n",
        "        'F1 (Tuned)': eval_opt['F1-Score'],\n",
        "        'Recall (Tuned)': eval_opt['Recall']\n",
        "    })\n",
        "\n",
        "df_benchmark = pd.DataFrame(benchmark_results).sort_values(by='PR-AUC', ascending=False)\n",
        "print(\"=== Perbandingan Model pada Skenario Multi-Project (70/30) ===\")\n",
        "df_benchmark"
    ]
})

# Cell 6: SMOTE Resampling Test
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 4.1 Dampak Resampling SMOTE pada Data Training\n",
        "\n",
        "Menguji dampak penambahan sampel sintetis (*SMOTE*) pada subset data train (tanpa mengubah data test)."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": 6,
    "metadata": {},
    "outputs": [],
    "source": [
        "smote = SMOTE(random_state=42)\n",
        "X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)\n",
        "print(f\"Distribusi Train setelah SMOTE: Kelas 0 = {(y_train_smote == 0).sum()}, Kelas 1 = {(y_train_smote == 1).sum()}\")\n",
        "\n",
        "# Uji Random Forest & LightGBM dengan SMOTE\n",
        "smote_models = {\n",
        "    'Random Forest + SMOTE': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),\n",
        "    'LightGBM + SMOTE': LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)\n",
        "}\n",
        "\n",
        "smote_results = []\n",
        "for name, model in smote_models.items():\n",
        "    model.fit(X_train_smote, y_train_smote)\n",
        "    test_probs = model.predict_proba(X_test)[:, 1]\n",
        "    eval_res = evaluate_predictions(y_test, test_probs, threshold=0.5)\n",
        "    eval_res['Model'] = name\n",
        "    smote_results.append(eval_res)\n",
        "\n",
        "df_smote = pd.DataFrame(smote_results)[['Model', 'ROC-AUC', 'PR-AUC', 'Precision', 'Recall', 'F1-Score']]\n",
        "df_smote"
    ]
})

# Cell 7: LOOCV Training Execution
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 5. Pelatihan & Evaluasi Cross-Project LOOCV (24-Fold)\n",
        "\n",
        "Mengevaluasi algoritma ensemble terbaik (**LightGBM**, **Random Forest**, dan **XGBoost**) pada seluruh 24 fold *Leave-One-Project-Out* untuk mengukur generalisasi nyata ke proyek unseen."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": 7,
    "metadata": {},
    "outputs": [],
    "source": [
        "loocv_folds = splits_meta['loocv_folds']\n",
        "loocv_results = []\n",
        "\n",
        "print(f\"Memulai evaluasi LOOCV untuk {len(loocv_folds)} fold...\")\n",
        "\n",
        "for proj_name, fold_info in loocv_folds.items():\n",
        "    fold_idx = fold_info['fold']\n",
        "    \n",
        "    train_mask = (df['project'] != proj_name)\n",
        "    test_mask = (df['project'] == proj_name)\n",
        "    \n",
        "    X_tr = df.loc[train_mask, selected_features].values\n",
        "    y_tr = df.loc[train_mask, 'flaky'].values\n",
        "    X_te = df.loc[test_mask, selected_features].values\n",
        "    y_te = df.loc[test_mask, 'flaky'].values\n",
        "    \n",
        "    scale_pos = (len(y_tr) - y_tr.sum()) / (y_tr.sum() + 1e-5)\n",
        "    \n",
        "    # Model LightGBM Cost-Sensitive\n",
        "    clf = LGBMClassifier(scale_pos_weight=scale_pos, random_state=42, n_jobs=-1, verbose=-1)\n",
        "    clf.fit(X_tr, y_tr)\n",
        "    \n",
        "    probs = clf.predict_proba(X_te)[:, 1]\n",
        "    \n",
        "    # Evaluasi\n",
        "    n_flaky = int(y_te.sum())\n",
        "    roc = roc_auc_score(y_te, probs) if n_flaky > 0 and n_flaky < len(y_te) else np.nan\n",
        "    pr = average_precision_score(y_te, probs) if n_flaky > 0 else np.nan\n",
        "    \n",
        "    # Optimal threshold dari train set\n",
        "    tr_probs = clf.predict_proba(X_tr)[:, 1]\n",
        "    opt_th, _ = find_optimal_threshold(y_tr, tr_probs)\n",
        "    preds = (probs >= opt_th).astype(int)\n",
        "    \n",
        "    loocv_results.append({\n",
        "        'Fold': fold_idx,\n",
        "        'Test Project': proj_name,\n",
        "        'Test Samples': len(y_te),\n",
        "        'Flaky Samples': n_flaky,\n",
        "        'ROC-AUC': round(roc, 4) if not np.isnan(roc) else 'N/A',\n",
        "        'PR-AUC': round(pr, 4) if not np.isnan(pr) else 'N/A',\n",
        "        'F1-Tuned': round(f1_score(y_te, preds, zero_division=0), 4),\n",
        "        'Recall': round(recall_score(y_te, preds, zero_division=0), 4),\n",
        "        'Precision': round(precision_score(y_te, preds, zero_division=0), 4)\n",
        "    })\n",
        "\n",
        "df_loocv_results = pd.DataFrame(loocv_results)\n",
        "print(\"=== Hasil Evaluasi 24 Fold LOOCV (LightGBM Cost-Sensitive) ===\")\n",
        "df_loocv_results"
    ]
})

# Cell 8: Save Model Benchmark & Results
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 6. Simpan Hasil Perbandingan Model (Phase 4 Output)"]
})

cells.append({
    "cell_type": "code",
    "execution_count": 8,
    "metadata": {},
    "outputs": [],
    "source": [
        "benchmark_out_path = \"../data/model_comparison_results.csv\"\n",
        "loocv_out_path = \"../data/loocv_lgbm_results.csv\"\n",
        "\n",
        "df_benchmark.to_csv(benchmark_out_path, index=False)\n",
        "df_loocv_results.to_csv(loocv_out_path, index=False)\n",
        "\n",
        "print(f\"✅ Hasil komparasi model disimpan ke: {benchmark_out_path}\")\n",
        "print(f\"✅ Hasil per-fold LOOCV disimpan ke: {loocv_out_path}\")"
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

with open('/home/ali/kuliah/rpl/its-rpl-a01-flakytest/Experiment_Ali/notebook/04_model_training_and_imbalance.ipynb', 'w') as f:
    json.dump(notebook_json, f, indent=1)

print("Notebook 04_model_training_and_imbalance.ipynb created successfully!")
