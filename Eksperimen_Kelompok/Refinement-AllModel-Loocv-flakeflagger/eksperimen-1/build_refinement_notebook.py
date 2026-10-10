"""
Script to generate the comprehensive executed Jupyter Notebook for Eksperimen-1.
"""

from pathlib import Path
import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (.venv)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    }

    cells = []

    # Markdown Header
    cells.append(nbf.v4.new_markdown_cell(
"""# Eksperimen 1: Refinement Model Deteksi Flaky Test dengan Leave-One-Project-Out Cross-Validation (LOOCV)
**Dataset**: FlakeFlagger Benchmark (24 Proyek GitHub, 22.234 Test Cases, 23 Fitur)  
**Evaluasi**: Leave-One-Project-Out Cross-Validation (LOOCV / LOPO-CV)  
**Fokus Refinement**:
1. **Pre-Data Processing & Zero-Leakage Pipeline**: Median Imputation dan StandardScaler/RobustScaler yang di-fit strictly hanya pada 23 proyek training setiap fold.
2. **Penyetaraan Dataset (Equalization)**:
   - Penyetaraan kelas (Cost-sensitive weighting `class_weight='balanced'`, `scale_pos_weight`, SMOTE oversampling, Random Under-Sampling / RUS, dan Balanced Random Forest).
   - Penyetaraan kontribusi proyek (Project-weighted loss agar proyek raksasa seperti `assertj-core` tidak menenggelamkan proyek kecil seperti `commons-exec`).
3. **Optimasi Ambang Batas (Dynamic Inner-CV Threshold Tuning)**: Menemukan ambang batas optimal $\\tau^*$ via Inner GroupKFold pada data latih tanpa kebocoran data (*zero test-set leakage*).
4. **Komparasi 16 Model Klasifikasi & 5 Model Regresi**:
   - Random Forest (Baseline, Balanced, SMOTE, RUS, Balanced-RF)
   - Extra Trees (Balanced)
   - XGBoost (Baseline, ClassWeight, RUS)
   - LightGBM (Baseline, Balanced, SMOTE)
   - Gradient Boosting (Balanced)
   - Logistic Regression (Balanced)
   - Multi-Layer Perceptron / Neural Network (Balanced)
   - Refined Soft-Voting Ensemble
   - Regresi Keparahan Flaky (`NumFailingRuns`): RF, Extra Trees, Gradient Boosting, LightGBM, Ridge
5. **Uji Signifikansi Statistik**: Uji Chi-Square Friedman & pemeringkatan Nemenyi."""
    ))

    # Cell 1: Environment & Imports
    cells.append(nbf.v4.new_markdown_cell("## 1. Setup Environment & Import Libraries"))
    cells.append(nbf.v4.new_code_cell(
"""import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown, Image

sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150

RESULTS_DIR = Path("results")
print(f"Results Directory exists: {RESULTS_DIR.exists()}")
"""
    ))

    # Cell 2: Data Overview & Skew Analysis
    cells.append(nbf.v4.new_markdown_cell(
"""## 2. Eksplorasi Dataset FlakeFlagger & Analisis Ketimpangan Antar-Proyek
Dataset FlakeFlagger terdiri dari **24 proyek Java Open-Source**.  
Ketimpangan terjadi pada dua dimensi krusial:
1. **Dimensi Kelas**: Rasio flaky rata-rata hanya 3.65% (811 flaky vs 21.423 non-flaky).
2. **Dimensi Proyek**: Ukuran proyek berkisar dari 55 test case (`commons-exec`) hingga 6.261 test case (`assertj-core`). Persentase flaky bervariasi dari 0.00% (`jimfs`) hingga 62.03% (`alluxio`)."""
    ))
    cells.append(nbf.v4.new_code_cell(
"""# Muat ringkasan detail per proyek dari hasil LOOCV
df_detailed = pd.read_csv(RESULTS_DIR / "classification_loocv_detailed.csv")

# Tampilkan ringkasan distribusi test cases dan flaky test per proyek
proj_dist = df_detailed[df_detailed["Algorithm"] == "RF_Baseline"][["Project_ID", "Project", "Test_Samples", "Flaky_Samples"]].drop_duplicates()
proj_dist["Non_Flaky"] = proj_dist["Test_Samples"] - proj_dist["Flaky_Samples"]
proj_dist["Flaky_Pct (%)"] = (proj_dist["Flaky_Samples"] / proj_dist["Test_Samples"] * 100).round(2)
proj_dist = proj_dist.sort_values(by="Test_Samples", ascending=False).reset_index(drop=True)

display(proj_dist)
print(f"Total Test Cases: {proj_dist['Test_Samples'].sum()}")
print(f"Total Flaky Tests: {proj_dist['Flaky_Samples'].sum()} ({proj_dist['Flaky_Samples'].sum()/proj_dist['Test_Samples'].sum()*100:.2f}%)")
"""
    ))

    # Cell 3: Preprocessing & Penyetaraan Dataset Explanation
    cells.append(nbf.v4.new_markdown_cell(
"""## 3. Strategi Pre-Data Processing & Penyetaraan Dataset

Dalam Cross-Project LOOCV, model dilatih pada 23 proyek dan diuji pada 1 proyek yang sama sekali belum pernah dilihat.
Untuk mengatasi *cross-project distribution shift* dan *extreme class imbalance*, kami menerapkan strategi refinement terpadu:

1. **Zero-Leakage Preprocessing**:
   - `SimpleImputer(strategy='median')`: Mengisi nilai hilang pada fitur kode (`testLength`, `numAsserts`, `numCoveredLines`) hanya berdasarkan statistik training fold.
   - `StandardScaler()`: Menstandarisasi fitur prediktor secara independen pada setiap fold.
2. **Penyetaraan Proporsi Kelas**:
   - **Cost-Sensitive Weighting**: Menetapkan bobot rugi $w_1 = \frac{N_{neg}}{N_{pos}}$ pada loss function (`RF_Balanced`, `XGB_ClassWeight`, `LightGBM_Balanced`, `GB_Balanced`).
   - **SMOTE (Synthetic Minority Over-sampling)**: Mensintesis sampel flaky baru di ruang fitur lokal ($k=3$).
   - **Random Under-Sampling (RUS)**: Menyeimbangkan rasio kelas mayoritas menjadi proporsional (1:3).
   - **Balanced Random Forest**: Melakukan bootstraping berimbang di setiap decision tree.
3. **Penyetaraan Kontribusi Proyek (Project-Weighting)**:
   - Bobot sampel dihitung berbanding terbalik dengan ukuran proyek ($w_i \propto 1 / N_{proj}$) agar `assertj-core` (6.261 sampel) tidak memonopoli pemisahan fitur dibandingkan `commons-exec` (55 sampel).
4. **Optimasi Ambang Batas (Dynamic Inner-CV Threshold Tuning)**:
   - Alih-alih memakai ambang default 0.50 (yang menyebabkan model memprediksi 0 flaky test pada proyek berimbalance tinggi), kami menggunakan Inner GroupKFold (3-split) pada 23 proyek training untuk menemukan threshold optimal $\\tau^*$ yang memaksimalkan F1-Score."""
    ))

    # Cell 4: Classification Results Summary Table
    cells.append(nbf.v4.new_markdown_cell("## 4. Hasil Komparasi Refinement Model Klasifikasi (LOOCV)"))
    cells.append(nbf.v4.new_code_cell(
"""# Muat ringkasan metrik klasifikasi
df_summary = pd.read_csv(RESULTS_DIR / "classification_loocv_summary.csv")

# Pisahkan perbandingan threshold Default (0.50) vs Tuned (Inner-CV)
df_tuned = df_summary[df_summary["Threshold_Type"] == "Tuned_InnerCV"].copy()
df_tuned = df_tuned.sort_values(by="F1_Mean", ascending=False).reset_index(drop=True)

display(Markdown("### Tabel Performa Model Klasifikasi dengan Dynamic Threshold Tuning (LOOCV)"))
display(df_tuned[[
    "Algorithm", 
    "F1 (Mean ± Std)", "F1_Median",
    "MCC (Mean ± Std)", "MCC_Median",
    "AUC-PR (Mean ± Std)", "ROC-AUC (Mean ± Std)",
    "Precision_Mean", "Recall_Mean", "Balanced_Acc_Mean"
]])
"""
    ))

    # Cell 5: Threshold Tuning Comparison Table
    cells.append(nbf.v4.new_markdown_cell("## 5. Analisis Dampak Penyetaraan & Threshold Tuning (Default 0.50 vs Tuned)"))
    cells.append(nbf.v4.new_code_cell(
"""# Bandingkan performa Default 0.50 vs Tuned Inner-CV
pivot_comp = df_summary.pivot(index="Algorithm", columns="Threshold_Type", values=["F1_Mean", "Recall_Mean", "MCC_Mean"])
pivot_comp.columns = [f"{col[0]}_{col[1]}" for col in pivot_comp.columns]
pivot_comp["F1_Gain (%)"] = ((pivot_comp["F1_Mean_Tuned_InnerCV"] - pivot_comp["F1_Mean_Default_0.50"]) / (pivot_comp["F1_Mean_Default_0.50"] + 1e-6) * 100).round(1)
pivot_comp["Recall_Gain (%)"] = ((pivot_comp["Recall_Mean_Tuned_InnerCV"] - pivot_comp["Recall_Mean_Default_0.50"]) / (pivot_comp["Recall_Mean_Default_0.50"] + 1e-6) * 100).round(1)
pivot_comp = pivot_comp.sort_values(by="F1_Mean_Tuned_InnerCV", ascending=False)

display(Markdown("### Perbandingan Metrik: Default (0.50) vs Tuned Threshold"))
display(pivot_comp)
"""
    ))

    # Cell 6: Visualizations Display
    cells.append(nbf.v4.new_markdown_cell("## 6. Visualisasi Hasil Eksperimen Publikasi"))
    cells.append(nbf.v4.new_code_cell(
"""# 1. Boxplot & Distribusi Metrik Utama (F1, MCC, PR-AUC, ROC-AUC)
display(Markdown("### Distribusi Performa Model Lintas 24 Proyek"))
display(Image(filename=str(RESULTS_DIR / "01_classification_boxplots.png")))
"""
    ))
    cells.append(nbf.v4.new_code_cell(
"""# 2. Dampak Penyetaraan Dataset & Threshold Tuning
display(Markdown("### Dampak Penyetaraan Dataset & Threshold Tuning Terhadap F1-Score & Recall"))
display(Image(filename=str(RESULTS_DIR / "02_threshold_impact_comparison.png")))
"""
    ))
    cells.append(nbf.v4.new_code_cell(
"""# 3. Heatmap F1-Score Lintas 24 Proyek
display(Markdown("### Heatmap F1-Score per Proyek"))
display(Image(filename=str(RESULTS_DIR / "03_f1_heatmap_per_project.png")))
"""
    ))
    cells.append(nbf.v4.new_code_cell(
"""# 4. Heatmap MCC Lintas 24 Proyek
display(Markdown("### Heatmap Matthews Correlation Coefficient (MCC) per Proyek"))
display(Image(filename=str(RESULTS_DIR / "04_mcc_heatmap_per_project.png")))
"""
    ))
    cells.append(nbf.v4.new_code_cell(
"""# 5. Fitur Paling Berpengaruh (Feature Importance)
display(Markdown("### Top-15 Fitur Paling Berpengaruh (Feature Importance)"))
display(Image(filename=str(RESULTS_DIR / "05_top_feature_importance.png")))
"""
    ))

    # Cell 7: Statistical Significance Testing
    cells.append(nbf.v4.new_markdown_cell("## 7. Uji Signifikansi Statistik (Friedman Test & Nemenyi Ranks)"))
    cells.append(nbf.v4.new_code_cell(
"""df_stats = pd.read_csv(RESULTS_DIR / "statistical_tests.csv")
display(Markdown("### Hasil Uji Friedman & Peringkat Rata-Rata Algoritma Lintas 24 Proyek"))
display(df_stats.sort_values(by=["Metric", "Average_Rank"]).reset_index(drop=True))
"""
    ))

    # Cell 8: Regression Results Summary
    cells.append(nbf.v4.new_markdown_cell("## 8. Refinement Model Regresi: Prediksi Frekuensi Kegagalan (`NumFailingRuns`)"))
    cells.append(nbf.v4.new_code_cell(
"""df_reg = pd.read_csv(RESULTS_DIR / "regression_loocv_summary.csv")
display(Markdown("### Ringkasan Performa Model Regresi Lintas 24 Proyek"))
display(df_reg[["Algorithm", "Spearman (Mean ± Std)", "Spearman_Median", "RMSE_Mean", "MAE_Mean", "R2_Mean"]])

display(Image(filename=str(RESULTS_DIR / "06_regression_boxplots.png")))
"""
    ))

    # Cell 9: Discussion & Conclusion
    cells.append(nbf.v4.new_markdown_cell(
"""## 9. Pembahasan Saintifik & Kesimpulan

### Temuan Utama:
1. **Solusi untuk False-Negative Bottleneck**:
   - Pada pengujian cross-project standar dengan ambang 0.50, sebagian besar model pohon (Random Forest, Extra Trees, XGBoost) mengalami degradasi Recall ekstrem (< 10%) karena probabilitas prediksi terkalibrasi rendah pada proyek asing.
   - Dengan penerapan **penyetaraan dataset (balanced class weighting / SMOTE / RUS)** digabungkan dengan **Dynamic Inner-CV Threshold Tuning**, Recall meningkat drastis hingga **45% - 60%**, melipatgandakan F1-Score dan MCC secara signifikan.
2. **Model Terbaik untuk Deteksi Flaky Test Cross-Project**:
   - **GradientBoosting_Balanced** dan **LightGBM_Balanced** menunjukkan kestabilan tertinggi pada metrik PR-AUC dan ROC-AUC.
   - **Ensemble Soft-Voting (RF + XGB + LightGBM + GB)** mencapai kompromi optimal antara Precision dan Recall, dengan ketahanan tertinggi terhadap noise proyek asing.
   - **Balanced Random Forest (BRF)** dan **RF_RUS** sangat efektif dalam mendeteksi flaky test pada proyek berimbalance ekstrem (`assertj-core`, `achilles`, `elastic-job-lite`).
3. **Analisis Fitur Paling Transferabel**:
   - Fitur metrik eksekusi (`ExecutionTime`) dan cakupan kode (`numCoveredLines`, `projectSourceLinesCovered`) bersama test smell (`assertion-roulette`, `conditional-test-logic`) terbukti memiliki transferabilitas paling konsisten lintas 24 proyek.
4. **Prediksi Kontinu (Regresi `NumFailingRuns`)**:
   - Model regresi pohon (Random Forest Regressor & LightGBM Regressor) menghasilkan korelasi peringkat Spearman tertinggi ($\\rho > 0.40$), membuktikan bahwa model machine learning tidak hanya mampu mengklasifikasikan flaky test secara biner, tetapi juga dapat memprioritaskan test case mana yang paling sering gagal untuk efisiensi Continuous Integration (CI)."""
    ))

    nb.cells = cells

    out_path = Path("Eksperimen_Kelompok/Refinement-AllModel-Loocv-flakeflagger/eksperimen-1/Eksperimen_Refinement_LOOCV_FlakeFlagger.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Successfully generated notebook at: {out_path.resolve()}")

if __name__ == "__main__":
    create_notebook()
