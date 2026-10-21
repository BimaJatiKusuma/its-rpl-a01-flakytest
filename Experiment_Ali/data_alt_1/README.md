# 📁 Data Alt 1 — FlakeFlagger (Skenario 90% Train, 10% Test Projects)

Direktori ini berisi dataset dan metadata untuk eksperimen **Alternatif 1**:
- **Pemisahan**: 90% Proyek Pelatihan (22 proyek) vs 10% Proyek Pengujian (2 proyek: `elastic-job-lite` dan `spring-boot`).

---

## 📑 File dalam Direktori Ini

1. **`engineered_test_features.csv`**:
   - Dataset dengan 58 kolom fitur ter-engineered (rasio, log, robust scaling, PCA).
2. **`validation_splits_metadata_90_10.json`**:
   - Metadata konfigurasi pemisahan indeks train dan test set untuk skenario 90/10.
   - Train samples: 19.551 (645 flaky, 3,30%).
   - Test samples: 2.685 (166 flaky, 6,18%).
3. **`model_comparison_90_10.csv`**:
   - Hasil evaluasi komparatif 7 model (Dummy, Logistic Regression, GaussianNB, Random Forest, XGBoost, LightGBM, CatBoost).
4. **`feature_importance_90_10.csv`**:
   - Ranking nilai kontribusi fitur global pada model LightGBM yang dilatih dengan 22 proyek sumber.
