# 📊 Data Repository: Combined Experiment 1 & 2
## Tri-Paradigm Data Cleaning Benchmark & Feature Dictionary

This directory curates the empirical datasets synthesizing **Experiment 1** (raw FlakeFlagger data with sample omission strategies) and **Experiment 2** (contextual group-aware imputation) to enable rigorous, controlled evaluation of *3 Data Cleaning / Preprocessing Paradigms* under standardized cross-project validation.

---

## 🗂️ Dataset Inventory

| Filename | Source / Provenance | Rows | Projects | Scientific Description |
|---|---|:---:|:---:|---|
| `test_features.csv` | Experiment 1 / FlakeFlagger | 22,236 | 24 | Raw FlakeFlagger feature matrix exhibiting 273 missing values across 5 Java repositories. Used to evaluate strategies **C1** and **C2**. |
| `cleaned_test_features.csv` | Experiment 2 | 22,236 | 24 | Curated dataset processed via *Group-Aware Per-Project Median & Probabilistic Imputation* (0 missing values, 811 intact flaky tests). Used for strategy **C3**. |
| `test_results.csv` | FlakeFlagger Benchmark | 22,236 | 24 | Historical execution telemetry across 10,000 reruns per test case for ground truth flakiness verification. |
| `Project_Info.csv` | FlakeFlagger Benchmark | 24 | 24 | Repository metadata across 24 Java systems (repository slug, application domain, total test count, empirical flakiness rate). |

---

## ⚙️ Tri-Paradigm Data Cleaning Specifications (C1, C2, C3)

1. **C1: Row-Level Discarding (Experiment 1)**:
   - Discards all 273 rows containing missing values.
   - Retains **21,963 test cases** across all **24 Java projects**.
   - Flaky test population: **808 flaky tests**.
2. **C2: Project-Level Discarding (Experiment 1)**:
   - Discards the 5 projects containing missing values (`wildfly`, `logback`, `spring-boot`, `wro4j`, `handlebars.java`).
   - Retains **16,458 test cases** across **19 Java projects** ($-20.8\%$ projects, $-24.8\%$ test cases).
   - Flaky test population: **586 flaky tests** (222 flaky tests discarded, including 163 from `spring-boot`).
3. **C3: Group-Aware Median Imputation (Experiment 2)**:
   - Imputes missing feature values using the median value computed within each respective project group.
   - Preserves **100% of the benchmark data**: **22,236 test cases** across all **24 Java projects**.
   - Flaky test population: **811 flaky tests intact** with zero sample loss.

---

## 🔍 Canonical FlakeFlagger 23 Predictive Features Taxonomy

- **8 Test Smells (Binary)**: `assertion-roulette`, `conditional-test-logic`, `eager-test`, `fire-and-forget`, `indirect-testing`, `mystery-guest`, `resource-optimism`, `test-run-war`.
- **7 Execution & Code Coverage Metrics (Continuous)**: `testLength`, `numAsserts`, `numCoveredLines`, `ExecutionTime`, `projectSourceLinesCovered`, `projectSourceClassesCovered`, `num_third_party_libs`.
- **8 Code Churn H-Index Metrics (Continuous)**: Sliding modification windows per covered line (`window5`, `window10`, `window25`, `window50`, `window75`, `window100`, `window500`, `window10000`).
