# 📊 Direktori Data Eksperimen Gabungan 1 & 2

Direktori ini memadukan dataset dari **Eksperimen 1** (Raw Data dengan strategi pembuangan) dan **Eksperimen 2** (Cleaned Data dengan imputasi cerdas) untuk memungkinkan perbandingan komprehensif *3 Data Cleaning / Preprocessing Paradigms* dalam satu kerangka kerja validasi lintas proyek yang konsisten.

---

## 🗂️ Berkas Data

| Nama Berkas | Sumber | Baris | Proyek | Deskripsi |
|---|---|:---:|:---:|---|
| `test_features.csv` | Eksperimen 1 | 22.236 | 24 | Dataset mentah FlakeFlagger (memiliki 273 missing values pada 5 proyek Java). Digunakan untuk strategi **C1** dan **C2**. |
| `cleaned_test_features.csv` | Eksperimen 2 | 22.236 | 24 | Dataset terimputasi bersih hasil *Group-Aware Per-Project Median & Probabilistic Imputation* (0 missing values, 811 flaky test utuh). Digunakan untuk strategi **C3**. |
| `test_results.csv` | FlakeFlagger | 22.236 | 24 | Data riwayat eksekusi rerun 10.000x per test case untuk verifikasi flakiness ground truth. |
| `Project_Info.csv` | FlakeFlagger | 24 | 24 | Metadata repositori Java (nama proyek, domain, total test case, rasio flaky). |

---

## ⚙️ Tiga (3) Paradigma Data Cleaning yang Dibandingkan (C1, C2, C3)

1. **C1: Row-Level Drop (Eksperimen 1)**:
   - Membuang 273 baris dengan missing value.
   - Menyisakan **21.963 baris data** pada seluruh **24 proyek Java**.
   - Populasi flaky: **808 flaky tests**.
2. **C2: Project-Level Drop (Eksperimen 1)**:
   - Membuang 5 proyek yang memiliki missing value (`wildfly`, `logback`, `spring-boot`, `wro4j`, `handlebars.java`).
   - Menyisakan **16.458 baris data** pada **19 proyek Java**.
   - Populasi flaky: **586 flaky tests** (222 flaky test terbuang, termasuk 163 dari `spring-boot`).
3. **C3: Group-Aware Median Imputation (Eksperimen 2)**:
   - Mengimputasi missing values berdasarkan nilai median pada grup proyek masing-masing.
   - Mempertahankan **100% data**: **22.236 baris data** pada seluruh **24 proyek Java**.
   - Populasi flaky: **811 flaky tests utuh** (tanpa data yang terbuang).

---

## 🔍 23 Fitur Prediktif Resmi FlakeFlagger

- **8 Test Smells (Biner)**: `assertion-roulette`, `conditional-test-logic`, `eager-test`, `fire-and-forget`, `indirect-testing`, `mystery-guest`, `resource-optimism`, `test-run-war`.
- **7 Execution & Coverage (Numerik)**: `testLength`, `numAsserts`, `numCoveredLines`, `ExecutionTime`, `projectSourceLinesCovered`, `projectSourceClassesCovered`, `num_third_party_libs`.
- **8 Code Churn H-Index (Numerik)**: Jendela modifikasi per baris cakupan (`window5`, `window10`, `window25`, `window50`, `window75`, `window100`, `window500`, `window10000`).
