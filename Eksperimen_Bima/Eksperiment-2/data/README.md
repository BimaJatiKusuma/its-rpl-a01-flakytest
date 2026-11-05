# 📂 Dataset Eksperimen 2 — Preprocessed & Imputed FlakeFlagger Dataset

Folder ini memuat dataset bersih yang digunakan pada **Eksperimen 2** (Full Factorial 24 Pipelines pada 24 Proyek Utuh).

---

## 📑 Berkas Dataset

| Berkas | Ukuran | Dimensi | Deskripsi |
| :--- | :--- | :--- | :--- |
| **`cleaned_test_features.csv`** | ~5.8 MB | 22.236 baris × 29 kolom | Dataset FlakeFlagger yang telah diimputasi menggunakan **Group-Aware Per-Project Median & Probabilistic Fallback Imputation**. Memiliki **0 missing values**, mempertahankan **100% dari 24 proyek**, dan melestarikan **seluruh 811 flaky test** tanpa pembuangan data (eliminasi strategi C1/C2). |

---

## 🛠️ Reproduksibilitas & Pembuatan Ulang Dataset

Dataset ini dihasilkan dari dataset mentah `test_features.csv` menggunakan algoritma imputasi cerdas. Anda dapat mereproduksi atau memvalidasi ulang dataset ini kapan saja dengan menjalankan:

```bash
# Menjalankan script imputasi mandiri:
python src/generate_cleaned_dataset.py

# Atau membangun ulang notebook Phase 1 lengkap:
python src/build_full_notebook.py
```

Notebook interaktif eksplorasi Phase 1 juga tersedia di [`../notebooks/00_eda_and_data_cleaning.ipynb`](../notebooks/00_eda_and_data_cleaning.ipynb).

---

## 🏷️ Skema Kolom `cleaned_test_features.csv`

1. **Metadata Identitas (5 kolom)**:
   - `Unnamed: 0`: Indeks baris asli dari dataset mentah
   - `test_name`: Nama lengkap test case (Primary Key pelacakan)
   - `project`: Nama repositori Java asal (variabel grouping LOPO-CV)
   - `testClassName`: Nama kelas pengujian
   - `testMethodName`: Nama metode pengujian

2. **Target Prediksi (1 kolom)**:
   - `flaky`: Indikator biner ke-flaky-an test case (`1` = Flaky, `0` = Non-Flaky)

3. **23 Fitur Prediktif Resmi**:
   - **8 Test Smells (Biner)**: `assertion-roulette`, `conditional-test-logic`, `eager-test`, `fire-and-forget`, `indirect-testing`, `mystery-guest`, `resource-optimism`, `test-run-war`.
   - **7 Metrik Eksekusi & Cakupan (Numerik)**: `testLength`, `numAsserts`, `numCoveredLines`, `ExecutionTime`, `projectSourceLinesCovered`, `projectSourceClassesCovered`, `num_third_party_libs`.
   - **8 Code Churn H-Index (Numerik)**: `hIndexModificationsPerCoveredLine_window5` s.d. `window10000`.

---

## 💡 Keunggulan Metodologis Imputasi Group-Aware

| Aspek | Eksperimen 1 (C1: Row Drop) | Eksperimen 1 (C2: Project Drop) | Eksperimen 2 (Group-Aware Median) |
| :--- | :--- | :--- | :--- |
| **Jumlah Proyek** | 24 proyek | 19 proyek (5 dibuang) | **24 proyek (100% utuh)** |
| **Total Test Case** | 21.963 baris | 16.458 baris | **22.236 baris (100% utuh)** |
| **Flaky Test Cases** | 808 kasus | 621 kasus | **811 kasus (100% utuh)** |
| **Proyek Kritis** | Terpotong sebagian | `spring-boot`, `wildfly` hilang | **Terpelihara utuh** |
| **Missing Values** | Dibuang paksa | Dibuang satu proyek penuh | **0 missing values (Imputasi Terarah)** |

