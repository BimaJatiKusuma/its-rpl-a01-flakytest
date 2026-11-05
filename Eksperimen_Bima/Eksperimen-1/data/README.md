# 📂 Dataset Eksperimen 1 — Raw FlakeFlagger Benchmark

Folder ini memuat dataset acuan mentah yang digunakan pada **Eksperimen 1** (Full Factorial 48 Pipelines: 2 Cleaning C1/C2 × 3 Scaling × 4 Imbalance × 2 Classifiers).

---

## 📑 Berkas Dataset

| Berkas | Ukuran | Jumlah Baris | Deskripsi |
| :--- | :--- | :--- | :--- |
| **`test_features.csv`** | ~5.7 MB | 22.237 baris (header + 22.236 data) | Matriks fitur test case FlakeFlagger (29 kolom: 5 metadata/identitas, 1 target `flaky`, 23 fitur prediktif). Terdapat 273 baris dengan missing value pada `testLength`, `numAsserts`, atau `numCoveredLines`. |
| **`test_results.csv`** | ~2.9 MB | 22.246 baris | Hasil empiris 10.000x eksekusi ulang (rerun) per test case beserta frekuensi flakiness dan klasifikasi status tes. |
| **`Project_Info.csv`** | ~1.7 KB | 25 baris | Informasi dan metadata 24 repositori Java open-source (nama proyek, URL commit, jumlah test case). |

---

## 🏷️ Skema Fitur `test_features.csv`

1. **Metadata & Target**:
   - `Unnamed: 0`: Indeks baris asli
   - `test_name`: Fully qualified name test case (Primary Key pelacakan)
   - `project`: Nama repositori Java asal (variabel grouping/stratifikasi LOPO-CV)
   - `testClassName`: Nama kelas pengujian
   - `testMethodName`: Nama metode pengujian
   - `flaky`: Label target biner (`1` = Flaky, `0` = Non-Flaky)

2. **23 Fitur Prediktif**:
   - **Test Smells (8 fitur biner)**: `assertion-roulette`, `conditional-test-logic`, `eager-test`, `fire-and-forget`, `indirect-testing`, `mystery-guest`, `resource-optimism`, `test-run-war`.
   - **Metrik Eksekusi & Cakupan (7 fitur numerik)**: `testLength`, `numAsserts`, `numCoveredLines`, `ExecutionTime`, `projectSourceLinesCovered`, `projectSourceClassesCovered`, `num_third_party_libs`.
   - **Code Churn H-Index (8 fitur numerik)**: `hIndexModificationsPerCoveredLine_window5` s.d. `window10000`.

---

## ⚙️ Penanganan Missing Values pada Eksperimen 1
Dataset mentah ini diproses dalam Eksperimen 1 melalui dua strategi manipulasi pembersihan:
- **C1 (Row-Level Drop)**: Menghapus 273 baris yang memiliki nilai null $\rightarrow$ menyisakan 21.963 baris pada 24 proyek utuh.
- **C2 (Project-Level Drop)**: Menghapus 5 proyek yang terdampak missing value (`wildfly`, `logback`, `spring-boot`, `wro4j`, `handlebars.java`) $\rightarrow$ menyisakan 16.458 baris pada 19 proyek bersih.

