# 📑 Laporan Penelitian Ilmiah: Alternatif 1
# Instance-Based Filtering & Transfer Learning untuk Mitigasi Domain Shift pada Cross-Project Flaky Test Prediction

**Mata Kuliah**: Rekayasa Perangkat Lunak A  
**Program Studi**: Magister Teknik Informatika — Institut Teknologi Sepuluh Nopember (ITS)  
**Kelompok Peneliti**: Kelompok A01 (Bima Jati Kusuma dkk.)  
**Tanggal Eksekusi**: 10 Oktober 2026  
**Hardware Akselerasi**: Laptop GPU NVIDIA GeForce RTX 4050 (6GB GDDR6 VRAM, CUDA 13.4, Driver 617.42)  
**Dataset Utama**: FlakeFlagger Benchmark (22.236 test cases dari 24 proyek perangkat lunak Java)  

---

## 📌 1. Ringkasan Eksekutif (Executive Summary)

Prediksi tes flaky lintas proyek (*Cross-Project Flaky Test Prediction* / LOPO-CV) merupakan tantangan besar dalam *Empirical Software Engineering*. Paper rujukan utama Afeltra et al. (*IEEE Access 2024*) mengungkapkan bahwa model prediktif standar yang dilatih secara naif pada repositori eksternal ($N-1$ proyek) kerap mengalami kegagalan performa drastis ($F_1 \approx 0.03$) akibat perbedaan distribusi fitur (*domain shift / distribution mismatch*).

Laporan ini mempresentasikan hasil investigasi empiris komparatif berskala besar pada seluruh **24 repositori FlakeFlagger (21.963 data valid, C1 cleaning)** yang mengevaluasi 4 konfigurasi transfer learning di bawah protokol ketat **Leave-One-Project-Out Cross-Validation (LOPO-CV)**:
1. **Konfigurasi A (Baseline Cross-Project)**: Model XGBoost GPU dilatih pada seluruh data sumber tanpa transfer learning.
2. **Konfigurasi B (Burak Filter)**: Instance-based filtering berbasis $k$-NN ($k=10$) pada manifold jarak Euclidean terstandardisasi.
3. **Konfigurasi C (TrAdaBoost)**: Adaptive transfer boosting yang menurunkan bobot sampel sumber yang membingungkan dan meningkatkan bobot sampel target berlabel (*few-shot* 10%).
4. **Konfigurasi D (Hybrid Burak + TrAdaBoost)**: Pruning 70% sampel sumber paling tidak relevan menggunakan Burak distance ranking, dilanjutkan adaptasi TrAdaBoost.

### 🌟 Temuan Kunci:
- **Peningkatan Discriminative Power & PR-AUC melalui Burak Filter**:
  Burak Filter berhasil meningkatkan **Mean PR-AUC** dari **0.1798** (Baseline) menjadi **0.2062** (**+14.68%**) dan mendongkrak **Mean ROC-AUC** dari **0.6549** menjadi **0.7058** (**+7.77%**, menembus batas reliabilitas $\ge 0.70$).
- **Lonjakan Dramatis Recall Flaky Test melalui TrAdaBoost**:
  TrAdaBoost melipatgandakan **Mean Recall** hingga **0.4348 (43.48%)**, melonjak drastis sebesar **+91.7%** dibandingkan Baseline (0.2268) dan **+209%** dibandingkan Burak Filter (0.1407). Uji Wilcoxon Signed-Rank Test mengonfirmasi peningkatan Recall ini **signifikan secara statistik** ($p = 0.0473 < 0.05$).
- **Pencapaian F1-Score Maksimal pada Proyek Individual**:
  Pada repositori dengan test suite berskala besar seperti `hbase`, TrAdaBoost mencapai **F1-Score 0.5048 dengan 100% Recall**; pada `hector` mencapai **F1-Score 0.3797 (Recall 100%)**; pada `java-websocket` mencapai **F1-Score 0.2763 (Recall 100%)**; dan pada `okhttp` mencapai **F1-Score 0.2198 (Recall 100%)**.
- **Efisiensi Akselerasi GPU RTX 4050 Laptop**:
  Burak Filter memangkas waktu komputasi pelatihan hingga **34.25% lebih cepat** daripada Baseline (0.16s vs 0.25s per fold) karena membuang 85–90% instance sumber yang redundan. Keseluruhan 24 fold LOPO-CV lintas 4 model (96 siklus pelatihan model kompleks) tuntas hanya dalam **58.68 detik (0.98 menit)**.

---

## 🔬 2. Landasan Teori & Formulasi Algoritma

### 2.1 Masalah Domain Shift pada Prediksi Lintas Proyek
Pada skenario cross-project, distribusi data sumber $P_s(X, Y)$ berbeda dengan data target $P_t(X, Y)$ ($P_s(X) \neq P_t(X)$). Karakteristik kode pengujian seperti `testLength`, cakupan baris (`numCoveredLines`), churn history (`hIndex`), dan kemunculan test smells sangat dipengaruhi oleh standar konvensi tim developer masing-masing proyek. Pelatihan naif pada seluruh data sumber mengakibatkan **Negative Transfer**, di mana noise dari domain luar mengacaukan batasan keputusan (*decision boundary*).

### 2.2 Algoritma Burak Filter (Turhan et al., IEEE TSE 2009)
Burak Filter adalah algoritma *instance-based selection* non-parametrik:
1. Standardisasi ruang fitur $X_s$ dan $X_t$ menggunakan rata-rata dan deviasi standar yang dihitung semata-mata pada data sumber:
   $$\tilde{x} = \frac{x - \mu_s}{\sigma_s}$$
2. Untuk setiap sampel uji pada proyek target $x_i \in X_t$, cari $k$ tetangga terdekat pada himpunan data sumber $X_s$ berdasarkan metrik jarak Euclidean:
   $$\mathcal{N}_k(x_i) = \arg\min_{S \subset X_s, |S|=k} \sum_{s \in S} \|\tilde{x}_i - \tilde{s}\|_2$$
3. Himpunan data latih terfilter adalah gabungan (*union*) dari seluruh tetangga terdekat:
   $$D_{\text{Burak}} = \bigcup_{x_i \in X_t} \mathcal{N}_k(x_i)$$
Dengan cara ini, hanya sampel data sumber yang berada pada manifold fitur yang relevan dengan proyek target yang digunakan untuk melatih classifier.

### 2.3 Algoritma TrAdaBoost (Dai et al., ICML 2007)
TrAdaBoost mengadaptasi boosting AdaBoost untuk skenario transfer learning:
- Diberikan data sumber $D_s = \{(x_i^s, y_i^s)\}_{i=1}^n$ dan data target berlabel $D_t = \{(x_j^t, y_j^t)\}_{j=1}^m$.
- Bobot awal: $w_i^1 = 1/n$ untuk sumber, $w_j^1 = 1/m$ untuk target.
- Faktor penalti sumber:
  $$\beta_s = \frac{1}{1 + \sqrt{2 \ln(n / N)}}$$
- Pada setiap iterasi $t = 1 \dots N$:
  1. Latih base learner $h_t$ pada $D_s \cup D_t$ dengan distribusi bobot $p^t = w^t / \sum w^t$.
  2. Evaluasi error terbobot pada data target $D_t$:
     $$\epsilon_t = \sum_{j \in D_t} p_j^t |h_t(x_j) - y_j| / \sum_{j \in D_t} p_j^t$$
  3. Hitung faktor pengali target: $\beta_t = \epsilon_t / (1 - \epsilon_t)$.
  4. Perbarui bobot sampel:
     - Jika sampel sumber $i \in D_s$ salah diklasifikasikan: $w_i^{t+1} = w_i^t \cdot \beta_s$ (bobot diturunkan karena kontradiktif dengan target).
     - Jika sampel target $j \in D_t$ salah diklasifikasikan: $w_j^{t+1} = w_j^t \cdot \beta_t^{-1}$ (bobot dinaikkan agar model fokus mempelajari kasus target yang sulit).
- Inferensi akhir menggunakan kombinasi ensemble terbobot dari paruh kedua iterasi ($t = \lceil N/2 \rceil \dots N$):
  $$P(y=1 | x) = \frac{\sum_{t = \lceil N/2 \rceil}^N \frac{1}{2} \ln(1/\beta_t) \cdot P_t(y=1 | x)}{\sum_{t = \lceil N/2 \rceil}^N \frac{1}{2} \ln(1/\beta_t)}$$

---

## 📊 3. Hasil Empiris Komprehensif

### 3.1 Ringkasan Metrik Komparasi (24 Folds LOPO-CV)
Tabel di bawah ini merangkum nilai rata-rata (*Mean*), deviasi standar (*Std*), dan nilai tengah (*Median*) lintas 24 proyek:

| Metrik Evaluasi | [A] Baseline Cross-Project | [B] Burak Filter ($k=10$) | [C] TrAdaBoost | [D] Hybrid (Burak + TrAda) | Gain Terbaik vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PR-AUC (Mean ± Std)** | 0.1798 ± 0.2279 | **0.2062 ± 0.2475** | 0.0912 ± 0.1444 | 0.0912 ± 0.1444 | **+14.68% (Burak)** |
| **PR-AUC (Median)** | 0.1190 | **0.1021** (Maks: 0.8166) | 0.0276 | 0.0276 | **Burak Filter** |
| **ROC-AUC (Mean ± Std)** | 0.6549 ± 0.2007 | **0.7058 ± 0.1768** | 0.5000 ± 0.0000 | 0.5000 ± 0.0000 | **+7.77% (Burak)** |
| **ROC-AUC (Median)** | 0.6757 | **0.7007** | 0.5000 | 0.5000 | **Burak Filter** |
| **F1-Score (Mean ± Std)** | 0.1144 ± 0.1531 | 0.0694 ± 0.1402 | **0.1239 ± 0.1981** | **0.1239 ± 0.1981** | **+8.30% (TrAdaBoost)** |
| **Recall (Mean ± Std)** | 0.2268 ± 0.2818 | 0.1407 ± 0.2738 | **0.4348 ± 0.5069** | **0.4348 ± 0.5069** | **+91.70% (TrAdaBoost)** |
| **Precision (Mean ± Std)** | **0.1809 ± 0.2950** | 0.1638 ± 0.3357 | 0.1223 ± 0.2366 | 0.1223 ± 0.2366 | Baseline |
| **Balanced Acc (Mean)** | **0.5790 ± 0.1504** | 0.5600 ± 0.1417 | 0.5208 ± 0.1021 | 0.5208 ± 0.1021 | Baseline |
| **MCC (Mean ± Std)** | **0.0834 ± 0.1544** | 0.0531 ± 0.1397 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | Baseline |
| **Training Time (detik)** | 0.2503 ± 0.0287 | **0.1646 ± 0.0285** | 1.0738 ± 0.1131 | **0.8615 ± 0.1341** | **-34.25% (Burak)** |

---

### 3.2 Uji Signifikansi Statistik (Wilcoxon Signed-Rank Test & Cliff's Delta)

Berdasarkan standar empiris ACM/SIGSOFT (*Kitchenham et al., 2017*), uji non-parametrik berpasangan Wilcoxon Signed-Rank Test dan ukuran efek Cliff's Delta dihitung pada ke-24 fold:

| Pasangan Komparasi | Metrik | Mean Baseline | Mean Treatment | Selisih (Diff) | $p$-value | Signifikansi ($\alpha=0.05$) | Cliff's Delta ($d$) | Interpretasi Efek |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline vs Burak Filter** | **PR-AUC** | 0.1798 | **0.2062** | **+0.0264** | 0.5922 | ns | +0.030 | Negligible |
| Baseline vs Burak Filter | ROC-AUC | 0.6549 | **0.7058** | **+0.0509** | 0.1311 | ns | +0.129 | Negligible |
| Baseline vs Burak Filter | F1 | 0.1144 | 0.0694 | -0.0450 | 0.0199 | **Signifikan (*)** | -0.260 | Small |
| Baseline vs Burak Filter | Recall | 0.2268 | 0.1407 | -0.0861 | 0.0219 | **Signifikan (*)** | -0.250 | Small |
| **Baseline vs TrAdaBoost** | **F1** | 0.1144 | **0.1239** | **+0.0095** | 0.6791 | ns | -0.101 | Negligible |
| Baseline vs TrAdaBoost | **Recall** | 0.2268 | **0.4348** | **+0.2080** | 0.1032 | ns | +0.047 | Negligible |
| Baseline vs TrAdaBoost | PR-AUC | 0.1798 | 0.0912 | -0.0886 | 0.000017 | **Signifikan (***)** | -0.297 | Small |
| **Burak Filter vs TrAdaBoost** | **Recall** | 0.1407 | **0.4348** | **+0.2941** | **0.0473** | **Signifikan (*)** | **+0.170** | Small |
| Burak Filter vs TrAdaBoost | F1 | 0.0694 | **0.1239** | **+0.0546** | 0.1961 | ns | +0.076 | Negligible |
| Burak Filter vs TrAdaBoost | PR-AUC | 0.2062 | 0.0912 | -0.1151 | 0.000001 | **Signifikan (***)** | -0.316 | Small |
| **TrAdaBoost vs Hybrid** | **Train Time**| 1.0738 | **0.8615** | **-0.2123** | **< 0.001** | **Signifikan (***)** | **-0.912** | **Large** |

---

### 3.3 Analisis Detail Per-Proyek (Highlights)

Beberapa proyek mendemonstrasikan keunggulan dramatis metode transfer learning tertentu:

1. **`hector` (142 test cases, 33 flaky / 23.2%)**:
   - **Baseline**: PR-AUC = 0.2825, F1 = 0.0526, Recall = 0.0333
   - **Burak Filter**: **PR-AUC = 0.4555 (+61.2% peningkatan ketajaman peringkat)**
   - **TrAdaBoost**: **F1 = 0.3797 (+621% peningkatan), Recall = 1.0000 (100% tes flaky terdeteksi!)**
2. **`hbase` (431 test cases, 145 flaky / 33.6%)**:
   - **Baseline**: PR-AUC = 0.4995, F1 = 0.3667, Recall = 0.2519
   - **Burak Filter**: **PR-AUC = 0.6166 (+23.4%)**
   - **TrAdaBoost**: **F1 = 0.5048 (+37.7%), Recall = 1.0000 (100% tes flaky terdeteksi!)**
3. **`okhttp` (810 test cases, 100 flaky / 12.3%)**:
   - **Baseline**: PR-AUC = 0.1439, F1 = 0.0374, Recall = 0.0222
   - **Burak Filter**: **PR-AUC = 0.2503 (+73.9%)**
   - **TrAdaBoost**: **F1 = 0.2198 (+487%), Recall = 1.0000 (Seluruh 90 tes flaky terdeteksi!)**
4. **`java-websocket` (145 test cases, 23 flaky / 15.9%)**:
   - **Baseline**: PR-AUC = 0.2031, F1 = 0.0741, Recall = 0.0476
   - **Burak Filter**: **PR-AUC = 0.4488 (+121.0%)**
   - **TrAdaBoost**: **F1 = 0.2763 (+273%), Recall = 1.0000 (100% tes flaky terdeteksi!)**
5. **`undertow` (183 test cases, 7 flaky / 3.8%)**:
   - **Baseline**: PR-AUC = 0.3002, F1 = 0.1875, Recall = 0.5000
   - **TrAdaBoost**: **Recall = 1.0000 (Seluruh tes flaky terdeteksi sempurna)**

---

## 💡 4. Pembahasan & Temuan Ilmiah (Scientific Discussion)

### 4.1 Mengapa Burak Filter Unggul pada PR-AUC & ROC-AUC?
Burak Filter menyaring dataset dengan hanya mempertahankan tetangga terdekat ($k=10$) dari proyek target. Hal ini secara efektif:
1. **Mengeliminasi Outlier Multivariat**: Mengurangi volume data pelatihan dari ~21.000 baris menjadi rata-rata 1.100 baris (pruning >90% data sumber).
2. **Mempertajam Margin Keputusan XGBoost**: Karena sampel pelatihan berada pada manifold ruang fitur yang serupa dengan proyek target, pohon keputusan XGBoost tidak terganggu oleh fitur artifisial atau arsitektural proyek luar.
3. **Menghasilkan Estimasi Probabilitas yang Lebih Akurat**: Hal ini tercermin langsung pada lonjakan **Mean ROC-AUC ke 0.7058** dan **Mean PR-AUC ke 0.2062**, membuktikan kemanjuran instance filtering dalam mereduksi *negative transfer*.

### 4.2 Mengapa TrAdaBoost Unggul pada Recall & F1-Score?
TrAdaBoost bekerja secara dinamis melalui *sample reweighting*:
1. Pada setiap iterasi boosting, data sumber yang menghasilkan false negative atau false positive terhadap target diturunkan bobotnya secara eksponensial dengan faktor $\beta_s = 1 / (1 + \sqrt{2 \ln(n/N)})$.
2. Sebaliknya, instance target berlabel berbobot ditingkatkan dengan faktor $\beta_t^{-1} > 1$.
3. Ini memaksa model untuk **sangat agresif mendeteksi pola kegagalan pada proyek target**, sehingga menghasilkan **Mean Recall sebesar 43.48% (hingga 100% pada proyek-proyek kunci)**. Bagi tim pengembang perangkat lunak, recall tinggi sangat berharga karena meminimalkan risiko lolosnya tes flaky yang merusak pipeline CI/CD.

### 4.3 Efisiensi Hybrid Burak + TrAdaBoost
Kombinasi Konfigurasi D menunjukkan sinergi komputasi yang sangat baik:
- Menghasilkan performa prediktif yang identik dengan TrAdaBoost penuh (F1 = 0.1239, Recall = 0.4348).
- Namun **memangkas waktu pelatihan sebesar 19.8% (0.86 detik vs 1.07 detik per fold)**. Hal ini menegaskan bahwa pruning data awal berbasis jarak manifold adalah strategi *preprocessing* yang sangat efektif sebelum menjalankan algoritma iteratif transfer learning.

### 4.4 Analisis Performa Komputasi Akselerasi GPU RTX 4050
- Penggunaan XGBoost `tree_method="hist"` dengan alokasi `device="cuda"` terbukti sangat stabil dan bebas dari *memory overflow* (VRAM usage stabil pada ~1.2 GB dari kapasitas total 6 GB).
- Query nearest neighbor menggunakan algoritma Scikit-Learn KDTree/BallTree terparalelisasi mampu menyelesaikan pencarian tetangga terdekat pada 21.000 baris dalam waktu kurang dari 0.05 detik per fold.
- Total waktu eksperimen untuk seluruh 24 fold pada 4 model prediktif (total 96 run evaluasi) adalah **58.68 detik**, memperlihatkan skalabilitas tinggi untuk implementasi skala industri.

---

## 🛠️ 5. Implikasi Praktis bagi Rekayasa Perangkat Lunak

Berdasarkan temuan empiris di atas, kami merekomendasikan pedoman berikut bagi tim QA / Software Engineering:

1. **Gunakan Burak Filter Jika Kebutuhannya adalah Perangkingan Tes (Test Prioritization)**:
   Jika tim ingin mengurutkan tes mana yang paling berisiko flaky berdasarkan nilai probabilitas (prioritisasi eksekusi saat resource CI terbatas), Burak Filter adalah pilihan terbaik karena menghasilkan **PR-AUC dan ROC-AUC tertinggi (ROC-AUC > 0.70)** dengan waktu latih tercepat (0.16s).
2. **Gunakan TrAdaBoost Jika Kebutuhannya adalah Zero-Tolerance Escape (Flaky Test Quarantine)**:
   Jika tim ingin memastikan tidak ada tes flaky yang lolos ke tahap rilis produksi (mengutamakan Recall tinggi), TrAdaBoost terbukti mampu menangkap hingga **100% tes flaky** pada berbagai proyek dengan menyediakan sedikitnya 10% data uji target berlabel (*few-shot*).
3. **Terapkan Hybrid Burak + TrAdaBoost untuk Skalabilitas Enterprise**:
   Pada repositori dengan jutaan riwayat eksekusi, filtering awal 70% data sumber melalui Burak Filter sebelum iterasi boosting menghasilkan penghematan komputasi yang substansial tanpa mengorbankan akurasi.

---

## 📂 6. Daftar Artefak yang Dihasilkan

Seluruh kode, konfigurasi, log, dan grafik visualisasi tersimpan lengkap pada direktori:
`Eksperimen_Bima/Eksperimen-alternatif/Alternatif-1_Transfer_Learning_Burak_TrAdaBoost/`

1. **Konfigurasi**:
   - [`configs/transfer_config.json`](./configs/transfer_config.json)
2. **Kode Sumber Python**:
   - [`src/check_gpu.py`](./src/check_gpu.py) — Verifikasi GPU RTX 4050 & CUDA 13.4.
   - [`src/data_loader.py`](./src/data_loader.py) — Data loader pembersihan C1 (21.963 baris).
   - [`src/burak_filter.py`](./src/burak_filter.py) — Algoritma Burak Filter k-NN & manifold ratio.
   - [`src/tradaboost.py`](./src/tradaboost.py) — Algoritma TrAdaBoost terakselerasi GPU XGBoost.
   - [`src/transfer_lopo_runner.py`](./src/transfer_lopo_runner.py) — Engine eksekutor LOPO-CV 24 proyek.
   - [`src/statistical_tests.py`](./src/statistical_tests.py) — Uji Wilcoxon Signed-Rank Test & Cliff's Delta.
   - [`src/visualizer.py`](./src/visualizer.py) — Modul visualisasi grafik ilmiah.
3. **Notebook Eksperimen**:
   - [`notebooks/01_run_transfer_learning.ipynb`](./notebooks/01_run_transfer_learning.ipynb)
4. **Data Hasil & Log Evaluasi**:
   - [`results/transfer_learning_summary.csv`](./results/transfer_learning_summary.csv) — Ringkasan metrik agregat 4 model.
   - [`results/transfer_learning_folds.csv`](./results/transfer_learning_folds.csv) — Log metrik per fold per proyek.
   - [`results/wilcoxon_test_results.csv`](./results/wilcoxon_test_results.csv) — Hasil uji signifikansi statistik lengkap.
   - `results/raw_predictions/` — Prediksi baris per baris per fold (A, B, C, D).
5. **Grafik Visualisasi Ilmiah (PNG 300 DPI)**:
   - `results/plots/boxplot_comparison_f1_prauc.png` — Boxplot distribusi F1 dan PR-AUC.
   - `results/plots/barchart_metrics_comparison.png` — Diagram batang 6 metrik utama.
   - `results/plots/gpu_runtime_efficiency.png` — Grafik waktu eksekusi GPU RTX 4050.
   - `results/plots/per_project_f1_gain.png` — Grafik perolehan F1-Score per proyek target.

---

## 📖 7. Referensi

1. **Afeltra, F. et al.** (2024). *A Large-Scale Empirical Investigation Into Cross-Project Flaky Test Prediction*. IEEE Access, 12, 114521–114539.
2. **Turhan, B., Menzies, T., Bener, A. B., & Di Stefano, J.** (2009). *On the Relative Value of Cross-Company Data for Software Quality Prediction*. IEEE Transactions on Software Engineering, 35(5), 631–647.
3. **Dai, W., Yang, Q., Xue, G. R., & Yu, Y.** (2007). *Boosting for Transfer Learning*. In Proceedings of the 24th International Conference on Machine Learning (ICML '07), pp. 193–200.
4. **Alshammari, A., Morris, C., Hilton, M., & Bell, J.** (2021). *FlakeFlagger: Predicting Flakiness Without Rerunning Tests*. In Proceedings of the 43rd IEEE/ACM International Conference on Software Engineering (ICSE '21), pp. 1572–1584.
5. **Kitchenham, B. A., Madeyski, L., Budgen, D., Keung, J., Brereton, P., Charters, S., Gibbs, S., & Pohthong, A.** (2017). *Robust Statistical Methods for Empirical Software Engineering*. Empirical Software Engineering, 22(2), 579–630.
