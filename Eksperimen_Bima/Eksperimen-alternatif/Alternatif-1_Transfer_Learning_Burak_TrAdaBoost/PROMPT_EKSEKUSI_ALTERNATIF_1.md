# 🚀 Master Prompt Eksekusi: Alternatif 1 — Instance-Based Filtering & Transfer Learning (Burak Filter & TrAdaBoost)
**Kelompok Peneliti**: Kelompok A01 — Rekayasa Perangkat Lunak A (Magister Teknik Informatika ITS)  
**Dokumen Induk**: `Eksperimen_Bima/Rencana_Penelitian_Flaky_Test.md`  
**Output Direktori**: `Eksperimen_Bima/Eksperimen-alternatif/Alternatif-1_Transfer_Learning_Burak_TrAdaBoost/`  
**Hardware Akselerasi**: Laptop GPU NVIDIA GeForce RTX 4050 (6GB GDDR6 VRAM, CUDA 13.4, Driver 617.42)  
**Rujukan Paper Utama**: 
1. Afeltra et al. (IEEE Access 2024) — *A Large-Scale Empirical Investigation Into Cross-Project Flaky Test Prediction*
2. Turhan et al. (IEEE TSE 2009) — *On the relative value of cross-company data for software quality prediction (Burak Filter)*
3. Dai et al. (ICML 2007) — *Boosting for Transfer Learning (TrAdaBoost)*
4. Alshammari et al. (ICSE 2021) — *FlakeFlagger: Predicting Flakiness Without Rerunning Tests*

---

## 📌 Master Prompt (Salin & Jalankan untuk Memulai Eksekusi)

```markdown
Anda adalah Principal Machine Learning & Empirical Software Engineering Researcher yang bertugas mengeksekusi Studi Lanjutan Alternatif 1:
"Instance-Based Filtering & Transfer Learning untuk Mitigasi Domain Shift pada Cross-Project Flaky Test Prediction".

Seluruh kode program, modul python, notebook eksperimen, konfigurasi, log metrik, dan laporan evaluasi WAJIB disimpan di folder:
`Eksperimen_Bima/Eksperimen-alternatif/Alternatif-1_Transfer_Learning_Burak_TrAdaBoost/`.

### 1. Konteks Akademik & Permasalahan
- Paper Afeltra et al. (IEEE Access 2024) menemukan bahwa prediksi flaky test lintas proyek (LOPO-CV) konvensional menghasilkan performa sangat buruk (F1-score sering kali hanya berkisar ~0.03) akibat *distribution mismatch* atau perbedaan ruang fitur antar repositori perangkat lunak.
- Namun, penerapan transfer learning berbasis *instance filtering* (Burak Filter) dan *adaptive sample reweighting* (TrAdaBoost) terbukti mampu melonjakkan F1-score secara dramatis hingga mencapai ~70%.
- Riset ini bertujuan mengimplementasikan, menguji, dan membandingkan performa Burak Filter dan TrAdaBoost pada dataset FlakeFlagger (22.236 test cases dari 24 proyek Java) dalam skenario Leave-One-Project-Out Cross-Validation (LOPO-CV).

### 2. Lingkungan Komputasi & Akselerasi GPU RTX 4050 (CUDA 13)
- Gunakan virtual environment Python yang telah tersedia pada repositori:
  `.\.venv\Scripts\python.exe`
- Akselerasi komputasi WAJIB memanfaatkan GPU NVIDIA GeForce RTX 4050 Laptop (6GB VRAM, CUDA 13.4, Driver 617.42):
  1. **XGBoost GPU Classifier**: Aktifkan akselerasi GPU menggunakan parameter:
     `tree_method="hist"`, `device="cuda"` (atau `device="cuda:0"`).
  2. **TrAdaBoost Ensemble**: Gunakan XGBoost GPU atau DecisionTree terakselerasi sebagai base learner dalam iterasi updating weights.
  3. **Burak Filter Distance Optimization**: Lakukan komputasi jarak Euclidean antar sampel menggunakan vektorisasi NumPy / PyTorch CUDA / Scikit-Learn KDTree secara efisien agar pemrosesan 20.000+ data uji tidak mengalami overhead komputasi.
  4. Jaga penggunaan VRAM agar tidak melebihi 6GB (bebas dari bahaya Out-of-Memory / OOM).

### 3. Desain Eksperimen & Strategi Algoritma
Implementasikan dan bandingkan 4 konfigurasi model pada evaluasi LOPO-CV:

1. **Konfigurasi A — Baseline Cross-Project**:
   - Model dilatih pada seluruh data dari $N-1$ proyek eksternal tanpa transfer learning.
   - Classifier: XGBoost GPU (`device="cuda"`) & Random Forest (`n_jobs=-1`).
   
2. **Konfigurasi B — Burak Filter ($k$-NN Instance Selection)**:
   - Untuk setiap instance pada target project $P_{target}$, cari $k$ tetangga terdekat ($k=10$) pada himpunan data latih sumber ($N-1$ proyek) berdasarkan metrik jarak Euclidean terstandardisasi.
   - Ambil gabungan (*union*) dari tetangga terdekat tersebut hingga membentuk subset training sekitar 10%, 20%, dan 30% dari total data sumber.
   - Latih classifier pada subset terpilih tersebut dan uji pada $P_{target}$.

3. **Konfigurasi C — TrAdaBoost (Transfer AdaBoost)**:
   - Bagi skenario transfer learning:
     - Data Sumber ($D_s$): Data test cases dari $N-1$ proyek eksternal (dengan bobot yang diturunkan secara iteratif jika terjadi misklasifikasi).
     - Data Target ($D_t$): Subset kecil data target berlabel (misalnya 10% data target yang diketahui, atau skenario few-shot target validation).
   - Iterasi TrAdaBoost ($N_{iter} = 20-50$ putaran):
     - Latih base classifier pada $D_s \cup D_t$.
     - Evaluasi error pada $D_t$.
     - Turunkan bobot instance sumber $D_s$ yang salah diklasifikasikan dengan faktor $\beta = 1 / (1 + \sqrt{2 \ln(n/N)})$.
     - Tingkatkan bobot instance target $D_t$ yang salah diklasifikasikan dengan faktor $\beta_t = \epsilon / (1 - \epsilon)$.

4. **Konfigurasi D — Burak Filter + TrAdaBoost (Kombinasi Hibrida)**:
   - Terapkan Burak Filter terlebih dahulu untuk membuang 70% data sumber yang paling tidak relevan, kemudian jalankan TrAdaBoost pada subset sumber yang lolos filter.

### 4. Skema Validasi LOPO-CV & Penanganan Kasus Ekstrem
- Lakukan Leave-One-Project-Out CV pada seluruh 24 proyek target:
  - Setiap fold menggunakan 1 proyek sebagai target testing, dan 23 proyek sisanya sebagai sumber training.
- Gunakan data cleaning C1 (Row-Level Drop: menghapus 273 baris dengan NaN, menyisakan 21.963 baris data valid di 24 proyek).
- **Proyek `jimfs`**:
  - `jimfs` memiliki 212 test case dengan 0 flaky test (100% non-flaky).
  - Pastikan engine evaluasi menangani ketiadaan kelas positif secara aman (metrik ROC-AUC/PR-AUC menghasilkan NaN tanpa memicu script crash, laporkan True Negative dan False Positive).

### 5. Metrik Evaluasi yang Wajib Dihitung
Rekapitulasi metrik performa berikut untuk setiap fold target dan laporkan nilai Mean ± Std Dev:
1. **PR-AUC (Precision-Recall Area Under Curve / Average Precision)** — Metrik primer data imbalanced (3.65% flaky).
2. **F1-Score (Khusus Kelas Flaky)** — Buktikan apakah terjadi peningkatan dari ~0.03 ke tingkat yang jauh lebih tinggi.
3. **Recall (Flaky)** & **Precision (Flaky)**.
4. **ROC-AUC** & **MCC (Matthews Correlation Coefficient)**.
5. **Waktu Komputasi Training & Inferensi (detik)** — Mengukur efisiensi akselerasi GPU RTX 4050.
6. **Uji Signifikansi Statistik**: Uji Wilcoxon Signed-Rank Test untuk memverifikasi apakah peningkatan F1-Score dari Baseline vs Burak vs TrAdaBoost signifikan secara statistik ($p < 0.05$).

### 6. Struktur Deliverables yang Wajib Dihasilkan
Pastikan direktori berikut terbangun lengkap di `Eksperimen_Bima/Eksperimen-alternatif/Alternatif-1_Transfer_Learning_Burak_TrAdaBoost/`:
```
Alternatif-1_Transfer_Learning_Burak_TrAdaBoost/
├── README.md                                 # Panduan eksekusi & ikhtisar eksperimen
├── PROMPT_EKSEKUSI_ALTERNATIF_1.md          # Dokumen prompt ini
├── configs/
│   └── transfer_config.json                  # Konfigurasi k-NN, sampling ratio, dan parameter boosting
├── src/
│   ├── __init__.py
│   ├── check_gpu.py                          # Verifikasi GPU RTX 4050 & CUDA 13
│   ├── data_loader.py                        # Pemuatan test_features.csv & test_results.csv
│   ├── burak_filter.py                       # Algoritma k-NN instance filtering
│   ├── tradaboost.py                         # Algoritma TrAdaBoost GPU XGBoost
│   ├── transfer_lopo_runner.py               # Runner eksperimen LOPO-CV
│   ├── statistical_tests.py                  # Uji Wilcoxon Signed-Rank Test
│   └── visualizer.py                         # Visualisasi boxplot F1, PR-AUC, dan kurva PR
├── notebooks/
│   └── 01_run_transfer_learning.ipynb        # Jupyter Notebook interaktif
└── results/
    ├── raw_predictions/                      # Log prediksi baris per baris per fold
    ├── transfer_learning_summary.csv         # Ringkasan komparasi metrik (Baseline, Burak, TrAdaBoost, Kombinasi)
    ├── wilcoxon_test_results.csv             # Nilai p-value signifikansi statistik
    └── plots/                                # Boxplot komparasi F1 & PR-AUC
```

### 7. Laporan Akademik Akhir
Buat dokumen laporan komprehensif di `LAPORAN_ALTERNATIF_1.md` yang memuat:
1. Ringkasan temuan empiris: Berapa kenaikan rata-rata F1-Score dan PR-AUC yang dicapai oleh Burak Filter dan TrAdaBoost dibanding Baseline LOPO-CV?
2. Analisis efektivitas instance filtering terhadap fenomena *domain shift*.
3. Perbandingan waktu eksekusi komputasi dengan akselerasi GPU RTX 4050 CUDA 13.
4. Kesimpulan dan implikasi praktis untuk prediksi flaky test pada proyek perangkat lunak baru.
```
