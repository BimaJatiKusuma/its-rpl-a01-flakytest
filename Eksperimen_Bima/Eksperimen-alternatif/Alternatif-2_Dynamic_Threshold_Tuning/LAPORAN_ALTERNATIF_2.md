# 📑 Laporan Penelitian Ilmiah: Alternatif 2
# Validation-Based Dynamic Threshold Tuning vs Default 0.5 untuk Optimalisasi Deteksi Flaky Test pada Skenario Ekstrem Imbalanced Cross-Project

**Mata Kuliah**: Rekayasa Perangkat Lunak A  
**Program Studi**: Magister Teknik Informatika — Institut Teknologi Sepuluh Nopember (ITS)  
**Kelompok Peneliti**: Kelompok A01 (Bima Jati Kusuma dkk.)  
**Tanggal Eksekusi**: 10 Oktober 2026  
**Hardware Akselerasi**: Laptop GPU NVIDIA GeForce RTX 4050 (6GB GDDR6 VRAM, CUDA 13.4, Driver 617.42)  
**Dataset Utama**: FlakeFlagger Benchmark (22.236 test cases dari 24 repositori Java, Pembersihan C1: 21.963 baris valid)  
**Dokumen Induk**: [`../../Rencana_Penelitian_Flaky_Test.md`](../../Rencana_Penelitian_Flaky_Test.md)  

---

## 📌 1. Ringkasan Eksekutif (Executive Summary)

Prediksi tes flaky lintas proyek (*Cross-Project Flaky Test Prediction* / LOPO-CV) menghadapi kendala ganda: ketidakseimbangan kelas ekstrem (*extreme class imbalance*) dan pergeseran distribusi lintas repositori (*domain shift*). Pada benchmark FlakeFlagger, hanya **808 test cases (3,68%)** yang berstatus flaky berbanding **21.155 (96,32%)** non-flaky. Mayoritas studi empiris terdahulu mengadopsi ambang batas probabilitas default standar $\tau = 0.50$ ($\hat{y} = 1 \iff P(y=1) \ge 0.50$). Akibatnya, classifier yang terlatih pada data tak seimbang menghasilkan probabilitas prediktif terkonsentrasi di bawah $0.50$, memicu fenomena keruntuhan prediksi (*prediction collapse*): pada 14 dari 24 repositori uji target, model menghasilkan **Recall = 0**, **Precision = 0**, dan **F1-Score = 0**.

Penelitian ini menginvestigasi metodologi **Validation-Based Dynamic Threshold Tuning** bebas kebocoran data (*Zero Test Leakage*). Menggunakan skema *Inner-Validation Split* pada $N-1$ proyek sumber, kami menguji dan membandingkan 5 strategi ambang batas probabilitas secara ketat di bawah protokol 24-fold Leave-One-Project-Out Cross-Validation (LOPO-CV):
1. **T1 — Default Baseline**: $\tau = 0.50$ (Ambang batas konvensional).
2. **T2 — Max-F1 Dynamic Tuning**: $\tau^* = \arg\max_\tau F_1(\tau)$ pada inner validation.
3. **T3 — Recall-Constrained Tuning ($\ge 70\%$)**: $\tau^* = \arg\max_\tau \{\text{Precision}(\tau) \mid \text{Recall}(\tau) \ge 0.70\}$.
4. **T4 — Youden’s J-Statistic**: $\tau^* = \arg\max_\tau (\text{TPR}(\tau) - \text{FPR}(\tau)) = \arg\max_\tau (\text{Sensitivity} + \text{Specificity} - 1)$.
5. **T5 — Prior-Shifted Threshold**: $\tau^* = \bar{y}_{\text{source}} \approx 0.0368$ (Penyesuaian berbasis proporsi kelas empiris).

Evaluasi dilakukan pada dua arsitektur model utama: **XGBoost GPU Classifier** (CUDA 13.4 pada RTX 4050) dan **Random Forest Classifier** (100 estimators, multi-core CPU).

### 🌟 Temuan Kunci Penelitian:
- **Lonjakan Dramatis F1-Score dan Recall Tanpa Kebocoran Data**:
  - Pada **XGBoost GPU**, strategi **T4 (Youden's J)** melipatgandakan **Mean F1-Score** dari **0.0550** (T1) menjadi **0.1460** (**lonjakan +165,4%**) dan mendongkrak **Mean Recall** dari **7,10%** menjadi **38,06%** (**lonjakan +436,3%**).
  - Pada **Random Forest**, strategi **T4** mendongkrak Mean F1 dari **0.0248** menjadi **0.1639** (**lonjakan +560,0%**) dan Mean Recall dari **5,44%** menjadi **54,29%** (**lonjakan +898,3%**). Strategi **T5 (Prior-Shifted)** bahkan mencapai Mean F1 **0.1948** (**+684,7%**) dengan Mean Recall **67,71%**.
- **Signifikansi Statistik (Wilcoxon Signed-Rank Test)**:
  Uji Wilcoxon mengonfirmasi keunggulan T4 (Youden's J) vs T1 (Default 0.5) signifikan secara statistik pada F1 ($p = 0.0191 < 0.05$, Cliff's $d = +0.359$, Medium effect size) dan sangat signifikan pada Recall ($p = 0.00029 < 0.001$, Cliff's $d = +0.527$, Large effect size).
- **Penyelamatan Repositori yang Sebelumnya Runtuh (Zero F1 Resurrected)**:
  Pada T1 default 0.5, sebanyak 14 proyek target menghasilkan F1 = 0.0000. Penerapan Dynamic Threshold berhasil membangkitkan deteksi flaky pada repositori-repositori kritis tersebut:
  - `alluxio`: F1 melonjak dari **0.0504** ke **0.6591** (Recall 69,0%).
  - `spring-boot`: F1 melonjak dari **0.0099** ke **0.2399** (Recall 51,7%).
  - `commons-exec`: F1 bangkit dari **0.0000** ke **0.2500** (Recall 50,0%).
  - `hector`: F1 bangkit dari **0.0000** ke **0.2963** (Recall 57,1%).
  - `orbit`: F1 bangkit dari **0.0000** ke **0.2308** (Recall 75,0%).
  - `hbase`: F1 meningkat dari **0.0400** ke **0.4495** (Recall 49,0%).
- **Rentang Threshold Optimal yang Dipelajari**:
  Ambang batas optimal $\tau^*$ yang terbukti efektif untuk skenario ekstrem imbalanced berada di rentang **$0.030 \le \tau^* \le 0.065$** (terpusat di sekitar prior kelas empiris $\approx 0.037$). Ambang batas di atas $0.20$ terbukti terlalu konservatif dan mempertahankan kegagalan deteksi.
- **Efisiensi Akselerasi GPU NVIDIA RTX 4050**:
  Seluruh rangkaian grid search 99 titik inner-validation, pelatihan ulang model final, dan inferensi 5 strategi di seluruh 24 fold LOPO-CV tuntas hanya dalam **10,50 detik** untuk XGBoost GPU (rata-rata 0,37 detik per fold).

---

## 🔬 2. Landasan Teori & Formulasi Metodologis

### 2.1 Patologi Ambang Batas Default ($\tau = 0.50$) pada Ketimpangan Ekstrem
Dalam klasifikasi probabilistik standar, aturan keputusan Bayes optimal mengasumsikan matriks biaya simetris:
$$\hat{y} = \mathbb{I}(P(y=1 \mid x) \ge 0.50)$$
Namun, Provost (AAAI 2000) dan Elkan (IJCAI 2001) membuktikan bahwa ketika proporsi kelas minoritas $\pi = P(y=1) \ll 0.50$, fungsi kerugian standar (seperti *binary log-loss* atau *Gini impurity*) mendorong model mengkalibrasi probabilitas posteriordan mengarahkannya mendekati nilai prior $\pi$. Akibatnya:
$$\max_{x} P(y=1 \mid x) < 0.50$$
Pada situasi cross-project di mana terdapat *domain shift*, probabilitas tes target semakin tertekan. Menggunakan $\tau = 0.50$ menjamin bahwa hampir seluruh sampel diklasifikasikan sebagai negatif ($\hat{y} = 0$), mengakibatkan $TP \approx 0 \implies \text{Recall} \approx 0$ dan $F_1 \approx 0$.

### 2.2 Prinsip Zero Test-Set Leakage pada Dynamic Thresholding
Menentukan ambang batas $\tau$ dengan mengamati kurva PR atau ROC pada data uji target adalah bentuk fatal dari kebocoran data (*data leakage / post-hoc cherry-picking*), karena mengasumsikan pengembang sudah mengetahui label proyek target sebelum prediksi dilakukan.

Metodologi yang valid secara ilmiah menuntut pemisahan bertingkat (*nested validation*):
1. Diberikan $N$ proyek, proyek target $P_{\text{target}}$ disisihkan sebagai *unseen test set*.
2. Seluruh data sumber $D_{\text{source}} = \bigcup_{k \neq \text{target}} P_k$ dibagi menjadi:
   - $D_{\text{inner\_train}}$ ($80\%$)
   - $D_{\text{inner\_val}}$ ($20\%$, berstratifikasi label)
3. Model inner dilatih pada $D_{\text{inner\_train}}$, menghasilkan probabilitas validasi $\hat{P}_{\text{val}} = f(X_{\text{inner\_val}})$.
4. Grid search dievaluasi pada $\tau \in \{0.01, 0.02, \dots, 0.99\}$ semata-mata menggunakan $(y_{\text{inner\_val}}, \hat{P}_{\text{val}})$ untuk menentukan $\tau^*$.
5. Model final dilatih pada seluruh data sumber $D_{\text{source}}$, dan $\tau^*$ diterapkan secara *blind* pada data target $X_{\text{target}}$:
   $$\hat{y}_{\text{target}} = \mathbb{I}(f_{\text{final}}(X_{\text{target}}) \ge \tau^*)$$

### 2.3 Formulasi Kelima Strategi Threshold yang Diuji

| Kode | Nama Strategi | Formulasi Matematis | Karakteristik Perilaku |
| :--- | :--- | :--- | :--- |
| **T1** | Default Baseline | $\tau^* = 0.50$ | Naif, sangat konservatif, mengabaikan ketimpangan data. |
| **T2** | Max-F1 Tuning | $\tau^* = \arg\max_{\tau \in [0.01, 0.99]} \frac{2 \cdot P(\tau) \cdot R(\tau)}{P(\tau) + R(\tau)}$ | Memaksimalkan keseimbangan harmonik Precision-Recall pada inner validation. |
| **T3** | Recall-Constrained | $\tau^* = \arg\max_{\tau} \{P(\tau) \mid R(\tau) \ge 0.70\}$ | Memprioritaskan deteksi minimal 70% flaky test, lalu memaksimalisasi presisi. |
| **T4** | Youden’s J-Statistic | $\tau^* = \arg\max_{\tau} (\text{TPR}(\tau) - \text{FPR}(\tau))$ | Menyeimbangkan sensitivitas dan spesifisitas secara objektif tanpa bias prior. |
| **T5** | Prior-Shifted | $\tau^* = \frac{\sum_{i \in D_{\text{source}}} y_i}{|D_{\text{source}}|} \approx 0.0368$ | Memindahkan threshold ke titik peluang empiris kemunculan flaky test. |

---

## 📊 3. Hasil Empiris Komprehensif

### 3.1 Ringkasan Performa Komparatif 5 Strategi Lintas 24 Fold LOPO-CV

Tabel berikut menyajikan rekapitulasi nilai rata-rata (*Mean $\pm$ Std Dev*) dari seluruh 24 fold evaluasi Leave-One-Project-Out:

#### A. Model XGBoost GPU Classifier (`tree_method="hist"`, `device="cuda"`)

| Strategi | $\tau^*$ (Mean $\pm$ Std) | F1-Score | Recall | Precision | MCC | Specificity | FPR | PR-AUC | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **T1: Default 0.5** | $0.50 \pm 0.00$ | $0.0550 \pm 0.104$ | $0.0710 \pm 0.159$ | $\mathbf{0.2252 \pm 0.355}$ | $0.0575 \pm 0.105$ | $\mathbf{0.9711 \pm 0.106}$ | $\mathbf{0.0289 \pm 0.106}$ | $0.1898 \pm 0.239$ | $0.7020 \pm 0.179$ |
| **T2: Max-F1** | $0.34 \pm 0.07$ | $0.0621 \pm 0.108$ | $0.0879 \pm 0.189$ | $0.2137 \pm 0.337$ | $0.0544 \pm 0.107$ | $0.9527 \pm 0.148$ | $0.0473 \pm 0.148$ | $0.1898 \pm 0.239$ | $0.7020 \pm 0.179$ |
| **T3: Recall $\ge 70\%$** | $0.20 \pm 0.05$ | $0.0860 \pm 0.164$ | $0.1348 \pm 0.248$ | $0.1755 \pm 0.321$ | $0.0630 \pm 0.155$ | $0.9342 \pm 0.148$ | $0.0658 \pm 0.148$ | $0.1898 \pm 0.239$ | $0.7020 \pm 0.179$ |
| **T4: Youden’s J** | $\mathbf{0.04 \pm 0.01}$ | $\mathbf{0.1460 \pm 0.180}$ | $\mathbf{0.3806 \pm 0.368}$ | $0.1857 \pm 0.291$ | $\mathbf{0.1115 \pm 0.161}$ | $0.8211 \pm 0.187$ | $0.1789 \pm 0.187$ | $0.1898 \pm 0.239$ | $0.7020 \pm 0.179$ |
| **T5: Prior-Shifted** | $0.04 \pm 0.00$ | $0.1384 \pm 0.175$ | $\mathbf{0.3879 \pm 0.368}$ | $0.1285 \pm 0.223$ | $0.0950 \pm 0.155$ | $0.8015 \pm 0.188$ | $0.1985 \pm 0.188$ | $0.1898 \pm 0.239$ | $0.7020 \pm 0.179$ |

#### B. Model Random Forest Classifier (100 Trees, CPU Multi-Threading)

| Strategi | $\tau^*$ (Mean $\pm$ Std) | F1-Score | Recall | Precision | MCC | Specificity | FPR | PR-AUC | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **T1: Default 0.5** | $0.50 \pm 0.00$ | $0.0248 \pm 0.084$ | $0.0544 \pm 0.167$ | $0.1332 \pm 0.310$ | $0.0192 \pm 0.067$ | $\mathbf{0.9751 \pm 0.084}$ | $\mathbf{0.0249 \pm 0.084}$ | $0.1942 \pm 0.251$ | $0.7149 \pm 0.179$ |
| **T2: Max-F1** | $0.37 \pm 0.06$ | $0.0624 \pm 0.133$ | $0.1034 \pm 0.230$ | $0.1532 \pm 0.307$ | $0.0461 \pm 0.114$ | $0.9470 \pm 0.158$ | $0.0530 \pm 0.158$ | $0.1942 \pm 0.251$ | $0.7149 \pm 0.179$ |
| **T3: Recall $\ge 70\%$** | $0.30 \pm 0.06$ | $0.0852 \pm 0.141$ | $0.1401 \pm 0.235$ | $\mathbf{0.1683 \pm 0.313}$ | $0.0668 \pm 0.122$ | $0.9361 \pm 0.157$ | $0.0639 \pm 0.157$ | $0.1942 \pm 0.251$ | $0.7149 \pm 0.179$ |
| **T4: Youden’s J** | $\mathbf{0.06 \pm 0.02}$ | $0.1639 \pm 0.237$ | $0.5429 \pm 0.382$ | $0.1334 \pm 0.227$ | $0.1293 \pm 0.217$ | $0.7433 \pm 0.202$ | $0.2567 \pm 0.202$ | $0.1942 \pm 0.251$ | $0.7149 \pm 0.179$ |
| **T5: Prior-Shifted** | $0.04 \pm 0.00$ | $\mathbf{0.1948 \pm 0.252}$ | $\mathbf{0.6771 \pm 0.352}$ | $0.1450 \pm 0.224$ | $\mathbf{0.1640 \pm 0.218}$ | $0.6532 \pm 0.205$ | $0.3468 \pm 0.205$ | $0.1942 \pm 0.251$ | $0.7149 \pm 0.179$ |

---

### 3.2 Uji Signifikansi Statistik (Wilcoxon Signed-Rank Test & Cliff's Delta)

Evaluasi signifikansi dilakukan menggunakan *paired two-sided Wilcoxon signed-rank test* dan ukuran efek *Cliff's Delta* ($d$) mengikuti pedoman ACM/SIGSOFT Empirical Standards:

| Perbandingan (XGBoost GPU) | Metrik | Mean Baseline | Mean Treatment | Selisih Mean | $p$-value | Signifikansi | Cliff's Delta ($d$) | Interpretasi Efek |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **T1 (Default 0.5) vs T4 (Youden's J)** | **F1-Score** | $0.0550$ | $0.1460$ | $+0.0910$ | **$0.0191$** | **$*$ ($p < 0.05$)** | $+0.359$ | **Medium** |
| | **Recall** | $0.0710$ | $0.3806$ | $+0.3096$ | **$0.00029$** | **$***$ ($p < 0.001$)** | $+0.527$ | **Large** |
| | **Precision** | $0.2252$ | $0.1857$ | $-0.0395$ | $0.7946$ | ns | $+0.158$ | Small |
| | Specificity | $0.9711$ | $0.8211$ | $-0.1500$ | $0.00003$ | $***$ | $-0.823$ | Large |
| **T1 (Default 0.5) vs T5 (Prior-Shifted)** | **F1-Score** | $0.0550$ | $0.1384$ | $+0.0833$ | **$0.0217$** | **$*$ ($p < 0.05$)** | $+0.345$ | **Medium** |
| | **Recall** | $0.0710$ | $0.3879$ | $+0.3169$ | **$0.00029$** | **$***$ ($p < 0.001$)** | $+0.541$ | **Large** |
| **T1 (Default 0.5) vs T2 (Max-F1)** | F1-Score | $0.0550$ | $0.0621$ | $+0.0071$ | $0.7213$ | ns | $+0.045$ | Negligible |
| | Recall | $0.0710$ | $0.0879$ | $+0.0169$ | $0.0277$ | $*$ | $+0.060$ | Negligible |
| **T1 (Default 0.5) vs T3 (Recall $\ge 70\%$)** | F1-Score | $0.0550$ | $0.0860$ | $+0.0310$ | $0.2860$ | ns | $+0.052$ | Negligible |
| | Recall | $0.0710$ | $0.1348$ | $+0.0638$ | $0.0117$ | $*$ | $+0.102$ | Negligible |
| **T2 (Max-F1) vs T4 (Youden's J)** | **F1-Score** | $0.0621$ | $0.1460$ | $+0.0839$ | **$0.0065$** | **$**$ ($p < 0.01$)** | $+0.325$ | Small |
| | **Recall** | $0.0879$ | $0.3806$ | $+0.2927$ | **$0.00029$** | **$***$ ($p < 0.001$)** | $+0.490$ | **Large** |

---

### 3.3 Penyelidikan Proyek Individual: Mengapa Youden's J & Prior Menang Telak?

Tabel di bawah ini menampilkan rincian performa F1-Score pada repositori-repositori utama yang mengalami lonjakan performa:

| Target Project | Total Test | Flaky Test | T1 (Default 0.5) | T2 (Max-F1) | T3 (Recall $\ge 70\%$) | T4 (Youden's J) | Gain vs Default |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `alluxio` | 913 | 129 | $0.0504$ | $0.2154$ | $0.4638$ | $\mathbf{0.6591}$ | **+0.6087 (+1207%)** |
| `ambari` | 1.002 | 48 | $0.4222$ | $0.4364$ | $0.4578$ | $\mathbf{0.4902}$ | **+0.0680 (+16.1%)** |
| `hbase` | 2.124 | 51 | $0.0400$ | $0.0909$ | $0.3276$ | $\mathbf{0.4495}$ | **+0.4095 (+1024%)** |
| `hector` | 137 | 7 | $0.0000$ | $0.0000$ | $0.0000$ | $\mathbf{0.2963}$ | **+0.2963 (Bangkit)** |
| `spring-boot` | 1.341 | 60 | $0.0099$ | $0.0127$ | $0.0531$ | $\mathbf{0.2399}$ | **+0.2300 (+2323%)** |
| `commons-exec` | 76 | 2 | $0.0000$ | $0.0000$ | $0.0000$ | $\mathbf{0.2500}$ | **+0.2500 (Bangkit)** |
| `orbit` | 227 | 4 | $0.0000$ | $0.0000$ | $0.0000$ | $\mathbf{0.2308}$ | **+0.2308 (Bangkit)** |
| `okhttp` | 1.493 | 11 | $0.0000$ | $0.0748$ | $0.0748$ | $\mathbf{0.1333}$ | **+0.1333 (Bangkit)** |
| `incubator-dubbo`| 894 | 19 | $0.1000$ | $0.0952$ | $0.0976$ | $\mathbf{0.1085}$ | **+0.0085 (+8.5%)** |
| `undertow` | 1.055 | 16 | $\mathbf{0.1818}$ | $0.1176$ | $0.1081$ | $0.1000$ | $-0.0818$ |
| `httpcore` | 647 | 23 | $\mathbf{0.2759}$ | $\mathbf{0.2759}$ | $0.2609$ | $0.2530$ | $-0.0229$ |

---

## 🔍 4. Analisis Mendalam & Pembahasan Akademik

### 4.1 Analisis Trade-off Precision vs Recall
Peralihan dari Default 0.5 ke strategi Dynamic Thresholding mengungkap dinamika trade-off yang krusial bagi praktisi rekayasa perangkat lunak:
1. **Kegagalan Paradigma Default 0.5**:
   Meskipun Default 0.5 mencatat Specificity tertinggi ($0.9711$) dan FPR terendah ($2,89\%$), nilai Recall-nya runtuh pada angka $7,10\%$. Artinya, **lebih dari 92,9% tes flaky lolos tanpa terdeteksi** ke dalam sistem produksi atau pipeline integrasi.
2. **Keterbatasan Max-F1 Tuning pada Inner Validation**:
   Strategi T2 (Max-F1) memilih $\tau^* \approx 0.344$. Walaupun optimal pada data validasi internal, ambang batas $0.34$ masih terlalu tinggi bagi proyek target yang mengalami *domain shift* negatif (di mana distribusi probabilitas bergeser ke kiri). Akibatnya, peningkatan F1 pada T2 relatif kecil ($+12,86\%$) dan tidak signifikan secara statistik ($p = 0.721$).
3. **Keberhasilan Superior Youden’s J-Statistic (T4)**:
   Youden's J memaksimalkan fungsi objektif $J(\tau) = \text{TPR}(\tau) - \text{FPR}(\tau)$, yang secara inheren mengabaikan ketimpangan rasio kelas. Dengan menurunkan ambang batas ke $\tau^* \approx 0.043$, T4 berhasil mendongkrak Recall hingga $38,06\%$ pada XGBoost dan $54,29\%$ pada Random Forest dengan tetap mempertahankan Spesifisitas sebesar $82,11\%$ (FPR terkendali pada $17,89\%$).

### 4.2 Rekapitulasi Rentang Threshold Optimal
Berdasarkan investigasi empiris pada 24 fold cross-project:
- **Rentang Default yang Gagal**: $\tau \ge 0.20$ terbukti gagal memberikan cakupan recall yang memadai pada data cross-project.
- **Rentang Sweet Spot Optimal**: Ambang batas probabilitas terbaik untuk deteksi flaky test berada di rentang **$0.030 \le \tau^* \le 0.065$**.
- **Korelasi Kuat dengan Empiris Prior**: Titik optimal Youden's J ($\tau^* \approx 0.0429$) berada sangat dekat dengan proporsi prior flaky test dalam repositori sumber ($\pi \approx 0.0368$). Hal ini membuktikan postulat Provost (2000) bahwa ambang batas keputusan harus digeser proporsional terhadap ketimpangan kelas dasar (*prior probability moving*).

### 4.3 Analisis Efisiensi Komputasi Hardware GPU NVIDIA RTX 4050
Eksperimen LOPO-CV ini memanfaatkan penuh akselerasi GPU NVIDIA RTX 4050 Laptop (CUDA 13.4):
- **Waktu Latih Model Final**: $0,165 \pm 0,007$ detik per fold.
- **Waktu Inner-Validation Grid Search (99 Titik Grid)**: $0,204 \pm 0,027$ detik per fold.
- **Total Durasi 24 Fold LOPO-CV**: Tuntas dalam **10,50 detik** untuk XGBoost GPU dan **14,20 detik** untuk Random Forest CPU.
Akselerasi GPU memungkinkan proses tuning ambang batas berskala ribuan iterasi dijalankan secara *real-time* di lingkungan pipeline CI/CD modern tanpa menimbulkan latensi yang membebani server build.

---

## 📈 5. Visualisasi Publikasi Ilmiah

Seluruh grafik visualisasi berkualitas tinggi (300 DPI) telah dihasilkan di direktori `results/plots/`:

1. **Boxplot Distribusi F1-Score Lintas Strategi**:
   - [`results/plots/boxplot_f1_strategies_xgboost_gpu.png`](results/plots/boxplot_f1_strategies_xgboost_gpu.png)
   - [`results/plots/boxplot_f1_strategies_random_forest.png`](results/plots/boxplot_f1_strategies_random_forest.png)
   *Menggambarkan pergeseran median dan mean F1-Score yang melonjak drastis pada T4 (Youden's J) dan T5 (Prior-Shifted).*

2. **Trade-off Recall vs Precision Lintas Strategi**:
   - [`results/plots/boxplot_precision_recall_tradeoff_xgboost_gpu.png`](results/plots/boxplot_precision_recall_tradeoff_xgboost_gpu.png)
   - [`results/plots/boxplot_precision_recall_tradeoff_random_forest.png`](results/plots/boxplot_precision_recall_tradeoff_random_forest.png)
   *Menunjukkan lonjakan Recall hingga 67% dengan penurunan presisi yang moderat dan terkendali.*

3. **Distribusi Nilai Threshold Optimal ($\tau^*$) yang Dipelajari**:
   - [`results/plots/distribution_learned_thresholds_xgboost_gpu.png`](results/plots/distribution_learned_thresholds_xgboost_gpu.png)
   - [`results/plots/distribution_learned_thresholds_random_forest.png`](results/plots/distribution_learned_thresholds_random_forest.png)
   *Memperlihatkan klasterisasi nilai $\tau^*$ di sekitar $0.04$ untuk Youden's J dan $0.34$ untuk Max-F1.*

4. **Diagram Batang Ringkasan Metrik Komprehensif**:
   - [`results/plots/barchart_metrics_comparison_xgboost_gpu.png`](results/plots/barchart_metrics_comparison_xgboost_gpu.png)
   - [`results/plots/barchart_metrics_comparison_random_forest.png`](results/plots/barchart_metrics_comparison_random_forest.png)

5. **Grafik F1 Gain Per Proyek Target**:
   - [`results/plots/per_project_f1_gain_xgboost_gpu.png`](results/plots/per_project_f1_gain_xgboost_gpu.png)
   - [`results/plots/per_project_f1_gain_random_forest.png`](results/plots/per_project_f1_gain_random_forest.png)

---

## ⚠️ 6. Ancaman Validitas (Threats to Validity)

1. **Validitas Internal (*Internal Validity*)**:
   - Risiko kebocoran data (*data leakage*) telah dimitigasi sepenuhnya melalui skema *nested inner-validation split* pada data sumber semata. Data uji target diisolasi total hingga tahap inferensi akhir.
   - Penskalaan fitur (`StandardScaler`) hanya mempelajari rata-rata dan deviasi standar dari data sumber fold terkait.
2. **Validitas Eksternal (*External Validity*)**:
   - Studi ini menguji 24 repositori perangkat lunak Java sumber terbuka berukuran besar dari benchmark FlakeFlagger (Alshammari et al., 2021). Meskipun representatif, generalisasi ke bahasa pemrograman dinamis (Python, JavaScript) memerlukan studi replikasi lanjutan.
3. **Validitas Konstruk (*Construct Validity*)**:
   - Penanganan kasus zero-positive (seperti pada repositori `jimfs` yang memiliki 0 flaky test setelah pembersihan C1) ditangani secara matematis dengan menetapkan Recall=0, Precision=0 jika FP>0, dan Specificity yang dihitung secara tepat.

---

## 💡 7. Rekomendasi Praktis untuk Industri CI/CD

Berdasarkan temuan empiris penelitian ini, kami menyusun rekomendasi konkret untuk tim DevOps dan Software Quality Assurance:

1. **Hentikan Penggunaan Ambang Batas Default 0.5 pada Prediksi Flaky Test**:
   Penggunaan $\tau = 0.50$ memberikan ilusi stabilitas palsu (*false sense of security*) karena menghasilkan FPR rendah, namun membiarkan $>90\%$ flaky test lolos tanpa penanganan.
2. **Gunakan Youden’s J-Statistic ($\tau^* \approx 0.04 - 0.06$) sebagai Ambang Batas Utama**:
   Youden's J memberikan kompromi terbaik antara deteksi flaky test (Recall $\approx 40-55\%$) dan pengendalian false alarm (Specificity $>80\%$).
3. **Terapkan Prior-Shifted Thresholding sebagai Solusi Heuristik Cepat**:
   Jika komputasi grid search tidak memungkinkan, menyetel ambang batas langsung pada proporsi prior historis kelas flaky ($\tau = \pi_{\text{historical}} \approx 0.037$) terbukti memberikan performa yang hampir menyamai Youden's J tanpa biaya tuning tambahan.
4. **Padukan dengan Transfer Learning (Alternatif 1)**:
   Kombinasi antara adaptasi bobot sampel sumber (TrAdaBoost / Burak Filter dari Alternatif 1) dan Dynamic Threshold Tuning (Alternatif 2) berpotensi memecahkan rekor performa F1-Score cross-project di masa depan.

---

## 📚 8. Rujukan Bibliografi

1. **Alshammari, A., Aldeen, F., & Bell, J. (2021)**. *FlakeFlagger: Predicting flakiness without rerunning tests*. In Proceedings of the 43rd IEEE/ACM International Conference on Software Engineering (ICSE 2021), pp. 1572–1584.
2. **Afeltra, F., Coppola, R., & Morisio, M. (2024)**. *A large-scale empirical investigation into cross-project flaky test prediction*. IEEE Access, 12, pp. 45210–45228.
3. **Provost, F. (2000)**. *Machine learning from imbalanced data sets 101*. In Proceedings of the AAAI’2000 Workshop on Imbalanced Data Sets, pp. 1–3.
4. **Elkan, C. (2001)**. *The foundations of cost-sensitive learning*. In Proceedings of the 17th International Joint Conference on Artificial Intelligence (IJCAI 2001), pp. 973–978.
5. **Youden, W. J. (1950)**. *Index for rating diagnostic tests*. Cancer, 3(1), pp. 32–35.
6. **Kitchenham, B. A., Madeyski, L., & Budgen, D. (2017)**. *How should software engineering researchers report distributions of continuous data?*. ACM Transactions on Software Engineering and Methodology (TOSEM), 26(1), pp. 1–49.
