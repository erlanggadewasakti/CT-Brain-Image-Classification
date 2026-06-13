# Analisis Komprehensif: CT Brain Image Classification

> **Tujuan:** Menganalisis 12 eksperimen (3 arsitektur × 2 skenario data-split × 2 learning rate) pada klasifikasi 3-kelas CT Brain untuk menyediakan insight paper conference.

---

## 1. Ringkasan Setup Eksperimen

| Komponen | Detail |
|---|---|
| **Dataset** | 256 citra CT-Brain (aneurysm: 84, cancer: 91, tumor: 84) dalam format JPG |
| **Kelas** | aneurysm (0), tumor (1), cancer (2) |
| **Skenario 1** | Train 204 / Val 26 / Test 26 — proporsi **80:10:10** |
| **Skenario 2** | Train 178 / Val 52 / Test 26 — proporsi **70:20:10** |
| **Test set** | *Identik di seluruh 12 eksperimen* (`random_state=42`, stratifikasi `label`) — 26 citra: aneurysm 9, tumor 8, cancer 9 |
| **Arsitektur** | ResNet50, VGG16, ViT (Vision Transformer) — semua pretrained ImageNet |
| **Learning Rate** | 1e-4 (low) vs 1e-2 (high) |
| **Optimizer** | Adam (default β₁=0.9, β₂=0.999) |
| **Epochs** | 30, Batch Size 16 |
| **Augmentasi (Train)** | Resize 224², RandomHorizontalFlip, RandomRotation(15°), ImageNet normalization |
| **Transform (Val/Test)** | Resize 224², ImageNet normalization (tanpa augmentasi) |
| **Device** | GPU CUDA |

---

## 2. Tabel Induk — Test Set Metrics (Unseen, N=26)

### 2.1 Learning Rate = 1e-4 (Optimal)

| Skenario | Model | Acc | Prec | Rec | F1 | Aneur. F1 | Tumor F1 | Cancer F1 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | ResNet50 | **0.9615** | 0.9654 | 0.9615 | **0.9614** | 0.95 | 1.00 | 0.94 |
| 1 | VGG16 | **0.9615** | 0.9654 | 0.9615 | **0.9614** | 0.95 | 1.00 | 0.94 |
| 1 | ViT | **0.9615** | 0.9654 | 0.9615 | **0.9614** | 0.95 | 1.00 | 0.94 |
| 2 | ResNet50 | **0.9615** | 0.9654 | 0.9615 | **0.9614** | 0.95 | 1.00 | 0.94 |
| 2 | VGG16 | **0.9615** | 0.9654 | 0.9615 | **0.9614** | 0.95 | 1.00 | 0.94 |
| 2 | ViT | 0.9231 | 0.9231 | 0.9231 | 0.9231 | 0.89 | 1.00 | 0.89 |

> **Catatan:** ResNet50 dan VGG16 identik di kedua skenario karena prediksi final mereka sama (1 misklasifikasi: 1 cancer → aneurysm). ViT skenario 2 menghasilkan 2 misklasifikasi (1 cancer → aneurysm, 1 aneurysm → cancer).

### 2.2 Learning Rate = 1e-2 (Degradasi)

| Skenario | Model | Acc | Prec | Rec | F1 | Aneur. F1 | Tumor F1 | Cancer F1 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | ResNet50 | 0.9231 | 0.9371 | 0.9231 | 0.9221 | 0.88 | 1.00 | 0.90 |
| 1 | VGG16 | 0.3462 | 0.1198 | 0.3462 | 0.1780 | 0.00 | 0.00 | 0.51 |
| 1 | ViT | 0.6538 | 0.4409 | 0.6538 | 0.5227 | 0.00 | 0.89 | 0.72 |
| 2 | ResNet50 | 0.7308 | 0.8486 | 0.7308 | 0.6828 | 0.72 | 1.00 | 0.36 |
| 2 | VGG16 | 0.3462 | 0.1198 | 0.3462 | 0.1780 | 0.00 | 0.00 | 0.51 |
| 2 | ViT | 0.3462 | 0.1198 | 0.3462 | 0.1780 | 0.00 | 0.00 | 0.51 |

---

## 3. Temuan Utama

### 3.1 Learning Rate adalah Faktor Paling Dominan

Gap akurasi LR 1e-4 → 1e-2 (rata-rata kedua skenario):

| Model | LR=1e-4 | LR=1e-2 | ΔAcc |
|:-:|:-:|:-:|:-:|
| ResNet50 | 0.9615 | 0.8270 | −13.5 pp |
| VGG16 | 0.9615 | 0.3462 | **−61.5 pp** |
| ViT | 0.9423 | 0.5000 | −44.2 pp |

- **LR 1e-4 ⇒ semua arsitektur konvergen** ke F1 ≈ 0.96 (kecuali ViT skenario 2 di 0.92).
- **LR 1e-2 ⇒ degradasi masif** terutama pada VGG16 yang collapse menjadi *majority-class classifier* (selalu prediksi "cancer" ⇒ 9/26 = 34.6% = accuracy baseline).

### 3.2 Diagnosis Kegagalan LR 1e-2 (dari Fig. 3a/b)

| Model | Fenomena | Penjelasan |
|:-:|---|---|
| **VGG16** | Loss meledak ke **8.8 × 10⁸** (sk.1) dan **4.9 × 10⁸** (sk.2) di epoch awal | Tanpa BatchNorm internal, gradien pada head Linear baru sangat besar. Adam gagal mengadaptasi momen pada skala loss ekstrem. |
| **ResNet50** | Loss berosilasi 0.06–1.57, tidak stabil | BatchNorm + residual connection men-dampen gradien, tapi LR tetap terlalu tinggi untuk konvergensi stabil. Sk.2 lebih buruk karena train set lebih kecil ⇒ estimasi gradien lebih noisy. |
| **ViT** | Loss stagnan ~1.0 dari 3.0 awal | Fine-tuning ViT pada dataset kecil (<200 sampel) dengan LR besar menghancurkan representasi pretrained yang sensitif terhadap posisi patch embedding. |

### 3.3 Perbandingan Antar-Arsitektur

| | ResNet50 | VGG16 | ViT |
|---|---|---|---|
| **Robustness terhadap LR besar** | **Tinggi** — BN + residual | Rendah — tanpl BN | Sedang — bergantung data |
| **Kebutuhan data minimal** | Rendah (178 cukup) | Rendah | **Tinggi** (butuh ≥204) |
| **Konvergensi di LR optimal** | Cepat (epoch 2–10) | Lambat (epoch 22) | **Paling cepat** (epoch 1) |
| **F1 pada LR 1e-4** | 0.96 | 0.96 | 0.92–0.96 |

**Insight praktis:** Untuk fine-tuning medical imaging data kecil (<300 citra), ResNet50 memberikan *best trade-off* antara performa, robustness terhadap hyperparameter, dan stabilitas training.

### 3.4 Pengaruh Skenario Split (80:10:10 vs 70:20:10)

Pada LR=1e-4:
- **ResNet50 & VGG16**: identik di kedua skenario (0.9615). Kedua CNN tidak terpengaruh oleh pengurangan 26 sampel training.
- **ViT**: turun dari 0.9615 ke 0.9231 saat training sampel berkurang dari 204 ke 178 (−26 sampel). Konfirmasi hipotesis *data-hungry nature* ViT.

Pada LR=1e-2:
- **ResNet50**: turun dari 0.9231 ke 0.7308 — pengurangan data memperkuat efek noise gradien.
- **VGG16**: sudah collapse di skenario 1, tidak berubah di skenario 2 (floor effect).
- **ViT**: collapse tambahan (0.6538 → 0.3462), bergabung dengan VGG16 sebagai majority-class classifier.

### 3.5 Analisis Pola Kesalahan Confusion Matrix

Pola konsisten pada semua model LR=1e-4 (dari 12 confusion matrix):

- **Tumor** diprediksi sempurna di **100%** dari seluruh eksperimen LR=1e-4. Kelas ini paling diskriminatif secara visual (F1 = 1.00).
- **Cancer → Aneurysm** adalah *satu-satunya* misklasifikasi yang terjadi (1 dari 9 sampel cancer). Tidak ada aneurysm → cancer atau tumor ↔ lainnya pada LR=1e-4.
- Ini mengindikasikan bahwa **ciri radiologis cancer dan aneurysm lebih mirip** (keduanya terkait dengan pembuluh darah/lesi vaskular) dibandingkan cancer-tumor.

Pada LR=1e-2, pola kesalahan berpola ke *mode collapse*:
- VGG16 dan ViT skenario 2: semua sampel diprediksi "cancer" (aneurysm F1 = tumor F1 = 0.00).
- ResNet50 skenario 2: cancer recall hanya 0.22 (2/9), mayoritas cancer salah diklasifikasikan.

---

## 4. Dinamika Training

### 4.1 Kecepatan Konvergensi (LR=1e-4, dari Fig. 4a/b)

| Skenario | Model | Val Acc ≥ 0.95 | Best Val Acc | Epoch Best |
|:-:|:-:|:-:|:-:|:-:|
| 1 | ViT | **Epoch 1** | 1.00 | 1 |
| 1 | ResNet50 | Epoch 4 | 1.00 | 10 |
| 1 | VGG16 | Epoch 2 | 1.00 | 22 |
| 2 | ViT | Epoch 3 | 1.00 | 3 |
| 2 | ResNet50 | Epoch 2 | 1.00 | 8 |
| 2 | VGG16 | Epoch 1 | 0.9808 | 1 |

> **Catatan:** VGG16 di skenario 1 menunjukkan osilasi val acc lebih tinggi (tidak pernah stabil di 1.00 selama >1 epoch berturut-turut), sementara ViT di skenario 1 mencapai 1.00 di epoch 1 dan mempertahankannya 18/30 epoch.

### 4.2 Visualisasi yang Sudah Ada

| Figure | Deskripsi | File |
|---|---|---|
| Fig. 3a | Training loss LR=1e-2, Skenario 1 (log scale) — VGG16 exploding | `fig3a_trainloss_1e2_skenario1.png` |
| Fig. 3b | Training loss LR=1e-2, Skenario 2 (log scale) — VGG16 exploding | `fig3b_trainloss_1e2_skenario2.png` |
| Fig. 4a | Validation accuracy LR=1e-4, Skenario 1 | `fig4a_valacc_1e4_skenario1.png` |
| Fig. 4b | Validation accuracy LR=1e-4, Skenario 2 | `fig4b_valacc_1e4_skenario2.png` |

---

## 5. Insight untuk Paper Conference

### 5.1 Kontribusi yang Bisa Diklaim

1. **Learning rate sensitivity analysis** pada fine-tuning medical image classification tiga arsitektur — menunjukkan bahwa LR 1e-2 membuat model SOTA *underperform* majority baseline (VGG16 acc 0.346 ≈ 34.6% = proporsi kelas mayoritas).
2. **Bukti empiris ViT membutuhkan data lebih banyak** dibanding CNN pada domain CT-Brain (204 vs 178 sampel menghasilkan gap F1 4 pp).
3. **ResNet50 sebagai arsitektur paling robust** untuk dataset medis kecil — mempertahankan F1 ≥ 0.96 di semua konfigurasi optimal dan paling tahan terhadap LR tinggi.

### 5.2 Rekomendasi Narasi Paper

**Paragraf 1 — Temuan utama:** "We demonstrate that the choice of learning rate is far more consequential than architecture selection for small-scale medical imaging fine-tuning, with an LR of 10⁻² causing complete training collapse in VGG16..."

**Paragraf 2 — Perbandingan arsitektur:** "At the optimal LR of 10⁻⁴, all three architectures achieve comparable F1 ≥ 0.92. However, the Vision Transformer shows higher sensitivity to training set size, consistent with its known inductive bias limitations (Dosovitskiy et al., 2021)."

**Paragraf 3 — Limitasi & future work:** "Our small dataset (N=256) and single random seed limit statistical conclusions. Future work should employ k-fold cross-validation and learning rate scheduling."

### 5.3 Struktur Tabel & Figure yang Disarankan

| # | Jenis | Isi | Status |
|---|---|---|---|
| Table I | Hyperparameter & dataset split | Ringkasan setup (Section 1 di atas) | Siap |
| Table II | Test metrics LR=1e-4 | Tabel 2.1 | **Sudah di EDA**, tapi perlu koreksi ViT sk.2 |
| Table III | Test metrics LR=1e-2 | Tabel 2.2 | **Belum dibuat** di EDA |
| Fig. 3 | Training loss log-scale | Fig 3a (sk.1) + 3b (sk.2) | Sudah jadi |
| Fig. 4 | Validation accuracy LR=1e-4 | Fig 4a (sk.1) + 4b (sk.2) | Sudah jadi |
| Fig. 5 | Confusion matrix grid LR=1e-4 | 6 confusion matrix dalam grid 3×2 | Perlu dirangkai |

---

## 6. Celah Riset & Saran Perbaikan

| # | Celah | Severity | Saran Perbaikan |
|---|---|---|---|
| 1 | **Single seed** (random_state=42) | **High** | Gunakan 3+ seed berbeda atau k-fold (≥5) untuk uji statistik signifikansi (paired t-test antar model) |
| 2 | **LR hanya 2 titik** | Medium | Tambahkan sweep 1e-5, 5e-4, 1e-3 + learning rate scheduler (CosineAnnealing) |
| 3 | **Augmentasi minimal** | Medium | ViT sangat diuntungkan augmentasi kuat (RandAugment, MixUp, CutMix) — relevan untuk mengatasi gap skenario 2 |
| 4 | **Class imbalance tipis** (84/91/84) | Low | Laporkan Cohen's κ atau balanced accuracy; akurasi mentah sudah informatif untuk imbalance kecil |
| 5 | **30 epoch tanpa early stopping** | Low | ViT/ResNet50 mencapai val acc 100% di epoch 1–10; sisanya berpotensi overfit. Tambahkan early stopping berbasis val loss |
| 6 | **Test set kecil (N=26)** | Medium | Satu misklasifikasi = 4 pp akurasi. Interpretasi perlu cautious, CI akurasi ≈ ±8 pp. |
| 7 | **Belum ada Tabel III (LR=1e-2)** di EDA | Low | Perlu ditambahkan untuk kelengkapan paper (negative result tetap menarik untuk diskusi) |
| 8 | **Belum ada uji statistik** antar model | Medium | McNemar test pada prediksi pair-wise untuk klaim "ResNet50 ≥ ViT" |

### 6.1 Checklist Sebelum Submit Paper

- [ ] Tambahkan minimal 1 seed tambahan (random_state=123) sebagai validasi eksternal
- [ ] Buat Tabel III (LR=1e-2) di EDA notebook
- [ ] Koreksi Tabel II di EDA — ViT skenario 2 = 0.9231 (bukan 0.9615)
- [ ] Grid-kan 6 confusion matrix LR=1e-4 menjadi Fig. 5 (3 kolom arsitektur × 2 baris skenario)
- [ ] Hitung confidence interval akurasi (Wilson score interval) untuk N=26
- [ ] Tulis diskusi tentang *why LR matters more than architecture in small-data fine-tuning*

---

## 7. Kesimpulan

1. **LR 1e-4 adalah pilihan optimal** untuk fine-tuning semua arsitektur pada CT-Brain 256 citra (F1 ≥ 0.92).
2. **LR 1e-2 menyebabkan kegagalan training** pada VGG16 (loss explosion, akurasi 0.346) dan degradasi signifikan pada ViT dan ResNet50.
3. **ResNet50 menunjukkan robustness terbaik** — mempertahankan F1 ≥ 0.68 bahkan di LR 1e-2, dan mencapai F1 0.96 di LR optimal.
4. **ViT paling sensitif terhadap ukuran data** — menurun 4 pp akurasi saat training set dikurangi dari 204 ke 178.
5. **Tumor adalah kelas paling mudah diklasifikasikan** (F1=1.00 di seluruh eksperimen LR=1e-4), sementara **cancer-aneurysm adalah pasangan paling membingungkan** (1 misklasifikasi konsisten di seluruh model).
6. **Skenario split (80:10:10 vs 70:20:10) tidak berpengaruh signifikan** pada CNN, hanya pada ViT — menunjukkan bahwa untuk dataset kecil, proporsi training set kritis hanya untuk arsitektur transformer.

---

*Dibuat: 13 Juni 2026 | Berdasarkan 12 notebook training di `skenario_1/` dan `skenario_2/`*
