# El Geometrisi ile Biyometrik Kimlik Tanıma

Mediapipe tabanlı el iskelet çıkarımı, geometrik özellik mühendisliği ve makine öğrenmesi / derin öğrenme modelleriyle kişi kimliği doğrulama sistemi. Geleneksel ML yöntemleri (SVM, MLP) ve son teknoloji derin öğrenme mimarileri (MBA-Net, ABD-Net, RGA-Net) karşılaştırmalı olarak değerlendirilmiştir.

---

## Proje Özeti

Bu proje, el görüntülerinden kişiyi tanımayı iki farklı yaklaşımla ele almaktadır:

**Geleneksel ML Yaklaşımı:** MediaPipe ile 21 iskelet noktası çıkarılır, 75 geometrik özellik hesaplanır ve SVM / MLP ile sınıflandırma yapılır.

**Derin Öğrenme Yaklaşımı:** Ham görüntüler ImageNet ön eğitimli ResNet-50 omurgası üzerine inşa edilmiş dikkat mekanizmalı mimarilerle (MBA-Net, ABD-Net, RGA-Net) doğrudan işlenir.

---

## Veri Seti

**11k Hands Dataset** — [Kaggle](https://www.kaggle.com/datasets/shyambhu/hands-and-palm-images-dataset)

| Özellik | Değer |
|---------|-------|
| Toplam görüntü | 11.076 |
| Kişi sayısı | 190 |
| Başarılı çıkarım | 10.765 (%97.2) |
| Görünüm türleri | Dorsal (el sırtı) + Palmar (avuç içi) |
| Metadata | Yaş, cinsiyet, ten rengi, aksesuar, tırnak durumu |

---

## Geleneksel ML Yaklaşımı

### 1. İskelet Çıkarımı (MediaPipe)

MediaPipe Hand Landmarker her görüntüden **21 anahtar nokta** tespit eder.

```
         8   12  16  20      ← Parmak uçları
         7   11  15  19
    4    6   10  14  18
    3    5    9  13  17
    2         \ | /
    1           0            ← Bilek (WRIST)
```

### 2. Normalizasyon

Ölçek ve konum bağımsızlığı için:
1. **Öteleme:** Tüm noktalardan WRIST çıkarılır
2. **Ölçekleme:** Tüm noktalar WRIST→ORTA_MCP mesafesine bölünür

### 3. Özellik Mühendisliği (75 özellik)

| Grup | Açıklama | Sayı |
|------|----------|------|
| Parmak uzunlukları | Her parmak için segment toplamı | 5 |
| Parmak ucu mesafeleri | 5 ucun tüm ikili kombinasyonları | 10 |
| Avuç genişlikleri | MCP eklemleri arası mesafeler | 4 |
| Uzunluk/genişlik oranları | Parmak uzunluğu ÷ avuç genişliği | 5 |
| PIP eklem açıları | Orta eklemlerdeki bükülme açısı | 4 |
| Uç→bilek mesafeleri | Her parmak ucunun bilek uzaklığı | 5 |
| **Ham landmark (x,y)** | 21 normalize noktanın koordinatları | **42** |
| **Toplam** | | **75** |

### 4. Modeller ve Hiperparametre Optimizasyonu

- **SVM** — RBF çekirdeği, GridSearchCV ile optimizasyon (C×gamma, cv=3)
- **MLP** — 3 katmanlı ağ (256→128→64), Adam, erken durdurma
- **Ön işleme** — Yalnızca eğitim setine fit edilen StandardScaler

### 5. Biyometrik Değerlendirme

Centroid tabanlı doğrulama senaryosu:
- Her sınıfın eğitim centroid'i hesaplanır
- **Genuine skoru** = prob'un kendi sınıf centroid'ine uzaklığı
- **Impostor skoru** = prob'un diğer sınıf centroid'lerine uzaklığı
- 500 eşik değeri taranarak FAR/FRR eğrisi çizilir
- **EER** eşik bağımsız özet metrik olarak raporlanır

### Geleneksel ML Sonuçları

| Deney | Özellik | Accuracy | Macro F1 | EER |
|-------|---------|----------|----------|-----|
| Baseline SVM | 33 | %78.85 | %74.00 | %36.93 |
| Baseline MLP | 33 | %75.94 | %71.20 | %37.07 |
| SVM_tum | 75 | %91.89 | %89.67 | %36.56 |
| MLP_tum | 75 | %86.78 | %83.32 | %36.56 |
| SVM_dorsal | 75 | %91.58 | %88.53 | %28.52 |
| MLP_dorsal | 75 | %83.91 | %78.78 | %28.52 |
| **SVM_palmar** | **75** | **%93.03** | **%90.29** | **%27.49** |
| MLP_palmar | 75 | %88.24 | %85.01 | %27.49 |

**En iyi geleneksel model: SVM_palmar — %93.03 Accuracy, %27.49 EER**

| Geliştirme | Etki |
|-----------|------|
| Ham landmark koordinatları (33→75 özellik) | Accuracy **+9–10 puan** |
| GridSearchCV (C=10→C=100) | Accuracy **+3.5 puan** |
| Dorsal/Palmar ayrımı | EER **%37→%27** (−10 puan) |

---

## Derin Öğrenme Yaklaşımı

Ham görüntüler (224×224) ImageNet ön eğitimli **ResNet-50** omurgası üzerine inşa edilmiş üç farklı mimariyle işlenir. Eğitim: CrossEntropy + Triplet Loss, Adam optimizer, 30 epoch, T4 GPU (Google Colab).

### MBA-Net — Multi-Branch Attention Network

SE (Squeeze-and-Excitation) dikkat bloğu + global branch (GAP) + local branch (4 yatay şerit). Global ve local özellikler birleştirilip L2-normalize edilir.

### ABD-Net — Attentive but Diverse Network

CBAM (Channel + Spatial Attention) branch ve dikkat-içermeyen backbone branch. Diversity loss iki dalı birbirinden farklı özellikler öğrenmeye zorlar.

### RGA-Net — Relation-Guided Attention Network

Uzamsal öz-ilişki matrisi ile özellik haritaları üzerinde ilişki-rehberli dikkat. RGA modülü ResNet-50'nin layer3 ve layer4 çıktılarına uygulanır.

### Derin Öğrenme Sonuçları (Palmar, 30 Epoch)

| Model | Parametre | Rank-1 | Accuracy | EER |
|-------|-----------|--------|----------|-----|
| MBA-Net | 26.3M | %99.81 | %96.89 | %0.12 |
| **ABD-Net** | **26.2M** | **%99.88** | **%99.69** | **%0.01** |
| RGA-Net | 35.1M | — | %99.19* | — |

*Epoch 21'de ölçülmüş değer, eğitim kesintiye uğramıştır.

---

## Tüm Modeller Karşılaştırması

| Model | Tür | Rank-1 / Accuracy | EER |
|-------|-----|-------------------|-----|
| SVM_palmar | Geleneksel ML | %93.03 | %27.49 |
| MLP_palmar | Geleneksel ML | %88.24 | %27.49 |
| MBA-Net | Derin Öğrenme | %99.81 | %0.12 |
| **ABD-Net** | **Derin Öğrenme** | **%99.88** | **%0.01** |
| RGA-Net | Derin Öğrenme | ~%99+ | — |

Derin öğrenme mimarileri geleneksel ML'yi hem doğruluk hem de EER açısından belirgin biçimde geçmektedir.

---

## Proje Yapısı

```
📁 Yerel Pipeline (sıralı çalıştırılır)
├── 01_veri_indir.py                          # kagglehub ile dataset indirme
├── 02_mediapipe_ozellik_cikar.py             # MediaPipe landmark çıkarımı + 75 özellik
├── 03_geleneksel_ml_egit.py                  # SVM/MLP/KNN/RF eğitim, FAR/FRR/EER, grafikler
├── 04_landmark_gorsellestir.py               # Landmark anotasyon görselleştirme
│
📓 Colab Notebooks
├── colab_odev_mediapipe_svm_mlp.ipynb        # Ödev: MediaPipe + SVM/MLP (uçtan uca)
├── colab_derin_ogrenme_mbanet_abdnet_rganet.ipynb  # MBA-Net + ABD-Net + RGA-Net
├── colab_abdnet_rganet.ipynb                 # ABD-Net + RGA-Net (Drive kayıt destekli)
│
🖼️ Sonuç Görselleri
├── sonuc_egitim_mbanet_palmar.png            # MBA-Net eğitim eğrisi
├── sonuc_egitim_abdnet_palmar.png            # ABD-Net eğitim eğrisi
├── sonuc_far_frr_mbanet_palmar.png           # MBA-Net FAR/FRR grafiği
├── sonuc_far_frr_abdnet_palmar.png           # ABD-Net FAR/FRR grafiği
│
📄 Dokümantasyon
├── notes.md                                  # Geliştirme logu (her adımın detayı)
└── output/
    ├── results_summary.csv                   # Geleneksel ML sonuçları
    └── plots/                                # Tüm grafikler
```

---

## Kurulum ve Kullanım

### Gereksinimler

```bash
pip install mediapipe scikit-learn matplotlib seaborn kagglehub tqdm opencv-python numpy pandas scipy
```

Python 3.10+ önerilir.

### Yerel Pipeline

**Adım 1 — Dataset İndir**

Kaggle API token'ı için: kaggle.com → Settings → API → Create New Token

```bash
python 01_veri_indir.py
```

**Adım 2 — MediaPipe Modelini İndir**

```bash
python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', 'hand_landmarker.task')"
```

**Adım 3 — Özellik Çıkar**

```bash
python 02_mediapipe_ozellik_cikar.py
```

~11k görüntü işlenir, `output/features.csv` üretilir. CPU'da ~5–10 dakika.

**Adım 4 — Eğit ve Değerlendir**

```bash
python 03_geleneksel_ml_egit.py
```

SVM+MLP × tüm/dorsal/palmar deneyleri çalışır, metrikler ve grafikler üretilir. ~20–30 dakika (GridSearchCV).

### Colab Notebooks

Derin öğrenme modelleri ve uçtan uca ödev pipeline'ı için Colab notebook'larını Google Colab'a yükleyin, T4 GPU seçin, Kaggle token'ınızı ilgili hücreye yapıştırın ve sırayla çalıştırın.

---

## Ders

Biyometrik Sistemler — Bilgisayar Mühendisliği
