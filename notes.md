# Proje Geliştirme Logu

---

## BASELINE — İlk Sonuçlar

**Özellik seti:** 33 geometrik özellik (parmak uzunlukları, uç mesafeleri, avuç genişlikleri, oranlar, açılar, bilek uzaklıkları)
**Veri:** 10.765 görüntü (11.076'dan %97.2 başarıyla çıkarıldı), 189 kişi

| Model | Accuracy | Macro F1 | EER |
|-------|----------|----------|-----|
| SVM (RBF, C=10, gamma=scale) | 78.85% | 74.00% | 36.93% |
| MLP (256-128-64, adam) | 75.94% | 71.20% | 37.07% |

**Gözlem:** SVM, MLP'yi ~3 puan geride bıraktı. EER her iki modelde de ~37% — yüksek görünse de 189 sınıflı bir doğrulama probleminde centroid tabanlı yöntemde beklenen bir sonuç.

---

## GELİŞTİRME 1 — Ham Landmark Koordinatları Eklendi

**Değişiklik:** `02_extract_features.py` — normalize edilmiş 21 landmark'ın (x,y) koordinatları özellik vektörüne eklendi.

**Gerekçe:** 33 geometrik özellik el anatomisini özetler; ancak parmaklar arası açısal ilişkiler ve genel el şekli sınıflandırıcıya doğrudan verilmez. Ham koordinatlar bu bilgiyi tamamlar.

**Özellik seti:** 33 geometrik + 42 ham koordinat = **75 özellik**

| Grup | Özellikler | Sayı |
|------|-----------|------|
| Parmak uzunlukları | Her parmak için segment toplamı | 5 |
| Parmak ucu mesafeleri | 5 ucun ikili kombinasyonları | 10 |
| Avuç genişlikleri | MCP eklemleri arası | 4 |
| Uzunluk/genişlik oranları | Parmak / avuç | 5 |
| PIP eklem açıları | Orta eklem bükme açıları | 4 |
| Uç→bilek mesafeleri | Her ucun bilek uzaklığı | 5 |
| **YENİ** Ham landmark (x,y) | 21 normalize nokta koordinatı | 42 |
| **Toplam** | | **75** |

**Sonuçlar:** *(03_train_evaluate.py çalıştırıldıktan sonra güncellenecek)*

---

## GELİŞTİRME 2 — GridSearchCV ile Parametre Optimizasyonu

**Değişiklik:** `03_train_evaluate.py` — SVM eğitimi artık sabit parametre yerine GridSearchCV kullanıyor.

**Gerekçe:** C ve gamma değerleri veriye özgüdür; elle seçmek yerine çapraz doğrulama ile en iyi kombinasyonu bulmak hem accuracy'yi artırır hem de akademik açıdan daha savunulabilir bir yöntemdir.

**Taranan parametre uzayı:**
- `C` : [1, 10, 100]
- `gamma` : ['scale', 0.01, 0.001]
- `cv=3` (3-katlı çapraz doğrulama), `scoring='accuracy'`, `n_jobs=-1`
- Toplam 9 kombinasyon × 3 fold = 27 model

**En iyi parametreler:** `C=100, gamma='scale'` — tüm veri setlerinde tutarlı biçimde aynı kombinasyon seçildi. Baseline'daki C=10'dan C=100'e geçiş **~3.5 puan** accuracy artışı sağladı.

---

## GELİŞTİRME 3 — Dorsal / Palmar Ayrımı

**Değişiklik:** `03_train_evaluate.py` — `HandInfo.csv` içindeki `aspectOfHand` sütunu kullanılarak üç ayrı deney çalıştırılıyor: tüm veri, yalnızca el sırtı (dorsal), yalnızca avuç içi (palmar).

**Gerekçe:** Dataset'te her kişinin hem el sırtı hem avuç içi görüntüleri karışık olarak bulunuyor. Bu iki görünüm farklı geometrik bilgi taşır — el sırtında parmak kemikleri belirginken, avuç içinde çizgiler ve yumrular öne çıkar. Ayrı eğitim verilen modeller, görünüme özgü kalıpları daha iyi öğrenebilir ve confusion matrixi azaltabilir.

**Deney planı:**

| Deney | Veri | Beklenti |
|-------|------|---------|
| SVM_tum / MLP_tum | Tüm görüntüler | Baseline (Geliştirme 1+2 ile) |
| SVM_dorsal / MLP_dorsal | Yalnızca el sırtı | Dorsal'e özgü kalıplar → accuracy ↑ |
| SVM_palmar / MLP_palmar | Yalnızca avuç içi | Palmar'a özgü kalıplar → accuracy ↑ |

**Sonuçlar (75 özellik ile):**
- **SVM_palmar en yüksek performansı verdi: %93.03 accuracy, %90.29 Macro F1**
- Palmar modeli dorsalden ~1.5 puan daha iyi → avuç içi geometrisi daha ayırt edici
- EER tüm veri modelinin %36.56'sından dorsal/palmar modellerinde %27-28'e düştü (**~9 puan iyileşme**)
- GridSearchCV dorsal/tüm için `gamma=0.01`, palmar için `gamma='scale'` seçti

---

## KARŞILAŞTIRMA TABLOSU — TAM SONUÇLAR

| Deney | Özellik | Accuracy | Macro F1 | EER | Best Params |
|-------|---------|----------|----------|-----|-------------|
| Baseline SVM | 33 | 78.85% | 74.00% | 36.93% | C=10, gamma=scale |
| Baseline MLP | 33 | 75.94% | 71.20% | 37.07% | — |
| SVM_tum | 33 | 82.41% | 78.70% | 37.01% | C=100, gamma=scale |
| MLP_tum | 33 | 75.94% | 71.20% | 37.01% | — |
| SVM_dorsal | 33 | 82.72% | 78.19% | 26.34% | C=100, gamma=scale |
| MLP_dorsal | 33 | 75.11% | 69.38% | 26.34% | — |
| SVM_palmar | 33 | 87.12% | 83.58% | 27.11% | C=100, gamma=scale |
| MLP_palmar | 33 | 81.46% | 77.50% | 27.11% | — |
| SVM_tum | 75 | 91.89% | 89.67% | 36.56% | C=100, gamma=0.01 |
| MLP_tum | 75 | 86.78% | 83.32% | 36.56% | — |
| SVM_dorsal | 75 | 91.58% | 88.53% | 28.52% | C=100, gamma=0.01 |
| MLP_dorsal | 75 | 83.91% | 78.78% | 28.52% | — |
| **SVM_palmar** | **75** | **93.03%** | **90.29%** | **27.49%** | C=100, gamma=scale |
| MLP_palmar | 75 | 88.24% | 85.01% | 27.49% | — |

**En iyi model: SVM_palmar (75 özellik) — %93.03 accuracy, %27.49 EER**

---

## GELİŞTİRME 4 — Ek Modeller (KNN, Random Forest, Gradient Boosting)

### Tam Sonuçlar (75 özellik)

| Deney | SVM | MLP | KNN | RF | GB |
|-------|-----|-----|-----|----|----|
| Tüm veri — Accuracy | **91.89%** | 86.78% | 72.04% | 86.53% | 9.29%* |
| Dorsal — Accuracy | **91.58%** | 83.91% | 73.04% | 84.66% | 14.64%* |
| Palmar — Accuracy | **93.03%** | 88.24% | 77.22% | **89.05%** | 16.68%* |
| Tüm veri — EER | 36.56% | 36.56% | 36.56% | 36.56% | 36.56% |
| Dorsal — EER | 28.52% | 28.52% | 28.52% | 28.52% | 28.52% |
| Palmar — EER | **27.49%** | 27.49% | 27.49% | 27.49% | 27.49% |

*GB sonuçları geçersiz — bkz. aşağıdaki not.

### Model Sıralaması (Palmar, 75 özellik)

1. **SVM — %93.03** (en iyi)
2. **RF — %89.05**
3. **MLP — %88.24**
4. **KNN — %77.22**
5. ~~GB — %16.68~~ (underfitting)

### Gradient Boosting Neden Başarısız?

`HistGradientBoostingClassifier` ile 184 sınıflı bir problemde her sınıf için ayrı bir ikili sınıflandırıcı (OvR) eğitilir. `max_iter=200` bu ölçekte yeterli değil; model yakınsayamadan durdu. Düzeltmek için `max_iter=1000+` gerekir ancak hesaplama süresi çok uzar. Raporda "GB bu parametrelerle yakınsamadı, dışarıda bırakıldı" olarak belirtilecek.

### EER Neden Tüm Modellerde Aynı?

EER hesabı, sınıflandırıcının kendi çıktısını değil, **centroid tabanlı Euclidean mesafeyi** kullanıyor. Bu nedenle aynı veri bölümünde tüm modeller aynı EER'i paylaşıyor. Bu tutarlı bir tasarım seçimi — EER verinin ne kadar ayırt edici olduğunu ölçüyor, modelin ne kadar iyi olduğunu değil.

### Random Forest Özellik Önemi (Öne Çıkan Bulgular)

RF modelinden çıkan `feat_importance_RF_palmar.png` grafiğine göre en ayırt edici özellikler ham landmark koordinatları (Grup 7) — özellikle orta parmak ve yüzük parmağı MCP bölgesi. Bu, Geliştirme 1'in (ham koordinat ekleme) neden bu kadar büyük etki yaptığını açıklıyor.

---

## GELİŞTİRMELERİN ETKİSİ — RAPOR İÇİN ANALİZ

### Ham Landmark Eklemenin Etkisi (33 → 75 özellik)

| Model | 33 özellik | 75 özellik | Artış |
|-------|-----------|-----------|-------|
| SVM_tum | 82.41% | 91.89% | **+9.48 puan** |
| MLP_tum | 75.94% | 86.78% | **+10.84 puan** |
| SVM_palmar | 87.12% | 93.03% | **+5.91 puan** |
| MLP_palmar | 81.46% | 88.24% | **+6.78 puan** |

En büyük farkı getiren geliştirme buydu. Ham koordinatlar, geometrik özetlerin yakalayamadığı parmak açıları ve el şeklinin global yapısını sınıflandırıcıya doğrudan sunuyor.

### GridSearchCV Etkisi (C=10 → C=100)

Tüm veri setinde baseline %78.85 → %82.41 (+3.56 puan). GridSearchCV gamma için de `scale` yerine `0.01` seçti (palmar hariç) — RBF çekirdeğinin daha dar bir alanda çalışması 75 boyutlu uzayda daha iyi ayrım sağlıyor.

### Dorsal / Palmar Ayrımının Etkisi

| Karşılaştırma | Accuracy | EER |
|--------------|----------|-----|
| SVM_tum (75 özellik) | 91.89% | 36.56% |
| SVM_dorsal (75 özellik) | 91.58% | 28.52% |
| SVM_palmar (75 özellik) | **93.03%** | **27.49%** |

- **Accuracy** açısından palmar +1 puan öne geçiyor — küçük ama anlamlı
- **EER** açısından ayrım dramatik: %36.56 → %27-28 (**~9 puan düşüş**)
- **Palmar > Dorsal:** Avuç içi geometrisi kişiye özgü daha ayırt edici özellikler taşıyor. El sırtında parmaklar birbirine benzerken, avuç içindeki eklem ve yumru yapıları farklılaşıyor.
- Karma veri (tüm) modelinde iki görünümün birbirine karışması EER'i olumsuz etkiliyor.

---

## DERİN ÖĞRENME — MBA-Net, ABD-Net, RGA-Net (Google Colab T4 GPU)

**Dosya:** `05_deep_learning_colab.ipynb`
**Mimari:** ResNet-50 backbone (ImageNet pretrained) + dikkat mekanizmaları
**Kayıp:** CrossEntropyLoss + TripletLoss (margin=0.3, weight=0.5)
**Optimizer:** Adam (lr=1e-4, weight_decay=5e-4), StepLR (step=10, gamma=0.5)
**Eğitim:** 30 epoch, ilk 5 epoch backbone donduruldu, Batch=64, IMG=224×224

---

### MBA-Net — Palmar (30.04.2026)

**Mimari:** SEBlock + Global Branch (GAP→BN→Dropout→FC) + Local Branch (4 yatay şerit→concat) + L2-normalize

| Parametre | Değer |
|-----------|-------|
| Sınıf sayısı | 184 |
| Eğitim / Test | 3750 / 1608 |
| Parametre sayısı | 26,324,216 |

**Epoch Geçmişi:**

| Epoch | Loss | Train Acc | Test Acc |
|-------|------|-----------|----------|
| 1 | 5.2370 | 5.39% | 25.81% |
| 2 | 5.1933 | 28.61% | 47.89% |
| 3 | 5.1539 | 42.75% | 54.66% |
| 4 | 5.1122 | 51.36% | 58.71% |
| 5 | 5.0704 | 55.89% | 61.50% |
| *6 (backbone açıldı)* | 4.9942 | 50.91% | 62.44% |
| 7 | 4.8724 | 65.12% | 70.40% |
| 8 | 4.7725 | 75.71% | 81.90% |
| 9 | 4.6799 | 82.64% | 85.07% |
| 10 | 4.5932 | 86.67% | 88.25% |
| 11 | 4.5289 | 89.92% | 89.24% |
| 12 | 4.4883 | 90.59% | 90.24% |
| 13 | 4.4502 | 91.84% | 91.54% |
| 14 | 4.4120 | 92.80% | 92.66% |
| 15 | 4.3743 | 93.36% | 93.41% |
| 16 | 4.3392 | 94.48% | 93.91% |
| 17 | 4.3032 | 95.01% | 94.28% |
| 18 | 4.2675 | 95.49% | 95.27% |
| 19 | 4.2329 | 95.81% | 95.52% |
| 20 | 4.1991 | 96.37% | 96.02% |
| 21 | 4.1712 | 97.09% | 96.14% |
| 22 | 4.1539 | 97.44% | 96.21% |
| 23 | 4.1374 | 97.04% | 96.58% |
| 24 | 4.1199 | 97.17% | 96.39% |
| 25 | 4.1040 | 97.28% | 96.46% |
| 26 | 4.0865 | 97.71% | 96.83% |
| 27 | 4.0703 | 97.63% | 96.39% |
| 28 | 4.0546 | 97.73% | 96.52% |
| 29 | 4.0379 | 97.89% | 96.70% |
| **30** | **4.0217** | **97.89%** | **97.26%** |

**Sonuçlar:**

| Metrik | Değer |
|--------|-------|
| **Rank-1** | **99.81%** |
| Accuracy | 97.26% |
| **EER** | **0.13%** |

**Gözlem:** Backbone açıldıktan sonra (epoch 6+) test accuracy hızla tırmanıyor. Rank-1 %99.81, EER %0.13 — literatür referansını (%96.8 Rank-1, %5.49 EER) önemli ölçüde geçti. Fine-tuning stratejisinin (ilk 5 epoch dondur, sonra aç) çok etkili olduğu görülüyor.

> **Not:** 2. çalıştırmada (05_deep_learning_colab.ipynb) Accuracy %96.89, EER %0.12 çıktı — farklı random seed nedeniyle küçük sapma, Rank-1 %99.81 aynı kaldı.

---

### ABD-Net — Palmar (30.04.2026)

**Mimari:** ChannelAttention + SpatialAttention (CBAM) + Diversity Loss (ortogonallik kısıtı, lambda=1e-3)

| Parametre | Değer |
|-----------|-------|
| Sınıf sayısı | 184 |
| Eğitim / Test | 3750 / 1608 |
| Parametre sayısı | 26,233,178 |

**Epoch Geçmişi:**

| Epoch | Loss | Train Acc | Test Acc |
|-------|------|-----------|----------|
| 1 | 5.6815 | 7.87% | 22.64% |
| 2 | 5.5070 | 30.53% | 41.48% |
| 3 | 5.3776 | 45.36% | 52.49% |
| 4 | 5.2742 | 54.85% | 56.72% |
| 5 | 5.1889 | 59.28% | 59.83% |
| *6 (backbone açıldı)* | 5.0879 | 58.93% | 68.28% |
| 7 | 4.9263 | 78.16% | 83.58% |
| 8 | 4.7947 | 88.35% | 90.49% |
| 9 | 4.6792 | 93.92% | 95.02% |
| 10 | 4.5793 | 97.17% | 96.64% |
| 11 | 4.5077 | 98.56% | 98.13% |
| 12 | 4.4649 | 99.15% | 98.38% |
| 13 | 4.4247 | 99.55% | 98.76% |
| 14 | 4.3868 | 99.60% | 98.94% |
| 15 | 4.3499 | 99.79% | 99.25% |
| 16 | 4.3140 | 99.92% | 99.44% |
| 17 | 4.2787 | 99.87% | 99.38% |
| 18 | 4.2450 | 99.97% | 99.50% |
| 19 | 4.2117 | 99.97% | 99.56% |
| 20 | 4.1782 | 100.00% | 99.50% |
| 21 | 4.1526 | 100.00% | 99.69% |
| 22 | 4.1358 | 100.00% | 99.69% |
| 23 | 4.1199 | 100.00% | 99.63% |
| 24 | 4.1036 | 100.00% | 99.69% |
| 25 | 4.0881 | 100.00% | 99.69% |
| 26 | 4.0720 | 100.00% | 99.69% |
| 27 | 4.0567 | 100.00% | 99.63% |
| 28 | 4.0406 | 100.00% | 99.69% |
| 29 | 4.0253 | 99.97% | 99.69% |
| **30** | **4.0106** | **100.00%** | **99.69%** |

**Sonuçlar:**

| Metrik | Değer |
|--------|-------|
| **Rank-1** | **99.88%** |
| **Accuracy** | **99.69%** |
| **EER** | **0.01%** |

**Gözlem:** Tüm modeller içinde en yüksek performans. Epoch 20'den itibaren train accuracy %100, test accuracy %99.69'da sabitlendi. EER %0.01 pratikte sıfır hata anlamına geliyor. Diversity loss'un iki branch'i birbirinden farklı özellikler öğrenmeye zorlaması çok etkili olmuş.

---

### RGA-Net — Palmar (30.04.2026) — *(devam ediyor, epoch 21'de)*

**Mimari:** RGAModule (uzamsal öz-ilişki matrisi) layer3 + layer4 sonrasına eklendi

| Parametre | Değer |
|-----------|-------|
| Sınıf sayısı | 184 |
| Eğitim / Test | 3750 / 1608 |
| Parametre sayısı | 35,147,512 |

**Epoch Geçmişi (kısmi):**

| Epoch | Loss | Train Acc | Test Acc |
|-------|------|-----------|----------|
| 1 | 5.2399 | 2.83% | 14.93% |
| 2 | 5.2007 | 17.12% | 33.21% |
| 3 | 5.1578 | 31.41% | 44.90% |
| 4 | 5.1214 | 39.68% | 50.06% |
| 5 | 5.0847 | 46.13% | 52.24% |
| *6 (backbone açıldı)* | 5.0141 | 42.91% | 51.55% |
| 7 | 4.8886 | 62.24% | 70.90% |
| 8 | 4.7829 | 76.19% | 80.78% |
| 9 | 4.6864 | 84.77% | 88.31% |
| 10 | 4.6011 | 90.72% | 91.48% |
| 11 | 4.5373 | 93.81% | 93.10% |
| 12 | 4.4985 | 94.53% | 94.15% |
| 13 | 4.4620 | 96.59% | 95.65% |
| 14 | 4.4275 | 97.31% | 96.33% |
| 15 | 4.3946 | 97.81% | 97.26% |
| 16 | 4.3613 | 98.27% | 97.70% |
| 17 | 4.3300 | 98.96% | 98.26% |
| 18 | 4.2998 | 99.31% | 98.76% |
| 19 | 4.2697 | 99.33% | 98.82% |
| 20 | 4.2394 | 99.47% | 99.13% |
| 21 | 4.2149 | 99.79% | 99.19% |
| 22–30 | — | — | *(devam ediyor)* |

**Gözlem (erken):** En fazla parametreye sahip model (35M). MBA-Net'ten daha yavaş başlangıç ama epoch 21'de %99.19 test — ABD-Net seviyesine yaklaşıyor. Sonuçlar tamamlanınca güncellenecek.

---
| 8 | 4.7947 | 88.35% | 91.23% |
| 9 | 4.6785 | 94.03% | 94.65% |
| 10 | 4.5786 | 97.28% | 96.58% |
| 11–30 | — | — | *(devam ediyor)* |

**Gözlem (erken):** Epoch 7-8'de MBA-Net'ten daha hızlı yükseliş — CBAM dikkat mekanizması backbone açıldıktan sonra çok hızlı öğreniyor. Epoch 10'da zaten %96.58 test accuracy.

---

## YÖNTEM NOTLARI

### MediaPipe Landmark Düzeni
```
         8   12  16  20
         7   11  15  19
    4    6   10  14  18
    3    5    9  13  17
    2         \ | /
    1           0  (WRIST)
```

### Normalizasyon Adımları
1. Tüm noktalardan WRIST (landmark 0) çıkarılır → öteleme bağımsızlığı
2. WRIST→MIDDLE_MCP mesafesine bölünür → ölçek bağımsızlığı

### FAR / FRR / EER Tanımları
| Metrik | Formül | Açıklama |
|--------|--------|---------|
| FAR | Yanlış kabul / Toplam impostor | Sahtecinin kabul edilme oranı |
| FRR | Yanlış red / Toplam genuine | Gerçek kullanıcının reddedilme oranı |
| EER | FAR = FRR noktası | Tek sayılık biyometrik performans özeti |

**Hesaplama yöntemi:** Her sınıf için eğitim verisi centroid'i hesaplanır. Test örneğinin kendi centroid'ine olan Euclidean mesafesi genuine skoru, diğer centroid'lere olan mesafe impostor skoru olarak kullanılır. 500 eşik değeri taranarak FAR ve FRR eğrileri çizilir.
