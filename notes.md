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
