# MRI Sınıflandırması - Yapay Zeka Projesi

MRI (Manyetik Rezonans Görüntüleme) görüntülerini kullanarak demans hastalığını sınıflandıran kapsamlı makine öğrenmesi projesi.

## 📋 Proje Yapısı

```
Machine_Learning/
├── Görüntü_On_Isleme/              # Görüntü ön işleme modülü
│   ├── requirements.txt             # Bağımlılıklar
│   ├── goruntu_isleme_kontrol_paneli.py  # Ana kontrol paneli
│   ├── goruntu_isleme_mri/          # Ön işleme araçları
│   │   ├── ayarlar.py               # Proje ayarları
│   │   ├── io_araclari.py           # Dosya okuma/yazma
│   │   ├── on_isleme_adimlari.py    # Ön işleme pipeline
│   │   ├── csv_olusturucu.py        # CSV oluşturma
│   │   ├── veri_artirma.py          # Veri artırma
│   │   ├── veri_normalizasyon.py    # Normalizasyon
│   │   ├── veri_boluntuleme.py      # Train/Val/Test ayırma
│   │   ├── gelismis_filtreler.py    # 20+ filtreleme fonksiyonu
│   │   ├── dosya_yoneticisi.py      # Dosya yönetimi
│   │   └── dosyalama_islemleri.py   # Dosyalama menüsü
│   └── scripts/
│       └── TUMU_ISLEMLER.py         # Tüm işlemler (Ana script)
│
├── Görüntüleri_Detayli_İncele/     # EDA (Exploratory Data Analysis)
│   ├── requirements.txt
│   ├── mri_eda_jpg/                 # EDA araçları
│   │   ├── ayarlar.py
│   │   ├── io_araclari.py
│   │   ├── grafik_araclari.py
│   │   └── istatistik_araclari.py
│   └── scripts/
│       └── analiz_calistir.py
│
├── Model/                           # Model eğitimi ve değerlendirmesi
│   ├── requirements.txt
│   ├── config.py                    # Merkezi konfigürasyon
│   ├── gradient_boosting_model.py   # XGBoost/LightGBM modeli
│   ├── linear_svm_model.py          # Linear SVM modeli
│   ├── model_evaluator.py           # Model değerlendirmesi
│   ├── model_manager.py             # Model yönetimi ve versiyonlama
│   ├── visualizer.py                # Görselleştirme araçları
│   ├── train_and_evaluate_models.py # Eğitim ve değerlendirme
│   ├── test_models.py               # Unit testler
│   ├── example_usage.py             # Örnek kullanım
│   └── outputs/                     # Çıktı dizini
│       ├── models/                  # Eğitilmiş modeller
│       ├── reports/                 # Raporlar
│       └── visualizations/          # Grafikler
│
├── Veri_Seti/                       # Ham veri
│   ├── NonDemented/                 # Normal bilişsel durumu olan hastalar
│   ├── VeryMildDemented/            # Çok hafif demans
│   ├── MildDemented/                # Hafif demans
│   └── ModerateDemented/            # Orta demans
│
└── README.md                        # Bu dosya
```

## 🎯 Demans Sınıfları

Proje 4 sınıfta demans hastalığını sınıflandırır:

| Sınıf | Açıklama |
|-------|----------|
| **Non Demented** | Normal bilişsel durumu olan bireyler |
| **Very Mild Demented** | Çok hafif demans (CDR=0.5) |
| **Mild Demented** | Hafif demans (CDR=1) |
| **Moderate Demented** | Orta demans (CDR=2) |

## 📦 Kurulum

### Gereksinimler
- Python 3.8+
- pip

### Adım 1: Proje Dosyalarını İndirin
```bash
cd Machine_Learning
```

### Adım 2: Virtual Environment Oluşturun (Opsiyonel ama Tavsiye Edilir)
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### Adım 3: Bağımlılıkları Kurun
```bash
# Görüntü işleme için
pip install -r Görüntü_On_Isleme/requirements.txt

# EDA için
pip install -r Görüntüleri_Detayli_İncele/requirements.txt

# Model eğitimi için
pip install -r Model/requirements.txt
```

## 🚀 Kullanım

### Görüntü Ön İşleme

#### Kontrol Paneli ile (İnteraktif)
```bash
cd Görüntü_On_Isleme
python goruntu_isleme_kontrol_paneli.py
```

**Menü Seçenekleri:**
1. **Toplu ön işleme**: Tüm görüntülere ön işleme uygula
2. **CSV oluşturma**: Görüntüleri CSV formatına çevir
3. **Tek görüntü inceleme**: Ön işleme adımlarını göster
4. **Veri bölüntüleme**: Train/Val/Test ayırma
5. **Veri seti kontrol**: İstatistik ve anomali tespiti
6. **CSV analiz ve export**: CSV analiz ve dışa aktarma

#### Command Line ile (Komut Satırından)
```bash
cd Görüntü_On_Isleme
python scripts/TUMU_ISLEMLER.py
```

### Model Eğitimi

```bash
cd Model
python train_and_evaluate_models.py
```

**Bu komut:**
- CSV verilerini yükler
- Eğitim/Doğrulama/Test setlerine böler
- Gradient Boosting modelini eğitir
- Linear SVM modelini eğitir
- Modelleri karşılaştırır
- Raporlar ve grafikler oluşturur

### Model Testleri

```bash
cd Model
python test_models.py
```

### Örnek Kullanım

```bash
cd Model
python example_usage.py
```

## 📊 Veri İşleme Pipeline'ı

```
Ham MRI Görüntüleri (JPG/PNG)
         ↓
   Ön İşleme Aşamaları
   ├─ Gri tonlamaya çevir
   ├─ Boyut standardizasyonu
   ├─ Arka plan maskeleme
   ├─ Kontrast normalizasyonu
   └─ Veri artırma (opsiyonel)
         ↓
   Özellikleri Çıkart
   ├─ İstatistiksel özellikler
   ├─ Doku analizi
   ├─ Histogram özellikleri
   └─ Entropi ve kontrast
         ↓
   CSV Dosyası Oluştur
         ↓
   Veri Bölüntüleme
   ├─ Eğitim seti (70%)
   ├─ Doğrulama seti (15%)
   └─ Test seti (15%)
         ↓
   Model Eğitimi
   ├─ Gradient Boosting
   ├─ Linear SVM
   └─ Karşılaştırma
         ↓
   Model Değerlendirmesi
   ├─ Doğruluk (Accuracy)
   ├─ Kesinlik (Precision)
   ├─ Geri Çağırma (Recall)
   ├─ F1 Skoru
   └─ Karmaşıklık Matrisi
```

## 🔧 Konfigürasyon

### Görüntü Ön İşleme Ayarları
Dosya: `Görüntü_On_Isleme/goruntu_isleme_mri/ayarlar.py`

```python
# Giriş/Çıkış klasörleri
GİRDİ_KLASORU = "veri/girdi"
CIKTI_KLASORU = "veri/çıktı"

# Görüntü ayarları
HEDEF_BOYUT = (256, 256)
HEDEF_KANAL = 'L'  # Gri tonlama

# Veri artırma
VERI_ARTIRMA_AKTIF = True
```

### Model Eğitimi Ayarları
Dosya: `Model/config.py`

```python
# Gradient Boosting
GRADIENT_BOOSTING_CONFIG = {
    'algorithm': 'xgboost',
    'n_estimators': 100,
    'max_depth': 7,
    'learning_rate': 0.1,
    ...
}

# Linear SVM
LINEAR_SVM_CONFIG = {
    'C': 1.0,
    'loss': 'squared_hinge',
    'max_iter': 2000,
    ...
}

# Veri Bölümleme
DATA_SPLIT_CONFIG = {
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    ...
}
```

## 📈 Özellikler (Features)

Çıkarılan özellikler CSV'ye kaydedilir:

| Özellik | Açıklama |
|---------|----------|
| `mean_intensity` | Ortalama piksel yoğunluğu |
| `std_intensity` | Standart sapma |
| `min_intensity` | Minimum yoğunluk |
| `max_intensity` | Maksimum yoğunluk |
| `entropy` | Shannon entropisi |
| `contrast` | Doku kontrastı |
| `homogeneity` | Doku homojenliği |
| `dissimilarity` | Doku farklılığı |
| ... | (20+ özellik) |

## 📝 Çıktılar

### CSV Dosyası
- **Konum**: `Görüntü_On_Isleme/çıktı/goruntu_ozellikleri_scaled.csv`
- **İçerik**: Görüntü özellikleri ve normalizasyon

### Modeller
- **Konum**: `Model/outputs/models/`
- **Format**: JSON ve Pickle

### Raporlar
- **Konum**: `Model/outputs/reports/`
- **İçerik**: Eğitim ve değerlendirme raporları

### Grafikler
- **Konum**: `Model/outputs/visualizations/`
- **İçerik**: Karmaşıklık matrisleri, ROC eğrileri, özelliklerin önemi

## 🛠️ Gelişmiş Filtreler

`gelismis_filtreler.py` modülü 20+ filtreleme fonksiyonu içerir:

- **Morfolojik**: Açılış, kapanış, gradient
- **Kenar Tespiti**: Sobel, Laplacian, Canny
- **Doku Analizi**: GLCM, LBP
- **Kontrol**: Medyan, Bilateral, Gaussian
- **Frekans Alanı**: FFT, Wavelet
- **Özel**: Arka plan maskeleme, entropikSharpen

## 🧪 Test Etme

Proje için yazılmış unit testler:

```bash
cd Model
python test_models.py
```

## 📚 Dokümantasyon

Her modülün başında detaylı docstring'ler bulunur. Örnek:

```python
def func_name(param1: str, param2: int) -> Dict:
    r"""
    Fonksiyonun açıklaması.
    
    Parametreler:
    -----------
    param1 : str
        Açıklama
    param2 : int
        Açıklama
    
    Döndürülen:
    ---------
    Dict
        Açıklama
    """
```

## ⚠️ Notlar

- **Veri Seti**: Ham MRI görüntüleri `Veri_Seti/` klasöründe bulunmalıdır
- **CSV Dosyası**: `scripts/TUMU_ISLEMLER.py` ile otomatik oluşturulur
- **Model Dosyaları**: `Model/outputs/models/` dizininde saklanır
- **Loglama**: `Model/outputs/training.log` dosyasına kaydedilir

## 🤝 Katkıda Bulunma

Projekti geliştirmek için:

1. Fork yapın
2. Feature branch'i oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişiklikleri commit edin (`git commit -m 'Add AmazingFeature'`)
4. Branch'e push yapın (`git push origin feature/AmazingFeature`)
5. Pull request açın

## 📄 Lisans

Bu proje MIT Lisansı altında dağıtılmaktadır. Detaylar için `LICENSE` dosyasını inceleyin.

## 👨‍💻 Yazar

**MRI Sınıflandırması Projesi**
- Repository: [MRI_Classification](https://github.com/mozybali/MRI_Classification)
- Geliştirici: mozybali

## 📞 İletişim

Sorularınız veya önerileriniz varsa lütfen bir issue açın.

---

**Son Güncelleme**: Aralık 2025
