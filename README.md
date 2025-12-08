# MRI Sınıflandırması - Yapay Zeka Projesi

Manyetik Rezonans Görüntüleme (MRI) görüntülerini kullanarak demans hastalığını sınıflandıran kapsamlı bir makine öğrenmesi projesi.

## 📋 Proje Özeti

Bu proje, MRI beyin görüntülerinden otomatik olarak demans hastalığı teşhisini yapmayı amaçlamaktadır. Proje, görüntü ön işleme, öznitelik çıkarma ve makine öğrenmesi modellerinin eğitilmesi olmak üzere üç ana bölümden oluşmaktadır.

### Sınıflandırma Kategorileri
- **NonDemented** - Sağlıklı (Demans yok)
- **VeryMildDemented** - Çok hafif demans
- **MildDemented** - Hafif demans
- **ModerateDemented** - Orta seviye demans

## 🏗️ Proje Mimarisi

```
Machine_Learning/
│
├── Görüntü_On_Isleme/              # Görüntü ön işleme ve CSV oluşturma
│   ├── requirements.txt             # Python bağımlılıkları
│   ├── goruntu_isleme_kontrol_paneli.py  # Ana menü arayüzü
│   ├── goruntu_isleme_mri/          # Ön işleme modülleri
│   │   ├── ayarlar.py               # Konfigürasyon
│   │   ├── io_araclari.py           # Dosya I/O işlemleri
│   │   ├── on_isleme_adimlari.py    # Ön işleme pipeline
│   │   ├── csv_olusturucu.py        # Öznitelik çıkarma
│   │   ├── veri_artirma.py          # Veri augmentation
│   │   ├── veri_normalizasyon.py    # Normalizasyon
│   │   ├── veri_boluntuleme.py      # Train/Val/Test bölümü
│   │   ├── gelismis_filtreler.py    # İleri filtreler
│   │   ├── arka_plan_isleme.py      # Background processing
│   │   ├── dosya_yoneticisi.py      # Dosya yönetimi
│   │   └── dosyalama_islemleri.py   # Veri seti organizasyonu
│   └── scripts/
│       └── TUMU_ISLEMLER.py         # Tüm işlemleri otomatik yapan script
│
├── Görüntüleri_Detayli_İncele/     # Veri analizi ve görselleştirme
│   ├── requirements.txt
│   ├── mri_eda_jpg/                 # EDA araçları
│   │   ├── ayarlar.py
│   │   ├── io_araclari.py
│   │   ├── grafik_araclari.py
│   │   └── istatistik_araclari.py
│   └── scripts/
│       └── analiz_calistir.py       # EDA analizi
│
├── Model/                           # Makine öğrenmesi modelleri
│   ├── requirements.txt
│   ├── config.py                    # Merkezi konfigürasyon ve hyperparametreler
│   ├── config.json                  # Config dosyası (JSON formatı)
│   ├── gradient_boosting_model.py   # XGBoost/LightGBM modeli
│   ├── linear_svm_model.py          # Linear SVM sınıflandırıcı
│   ├── model_evaluator.py           # Model değerlendirme metrikleri
│   ├── model_manager.py             # Model versiyonlama ve yönetimi
│   ├── visualizer.py                # Sonuç görselleştirmesi
│   ├── train_and_evaluate_models.py # Ana eğitim script'i
│   ├── test_models.py               # Unit testler
│   ├── example_usage.py             # Örnek kullanım
│   └── outputs/                     # Çıktı klasörü
│       ├── models/                  # Eğitilmiş modeller
│       ├── reports/                 # Performans raporları
│       └── visualizations/          # Grafik ve görseller
│
├── Veri_Seti/                       # Orijinal MRI görüntüleri
│   ├── NonDemented/
│   ├── VeryMildDemented/
│   ├── MildDemented/
│   └── ModerateDemented/
│
└── LICENSE

```

## 🚀 Başlangıç

### Sistem Gereksinimleri
- Python 3.8 veya üzeri
- 4GB+ RAM (model eğitimi için 8GB+ önerilir)
- 2GB+ disk alanı (çıktı dosyaları için)

### Kurulum

1. **Proje klasörüne gidin:**
   ```bash
   cd c:\Users\HectoRSheesh\Desktop\Machine_Learning
   ```

2. **Python paketlerini kurun:**
   ```bash
   # Görüntü ön işleme
   pip install -r Görüntü_On_Isleme\requirements.txt
   
   # Model eğitimi
   pip install -r Model\requirements.txt
   
   # EDA (isteğe bağlı)
   pip install -r Görüntüleri_Detayli_İncele\requirements.txt
   ```

### Çalıştırma

**Seçenek 1: Menü arayüzü ile**
```bash
cd Görüntü_On_Isleme
python goruntu_isleme_kontrol_paneli.py
```

**Seçenek 2: Otomatik olarak tüm işlemler**
```bash
cd Görüntü_On_Isleme\scripts
python TUMU_ISLEMLER.py
```

**Seçenek 3: Model eğitimi (ön işleme yapıldıktan sonra)**
```bash
cd Model
python train_and_evaluate_models.py
```

## 📊 İş Akışı

```
1. VERİ HAZIRLAMA
   ├─ Görüntüleri oku
   ├─ Arka plan tespiti
   ├─ Maske oluşturma
   └─ Kırpma ve boyutlandırma
   
2. ÖN İŞLEME
   ├─ Yoğunluk normalizasyonu
   ├─ Gürültü azaltma
   ├─ Histogram eşitleme
   └─ Veri augmentation
   
3. ÖZNİTELİK ÇIKARMA
   ├─ İstatistiksel öznitelikler
   ├─ Doku analizi
   ├─ Şekil öznitelikleri
   └─ CSV dosyası oluşturma
   
4. VERİ BÖLÜMLEME
   ├─ Eğitim seti (70%)
   ├─ Doğrulama seti (15%)
   └─ Test seti (15%)
   
5. MODEL EĞİTİMİ
   ├─ Gradient Boosting (XGBoost/LightGBM)
   └─ Linear SVM
   
6. DEĞERLENDİRME
   ├─ Doğruluk (Accuracy)
   ├─ Precision/Recall
   ├─ F1-Score
   └─ ROC-AUC
```

## 🔧 Konfigürasyon

Hyperparametreler `Model/config.py` dosyasında tanımlanmıştır:

```python
# Gradient Boosting
GRADIENT_BOOSTING_CONFIG = {
    'algorithm': 'xgboost',  # veya 'lightgbm'
    'n_estimators': 100,
    'max_depth': 7,
    'learning_rate': 0.1,
}

# Linear SVM
LINEAR_SVM_CONFIG = {
    'C': 1.0,
    'kernel': 'rbf',
    'gamma': 'scale',
}

# Veri bölümleme
DATA_SPLIT_CONFIG = {
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
}
```

Ayarları değiştirerek model performansını optimize edebilirsiniz.

## 📈 Proje Modülleri

### Görüntü_On_Isleme
MRI görüntülerinin ön işlenmesi ve öznitelik çıkarılması:
- Görüntü yükleme ve gri dönüştürme
- Arka plan tespiti ve maskeleme
- Histogram eşitleme (CLAHE)
- Gürültü azaltma (bilateral, NLM)
- Min-Max normalizasyon
- Veri artırma (rotation, scaling)

### Görüntüleri_Detayli_İncele
Veri seti analizi ve istatistiksel inceleme:
- Sınıf dağılımı analizi
- Görüntü istatistikleri
- Öznitelik dağılımı
- Korelasyon analizi

### Model
Makine öğrenmesi modelleri:
- **Gradient Boosting:** XGBoost veya LightGBM kullanarak yüksek performanslı sınıflandırma
- **Linear SVM:** Doğrusal kernel kullanan destek vektör makinesi
- **Evaluator:** Modelleri değerlendirme ve karşılaştırma
- **Visualizer:** Confusion matrix, ROC eğrileri, feature importance
- **Model Manager:** Modelleri kaydetme ve versiyon kontrolü

## 💾 Çıktılar

Model eğitimi tamamlandığında aşağıdaki dosyalar oluşturulur:

```
Model/outputs/
├── models/
│   ├── gradient_boosting_latest.pkl
│   ├── linear_svm_latest.pkl
│   └── model_metadata.json
├── reports/
│   ├── evaluation_report.json
│   ├── confusion_matrices.json
│   └── metrics_summary.txt
└── visualizations/
    ├── confusion_matrix_gb.png
    ├── confusion_matrix_svm.png
    ├── roc_curves.png
    └── feature_importance.png
```

## 📝 Örnek Kullanım

```python
# Model yükle
from Model.model_manager import ModelManager
from Model.config import config

manager = ModelManager()
gb_model = manager.load_model('latest', 'gradient_boosting')
svm_model = manager.load_model('latest', 'linear_svm')

# Tahmin yap
import numpy as np
X_new = np.random.rand(10, 45)  # 45 öznitelik
predictions_gb = gb_model.predict(X_new)
predictions_svm = svm_model.predict(X_new)

# Sonuçları göster
print(f"GB Predictions: {predictions_gb}")
print(f"SVM Predictions: {predictions_svm}")
```

## 🧪 Testler

Unit testleri çalıştırmak için:
```bash
cd Model
python test_models.py
```

## 📦 Bağımlılıklar

### Temel Paketler
- **numpy** - Sayısal işlemler
- **pandas** - Veri manipülasyonu
- **scikit-learn** - Makine öğrenmesi
- **opencv-python** - Görüntü işleme
- **scikit-image** - Gelişmiş görüntü işleme
- **scipy** - Bilimsel hesaplamalar

### Model Paketleri
- **xgboost** - Gradient boosting modeli
- **lightgbm** - Alternatif gradient boosting
- **pillow** - Görüntü I/O

### Görselleştirme
- **matplotlib** - 2D grafikler
- **seaborn** - İstatistiksel görselleştirme
- **plotly** - İnteraktif grafikler

## 📄 Lisans

Bu proje [LICENSE](LICENSE) dosyası altında lisanslanmıştır.

## 👥 Katkıda Bulunma

Hata raporları ve öneriler için lütfen issue açınız.

## 📞 İletişim

Sorularınız veya önerileriniz için iletişime geçin.

---

**Son Güncelleme:** Aralık 2025  
**Proje Durumu:** Aktif Geliştirme
