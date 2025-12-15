# MRI Beyin Görüntüsü Sınıflandırma

MRI beyin görüntülerinden demans seviyesini tahmin etmek için uçtan uca bir makine öğrenmesi projesi. Görüntü işleme, özellik çıkarma, EDA, klasik ML modelleri (XGBoost, LightGBM, Linear SVM) ve testler tek bir depo içinde.

## Proje Yapısı

```
MRI_Classification/
├── Veri_Seti/                 # Ham görüntüler (sınıf klasörleri: NonDemented, VeryMildDemented, MildDemented, ModerateDemented)
├── goruntu_isleme/            # Ön işleme + özellik çıkarma
│   ├── ana_islem.py           # Menü tabanlı ana akış
│   ├── goruntu_isleyici.py    # Ön işleme pipeline'ı (bias correction, skull stripping, hizalama, CLAHE, augmentasyon)
│   ├── ozellik_cikarici.py    # 20+ özellik çıkarımı ve CSV oluşturma
│   ├── pipeline_quick_test.py # Tek görüntü için hızlı kontrol
│   ├── test_pipeline.py       # Pipeline testi
│   └── ayarlar.py             # Konfigürasyon
├── eda_analiz/                # Keşifsel veri analizi
│   ├── eda_calistir.py        # Basit arayüz
│   ├── eda_araclar.py         # Paralel istatistik + görselleştirme
│   └── requirements.txt       # Minimal bağımlılıklar
├── model/                     # Model eğitimi ve tahmin
│   ├── train.py               # İnteraktif/otomatik eğitim
│   ├── inference.py           # Tek/batch tahmin
│   ├── model_comparison.py    # Eğitilmiş modelleri kıyaslama
│   ├── model_egitici.py       # Eğitim mantığı ve raporlama
│   └── ayarlar.py             # Model ayarları
├── tests/                     # Pytest senaryoları
├── requirements.txt           # Tüm bağımlılıklar (dev dahil)
└── LICENSE
```

## Kurulum

1) Python ortamınızı hazırlayın (ör. `python -m venv .venv` ve etkinleştirin).  
2) Ana dizinde bağımlılıkları kurun:
```bash
pip install -r requirements.txt
```
3) Ham veri klasörünün `Veri_Seti/<sınıf_adı>/` altında yer aldığından emin olun.

Minimal EDA kurulumu gerekiyorsa `eda_analiz/requirements.txt` dosyasını kullanabilirsiniz.

## Hızlı Başlangıç İş Akışı

1) **Görüntü ön işleme ve özellik çıkarma**  
```bash
cd goruntu_isleme
python ana_islem.py   # Menüde 6 → tüm adımlar
```
İşlenmiş görüntüler ve özellik CSV'leri `goruntu_isleme/cikti/` altında oluşur.

2) **Model eğitimi**  
```bash
cd ../model
python train.py --auto                 # Varsayılan ayarlarla XGBoost
# veya etkileşimli seçim için
python train.py
```
Modeller ve raporlar `model/ciktilar/` klasöründe saklanır.

3) **Tahmin**  
```bash
# Tek görüntü
python inference.py --model model/ciktilar/modeller/xgboost_YYYYMMDD_HHMMSS.pkl --image /path/to/image.jpg
# Klasör içindeki tüm görüntüler
python inference.py --model model/ciktilar/modeller/xgboost_YYYYMMDD_HHMMSS.pkl --batch /path/to/folder/
```

4) **EDA (isteğe bağlı)**  
```bash
cd ../eda_analiz
python eda_calistir.py    # Çıktılar: eda_ciktilar/
```

## Modül Detayları

- **goruntu_isleme**: Bias field correction (SimpleITK mevcutsa N4ITK), skull stripping, hizalama, adaptif CLAHE, z-score normalizasyonu, medikal augmentasyon, sınıf bazlı artırma ve çok çekirdekli toplu işleme. `ozellik_cikarici.py` 20+ öznitelik çıkarır, `ayarlar.py` üzerinden değiştirilebilir.
- **eda_analiz**: Paralel temel istatistik hesaplama, sınıf dağılımı, boyut analizi, yoğunluk dağılımı, korelasyon matrisi ve PCA grafikleri üretir.
- **model**: SMOTE ile dengeleme, sınıf ağırlıkları, isteğe bağlı özellik seçimi ve grid search; XGBoost/LightGBM/Linear SVM desteği; JSON metadata ve görsellerle raporlama; tek veya toplu tahmin.

## Ayarlar

- `goruntu_isleme/ayarlar.py`: hedef boyut, normalizasyon stratejisi, bias correction, skull stripping, registration, augmentasyon ve ölçekleme yöntemi (`SCALING_METODU`).
- `model/ayarlar.py`: veri yolları, train/val/test oranları, model hiperparametreleri, grid search parametreleri, log ve çıktı yolları.

## Testler

Testleri çalıştırmak için ana dizinde:
```bash
pytest
```
Belirli modüller için:
```bash
pytest tests/test_goruntu_isleyici.py
pytest tests/test_model_egitici.py
```

## Çıktılar

- `goruntu_isleme/cikti/`: İşlenmiş görüntüler, ham ve ölçekli özellik CSV'leri, stratified bölünmüş `egitim.csv`/`dogrulama.csv`/`test.csv`.
- `model/ciktilar/`: Eğitilmiş modeller (`.pkl`), metadata (`.json`), raporlar ve değerlendirme görselleri.
- `eda_analiz/eda_ciktilar/`: EDA grafikleri ve özet CSV.

## Lisans

MIT lisansı için `LICENSE` dosyasına bakın.
