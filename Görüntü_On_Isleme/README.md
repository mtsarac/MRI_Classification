# MRI Görüntü Ön İşleme Projesi (JPEG/PNG)

Bu proje, siyah veya gri arka plana sahip 2B MRI görüntülerini (JPEG/PNG)
**derin öğrenme / makine öğrenmesi model eğitimi için en iyi şekilde hazırlamak**
amacıyla tasarlanmıştır.

Ana hedef:
- Arka planı (siyah veya gri) mümkün olduğunca temizleyip
- Sadece ilgi bölgesini (vücut dokusu) içeren, normalize edilmiş,
  tekdüze boyutlu görüntüler üretmek
- İsteğe bağlı olarak veri artırma (augmentation) ile modele daha zengin bir eğitim seti sağlamak

## Klasör Yapısı

Önerilen yapı:

```text
mri_on_isleme_projesi/
  goruntu_isleme_mri/
    __init__.py
    ayarlar.py
    io_araclari.py
    arka_plan_isleme.py
    on_isleme_adimlari.py
    artirma.py
  scripts/
    toplu_on_isleme.py
    tek_goruntuyu_incele.py
  veri/
    girdi/
      ... burada ham (orijinal) MRI JPEG/PNG dosyaların ...
    cikti/
      ... otomatik oluşturulacak, ön işlenmiş görüntüler ...
  requirements.txt
  README.md
```

Sen sadece:
- Ham görüntülerini `veri/girdi/` klasörüne koy
- Gerekirse `goruntu_isleme_mri/ayarlar.py` dosyasındaki ayarları düzenle
- Aşağıdaki komutları çalıştır

## Kurulum

1. (Opsiyonel ama önerilir) Sanal ortam oluştur:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. Gerekli paketleri kur:

   ```bash
   pip install -r requirements.txt
   ```

## Toplu Ön İşleme

Tüm görüntüleri ön işlemek için:

```bash
python scripts/toplu_on_isleme.py
```

Bu script:

- `veri/girdi/` içindeki tüm `.jpg`, `.jpeg`, `.png` dosyalarını bulur
- Her biri için:
  - Arka plan tipini tahmin eder (siyah/gri/diger)
  - Otsu eşiği ile ikili maske çıkarır
  - Maskeye göre sınır kutusunu bulup kenar payı ile genişletir
  - Bu sınır kutusuna göre görüntüyü kırpar
  - Yoğunluk normalizasyonu yapar (örn. %1-%99 arasında kırpıp 0-255 aralığına çeker)
  - **Gelişmiş filtreleri uygular** (gürültü azaltma, keskinlik artırma)
  - İsteğe bağlı CLAHE benzeri adaptif histogram eşitleme uygular
  - Sabit boyuta (örn. 256x256) yeniden boyutlandırır
  - Sonucu `veri/cikti/` altına kaydeder (klasör yapısı korunur)
- `veri/cikti/on_isleme_log.csv` dosyasında tüm adımların özetini tutar
  (orijinal boyutlar, kırpma kutusu, arka plan tipi vb.)

Ayrıca `ayarlar.py` içinde `VERI_ARTIRMA_AKTIF = True` ise,
her görüntü için belirttiğin sayıda (`EKSTRA_KOPYA_SAYISI`) artırılmış kopya üretir:
- Yatay/dikey ayna
- 90/180/270 derece döndürme
- Küçük parlaklık/kontrast oynamaları

## Tek Görüntüyü İnceleme

Bir görüntü üzerinde ön işleme adımlarının etkisini hızlıca görmek için:

```bash
python scripts/tek_goruntuyu_incele.py veri/girdi/ornek.jpg
```

Bu komut:
- Orijinal ve ön işlenmiş görüntüyü yan yana gösterir
- Konsola kırpma kutusu ve arka plan tipi gibi meta bilgileri yazar

## Ön İşlenmiş Verileri Sınıflara Göre Dosyalama

Ön işleme tamamlandıktan sonra, görüntüleri sınıf klasörlerine organize etmek ve
eğitim/doğrulama/test setlerine ayırmak için:

```bash
python scripts/verileri_dosyala.py
```

Bu script:

1. **Hızlı Dosyalama (Önerilen):**
   - Ön işlenmiş görüntüleri otomatik sınıf klasörlerine taşır
   - Eğitim/doğrulama/test setlerine özelleştirilebilir oranlarda böler
   - Veri seti bilgisini JSON dosyasına kaydeder

2. **Adım Adım Dosyalama:**
   - Her adımı ayrı ayrı gerçekleştirme seçeneği
   - Daha fazla kontrol ve esneklik sağlar

3. **Mevcut Veri Setini Yeniden Organize Etme:**
   - Zaten sınıf klasörlerine dosyalanmış verileri yeniden yapılandırma

**Çıktı Yapısı:**
```text
veri/veri_seti/
  eğitim/
    NonDemented/          # 70% (varsayılan)
    VeryMildDemented/
    MildDemented/
    ModerateDemented/
  doğrulama/
    NonDemented/          # 15% (varsayılan)
    ...
  test/
    NonDemented/          # 15% (varsayılan)
    ...
  tüm_veriler/            # Bölüntüden önce
    NonDemented/
    ...
  veri_seti_bilgisi.json  # Meta veriler
```

## Veri Seti Doğrulama ve İstatistikler

Oluşturulan veri setinin bütünlüğünü kontrol etmek ve istatistiklerini görmek için:

```bash
python scripts/veri_seti_kontrol_et.py
```

Bu script:

1. **Veri Seti İstatistikleri:** Her bölümdeki dosya sayısını gösterir
2. **Doğrulama Raporu:** Sorunları ve uyarıları tespit eder
3. **Klasör Boyutları:** Disk kullanımını gösterir
4. **Dosya İstatistikleri:** Uzantı dağılımını ve ayrıntıları gösterir

## Ayarları Özelleştirme

`goruntu_isleme_mri/ayarlar.py` dosyasında:

- `HEDEF_GENISLIK`, `HEDEF_YUKSEKLIK` → modeline uygun hedef boyut
- `KIRPMA_YUZDELERI` → yoğunluk normalizasyonunda uç değer kırpma
- `HISTOGRAM_ESITLEME_AKTIF` → CLAHE kullanmak isteyip istemediğin
- `MASKE_KENAR_PAYI` → kırpma kutusunun etrafına eklenilecek güvenlik payı
- `VERI_ARTIRMA_AKTIF`, `EKSTRA_KOPYA_SAYISI` → veri artırma ayarları
- `GELISMIS_FILTRE_AKTIF` → gelişmiş filtreleri etkinleştir/devre dışı bırak
- `FILTRE_KALITESI` → "düşük", "standart", "yüksek" arasında seç

gibi parametreleri kolayca değiştirebilirsin.

## Gelişmiş Filtreler

Proje artık şu gelişmiş filtreler içermektedir:

### Gürültü Azaltma:
- **Non-Lokal Ortalama (NLM)** - Yüksek kaliteli gürültü azaltma
- **Bilateral Filtre** - Kenarları koruyarak gürültü azaltma
- **Medyan Filtresi** - Tuz-biber gürültüne karşı etkili
- **Gauss Bulanıklaştırma** - Hafif gürültü azaltma

### Kontrast Iyileştirme:
- **Adaptif Histogram Eşitleme (CLAHE)** - Yerel kontrast iyileştirme
- **Kontrast Uzatma** - Histogramın tüm aralığını kullanma
- **Unsharp Masking** - Keskinlik artırma

### Kenar Tespiti:
- **Canny Kenar Dedektörü** - Yüksek kaliteli kenar tespiti
- **Sobel Filtreleri** - X ve Y yönlü kenarlar
- **Laplacian Filtresi** - İkinci derece kenar tespiti

### Morfolojik İşlemler:
- **Morfolojik Kapanış** - Küçük boşlukları doldurma
- **Morfolojik Açılış** - Küçük objeleri temizleme
- **Top-Hat Filtresi** - Küçük yapıları vurgulama
- **Black-Hat Filtresi** - Gölgeleri vurgulama

## Filtreleri Test Etme

Farklı filtrelerin ve kombinasyonlarının etkisini görmek için:

```bash
python scripts/filtre_test_et.py veri/girdi/ornek.jpg
```

Bu komut 3x4 bir grid'de gösterir:
- **Satır 1**: Orijinal, NLM, Bilateral, Medyan filtreleri
- **Satır 2**: Canny, Sobel, Laplacian kenarları ve otomatik kombinasyon
- **Satır 3**: Kontrast uzatma, unsharp masking, histogram ve istatistikler

## Notlar

- Arka plan tespiti, görüntülerin çoğunda siyah arka plan olduğu varsayımıyla tasarlanmıştır.
  Yine de kenar piksellere bakarak gri arka planı da ayırt etmeye çalışır.
- Otsu eşiği ile maske çıkarımında, ilgi bölgesinin arka plandan **daha parlak** olduğu
  senaryo varsayılmıştır. Eğer senin verinde tam tersi durum baskınsa,
  `arka_plan_isleme.py` içindeki `ikili_maske_olustur` fonksiyonunda
  `goruntu > esik` yerine `goruntu < esik` kullanabilirsin.
- Proje tamamen modüler yazılmıştır; ihtiyaç duyduğun kısma kolayca müdahale edebilirsin.
