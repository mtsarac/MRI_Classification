# Görüntü İşleme Modülü

MRI görüntülerini temizler, hizalar, normalleştirir, augmentasyon uygular ve model eğitimine hazır özellik CSV'leri üretir. İşlem adımları bias field correction (SimpleITK varsa N4ITK), skull stripping, hizalama, adaptif CLAHE, z-score normalizasyonu ve sınıf bazlı augmentasyonu içerir. Toplu işler için çok çekirdek kullanılır.

## Kurulum

Ana dizinden bağımlılıkları yükleyin:
```bash
pip install -r ../requirements.txt
```
Ham veri klasörünüzün `../Veri_Seti/<sınıf_adı>/` yapısında olduğundan emin olun.

## Kullanım

1) **Hızlı kontrol (önerilir)**  
```bash
python pipeline_quick_test.py
```
Paketleri ve veri dizinini doğrular.

2) **Ana menü**  
```bash
python ana_islem.py
```
Menü seçenekleri:
- 1: Görüntüleri ön işle (kalite kontrol, bias correction, skull stripping, hizalama, normalizasyon, yeniden boyutlandırma, augmentasyon)
- 2: Özellik çıkar ve `goruntu_ozellikleri.csv` oluştur
- 3: CSV'deki NaN değerlerini temizle
- 4: CSV'ye ölçeklendirme uygula (`SCALING_METODU`: minmax/robust/standard/maxabs)
- 5: Özellik CSV'yi train/val/test olarak böl (`egitim.csv`, `dogrulama.csv`, `test.csv`)
- 6: İstatistik raporu göster (CSV özetleri)
- 7: Tüm adımları sırayla çalıştır (önerilen)

3) **Tek görüntü testi**  
```bash
python test_pipeline.py /path/to/image.jpg   # Argüman vermezseniz örnek akışı dener
```

## Çıktılar

`cikti/` altında oluşanlar:
- İşlenmiş görüntüler (sınıf klasörlerinde `.png`)
- `goruntu_ozellikleri.csv` (ham özellikler)
- `goruntu_ozellikleri_scaled.csv` (seçilen ölçekleme ile)
- `egitim.csv`, `dogrulama.csv`, `test.csv` (taban CSV'den stratified split)

## Ayarlar

`ayarlar.py` içinden başlıca kontroller:
- Görüntü boyutu: `HEDEF_GENISLIK`, `HEDEF_YUKSEKLIK`
- Normalizasyon stratejisi: `NORMALIZASYON_STRATEJISI` (`minimal/standard/aggressive`)
- Bias correction, skull stripping, registration anahtarları
- Augmentasyon açık/kapalı (`VERI_ARTIRMA_AKTIF`) ve sınıf bazlı çarpanlar (`SINIF_BAZLI_CARPANLAR`)
- Özellik ölçekleme metodu: `SCALING_METODU`

## İpuçları

- SimpleITK kurulu değilse bias correction basit metoda düşer; kurulumda uyarı görürsünüz.  
- İşlem süresi için CPU çekirdekleri otomatik ayarlanır; çekirdek sayısını `GorselIsleyici.n_jobs` üzerinde değiştirebilirsiniz.  
- CSV'lerde NaN oluşursa önce 3. adımı, ardından 4. adımı çalıştırın.
