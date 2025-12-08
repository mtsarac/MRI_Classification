"""
ayarlar.py
----------
MRI JPEG/PNG görüntülerinin model eğitimi öncesi ön işlenmesi için temel ayarlar.
Bu dosyadaki değerleri kendi proje yapına göre rahatça değiştirebilirsin.
"""

# Girdi ve çıktı klasörleri
# Veri seti klasörleri: NonDemented, VeryMildDemented, MildDemented, ModerateDemented
GIRDİ_KLASORU = "../Veri_Seti"   # Ham (orijinal) görüntülerin olduğu klasör
CIKTI_KLASORU = "veri/cikti"   # Ön işlenmiş görüntülerin kaydedileceği klasör

# Sınıf klasörleri ve etiketleri
SINIF_KLASORLERI = [
    "NonDemented",       # Etiket: 0
    "VeryMildDemented",  # Etiket: 1
    "MildDemented",      # Etiket: 2
    "ModerateDemented",  # Etiket: 3
]

SINIF_ETIKETI = {
    "NonDemented": 0,
    "VeryMildDemented": 1,
    "MildDemented": 2,
    "ModerateDemented": 3,
}

# İzin verilen görüntü uzantıları
GORUNTU_UZANTILARI = [".jpg", ".jpeg", ".png"]

# Sınıf haritası (isteğe bağlı - daha anlamlı isimler için)
ETIKET_ISIM_HARITASI = {
    0: "NonDemented",
    1: "VeryMildDemented",
    2: "MildDemented",
    3: "ModerateDemented",
}
HEDEF_GENISLIK = 256
HEDEF_YUKSEKLIK = 256

# Yoğunluk normalizasyonu için kullanılacak yüzdelikler (alt, üst)
# Örn: (1, 99) => en düşük %1 ve en yüksek %1 uç değerleri kırp
KIRPMA_YUZDELERI = (1, 99)

# Z-score normalizasyonu (mean=0, std=1) aktif mi?
# Eğer True ise, ön işlenmiş görüntüler Z-score normalize edilecek
Z_SCORE_NORMALIZASYON_AKTIF = True

# Adaptif histogram eşitleme (CLAHE benzeri) kullanılsın mı?
HISTOGRAM_ESITLEME_AKTIF = True
# CLAHE clip_limit değeri (daha yüksek = daha güçlü kontrast)
CLAHE_CLIP_LIMIT = 0.03

# Maskeden kırpma yaparken etrafına eklenecek kenar payı (piksel cinsinden)
MASKE_KENAR_PAYI = 5

# Rastgelelik için sabit tohum (reproducible olsun diye)
RASTGELE_TOHUM = 42

# Veri artırma (augmentation) ayarları
VERI_ARTIRMA_AKTIF = True
# Her orijinal görüntü için kaç ekstra artırılmış kopya üretilecek
EKSTRA_KOPYA_SAYISI = 2

# Sınıf dengesizliğine karşı koşullu augmentation
# ModerateDemented sınıfı için daha fazla augmentation kopyası üret
SINIF_DENGELEME_AKTIF = True
SINIF_AUGMENTATION_SAYILARI = {
    "NonDemented": 2,          # Standart: 2 kopya
    "VeryMildDemented": 2,     # Standart: 2 kopya
    "MildDemented": 2,         # Standart: 2 kopya
    "ModerateDemented": 5,     # Az temsil edilen sınıf: 5 kopya (dengeleme için)
}

# Gelişmiş filtre ayarları
GELISMIS_FILTRE_AKTIF = True
# Filtre kalite seviyesi: "düşük", "standart", "yüksek"
FILTRE_KALITESI = "standart"
# Non-lokal ortalama gücü (h parametresi)
NLM_GUCU = 10.0
# Bilateral filtre sigma değerleri
BILATERAL_SIGMA_RENK = 75.0
BILATERAL_SIGMA_MEKAN = 75.0
# Keskinlik artırma (unsharp masking) gücü
KESKINLIK_GUCU = 1.0

# Pre-filtering: Gaussian Blur ayarları
GAUSSIAN_BLUR_AKTIF = True
GAUSSIAN_BLUR_SIGMA = 0.5  # Çok hafif blur (yapı korunur)

# Morfolojik işlemler
MORFOLOJIK_OPERASYONLAR_AKTIF = True
MORFOLOJIK_KERNEL_BOYUTU = 3  # Küçük kernel (detay koruma)

# Kenar tespiti (Canny Edge Detection)
KENAR_TESPITI_AKTIF = False  # İsteğe bağlı - verbose output için
CANNY_ESIK1 = 100
CANNY_ESIK2 = 200

# Şiddetli outlier'lar için robust normalizasyon
ROBUST_NORMALIZASYON_AKTIF = True
# Robust normalizasyon yöntemi: "percentile", "iqr", "mad"
ROBUST_NORMALIZASYON_METODU = "iqr"