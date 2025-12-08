"""
ayarlar.py
----------
JPEG (ve benzeri 2B) MRI görüntüleri için EDA ayarları.
"""

# Veri seti klasörü - sınıf klasörleri: NonDemented, VeryMildDemented, MildDemented, ModerateDemented
VERI_KLASORU = "../../Veri_Seti"

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

# Tüm grafiklerin kaydedileceği çıktı klasörü
CIKTI_KLASORU = "eda_ciktlari"

# Tekrarlanabilirlik için rastgelelik tohumu
RASTGELE_TOHUM = 42

# Global yoğunluk histogramları için kaç görüntü kullanılacak
N_GORSELLER_YOGUNLUK_ORNEGI = 40

# Her görüntüden kaç piksel örneklenecek
N_PIKSEL_ORNEK_SAYISI = 5000

# PCA / t-SNE gömme için maksimum kaç görüntü kullanılacak
N_GORSELLER_GOMULEME = 200

# Sınıf etiketlerini daha anlamlı isimlere çevirmek istersen:
ETIKET_ISIM_HARITASI = {
    0: "NonDemented",
    1: "VeryMildDemented",
    2: "MildDemented",
    3: "ModerateDemented",
}

# Görüntüleri gri tonlamaya mı çevirelim? (Önerilen: True)
GORSELU_GRIDE_CEV = True
