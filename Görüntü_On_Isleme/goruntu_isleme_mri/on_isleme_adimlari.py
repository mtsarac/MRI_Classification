"""
on_isleme_adimlari.py
---------------------
Tek bir MRI görüntüsü üzerinde uygulanacak ön işleme adımlarını içerir:
- Arka plan tespiti
- Maske oluşturma ve kırpma
- Yoğunluk normalizasyonu
- Gelişmiş gürültü azaltma filtreleri (bilateral, NLM, unsharp masking)
- Histogram eşitleme (isteğe bağlı)
- Yeniden boyutlandırma
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Optional
from skimage import exposure

from .ayarlar import (
    HEDEF_GENISLIK,
    HEDEF_YUKSEKLIK,
    KIRPMA_YUZDELERI,
    HISTOGRAM_ESITLEME_AKTIF,
    CLAHE_CLIP_LIMIT,
    MASKE_KENAR_PAYI,
    GELISMIS_FILTRE_AKTIF,
    FILTRE_KALITESI,
    Z_SCORE_NORMALIZASYON_AKTIF,
    GAUSSIAN_BLUR_AKTIF,
    GAUSSIAN_BLUR_SIGMA,
    MORFOLOJIK_OPERASYONLAR_AKTIF,
    MORFOLOJIK_KERNEL_BOYUTU,
    KENAR_TESPITI_AKTIF,
    CANNY_ESIK1,
    CANNY_ESIK2,
    ROBUST_NORMALIZASYON_AKTIF,
    ROBUST_NORMALIZASYON_METODU,
)
from .arka_plan_isleme import (
    arka_plan_tipi_belirle,
    ikili_maske_olustur,
    maske_sinir_kutusu_bul,
    maske_sinir_kutusunu_genislet,
)
from .gelismis_filtreler import GelismisFiltreler, otomatik_filtre_uygulamasi


def yogunluk_normalize_et(goruntu: np.ndarray, yuzdelikler: Tuple[int, int] = (1, 99)) -> np.ndarray:
    """
    MRI görüntüsünün yoğunluğunu normalize eder:
    - Belirtilen yüzdeliklere göre alt/üst kırpma (örn: %1 ve %99)
    - 0-255 aralığına yeniden ölçekleme

    Parametreler:
    - goruntu: Input görüntü (np.ndarray)
    - yuzdelikler: Alt ve üst yüzdelik değerleri (tuple)
    
    Çıktı: uint8 [0, 255] şeklinde normalize edilmiş görüntü.
    
    Raises:
        ValueError: Geçersiz input görüntü veya yüzdelik değerleri
    """
    try:
        if goruntu is None or goruntu.size == 0:
            raise ValueError("Geçersiz görüntü: Boş veya None")
        
        if len(yuzdelikler) != 2 or not (0 <= yuzdelikler[0] < yuzdelikler[1] <= 100):
            raise ValueError("Yüzdelik değerleri 0-100 arasında ve artan sırada olmalı")
        
        alt_yuzde, ust_yuzde = yuzdelikler
        flat = goruntu.flatten()
        alt_deger, ust_deger = np.percentile(flat, [alt_yuzde, ust_yuzde])

        goruntu_kirp = np.clip(goruntu, alt_deger, ust_deger)
        if ust_deger - alt_deger < 1e-6:
            norm = np.zeros_like(goruntu_kirp)
        else:
            norm = (goruntu_kirp - alt_deger) / (ust_deger - alt_deger)

        norm = (norm * 255.0).astype("uint8")
        return norm
    except Exception as e:
        print(f"[HATA] yogunluk_normalize_et içinde hata: {str(e)}")
        raise


def histogram_esit(goruntu_uint8: np.ndarray) -> np.ndarray:
    """
    Adaptif histogram eşitleme (CLAHE benzeri) uygular.
    skimage.exposure.equalize_adapthist kullanır.

    Girdi: uint8 [0,255] görüntü
    Çıktı: uint8 [0,255] görüntü
    
    Raises:
        ValueError: Geçersiz input görüntü
    """
    try:
        if goruntu_uint8 is None or goruntu_uint8.size == 0:
            raise ValueError("Geçersiz görüntü: Boş veya None")
        
        # 0-1 aralığına getir
        goruntu_float = goruntu_uint8.astype("float32") / 255.0
        goruntu_eq = exposure.equalize_adapthist(goruntu_float, clip_limit=CLAHE_CLIP_LIMIT)
        goruntu_eq_uint8 = (goruntu_eq * 255.0).clip(0, 255).astype("uint8")
        return goruntu_eq_uint8
    except Exception as e:
        print(f"[HATA] histogram_esit içinde hata: {str(e)}")
        raise


def boyutu_degistir(goruntu_uint8: np.ndarray, genislik: int = HEDEF_GENISLIK, yukseklik: int = HEDEF_YUKSEKLIK) -> np.ndarray:
    """
    Görüntüyü hedef boyuta yeniden boyutlandırır (bilineer interpolasyon).
    
    Parametreler:
    - goruntu_uint8: Input görüntü
    - genislik: Hedef genişlik (pixel)
    - yukseklik: Hedef yükseklik (pixel)
    
    Raises:
        ValueError: Geçersiz input veya boyut
    """
    try:
        if goruntu_uint8 is None or goruntu_uint8.size == 0:
            raise ValueError("Geçersiz görüntü: Boş veya None")
        
        if genislik <= 0 or yukseklik <= 0:
            raise ValueError("Genişlik ve yükseklik pozitif olmalı")
        
        hedef_boyut = (genislik, yukseklik)
        # cv2.resize boyut parametresi (width, height) sırasıyla alır
        yeniden = cv2.resize(goruntu_uint8, hedef_boyut, interpolation=cv2.INTER_LINEAR)
        return yeniden
    except Exception as e:
        print(f"[HATA] boyutu_degistir içinde hata: {str(e)}")
        raise


def z_score_normalizasyon(goruntu: np.ndarray) -> np.ndarray:
    """
    Z-score normalizasyonu uygular (mean=0, std=1).
    
    Girdi: uint8 [0,255] görüntü
    Çıktı: float32 [-N, N] aralığında Z-score normalize edilmiş görüntü
    
    Raises:
        ValueError: Geçersiz input görüntü
    """
    try:
        if goruntu is None or goruntu.size == 0:
            raise ValueError("Geçersiz görüntü: Boş veya None")
        
        goruntu_float = goruntu.astype("float32")
        ortalama = goruntu_float.mean()
        standart_sapma = goruntu_float.std()
        
        if standart_sapma < 1e-6:
            # Tüm değerler aynı ise, sıfır döndür
            return np.zeros_like(goruntu_float)
        
        z_score = (goruntu_float - ortalama) / standart_sapma
        return z_score
    except Exception as e:
        print(f"[HATA] z_score_normalizasyon içinde hata: {str(e)}")
        raise


def robust_normalizasyon(goruntu: np.ndarray, metodu: str = "iqr") -> np.ndarray:
    """
    Robust normalizasyon - outlier'lara karşı dayanıklı.
    
    Parametreler:
    - goruntu: Input görüntü
    - metodu: "percentile", "iqr" (Interquartile Range), "mad" (Median Absolute Deviation)
    
    Çıktı: uint8 [0,255] normalize edilmiş görüntü
    
    Raises:
        ValueError: Geçersiz input veya normalizasyon metodu
    """
    try:
        if goruntu is None or goruntu.size == 0:
            raise ValueError("Geçersiz görüntü: Boş veya None")
        
        valid_methods = ["iqr", "mad", "percentile"]
        if metodu not in valid_methods:
            print(f"[UYARI] Bilinmeyen normalizasyon metodu '{metodu}'. Varsayılan 'iqr' kullanılıyor.")
            metodu = "iqr"
        
        goruntu_float = goruntu.astype("float32")
        
        if metodu == "iqr":
            # IQR yöntemi - outlier'ları kırpma
            Q1 = np.percentile(goruntu_float, 25)
            Q3 = np.percentile(goruntu_float, 75)
            IQR = Q3 - Q1
            
            alt_sinir = Q1 - 1.5 * IQR
            ust_sinir = Q3 + 1.5 * IQR
            
            goruntu_kirp = np.clip(goruntu_float, alt_sinir, ust_sinir)
            
        elif metodu == "mad":
            # MAD yöntemi - Median Absolute Deviation
            medyan = np.median(goruntu_float)
            sapma = np.abs(goruntu_float - medyan)
            mad = np.median(sapma)
            
            if mad < 1e-6:
                return np.clip(goruntu_float, 0, 255).astype("uint8")
            
            alt_sinir = medyan - 3 * mad
            ust_sinir = medyan + 3 * mad
            
            goruntu_kirp = np.clip(goruntu_float, alt_sinir, ust_sinir)
            
        else:  # percentile (default)
            # Percentile yöntemi - %2 ve %98 percentile
            alt_deger = np.percentile(goruntu_float, 2)
            ust_deger = np.percentile(goruntu_float, 98)
            
            goruntu_kirp = np.clip(goruntu_float, alt_deger, ust_deger)
        
        # 0-255 aralığına ölçekle
        min_val = goruntu_kirp.min()
        max_val = goruntu_kirp.max()
        
        if max_val - min_val < 1:
            return np.zeros_like(goruntu_kirp).astype("uint8")
        
        normalized = (goruntu_kirp - min_val) / (max_val - min_val) * 255.0
        return normalized.astype("uint8")
    except Exception as e:
        print(f"[HATA] robust_normalizasyon içinde hata: {str(e)}")
        raise


def gaussian_bulaniklastir(goruntu: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """
    Hafif Gaussian bulanıklaştırma - pre-filtering.
    
    Parametreler:
    - goruntu: Input görüntü
    - sigma: Standart sapma (düşük = hafif blur)
    
    Çıktı: Bulanıklaştırılmış görüntü (aynı tip)
    
    Raises:
        ValueError: Geçersiz input veya sigma değeri
    """
    try:
        from scipy import ndimage
        
        if goruntu is None or goruntu.size == 0:
            raise ValueError("Geçersiz görüntü: Boş veya None")
        
        if sigma < 0:
            raise ValueError("Sigma değeri negatif olamaz")
        
        goruntu_float = goruntu.astype("float32")
        bulanik = ndimage.gaussian_filter(goruntu_float, sigma=sigma)
        return bulanik.astype(goruntu.dtype)
    except Exception as e:
        print(f"[HATA] gaussian_bulaniklastir içinde hata: {str(e)}")
        raise


def morfolojik_islemler(goruntu: np.ndarray, kernel_boyutu: int = 3) -> np.ndarray:
    """
    Morfolojik açılış (opening) işlemi - ufak gürültüyü temizleme.
    
    Parametreler:
    - goruntu: Input görüntü
    - kernel_boyutu: Yapısal eleman boyutu (pozitif tek sayı)
    
    Çıktı: İşlenmiş görüntü (uint8)
    
    Raises:
        ValueError: Geçersiz input veya kernel boyutu
    """
    try:
        if goruntu is None or goruntu.size == 0:
            raise ValueError("Geçersiz görüntü: Boş veya None")
        
        if kernel_boyutu <= 0 or kernel_boyutu % 2 == 0:
            raise ValueError("Kernel boyutu pozitif tek sayı olmalı")
        
        goruntu_uint8 = np.clip(goruntu, 0, 255).astype("uint8")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_boyutu, kernel_boyutu))
        
        # Opening: Erosion -> Dilation (gürültü temizleme)
        acilis = cv2.morphologyEx(goruntu_uint8, cv2.MORPH_OPEN, kernel)
        
        return acilis
    except Exception as e:
        print(f"[HATA] morfolojik_islemler içinde hata: {str(e)}")
        raise


def canny_kenar_haritasi(goruntu: np.ndarray, esik1: int = 100, esik2: int = 200) -> np.ndarray:
    """
    Canny kenar dedektörü - ilgi bölgesi analizi için.
    
    Parametreler:
    - goruntu: Input görüntü
    - esik1: Düşük eşik (0-255)
    - esik2: Yüksek eşik (0-255, esik1 < esik2 olmalı)
    
    Çıktı: Binary kenar haritası (0 veya 255)
    
    Raises:
        ValueError: Geçersiz input veya eşik değerleri
    """
    try:
        if goruntu is None or goruntu.size == 0:
            raise ValueError("Geçersiz görüntü: Boş veya None")
        
        if not (0 <= esik1 < esik2 <= 255):
            raise ValueError("Eşik değerleri: 0 <= esik1 < esik2 <= 255 olmalı")
        
        goruntu_uint8 = np.clip(goruntu, 0, 255).astype("uint8")
        
        # Canny kenar dedeksiyonu
        kenarlar = cv2.Canny(goruntu_uint8, esik1, esik2)
        
        return kenarlar
    except Exception as e:
        print(f"[HATA] canny_kenar_haritasi içinde hata: {str(e)}")
        raise


def tek_goruntu_on_isle(goruntu_gri: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Tek bir gri tonlamalı MRI görüntüsüne tüm ön işleme adımlarını uygular.
    
    Parametreler:
    - goruntu_gri: float32 [0,255] veya benzeri aralıkta 2B numpy dizisi

    Döndürülen:
    - on_islenmis_goruntu: uint8 [0,255] veya float32, (HEDEF_YUKSEKLIK, HEDEF_GENISLIK)
    - meta_bilgi: dict - İşleme ile ilgili metadata
    
    Raises:
        ValueError: Geçersiz input görüntü
        Exception: İşleme adımlarından herhangi birinde hata
    """
    try:
        if goruntu_gri is None or goruntu_gri.size == 0:
            raise ValueError("Geçersiz görüntü: Boş veya None")
        
        if len(goruntu_gri.shape) != 2:
            raise ValueError("Görüntü 2B olmalı (gri tonlamalı)")
        
        orijinal_h, orijinal_w = goruntu_gri.shape

        # 0) Pre-filtering: Hafif Gaussian Blur (opsiyonel)
        if GAUSSIAN_BLUR_AKTIF:
            goruntu_gri = gaussian_bulaniklastir(goruntu_gri, sigma=GAUSSIAN_BLUR_SIGMA)

        # 1) Arka plan tipini belirle (bilgi amaçlı)
        arka_plan_tipi = arka_plan_tipi_belirle(goruntu_gri)

        # 2) Maske oluştur
        maske = ikili_maske_olustur(goruntu_gri)

        # 3) Maske sınır kutusunu bul ve kenar payı ile genişlet
        sinir_kutusu = maske_sinir_kutusu_bul(maske)
        sinir_kutusu_genis = maske_sinir_kutusunu_genislet(sinir_kutusu, goruntu_gri.shape, MASKE_KENAR_PAYI)

        if sinir_kutusu_genis is not None:
            y_min, y_max, x_min, x_max = sinir_kutusu_genis
            kirpilmis = goruntu_gri[y_min:y_max, x_min:x_max]
        else:
            # Maske boş ise görüntüyü kırpmadan kullan
            y_min, y_max, x_min, x_max = 0, orijinal_h, 0, orijinal_w
            kirpilmis = goruntu_gri

        # 4) Robust yoğunluk normalizasyonu (outlier'lara karşı dayanıklı)
        if ROBUST_NORMALIZASYON_AKTIF:
            norm = robust_normalizasyon(kirpilmis, metodu=ROBUST_NORMALIZASYON_METODU)
        else:
            norm = yogunluk_normalize_et(kirpilmis, yuzdelikler=KIRPMA_YUZDELERI)

        # 4.5) Morfolojik işlemler (ufak gürültüyü temizleme)
        if MORFOLOJIK_OPERASYONLAR_AKTIF:
            norm = morfolojik_islemler(norm, kernel_boyutu=MORFOLOJIK_KERNEL_BOYUTU)

        # 5) Gelişmiş filtreler (isteğe bağlı)
        if GELISMIS_FILTRE_AKTIF:
            norm = otomatik_filtre_uygulamasi(norm, kalite=FILTRE_KALITESI)

        # 5.5) Histogram eşitleme (isteğe bağlı)
        if HISTOGRAM_ESITLEME_AKTIF:
            norm = histogram_esit(norm)

        # 6) Hedef boyuta yeniden boyutlandırma
        yeniden = boyutu_degistir(norm, HEDEF_GENISLIK, HEDEF_YUKSEKLIK)
        
        # 6.5) Kenar tespiti (isteğe bağlı - verbose bilgi)
        kenar_haritasi = None
        if KENAR_TESPITI_AKTIF:
            kenar_haritasi = canny_kenar_haritasi(yeniden, esik1=CANNY_ESIK1, esik2=CANNY_ESIK2)
        
        # 7) Z-score normalizasyonu (isteğe bağlı)
        if Z_SCORE_NORMALIZASYON_AKTIF:
            yeniden = z_score_normalizasyon(yeniden)
        
        # Z-score yapılmışsa uint8 döndürme; aksi takdirde uint8 döndür
        if not Z_SCORE_NORMALIZASYON_AKTIF:
            yeniden = yeniden.astype("uint8")

        meta_bilgi = {
            "orijinal_genislik": int(orijinal_w),
            "orijinal_yukseklik": int(orijinal_h),
            "kirp_y_min": int(y_min),
            "kirp_y_max": int(y_max),
            "kirp_x_min": int(x_min),
            "kirp_x_max": int(x_max),
            "arka_plan_tipi": arka_plan_tipi,
            "gaussian_blur_aktif": GAUSSIAN_BLUR_AKTIF,
            "morfolojik_aktif": MORFOLOJIK_OPERASYONLAR_AKTIF,
            "robust_normalizasyon_aktif": ROBUST_NORMALIZASYON_AKTIF,
            "robust_metodu": ROBUST_NORMALIZASYON_METODU if ROBUST_NORMALIZASYON_AKTIF else "N/A",
            "gelismis_filtre_aktif": GELISMIS_FILTRE_AKTIF,
            "filtre_kalitesi": FILTRE_KALITESI if GELISMIS_FILTRE_AKTIF else "N/A",
            "kenar_tespiti_aktif": KENAR_TESPITI_AKTIF,
            "z_score_aktif": Z_SCORE_NORMALIZASYON_AKTIF,
        }

        return yeniden, meta_bilgi
    
    except Exception as e:
        print(f"[HATA] tek_goruntu_on_isle içinde hata: {str(e)}")
        raise
