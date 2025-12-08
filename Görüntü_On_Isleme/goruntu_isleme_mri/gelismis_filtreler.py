"""
gelismis_filtreler.py
---------------------
MRI görüntüleri için gelişmiş filtre işlemleri.
Gürültü azaltma, kontrast iyileştirme ve kenar tespiti gibi işlemler.
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage import filters, exposure


class GelismisFiltreler:
    """MRI görüntüleri için filtre işlemleri."""
    
    @staticmethod
    def bilateral_filtre(goruntu: np.ndarray, 
                        cerceve_eni: int = 9,
                        sigma_renk: float = 75.0,
                        sigma_mekan: float = 75.0) -> np.ndarray:
        """
        Bilateral filtre - kenarları koruyarak gürültü azaltma.
        
        Parametreler:
        - cerceve_eni: Filtre çerçevesi boyutu (tek sayı)
        - sigma_renk: Renk uzayındaki standart sapma
        - sigma_mekan: Mekan uzayındaki standart sapma
        
        Çıktı: Filtrele yapılmış uint8 görüntü
        """
        goruntu_uint8 = np.clip(goruntu, 0, 255).astype(np.uint8)
        
        filtreli = cv2.bilateralFilter(
            goruntu_uint8,
            d=cerceve_eni,
            sigmaColor=sigma_renk,
            sigmaSpace=sigma_mekan
        )
        return filtreli.astype(np.float32)
    
    @staticmethod
    def non_lokal_ortalama(goruntu: np.ndarray,
                           h: float = 10.0,
                           template_penceresi: int = 7,
                           arama_penceresi: int = 21) -> np.ndarray:
        """
        Non-Lokal Ortalama (NLM) - yüksek kaliteli gürültü azaltma.
        
        Parametreler:
        - h: Filtre gücü (yüksek = daha fazla gürültü azaltma)
        - template_penceresi: Şablon penceresi boyutu
        - arama_penceresi: Arama alanı boyutu
        
        Çıktı: Filtrele yapılmış uint8 görüntü
        """
        goruntu_uint8 = np.clip(goruntu, 0, 255).astype(np.uint8)
        
        filtreli = cv2.fastNlMeansDenoising(
            goruntu_uint8,
            h=h,
            templateWindowSize=template_penceresi,
            searchWindowSize=arama_penceresi
        )
        return filtreli.astype(np.float32)
    
    @staticmethod
    def adaptif_histogram_esit(goruntu: np.ndarray,
                               clip_limit: float = 2.0,
                               tile_size: tuple = (8, 8)) -> np.ndarray:
        """
        Adaptif Histogram Eşitleme (CLAHE) - kontrast iyileştirme.
        
        Parametreler:
        - clip_limit: Kontrast sınırı
        - tile_size: Bölge boyutu
        
        Çıktı: Kontrast iyileştirilmiş uint8 görüntü
        """
        goruntu_float = goruntu.astype("float32") / 255.0
        goruntu_eq = exposure.equalize_adapthist(
            goruntu_float,
            clip_limit=clip_limit / 100.0
        )
        return (goruntu_eq * 255.0).astype(np.uint8)
    
    @staticmethod
    def gauss_bulaniklastirma(goruntu: np.ndarray,
                              sigma: float = 1.0) -> np.ndarray:
        """
        Gauss bulanıklaştırma - hafif gürültü azaltma.
        
        Parametreler:
        - sigma: Standart sapma
        
        Çıktı: Bulanıklaştırılmış uint8 görüntü
        """
        filtreli = ndimage.gaussian_filter(goruntu, sigma=sigma)
        return np.clip(filtreli, 0, 255).astype(np.uint8)
    
    @staticmethod
    def medyan_filtresi(goruntu: np.ndarray,
                       kernel_boyutu: int = 5) -> np.ndarray:
        """
        Medyan filtresi - tuz-biber gürültüne karşı etkili.
        
        Parametreler:
        - kernel_boyutu: Kernel boyutu (tek sayı)
        
        Çıktı: Filtrele yapılmış uint8 görüntü
        """
        goruntu_uint8 = np.clip(goruntu, 0, 255).astype(np.uint8)
        
        filtreli = cv2.medianBlur(goruntu_uint8, kernel_boyutu)
        return filtreli.astype(np.float32)
    
    @staticmethod
    def morfolojik_kapanma(goruntu: np.ndarray,
                          kernel_boyutu: int = 5) -> np.ndarray:
        """
        Morfolojik kapanış işlemi - küçük boşlukları doldurma.
        
        Parametreler:
        - kernel_boyutu: Yapısal eleman boyutu
        
        Çıktı: İşlenmiş uint8 görüntü
        """
        goruntu_uint8 = np.clip(goruntu, 0, 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_boyutu, kernel_boyutu))
        
        kapanmis = cv2.morphologyEx(goruntu_uint8, cv2.MORPH_CLOSE, kernel)
        return kapanmis.astype(np.float32)
    
    @staticmethod
    def morfolojik_acilma(goruntu: np.ndarray,
                         kernel_boyutu: int = 5) -> np.ndarray:
        """
        Morfolojik açılış işlemi - küçük objeleri temizleme.
        
        Parametreler:
        - kernel_boyutu: Yapısal eleman boyutu
        
        Çıktı: İşlenmiş uint8 görüntü
        """
        goruntu_uint8 = np.clip(goruntu, 0, 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_boyutu, kernel_boyutu))
        
        acilis = cv2.morphologyEx(goruntu_uint8, cv2.MORPH_OPEN, kernel)
        return acilis.astype(np.float32)
    
    @staticmethod
    def laplacian_kenar_tespiti(goruntu: np.ndarray) -> np.ndarray:
        """
        Laplacian filtresi ile kenar tespiti.
        
        Çıktı: Kenar haritası (uint8)
        """
        goruntu_uint8 = np.clip(goruntu, 0, 255).astype(np.uint8)
        
        laplacian = cv2.Laplacian(goruntu_uint8, cv2.CV_32F)
        laplacian = np.clip(np.abs(laplacian), 0, 255).astype(np.uint8)
        return laplacian
    
    @staticmethod
    def sobel_kenar_tespiti(goruntu: np.ndarray) -> np.ndarray:
        """
        Sobel filtreleri ile kenar tespiti (X ve Y yönleri).
        
        Çıktı: Kenar şiddeti haritası (uint8)
        """
        goruntu_uint8 = np.clip(goruntu, 0, 255).astype(np.uint8)
        
        sobelx = cv2.Sobel(goruntu_uint8, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(goruntu_uint8, cv2.CV_32F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
        return magnitude
    
    @staticmethod
    def canny_kenar_tespiti(goruntu: np.ndarray,
                           esik1: int = 100,
                           esik2: int = 200) -> np.ndarray:
        """
        Canny kenar dedektörü.
        
        Parametreler:
        - esik1: Düşük eşik
        - esik2: Yüksek eşik
        
        Çıktı: Binary kenar haritası (uint8)
        """
        goruntu_uint8 = np.clip(goruntu, 0, 255).astype(np.uint8)
        
        kenarlar = cv2.Canny(goruntu_uint8, esik1, esik2)
        return kenarlar
    
    @staticmethod
    def kontrast_uzatma(goruntu: np.ndarray) -> np.ndarray:
        """
        Kontrast uzatma - histogramın tüm aralığını kullanma.
        
        Çıktı: Uzatılmış kontrast uint8 görüntü
        """
        goruntu_uint8 = np.clip(goruntu, 0, 255).astype(np.uint8)
        
        min_val = goruntu_uint8.min()
        max_val = goruntu_uint8.max()
        
        if max_val - min_val < 1:
            return goruntu_uint8
        
        uzatilmis = ((goruntu_uint8 - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        return uzatilmis
    
    @staticmethod
    def unsharp_masking(goruntu: np.ndarray,
                       sigma: float = 1.0,
                       strength: float = 1.0) -> np.ndarray:
        """
        Unsharp masking - keskinlik artırma.
        
        Parametreler:
        - sigma: Gauss bulanıklığının sigma değeri
        - strength: Maskeleme gücü
        
        Çıktı: Keskinleştirilmiş uint8 görüntü
        """
        goruntu_float = goruntu.astype(np.float32)
        
        # Bulanık versiyonu oluştur
        bulanik = ndimage.gaussian_filter(goruntu_float, sigma=sigma)
        
        # Unsharp mask uygula
        maske = goruntu_float - bulanik
        keskinlesmis = goruntu_float + strength * maske
        
        return np.clip(keskinlesmis, 0, 255).astype(np.uint8)
    
    @staticmethod
    def top_hat_filtresi(goruntu: np.ndarray,
                        kernel_boyutu: int = 9) -> np.ndarray:
        """
        Top-hat filtresi - küçük yapıları vurgulama.
        
        Parametreler:
        - kernel_boyutu: Yapısal eleman boyutu
        
        Çıktı: Filtrelenmiş uint8 görüntü
        """
        goruntu_uint8 = np.clip(goruntu, 0, 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_boyutu, kernel_boyutu))
        
        tophat = cv2.morphologyEx(goruntu_uint8, cv2.MORPH_TOPHAT, kernel)
        return tophat
    
    @staticmethod
    def black_hat_filtresi(goruntu: np.ndarray,
                          kernel_boyutu: int = 9) -> np.ndarray:
        """
        Black-hat filtresi - gölgeleri vurgulama.
        
        Parametreler:
        - kernel_boyutu: Yapısal eleman boyutu
        
        Çıktı: Filtrelenmiş uint8 görüntü
        """
        goruntu_uint8 = np.clip(goruntu, 0, 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_boyutu, kernel_boyutu))
        
        blackhat = cv2.morphologyEx(goruntu_uint8, cv2.MORPH_BLACKHAT, kernel)
        return blackhat


def otomatik_filtre_uygulamasi(goruntu: np.ndarray,
                               kalite: str = "standart") -> np.ndarray:
    """
    Görüntü kalitesine göre otomatik filtre kombinasyonu uygula.
    
    Parametreler:
    - goruntu: İnput görüntü (uint8 veya float)
    - kalite: "düşük", "standart", "yüksek"
    
    Çıktı: İşlenmiş görüntü
    """
    goruntu_uint8 = np.clip(goruntu, 0, 255).astype(np.uint8)
    
    if kalite == "düşük":
        # Düşük kalite - aggressive denoising
        islemli = GelismisFiltreler.non_lokal_ortalama(goruntu_uint8, h=15.0)
        islemli = GelismisFiltreler.adaptif_histogram_esit(islemli, clip_limit=3.0)
        islemli = GelismisFiltreler.unsharp_masking(islemli, sigma=0.8, strength=1.5)
        
    elif kalite == "yüksek":
        # Yüksek kalite - moderate denoising
        islemli = GelismisFiltreler.bilateral_filtre(goruntu_uint8, cerceve_eni=7, sigma_renk=50, sigma_mekan=50)
        islemli = GelismisFiltreler.adaptif_histogram_esit(islemli, clip_limit=2.0)
        islemli = GelismisFiltreler.unsharp_masking(islemli, sigma=1.0, strength=0.8)
        
    else:  # standart
        # Standart - balanced denoising
        islemli = GelismisFiltreler.non_lokal_ortalama(goruntu_uint8, h=10.0)
        islemli = GelismisFiltreler.adaptif_histogram_esit(islemli, clip_limit=2.5)
        islemli = GelismisFiltreler.unsharp_masking(islemli, sigma=1.0, strength=1.0)
    
    return islemli
