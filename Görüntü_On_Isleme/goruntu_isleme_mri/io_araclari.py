"""
io_araclari.py
--------------
Dosya sisteminden görüntüleri bulma, okuma ve kaydetme ile ilgili yardımcı fonksiyonlar.
"""

import os
import random
from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image

from .ayarlar import (
    GIRDİ_KLASORU,
    CIKTI_KLASORU,
    GORUNTU_UZANTILARI,
    RASTGELE_TOHUM,
    SINIF_KLASORLERI,
    SINIF_ETIKETI,
)


def rastgele_tohum_ayarla(tohum: int = RASTGELE_TOHUM) -> None:
    """Tüm rastgelelik kaynakları için sabit tohum ayarla."""
    random.seed(tohum)
    np.random.seed(tohum)


def klasor_olustur_yoksa(klasor_yolu: str) -> None:
    """Verilen klasör yolu yoksa oluştur."""
    Path(klasor_yolu).mkdir(parents=True, exist_ok=True)


def girdi_gorsellerini_listele(klasor_yolu: str = GIRDİ_KLASORU) -> List[Dict]:
    """
    Girdi klasörü altında izin verilen uzantılara sahip tüm görüntü dosyalarını listele.
    Alt klasörler de taranır. Sınıf etiketi bilgisini de döndürür.
    
    Döndürülen: [{"path": dosya_yolu, "sınıf": sınıf_adı, "etiket": etiket_numarası}, ...]
    """
    klasor = Path(klasor_yolu)
    dosya_listesi = []
    
    for kok, alt_klasorler, dosyalar in os.walk(klasor):
        # Mevcut klasör adını kontrol et
        mevcut_klasor_adi = Path(kok).name
        
        for dosya in dosyalar:
            alt_uzanti = Path(dosya).suffix.lower()
            if alt_uzanti in GORUNTU_UZANTILARI:
                tam_yol = str(Path(kok) / dosya)
                
                # Sınıf etiketini belirle
                if mevcut_klasor_adi in SINIF_ETIKETI:
                    etiket = SINIF_ETIKETI[mevcut_klasor_adi]
                    dosya_listesi.append({
                        "path": tam_yol,
                        "sinif": mevcut_klasor_adi,
                        "etiket": etiket,
                    })
    
    # Dosya yoluna göre sırala
    dosya_listesi.sort(key=lambda x: x["path"])
    return dosya_listesi


def goruntu_gri_olarak_oku(yol: str) -> np.ndarray:
    """
    Verilen dosya yolundaki görüntüyü oku ve gri tonlamaya çevir.
    
    Parametreler:
    -----------
    yol : str
        Görüntü dosyasının yolu
    
    Döndürülen:
    ---------
    np.ndarray
        Gri tonlama görüntü (H, W) float32 [0, 255] aralığında
    
    Raises:
    ------
    FileNotFoundError: Dosya bulunamadığında
    ValueError: Görüntü formatı geçersizse
    """
    try:
        dosya_yolu = Path(yol)
        if not dosya_yolu.exists():
            raise FileNotFoundError(f"Görüntü dosyası bulunamadı: {yol}")
        
        img = Image.open(yol).convert("L")  # 8-bit gri tonlama
        arr = np.array(img).astype(np.float32)
        
        if arr.size == 0:
            raise ValueError(f"Boş görüntü yüklendi: {yol}")
        
        return arr
    except FileNotFoundError as e:
        print(f"[HATA] {str(e)}")
        raise
    except Exception as e:
        print(f"[HATA] Görüntü okuma hatası ({yol}): {str(e)}")
        raise


def goruntu_dosyaya_kaydet(yol: str, goruntu: np.ndarray) -> None:
    """
    Verilen numpy dizisini JPEG/PNG olarak kaydet.
    
    Parametreler:
    -----------
    yol : str
        Kaydedilecek dosyanın yolu
    goruntu : np.ndarray
        (H, W) veya (H, W, 3) numpy dizisi [0, 255] aralığında
    
    Raises:
    ------
    ValueError: Geçersiz görüntü şekli veya veri tipi
    IOError: Dosya yazma hatası
    """
    try:
        if goruntu is None or goruntu.size == 0:
            raise ValueError("Geçersiz görüntü: Boş veya None")
        
        if goruntu.ndim not in (2, 3):
            raise ValueError(f"Geçersiz görüntü şekli: {goruntu.ndim}D (beklenen: 2D veya 3D)")
        
        arr = goruntu
        if arr.ndim == 2:
            img = Image.fromarray(arr.astype("uint8"), mode="L")
        else:
            img = Image.fromarray(arr.astype("uint8"), mode="RGB")

        kayit_yolu = Path(yol)
        kayit_yolu.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(kayit_yolu))
    except Exception as e:
        print(f"[HATA] Görüntü kaydetme hatası ({yol}): {str(e)}")
        raise


def cikti_yolu_uretle(girdi_dosyasi: str, girdi_kok: str = GIRDİ_KLASORU, cikti_kok: str = CIKTI_KLASORU) -> str:
    """
    Girdi dosyasının CIKTI_KLASORU içindeki karşılık gelen yolunu üretir.
    Örnek:
        girdi: veri/girdi/alt/hasta_001.jpg
        çıktı: veri/cikti/alt/hasta_001.jpg
    """
    girdi_path = Path(girdi_dosyasi)
    girdi_kok = Path(girdi_kok)
    nispi_yol = girdi_path.relative_to(girdi_kok)
    cikti_path = Path(cikti_kok) / nispi_yol
    return str(cikti_path)
