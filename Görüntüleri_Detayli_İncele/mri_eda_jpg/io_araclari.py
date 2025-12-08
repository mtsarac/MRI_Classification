"""
io_araclari.py
--------------
Veri okuma, görüntü yükleme ve temel yardımcı fonksiyonlar.
"""

import random
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .ayarlar import (
    RASTGELE_TOHUM,
    CIKTI_KLASORU,
    ETIKET_ISIM_HARITASI,
    N_PIKSEL_ORNEK_SAYISI,
    GORSELU_GRIDE_CEV,
    VERI_KLASORU,
    SINIF_KLASORLERI,
    SINIF_ETIKETI,
)


def cikti_klasorunu_olustur():
    """Çıktı klasörünün var olduğundan emin ol."""
    Path(CIKTI_KLASORU).mkdir(parents=True, exist_ok=True)


def rastgele_tohum_ayarla(tohum: int = RASTGELE_TOHUM):
    """Tüm rastgelelik kaynakları için sabit tohum ayarla."""
    random.seed(tohum)
    np.random.seed(tohum)


def etiket_tablosu_yukle(csv_yolu: str = None) -> pd.DataFrame:
    """
    CSV dosyasını oku VEYA sınıf klasörlerinden veri yükle.
    
    Eğer csv_yolu None ise, sınıf klasörlerini tarayarak otomatik veri oluştur.
    Aksi takdirde labels.csv dosyasını oku.
    
    Beklenen kolonlar: id, filepath, label (CSV için) 
    Döndürülen DataFrame: id, filepath, label, label_name

    Returns:
        pd.DataFrame: Görüntü yolları ve etiketleri içeren DataFrame
    """
    if csv_yolu is None:
        # Sınıf klasörlerinden otomatik veri yükle
        return _sinif_klasorlerinden_veri_yukle()
    else:
        # CSV dosyasından yükle
        return _csv_dosyasından_veri_yukle(csv_yolu)


def _sinif_klasorlerinden_veri_yukle() -> pd.DataFrame:
    """Sınıf klasörlerini tarayarak veri DataFrame oluştur."""
    veriler = []
    id_sayaci = 0
    
    veri_dir = Path(VERI_KLASORU)
    
    for sinif_adi in SINIF_KLASORLERI:
        sinif_klasoru = veri_dir / sinif_adi
        
        if not sinif_klasoru.exists():
            print(f"[UYARI] Sınıf klasörü bulunamadı: {sinif_klasoru}")
            continue
        
        etiket = SINIF_ETIKETI[sinif_adi]
        
        # Tüm görüntü dosyalarını bul
        for dosya in sinif_klasoru.glob("*.[jJ][pP][gG]"):
            veriler.append({
                "id": id_sayaci,
                "filepath": str(dosya),
                "label": etiket,
            })
            id_sayaci += 1
        
        # PNG dosyaları da ekle
        for dosya in sinif_klasoru.glob("*.[pP][nN][gG]"):
            veriler.append({
                "id": id_sayaci,
                "filepath": str(dosya),
                "label": etiket,
            })
            id_sayaci += 1
    
    df = pd.DataFrame(veriler)
    
    if len(df) == 0:
        print("[UYARI] Hiçbir görüntü dosyası bulunamadı!")
        return df
    
    # label_name ekle
    if ETIKET_ISIM_HARITASI is not None:
        df["label_name"] = df["label"].map(ETIKET_ISIM_HARITASI).astype(str)
    else:
        df["label_name"] = df["label"].astype(str)
    
    print(f"[BILGI] Toplam {len(df)} adet görüntü yüklendi")
    print(f"[BILGI] Sınıf dağılımı:")
    for sinif_adi in SINIF_KLASORLERI:
        etiket = SINIF_ETIKETI[sinif_adi]
        adet = len(df[df["label"] == etiket])
        print(f"  {sinif_adi}: {adet}")
    
    return df


def _csv_dosyasından_veri_yukle(csv_yolu: str) -> pd.DataFrame:
    """CSV dosyasından veri yükle."""
    df = pd.read_csv(csv_yolu)
    beklenen_kolonlar = {"id", "filepath", "label"}
    if not beklenen_kolonlar.issubset(df.columns):
        raise ValueError(f"CSV şu kolonları içermeli: {beklenen_kolonlar}, mevcut: {df.columns}")

    # Dosya yollarının gerçekten var olup olmadığına hızlı bir bakış
    eksik = []
    for yol in df["filepath"]:
        if not Path(yol).exists():
            eksik.append(yol)
    if len(eksik) > 0:
        print(f"[UYARI] {len(eksik)} adet dosya bulunamadı. İlk birkaç örnek:")
        print("\n".join(map(str, eksik[:10])))

    # label'ı kategorik stringe çevir
    if ETIKET_ISIM_HARITASI is not None:
        df["label_name"] = df["label"].map(ETIKET_ISIM_HARITASI).astype(str)
    else:
        df["label_name"] = df["label"].astype(str)

    return df


def goruntu_yukle_yoksa_gri(yol: str) -> np.ndarray:
    """
    JPEG/PNG görüntüyü oku ve numpy dizisi olarak döndür.

    - Eğer GORSELU_GRIDE_CEV True ise: (H, W) gri tonlamalı
    - Aksi halde: (H, W, C) RGB
    """
    img = Image.open(yol)
    if GORSELU_GRIDE_CEV:
        img = img.convert("L")  # 8-bit gri tonlama
    else:
        img = img.convert("RGB")
    dizi = np.array(img).astype(np.float32)
    return dizi


def gorselu_normalize_et(goruntu: np.ndarray) -> np.ndarray:
    """
    Görselleştirme için 2B görüntüyü normalize et:
    - %1 ve %99 percentilleri arasında kırp
    - 0-1 aralığına ölçekle
    """
    flat = goruntu.flatten()
    vmin, vmax = np.percentile(flat, [1, 99])
    goruntu_kirp = np.clip(goruntu, vmin, vmax)
    if vmax - vmin < 1e-6:
        return np.zeros_like(goruntu_kirp)
    return (goruntu_kirp - vmin) / (vmax - vmin)


def rastgele_piksel_ornekle(yol: str,
                             n_piksel: int = N_PIKSEL_ORNEK_SAYISI) -> np.ndarray:
    """
    Global yoğunluk histogramı için tek bir görüntüden rastgele piksel örnekle.
    """
    goruntu = goruntu_yukle_yoksa_gri(yol)
    flat = goruntu.flatten()
    flat = flat[~np.isnan(flat)]

    if flat.size == 0:
        return np.array([])

    n = min(n_piksel, flat.size)
    idx = np.random.choice(flat.size, size=n, replace=False)
    return flat[idx]
