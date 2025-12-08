#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
filtre_test_et.py
-----------------
Gelişmiş filtrelerin görüntüler üzerindeki etkisini test etme script'i.

Kullanım:
    python scripts/filtre_test_et.py veri/girdi/ornek.jpg
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from goruntu_isleme_mri.io_araclari import goruntu_gri_olarak_oku
from goruntu_isleme_mri.gelismis_filtreler import (
    GelismisFiltreler,
    otomatik_filtre_uygulamasi,
)


def main(girdi_yolu: str):
    goruntu_gri = goruntu_gri_olarak_oku(girdi_yolu)
    
    # Orijinal görüntüyü normalize et
    goruntu_gri = np.clip(goruntu_gri, 0, 255).astype(np.uint8)
    
    print("[BILGI] Filtreler uygulanıyor...")
    
    # Farklı filtreleri uygula
    filtrelenmis_nlm = GelismisFiltreler.non_lokal_ortalama(goruntu_gri, h=10.0)
    filtrelenmis_bilateral = GelismisFiltreler.bilateral_filtre(goruntu_gri)
    filtrelenmis_medyan = GelismisFiltreler.medyan_filtresi(goruntu_gri)
    filtrelenmis_otomatik = otomatik_filtre_uygulamasi(goruntu_gri, kalite="standart")
    
    # Kenar tespitleri
    kenarlar_canny = GelismisFiltreler.canny_kenar_tespiti(goruntu_gri)
    kenarlar_sobel = GelismisFiltreler.sobel_kenar_tespiti(goruntu_gri)
    kenarlar_laplacian = GelismisFiltreler.laplacian_kenar_tespiti(goruntu_gri)
    
    # Kontrast işlemleri
    kontrast_uzatilmis = GelismisFiltreler.kontrast_uzatma(goruntu_gri)
    unsharp_maskeli = GelismisFiltreler.unsharp_masking(goruntu_gri, sigma=1.0, strength=1.0)
    
    # Görselleştirme
    fig, eksenler = plt.subplots(3, 4, figsize=(16, 12))
    
    # Satır 1: Gürültü azaltma
    eksenler[0, 0].imshow(goruntu_gri, cmap="gray")
    eksenler[0, 0].set_title("Orijinal")
    eksenler[0, 0].axis("off")
    
    eksenler[0, 1].imshow(filtrelenmis_nlm, cmap="gray")
    eksenler[0, 1].set_title("Non-Lokal Ortalama (NLM)")
    eksenler[0, 1].axis("off")
    
    eksenler[0, 2].imshow(filtrelenmis_bilateral, cmap="gray")
    eksenler[0, 2].set_title("Bilateral Filtre")
    eksenler[0, 2].axis("off")
    
    eksenler[0, 3].imshow(filtrelenmis_medyan, cmap="gray")
    eksenler[0, 3].set_title("Medyan Filtresi")
    eksenler[0, 3].axis("off")
    
    # Satır 2: Kenar tespitleri
    eksenler[1, 0].imshow(kenarlar_canny, cmap="gray")
    eksenler[1, 0].set_title("Canny Kenarlar")
    eksenler[1, 0].axis("off")
    
    eksenler[1, 1].imshow(kenarlar_sobel, cmap="gray")
    eksenler[1, 1].set_title("Sobel Kenarlar")
    eksenler[1, 1].axis("off")
    
    eksenler[1, 2].imshow(kenarlar_laplacian, cmap="gray")
    eksenler[1, 2].set_title("Laplacian Kenarlar")
    eksenler[1, 2].axis("off")
    
    eksenler[1, 3].imshow(filtrelenmis_otomatik, cmap="gray")
    eksenler[1, 3].set_title("Otomatik Kombinasyon")
    eksenler[1, 3].axis("off")
    
    # Satır 3: Kontrast işlemleri
    eksenler[2, 0].imshow(kontrast_uzatilmis, cmap="gray")
    eksenler[2, 0].set_title("Kontrast Uzatma")
    eksenler[2, 0].axis("off")
    
    eksenler[2, 1].imshow(unsharp_maskeli, cmap="gray")
    eksenler[2, 1].set_title("Unsharp Masking")
    eksenler[2, 1].axis("off")
    
    # Histogramlar
    eksenler[2, 2].hist(goruntu_gri.flatten(), bins=256, alpha=0.7, label="Orijinal")
    eksenler[2, 2].hist(filtrelenmis_otomatik.flatten(), bins=256, alpha=0.7, label="Filtrelenmiş")
    eksenler[2, 2].set_title("Histogram Karşılaştırması")
    eksenler[2, 2].legend()
    
    # İstatistikler
    ax_stats = eksenler[2, 3]
    ax_stats.axis("off")
    stats_text = f"""
    Orijinal:
    Min: {goruntu_gri.min():.0f}
    Max: {goruntu_gri.max():.0f}
    Mean: {goruntu_gri.mean():.1f}
    Std: {goruntu_gri.std():.1f}
    
    Filtrelenmiş (Otomatik):
    Min: {filtrelenmis_otomatik.min():.0f}
    Max: {filtrelenmis_otomatik.max():.0f}
    Mean: {filtrelenmis_otomatik.mean():.1f}
    Std: {filtrelenmis_otomatik.std():.1f}
    """
    ax_stats.text(0.1, 0.5, stats_text, fontsize=10, family="monospace",
                  verticalalignment="center")
    
    plt.suptitle("Gelişmiş Filtre Testi", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Lütfen bir görüntü yolu verin. Örnek:")
        print("  python scripts/filtre_test_et.py veri/girdi/ornek.jpg")
        sys.exit(1)
    
    girdi_yolu = sys.argv[1]
    if not Path(girdi_yolu).exists():
        print(f"Girdi yolu bulunamadı: {girdi_yolu}")
        sys.exit(1)
    
    main(girdi_yolu)
