#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
toplu_on_isleme.py
------------------
Girdi klasöründeki tüm MRI görüntülerine ön işleme pipeline'ını uygular.

Adımlar:
  1) Girdi klasöründeki tüm JPEG/PNG dosyalarını bul
  2) Her görüntüyü gri tonlamada oku
  3) Arka plan tespiti, maskeleme, kırpma, normalizasyon, histogram eşitleme, yeniden boyutlandırma
  4) Ön işlenmiş görüntüyü çıktı klasörüne kaydet
  5) (İsteğe bağlı) Veri artırma ile ekstra kopyalar üret
  6) Tüm işlemleri CSV log dosyasında özetle

Çalıştırma:
    python scripts/toplu_on_isleme.py
"""

import os
from pathlib import Path

from goruntu_isleme_mri.ayarlar import (
    GIRDİ_KLASORU,
    CIKTI_KLASORU,
    VERI_ARTIRMA_AKTIF,
    EKSTRA_KOPYA_SAYISI,
    SINIF_DENGELEME_AKTIF,
    SINIF_AUGMENTATION_SAYILARI,
)
from goruntu_isleme_mri.io_araclari import (
    rastgele_tohum_ayarla,
    klasor_olustur_yoksa,
    girdi_gorsellerini_listele,
    goruntu_gri_olarak_oku,
    goruntu_dosyaya_kaydet,
    cikti_yolu_uretle,
)
from goruntu_isleme_mri.on_isleme_adimlari import tek_goruntu_on_isle
from goruntu_isleme_mri.veri_artirma import rastgele_artirma_uygula
from goruntu_isleme_mri.csv_olusturucu import on_isleme_log_kaydet


def ana():
    rastgele_tohum_ayarla()
    klasor_olustur_yoksa(GIRDİ_KLASORU)
    klasor_olustur_yoksa(CIKTI_KLASORU)

    print(f"[BILGI] Girdi klasörü: {GIRDİ_KLASORU}")
    print(f"[BILGI] Çıktı klasörü: {CIKTI_KLASORU}")

    girdi_listesi = girdi_gorsellerini_listele(GIRDİ_KLASORU)
    print(f"[BILGI] Toplam {len(girdi_listesi)} adet görüntü bulundu.")

    log_kayitlari = []

    for i, dosya_bilgisi in enumerate(girdi_listesi, start=1):
        girdi_yolu = dosya_bilgisi["path"]
        sinif_adi = dosya_bilgisi["sinif"]
        etiket = dosya_bilgisi["etiket"]
        
        print(f"[{i}/{len(girdi_listesi)}] İşleniyor: {girdi_yolu} (Sınıf: {sinif_adi}, Etiket: {etiket})")

        goruntu_gri = goruntu_gri_olarak_oku(girdi_yolu)
        on_islenmis, meta = tek_goruntu_on_isle(goruntu_gri)

        # Ana çıktı dosyasının yolunu üret ve kaydet
        cikti_yolu = cikti_yolu_uretle(girdi_yolu)
        goruntu_dosyaya_kaydet(cikti_yolu, on_islenmis)

        # Log kaydı - sınıf ve etiket bilgisini ekle
        kayit = {
            "girdi_yolu": girdi_yolu,
            "sinif": sinif_adi,
            "etiket": etiket,
            "cikti_yolu": cikti_yolu,
            **meta,
        }
        log_kayitlari.append(kayit)

        # Veri artırma aktif ise ekstra kopyalar üret
        if VERI_ARTIRMA_AKTIF:
            # Sınıf dengeleme aktif ise, sınıfa özel augmentation sayısı kullan
            if SINIF_DENGELEME_AKTIF and sinif_adi in SINIF_AUGMENTATION_SAYILARI:
                aug_sayisi = SINIF_AUGMENTATION_SAYILARI[sinif_adi]
            else:
                aug_sayisi = EKSTRA_KOPYA_SAYISI
            
            if aug_sayisi > 0:
                ana_govde, uzanti = os.path.splitext(cikti_yolu)
                for k in range(aug_sayisi):
                    artirilmis = rastgele_artirma_uygula(on_islenmis)
                    artirilmis_yol = f"{ana_govde}_aug{k+1}{uzanti}"
                    goruntu_dosyaya_kaydet(artirilmis_yol, artirilmis)
                    # Augmented örnekler için de log ekle
                    log_kayitlari.append({
                        "girdi_yolu": girdi_yolu,
                        "sinif": sinif_adi,
                        "etiket": etiket,
                        "cikti_yolu": artirilmis_yol,
                        **meta,
                        "aug_kopya": k + 1,
                    })

    # Log dosyasını CSV'ye kaydet
    on_isleme_log_kaydet(log_kayitlari, CIKTI_KLASORU)
    print(f"[TAMAMLANDI] Ön işleme tamamlandı.")


if __name__ == "__main__":
    ana()
