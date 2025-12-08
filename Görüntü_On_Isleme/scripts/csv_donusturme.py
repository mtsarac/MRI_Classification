#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
csv_donusturme.py
-----------------
Ön işlenmiş MRI görüntülerini CSV formatına dönüştüren ana script.

Çalıştırma:
    python scripts/csv_donusturme.py
    
Çıktı:
    1. goruntu_ozellikleri.csv - Her görüntü için 15+ öznitelik
    2. istatistikler.csv - Sınıf bazında özet istatistikler
"""

import sys
import argparse
from pathlib import Path

# Proje modüllerini import et
sys.path.insert(0, str(Path(__file__).parent.parent))

from goruntu_isleme_mri.csv_olusturucu import (
    tum_gorseller_icin_csv_olustur,
    istatistikleri_kaydet,
)
from goruntu_isleme_mri.ayarlar import CIKTI_KLASORU


def main():
    """Ana fonksiyon - CSV dönüşümünü başlat."""
    
    parser = argparse.ArgumentParser(
        description="Ön işlenmiş MRI görüntülerini CSV'ye dönüştür"
    )
    parser.add_argument(
        '--cikti-klasoru',
        type=str,
        default=CIKTI_KLASORU,
        help='Ön işlenmiş görüntülerin klasörü'
    )
    parser.add_argument(
        '--csv-adi',
        type=str,
        default='goruntu_ozellikleri.csv',
        help='Oluşturulacak CSV dosyasının adı'
    )
    parser.add_argument(
        '--istatistik-csv-adi',
        type=str,
        default='istatistikler.csv',
        help='Istatistik CSV dosyasının adı'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("MRI GÖRÜNTÜ CSV DÖNÜŞTÜRME TOOL'I")
    print("="*80)
    print(f"\nGörüntü Klasörü: {args.cikti_klasoru}")
    print(f"CSV Dosyası: {args.csv_adi}")
    print(f"Istatistik CSV: {args.istatistik_csv_adi}\n")
    
    # CSV oluştur
    csv_yolu = tum_gorseller_icin_csv_olustur(
        cikti_klasoru=args.cikti_klasoru,
        csv_dosya_adi=args.csv_adi
    )
    
    if not Path(csv_yolu).exists():
        print("[HATA] CSV dosyası oluşturulamadı!")
        return 1
    
    # Istatistikleri hesapla
    istatistik_yolu = istatistikleri_kaydet(
        csv_dosya_yolu=csv_yolu,
        istatistik_csv_adi=args.istatistik_csv_adi
    )
    
    print("\n" + "="*80)
    print("IŞLEM TAMAMLANDI")
    print("="*80)
    print(f"\n✅ CSV dosyaları başarıyla oluşturuldu:")
    print(f"   1. {csv_yolu}")
    print(f"   2. {istatistik_yolu}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
