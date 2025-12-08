#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
csv_analiz_ornegi.py
--------------------
Oluşturulan CSV dosyalarını analiz etme örnekleri.

Çalıştırma:
    python scripts/csv_analiz_ornegi.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


def csv_analiz():
    """CSV dosyalarının temel analizini yap."""
    
    csv_yolu = Path("veri/cikti/goruntu_ozellikleri.csv")
    
    if not csv_yolu.exists():
        print(f"[HATA] {csv_yolu} dosyası bulunamadı!")
        print("Önce 'python scripts/csv_donusturme.py' komutunu çalıştırın.")
        return
    
    print("="*80)
    print("CSV ANALIZ ÖRNEKLERİ")
    print("="*80)
    
    # CSV yükle
    df = pd.read_csv(csv_yolu)
    
    print(f"\n📊 VERİ SETI İSTATİSTİKLERİ")
    print(f"-"*80)
    print(f"Toplam Görüntü: {len(df):,}")
    print(f"Sütun Sayısı: {len(df.columns)}")
    print(f"Bellek Kullanımı: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Sınıf dağılımı
    print(f"\n📈 SINIF DAĞILIMI")
    print(f"-"*80)
    sinif_dagilim = df['sinif'].value_counts()
    for sinif, sayi in sinif_dagilim.items():
        yuzde = (sayi / len(df)) * 100
        print(f"{sinif:20s}: {sayi:6,d} ({yuzde:5.1f}%)")
    
    # Boyut istatistikleri
    print(f"\n📏 BOYUT İSTATİSTİKLERİ")
    print(f"-"*80)
    print(f"Genişlik (piksel):")
    print(f"  Min: {df['genislik'].min():5.0f}")
    print(f"  Ort: {df['genislik'].mean():5.1f}")
    print(f"  Max: {df['genislik'].max():5.0f}")
    print(f"  Std: {df['genislik'].std():5.1f}")
    
    print(f"\nYükseklik (piksel):")
    print(f"  Min: {df['yukseklik'].min():5.0f}")
    print(f"  Ort: {df['yukseklik'].mean():5.1f}")
    print(f"  Max: {df['yukseklik'].max():5.0f}")
    print(f"  Std: {df['yukseklik'].std():5.1f}")
    
    print(f"\nEn/Boy Oranı:")
    print(f"  Min: {df['en_boy_orani'].min():.4f}")
    print(f"  Ort: {df['en_boy_orani'].mean():.4f}")
    print(f"  Max: {df['en_boy_orani'].max():.4f}")
    
    # Yoğunluk istatistikleri
    print(f"\n🌡️  YOĞUNLUK İSTATİSTİKLERİ (0-255)")
    print(f"-"*80)
    print(f"Ortalama Yoğunluk:")
    print(f"  Min: {df['ort_yogunluk'].min():7.2f}")
    print(f"  Ort: {df['ort_yogunluk'].mean():7.2f}")
    print(f"  Max: {df['ort_yogunluk'].max():7.2f}")
    print(f"  Std: {df['ort_yogunluk'].std():7.2f}")
    
    print(f"\nStandart Sapma Yoğunluk:")
    print(f"  Min: {df['std_yogunluk'].min():7.2f}")
    print(f"  Ort: {df['std_yogunluk'].mean():7.2f}")
    print(f"  Max: {df['std_yogunluk'].max():7.2f}")
    
    # Entropi istatistikleri
    print(f"\n🔗 ENTROPI İSTATİSTİKLERİ")
    print(f"-"*80)
    print(f"Entropi (0-8):")
    print(f"  Min: {df['entropi'].min():.4f}")
    print(f"  Ort: {df['entropi'].mean():.4f}")
    print(f"  Max: {df['entropi'].max():.4f}")
    print(f"  Std: {df['entropi'].std():.4f}")
    
    # Kontrast istatistikleri
    print(f"\n⚡ KONTRAST İSTATİSTİKLERİ")
    print(f"-"*80)
    print(f"Kontrast:")
    print(f"  Min: {df['kontrast'].min():.4f}")
    print(f"  Ort: {df['kontrast'].mean():.4f}")
    print(f"  Max: {df['kontrast'].max():.4f}")
    print(f"  Std: {df['kontrast'].std():.4f}")
    
    # Sınıf bazında karşılaştırma
    print(f"\n🔍 SINIF BAZINDA KARŞILAŞTIRMA")
    print(f"-"*80)
    
    for sinif in sorted(df['sinif'].unique()):
        sinif_df = df[df['sinif'] == sinif]
        print(f"\n{sinif}:")
        print(f"  Örnek: {len(sinif_df):6,d}")
        print(f"  Ort. Yoğunluk: {sinif_df['ort_yogunluk'].mean():7.2f} ± {sinif_df['ort_yogunluk'].std():6.2f}")
        print(f"  Ort. Entropi: {sinif_df['entropi'].mean():.4f} ± {sinif_df['entropi'].std():.4f}")
        print(f"  Ort. Kontrast: {sinif_df['kontrast'].mean():.4f} ± {sinif_df['kontrast'].std():.4f}")
    
    # İlişkiler
    print(f"\n🔗 ÖZNİTELİKLER ARASI İLİŞKİ (Correlation)")
    print(f"-"*80)
    
    # İlişki matrisi
    ozellikler = ['ort_yogunluk', 'std_yogunluk', 'entropi', 'kontrast']
    iliski = df[ozellikler].corr()
    
    print(iliski.to_string())
    
    # En yüksek korelasyonlar
    print(f"\nEn Yüksek Korelasyonlar:")
    for i in range(len(ozellikler)):
        for j in range(i+1, len(ozellikler)):
            korelasyon = iliski.iloc[i, j]
            print(f"  {ozellikler[i]:20s} <-> {ozellikler[j]:20s}: {korelasyon:+.4f}")
    
    print("\n" + "="*80)


def anomali_tespiti():
    """Anomali (outlier) tespit et."""
    
    csv_yolu = Path("veri/cikti/goruntu_ozellikleri.csv")
    
    if not csv_yolu.exists():
        print("CSV dosyası bulunamadı!")
        return
    
    df = pd.read_csv(csv_yolu)
    
    print("\n" + "="*80)
    print("ANOMALİ TESPİTİ (Outlier Detection)")
    print("="*80)
    
    # Z-score ile anomali tespit et
    from scipy import stats
    
    ozellikler = ['ort_yogunluk', 'entropi', 'kontrast']
    
    for ozellik in ozellikler:
        z_score = np.abs(stats.zscore(df[ozellik]))
        anomali_indeksleri = np.where(z_score > 3)[0]  # 3-sigma rule
        
        print(f"\n{ozellik}:")
        print(f"  Toplam: {len(df)}")
        print(f"  Anomali Sayısı (|Z| > 3): {len(anomali_indeksleri)}")
        print(f"  Anomali Yüzdesi: {(len(anomali_indeksleri) / len(df)) * 100:.2f}%")
        
        if len(anomali_indeksleri) > 0:
            print(f"  Örnek Anomali Değerleri:")
            for idx in anomali_indeksleri[:3]:
                print(f"    {df.iloc[idx]['dosya_adı']}: {df.iloc[idx][ozellik]:.4f}")


def csv_dısa_aktar(format='excel'):
    """CSV'yi farklı formatlara aktar."""
    
    csv_yolu = Path("veri/cikti/goruntu_ozellikleri.csv")
    
    if not csv_yolu.exists():
        print("CSV dosyası bulunamadı!")
        return
    
    df = pd.read_csv(csv_yolu)
    
    if format == 'excel':
        excel_yolu = csv_yolu.parent / "goruntu_ozellikleri.xlsx"
        df.to_excel(excel_yolu, index=False)
        print(f"[TAMAMLANDI] Excel dosyası kaydedildi: {excel_yolu}")
    
    elif format == 'json':
        json_yolu = csv_yolu.parent / "goruntu_ozellikleri.json"
        df.to_json(json_yolu, orient='records', indent=2)
        print(f"[TAMAMLANDI] JSON dosyası kaydedildi: {json_yolu}")
    
    elif format == 'parquet':
        parquet_yolu = csv_yolu.parent / "goruntu_ozellikleri.parquet"
        df.to_parquet(parquet_yolu, index=False)
        print(f"[TAMAMLANDI] Parquet dosyası kaydedildi: {parquet_yolu}")


if __name__ == "__main__":
    csv_analiz()
    anomali_tespiti()
    
    print("\n" + "="*80)
    print("Ek İşlemler:")
    print("="*80)
    print("Excel'e aktar: csv_dısa_aktar('excel')")
    print("JSON'a aktar: csv_dısa_aktar('json')")
    print("Parquet'a aktar: csv_dısa_aktar('parquet')")
