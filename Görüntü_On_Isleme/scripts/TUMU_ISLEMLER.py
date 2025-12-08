#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TUMU_ISLEMLER.py (Tüm İşlemler - Ana Script)
=============================================
Model eğitimi için gerekli tüm işlemleri tek bir script'ten yapın.

İçerik:
  1. Ön işleme (Toplu görüntü işleme)
  2. CSV oluşturma ve normalizasyon
  3. Tek görüntü inceleme
  4. Veri bölüntüleme (Train/Val/Test)
  5. Veri seti kontrol ve istatistikler
  6. CSV analiz ve export
  
Kullanım:
    python scripts/TUMU_ISLEMLER.py

Seçenekler:
    1. Toplu ön işleme
    2. CSV oluşturma ve normalizasyon
    3. Tek görüntü inceleme
    4. Veri bölüntüleme
    5. Veri seti kontrol
    6. CSV analiz ve export
    0. Çıkış
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

# Proje modüllerini import et
sys.path.insert(0, str(Path(__file__).parent.parent))

from goruntu_isleme_mri.ayarlar import (
    GIRDİ_KLASORU,
    CIKTI_KLASORU,
    VERI_ARTIRMA_AKTIF,
    RASTGELE_TOHUM,
)
from goruntu_isleme_mri.io_araclari import (
    rastgele_tohum_ayarla,
    klasor_olustur_yoksa,
    girdi_gorsellerini_listele,
    goruntu_gri_olarak_oku,
    goruntu_dosyaya_kaydet,
)
from goruntu_isleme_mri.on_isleme_adimlari import tek_goruntu_on_isle
from goruntu_isleme_mri.veri_artirma import rastgele_artirma_uygula
from goruntu_isleme_mri.csv_olusturucu import (
    tum_gorseller_icin_csv_olustur,
    istatistikleri_kaydet,
    csv_ye_minmax_scaling_uygula,
)
from goruntu_isleme_mri.veri_boluntuleme import VeriboluntulemeManager


# ========================= BÖLÜM 1: TOPLU ÖN İŞLEME =========================

def menu_toplu_on_isleme():
    """Toplu ön işleme menüsü."""
    print("\n" + "="*70)
    print("1. TOPLU ÖN İŞLEME")
    print("="*70)
    print("\nGirdi klasöründeki tüm MRI görüntülerine ön işleme uygula.")
    print("Çıktı: Normalize edilmiş, boyutlandırılmış görüntüler")
    
    confirm = input("\nDevam etmek istediğinize emin misiniz? (evet/hayır): ").strip().lower()
    if confirm not in ["evet", "yes", "y"]:
        print("[İPTAL]")
        return
    
    try:
        toplu_on_isleme_calistir()
    except Exception as e:
        print(f"[HATA] {str(e)}")
        import traceback
        traceback.print_exc()


def toplu_on_isleme_calistir():
    """Toplu ön işleme işlemini çalıştır."""
    print("\n[BAŞLATILIYOR] Toplu ön işleme başlatılıyor...\n")
    
    rastgele_tohum_ayarla()
    klasor_olustur_yoksa(GIRDİ_KLASORU)
    klasor_olustur_yoksa(CIKTI_KLASORU)
    
    print(f"[BİLGİ] Girdi klasörü: {GIRDİ_KLASORU}")
    print(f"[BİLGİ] Çıktı klasörü: {CIKTI_KLASORU}")
    
    girdi_listesi = girdi_gorsellerini_listele(GIRDİ_KLASORU)
    print(f"[BİLGİ] Toplam {len(girdi_listesi)} adet görüntü bulundu.\n")
    
    if not girdi_listesi:
        print("[UYARI] Girdi klasöründe görüntü bulunamadı!")
        return
    
    log_kayitlari = []
    basarili = 0
    basarisiz = 0
    
    for i, dosya_bilgisi in enumerate(girdi_listesi, start=1):
        girdi_yolu = dosya_bilgisi["path"]
        sinif_adi = dosya_bilgisi["sinif"]
        
        try:
            # Görüntüyü gri tonlamada oku
            goruntu_gri = goruntu_gri_olarak_oku(girdi_yolu)
            
            # Ön işlemeyi uygula
            goruntu_isle, log_bilgisi = tek_goruntu_on_isle(goruntu_gri)
            
            # Çıktı yolunu oluştur
            dosya_adi = os.path.basename(girdi_yolu)
            cikti_yolu = os.path.join(CIKTI_KLASORU, sinif_adi, dosya_adi)
            klasor_olustur_yoksa(os.path.dirname(cikti_yolu))
            
            # Ön işlenmiş görüntüyü kaydet
            goruntu_dosyaya_kaydet(cikti_yolu, goruntu_isle)
            
            # Veri artırma uygulandıysa
            if VERI_ARTIRMA_AKTIF:
                rastgele_artirma_uygula(goruntu_isle, cikti_yolu)
            
            basarili += 1
            
            # Log kaydını ekle
            log_bilgisi["dosya_adi"] = dosya_adi
            log_bilgisi["sinif"] = sinif_adi
            log_kayitlari.append(log_bilgisi)
            
            if i % 10 == 0 or i == len(girdi_listesi):
                print(f"[İLERLEME] {i}/{len(girdi_listesi)} işlendi ({basarili} başarılı, {basarisiz} başarısız)")
        
        except Exception as e:
            basarisiz += 1
            print(f"[UYARI] {girdi_yolu} işlenirken hata: {str(e)}")
    
    # Log dosyasını kaydet
    if log_kayitlari:
        log_csv_yolu = os.path.join(CIKTI_KLASORU, "on_isleme_log.csv")
        on_isleme_log_kaydet(log_kayitlari, log_csv_yolu)
        print(f"\n[BAŞARILI] Log dosyası kaydedildi: {log_csv_yolu}")
    
    print(f"\n[TAMAMLANDI] Toplu ön işleme tamamlandı!")
    print(f"  Başarılı: {basarili}")
    print(f"  Başarısız: {basarisiz}")


# ==================== BÖLÜM 2: CSV OLUŞTURMA VE NORMALIZASYON ====================

def menu_csv_ve_normalizasyon():
    """CSV oluşturma ve normalizasyon menüsü."""
    print("\n" + "="*70)
    print("2. CSV OLUŞTURMA VE NORMALIZASYON")
    print("="*70)
    print("\nSeçenekler:")
    print("  1. CSV oluştur")
    print("  2. Min-Max normalizasyon uygula")
    print("  3. Her ikisini yap (adım adım)")
    print("  0. Geri dön")
    
    secim = input("\nSeçiminiz (0-3): ").strip()
    
    if secim == "1":
        csv_olustur()
    elif secim == "2":
        csv_normalizasyon()
    elif secim == "3":
        csv_tam_islem()
    elif secim != "0":
        print("[HATA] Geçersiz seçim!")


def csv_olustur():
    """CSV dosyası oluştur."""
    print("\n[CSV OLUŞTURMA]")
    print("Ön işlenmiş görüntülerden öznitelikler çıkarılıyor...\n")
    
    try:
        csv_yolu = tum_gorseller_icin_csv_olustur(
            cikti_klasoru=CIKTI_KLASORU,
            csv_dosya_adi="goruntu_ozellikleri.csv"
        )
        
        if os.path.exists(csv_yolu):
            df = pd.read_csv(csv_yolu)
            print(f"[BAŞARILI] CSV oluşturuldu: {csv_yolu}")
            print(f"  Satır: {len(df)}, Sütun: {len(df.columns)}")
            print(f"  Sütunlar: {df.columns.tolist()}")
        else:
            print("[HATA] CSV dosyası oluşturulamadı!")
    
    except Exception as e:
        print(f"[HATA] {str(e)}")


def csv_normalizasyon():
    """CSV'ye normalizasyon uygula."""
    print("\n[CSV NORMALIZASYON]")
    print("Min-Max scaling uygulanıyor...\n")
    
    try:
        csv_dosya = os.path.join(CIKTI_KLASORU, "goruntu_ozellikleri.csv")
        
        if not os.path.exists(csv_dosya):
            print(f"[HATA] CSV dosyası bulunamadı: {csv_dosya}")
            print("Önce CSV oluşturun!")
            return
        
        scaled_csv, stats = csv_ye_minmax_scaling_uygula(csv_dosya)
        print(f"[BAŞARILI] Normalizasyon tamamlandı!")
        print(f"  Çıktı: {scaled_csv}")
        print(f"  Min-Max aralığı: [0, 1]")
    
    except Exception as e:
        print(f"[HATA] {str(e)}")


def csv_tam_islem():
    """CSV oluştur ve normalizasyon uygula."""
    print("\n[ADIM 1/2] CSV oluşturuluyor...")
    csv_olustur()
    
    print("\n[ADIM 2/2] Normalizasyon uygulanıyor...")
    csv_normalizasyon()
    
    print("\n[TAMAMLANDI] Tüm işlemler tamamlandı!")


# ===================== BÖLÜM 3: TEK GÖRÜNTÜ İNCELEME =====================

def menu_tek_goruntu_incele():
    """Tek görüntü inceleme menüsü."""
    print("\n" + "="*70)
    print("3. TEK GÖRÜNTÜ İNCELEME")
    print("="*70)
    print("\nÖn işleme adımlarının etkisini görmek için tek görüntü inceleyin.")
    
    girdi_yolu = input("\nGörüntü yolunu girin (örn: veri/girdi/Nondemented/img.jpg): ").strip()
    
    if not girdi_yolu:
        print("[İPTAL]")
        return
    
    if not os.path.exists(girdi_yolu):
        print(f"[HATA] Görüntü bulunamadı: {girdi_yolu}")
        return
    
    try:
        import matplotlib.pyplot as plt
        
        # Orijinal görüntüyü oku
        goruntu_orijinal = goruntu_gri_olarak_oku(girdi_yolu)
        
        # Ön işleme uygula
        goruntu_isle, log_bilgisi = tek_goruntu_on_isle(goruntu_orijinal)
        
        # Sonuçları göster
        print(f"\n[BAŞARILI] Görüntü işlendi!")
        print(f"  Orijinal boyut: {goruntu_orijinal.shape}")
        print(f"  İşlenmiş boyut: {goruntu_isle.shape}")
        
        # İstatistikleri göster
        print(f"\n[İSTATİSTİKLER]")
        for key, value in log_bilgisi.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value}")
        
        # Görüntüleri yan yana göster
        print("\n[VİZÜALİZASYON] Görüntüler ekranda gösteriliyor...")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(goruntu_orijinal, cmap='gray')
        axes[0].set_title('Orijinal Görüntü')
        axes[0].axis('off')
        
        axes[1].imshow(goruntu_isle, cmap='gray')
        axes[1].set_title('İşlenmiş Görüntü')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    except ImportError:
        print("[UYARI] Matplotlib kurulu değil. Sadece metin çıktısı gösterildi.")
    except Exception as e:
        print(f"[HATA] {str(e)}")


# =================== BÖLÜM 4: VERİ BÖLÜNTÜLEME ==================

def menu_veri_boluntuleme():
    """Veri bölüntüleme menüsü."""
    print("\n" + "="*70)
    print("4. VERİ BÖLÜNTÜLEME")
    print("="*70)
    print("\nSeçenekler:")
    print("  1. Meta veri bölüntüleme (hızlı)")
    print("  2. NumPy array'lere yükle")
    print("  3. TensorFlow Dataset yükle")
    print("  4. PyTorch DataLoader yükle")
    print("  0. Geri dön")
    
    secim = input("\nSeçiminiz (0-4): ").strip()
    
    if secim == "1":
        boluntuleme_meta()
    elif secim == "2":
        boluntuleme_numpy()
    elif secim == "3":
        boluntuleme_tensorflow()
    elif secim == "4":
        boluntuleme_pytorch()
    elif secim != "0":
        print("[HATA] Geçersiz seçim!")


def boluntuleme_meta():
    """Meta veri bölüntüleme."""
    print("\n[META VERİ BÖLÜNTÜLEME]")
    
    csv_dosya = os.path.join(CIKTI_KLASORU, "goruntu_ozellikleri.csv")
    resim_klasoru = CIKTI_KLASORU
    
    if not os.path.exists(csv_dosya):
        print(f"[HATA] CSV dosyası bulunamadı!")
        return
    
    try:
        manager = VeriboluntulemeManager(
            csv_dosya=csv_dosya,
            resim_klasoru=resim_klasoru,
            verbose=True
        )
        
        train_data, val_data, test_data = manager.boluntule(stratified=True)
        istatistikler = manager.boluntuleme_istatistikleri()
        
        manager.istatistikleri_kaydet("boluntuleme_istatistikleri.json")
        
        print(f"\n[BAŞARILI] İstatistikler kaydedildi: boluntuleme_istatistikleri.json")
    
    except Exception as e:
        print(f"[HATA] {str(e)}")


def boluntuleme_numpy():
    """NumPy array'lere yükle."""
    print("\n[NUMPY YÜKLEME]")
    
    csv_dosya = os.path.join(CIKTI_KLASORU, "goruntu_ozellikleri.csv")
    resim_klasoru = CIKTI_KLASORU
    
    if not os.path.exists(csv_dosya):
        print(f"[HATA] CSV dosyası bulunamadı!")
        return
    
    try:
        manager = VeriboluntulemeManager(
            csv_dosya=csv_dosya,
            resim_klasoru=resim_klasoru,
            verbose=True
        )
        
        manager.boluntule(stratified=True)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = manager.veri_seti_olustur()
        
        print(f"\n[BAŞARILI] Veri setleri yüklendi!")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_val: {X_val.shape}")
        print(f"  X_test: {X_test.shape}")
        
        # Kaydet
        np.save("X_train.npy", X_train)
        np.save("y_train.npy", y_train)
        np.save("X_val.npy", X_val)
        np.save("y_val.npy", y_val)
        np.save("X_test.npy", X_test)
        np.save("y_test.npy", y_test)
        
        print(f"✓ NumPy dosyaları kaydedildi")
    
    except Exception as e:
        print(f"[HATA] {str(e)}")


def boluntuleme_tensorflow():
    """TensorFlow Dataset."""
    print("\n[TENSORFLOW DATASET]")
    
    try:
        import tensorflow as tf
    except ImportError:
        print("[HATA] TensorFlow kurulu değil!")
        return
    
    csv_dosya = os.path.join(CIKTI_KLASORU, "goruntu_ozellikleri.csv")
    resim_klasoru = CIKTI_KLASORU
    
    if not os.path.exists(csv_dosya):
        print(f"[HATA] CSV dosyası bulunamadı!")
        return
    
    try:
        manager = VeriboluntulemeManager(
            csv_dosya=csv_dosya,
            resim_klasoru=resim_klasoru,
            verbose=True
        )
        
        manager.boluntule(stratified=True)
        train_ds, val_ds, test_ds = manager.tensorflow_veri_yukle(batch_size=32)
        
        print(f"\n[BAŞARILI] TensorFlow Dataset'leri oluşturuldu!")
    
    except Exception as e:
        print(f"[HATA] {str(e)}")


def boluntuleme_pytorch():
    """PyTorch DataLoader."""
    print("\n[PYTORCH DATALOADER]")
    
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("[HATA] PyTorch kurulu değil!")
        return
    
    csv_dosya = os.path.join(CIKTI_KLASORU, "goruntu_ozellikleri.csv")
    resim_klasoru = CIKTI_KLASORU
    
    if not os.path.exists(csv_dosya):
        print(f"[HATA] CSV dosyası bulunamadı!")
        return
    
    try:
        manager = VeriboluntulemeManager(
            csv_dosya=csv_dosya,
            resim_klasoru=resim_klasoru,
            verbose=True
        )
        
        manager.boluntule(stratified=True)
        train_loader, val_loader, test_loader = manager.pytorch_veri_yukle(batch_size=32)
        
        print(f"\n[BAŞARILI] PyTorch DataLoader'ları oluşturuldu!")
    
    except Exception as e:
        print(f"[HATA] {str(e)}")


# =================== BÖLÜM 5: VERİ SETİ KONTROL ===================

def menu_veri_seti_kontrol():
    """Veri seti kontrol menüsü."""
    print("\n" + "="*70)
    print("5. VERİ SETİ KONTROL VE İSTATİSTİKLER")
    print("="*70)
    print("\nSeçenekler:")
    print("  1. CSV istatistikleri")
    print("  2. Sınıf dağılımı")
    print("  3. Anomali tespiti")
    print("  0. Geri dön")
    
    secim = input("\nSeçiminiz (0-3): ").strip()
    
    if secim == "1":
        csv_istatistikleri()
    elif secim == "2":
        sinif_dagilimi()
    elif secim == "3":
        anomali_tespiti()
    elif secim != "0":
        print("[HATA] Geçersiz seçim!")


def csv_istatistikleri():
    """CSV istatistiklerini göster."""
    print("\n[CSV İSTATİSTİKLERİ]")
    
    csv_dosya = os.path.join(CIKTI_KLASORU, "goruntu_ozellikleri.csv")
    
    if not os.path.exists(csv_dosya):
        print(f"[HATA] CSV dosyası bulunamadı!")
        return
    
    try:
        df = pd.read_csv(csv_dosya)
        
        print(f"\nVeri seti bilgileri:")
        print(f"  Toplam satır: {len(df)}")
        print(f"  Toplam sütun: {len(df.columns)}")
        
        print(f"\nSütunlar:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")
        
        print(f"\nSayısal istatistikler:")
        print(df.describe())
        
        if "etiket" in df.columns:
            print(f"\nSınıf dağılımı:")
            print(df["etiket"].value_counts().sort_index())
    
    except Exception as e:
        print(f"[HATA] {str(e)}")


def sinif_dagilimi():
    """Sınıf dağılımı göster."""
    print("\n[SINIF DAĞILIMI]")
    
    csv_dosya = os.path.join(CIKTI_KLASORU, "goruntu_ozellikleri.csv")
    
    if not os.path.exists(csv_dosya):
        print(f"[HATA] CSV dosyası bulunamadı!")
        return
    
    try:
        df = pd.read_csv(csv_dosya)
        
        if "etiket" in df.columns:
            dagilim = df["etiket"].value_counts().sort_index()
            toplam = len(df)
            
            sinif_adlari = {
                0: "NonDemented",
                1: "VeryMildDemented",
                2: "MildDemented",
                3: "ModerateDemented"
            }
            
            print(f"\nSınıf dağılımı (toplam: {toplam}):")
            for sinif, sayı in dagilim.items():
                yuzde = (sayı / toplam) * 100
                ad = sinif_adlari.get(sinif, f"Sınıf {sinif}")
                print(f"  {ad:20s}: {sayı:4d} ({yuzde:5.1f}%)")
    
    except Exception as e:
        print(f"[HATA] {str(e)}")


def anomali_tespiti():
    """Anomali (outlier) tespit et."""
    print("\n[ANOMALI TESPİTİ]")
    
    csv_dosya = os.path.join(CIKTI_KLASORU, "goruntu_ozellikleri.csv")
    
    if not os.path.exists(csv_dosya):
        print(f"[HATA] CSV dosyası bulunamadı!")
        return
    
    try:
        df = pd.read_csv(csv_dosya)
        
        # Sayısal sütunları bul
        sayisal_sutunlar = df.select_dtypes(include=[np.number]).columns
        
        anomali_sayilari = {}
        
        for sutun in sayisal_sutunlar:
            Q1 = df[sutun].quantile(0.25)
            Q3 = df[sutun].quantile(0.75)
            IQR = Q3 - Q1
            
            alt_limit = Q1 - 1.5 * IQR
            ust_limit = Q3 + 1.5 * IQR
            
            anomali_mask = (df[sutun] < alt_limit) | (df[sutun] > ust_limit)
            anomali_sayilari[sutun] = anomali_mask.sum()
        
        print(f"\nAnomali sayıları (IQR yöntemi):")
        for sutun, sayı in sorted(anomali_sayilari.items(), key=lambda x: x[1], reverse=True):
            if sayı > 0:
                yuzde = (sayı / len(df)) * 100
                print(f"  {sutun:30s}: {sayı:3d} ({yuzde:5.2f}%)")
    
    except Exception as e:
        print(f"[HATA] {str(e)}")


# =================== BÖLÜM 6: CSV ANALİZ VE EXPORT ===================

def menu_csv_analiz():
    """CSV analiz menüsü."""
    print("\n" + "="*70)
    print("6. CSV ANALİZ VE EXPORT")
    print("="*70)
    print("\nSeçenekler:")
    print("  1. CSV analiz istatistikleri")
    print("  2. CSV'yi Excel'e aktar")
    print("  3. CSV'yi JSON'a aktar")
    print("  0. Geri dön")
    
    secim = input("\nSeçiminiz (0-3): ").strip()
    
    if secim == "1":
        csv_ayrinti_analiz()
    elif secim == "2":
        csv_export_excel()
    elif secim == "3":
        csv_export_json()
    elif secim != "0":
        print("[HATA] Geçersiz seçim!")


def csv_ayrinti_analiz():
    """CSV detaylı analizi."""
    print("\n[CSV DETAİLLİ ANALİZİ]")
    
    csv_dosya = os.path.join(CIKTI_KLASORU, "goruntu_ozellikleri.csv")
    
    if not os.path.exists(csv_dosya):
        print(f"[HATA] CSV dosyası bulunamadı!")
        return
    
    try:
        df = pd.read_csv(csv_dosya)
        
        print(f"\nVeri seti özeti:")
        print(f"  Toplam görüntü: {len(df):,}")
        print(f"  Özelliklerin sayısı: {len(df.columns)}")
        
        # Sayısal özniteliklerin dağılımı
        sayisal = df.select_dtypes(include=[np.number])
        
        print(f"\nSayısal özniteliklerin istatistikleri:")
        print(f"  En yüksek korelasyon: {sayisal.corr().values[~np.eye(len(sayisal), dtype=bool)].max():.3f}")
        
        # Bellek kullanımı
        bellek = df.memory_usage(deep=True).sum() / (1024**2)
        print(f"  Bellek kullanımı: {bellek:.2f} MB")
        
        # Eksik veriler
        eksik = df.isnull().sum()
        if eksik.sum() > 0:
            print(f"  Eksik veriler:")
            for col, sayı in eksik[eksik > 0].items():
                print(f"    {col}: {sayı}")
        else:
            print(f"  Eksik veri: Yok")
    
    except Exception as e:
        print(f"[HATA] {str(e)}")


def csv_export_excel():
    """CSV'yi Excel'e aktar."""
    print("\n[EXCEL EXPORT]")
    
    csv_dosya = os.path.join(CIKTI_KLASORU, "goruntu_ozellikleri.csv")
    
    if not os.path.exists(csv_dosya):
        print(f"[HATA] CSV dosyası bulunamadı!")
        return
    
    try:
        try:
            import openpyxl
        except ImportError:
            print("[UYARI] openpyxl kurulu değil. 'pip install openpyxl' ile kurun.")
            return
        
        df = pd.read_csv(csv_dosya)
        excel_dosya = csv_dosya.replace('.csv', '.xlsx')
        df.to_excel(excel_dosya, index=False)
        
        print(f"[BAŞARILI] Excel dosyası oluşturuldu: {excel_dosya}")
    
    except Exception as e:
        print(f"[HATA] {str(e)}")


def csv_export_json():
    """CSV'yi JSON'a aktar."""
    print("\n[JSON EXPORT]")
    
    csv_dosya = os.path.join(CIKTI_KLASORU, "goruntu_ozellikleri.csv")
    
    if not os.path.exists(csv_dosya):
        print(f"[HATA] CSV dosyası bulunamadı!")
        return
    
    try:
        df = pd.read_csv(csv_dosya)
        json_dosya = csv_dosya.replace('.csv', '.json')
        
        df.to_json(json_dosya, orient='records', indent=2)
        
        print(f"[BAŞARILI] JSON dosyası oluşturuldu: {json_dosya}")
    
    except Exception as e:
        print(f"[HATA] {str(e)}")


# ====================== ANA MENÜ ======================

def ana_menu():
    """Ana menü."""
    print("\n" + "="*70)
    print(" "*15 + "TÜM İŞLEMLER - ANA MENU")
    print("="*70)
    print("\nSeçenekler:")
    print("  1. Toplu ön işleme")
    print("  2. CSV oluşturma ve normalizasyon")
    print("  3. Tek görüntü inceleme")
    print("  4. Veri bölüntüleme")
    print("  5. Veri seti kontrol")
    print("  6. CSV analiz ve export")
    print("  0. Çıkış")
    print("-"*70)
    
    secim = input("Seçiminiz (0-6): ").strip()
    return secim


def main():
    """Ana işlev."""
    print("\n" + "="*70)
    print(" "*10 + "MRI KLASİFİKASYON - TÜM İŞLEMLER")
    print("="*70)
    
    while True:
        secim = ana_menu()
        
        if secim == "1":
            menu_toplu_on_isleme()
        elif secim == "2":
            menu_csv_ve_normalizasyon()
        elif secim == "3":
            menu_tek_goruntu_incele()
        elif secim == "4":
            menu_veri_boluntuleme()
        elif secim == "5":
            menu_veri_seti_kontrol()
        elif secim == "6":
            menu_csv_analiz()
        elif secim == "0":
            print("\n[ÇIKILIYOR] Program sonlandırılıyor...")
            break
        else:
            print("[HATA] Geçersiz seçim! Lütfen 0-6 arasında bir sayı girin.")
        
        input("\nDevam etmek için Enter tuşuna basınız...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[DURDURULDU] Program kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\n[HATA] Beklenmeyen hata: {str(e)}")
        import traceback
        traceback.print_exc()
