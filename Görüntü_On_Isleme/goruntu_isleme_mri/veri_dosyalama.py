"""
veri_dosyalama.py
-----------------
Ön işlenmiş görüntüleri sınıflarına göre otomatik dosyalama işlemleri.
Veri seti yapısını oluşturma ve yönetme.
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


class VeriDosyalayici:
    """Ön işlenmiş görüntüleri sınıflara göre dosyala ve veri seti oluştur."""
    
    def __init__(self, 
                 giris_klasoru: str,
                 cikti_kok_klasoru: str,
                 sinif_haritasi: Dict[str, int] = None):
        """
        Başlatma.
        
        Parametreler:
        - giris_klasoru: Ön işlenmiş görüntülerin bulunduğu klasör
        - cikti_kok_klasoru: Çıktı veri setinin oluşturulacağı ana klasör
        - sinif_haritasi: Sınıf adı -> etiket haritası
        """
        self.giris_klasoru = Path(giris_klasoru)
        self.cikti_kok_klasoru = Path(cikti_kok_klasoru)
        self.sinif_haritasi = sinif_haritasi or {}
        
        # Ters harita oluştur (etiket -> sınıf adı)
        self.ters_sinif_haritasi = {v: k for k, v in self.sinif_haritasi.items()}
    
    def dosya_yapisi_olustur(self, sinif_listesi: List[str] = None) -> Dict:
        """
        Veri seti klasor yapisini olustur.
        
        Yapı:
        cikti_kok_klasoru/
            eğitim/
                NonDemented/
                VeryMildDemented/
                ...
            doğrulama/
                NonDemented/
                ...
            test/
                NonDemented/
                ...
            tüm_veriler/
                NonDemented/
                ...
        """
        if sinif_listesi is None:
            sinif_listesi = list(self.sinif_haritasi.keys())
        
        klasor_yapisi = {}
        bolumler = ["eğitim", "doğrulama", "test", "tüm_veriler"]
        
        for bolum in bolumler:
            bolum_yolu = self.cikti_kok_klasoru / bolum
            klasor_yapisi[bolum] = {}
            
            for sinif in sinif_listesi:
                sinif_yolu = bolum_yolu / sinif
                sinif_yolu.mkdir(parents=True, exist_ok=True)
                klasor_yapisi[bolum][sinif] = str(sinif_yolu)
        
        print(f"[TAMAMLANDI] Veri seti klasor yapisi olusturuldu:")
        print(f"  Konum: {self.cikti_kok_klasoru}")
        for bolum in bolumler:
            print(f"  ├─ {bolum}/")
            for sinif in sinif_listesi:
                print(f"  │  └─ {sinif}/")
        
        return klasor_yapisi
    
    def gorselleri_sinif_klasorlerine_dosyala(self, log_csv: str = None) -> Dict:
        """
        Ön işlenmiş görüntüleri sınıf klasörlerine dosyala.
        
        Parametreler:
        - log_csv: Ön işleme log dosyası (sinif ve etiket bilgisi için)
        
        Çıktı: İstatistik sözlüğü
        """
        print("[BILGI] Görüntüler sınıf klasörlerine dosyalanıyor...")
        
        # Log dosyasından sınıf bilgisini oku
        sinif_bilgileri = self._log_csv_oku(log_csv)
        
        # "tüm_veriler" klasörü oluştur
        tum_veriler_yolu = self.cikti_kok_klasoru / "tüm_veriler"
        tum_veriler_yolu.mkdir(parents=True, exist_ok=True)
        
        istatistikler = {}
        
        for sinif, etiket in self.sinif_haritasi.items():
            sinif_klasoru = tum_veriler_yolu / sinif
            sinif_klasoru.mkdir(parents=True, exist_ok=True)
            istatistikler[sinif] = {"toplam": 0, "dosyalanan": 0, "hata": 0}
        
        # Tüm ön işlenmiş görüntüleri tara
        for dosya in sorted(self.giris_klasoru.rglob("*")):
            if not dosya.is_file():
                continue
            
            # Dosya uzantısını kontrol et
            if dosya.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            
            # Log'dan sınıf bilgisini al
            sinif = sinif_bilgileri.get(dosya.name)
            
            if sinif is None:
                print(f"[UYARI] Sınıf bilgisi bulunamadı: {dosya.name}")
                continue
            
            if sinif not in istatistikler:
                print(f"[UYARI] Bilinmeyen sınıf: {sinif}")
                continue
            
            # Dosyayı hedef klasöre kopyala
            hedef_yolu = (tum_veriler_yolu / sinif / dosya.name)
            
            try:
                shutil.copy2(dosya, hedef_yolu)
                istatistikler[sinif]["toplam"] += 1
                istatistikler[sinif]["dosyalanan"] += 1
            except Exception as e:
                print(f"[HATA] Dosya kopyalanamadı {dosya.name}: {str(e)}")
                istatistikler[sinif]["hata"] += 1
        
        print("[TAMAMLANDI] Görüntüler sınıf klasörlerine dosyalandı:")
        for sinif, stats in istatistikler.items():
            print(f"  {sinif}: {stats['dosyalanan']}/{stats['toplam']} dosya")
            if stats['hata'] > 0:
                print(f"    Hatalar: {stats['hata']}")
        
        return istatistikler
    
    def egitim_dogrulama_test_ayir(self,
                                    egitim_oran: float = 0.7,
                                    dogrulama_oran: float = 0.15,
                                    test_oran: float = 0.15,
                                    rastgele_tohum: int = 42) -> Dict:
        """
        Tüm veriler klasörünü eğitim/doğrulama/test olarak ayır.
        
        Parametreler:
        - egitim_oran: Eğitim seti oranı (default: 0.70)
        - dogrulama_oran: Doğrulama seti oranı (default: 0.15)
        - test_oran: Test seti oranı (default: 0.15)
        - rastgele_tohum: Tekrarlanabilirlik için
        
        Çıktı: Bölümlere göre istatistikler
        """
        print("[BILGI] Veri setleri eğitim/doğrulama/test olarak ayrılıyor...")
        
        # Oranların toplamını kontrol et
        toplam_oran = egitim_oran + dogrulama_oran + test_oran
        if abs(toplam_oran - 1.0) > 1e-6:
            raise ValueError(f"Oranların toplamı 1.0 olmalı, fakat {toplam_oran}")
        
        tum_veriler_yolu = self.cikti_kok_klasoru / "tüm_veriler"
        
        istatistikler = {
            "eğitim": {},
            "doğrulama": {},
            "test": {},
        }
        
        # Her sınıf için ayrım yap
        for sinif in self.sinif_haritasi.keys():
            sinif_klasoru = tum_veriler_yolu / sinif
            
            if not sinif_klasoru.exists():
                print(f"[UYARI] Sınıf klasörü bulunamadı: {sinif}")
                continue
            
            # Sınıftaki tüm dosyaları al
            dosyalar = sorted([f for f in sinif_klasoru.glob("*") if f.is_file()])
            
            if len(dosyalar) == 0:
                print(f"[UYARI] {sinif} sınıfında dosya bulunamadı")
                continue
            
            # Rastgele karıştur
            import random
            random.seed(rastgele_tohum)
            random.shuffle(dosyalar)
            
            # Bölüntü noktalarını hesapla
            n = len(dosyalar)
            egitim_indis = int(n * egitim_oran)
            dogrulama_indis = egitim_indis + int(n * dogrulama_oran)
            
            egitim_dosyalari = dosyalar[:egitim_indis]
            dogrulama_dosyalari = dosyalar[egitim_indis:dogrulama_indis]
            test_dosyalari = dosyalar[dogrulama_indis:]
            
            # Dosyaları hedef klasörlere taşı
            for dosya in egitim_dosyalari:
                hedef = self.cikti_kok_klasoru / "eğitim" / sinif / dosya.name
                shutil.move(str(dosya), str(hedef))
            
            for dosya in dogrulama_dosyalari:
                hedef = self.cikti_kok_klasoru / "doğrulama" / sinif / dosya.name
                shutil.move(str(dosya), str(hedef))
            
            for dosya in test_dosyalari:
                hedef = self.cikti_kok_klasoru / "test" / sinif / dosya.name
                shutil.move(str(dosya), str(hedef))
            
            # İstatistikleri kaydet
            istatistikler["eğitim"][sinif] = len(egitim_dosyalari)
            istatistikler["doğrulama"][sinif] = len(dogrulama_dosyalari)
            istatistikler["test"][sinif] = len(test_dosyalari)
        
        # Sonuçları yazdır
        print("[TAMAMLANDI] Veri setleri ayrıldı:")
        for bolum, sinif_stats in istatistikler.items():
            toplam = sum(sinif_stats.values())
            print(f"  {bolum.capitalize()}: {toplam} dosya")
            for sinif, adet in sinif_stats.items():
                print(f"    {sinif}: {adet}")
        
        return istatistikler
    
    def veri_seti_bilgisi_kaydet(self, 
                                  istatistikler: Dict,
                                  cikti_dosyasi: str = "veri_seti_bilgisi.json") -> str:
        """
        Veri seti bilgilerini JSON dosyasına kaydet.
        
        Parametreler:
        - istatistikler: Bölümlere göre istatistikler
        - cikti_dosyasi: Çıktı JSON dosyasının adı
        
        Çıktı: Kaydedilen dosyasının tam yolu
        """
        veri_seti_bilgisi = {
            "oluşturulma_tarihi": pd.Timestamp.now().isoformat(),
            "giriş_klasörü": str(self.giris_klasoru),
            "çıktı_klasörü": str(self.cikti_kok_klasoru),
            "sınıf_haritası": self.sinif_haritasi,
            "istatistikler": istatistikler,
        }
        
        cikti_yolu = self.cikti_kok_klasoru / cikti_dosyasi
        
        with open(cikti_yolu, "w", encoding="utf-8") as f:
            json.dump(veri_seti_bilgisi, f, ensure_ascii=False, indent=2)
        
        print(f"[KAYDEDILDI] Veri seti bilgisi: {cikti_yolu}")
        return str(cikti_yolu)
    
    def _log_csv_oku(self, log_csv: str = None) -> Dict[str, str]:
        """
        Ön işleme log CSV dosyasından sınıf bilgisini oku.
        
        Çıktı: dosya_adı -> sınıf_adı haritası
        """
        sinif_bilgileri = {}
        
        if log_csv is None:
            log_csv = self.giris_klasoru / "on_isleme_log.csv"
        
        log_csv = Path(log_csv)
        
        if not log_csv.exists():
            print(f"[UYARI] Log dosyası bulunamadı: {log_csv}")
            print("[BILGI] Sınıf bilgisi dosya yolundan çıkarılacak...")
            return {}
        
        try:
            df = pd.read_csv(log_csv)
            
            # log'da 'sinif' kolonu varsa kullan
            if "sinif" in df.columns and "cikti_yolu" in df.columns:
                for _, row in df.iterrows():
                    cikti_yolu = Path(row["cikti_yolu"])
                    dosya_adi = cikti_yolu.name
                    sinif = row["sinif"]
                    sinif_bilgileri[dosya_adi] = sinif
            
            print(f"[BILGI] Log dosyasından {len(sinif_bilgileri)} sınıf bilgisi okundu")
            
        except Exception as e:
            print(f"[HATA] Log dosyası okunamadı: {str(e)}")
        
        return sinif_bilgileri
    
    def sinif_klasorlerinden_yeniden_olustur(self,
                                             giris_veri_seti_yolu: str) -> Dict:
        """
        Zaten sınıf klasörlerine dosyalanmış verileri yeniden organize et.
        
        Parametreler:
        - giris_veri_seti_yolu: Sınıf klasörlerinin bulunduğu ana klasör
        
        Çıktı: İstatistikler
        """
        print(f"[BILGI] Mevcut veri seti yeniden organize ediliyor: {giris_veri_seti_yolu}")
        
        giris_yolu = Path(giris_veri_seti_yolu)
        istatistikler = {}
        
        # Her sınıf klasörünü tara
        for sinif_klasoru in giris_yolu.iterdir():
            if not sinif_klasoru.is_dir():
                continue
            
            sinif_adi = sinif_klasoru.name
            if sinif_adi not in self.sinif_haritasi:
                print(f"[UYARI] Tanınmayan sınıf: {sinif_adi}")
                continue
            
            # Dosyaları say
            dosya_sayisi = len(list(sinif_klasoru.glob("*")))
            istatistikler[sinif_adi] = dosya_sayisi
            print(f"  {sinif_adi}: {dosya_sayisi} dosya")
        
        return istatistikler


def hizli_dosyalama(giris_klasoru: str,
                    cikti_klasoru: str,
                    log_csv: str = None,
                    egitim_oran: float = 0.7,
                    dogrulama_oran: float = 0.15,
                    test_oran: float = 0.15,
                    sinif_haritasi: Dict[str, int] = None):
    """
    Hızlı dosyalama işlemini bir fonksiyonda gerçekleştir.
    
    Parametreler:
    - giris_klasoru: Ön işlenmiş görüntülerin bulunduğu klasör
    - cikti_klasoru: Çıktı veri setinin oluşturulacağı klasör
    - log_csv: Ön işleme log dosyası
    - egitim_oran: Eğitim seti oranı
    - dogrulama_oran: Doğrulama seti oranı
    - test_oran: Test seti oranı
    - sinif_haritasi: Sınıf adı -> etiket haritası
    """
    if sinif_haritasi is None:
        sinif_haritasi = {
            "NonDemented": 0,
            "VeryMildDemented": 1,
            "MildDemented": 2,
            "ModerateDemented": 3,
        }
    
    # Dosyalayici olustur
    dosyalayici = VeriDosyalayici(giris_klasoru, cikti_klasoru, sinif_haritasi)
    
    # Klasor yapisini olustur
    dosyalayici.dosya_yapisi_olustur()
    
    # Goruntuleri dosyala
    istatistikler_dosyalama = dosyalayici.gorselleri_sinif_klasorlerine_dosyala(log_csv)
    
    # Eğitim/doğrulama/test olarak ayır
    istatistikler_ayirma = dosyalayici.egitim_dogrulama_test_ayir(
        egitim_oran=egitim_oran,
        dogrulama_oran=dogrulama_oran,
        test_oran=test_oran
    )
    
    # Bilgileri kaydet
    dosyalayici.veri_seti_bilgisi_kaydet(istatistikler_ayirma)
    
    print("[BAŞARILI] Tüm işlemler tamamlandı!")
    return dosyalayici, istatistikler_dosyalama, istatistikler_ayirma
