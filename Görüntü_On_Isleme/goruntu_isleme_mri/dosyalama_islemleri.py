"""
dosyalama_islemleri.py
----------------------
Görüntülerin sınıflara göre dosyalanması işlemleri.
Verilerin organize edilmesi ve bölüntü ayırması.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple


class DosyalamaIslemleri:
    """Görüntü dosyalama işlemleri için yardımcı sınıf."""
    
    @staticmethod
    def giris_klasoru_al(varsayilan: str = None) -> str:
        """
        Kullanıcıdan giriş klasörü al.
        
        Parametreler:
        - varsayilan: Varsayılan klasör yolu
        
        Çıktı: Seçilen klasör yolu
        """
        if varsayilan:
            prompt = f"Giriş klasörü (varsayılan: {varsayilan}): "
        else:
            prompt = "Giriş klasörü: "
        
        giris = input(prompt).strip()
        
        if not giris and varsayilan:
            return str(varsayilan)
        
        return giris
    
    @staticmethod
    def cikti_klasoru_al(varsayilan: str = None) -> str:
        """
        Kullanıcıdan çıktı klasörü al.
        
        Parametreler:
        - varsayilan: Varsayılan klasör yolu
        
        Çıktı: Seçilen klasör yolu
        """
        if varsayilan:
            prompt = f"Çıktı klasörü (varsayılan: {varsayilan}): "
        else:
            prompt = "Çıktı klasörü: "
        
        cikti = input(prompt).strip()
        
        if not cikti and varsayilan:
            return str(varsayilan)
        
        return cikti
    
    @staticmethod
    def log_csv_dosyasi_al() -> str:
        """
        Kullanıcıdan ön işleme log CSV dosyası al.
        
        Çıktı: Log dosyasının yolu (None ise otomatik arama)
        """
        log_csv = input("Log CSV dosyası (varsayılan: otomatik arama): ").strip()
        return log_csv if log_csv else None
    
    @staticmethod
    def oranlar_al() -> Tuple[float, float, float]:
        """
        Kullanıcıdan veri seti oranlarını al.
        
        Çıktı: (egitim_oran, dogrulama_oran, test_oran) tuple'ı
        """
        print("\nOranlar (varsayılan: Eğitim=70%, Doğrulama=15%, Test=15%):")
        
        egitim_oran_str = input("Eğitim oranı (0.0-1.0): ").strip()
        egitim_oran = float(egitim_oran_str) if egitim_oran_str else 0.70
        
        dogrulama_oran_str = input("Doğrulama oranı (0.0-1.0): ").strip()
        dogrulama_oran = float(dogrulama_oran_str) if dogrulama_oran_str else 0.15
        
        test_oran_str = input("Test oranı (0.0-1.0): ").strip()
        test_oran = float(test_oran_str) if test_oran_str else 0.15
        
        # Oranları doğrula
        toplam_oran = egitim_oran + dogrulama_oran + test_oran
        if abs(toplam_oran - 1.0) > 1e-6:
            print(f"\n[HATA] Oranların toplamı 1.0 olmalı (toplam: {toplam_oran})")
            raise ValueError(f"Oranların toplamı 1.0 değil: {toplam_oran}")
        
        return egitim_oran, dogrulama_oran, test_oran
    
    @staticmethod
    def sinif_haritasi_olustur() -> Dict[str, int]:
        """
        Sınıf haritasını oluştur.
        
        Çıktı: Sınıf adı -> etiket haritası
        """
        return {
            "NonDemented": 0,
            "VeryMildDemented": 1,
            "MildDemented": 2,
            "ModerateDemented": 3,
        }
    
    @staticmethod
    def dosyalama_ozeti_yazdir(giris: str, cikti: str, oranlar: Tuple[float, float, float]):
        """
        Dosyalama işleminin özetini yazdır.
        
        Parametreler:
        - giris: Giriş klasörü
        - cikti: Çıktı klasörü
        - oranlar: (egitim_oran, dogrulama_oran, test_oran)
        """
        print("\n" + "="*60)
        print("DOSYALAMA İŞLEMİ ÖZETİ")
        print("="*60)
        print(f"\nGiriş klasörü: {giris}")
        print(f"Çıktı klasörü: {cikti}")
        print(f"\nVeri seti oranları:")
        print(f"  Eğitim: {oranlar[0]*100:.0f}%")
        print(f"  Doğrulama: {oranlar[1]*100:.0f}%")
        print(f"  Test: {oranlar[2]*100:.0f}%")
        print("\n" + "-"*60)
    
    @staticmethod
    def onayla_devam_et(mesaj: str = "Devam etmek istiyor musunuz?") -> bool:
        """
        Kullanıcıdan onay al.
        
        Parametreler:
        - mesaj: Gösterilecek mesaj
        
        Çıktı: Onay (True/False)
        """
        onay = input(f"\n{mesaj} (e/H): ").strip().lower()
        return onay == "e"


class HizliDosyalamaMenu:
    """Hızlı dosyalama menüsü işlemleri."""
    
    def __init__(self, veri_klasoru):
        self.veri_klasoru = veri_klasoru
        self.islemler = DosyalamaIslemleri()
    
    def calistir(self):
        """Hızlı dosyalama menüsünü çalıştır."""
        print("\n[HIZLI DOSYALAMA]")
        print("-" * 40)
        
        # Giriş ve çıktı klasörlerini al
        giris_klasoru = self.islemler.giris_klasoru_al(
            varsayilan=self.veri_klasoru / "cikti"
        )
        cikti_klasoru = self.islemler.cikti_klasoru_al(
            varsayilan=self.veri_klasoru / "veri_seti"
        )
        
        # Log dosyasını al
        log_csv = self.islemler.log_csv_dosyasi_al()
        
        # Oranları al
        egitim_oran, dogrulama_oran, test_oran = self.islemler.oranlar_al()
        
        # Özeti yazdır
        self.islemler.dosyalama_ozeti_yazdir(
            giris_klasoru, cikti_klasoru, 
            (egitim_oran, dogrulama_oran, test_oran)
        )
        
        # Onay al
        if not self.islemler.onayla_devam_et("Dosyalama işlemine başlansın mı?"):
            print("[İPTAL] İşlem iptal edildi")
            return None
        
        return {
            "giris_klasoru": str(giris_klasoru),
            "cikti_klasoru": str(cikti_klasoru),
            "log_csv": log_csv,
            "egitim_oran": egitim_oran,
            "dogrulama_oran": dogrulama_oran,
            "test_oran": test_oran,
        }


class AdimAdimMenu:
    """Adım adım dosyalama menüsü işlemleri."""
    
    def __init__(self):
        self.islemler = DosyalamaIslemleri()
    
    def dosyalayici_olustur(self):
        """Dosyalayıcı oluştur ve adım adım menüsünü başlat."""
        from goruntu_isleme_mri.veri_dosyalama import VeriDosyalayici
        
        print("\n[ADIM ADIM DOSYALAMA]")
        print("-" * 40)
        
        # Giriş klasörü
        giris_klasoru = self.islemler.giris_klasoru_al()
        if not giris_klasoru:
            print("[HATA] Giriş klasörü zorunludur!")
            return None
        
        # Çıktı klasörü
        cikti_klasoru = self.islemler.cikti_klasoru_al()
        if not cikti_klasoru:
            print("[HATA] Çıktı klasörü zorunludur!")
            return None
        
        # Dosyalayıcı oluştur
        dosyalayici = VeriDosyalayici(
            giris_klasoru=giris_klasoru,
            cikti_klasoru=cikti_klasoru,
            sinif_haritasi=self.islemler.sinif_haritasi_olustur()
        )
        
        return dosyalayici
    
    def adim_secim_al(self) -> str:
        """
        Adım seçimini al.
        
        Çıktı: Seçim (0-4)
        """
        print("\nAdımlar:")
        print("1. Klasör yapısını oluştur")
        print("2. Görüntüleri sınıf klasörlerine dosyala")
        print("3. Eğitim/doğrulama/test olarak ayır")
        print("4. Veri seti bilgisini kaydet")
        print("0. Geri dön")
        
        secim = input("\nAdım seçimi (0-4): ").strip()
        return secim
    
    def calistir(self):
        """Adım adım menüsünü çalıştır."""
        dosyalayici = self.dosyalayici_olustur()
        
        if dosyalayici is None:
            return
        
        while True:
            secim = self.adim_secim_al()
            
            if secim == "1":
                print("\n[ADIM 1] Klasör yapısı oluşturuluyor...")
                dosyalayici.dosya_yapisi_olustur()
                
            elif secim == "2":
                print("\n[ADIM 2] Görüntüler dosyalanıyor...")
                log_csv = self.islemler.log_csv_dosyasi_al()
                dosyalayici.gorselleri_sinif_klasorlerine_dosyala(
                    log_csv=log_csv if log_csv else None
                )
                
            elif secim == "3":
                print("\n[ADIM 3] Veri setleri ayrılıyor...")
                egitim_oran, dogrulama_oran, test_oran = self.islemler.oranlar_al()
                dosyalayici.egitim_dogrulama_test_ayir(
                    egitim_oran=egitim_oran,
                    dogrulama_oran=dogrulama_oran,
                    test_oran=test_oran
                )
                
            elif secim == "4":
                print("\n[ADIM 4] Veri seti bilgisi kaydediliyor...")
                istatistikler = {
                    "egitim": {},
                    "dogrulama": {},
                    "test": {},
                }
                dosyalayici.veri_seti_bilgisi_kaydet(istatistikler)
                
            elif secim == "0":
                break
            else:
                print("[HATA] Geçersiz seçim!")


class MevcutVeriSetiniReorganizeEt:
    """Mevcut veri setini reorganize etme işlemleri."""
    
    def __init__(self):
        self.islemler = DosyalamaIslemleri()
    
    def calistir(self):
        """Reorganize etme işlemini çalıştır."""
        from goruntu_isleme_mri.veri_dosyalama import VeriDosyalayici
        
        print("\n[MEVCUT VERİ SETİNİ YENİDEN ORGANIZE ET]")
        print("-" * 40)
        
        veri_seti_yolu = self.islemler.giris_klasoru_al()
        if not veri_seti_yolu:
            print("[HATA] Veri seti klasörü zorunludur!")
            return
        
        cikti_klasoru = self.islemler.cikti_klasoru_al()
        if not cikti_klasoru:
            print("[HATA] Çıktı klasörü zorunludur!")
            return
        
        dosyalayici = VeriDosyalayici(
            giris_klasoru=veri_seti_yolu,
            cikti_klasoru=cikti_klasoru,
            sinif_haritasi=self.islemler.sinif_haritasi_olustur()
        )
        
        try:
            istatistikler = dosyalayici.sinif_klasorlerinden_yeniden_organize_et(veri_seti_yolu)
            print("\n[BAŞARILI] Veri seti yeniden organize edildi!")
            print(f"İstatistikler: {istatistikler}")
            
        except Exception as e:
            print(f"\n[HATA] İşlem sırasında hata oluştu: {str(e)}")
            import traceback
            traceback.print_exc()
