"""
verileri_dosyala.py
-------------------
Ön işlenmiş görüntüleri sınıflarına göre otomatik dosyalayan script.
Eğitim, doğrulama ve test setlerine ayrılan veri seti oluşturur.
"""

import sys
from pathlib import Path

# Proje modüllerini import et
sys.path.insert(0, str(Path(__file__).parent))

from goruntu_isleme_mri.ayarlar import VERI_KLASORU
from goruntu_isleme_mri.veri_dosyalama import hizli_dosyalama
from goruntu_isleme_mri.dosyalama_islemleri import (
    HizliDosyalamaMenu,
    AdimAdimMenu,
    MevcutVeriSetiniReorganizeEt
)


def ana_menu():
    """Ana menüyü göster ve kullanıcı seçimini al."""
    print("\n" + "="*60)
    print("VERİ DOSYALAMA VE VERI SETİ OLUŞTURMA")
    print("="*60)
    print("\nSeçenekler:")
    print("1. Hızlı dosyalama (önerilen)")
    print("2. Adım adım dosyalama (özelleştirilmiş)")
    print("3. Mevcut veri setini yeniden organize et")
    print("0. Çıkış")
    print("\n" + "-"*60)
    
    secim = input("Seçiminiz (0-3): ").strip()
    return secim


def main():
    """Ana işlev."""
    while True:
        secim = ana_menu()
        
        if secim == "1":
            menu = HizliDosyalamaMenu(VERI_KLASORU)
            parametreler = menu.calistir()
            
            if parametreler:
                print("\n[BAŞLATILIYOR] Dosyalama işlemi başlatılıyor...")
                try:
                    hizli_dosyalama(
                        giris_klasoru=parametreler["giris_klasoru"],
                        cikti_klasoru=parametreler["cikti_klasoru"],
                        log_csv=parametreler["log_csv"],
                        egitim_oran=parametreler["egitim_oran"],
                        dogrulama_oran=parametreler["dogrulama_oran"],
                        test_oran=parametreler["test_oran"]
                    )
                    print("\n[BAŞARILI] Tüm işlemler tamamlandı!")
                except Exception as e:
                    print(f"\n[HATA] İşlem sırasında hata oluştu: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
        elif secim == "2":
            menu = AdimAdimMenu()
            menu.calistir()
            
        elif secim == "3":
            menu = MevcutVeriSetiniReorganizeEt()
            menu.calistir()
            
        elif secim == "0":
            print("\n[ÇIKILIYOR] Program sonlandırılıyor...")
            break
        else:
            print("[HATA] Geçersiz seçim! Lütfen 0-3 arasında bir sayı girin.")


if __name__ == "__main__":
    main()
