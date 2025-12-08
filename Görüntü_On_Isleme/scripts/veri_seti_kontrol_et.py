"""
veri_seti_kontrol_et.py
-----------------------
Veri seti doğrulama ve istatistik raporlama script'i.
Oluşturulan veri setinin bütünlüğü ve dağılımını kontrol eder.
"""

import sys
from pathlib import Path

# Proje modüllerini import et
sys.path.insert(0, str(Path(__file__).parent.parent))

from goruntu_isleme_mri.ayarlar import VERI_KLASORU
from goruntu_isleme_mri.dosya_yoneticisi import DosyaYoneticisi, VeriSeti


def ana_menu():
    """Ana menüyü göster."""
    print("\n" + "="*60)
    print("VERİ SETİ KONTROL VE İSTATİSTİK")
    print("="*60)
    print("\nSeçenekler:")
    print("1. Veri seti istatistiklerini göster")
    print("2. Veri seti doğrulaması yap")
    print("3. Klasör boyutlarını kontrol et")
    print("4. Tüm dosya istatistiklerini göster")
    print("0. Çıkış")
    print("\n" + "-"*60)
    
    secim = input("Seçiminiz (0-4): ").strip()
    return secim


def veri_seti_istatistikleri_goster():
    """Veri seti istatistiklerini göster."""
    print("\n[VERİ SETİ İSTATİSTİKLERİ]")
    print("-" * 40)
    
    veri_seti_yolu = input("Veri seti yolu (varsayılan: veri/veri_seti): ").strip()
    if not veri_seti_yolu:
        veri_seti_yolu = VERI_KLASORU / "veri_seti"
    
    veri_seti = VeriSeti(str(veri_seti_yolu))
    veri_seti.istatistikleri_yazdir()


def veri_seti_dogulamasi_yap():
    """Veri seti dogulamasi yap."""
    print("\n[VERİ SETİ DOĞRULAMASI]")
    print("-" * 40)
    
    veri_seti_yolu = input("Veri seti yolu (varsayılan: veri/veri_seti): ").strip()
    if not veri_seti_yolu:
        veri_seti_yolu = VERI_KLASORU / "veri_seti"
    
    veri_seti = VeriSeti(str(veri_seti_yolu))
    veri_seti.dogrulama_raporu_yazdir()


def klasor_boyutlarini_kontrol_et():
    """Klasör boyutlarını kontrol et."""
    print("\n[KLASÖR BOYUTLARI]")
    print("-" * 40)
    
    veri_seti_yolu = input("Veri seti yolu (varsayılan: veri/veri_seti): ").strip()
    if not veri_seti_yolu:
        veri_seti_yolu = VERI_KLASORU / "veri_seti"
    
    veri_seti_yolu = Path(veri_seti_yolu)
    
    if not veri_seti_yolu.exists():
        print(f"[HATA] Veri seti klasörü bulunamadı: {veri_seti_yolu}")
        return
    
    print(f"\nVeri seti ana klasörü: {veri_seti_yolu}")
    print("-" * 40)
    
    toplam_boyut = 0
    
    for item in sorted(veri_seti_yolu.iterdir()):
        if not item.is_dir():
            continue
        
        boyut = DosyaYoneticisi.klasor_boyutu_hesapla(str(item))
        boyut_str = DosyaYoneticisi.boyutu_insan_okunabilir_formata_cevir(boyut)
        
        print(f"  {item.name}: {boyut_str}")
        toplam_boyut += boyut
    
    toplam_boyut_str = DosyaYoneticisi.boyutu_insan_okunabilir_formata_cevir(toplam_boyut)
    print("-" * 40)
    print(f"Toplam boyut: {toplam_boyut_str}")


def tum_dosya_istatistiklerini_goster():
    """Tüm dosya istatistiklerini göster."""
    print("\n[DOSYA İSTATİSTİKLERİ]")
    print("-" * 40)
    
    veri_seti_yolu = input("Veri seti yolu (varsayılan: veri/veri_seti): ").strip()
    if not veri_seti_yolu:
        veri_seti_yolu = VERI_KLASORU / "veri_seti"
    
    istatistikler = DosyaYoneticisi.dosya_istatistikleri_al(str(veri_seti_yolu))
    DosyaYoneticisi.dosya_istatistiklerini_yazdir(istatistikler)


def main():
    """Ana işlev."""
    while True:
        secim = ana_menu()
        
        if secim == "1":
            veri_seti_istatistikleri_goster()
        elif secim == "2":
            veri_seti_dogulamasi_yap()
        elif secim == "3":
            klasor_boyutlarini_kontrol_et()
        elif secim == "4":
            tum_dosya_istatistiklerini_goster()
        elif secim == "0":
            print("\n[ÇIKILIYOR] Program sonlandırılıyor...")
            break
        else:
            print("[HATA] Geçersiz seçim! Lütfen 0-4 arasında bir sayı girin.")


if __name__ == "__main__":
    main()
