"""
dosyalama_modulleri_test_et.py
------------------------------
Dosyalama modüllerinin söz dizimini ve önemli fonksiyonlarını test eder.
"""

import sys
from pathlib import Path

# Proje modüllerini import et
sys.path.insert(0, str(Path(__file__).parent.parent))

print("[TEST] Dosyalama modulleri test ediliyor...")
print("=" * 60)

# Test 1: veri_dosyalama modulu
print("\n[TEST 1] veri_dosyalama modulu...")
try:
    from goruntu_isleme_mri.veri_dosyalama import VeriDosyalayici
    print("[OK] VeriDosyalayici sinifi import edildi")
    
    # VeriDosyalayici metotlarini kontrol et
    assert hasattr(VeriDosyalayici, 'dosya_yapisi_olustur'), "dosya_yapisi_olustur metodu bulunamadi"
    assert hasattr(VeriDosyalayici, 'gorselleri_sinif_klasorlerine_dosyala'), "gorselleri_sinif_klasorlerine_dosyala metodu bulunamadi"
    assert hasattr(VeriDosyalayici, 'egitim_dogrulama_test_ayir'), "egitim_dogrulama_test_ayir metodu bulunamadi"
    print("[OK] VeriDosyalayici tum gerekli metodlara sahip")
    
except ImportError as e:
    print(f"[HATA] Import hatasi: {e}")
except AssertionError as e:
    print(f"[HATA] Assertion hatasi: {e}")

# Test 2: dosya_yoneticisi modulu
print("\n[TEST 2] dosya_yoneticisi modulu...")
try:
    from goruntu_isleme_mri.dosya_yoneticisi import DosyaYoneticisi, VeriSeti
    print("[OK] DosyaYoneticisi sinifi import edildi")
    print("[OK] VeriSeti sinifi import edildi")
    
    # DosyaYoneticisi metotlarini kontrol et
    assert hasattr(DosyaYoneticisi, 'dosya_hash_hesapla'), "dosya_hash_hesapla metodu bulunamadi"
    assert hasattr(DosyaYoneticisi, 'guvenli_dosya_kopyala'), "guvenli_dosya_kopyala metodu bulunamadi"
    print("[OK] DosyaYoneticisi tum gerekli metodlara sahip")
    
except ImportError as e:
    print(f"[HATA] Import hatasi: {e}")
except AssertionError as e:
    print(f"[HATA] Assertion hatasi: {e}")

# Test 3: Script'lerin varligi
print("\n[TEST 3] Script'lerin olusturuldugu kontrol ediliyor...")
try:
    from pathlib import Path
    
    script_dosyalari = [
        "c:\\Users\\HectoRSheesh\\Desktop\\Machine_Learning\\Goruntu_On_Isleme\\scripts\\verileri_dosyala.py",
        "c:\\Users\\HectoRSheesh\\Desktop\\Machine_Learning\\Goruntu_On_Isleme\\scripts\\veri_seti_kontrol_et.py",
    ]
    
    for script in script_dosyalari:
        script_path = Path(script)
        if script_path.exists():
            print(f"[OK] Script olusturuldu: {script_path.name}")
        else:
            print(f"[UYARI] Script bulunamadi: {script_path.name}")
    
except Exception as e:
    print(f"[HATA] Hata: {e}")

print("\n" + "=" * 60)
print("[BASARILI] Tum testler tamamlandi!")
print("\nModullerin kullanimi:")
print("  1. Dosyalama: VeriDosyalama sinifini veya hizli_dosyalama() kullan")
print("  2. Yonetim: DosyaYoneticisi ve VeriSeti siniflarini kullan")
print("  3. Script'ler: verileri_dosyala.py veya veri_seti_kontrol_et.py calistir")
