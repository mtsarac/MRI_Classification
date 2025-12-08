"""
dosya_yoneticisi.py
-------------------
Dosya yönetimi ve veri seti işlemleri için yardımcı fonksiyonlar.
Güvenli dosya kopyalama, taşıma, silerek silme ve doğrulama işlemleri.
"""

import os
import shutil
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
import json


class DosyaYoneticisi:
    """Dosya yönetimi işlemleri için yardımcı sınıf."""
    
    @staticmethod
    def dosya_hash_hesapla(dosya_yolu: str, algoritma: str = "md5") -> str:
        """
        Dosyanın hash değerini hesapla (bütünlük kontrolü için).
        
        Parametreler:
        - dosya_yolu: Dosyanın yolu
        - algoritma: Hash algoritması (md5, sha1, sha256)
        
        Çıktı: Hash değeri (hex formatında)
        """
        hash_obj = hashlib.new(algoritma)
        dosya_yolu = Path(dosya_yolu)
        
        with open(dosya_yolu, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    @staticmethod
    def dosyalar_ayni_mi(dosya1: str, dosya2: str) -> bool:
        """
        İki dosyanın aynı olup olmadığını kontrol et.
        
        Parametreler:
        - dosya1: Birinci dosyanın yolu
        - dosya2: İkinci dosyanın yolu
        
        Çıktı: True if aynı, False otherwise
        """
        hash1 = DosyaYoneticisi.dosya_hash_hesapla(dosya1)
        hash2 = DosyaYoneticisi.dosya_hash_hesapla(dosya2)
        return hash1 == hash2
    
    @staticmethod
    def guvenli_dosya_kopyala(kaynak: str,
                              hedef: str,
                              ustune_yaz: bool = False,
                              hash_kontrol: bool = True) -> bool:
        """
        Dosyayı güvenli şekilde kopyala.
        
        Parametreler:
        - kaynak: Kaynak dosya yolu
        - hedef: Hedef dosya yolu
        - ustune_yaz: Varolan dosyayı üzerine yazma (varsayılan: False)
        - hash_kontrol: Kopyalamadan sonra hash kontrolü yap
        
        Çıktı: Başarı durumu (True/False)
        """
        kaynak = Path(kaynak)
        hedef = Path(hedef)
        
        # Kaynak dosyasının varlığını kontrol et
        if not kaynak.exists():
            print(f"[HATA] Kaynak dosya bulunamadı: {kaynak}")
            return False
        
        if not kaynak.is_file():
            print(f"[HATA] Kaynak bir dosya değil: {kaynak}")
            return False
        
        # Hedef klasörün varlığını kontrol et
        hedef.parent.mkdir(parents=True, exist_ok=True)
        
        # Hedef dosyasının varlığını kontrol et
        if hedef.exists() and not ustune_yaz:
            print(f"[UYARI] Hedef dosya zaten var (üzerine yazılmadı): {hedef}")
            return False
        
        try:
            # Dosyayı kopyala
            shutil.copy2(str(kaynak), str(hedef))
            
            # Hash kontrolü yap
            if hash_kontrol:
                if DosyaYoneticisi.dosyalar_ayni_mi(str(kaynak), str(hedef)):
                    print(f"[BAŞARILI] Dosya kopyalandı: {kaynak.name}")
                    return True
                else:
                    print(f"[HATA] Hash kontrolü başarısız: {hedef}")
                    return False
            else:
                print(f"[BAŞARILI] Dosya kopyalandı: {kaynak.name}")
                return True
                
        except Exception as e:
            print(f"[HATA] Dosya kopyalanamadı: {str(e)}")
            return False
    
    @staticmethod
    def guvenli_dosya_tasi(kaynak: str,
                           hedef: str,
                           ustune_yaz: bool = False) -> bool:
        """
        Dosyayı güvenli şekilde taşı.
        
        Parametreler:
        - kaynak: Kaynak dosya yolu
        - hedef: Hedef dosya yolu
        - ustune_yaz: Varolan dosyayı üzerine yazma
        
        Çıktı: Başarı durumu
        """
        kaynak = Path(kaynak)
        hedef = Path(hedef)
        
        if not kaynak.exists():
            print(f"[HATA] Kaynak dosya bulunamadı: {kaynak}")
            return False
        
        # Hedef klasörün varlığını kontrol et
        hedef.parent.mkdir(parents=True, exist_ok=True)
        
        if hedef.exists() and not ustune_yaz:
            print(f"[UYARI] Hedef dosya zaten var (taşınmadı): {hedef}")
            return False
        
        try:
            shutil.move(str(kaynak), str(hedef))
            print(f"[BAŞARILI] Dosya taşındı: {kaynak.name}")
            return True
        except Exception as e:
            print(f"[HATA] Dosya taşınamadı: {str(e)}")
            return False
    
    @staticmethod
    def klasor_simdiklik_olustur(klasor_yolu: str, silmeden_once_sorma: bool = True) -> bool:
        """
        Klasörü oluştur (önceden silme seçeneği ile).
        
        Parametreler:
        - klasor_yolu: Klasör yolu
        - silmeden_once_sorma: Varsa silmeden önce sor
        
        Çıktı: Başarı durumu
        """
        klasor_yolu = Path(klasor_yolu)
        
        if klasor_yolu.exists():
            if silmeden_once_sorma:
                cevap = input(f"[SORU] Klasör zaten var. Silinsin mi? ({klasor_yolu}) [e/H]: ").strip().lower()
                if cevap != "e":
                    print("[IPTAL] İşlem iptal edildi")
                    return False
            
            try:
                shutil.rmtree(klasor_yolu)
                print(f"[SILINDI] Eski klasör silindi: {klasor_yolu}")
            except Exception as e:
                print(f"[HATA] Klasör silinemedi: {str(e)}")
                return False
        
        try:
            klasor_yolu.mkdir(parents=True, exist_ok=True)
            print(f"[OLUSTURULDU] Klasör oluşturuldu: {klasor_yolu}")
            return True
        except Exception as e:
            print(f"[HATA] Klasör oluşturulamadı: {str(e)}")
            return False
    
    @staticmethod
    def klasor_boyutu_hesapla(klasor_yolu: str) -> int:
        """
        Klasörün toplam boyutunu byte cinsinden hesapla.
        
        Parametreler:
        - klasor_yolu: Klasör yolu
        
        Çıktı: Toplam boyut (byte)
        """
        klasor_yolu = Path(klasor_yolu)
        
        if not klasor_yolu.exists():
            return 0
        
        toplam_boyut = 0
        for dosya in klasor_yolu.rglob("*"):
            if dosya.is_file():
                toplam_boyut += dosya.stat().st_size
        
        return toplam_boyut
    
    @staticmethod
    def boyutu_insan_okunabilir_formata_cevir(byte_sayisi: int) -> str:
        """
        Byte cinsinden boyutu insan okunabilir formata çevir.
        
        Parametreler:
        - byte_sayisi: Byte cinsinden boyut
        
        Çıktı: Formatlı boyut (KiB, MiB, GiB, TiB)
        """
        for birim in ["B", "KiB", "MiB", "GiB", "TiB"]:
            if byte_sayisi < 1024.0:
                return f"{byte_sayisi:.2f} {birim}"
            byte_sayisi /= 1024.0
        
        return f"{byte_sayisi:.2f} PiB"
    
    @staticmethod
    def dosya_istatistikleri_al(klasor_yolu: str) -> Dict:
        """
        Klasördeki dosya istatistiklerini al.
        
        Parametreler:
        - klasor_yolu: Klasör yolu
        
        Çıktı: İstatistikler sözlüğü
        """
        klasor_yolu = Path(klasor_yolu)
        
        istatistikler = {
            "toplam_dosya": 0,
            "toplam_klasor": 0,
            "toplam_boyut": 0,
            "uzanti_dagilimi": {},
        }
        
        for item in klasor_yolu.rglob("*"):
            if item.is_file():
                istatistikler["toplam_dosya"] += 1
                istatistikler["toplam_boyut"] += item.stat().st_size
                
                # Uzantı dağılımı
                uzanti = item.suffix.lower()
                if uzanti:
                    istatistikler["uzanti_dagilimi"][uzanti] = \
                        istatistikler["uzanti_dagilimi"].get(uzanti, 0) + 1
            
            elif item.is_dir():
                istatistikler["toplam_klasor"] += 1
        
        return istatistikler
    
    @staticmethod
    def dosya_istatistiklerini_yazdir(istatistikler: Dict):
        """
        Dosya istatistiklerini güzel formatta yazdır.
        
        Parametreler:
        - istatistikler: İstatistikler sözlüğü
        """
        print("\n[VERİ SETİ İSTATİSTİKLERİ]")
        print("-" * 40)
        print(f"Toplam dosya: {istatistikler['toplam_dosya']}")
        print(f"Toplam klasör: {istatistikler['toplam_klasor']}")
        print(f"Toplam boyut: {DosyaYoneticisi.boyutu_insan_okunabilir_formata_cevir(istatistikler['toplam_boyut'])}")
        
        if istatistikler['uzanti_dagilimi']:
            print("\nUzantı dağılımı:")
            for uzanti, adet in sorted(istatistikler['uzanti_dagilimi'].items(),
                                      key=lambda x: x[1], reverse=True):
                print(f"  {uzanti}: {adet} dosya")


class VeriSeti:
    """Veri seti yönetimi için yardımcı sınıf."""
    
    def __init__(self, veri_seti_yolu: str):
        """
        Başlatma.
        
        Parametreler:
        - veri_seti_yolu: Veri setinin ana klasörü
        """
        self.veri_seti_yolu = Path(veri_seti_yolu)
        self.bolumler = {}
        self.istatistikler = {}
        self._yeniden_yuksle()
    
    def _yeniden_yuksle(self):
        """Veri seti bilgisini yeniden yükle."""
        # Bilinen bölümleri tara
        for bolum in ["eğitim", "doğrulama", "test", "tüm_veriler"]:
            bolum_yolu = self.veri_seti_yolu / bolum
            if bolum_yolu.exists():
                self.bolumler[bolum] = bolum_yolu
    
    def bolum_istatistikleri_al(self, bolum_adi: str) -> Dict:
        """
        Bir bölümün istatistiklerini al.
        
        Parametreler:
        - bolum_adi: Bölüm adı (eğitim, doğrulama, test, vb.)
        
        Çıktı: İstatistikler sözlüğü
        """
        if bolum_adi not in self.bolumler:
            return {}
        
        bolum_yolu = self.bolumler[bolum_adi]
        istatistikler = {}
        
        for sinif_klasoru in bolum_yolu.iterdir():
            if not sinif_klasoru.is_dir():
                continue
            
            dosya_sayisi = len(list(sinif_klasoru.glob("*")))
            istatistikler[sinif_klasoru.name] = dosya_sayisi
        
        return istatistikler
    
    def tum_istatistikleri_al(self) -> Dict:
        """
        Tüm bölümlerin istatistiklerini al.
        
        Çıktı: Tüm istatistikler
        """
        tum_istatistikler = {}
        
        for bolum in self.bolumler.keys():
            tum_istatistikler[bolum] = self.bolum_istatistikleri_al(bolum)
        
        return tum_istatistikler
    
    def istatistikleri_yazdir(self):
        """Veri seti istatistiklerini güzel formatta yazdır."""
        print("\n[VERİ SETİ YAPISI VE İSTATİSTİKLERİ]")
        print("=" * 60)
        
        for bolum, istatistik in self.tum_istatistikleri_al().items():
            toplam = sum(istatistik.values())
            print(f"\n{bolum.upper()}: {toplam} dosya")
            for sinif, adet in sorted(istatistik.items()):
                print(f"  {sinif}: {adet}")
    
    def dogrulama_raporu_olustur(self) -> Dict:
        """
        Veri seti doğrulaması yap ve rapor oluştur.
        
        Çıktı: Doğrulama raporu
        """
        rapor = {
            "bolumler": {},
            "sorunlar": [],
            "uyarilar": [],
        }
        
        for bolum, bolum_yolu in self.bolumler.items():
            rapor["bolumler"][bolum] = {}
            
            for sinif_klasoru in bolum_yolu.iterdir():
                if not sinif_klasoru.is_dir():
                    continue
                
                dosya_sayisi = len(list(sinif_klasoru.glob("*")))
                
                if dosya_sayisi == 0:
                    rapor["uyarilar"].append(
                        f"[{bolum}] {sinif_klasoru.name} klasöründe dosya yok"
                    )
                
                rapor["bolumler"][bolum][sinif_klasoru.name] = dosya_sayisi
        
        # Sınıflar arasında dengesizlik kontrol et
        for bolum, istatistik in rapor["bolumler"].items():
            adetler = list(istatistik.values())
            if len(adetler) > 1:
                min_adet = min(adetler)
                max_adet = max(adetler)
                oranı = max_adet / min_adet if min_adet > 0 else float('inf')
                
                if oranı > 1.5:
                    rapor["uyarilar"].append(
                        f"[{bolum}] Sınıflar arasında dengesizlik: "
                        f"min={min_adet}, max={max_adet}, oran={oranı:.2f}"
                    )
        
        return rapor
    
    def dogrulama_raporu_yazdir(self):
        """Dogrulama raporunu yazdir."""
        rapor = self.dogrulama_raporu_olustur()
        
        print("\n[VERİ SETİ DOĞRULAMA RAPORU]")
        print("=" * 60)
        
        # Sorunlar
        if rapor["sorunlar"]:
            print("\nSONUNLAR:")
            for sorun in rapor["sorunlar"]:
                print(f"  ✗ {sorun}")
        
        # Uyarılar
        if rapor["uyarilar"]:
            print("\nUYARILAR:")
            for uyari in rapor["uyarilar"]:
                print(f"  ⚠ {uyari}")
        
        if not rapor["sorunlar"] and not rapor["uyarilar"]:
            print("\n✓ Veri seti başarıyla doğrulandı!")
        
        # İstatistikler
        print("\nBÖLÜMLER VE SINIFLARI:")
        for bolum, istatistik in rapor["bolumler"].items():
            toplam = sum(istatistik.values())
            print(f"  {bolum}: {toplam} dosya")
            for sinif, adet in sorted(istatistik.items()):
                print(f"    - {sinif}: {adet}")
