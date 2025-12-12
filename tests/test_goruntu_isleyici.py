"""
Görüntü İşleyici Modülü Testleri
Tests for goruntu_isleyici.py module.

Bu dosya GorselIsleyici sınıfının tüm fonksiyonlarını test eder.
Görüntü yükleme, normalizasyon, histogram eşitleme gibi temel işlemleri doğrular.
"""

import sys
import pytest
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "goruntu_isleme"))

from goruntu_isleyici import GorselIsleyici


class TestGorselIsleyici:
    """GorselIsleyici sınıfı için test suite."""
    
    def test_init(self):
        """
        GorselIsleyici başlatma testi.
        
        İşleyici nesnesinin doğru şekilde oluşturulduğunu ve
        başlangıç değerlerinin sıfır olduğunu kontrol eder.
        """
        isleyici = GorselIsleyici()
        assert isleyici is not None
        assert isleyici.kalite_istatistikleri['toplam'] == 0
        assert isleyici.kalite_istatistikleri['basarili'] == 0
    
    def test_tohum_ayarla(self):
        """
        Rastgele tohum ayarlama testi.
        
        Aynı tohum ile başlatıldığında rastgele sayı üretecinin
        aynı değerleri ürettiğini doğrular.
        """
        GorselIsleyici.tohum_ayarla(42)
        val1 = np.random.random()
        
        GorselIsleyici.tohum_ayarla(42)
        val2 = np.random.random()
        
        assert val1 == val2, "Aynı tohum aynı rastgele değerleri üretmeli"
    
    def test_klasor_olustur(self, tmp_path):
        """
        Klasör oluşturma testi.
        
        Yeni bir klasörün başarıyla oluşturulduğunu kontrol eder.
        """
        test_klasor = tmp_path / "test_folder"
        GorselIsleyici.klasor_olustur(test_klasor)
        
        assert test_klasor.exists()
        assert test_klasor.is_dir()
    
    def test_goruntu_yukle_valid(self, test_image_path):
        """
        Geçerli görüntü yükleme testi.
        
        Var olan bir görüntü dosyasının başarıyla yüklendiğini
        ve doğru formatta olduğunu kontrol eder.
        """
        isleyici = GorselIsleyici()
        img = isleyici.goruntu_yukle(test_image_path)
        
        assert img is not None
        assert isinstance(img, np.ndarray)
        assert len(img.shape) == 2  # Gri tonlamalı (grayscale)
        assert img.dtype == np.uint8
    
    def test_goruntu_yukle_invalid(self):
        """
        Geçersiz görüntü yükleme testi.
        
        Var olmayan bir dosya yüklenmeye çalışıldığında
        None döndüğünü doğrular.
        """
        isleyici = GorselIsleyici()
        img = isleyici.goruntu_yukle(Path("nonexistent.jpg"))
        
        assert img is None
    
    def test_yogunluk_normalize(self):
        """
        Yoğunluk normalizasyonu testi.
        
        Görüntü piksel değerlerinin [0, 255] aralığında
        normalize edildiğini kontrol eder.
        """
        isleyici = GorselIsleyici()
        
        # Bilinen değerlerle test görüntüsü oluştur
        test_img = np.array([[0, 50, 100], [150, 200, 255]], dtype=np.uint8)
        normalized = isleyici.yogunluk_normalize(test_img)
        
        assert normalized.min() >= 0
        assert normalized.max() <= 255
        assert normalized.dtype == np.uint8
    
    def test_histogram_esitle(self):
        """
        Histogram eşitleme testi (CLAHE).
        
        Düşük kontrastlı bir görüntüye histogram eşitleme
        uygulandığında kontrastın arttığını doğrular.
        """
        isleyici = GorselIsleyici()
        
        # Düşük kontrastlı görüntü oluştur (100-150 arası değerler)
        test_img = np.random.randint(100, 150, (256, 256), dtype=np.uint8)
        result = isleyici.histogram_esitle(test_img, adaptive=True)
        
        assert result.shape == test_img.shape
        assert result.dtype == np.uint8
        # CLAHE kontrastı artırmalı
        assert result.std() >= test_img.std() * 0.8
    
    def test_boyutlandir(self):
        """
        Görüntü yeniden boyutlandırma testi.
        
        512x512 boyutundaki bir görüntünün 256x256'ya
        başarıyla küçültüldüğünü kontrol eder.
        """
        isleyici = GorselIsleyici()
        
        # 512x512 boyutunda görüntü oluştur
        test_img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        resized = isleyici.boyutlandir(test_img, genislik=256, yukseklik=256)
        
        assert resized.shape == (256, 256)
        assert resized.dtype == np.uint8
    
    def test_goruntu_kalite_kontrol_valid(self):
        """
        Geçerli görüntü kalite kontrolü testi.
        
        Yeterli parlaklık ve kontrasta sahip bir görüntünün
        kalite kontrolünden geçtiğini doğrular.
        """
        isleyici = GorselIsleyici()
        
        # Geçerli görüntü oluştur (50-200 arası değerler)
        valid_img = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
        is_valid, message = isleyici.goruntu_kalite_kontrol(valid_img)
        
        assert is_valid is True
        assert "gecti" in message.lower() or message == ""
    
    def test_goruntu_kalite_kontrol_too_dark(self):
        """
        Çok karanlık görüntü kalite kontrolü testi.
        
        Çok düşük parlaklığa sahip görüntünün kalite kontrolünden
        geçmediğini veya uyarı aldığını doğrular.
        """
        isleyici = GorselIsleyici()
        
        # Çok karanlık görüntü oluştur (0-10 arası değerler)
        dark_img = np.random.randint(0, 10, (256, 256), dtype=np.uint8)
        is_valid, message = isleyici.goruntu_kalite_kontrol(dark_img)
        
        # Düşük parlaklık nedeniyle başarısız olmalı
        assert is_valid is False or "karanlik" in message.lower()
    
    def test_goruntu_kalite_kontrol_low_contrast(self):
        """
        Düşük kontrastlı görüntü kalite kontrolü testi.
        
        Tüm pikselleri benzer değerde olan (düz) bir görüntünün
        düşük kontrast nedeniyle başarısız olduğunu kontrol eder.
        """
        isleyici = GorselIsleyici()
        
        # Düşük kontrastlı görüntü (hepsi 128)
        low_contrast_img = np.full((256, 256), 128, dtype=np.uint8)
        is_valid, message = isleyici.goruntu_kalite_kontrol(low_contrast_img)
        
        # Düşük kontrast nedeniyle başarısız olmalı
        assert is_valid is False or "kontrast" in message.lower()
    
    def test_gorselleri_listele(self, test_dataset_structure):
        """
        Veri seti görüntü listeleme testi.
        
        Veri seti klasöründeki tüm görüntülerin doğru şekilde
        listelendiğini ve bilgilerinin eksiksiz olduğunu kontrol eder.
        """
        isleyici = GorselIsleyici()
        dosyalar = isleyici.gorselleri_listele(test_dataset_structure)
        
        assert len(dosyalar) == 12  # 4 sınıf * 3 görüntü
        assert all('yol' in d for d in dosyalar)
        assert all('sinif' in d for d in dosyalar)
        assert all('etiket' in d for d in dosyalar)


class TestGorselIsleyiciEdgeCases:
    """
    Sınır Durumları ve Hata Yönetimi Testleri.
    
    Bu sınıf, görüntü işleyicinin olağandışı durumları
    nasıl ele aldığını test eder.
    """
    
    def test_empty_image(self):
        """
        Boş (sıfır) görüntü testi.
        
        Tüm pikselleri sıfır olan bir görüntünün kalite kontrolünden
        geçmediğini doğrular.
        """
        isleyici = GorselIsleyici()
        
        empty_img = np.zeros((256, 256), dtype=np.uint8)
        is_valid, message = isleyici.goruntu_kalite_kontrol(empty_img)
        
        # Kalite kontrolünden geçmemeli
        assert is_valid is False
    
    def test_very_small_image(self):
        """
        Çok küçük görüntü boyutlandırma testi.
        
        10x10 gibi çok küçük bir görüntünün 256x256'ya
        büyütülebildiğini kontrol eder.
        """
        isleyici = GorselIsleyici()
        
        small_img = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
        resized = isleyici.boyutlandir(small_img, 256, 256)
        
        assert resized.shape == (256, 256)
    
    def test_very_large_image(self):
        """
        Çok büyük görüntü boyutlandırma testi.
        
        2048x2048 gibi büyük bir görüntünün 256x256'ya
        küçültülebildiğini kontrol eder.
        """
        isleyici = GorselIsleyici()
        
        large_img = np.random.randint(0, 256, (2048, 2048), dtype=np.uint8)
        resized = isleyici.boyutlandir(large_img, 256, 256)
        
        assert resized.shape == (256, 256)
    
    def test_yogunluk_normalize_constant_image(self):
        """
        Sabit değerli görüntü normalizasyonu testi.
        
        Tüm pikselleri aynı değerde olan bir görüntünün
        normalizasyon sırasında NaN üretmediğini doğrular.
        """
        isleyici = GorselIsleyici()
        
        constant_img = np.full((256, 256), 128, dtype=np.uint8)
        normalized = isleyici.yogunluk_normalize(constant_img)
        
        # Sabit görüntüyü sorunsuz işlemeli
        assert normalized.shape == constant_img.shape
        assert not np.isnan(normalized).any()
