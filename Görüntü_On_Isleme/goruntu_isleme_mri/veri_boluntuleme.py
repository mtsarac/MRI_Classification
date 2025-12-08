"""
veri_boluntuleme.py
-------------------
Model eğitimi için test/eğitim/doğrulama setlerine veri bölüntüleme modülü.

Özellikler:
  - Stratified split (sınıf dengesini koruma)
  - Train/Validation/Test (70/15/15 varsayılan)
  - CSV + Görüntü yüklemesi
  - NumPy/DataFrame/PyTorch desteği
  - Rastgelelik kontrolü (Reproducibility)
  
Kullanım Örneği:
    from veri_boluntuleme import VeriboluntulemeManager
    
    manager = VeriboluntulemeManager(
        csv_dosya="goruntu_ozellikleri.csv",
        resim_klasoru="veri/cikti"
    )
    
    # Basit bölüntüleme
    train_loader, val_loader, test_loader = manager.tensorflow_veri_yukle()
    
    # Detaylı kontrol
    istatistikler = manager.boluntuleme_istatistikleri()
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from collections import Counter
import json
from datetime import datetime

# Opsiyonel importlar
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class VeriboluntulemeManager:
    """Veri bölüntüleme ve yönetim sınıfı."""
    
    def __init__(self,
                 csv_dosya: str,
                 resim_klasoru: str,
                 rastgele_tohum: int = 42,
                 egitim_orani: float = 0.70,
                 dogrulama_orani: float = 0.15,
                 test_orani: float = 0.15,
                 verbose: bool = True):
        """
        Başlatma.
        
        Parametreler:
        -----------
        csv_dosya : str
            Görüntü özelikleri CSV dosyasının yolu
        resim_klasoru : str
            Ön işlenmiş görüntülerin bulunduğu klasör
        rastgele_tohum : int
            Reproducibility için sabit tohum (varsayılan: 42)
        egitim_orani : float
            Eğitim seti oranı (0.0-1.0, varsayılan: 0.70)
        dogrulama_orani : float
            Doğrulama seti oranı (0.0-1.0, varsayılan: 0.15)
        test_orani : float
            Test seti oranı (0.0-1.0, varsayılan: 0.15)
        verbose : bool
            Detaylı çıktı göster (varsayılan: True)
        
        Raises:
        ------
        ValueError: Oranlar 1.0'a eşit değilse veya CSV dosyası yoksa
        FileNotFoundError: Dosya bulunamadığında
        """
        # Sabit tohum ayarla
        np.random.seed(rastgele_tohum)
        self.rastgele_tohum = rastgele_tohum
        
        # Oranları doğrula
        toplam_oran = egitim_orani + dogrulama_orani + test_orani
        if not np.isclose(toplam_oran, 1.0, atol=1e-6):
            raise ValueError(
                f"Oranların toplamı 1.0 olmalı (toplam: {toplam_oran})\n"
                f"  Eğitim: {egitim_orani}\n"
                f"  Doğrulama: {dogrulama_orani}\n"
                f"  Test: {test_orani}"
            )
        
        self.egitim_orani = egitim_orani
        self.dogrulama_orani = dogrulama_orani
        self.test_orani = test_orani
        
        # Dosya yolları
        if not os.path.exists(csv_dosya):
            raise FileNotFoundError(f"CSV dosyası bulunamadı: {csv_dosya}")
        if not os.path.exists(resim_klasoru):
            raise FileNotFoundError(f"Görüntü klasörü bulunamadı: {resim_klasoru}")
        
        self.csv_dosya = csv_dosya
        self.resim_klasoru = resim_klasoru
        self.verbose = verbose
        
        # Veri yükle
        self.veri = self._csv_yukle()
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.boluntuleme_yapilmis = False
        
        if self.verbose:
            print(f"\n[BAŞLATILDI] VeriboluntulemeManager")
            print(f"  CSV: {self.csv_dosya}")
            print(f"  Klasör: {self.resim_klasoru}")
            print(f"  Tohum: {self.rastgele_tohum}")
            print(f"  Oranlar: Eğitim={egitim_orani}, Doğrulama={dogrulama_orani}, Test={test_orani}")
    
    def _csv_yukle(self) -> pd.DataFrame:
        """
        CSV dosyasını yükle ve temel doğrulama yap.
        
        Döndürülen:
        ----------
        pd.DataFrame
            Yüklenen veri seti
        """
        try:
            veri = pd.read_csv(self.csv_dosya)
            
            # Gerekli sütunları kontrol et
            gerekli_sutunlar = ["dosya_adı", "etiket"]
            eksik_sutunlar = [s for s in gerekli_sutunlar if s not in veri.columns]
            if eksik_sutunlar:
                raise ValueError(f"CSV'de eksik sütunlar: {eksik_sutunlar}")
            
            if self.verbose:
                print(f"\n[BAŞARILI] CSV yüklendi: {len(veri)} satır")
                print(f"  Sütunlar: {veri.columns.tolist()}")
            
            return veri
        except Exception as e:
            print(f"[HATA] CSV yükleme hatası: {str(e)}")
            raise
    
    def boluntule(self, stratified: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Veriyi eğitim/doğrulama/test setlerine böl.
        
        Stratified split sınıf dağılımını korur.
        
        Parametreler:
        -----------
        stratified : bool
            Stratified split kullan (sınıf dengesini koru)
        
        Döndürülen:
        ----------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (train_data, val_data, test_data)
        
        Raises:
        ------
        ValueError: Etiket sütunu eksikse veya oran sorunları varsa
        """
        if "etiket" not in self.veri.columns:
            raise ValueError("CSV'de 'etiket' sütunu bulunamadı")
        
        if self.verbose:
            print(f"\n[BÖLÜNTÜLEME BAŞLANIYOR]")
            print(f"  Toplam örnek: {len(self.veri)}")
            self._sınif_dagilimi_goster("Orijinal Veri", self.veri)
        
        # 1. adım: Train+Val vs Test (85% vs 15%)
        test_orani_temp = self.test_orani
        train_val_orani = 1.0 - test_orani_temp
        
        if stratified:
            train_val_indices, test_indices = self._stratified_split(
                self.veri,
                train_val_orani
            )
            train_val_data = self.veri.iloc[train_val_indices].reset_index(drop=True)
            self.test_data = self.veri.iloc[test_indices].reset_index(drop=True)
        else:
            train_val_data, self.test_data = train_test_split(
                self.veri,
                test_size=test_orani_temp,
                random_state=self.rastgele_tohum,
                stratify=self.veri["etiket"] if stratified else None
            )
        
        # 2. adım: Train vs Val (70% vs 30% of train+val)
        # İçinde 70% train, 15% test var, artanı (15%) val olsun
        # train+val = 85%, bunun içinde: 70/85 = 82.35% train, 15/85 = 17.65% val
        
        val_orani_train_val = self.dogrulama_orani / (self.egitim_orani + self.dogrulama_orani)
        
        if stratified:
            train_indices, val_indices = self._stratified_split(
                train_val_data,
                1.0 - val_orani_train_val
            )
            self.train_data = train_val_data.iloc[train_indices].reset_index(drop=True)
            self.val_data = train_val_data.iloc[val_indices].reset_index(drop=True)
        else:
            self.train_data, self.val_data = train_test_split(
                train_val_data,
                test_size=val_orani_train_val,
                random_state=self.rastgele_tohum,
                stratify=train_val_data["etiket"] if stratified else None
            )
        
        self.boluntuleme_yapilmis = True
        
        if self.verbose:
            print(f"\n[BÖLÜNTÜLEME TAMAMLANDI]")
            self._sınif_dagilimi_goster("Eğitim Seti", self.train_data)
            self._sınif_dagilimi_goster("Doğrulama Seti", self.val_data)
            self._sınif_dagilimi_goster("Test Seti", self.test_data)
        
        return self.train_data, self.val_data, self.test_data
    
    def _stratified_split(self, 
                         data: pd.DataFrame, 
                         train_oran: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stratified split uygula.
        
        Parametreler:
        -----------
        data : pd.DataFrame
            Veri seti
        train_oran : float
            Eğitim seti oranı
        
        Döndürülen:
        ----------
        Tuple[np.ndarray, np.ndarray]
            (train_indices, test_indices)
        """
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=1.0 - train_oran,
            random_state=self.rastgele_tohum
        )
        
        train_idx, test_idx = next(splitter.split(data, data["etiket"]))
        return train_idx, test_idx
    
    def _sınif_dagilimi_goster(self, baslik: str, data: pd.DataFrame):
        """Sınıf dağılımını göster."""
        if not self.verbose:
            return
        
        dagilim = data["etiket"].value_counts().sort_index()
        toplam = len(data)
        
        print(f"\n  {baslik} (toplam: {toplam}):")
        for etiket, sayı in dagilim.items():
            yuzde = (sayı / toplam) * 100
            print(f"    Sınıf {etiket}: {sayı:4d} ({yuzde:5.1f}%)")
    
    def goruntu_yukle(self, 
                     dosya_adi: str, 
                     boyut: Optional[Tuple[int, int]] = None,
                     normalize: bool = True) -> np.ndarray:
        """
        Tek bir görüntüyü yükle.
        
        Parametreler:
        -----------
        dosya_adi : str
            Görüntü dosyasının adı (CSV'den)
        boyut : Optional[Tuple[int, int]]
            Yeniden boyutlandırma (genişlik, yükseklik)
        normalize : bool
            Görüntüyü 0-1 aralığına normalize et
        
        Döndürülen:
        ----------
        np.ndarray
            Görüntü array'i
        """
        # Görüntü yolunu bul
        goruntu_yolu = self._goruntu_yolunu_bul(dosya_adi)
        
        if not goruntu_yolu:
            raise FileNotFoundError(f"Görüntü bulunamadı: {dosya_adi}")
        
        # Görüntüyü yükle
        if CV2_AVAILABLE:
            goruntu = cv2.imread(goruntu_yolu, cv2.IMREAD_GRAYSCALE)
        elif PIL_AVAILABLE:
            goruntu = np.array(Image.open(goruntu_yolu).convert('L'))
        else:
            raise ImportError("PIL veya OpenCV gerekli")
        
        if goruntu is None:
            raise ValueError(f"Görüntü yüklenemedi: {goruntu_yolu}")
        
        # Boyutlandır
        if boyut is not None:
            if CV2_AVAILABLE:
                goruntu = cv2.resize(goruntu, boyut, interpolation=cv2.INTER_LINEAR)
            elif PIL_AVAILABLE:
                goruntu = np.array(Image.fromarray(goruntu).resize(boyut))
        
        # Normalize et
        if normalize:
            goruntu = goruntu.astype(np.float32) / 255.0
        
        return goruntu
    
    def _goruntu_yolunu_bul(self, dosya_adi: str) -> Optional[str]:
        """
        Görüntü dosyasının tam yolunu bul.
        
        Klasör yapısı:
        - veri/cikti/NonDemented/...
        - veri/cikti/VeryMildDemented/...
        vb.
        """
        siniflar = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
        
        for sinif in siniflar:
            yol = os.path.join(self.resim_klasoru, sinif, dosya_adi)
            if os.path.exists(yol):
                return yol
        
        return None
    
    def veri_seti_olustur(self,
                         boyut: Tuple[int, int] = (256, 256),
                         normalize: bool = True) -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                                         Tuple[np.ndarray, np.ndarray],
                                                         Tuple[np.ndarray, np.ndarray]]:
        """
        Görüntü ve etiketlerle tam veri seti oluştur.
        
        Parametreler:
        -----------
        boyut : Tuple[int, int]
            Görüntü boyutu (varsayılan: 256x256)
        normalize : bool
            Görüntüleri 0-1 aralığına normalize et
        
        Döndürülen:
        ----------
        Tuple[Tuple, Tuple, Tuple]
            ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        
        Not: Bu işlem ağır olabilir. İlk çalıştırmada biraz zaman alabilir.
        """
        if not self.boluntuleme_yapilmis:
            self.boluntule()
        
        print(f"\n[VERI SETİ OLUŞTURULUYOR]")
        print(f"  Boyut: {boyut}")
        print(f"  Normalize: {normalize}")
        
        # Eğitim seti
        print(f"\n  Eğitim seti yükleniyor... ({len(self.train_data)} görüntü)")
        X_train, y_train = self._veri_yukle_ve_hazirla(self.train_data, boyut, normalize)
        
        # Doğrulama seti
        print(f"  Doğrulama seti yükleniyor... ({len(self.val_data)} görüntü)")
        X_val, y_val = self._veri_yukle_ve_hazirla(self.val_data, boyut, normalize)
        
        # Test seti
        print(f"  Test seti yükleniyor... ({len(self.test_data)} görüntü)")
        X_test, y_test = self._veri_yukle_ve_hazirla(self.test_data, boyut, normalize)
        
        print(f"\n[BAŞARILI] Tüm veri setleri yüklendi")
        print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def _veri_yukle_ve_hazirla(self,
                              veri: pd.DataFrame,
                              boyut: Tuple[int, int],
                              normalize: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Veri setini yükle ve hazırla.
        
        Döndürülen:
        ----------
        Tuple[np.ndarray, np.ndarray]
            (X, y) - görüntüler ve etiketler
        """
        X = []
        y = []
        basarisiz = 0
        
        for idx, satir in veri.iterrows():
            try:
                goruntu = self.goruntu_yukle(
                    satir["dosya_adı"],
                    boyut=boyut,
                    normalize=normalize
                )
                X.append(goruntu)
                y.append(satir["etiket"])
            except Exception as e:
                basarisiz += 1
                if self.verbose and basarisiz <= 3:
                    print(f"    [UYARI] Görüntü yükleme hatası: {satir['dosya_adı']} - {str(e)}")
        
        if basarisiz > 0:
            print(f"    [UYARI] {basarisiz} görüntü yüklenemedi")
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        
        # Kanal boyutu ekle (grayscale için)
        if len(X.shape) == 3:
            X = X[:, :, :, np.newaxis]
        
        return X, y
    
    def tensorflow_veri_yukle(self,
                             boyut: Tuple[int, int] = (256, 256),
                             batch_size: int = 32) -> Tuple:
        """
        TensorFlow/Keras için veri yükle (tf.data.Dataset).
        
        Parametreler:
        -----------
        boyut : Tuple[int, int]
            Görüntü boyutu
        batch_size : int
            Batch boyutu
        
        Döndürülen:
        ----------
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]
            (train_dataset, val_dataset, test_dataset)
        
        Not: TensorFlow kurulu olmalı
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow gerekli. 'pip install tensorflow' ile kurun")
        
        if not self.boluntuleme_yapilmis:
            self.boluntule()
        
        print(f"\n[TENSORFLOW VERİ YÜKLEME]")
        
        # Veri setlerini oluştur
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.veri_seti_olustur(
            boyut=boyut,
            normalize=True
        )
        
        # tf.data.Dataset'e dönüştür
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        
        # Batch ve shuffle
        train_dataset = train_dataset.shuffle(len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        print(f"  Train: {train_dataset}")
        print(f"  Validation: {val_dataset}")
        print(f"  Test: {test_dataset}")
        
        return train_dataset, val_dataset, test_dataset
    
    def pytorch_veri_yukle(self,
                          boyut: Tuple[int, int] = (256, 256),
                          batch_size: int = 32,
                          num_workers: int = 0) -> Tuple:
        """
        PyTorch için veri yükle (DataLoader).
        
        Parametreler:
        -----------
        boyut : Tuple[int, int]
            Görüntü boyutu
        batch_size : int
            Batch boyutu
        num_workers : int
            Paralel işçi sayısı
        
        Döndürülen:
        ----------
        Tuple[DataLoader, DataLoader, DataLoader]
            (train_loader, val_loader, test_loader)
        
        Not: PyTorch ve torchvision kurulu olmalı
        """
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("PyTorch gerekli. 'pip install torch torchvision' ile kurun")
        
        if not self.boluntuleme_yapilmis:
            self.boluntule()
        
        print(f"\n[PYTORCH VERİ YÜKLEME]")
        
        # Veri setlerini oluştur
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.veri_seti_olustur(
            boyut=boyut,
            normalize=True
        )
        
        # PyTorch tensörlere dönüştür
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).long()
        
        X_val = torch.from_numpy(X_val).float()
        y_val = torch.from_numpy(y_val).long()
        
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).long()
        
        # TensorDataset oluştur
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        # DataLoader oluştur
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        print(f"  Train: {len(train_loader)} batches")
        print(f"  Validation: {len(val_loader)} batches")
        print(f"  Test: {len(test_loader)} batches")
        
        return train_loader, val_loader, test_loader
    
    def boluntuleme_istatistikleri(self) -> Dict:
        """
        Bölüntüleme istatistiklerini hesapla ve döndür.
        
        Döndürülen:
        ----------
        Dict
            İstatistikler sözlüğü
        """
        if not self.boluntuleme_yapilmis:
            self.boluntule()
        
        istatistikler = {
            "timestamp": datetime.now().isoformat(),
            "toplam_ornekler": len(self.veri),
            "rastgele_tohum": self.rastgele_tohum,
            "stratified": True,
            "oranlar": {
                "egitim": self.egitim_orani,
                "dogrulama": self.dogrulama_orani,
                "test": self.test_orani
            },
            "boyutlar": {
                "egitim": len(self.train_data),
                "dogrulama": len(self.val_data),
                "test": len(self.test_data)
            },
            "sinif_dagilimi": {
                "orijinal": self.veri["etiket"].value_counts().to_dict(),
                "egitim": self.train_data["etiket"].value_counts().to_dict(),
                "dogrulama": self.val_data["etiket"].value_counts().to_dict(),
                "test": self.test_data["etiket"].value_counts().to_dict()
            }
        }
        
        return istatistikler
    
    def istatistikleri_kaydet(self, dosya_adi: str = "boluntuleme_istatistikleri.json"):
        """
        Bölüntüleme istatistiklerini JSON dosyasına kaydet.
        
        Parametreler:
        -----------
        dosya_adi : str
            Kaydedilecek dosyanın adı
        """
        istatistikler = self.boluntuleme_istatistikleri()
        
        with open(dosya_adi, 'w', encoding='utf-8') as f:
            json.dump(istatistikler, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"\n[BAŞARILI] İstatistikler kaydedildi: {dosya_adi}")


if __name__ == "__main__":
    """
    Kullanım örneği ve test.
    """
    import sys
    
    # Proje ana dizinine git
    proje_adi = "Görüntü_On_Isleme"
    ana_dizin = Path(__file__).parent.parent
    
    csv_dosya = os.path.join(ana_dizin, proje_adi, "veri", "cikti", "goruntu_ozellikleri.csv")
    resim_klasoru = os.path.join(ana_dizin, proje_adi, "veri", "cikti")
    
    print("="*70)
    print("VERİ BÖLÜNTÜLEME ÖRNEK KULLANIMI")
    print("="*70)
    
    # Manager oluştur
    manager = VeriboluntulemeManager(
        csv_dosya=csv_dosya,
        resim_klasoru=resim_klasoru,
        rastgele_tohum=42,
        egitim_orani=0.70,
        dogrulama_orani=0.15,
        test_orani=0.15,
        verbose=True
    )
    
    # Veriyi böl
    train_data, val_data, test_data = manager.boluntule(stratified=True)
    
    # İstatistikleri göster
    print("\n" + "="*70)
    print("BÖLÜNTÜLEME İSTATİSTİKLERİ")
    print("="*70)
    istatistikler = manager.boluntuleme_istatistikleri()
    print(json.dumps(istatistikler, indent=2, ensure_ascii=False))
    
    # İstatistikleri kaydet
    manager.istatistikleri_kaydet("boluntuleme_istatistikleri.json")
    
    print("\n" + "="*70)
    print("ÖRNEK VERİLER")
    print("="*70)
    print(f"\nTrain seti ilk 5 satır:")
    print(train_data.head())
    print(f"\nVal seti ilk 5 satır:")
    print(val_data.head())
    print(f"\nTest seti ilk 5 satır:")
    print(test_data.head())
