#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
test_models.py
--------------
Model sınıflarını test etmek için basit test script'i.

Bu script, modellerin doğru şekilde çalışıp çalışmadığını kontrol eder.
Gerçek veri kullanmadan yapay veri üzerinde çalışır.

Çalıştırma:
  python test_models.py
"""

import numpy as np
import sys
from pathlib import Path

# Parent dizini sys.path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from Model.gradient_boosting_model import GradientBoostingModel
from Model.linear_svm_model import LinearSVMModel
from Model.model_evaluator import ModelEvaluator, ReportGenerator


def generate_synthetic_data(n_samples: int = 1000,
                           n_features: int = 50,
                           n_classes: int = 4,
                           random_state: int = 42) -> tuple:
    r"""
    Test için yapay veri oluştur.
    
    Parametreler:
    -----------
    n_samples : int
        Örnek sayısı
    n_features : int
        Öznitelik sayısı
    n_classes : int
        Sınıf sayısı
    random_state : int
        Rastgele tohum
    
    Döndürülen:
    ---------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    np.random.seed(random_state)
    
    print("[YAPAY VERİ OLUŞTURULUYOR]")
    
    # Yapay veri oluştur
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples).astype(np.int32)
    
    # Train/Test bölü (80/20)
    split_idx = int(0.8 * n_samples)
    indices = np.random.permutation(n_samples)
    
    X_train = X[indices[:split_idx]]
    y_train = y[indices[:split_idx]]
    X_test = X[indices[split_idx:]]
    y_test = y[indices[split_idx:]]
    
    print(f"  Eğitim: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    print(f"  Sınıflar: {np.unique(y)}")
    
    return X_train, X_test, y_train, y_test


def test_gradient_boosting(X_train, X_test, y_train, y_test):
    """Gradient Boosting modelini test et."""
    print("\n" + "="*60)
    print("[TEST] Gradient Boosting Model")
    print("="*60)
    
    try:
        # Model oluştur ve eğit
        model = GradientBoostingModel(
            algorithm='xgboost',
            n_estimators=10,  # Test için az sayıda
            max_depth=5,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Tahmin yap
        predictions = model.predict(X_test)
        print(f"✓ Tahminler başarılı: {predictions.shape}")
        
        # Değerlendir
        metrics = model.evaluate(X_test, y_test)
        print(f"✓ Doğruluk: {metrics['accuracy']:.4f}")
        
        return True
    except ImportError as e:
        print(f"✗ XGBoost kurulu değil: {e}")
        print("  LightGBM deniyor...")
        try:
            model = GradientBoostingModel(
                algorithm='lightgbm',
                n_estimators=10,
                max_depth=5,
                random_state=42
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            print(f"✓ Tahminler başarılı: {predictions.shape}")
            
            metrics = model.evaluate(X_test, y_test)
            print(f"✓ Doğruluk: {metrics['accuracy']:.4f}")
            return True
        except ImportError as e2:
            print(f"✗ LightGBM da kurulu değil: {e2}")
            return False
    except Exception as e:
        print(f"✗ Hata: {e}")
        return False


def test_linear_svm(X_train, X_test, y_train, y_test):
    """Lineer SVM modelini test et."""
    print("\n" + "="*60)
    print("[TEST] Lineer SVM Model")
    print("="*60)
    
    try:
        # Model oluştur ve eğit
        model = LinearSVMModel(
            C=1.0,
            max_iter=1000,
            random_state=42,
            verbose=0
        )
        
        model.fit(X_train, y_train, scale_features=True)
        
        # Tahmin yap
        predictions = model.predict(X_test)
        print(f"✓ Tahminler başarılı: {predictions.shape}")
        
        # Değerlendir
        metrics = model.evaluate(X_test, y_test)
        print(f"✓ Doğruluk: {metrics['accuracy']:.4f}")
        
        # Katsayıları al
        coef_df = model.get_coefficients()
        print(f"✓ Katsayılar alındı: {coef_df.shape[0]} öznitelik")
        
        return True
    except Exception as e:
        print(f"✗ Hata: {e}")
        return False


def test_model_evaluator(X_test, y_test):
    """Model Evaluator'ı test et."""
    print("\n" + "="*60)
    print("[TEST] Model Evaluator")
    print("="*60)
    
    try:
        # Yapay tahminler oluştur
        y_pred = np.random.randint(0, 4, len(y_test))
        
        # Evaluator oluştur ve değerlendir
        evaluator = ModelEvaluator("Test Model")
        metrics = evaluator.evaluate(y_test, y_pred)
        
        print(f"✓ Evaluasyon başarılı")
        print(f"✓ Metrikler: {len(metrics)} adet")
        
        return True
    except Exception as e:
        print(f"✗ Hata: {e}")
        return False


def test_report_generator(y_test):
    """Report Generator'ı test et."""
    print("\n" + "="*60)
    print("[TEST] Report Generator")
    print("="*60)
    
    try:
        # Yapay tahminler oluştur
        y_pred1 = np.random.randint(0, 4, len(y_test))
        y_pred2 = np.random.randint(0, 4, len(y_test))
        
        # Reporter oluştur
        reporter = ReportGenerator()
        reporter.add_model("Model 1", y_test, y_pred1)
        reporter.add_model("Model 2", y_test, y_pred2)
        
        # Karşılaştır
        comparison = reporter.compare_models()
        print(f"✓ Karşılaştırma başarılı: {comparison.shape}")
        print(f"\n{comparison.to_string()}")
        
        return True
    except Exception as e:
        print(f"✗ Hata: {e}")
        return False


def main():
    """Ana test fonksiyonu."""
    print("\n" + "="*60)
    print("MODEL TEST SERİSİ")
    print("="*60)
    
    # Yapay veri oluştur
    X_train, X_test, y_train, y_test = generate_synthetic_data()
    
    # Testleri çalıştır
    results = {
        'Gradient Boosting': test_gradient_boosting(X_train, X_test, y_train, y_test),
        'Lineer SVM': test_linear_svm(X_train, X_test, y_train, y_test),
        'Model Evaluator': test_model_evaluator(X_test, y_test),
        'Report Generator': test_report_generator(y_test),
    }
    
    # Sonuçları özetle
    print("\n" + "="*60)
    print("[TEST SONUÇLARI]")
    print("="*60)
    
    for test_name, success in results.items():
        status = "✓ GEÇTI" if success else "✗ BAŞARISIZ"
        print(f"  {test_name}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nToplam: {passed}/{total} test geçti")
    
    if passed == total:
        print("\n✓ Tüm testler başarılı!")
        return 0
    else:
        print(f"\n✗ {total - passed} test başarısız.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
