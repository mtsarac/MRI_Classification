#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
example_usage.py
----------------
Yeni özellikler için örnek kullanım.

Özellikleri:
  - Konfigürasyon yönetimi (config.py)
  - Model görselleştirmesi (visualizer.py)
  - Model persistence ve versiyon kontrolü (model_manager.py)

Çalıştırma:
    python example_usage.py
"""

import numpy as np
from pathlib import Path
import sys

# Parent dizini sys.path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from Model.config import config, ConfigManager
from Model.visualizer import ModelVisualizer
from Model.model_manager import ModelManager
from Model.gradient_boosting_model import GradientBoostingModel
from Model.linear_svm_model import LinearSVMModel


def example_config_management():
    """Konfigürasyon yönetimi örneği."""
    print("\n" + "="*60)
    print("[ÖRNEK 1] Konfigürasyon Yönetimi")
    print("="*60)
    
    # Varsayılan konfigürasyonu al
    print("\nVarsayılan GB Konfigürasyonu:")
    gb_config = config.get_gb_config()
    print(f"  Algorithm: {gb_config['algorithm']}")
    print(f"  N Estimators: {gb_config['n_estimators']}")
    print(f"  Max Depth: {gb_config['max_depth']}")
    
    # Konfigürasyon değerini al
    csv_path = config.get('data_paths.csv_file')
    print(f"\nCSV Yolu: {csv_path}")
    
    # Konfigürasyon dosyasını kaydet
    config.save_to_file('Model/example_config.json')
    print("✓ Konfigürasyon kaydedildi")


def example_visualization():
    """Görselleştirme örneği."""
    print("\n" + "="*60)
    print("[ÖRNEK 2] Model Görselleştirmesi")
    print("="*60)
    
    # Yapay veri oluştur
    np.random.seed(42)
    y_true = np.random.randint(0, 4, 100)
    y_pred = np.random.randint(0, 4, 100)
    
    # Visualizer'ı oluştur
    visualizer = ModelVisualizer(output_dir='Model/outputs/visualizations')
    
    # Karmaşıklık matrisi çiz
    print("\nKarmaşıklık matrisi çiziliyor...")
    visualizer.plot_confusion_matrix(
        y_true, y_pred,
        class_names=['Sınıf 0', 'Sınıf 1', 'Sınıf 2', 'Sınıf 3'],
        title="Örnek Karmaşıklık Matrisi",
        filename="example_confusion_matrix.png"
    )
    
    # Öznitelik önem sıralaması
    print("Öznitelik önem sıralaması çiziliyor...")
    feature_names = [f"Öznitelik_{i}" for i in range(20)]
    importance = np.random.rand(20)
    visualizer.plot_feature_importance(
        feature_names, importance,
        top_n=10,
        title="Örnek Öznitelik Önem Sıralaması",
        filename="example_feature_importance.png"
    )
    
    print("✓ Görseller oluşturuldu")


def example_model_management():
    """Model yönetimi örneği."""
    print("\n" + "="*60)
    print("[ÖRNEK 3] Model Persistence ve Versiyon Kontrolü")
    print("="*60)
    
    # Yapay veri oluştur
    np.random.seed(42)
    X_train = np.random.randn(100, 10).astype(np.float32)
    y_train = np.random.randint(0, 2, 100)
    
    # Model oluştur ve eğit
    print("\nModel oluşturuluyor ve eğitiliyor...")
    svm_model = LinearSVMModel(C=1.0, max_iter=500)
    svm_model.fit(X_train, y_train, scale_features=True)
    
    # Model Manager'ı oluştur
    manager = ModelManager(models_dir='Model/outputs/models')
    
    # Model 1'i kaydet (v001)
    print("\nModel v001 kaydediliyor...")
    manager.save_model(
        svm_model,
        model_name="example_svm",
        metrics={'accuracy': 0.95, 'f1_weighted': 0.94},
        config={'C': 1.0, 'loss': 'squared_hinge'},
        training_info={'samples': 100, 'features': 10}
    )
    
    # Model 2'yi kaydet (v002)
    print("Model v002 kaydediliyor...")
    svm_model.C = 0.5  # Parametreyi değiştir
    manager.save_model(
        svm_model,
        model_name="example_svm",
        metrics={'accuracy': 0.96, 'f1_weighted': 0.95},
        config={'C': 0.5, 'loss': 'squared_hinge'},
        training_info={'samples': 100, 'features': 10}
    )
    
    # Modelleri listele
    print("\nTüm modeller:")
    models_dict = manager.list_models()
    for model_name, versions in models_dict.items():
        print(f"  {model_name}:")
        for v in versions:
            print(f"    - Version {v['version']}: {v['timestamp']}")
    
    # Model geçmişini al
    print("\nModel geçmişi (history):")
    history = manager.get_model_history('example_svm')
    print(history.to_string())
    
    # En son modeli yükle
    print("\nEn son model yükleniyor...")
    loaded_model, metadata = manager.load_model('example_svm')
    print(f"  Model Type: {type(loaded_model).__name__}")
    print(f"  Metadata: {metadata}")
    
    print("\n✓ Model yönetimi örneği tamamlandı")


def main():
    """Ana fonksiyon."""
    print("\n" + "="*60)
    print("YENİ ÖZELLİKLER - ÖRNEK KULLANIM")
    print("="*60)
    
    # Örnekleri çalıştır
    example_config_management()
    example_visualization()
    example_model_management()
    
    print("\n" + "="*60)
    print("✓ TÜM ÖRNEKLER TAMAMLANDI")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
