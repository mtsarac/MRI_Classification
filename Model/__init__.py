"""
Model
------
MRI sınıflandırması için makine öğrenmesi modelleri.

Modüller:
  - gradient_boosting_model.py : XGBoost/LightGBM tabanlı Gradient Boosting modeli
  - linear_svm_model.py        : Lineer SVM sınıflandırıcısı
  - model_evaluator.py         : Model değerlendirme ve analiz araçları
  - train_and_evaluate_models.py : Ana eğitim ve değerlendirme script'i
  - config.py                  : Merkezi konfigürasyon yönetimi
  - visualizer.py              : Model sonuçlarının görselleştirilmesi
  - model_manager.py           : Model versiyon yönetimi ve persistence

Sınıflar:
  - GradientBoostingModel : Gradient Boosting tabanlı sınıflandırıcı
  - LinearSVMModel : Lineer SVM sınıflandırıcısı
  - ModelEvaluator : Tek model değerlendirmesi
  - ReportGenerator : Çok model karşılaştırması ve rapor oluşturma
  - ConfigManager : Konfigürasyon yönetimi
  - ModelVisualizer : Model sonuçlarının görselleştirilmesi
  - ModelManager : Model kaydetme ve yükleme

Kullanım:
  from Model.gradient_boosting_model import GradientBoostingModel
  from Model.linear_svm_model import LinearSVMModel
  from Model.model_evaluator import ModelEvaluator, ReportGenerator
  from Model.config import config
  from Model.visualizer import ModelVisualizer
  from Model.model_manager import ModelManager
"""

try:
    from .gradient_boosting_model import GradientBoostingModel
    from .linear_svm_model import LinearSVMModel
    from .model_evaluator import ModelEvaluator, ReportGenerator
    from .config import ConfigManager, config
    from .visualizer import ModelVisualizer
    from .model_manager import ModelManager, ModelVersion
    
    __all__ = [
        'GradientBoostingModel',
        'LinearSVMModel',
        'ModelEvaluator',
        'ReportGenerator',
        'ConfigManager',
        'config',
        'ModelVisualizer',
        'ModelManager',
        'ModelVersion',
    ]
except ImportError as e:
    print(f"[UYARI] Model modülleri yüklenemedi: {e}")
    __all__ = []


