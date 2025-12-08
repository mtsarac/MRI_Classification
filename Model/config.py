#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
config.py
---------
Model eğitimi için merkezi konfigürasyon dosyası.

Tüm hiperparametreler bu dosyada tanımlanır ve yönetilir.
"""

import json
from typing import Dict, Any, Tuple
from pathlib import Path

# Proje kök dizini
PROJECT_ROOT = Path(__file__).parent.parent

# ==================== GRADIENT BOOSTING AYARLARI ====================
GRADIENT_BOOSTING_CONFIG = {
    'algorithm': 'xgboost',  # 'xgboost' veya 'lightgbm'
    'n_estimators': 100,
    'max_depth': 7,
    'learning_rate': 0.1,
    'random_state': 42,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1.0,
    'reg_alpha': 0.0,
    'early_stopping_rounds': 10,
}

# ==================== LINEAR SVM AYARLARI ====================
LINEAR_SVM_CONFIG = {
    'C': 1.0,
    'loss': 'squared_hinge',
    'max_iter': 2000,
    'random_state': 42,
    'dual': True,
    'tol': 1e-4,
    'class_weight': 'balanced',
    'verbose': 0,
}

# ==================== VERİ BÖLÜMLEME AYARLARI ====================
DATA_SPLIT_CONFIG = {
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'random_state': 42,
    'stratify': True,  # Sınıf dengesi korunacak mı?
}

# ==================== HİPERPARAMETRE AYARLAMA (GRID SEARCH) ====================
HYPERPARAMETER_TUNING_CONFIG = {
    'cv_folds': 5,
    'n_jobs': -1,  # -1 = tüm CPU çekirdekleri
    'verbose': 1,
}

# Grid Search parametreleri
GB_GRID_SEARCH_PARAMS = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 7, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
}

SVM_GRID_SEARCH_PARAMS = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'loss': ['hinge', 'squared_hinge'],
    'max_iter': [1000, 2000, 5000],
}

# ==================== DOSYA YOLLARI ====================
# Dosya yolları proje kök dizinine göre belirlenir
_POSSIBLE_CSV_NAMES = [
    'goruntu_ozellikleri_scaled.csv',
    'goruntu_ozellikleri.csv',
]

# ASCII yollarını tercih et, Türkçe yolları fallback olarak kullan
_POSSIBLE_CSV_DIRS = [
    PROJECT_ROOT / 'Goruntu_On_Isleme' / 'cikti',      # ASCII - tercih edilen
    PROJECT_ROOT / 'Goruntu_On_Isleme' / 'outputs',    # ASCII alternatif
    PROJECT_ROOT / 'Görüntü_On_Isleme' / 'cikti',      # Türkçe - fallback
    PROJECT_ROOT / 'Görüntü_On_Isleme' / 'çıktı',      # Türkçe - fallback
    PROJECT_ROOT / 'Görüntü_On_Isleme' / 'outputs',    # Türkçe - fallback
]

# CSV dosyasını otomatik bulma
_CSV_FILE = None
for csv_dir in _POSSIBLE_CSV_DIRS:
    if not csv_dir.exists():
        continue
    for csv_name in _POSSIBLE_CSV_NAMES:
        potential_csv = csv_dir / csv_name
        if potential_csv.exists():
            _CSV_FILE = str(potential_csv)
            break
    if _CSV_FILE:
        break

# Eğer CSV bulunamazsa varsayılan ASCII yolu belirle
if not _CSV_FILE:
    _CSV_FILE = str(PROJECT_ROOT / 'Goruntu_On_Isleme' / 'cikti' / 'goruntu_ozellikleri_scaled.csv')

DATA_CONFIG = {
    'csv_file': _CSV_FILE,
    'veri_seti_klasoru': str(PROJECT_ROOT / 'Veri_Seti'),
    'output_dir': str(PROJECT_ROOT / 'Model' / 'outputs'),
    'models_dir': str(PROJECT_ROOT / 'Model' / 'outputs' / 'models'),
    'reports_dir': str(PROJECT_ROOT / 'Model' / 'outputs' / 'reports'),
    'visualizations_dir': str(PROJECT_ROOT / 'Model' / 'outputs' / 'visualizations'),
}

# ==================== VİSUALİZASYON AYARLARI ====================
VISUALIZATION_CONFIG = {
    'confusion_matrix_figsize': (10, 8),
    'feature_importance_figsize': (12, 6),
    'roc_curve_figsize': (8, 6),
    'dpi': 100,
    'style': 'seaborn-v0_8-darkgrid',  # matplotlib style
    'color_palette': 'Set2',  # seaborn color palette
    'font_size': 12,
}

# ==================== MODEL VERSİYON AYARLARI ====================
MODEL_VERSION_CONFIG = {
    'save_format': 'json',  # 'json' veya 'pickle'
    'include_metadata': True,
    'include_metrics': True,
    'backup_old_versions': True,
    'max_versions_to_keep': 5,
}

# ==================== LOGGING AYARLARI ====================
LOGGING_CONFIG = {
    'log_file': str(PROJECT_ROOT / 'Model' / 'outputs' / 'training.log'),
    'log_level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
}

# Çıktı klasörlerini otomatik oluştur
def _create_output_directories():
    """Gerekli çıktı klasörlerini oluştur."""
    for dir_key in ['output_dir', 'models_dir', 'reports_dir', 'visualizations_dir']:
        dir_path = Path(DATA_CONFIG[dir_key])
        dir_path.mkdir(parents=True, exist_ok=True)


def validate_csv_file(csv_path: str = None) -> Tuple[bool, str]:
    """
    CSV dosyasının varlığını ve geçerliliğini kontrol et.
    
    Args:
        csv_path: Kontrol edilecek CSV dosyasının yolu (varsayılan: DATA_CONFIG['csv_file'])
    
    Returns:
        Tuple[bool, str]: (Geçerli mi, İleti)
    """
    csv_file = csv_path or DATA_CONFIG['csv_file']
    csv_path_obj = Path(csv_file)
    
    if not csv_path_obj.exists():
        return False, f"[HATA] CSV dosyası bulunamadı: {csv_file}\n\nEnsure veri ön işleme adımlarını çalıştırın."
    
    if not csv_path_obj.is_file():
        return False, f"[HATA] Yol bir dosya değil: {csv_file}"
    
    if csv_path_obj.stat().st_size == 0:
        return False, f"[HATA] CSV dosyası boş: {csv_file}"
    
    return True, f"[OK] CSV dosyası geçerli: {csv_file}"


class ConfigManager:
    """Konfigürasyon yönetimi için sınıf."""
    
    def __init__(self, config_file: str = None):
        """
        Konfigürasyon yöneticisini başlat.
        
        Args:
            config_file: Konfigürasyon JSON dosyasının yolu (opsiyonel)
        """
        self.config = self._get_default_config()
        
        if config_file and Path(config_file).exists():
            self.load_from_file(config_file)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Varsayılan konfigürasyonları al."""
        return {
            'gradient_boosting': GRADIENT_BOOSTING_CONFIG.copy(),
            'linear_svm': LINEAR_SVM_CONFIG.copy(),
            'data_split': DATA_SPLIT_CONFIG.copy(),
            'hyperparameter_tuning': HYPERPARAMETER_TUNING_CONFIG.copy(),
            'data_paths': DATA_CONFIG.copy(),
            'visualization': VISUALIZATION_CONFIG.copy(),
            'model_version': MODEL_VERSION_CONFIG.copy(),
            'logging': LOGGING_CONFIG.copy(),
        }
    
    def load_from_file(self, filepath: str):
        """JSON dosyasından konfigürasyon yükle."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            self.config.update(file_config)
            print(f"[BILGI] Konfigürasyon yüklendi: {filepath}")
        except Exception as e:
            print(f"[HATA] Konfigürasyon yükleme hatası: {e}")
    
    def save_to_file(self, filepath: str):
        """Konfigürasyonu JSON dosyasına kaydet."""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"[BILGI] Konfigürasyon kaydedildi: {filepath}")
        except Exception as e:
            print(f"[HATA] Konfigürasyon kaydetme hatası: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Konfigürasyon değerini al."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Konfigürasyon değerini ayarla."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_gb_config(self) -> Dict[str, Any]:
        """Gradient Boosting konfigürasyonunu al."""
        return self.config['gradient_boosting'].copy()
    
    def get_svm_config(self) -> Dict[str, Any]:
        """Linear SVM konfigürasyonunu al."""
        return self.config['linear_svm'].copy()
    
    def get_data_split_config(self) -> Dict[str, Any]:
        """Veri bölümleme konfigürasyonunu al."""
        return self.config['data_split'].copy()
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Görselleştirme konfigürasyonunu al."""
        return self.config['visualization'].copy()
    
    def print_config(self):
        """Konfigürasyonları yazdır."""
        print("\n" + "="*60)
        print("[KONFIGÜRASYON]")
        print("="*60)
        print(json.dumps(self.config, indent=2, ensure_ascii=False))
    
    def __repr__(self) -> str:
        """String gösterimi."""
        return f"ConfigManager(config_keys={list(self.config.keys())})"


# Gerekli çıktı klasörlerini otomatik oluştur
_create_output_directories()

# Global konfigürasyon yöneticisi
config = ConfigManager()

