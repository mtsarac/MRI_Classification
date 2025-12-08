#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
model_manager.py
----------------
Modellerin kaydedilmesi, yüklenmesi ve versiyon kontrolü.

Özellikler:
  - Model kaydı (pickle ve JSON formatları)
  - Model yükleme
  - Versiyon yönetimi
  - Model metadata (eğitim parametreleri, metrikler)
  - Yedekleme ve versiyon geçmişi
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import shutil
import numpy as np
import pandas as pd

# Pickle güvenliği için
pickle.DEFAULT_PROTOCOL = 4


class ModelVersion:
    """Model versiyonu bilgilerini saklayan sınıf."""
    
    def __init__(self,
                 model_name: str,
                 version: int,
                 timestamp: str = None):
        r"""
        Model versiyonunu başlat.
        
        Parametreler:
        -----------
        model_name : str
            Model adı
        version : int
            Versiyon numarası
        timestamp : str
            Oluşturma tarihi/saati
        """
        self.model_name = model_name
        self.version = version
        self.timestamp = timestamp or datetime.now().isoformat()
        self.metrics = {}
        self.config = {}
        self.training_info = {}
    
    def to_dict(self) -> Dict:
        """Sözlüğe dönüştür."""
        return {
            'model_name': self.model_name,
            'version': self.version,
            'timestamp': self.timestamp,
            'metrics': self.metrics,
            'config': self.config,
            'training_info': self.training_info,
        }


class ModelManager:
    """Model yönetimi için sınıf."""
    
    def __init__(self,
                 models_dir: str = 'Model/outputs/models',
                 save_format: str = 'pickle',
                 max_versions: int = 5):
        r"""
        Model yöneticisini başlat.
        
        Parametreler:
        -----------
        models_dir : str
            Modellerin kaydedileceği dizin
        save_format : str
            Kaydetme formatı ('pickle' veya 'json')
        max_versions : int
            Tutulacak maksimum versiyon sayısı
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.save_format = save_format
        self.max_versions = max_versions
    
    def save_model(self,
                   model: Any,
                   model_name: str,
                   metrics: Dict = None,
                   config: Dict = None,
                   training_info: Dict = None) -> Path:
        r"""
        Modeli kaydet.
        
        Parametreler:
        -----------
        model : Any
            Kaydedilecek model objesi
        model_name : str
            Model adı
        metrics : dict, optional
            Model metrikleri
        config : dict, optional
            Model konfigürasyonu
        training_info : dict, optional
            Eğitim bilgileri
        
        Döndürülen:
        ---------
        Path
            Kaydedilen dosya yolu
        """
        # Modelin klasörünü oluştur
        model_dir = self.models_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Versiyon numarasını belirle
        version = self._get_next_version(model_dir)
        
        # Metadata oluştur
        version_info = ModelVersion(model_name, version)
        version_info.metrics = metrics or {}
        version_info.config = config or {}
        version_info.training_info = training_info or {}
        
        # Model dosyası adı
        if self.save_format == 'pickle':
            model_file = model_dir / f"v{version:03d}_model.pkl"
        else:
            model_file = model_dir / f"v{version:03d}_model.json"
        
        # Metadata dosyası
        metadata_file = model_dir / f"v{version:03d}_metadata.json"
        
        # Modeli kaydet
        try:
            if self.save_format == 'pickle':
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
            else:
                # JSON formatında kaydet (model summary)
                self._save_model_as_json(model, model_file, model_name)
            
            print(f"[KAYDEDILDI] Model ({model_name} v{version}): {model_file}")
        except Exception as e:
            print(f"[HATA] Model kaydetme hatası: {e}")
            raise
        
        # Metadata'yı kaydet
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(version_info.to_dict(), f, indent=2, 
                         ensure_ascii=False, default=str)
            print(f"[KAYDEDILDI] Metadata: {metadata_file}")
        except Exception as e:
            print(f"[UYARI] Metadata kaydetme hatası: {e}")
        
        # Eski versiyonları temizle
        self._cleanup_old_versions(model_dir, model_name)
        
        return model_file
    
    def load_model(self,
                   model_name: str,
                   version: int = None) -> Tuple[Any, Dict]:
        r"""
        Modeli yükle.
        
        Parametreler:
        -----------
        model_name : str
            Model adı
        version : int, optional
            Versiyon numarası (None = son versiyon)
        
        Döndürülen:
        ---------
        Tuple[Any, Dict]
            (Model objesi, Metadata)
        """
        model_dir = self.models_dir / model_name
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model dizini bulunamadı: {model_dir}")
        
        # Versiyon numarasını belirle
        if version is None:
            version = self._get_latest_version(model_dir)
        
        # Model dosyasını bul
        if self.save_format == 'pickle':
            model_file = model_dir / f"v{version:03d}_model.pkl"
        else:
            model_file = model_dir / f"v{version:03d}_model.json"
        
        metadata_file = model_dir / f"v{version:03d}_metadata.json"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_file}")
        
        # Modeli yükle
        try:
            if self.save_format == 'pickle':
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
            else:
                # JSON'dan yükle (summary olarak)
                with open(model_file, 'r', encoding='utf-8') as f:
                    model = json.load(f)
            
            print(f"[YÜKLENDI] Model ({model_name} v{version}): {model_file}")
        except Exception as e:
            print(f"[HATA] Model yükleme hatası: {e}")
            raise
        
        # Metadata'yı yükle
        metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                print(f"[YÜKLENDI] Metadata: {metadata_file}")
            except Exception as e:
                print(f"[UYARI] Metadata yükleme hatası: {e}")
        
        return model, metadata
    
    def list_models(self) -> Dict[str, list]:
        """Tüm modelleri ve versiyonlarını listele."""
        models_dict = {}
        
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                versions = []
                
                for metadata_file in sorted(model_dir.glob('v*_metadata.json')):
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            info = json.load(f)
                        versions.append({
                            'version': info['version'],
                            'timestamp': info['timestamp'],
                            'metrics': info.get('metrics', {}),
                        })
                    except:
                        pass
                
                if versions:
                    models_dict[model_name] = versions
        
        return models_dict
    
    def get_model_history(self, model_name: str) -> pd.DataFrame:
        """Model versiyon geçmişini DataFrame olarak al."""
        model_dir = self.models_dir / model_name
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model bulunamadı: {model_name}")
        
        history_data = []
        
        for metadata_file in sorted(model_dir.glob('v*_metadata.json')):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                
                row = {
                    'Version': info['version'],
                    'Timestamp': info['timestamp'],
                }
                
                # Metrikleri ekle
                for metric_name, metric_value in info.get('metrics', {}).items():
                    row[metric_name] = metric_value
                
                history_data.append(row)
            except:
                pass
        
        if not history_data:
            return pd.DataFrame()
        
        return pd.DataFrame(history_data)
    
    def delete_model(self, model_name: str, version: int = None):
        """Modeli sil."""
        model_dir = self.models_dir / model_name
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model bulunamadı: {model_name}")
        
        if version is None:
            # Tüm modeli sil
            shutil.rmtree(model_dir)
            print(f"[SİLİNDİ] Model klasörü: {model_dir}")
        else:
            # Belirli versiyonu sil
            if self.save_format == 'pickle':
                model_file = model_dir / f"v{version:03d}_model.pkl"
            else:
                model_file = model_dir / f"v{version:03d}_model.json"
            
            metadata_file = model_dir / f"v{version:03d}_metadata.json"
            
            if model_file.exists():
                model_file.unlink()
                print(f"[SİLİNDİ] Model: {model_file}")
            
            if metadata_file.exists():
                metadata_file.unlink()
                print(f"[SİLİNDİ] Metadata: {metadata_file}")
    
    def _get_next_version(self, model_dir: Path) -> int:
        """Sonraki versiyon numarasını al."""
        existing_versions = []
        
        for file in model_dir.glob('v*_metadata.json'):
            try:
                version_num = int(file.name[1:4])  # v001 formatından
                existing_versions.append(version_num)
            except:
                pass
        
        return max(existing_versions) + 1 if existing_versions else 1
    
    def _get_latest_version(self, model_dir: Path) -> int:
        """Son versiyon numarasını al."""
        version_num = self._get_next_version(model_dir) - 1
        return max(version_num, 1)
    
    def _cleanup_old_versions(self, model_dir: Path, model_name: str):
        """Eski versiyonları sil."""
        metadata_files = sorted(model_dir.glob('v*_metadata.json'),
                               key=lambda f: f.name)
        
        if len(metadata_files) > self.max_versions:
            # En eski versiyonları sil
            to_delete = len(metadata_files) - self.max_versions
            
            for metadata_file in metadata_files[:to_delete]:
                version_num = int(metadata_file.name[1:4])
                self.delete_model(model_name, version_num)
    
    def _save_model_as_json(self, model: Any, filepath: Path, model_name: str):
        """Modeli JSON formatında kaydet (summary)."""
        model_summary = {
            'model_type': type(model).__name__,
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Model özelliklerini ekle
        if hasattr(model, '__dict__'):
            for key, value in model.__dict__.items():
                if not key.startswith('_') and not callable(value):
                    try:
                        # JSON-serializable yap
                        if isinstance(value, (np.ndarray, np.integer, np.floating)):
                            model_summary[key] = str(value)
                        elif isinstance(value, dict):
                            model_summary[key] = value
                        else:
                            model_summary[key] = str(value)
                    except:
                        pass
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_summary, f, indent=2, ensure_ascii=False, default=str)
    
    def export_model_info(self, model_name: str, filepath: str = None) -> Dict:
        """Model bilgilerini CSV olarak dışa aktar."""
        history_df = self.get_model_history(model_name)
        
        if filepath is None:
            filepath = self.models_dir / f"{model_name}_history.csv"
        
        history_df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"[KAYDEDILDI] Model Geçmişi: {filepath}")
        
        return history_df.to_dict('records')

