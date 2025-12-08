#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gradient_boosting_model.py
--------------------------
MRI sınıflandırması için Gradient Boosting tabanlı model.

Desteklenen Algoritma:
  - XGBoost: Üretim ortamında en iyi performans ve hız
  - LightGBM: Daha düşük bellek kullanımı, alternatif seçenek
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import warnings

warnings.filterwarnings('ignore')


class GradientBoostingModel:
    """Gradient Boosting tabanlı MRI sınıflandırma modeli."""
    
    def __init__(self,
                 algorithm: str = 'xgboost',
                 n_estimators: int = 100,
                 max_depth: int = 7,
                 learning_rate: float = 0.1,
                 random_state: int = 42,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 reg_lambda: float = 1.0,
                 reg_alpha: float = 0.0):
        """
        Gradient Boosting modelini başlat.
        
        Args:
            algorithm: 'xgboost' veya 'lightgbm'
            n_estimators: Ağaç sayısı
            max_depth: Maksimum ağaç derinliği
            learning_rate: Öğrenme oranı
            random_state: Rastgele tohum
            subsample: Örnek oranı
            colsample_bytree: Öznitelik oranı
            reg_lambda: L2 regularizasyon
            reg_alpha: L1 regularizasyon
        """
        self.algorithm = algorithm.lower()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        
        self.model = None
        self.is_fitted = False
        self.n_classes = None
        self.feature_names = None
        self.feature_importance = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Seçilen algoritma için modeli başlat."""
        try:
            if self.algorithm == 'xgboost':
                import xgboost as xgb
                
                self.model = xgb.XGBClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                    reg_lambda=self.reg_lambda,
                    reg_alpha=self.reg_alpha,
                    verbosity=1,
                    use_label_encoder=False,
                    eval_metric='mlogloss'
                )
                print("[BILGI] XGBoost modeli başlatıldı")
                
            elif self.algorithm == 'lightgbm':
                import lightgbm as lgb
                
                self.model = lgb.LGBMClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                    reg_lambda=self.reg_lambda,
                    reg_alpha=self.reg_alpha,
                    verbose=1,
                    num_leaves=2 ** self.max_depth - 1
                )
                print("[BILGI] LightGBM modeli başlatıldı")
                
            else:
                raise ValueError(f"Bilinmeyen algoritma: {self.algorithm}")
        except ImportError as e:
            raise ImportError(f"Gerekli paket yüklenmemiş: {str(e)}")
    
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            early_stopping_rounds: int = 10):
        """
        Modeli eğitim veri seti ile eğit.
        
        Args:
            X_train: Eğitim özellikleri
            y_train: Eğitim etiketleri
            X_val: Doğrulama özellikleri
            y_val: Doğrulama etiketleri
            early_stopping_rounds: Early stopping turları
        """
        print(f"\n[MODEL EĞİTİMİ] Gradient Boosting ({self.algorithm.upper()})")
        print(f"  Eğitim verisi: {X_train.shape}")
        print(f"  Öznitelik sayısı: {X_train.shape[1]}")
        print(f"  Sınıf sayısı: {len(np.unique(y_train))}")
        
        self.n_classes = len(np.unique(y_train))
        self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        try:
            if self.algorithm == 'xgboost' and X_val is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False
                )
                print(f"  Early stopping en iyi: {self.model.best_iteration}")
            else:
                self.model.fit(X_train, y_train)
            
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = self.model.feature_importances_
            
            self.is_fitted = True
            print("[TAMAMLANDI] Model başarıyla eğitildi")
            
        except Exception as e:
            print(f"[HATA] Eğitim sırasında hata: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Test verileri için tahmin yap."""
        if not self.is_fitted:
            raise RuntimeError("Model henüz eğitilmemiş.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Her sınıf için tahmin olasılıklarını al."""
        if not self.is_fitted:
            raise RuntimeError("Model henüz eğitilmemiş.")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise AttributeError("Bu model olasılık tahmini desteklemiyor.")
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Model doğruluğunu hesapla."""
        if not self.is_fitted:
            raise RuntimeError("Model henüz eğitilmemiş.")
        
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Modeli kapsamlı şekilde değerlendir."""
        if not self.is_fitted:
            raise RuntimeError("Model henüz eğitilmemiş.")
        
        predictions = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision_weighted': precision_score(y, predictions, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y, predictions, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y, predictions, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y, predictions),
            'classification_report': classification_report(y, predictions, zero_division=0)
        }
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Özniteliklerin önem sıralamasını al."""
        if self.feature_importance is None:
            raise RuntimeError("Feature importance desteklenmiyor.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def hyperparameter_tuning(self,
                             X_train: np.ndarray,
                             y_train: np.ndarray,
                             cv: int = 5,
                             n_jobs: int = -1) -> dict:
        """Hiperparametre ayarlamayı gerçekleştir."""
        print("\n[HİPERPARAMETRE AYARLAMA] Grid Search")
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 7, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9]
        }
        
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            n_jobs=n_jobs,
            verbose=1,
            scoring='accuracy'
        )
        
        grid_search.fit(X_train, y_train)
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
        }
        
        print(f"  En iyi parametreler: {results['best_params']}")
        print(f"  En iyi CV skoru: {results['best_score']:.4f}")
        
        self.model.set_params(**grid_search.best_params_)
        
        return results
    
    def __repr__(self) -> str:
        """Modelin string gösterimi."""
        return (
            f"GradientBoostingModel("
            f"algorithm='{self.algorithm}', "
            f"n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, "
            f"learning_rate={self.learning_rate})"
        )
