r"""
linear_svm_model.py
-------------------
MRI sınıflandırması için Lineer Support Vector Machine (SVM) modeli.

Özellikler:
  - Lineer kernel kullanarak hızlı ve verimli sınıflandırma
  - Düşük boyutlu veya yüksek boyutlu veri setlerine uygun
  - One-vs-Rest stratejisi ile çok sınıflı sınıflandırma
  - L2 regularizasyon desteği

Sınıf: LinearSVMModel
Parametreler:
  - C: Regularizasyon parametresi (varsayılan: 1.0)
  - max_iter: Maksimum iterasyon sayısı (varsayılan: 1000)
  - dual: Dual veya primal formülasyon
  - random_state: Tekrarlanabilirlik için (varsayılan: 42)

Kullanım:
    from Model.linear_svm_model import LinearSVMModel
    
    model = LinearSVMModel(C=1.0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = model.score(X_test, y_test)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings('ignore')


class LinearSVMModel:
    """Lineer SVM tabanlı MRI sınıflandırma modeli."""
    
    def __init__(self,
                 C: float = 1.0,
                 loss: str = 'squared_hinge',
                 max_iter: int = 2000,
                 random_state: int = 42,
                 dual: bool = True,
                 tol: float = 1e-4,
                 class_weight: Optional[str] = 'balanced',
                 verbose: int = 0):
        r"""
        Lineer SVM modelini başlat.
        
        Parametreler:
        -----------
        C : float
            Regularizasyon parametresi. Küçük değer daha sağlam model.
            Büyük değer eğitim verilerine daha uyumlu model.
        loss : str
            Kayıp fonksiyonu ('hinge' veya 'squared_hinge')
        max_iter : int
            Maksimum iterasyon sayısı
        random_state : int
            Rastgele tohum tekrarlanabilirlik için
        dual : bool
            Dual (True) veya primal (False) formülasyon
        tol : float
            Eğitim toleransı
        class_weight : str, optional
            Sınıf ağırlıklandırması ('balanced' = sınıf dengesini otomatik ayarla)
        verbose : int
            Verbose seviyesi (0 = sessiz)
        """
        self.C = C
        self.loss = loss
        self.max_iter = max_iter
        self.random_state = random_state
        self.dual = dual
        self.tol = tol
        self.class_weight = class_weight
        self.verbose = verbose
        
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.n_classes = None
        self.feature_names = None
        self.classes = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Lineer SVM modelini başlat."""
        try:
            self.model = LinearSVC(
                C=self.C,
                loss=self.loss,
                max_iter=self.max_iter,
                random_state=self.random_state,
                dual=self.dual,
                tol=self.tol,
                class_weight=self.class_weight,
                verbose=self.verbose
            )
            self.scaler = StandardScaler()
            print("[BILGI] Lineer SVM modeli başlatıldı")
        except Exception as e:
            raise RuntimeError(f"Model başlatma hatası: {str(e)}")
    
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            scale_features: bool = True):
        r"""
        Modeli eğitim veri seti ile eğit.
        
        Parametreler:
        -----------
        X_train : np.ndarray
            Eğitim özellikleri (n_samples, n_features)
        y_train : np.ndarray
            Eğitim etiketleri (n_samples,)
        scale_features : bool
            Özellikleri StandardScaler ile ölçeklendir
        """
        print(f"\n[MODEL EĞİTİMİ] Lineer SVM")
        print(f"  Eğitim verisi: {X_train.shape}")
        print(f"  Öznitelik sayısı: {X_train.shape[1]}")
        print(f"  Sınıf sayısı: {len(np.unique(y_train))}")
        
        self.n_classes = len(np.unique(y_train))
        self.classes = np.unique(y_train)
        self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        try:
            # Özellikleri ölçeklendir (SVM için önemli)
            if scale_features:
                X_train_scaled = self.scaler.fit_transform(X_train)
                print("  Öznitelikler StandardScaler ile ölçeklendirildi")
            else:
                X_train_scaled = X_train
            
            # Modeli eğit
            self.model.fit(X_train_scaled, y_train)
            self.is_fitted = True
            print("[TAMAMLANDI] Model başarıyla eğitildi")
            
        except Exception as e:
            print(f"[HATA] Eğitim sırasında hata oluştu: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        r"""
        Test verileri için tahmin yap.
        
        Parametreler:
        -----------
        X : np.ndarray
            Tahmin yapılacak öznitelikler (n_samples, n_features)
        
        Döndürülen:
        ---------
        np.ndarray
            Tahmin edilen etiketler (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model henüz eğitilmemiş. Önce fit() metodunu çağırın.")
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        r"""
        Karar fonksiyonunun çıktısını al (sınırından uzaklık).
        
        Parametreler:
        -----------
        X : np.ndarray
            Öznitelikler
        
        Döndürülen:
        ---------
        np.ndarray
            Karar fonksiyonu skoru
        """
        if not self.is_fitted:
            raise RuntimeError("Model henüz eğitilmemiş.")
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.decision_function(X_scaled)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        r"""
        Model doğruluğunu hesapla.
        
        Parametreler:
        -----------
        X : np.ndarray
            Test özellikleri
        y : np.ndarray
            Gerçek etiketler
        
        Döndürülen:
        ---------
        float
            Doğruluk skoru (0-1)
        """
        if not self.is_fitted:
            raise RuntimeError("Model henüz eğitilmemiş.")
        
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        r"""
        Modeli kapsamlı şekilde değerlendir.
        
        Parametreler:
        -----------
        X : np.ndarray
            Test özellikleri
        y : np.ndarray
            Gerçek etiketler
        
        Döndürülen:
        ---------
        dict
            Değerlendirme metrikleri
        """
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
    
    def get_coefficients(self, feature_names: Optional[list] = None) -> pd.DataFrame:
        r"""
        Model katsayılarını al (öznitelik ağırlıkları).
        
        Parametreler:
        -----------
        feature_names : list, optional
            Öznitelik adları
        
        Döndürülen:
        ---------
        pd.DataFrame
            Öznitelikler ve katsayıları
        """
        if not self.is_fitted:
            raise RuntimeError("Model henüz eğitilmemiş.")
        
        if feature_names is None:
            feature_names = self.feature_names
        
        if self.model.coef_.shape[0] == 1:
            # İkili sınıflandırma
            coef_values = self.model.coef_[0]
        else:
            # Çok sınıflı sınıflandırma - ortalama almayı seçebilir
            coef_values = np.mean(np.abs(self.model.coef_), axis=0)
        
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coef_values,
            'abs_coefficient': np.abs(coef_values)
        }).sort_values('abs_coefficient', ascending=False)
        
        return coef_df
    
    def hyperparameter_tuning(self,
                             X_train: np.ndarray,
                             y_train: np.ndarray,
                             cv: int = 5,
                             n_jobs: int = -1) -> Dict:
        r"""
        Hiperparametre ayarlamayı gerçekleştir (Grid Search).
        
        Parametreler:
        -----------
        X_train : np.ndarray
            Eğitim özellikleri
        y_train : np.ndarray
            Eğitim etiketleri
        cv : int
            Cross-validation katları
        n_jobs : int
            Paralel işler (-1 = tüm çekirdekler)
        
        Döndürülen:
        ---------
        dict
            En iyi parametreler ve skor
        """
        print("\n[HİPERPARAMETRE AYARLAMA] Grid Search")
        
        # Özellikleri ölçeklendir
        X_scaled = self.scaler.fit_transform(X_train)
        
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'loss': ['hinge', 'squared_hinge'],
            'max_iter': [1000, 2000, 5000]
        }
        
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            n_jobs=n_jobs,
            verbose=1,
            scoring='accuracy'
        )
        
        grid_search.fit(X_scaled, y_train)
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        print(f"  En iyi parametreler: {results['best_params']}")
        print(f"  En iyi CV skoru: {results['best_score']:.4f}")
        
        # Modeli en iyi parametrelerle güncelle
        self.model.set_params(**grid_search.best_params_)
        
        return results
    
    def __repr__(self) -> str:
        """Modelin string gösterimi."""
        return (
            f"LinearSVMModel("
            f"C={self.C}, "
            f"loss='{self.loss}', "
            f"max_iter={self.max_iter})"
        )
