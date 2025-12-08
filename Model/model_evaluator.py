r"""
model_evaluator.py
------------------
Eğitilmiş modelleri değerlendirme ve karşılaştırma araçları.

Sınıflar:
  - ModelEvaluator: Tek model veya birden fazla modeli değerlendirin ve karşılaştırın
  - ReportGenerator: Değerlendirme raporları oluşturun

Özellikleri:
  - Doğruluk, kesinlik, geri çağırma, F1 skoru hesaplaması
  - Karmaşıklık matrisi görselleştirmesi
  - Sınıf başına performans analizi
  - Model karşılaştırma tabloları
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import warnings

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Makine öğrenmesi modellerini değerlendirmek için sınıf."""
    
    def __init__(self, model_name: str = "Model"):
        r"""
        Evaluatörü başlat.
        
        Parametreler:
        -----------
        model_name : str
            Modelin adı
        """
        self.model_name = model_name
        self.metrics = {}
        self.predictions = None
        self.y_true = None
    
    def evaluate(self, 
                 y_true: np.ndarray,
                 y_pred: np.ndarray,
                 y_proba: Optional[np.ndarray] = None) -> Dict:
        r"""
        Modeli kapsamlı şekilde değerlendir.
        
        Parametreler:
        -----------
        y_true : np.ndarray
            Gerçek etiketler
        y_pred : np.ndarray
            Tahmin edilen etiketler
        y_proba : np.ndarray, optional
            Sınıf olasılıkları (ROC-AUC için)
        
        Döndürülen:
        ---------
        dict
            Tüm değerlendirme metrikleri
        """
        self.y_true = y_true
        self.predictions = y_pred
        
        unique_classes = np.unique(y_true)
        self.metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred, zero_division=0),
            'n_samples': len(y_true),
            'n_classes': len(unique_classes),
            'classes': unique_classes
        }
        
        # ROC-AUC (ikili sınıflandırma için)
        if len(unique_classes) == 2 and y_proba is not None:
            try:
                self.metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            except:
                self.metrics['roc_auc'] = None
        
        return self.metrics
    
    def print_summary(self):
        """Değerlendirme özetini yazdır."""
        if not self.metrics:
            print("[UYARI] Henüz değerlendirme yapılmamış.")
            return
        
        print(f"\n[DEĞERLENDIRME SONUÇLARI] {self.model_name}")
        print("=" * 60)
        
        print(f"\nGenel Metrikler:")
        print(f"  Doğruluk (Accuracy): {self.metrics['accuracy']:.4f}")
        print(f"  Kesinlik (Precision - Weighted): {self.metrics['precision_weighted']:.4f}")
        print(f"  Geri Çağırma (Recall - Weighted): {self.metrics['recall_weighted']:.4f}")
        print(f"  F1 Skoru (Weighted): {self.metrics['f1_weighted']:.4f}")
        
        if 'roc_auc' in self.metrics and self.metrics['roc_auc'] is not None:
            print(f"  ROC-AUC: {self.metrics['roc_auc']:.4f}")
        
        print(f"\nVeri Seti Bilgileri:")
        print(f"  Örnek Sayısı: {self.metrics['n_samples']}")
        print(f"  Sınıf Sayısı: {self.metrics['n_classes']}")
        
        print(f"\nSınıflandırma Raporu:")
        print(self.metrics['classification_report'])
        
        print(f"\nKarmaşıklık Matrisi:")
        self._print_confusion_matrix()
    
    def _print_confusion_matrix(self):
        """Karmaşıklık matrisini güzel formatta yazdır."""
        cm = self.metrics['confusion_matrix']
        classes = self.metrics['classes']
        
        # Header
        print(f"\n{'Sınıf':>12} ", end='')
        for cls in classes:
            print(f"{cls:>8}", end='')
        print(f" {'Toplam':>8}")
        
        # Satırlar
        for i, cls in enumerate(classes):
            print(f"{cls:>12} ", end='')
            for j in range(len(classes)):
                print(f"{cm[i, j]:>8}", end='')
            print(f" {cm[i].sum():>8}")
        
        # Toplam satırı
        print(f"{'Toplam':>12} ", end='')
        for j in range(len(classes)):
            print(f"{cm[:, j].sum():>8}", end='')
        print(f" {cm.sum():>8}")
    
    def get_per_class_metrics(self) -> pd.DataFrame:
        r"""
        Sınıf başına performans metriklerini al.
        
        Döndürülen:
        ---------
        pd.DataFrame
            Sınıf başına kesinlik, geri çağırma, F1 skoru
        """
        if not self.metrics:
            raise RuntimeError("Henüz değerlendirme yapılmamış.")
        
        report_dict = classification_report(
            self.y_true, 
            self.predictions, 
            output_dict=True,
            zero_division=0
        )
        
        # DataFrame'e dönüştür
        df = pd.DataFrame(report_dict).T
        df.index.name = 'class'
        
        return df[['precision', 'recall', 'f1-score', 'support']]
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        r"""
        Metrikleri DataFrame olarak al.
        
        Döndürülen:
        ---------
        pd.DataFrame
            Metrikler tablosu
        """
        if not self.metrics:
            raise RuntimeError("Henüz değerlendirme yapılmamış.")
        
        metrics_dict = {
            'Metric': [
                'Accuracy', 'Precision (Weighted)', 'Recall (Weighted)',
                'F1-Score (Weighted)', 'Precision (Macro)', 'Recall (Macro)',
                'F1-Score (Macro)'
            ],
            'Value': [
                self.metrics['accuracy'],
                self.metrics['precision_weighted'],
                self.metrics['recall_weighted'],
                self.metrics['f1_weighted'],
                self.metrics['precision_macro'],
                self.metrics['recall_macro'],
                self.metrics['f1_macro']
            ]
        }
        
        return pd.DataFrame(metrics_dict)


class ReportGenerator:
    """Modelleri karşılaştırmak ve rapor oluşturmak için sınıf."""
    
    def __init__(self):
        """Rapor üretecini başlat."""
        self.evaluators = {}
    
    def add_model(self, 
                  model_name: str,
                  y_true: np.ndarray,
                  y_pred: np.ndarray,
                  y_proba: Optional[np.ndarray] = None):
        r"""
        Karşılaştırma için bir modeli ekle.
        
        Parametreler:
        -----------
        model_name : str
            Modelin adı
        y_true : np.ndarray
            Gerçek etiketler
        y_pred : np.ndarray
            Tahmin edilen etiketler
        y_proba : np.ndarray, optional
            Sınıf olasılıkları
        """
        evaluator = ModelEvaluator(model_name)
        evaluator.evaluate(y_true, y_pred, y_proba)
        self.evaluators[model_name] = evaluator
    
    def compare_models(self) -> pd.DataFrame:
        r"""
        Tüm modelleri karşılaştır.
        
        Döndürülen:
        ---------
        pd.DataFrame
            Model karşılaştırma tablosu
        """
        if not self.evaluators:
            raise RuntimeError("Henüz model eklenmemiş.")
        
        comparison_data = []
        
        for model_name, evaluator in self.evaluators.items():
            row = {
                'Model': model_name,
                'Accuracy': evaluator.metrics['accuracy'],
                'Precision (W)': evaluator.metrics['precision_weighted'],
                'Recall (W)': evaluator.metrics['recall_weighted'],
                'F1-Score (W)': evaluator.metrics['f1_weighted'],
                'F1-Score (M)': evaluator.metrics['f1_macro'],
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data).sort_values('Accuracy', ascending=False)
        return df
    
    def print_comparison(self):
        """Model karşılaştırmasını yazdır."""
        comparison_df = self.compare_models()
        
        print("\n[MODEL KARŞILAŞTIRMASI]")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
        print("=" * 80)
        
        # En iyi model
        best_model = comparison_df.iloc[0]['Model']
        best_accuracy = comparison_df.iloc[0]['Accuracy']
        print(f"\n✓ En iyi model: {best_model} (Doğruluk: {best_accuracy:.4f})")
    
    def print_detailed_reports(self):
        """Tüm modellerin detaylı raporlarını yazdır."""
        for model_name, evaluator in self.evaluators.items():
            evaluator.print_summary()
    
    def export_comparison(self, filepath: str):
        r"""
        Model karşılaştırmasını dosyaya kaydet.
        
        Parametreler:
        -----------
        filepath : str
            CSV dosyasının yolu
        """
        comparison_df = self.compare_models()
        comparison_df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"[KAYDEDILDI] Karşılaştırma: {filepath}")
    
    def __repr__(self) -> str:
        """String gösterimi."""
        return f"ReportGenerator(models={len(self.evaluators)})"
