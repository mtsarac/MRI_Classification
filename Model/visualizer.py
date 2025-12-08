#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
visualizer.py
--------------
Model değerlendirmesi için görselleştirme araçları.

Özellikler:
  - Karmaşıklık matrisi (Confusion Matrix) heatmap
  - Öznitelik önem sıralaması (Feature Importance)
  - ROC eğrisi ve AUC skoru
  - Model performans karşılaştırma grafikleri
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Matplotlib uyarılarını bastır
import matplotlib
matplotlib.use('Agg')  # Headless ortam için


class ModelVisualizer:
    """Model sonuçlarını görselleştirmek için sınıf."""
    
    def __init__(self,
                 output_dir: str = 'Model/outputs/visualizations',
                 figsize: Tuple[int, int] = (12, 8),
                 dpi: int = 100,
                 style: str = 'default'):
        r"""
        Visualizer'ı başlat.
        
        Parametreler:
        -----------
        output_dir : str
            Grafiklerin kaydedileceği dizin
        figsize : tuple
            Grafik boyutu (genişlik, yükseklik)
        dpi : int
            Grafik çözünürlüğü
        style : str
            Matplotlib stili
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        self.dpi = dpi
        
        # Stil ayarla
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # Türkçe karakter desteği
        plt.rcParams['font.family'] = 'DejaVu Sans'
    
    def plot_confusion_matrix(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             class_names: Optional[List[str]] = None,
                             title: str = "Karmaşıklık Matrisi",
                             filename: str = "confusion_matrix.png",
                             normalize: bool = False) -> Path:
        r"""
        Karmaşıklık matrisini çiz.
        
        Parametreler:
        -----------
        y_true : np.ndarray
            Gerçek etiketler
        y_pred : np.ndarray
            Tahmin edilen etiketler
        class_names : list, optional
            Sınıf adları
        title : str
            Grafik başlığı
        filename : str
            Kaydedilecek dosya adı
        normalize : bool
            Normalize edilsin mi?
        
        Döndürülen:
        ---------
        Path
            Kaydedilen dosya yolu
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        # Grafik oluştur
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Sınıf adları ayarla
        if class_names is None:
            class_names = [f"Sınıf {i}" for i in range(len(cm))]
        
        # Heatmap çiz
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cbar_kws={'label': 'Sayı'},
                   ax=ax)
        
        ax.set_xlabel('Tahmin Edilen', fontsize=12, fontweight='bold')
        ax.set_ylabel('Gerçek', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Kaydet
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"[KAYDEDILDI] Karmaşıklık Matrisi: {filepath}")
        return filepath
    
    def plot_feature_importance(self,
                               feature_names: List[str],
                               importance_values: np.ndarray,
                               top_n: int = 20,
                               title: str = "Öznitelik Önem Sıralaması",
                               filename: str = "feature_importance.png") -> Path:
        r"""
        Öznitelik önem sıralamasını çiz.
        
        Parametreler:
        -----------
        feature_names : list
            Öznitelik adları
        importance_values : np.ndarray
            Önem değerleri
        top_n : int
            Kaç adet top öznitelik gösterilecek
        title : str
            Grafik başlığı
        filename : str
            Kaydedilecek dosya adı
        
        Döndürülen:
        ---------
        Path
            Kaydedilen dosya yolu
        """
        # DataFrame oluştur ve sırala
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_values
        }).sort_values('Importance', ascending=True).tail(top_n)
        
        # Grafik oluştur
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
        df.plot(kind='barh', x='Feature', y='Importance',
               color=colors, ax=ax, legend=False)
        
        ax.set_xlabel('Önem Değeri', fontsize=12, fontweight='bold')
        ax.set_ylabel('Öznitelik', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Kaydet
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"[KAYDEDILDI] Öznitelik Önem: {filepath}")
        return filepath
    
    def plot_roc_curve(self,
                      y_true: np.ndarray,
                      y_scores: np.ndarray,
                      title: str = "ROC Eğrisi",
                      filename: str = "roc_curve.png") -> Path:
        r"""
        ROC eğrisini çiz (ikili sınıflandırma için).
        
        Parametreler:
        -----------
        y_true : np.ndarray
            Gerçek etiketler
        y_scores : np.ndarray
            Tahmin olasılıkları (pozitif sınıf)
        title : str
            Grafik başlığı
        filename : str
            Kaydedilecek dosya adı
        
        Döndürülen:
        ---------
        Path
            Kaydedilen dosya yolu
        """
        from sklearn.metrics import roc_curve, auc
        
        # ROC eğrisi hesapla
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Grafik oluştur
        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
        
        ax.plot(fpr, tpr, color='darkorange', lw=2,
               label=f'ROC Eğrisi (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
               label='Rastgele Sınıflandırıcı')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Yanlış Pozitif Oranı (FPR)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Doğru Pozitif Oranı (TPR)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Kaydet
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"[KAYDEDILDI] ROC Eğrisi (AUC={roc_auc:.3f}): {filepath}")
        return filepath
    
    def plot_model_comparison(self,
                             models_data: Dict[str, Dict],
                             metric: str = 'accuracy',
                             title: str = "Model Karşılaştırması",
                             filename: str = "model_comparison.png") -> Path:
        r"""
        Model performanslarını karşılaştır.
        
        Parametreler:
        -----------
        models_data : dict
            Model adları ve metriklerini içeren sözlük
            Format: {'Model Adı': {'accuracy': 0.95, 'f1': 0.93, ...}, ...}
        metric : str
            Karşılaştırılacak metrik
        title : str
            Grafik başlığı
        filename : str
            Kaydedilecek dosya adı
        
        Döndürülen:
        ---------
        Path
            Kaydedilen dosya yolu
        """
        # Veriyi hazırla
        model_names = list(models_data.keys())
        metric_values = [models_data[m].get(metric, 0) for m in model_names]
        
        # Grafik oluştur
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        bars = ax.bar(model_names, metric_values, color=colors, edgecolor='black', linewidth=1.5)
        
        # Değerleri çubuk üstüne yaz
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel(metric.capitalize(), fontsize=12, fontweight='bold')
        ax.set_xlabel('Modeller', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim([0, max(metric_values) * 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Kaydet
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"[KAYDEDILDI] Model Karşılaştırması: {filepath}")
        return filepath
    
    def plot_metrics_heatmap(self,
                            metrics_df: pd.DataFrame,
                            title: str = "Model Metrikleri Heatmap",
                            filename: str = "metrics_heatmap.png") -> Path:
        r"""
        Model metriklerini heatmap olarak çiz.
        
        Parametreler:
        -----------
        metrics_df : pd.DataFrame
            Metrikler tablosu (satır: modeller, sütun: metrikler)
        title : str
            Grafik başlığı
        filename : str
            Kaydedilecek dosya adı
        
        Döndürülen:
        ---------
        Path
            Kaydedilen dosya yolu
        """
        # Grafik oluştur
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlGn',
                   center=0.5, vmin=0, vmax=1,
                   cbar_kws={'label': 'Metrik Değeri'},
                   ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Kaydet
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"[KAYDEDILDI] Metrikler Heatmap: {filepath}")
        return filepath
    
    def create_report_summary(self,
                             report_data: Dict,
                             filename: str = "report_summary.txt") -> Path:
        r"""
        Rapor özetini metin dosyası olarak kaydet.
        
        Parametreler:
        -----------
        report_data : dict
            Rapor verileri
        filename : str
            Dosya adı
        
        Döndürülen:
        ---------
        Path
            Kaydedilen dosya yolu
        """
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("MODEL DEĞERLENDİRME RAPORU\n")
            f.write("="*60 + "\n\n")
            
            for key, value in report_data.items():
                f.write(f"{key}:\n")
                if isinstance(value, dict):
                    for k, v in value.items():
                        f.write(f"  {k}: {v}\n")
                elif isinstance(value, (list, np.ndarray)):
                    f.write(f"  {value}\n")
                else:
                    f.write(f"  {value}\n")
                f.write("\n")
        
        print(f"[KAYDEDILDI] Rapor Özeti: {filepath}")
        return filepath

