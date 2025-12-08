#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
train_and_evaluate_models.py
-----------------------------
MRI sınıflandırması için Gradient Boosting ve Linear SVM modellerini eğit ve değerlendir.

Özellikleri:
  - CSV verisi yükleme (ön işlemli görüntü özellikleri)
  - Train/Val/Test bölümleri ile veri hazırlama
  - Gradient Boosting (XGBoost) modelini eğitme
  - Lineer SVM modelini eğitme
  - Her iki modeli değerlendirme
  - Modelleri karşılaştırma
  - Raporları dosyaya kaydetme

Kullanım:
    python train_and_evaluate_models.py

Gerekli Dosyalar:
    - Görüntü_On_Isleme/çıktı/goruntu_ozellikleri_scaled.csv
    - Veri_Seti/eğitim/*, Veri_Seti/doğrulama/*, Veri_Seti/test/*
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import pickle
import json

# Parent dizini sys.path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

# Model sınıflarını içeri aktar
from Model.gradient_boosting_model import GradientBoostingModel
from Model.linear_svm_model import LinearSVMModel
from Model.model_evaluator import ModelEvaluator, ReportGenerator
from Model.config import config, validate_csv_file
from Model.visualizer import ModelVisualizer
from Model.model_manager import ModelManager


# Konfigürasyon (config modülünden yükle)
CSV_FILE_PATH = config.get('data_paths.csv_file', 'Görüntü_On_Isleme/çıktı/goruntu_ozellikleri_scaled.csv')
VERI_SETI_KLASORU = config.get('data_paths.veri_seti_klasoru', 'Veri_Seti')
OUTPUT_DIR = config.get('data_paths.output_dir', 'Model/outputs')
MODELS_DIR = config.get('data_paths.models_dir', 'Model/outputs/models')
VISUALIZATIONS_DIR = config.get('data_paths.visualizations_dir', 'Model/outputs/visualizations')
RANDOM_STATE = config.get('gradient_boosting.random_state', 42)


def create_output_directory():
    """Çıktı dizinlerini oluştur."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    print(f"[BILGI] Çıktı dizinleri oluşturuldu:")
    print(f"  - {OUTPUT_DIR}")
    print(f"  - {MODELS_DIR}")
    print(f"  - {VISUALIZATIONS_DIR}")


def load_csv_data(csv_path: str) -> pd.DataFrame:
    r"""
    CSV dosyasından veri yükle.
    
    Parametreler:
    -----------
    csv_path : str
        CSV dosyasının yolu
    
    Döndürülen:
    ---------
    pd.DataFrame
        Yüklenen veri
    """
    # CSV dosyasını valide et
    is_valid, message = validate_csv_file(csv_path)
    print(message)
    
    if not is_valid:
        print("\n[HATA] Veri yükleme iptal edildi!")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        print(f"\n[VERİ YÜKLEME] CSV başarıyla yüklendi")
        print(f"  Dosya: {csv_path}")
        print(f"  Satır: {len(df)}, Sütun: {len(df.columns)}")
        print(f"  Özellikleri: {df.columns.tolist()[:10]}...")
        return df
    except Exception as e:
        print(f"[HATA] CSV yükleme hatası: {str(e)}")
        return None


def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    r"""
    Veri setini hazırla (öznitelikler ve etiketler).
    
    Parametreler:
    -----------
    df : pd.DataFrame
        Yüklenen CSV veri seti
    
    Döndürülen:
    ---------
    Tuple[np.ndarray, np.ndarray, list]
        (Öznitelikler, Etiketler, Öznitelik adları)
    """
    print(f"\n[VERİ HAZIRLAMA]")
    
    # Etiket sütununu belirle
    if 'etiket' in df.columns:
        label_col = 'etiket'
    elif 'label' in df.columns:
        label_col = 'label'
    else:
        print("[UYARI] Etiket sütunu bulunamadı. Son sütun etiket olarak kullanılıyor.")
        label_col = df.columns[-1]
    
    # Hariç tutulacak sütunlar
    exclude_cols = {'dosya_adı', 'dosya_yolu', 'sinif', 'etiket', 'label', 
                    'id', 'filepath'}
    
    # Öznitelik sütunlarını seç
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
    
    # Verileri hazırla
    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.int32)
    
    print(f"  Seçilen öznitelikler: {len(feature_cols)}")
    print(f"  Örnekler: {X.shape[0]}")
    print(f"  Sınıflar: {np.unique(y)}")
    print(f"  Sınıf dağılımı:")
    for cls in np.unique(y):
        count = np.sum(y == cls)
        print(f"    Sınıf {cls}: {count} ({count/len(y)*100:.1f}%)")
    
    return X, y, feature_cols


def train_gradient_boosting(X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_val: Optional[np.ndarray] = None,
                           y_val: Optional[np.ndarray] = None,
                           algorithm: str = None) -> GradientBoostingModel:
    r"""
    Gradient Boosting modelini eğit.
    
    Parametreler:
    -----------
    X_train : np.ndarray
        Eğitim özellikleri
    y_train : np.ndarray
        Eğitim etiketleri
    X_val : np.ndarray, optional
        Doğrulama özellikleri (early stopping için)
    y_val : np.ndarray, optional
        Doğrulama etiketleri
    algorithm : str
        'xgboost' veya 'lightgbm'
    
    Döndürülen:
    ---------
    GradientBoostingModel
        Eğitilmiş model
    """
    print(f"\n{'='*60}")
    print(f"[GRADIENT BOOSTING MODELİ EĞİTİMİ]")
    print(f"{'='*60}")
    
    # Config'den parametreleri al
    gb_config = config.get_gb_config()
    if algorithm is None:
        algorithm = gb_config.get('algorithm', 'xgboost')
    
    model = GradientBoostingModel(
        algorithm=algorithm,
        n_estimators=gb_config.get('n_estimators', 100),
        max_depth=gb_config.get('max_depth', 7),
        learning_rate=gb_config.get('learning_rate', 0.1),
        random_state=gb_config.get('random_state', RANDOM_STATE),
        subsample=gb_config.get('subsample', 0.8),
        colsample_bytree=gb_config.get('colsample_bytree', 0.8),
        reg_lambda=gb_config.get('reg_lambda', 1.0),
        reg_alpha=gb_config.get('reg_alpha', 0.0)
    )
    
    early_stopping_rounds = gb_config.get('early_stopping_rounds', 10)
    model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds)
    
    return model


def train_linear_svm(X_train: np.ndarray,
                     y_train: np.ndarray) -> LinearSVMModel:
    r"""
    Lineer SVM modelini eğit.
    
    Parametreler:
    -----------
    X_train : np.ndarray
        Eğitim özellikleri
    y_train : np.ndarray
        Eğitim etiketleri
    
    Döndürülen:
    ---------
    LinearSVMModel
        Eğitilmiş model
    """
    print(f"\n{'='*60}")
    print(f"[LİNEER SVM MODELİ EĞİTİMİ]")
    print(f"{'='*60}")
    
    # Config'den parametreleri al
    svm_config = config.get_svm_config()
    
    model = LinearSVMModel(
        C=svm_config.get('C', 1.0),
        loss=svm_config.get('loss', 'squared_hinge'),
        max_iter=svm_config.get('max_iter', 2000),
        random_state=svm_config.get('random_state', RANDOM_STATE),
        dual=svm_config.get('dual', True),
        tol=svm_config.get('tol', 1e-4),
        class_weight=svm_config.get('class_weight', 'balanced'),
        verbose=svm_config.get('verbose', 0)
    )
    
    model.fit(X_train, y_train, scale_features=True)
    
    return model


def evaluate_models(gb_model: GradientBoostingModel,
                   svm_model: LinearSVMModel,
                   X_test: np.ndarray,
                   y_test: np.ndarray) -> Tuple[dict, dict]:
    r"""
    Her iki modeli test verisi üzerinde değerlendir.
    
    Parametreler:
    -----------
    gb_model : GradientBoostingModel
        Eğitilmiş Gradient Boosting modeli
    svm_model : LinearSVMModel
        Eğitilmiş SVM modeli
    X_test : np.ndarray
        Test özellikleri
    y_test : np.ndarray
        Test etiketleri
    
    Döndürülen:
    ---------
    Tuple[dict, dict]
        GB metrikleri, SVM metrikleri
    """
    print(f"\n{'='*60}")
    print(f"[MODEL DEĞERLENDİRMESİ] Test Verisi")
    print(f"{'='*60}")
    
    # Gradient Boosting
    print("\n[Gradient Boosting Modeli]")
    gb_metrics = gb_model.evaluate(X_test, y_test)
    gb_evaluator = ModelEvaluator("Gradient Boosting (XGBoost)")
    gb_evaluator.evaluate(y_test, gb_model.predict(X_test),
                         gb_model.predict_proba(X_test) 
                         if hasattr(gb_model.model, 'predict_proba') else None)
    gb_evaluator.print_summary()
    
    # Lineer SVM
    print("\n[Lineer SVM Modeli]")
    svm_metrics = svm_model.evaluate(X_test, y_test)
    svm_evaluator = ModelEvaluator("Lineer SVM")
    svm_evaluator.evaluate(y_test, svm_model.predict(X_test))
    svm_evaluator.print_summary()
    
    return gb_metrics, svm_metrics, gb_evaluator, svm_evaluator


def compare_models(gb_evaluator: ModelEvaluator,
                  svm_evaluator: ModelEvaluator):
    """Modelleri karşılaştır."""
    print(f"\n{'='*60}")
    print(f"[MODEL KARŞILAŞTIRMASI]")
    print(f"{'='*60}")
    
    reporter = ReportGenerator()
    reporter.add_model(
        "Gradient Boosting",
        gb_evaluator.y_true,
        gb_evaluator.predictions
    )
    reporter.add_model(
        "Lineer SVM",
        svm_evaluator.y_true,
        svm_evaluator.predictions
    )
    
    reporter.print_comparison()
    
    # CSV'ye kaydet
    output_csv = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    reporter.export_comparison(output_csv)


def visualize_results(gb_model: GradientBoostingModel,
                      svm_model: LinearSVMModel,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      gb_eval: ModelEvaluator,
                      svm_eval: ModelEvaluator,
                      feature_names: list):
    """Model sonuçlarını görselleştir."""
    print(f"\n{'='*60}")
    print(f"[GÖRSELLEŞTIRME] Model Sonuçları")
    print(f"{'='*60}")
    
    visualizer = ModelVisualizer(output_dir=VISUALIZATIONS_DIR)
    
    # Tahminleri al
    gb_pred = gb_model.predict(X_test)
    svm_pred = svm_model.predict(X_test)
    
    # Karmaşıklık matrisleri çiz
    visualizer.plot_confusion_matrix(
        y_test, gb_pred,
        class_names=['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented'],
        title="Gradient Boosting - Karmaşıklık Matrisi",
        filename="gb_confusion_matrix.png"
    )
    
    visualizer.plot_confusion_matrix(
        y_test, svm_pred,
        class_names=['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented'],
        title="Linear SVM - Karmaşıklık Matrisi",
        filename="svm_confusion_matrix.png"
    )
    
    # Feature importance (Gradient Boosting için)
    if gb_model.feature_importance is not None:
        visualizer.plot_feature_importance(
            feature_names[:len(gb_model.feature_importance)],
            gb_model.feature_importance,
            top_n=20,
            title="Gradient Boosting - Öznitelik Önem Sıralaması",
            filename="gb_feature_importance.png"
        )
    
    # Model karşılaştırması
    models_data = {
        'Gradient Boosting': gb_eval.metrics,
        'Linear SVM': svm_eval.metrics,
    }
    
    visualizer.plot_model_comparison(
        models_data,
        metric='accuracy',
        title="Model Karşılaştırması - Doğruluk",
        filename="model_comparison_accuracy.png"
    )
    
    visualizer.plot_model_comparison(
        models_data,
        metric='f1_weighted',
        title="Model Karşılaştırması - F1 Skoru",
        filename="model_comparison_f1.png"
    )
    
    print(f"✓ Tüm görseller kaydedildi: {VISUALIZATIONS_DIR}")



def save_models(gb_model: GradientBoostingModel,
               svm_model: LinearSVMModel):
    """Eğitilmiş modelleri kaydet."""
    print(f"\n{'='*60}")
    print(f"[MODELLERİ KAYDETME]")
    print(f"{'='*60}")
    
    # Model Manager'ı oluştur
    manager = ModelManager(models_dir=MODELS_DIR)
    
    # Gradient Boosting modelini kaydet
    gb_config = config.get_gb_config()
    gb_manager = ModelManager(models_dir=MODELS_DIR)
    gb_manager.save_model(
        gb_model,
        model_name="gradient_boosting",
        config=gb_config,
        training_info={'algorithm': gb_config.get('algorithm', 'xgboost')}
    )
    
    # SVM modelini kaydet
    svm_config = config.get_svm_config()
    svm_manager = ModelManager(models_dir=MODELS_DIR)
    svm_manager.save_model(
        svm_model,
        model_name="linear_svm",
        config=svm_config,
        training_info={'loss': svm_config.get('loss', 'squared_hinge')}
    )
    
    print(f"✓ Modeller başarıyla kaydedildi")


def main():
    """Ana çalıştırma fonksiyonu."""
    print("\n" + "="*60)
    print("MRI SINIFLANDIRMASI - MODEL EĞİTİMİ VE DEĞERLENDİRMESİ")
    print("="*60)
    
    # Çıktı dizini oluştur
    create_output_directory()
    
    # Veri yükle
    csv_path = CSV_FILE_PATH
    if not os.path.exists(csv_path):
        print(f"[UYARI] CSV dosyası bulunamadı: {csv_path}")
        print("[BİLGİ] Örnek CSV oluşturuluyor...")
        # Burada örnek veri oluşturabilir veya kullanıcıya uyarı gösterebilirsiniz
        return
    
    df = load_csv_data(csv_path)
    if df is None:
        return
    
    # Verileri hazırla
    X, y, feature_cols = prepare_data(df)
    
    # Train/Val/Test bölümü (70/15/15)
    from sklearn.model_selection import train_test_split
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15/0.85, 
        random_state=RANDOM_STATE, stratify=y_temp
    )
    
    print(f"\n[VERİ BÖLÜMLEME]")
    print(f"  Eğitim: {X_train.shape[0]} ({X_train.shape[0]/len(y)*100:.1f}%)")
    print(f"  Doğrulama: {X_val.shape[0]} ({X_val.shape[0]/len(y)*100:.1f}%)")
    print(f"  Test: {X_test.shape[0]} ({X_test.shape[0]/len(y)*100:.1f}%)")
    
    # Modelleri eğit
    try:
        gb_model = train_gradient_boosting(X_train, y_train, X_val, y_val, 
                                          algorithm='xgboost')
    except ImportError as e:
        print(f"[UYARI] XGBoost kurulu değil: {e}")
        print("[BİLGİ] LightGBM deniyor...")
        try:
            gb_model = train_gradient_boosting(X_train, y_train, X_val, y_val,
                                              algorithm='lightgbm')
        except ImportError:
            print("[HATA] XGBoost ve LightGBM hiçbiri kurulu değil.")
            return
    
    svm_model = train_linear_svm(X_train, y_train)
    
    # Modelleri değerlendir
    gb_metrics, svm_metrics, gb_eval, svm_eval = evaluate_models(
        gb_model, svm_model, X_test, y_test
    )
    
    # Karşılaştır
    compare_models(gb_eval, svm_eval)
    
    # Görselleştir
    visualize_results(gb_model, svm_model, X_test, y_test, gb_eval, svm_eval, feature_cols)
    
    # Modelleri kaydet
    save_models(gb_model, svm_model)
    
    print(f"\n{'='*60}")
    print("[TAMAMLANDI] Eğitim ve değerlendirme işlemi tamamlandı")
    print(f"Çıktı dizini: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # CSV dosyasını valide et
    is_valid, msg = validate_csv_file(CSV_FILE_PATH)
    if not is_valid:
        print(msg)
        print("\n[HATA] CSV dosyası bulunamadı. Lütfen şu adımları yapın:")
        print("1. Görüntü_On_Isleme/goruntu_isleme_kontrol_paneli.py'yi çalıştırın")
        print("2. Toplu ön işleme yapın")
        print("3. CSV oluşturun")
        print("4. Normalizasyon uygulayın")
        sys.exit(1)
    
    main()
