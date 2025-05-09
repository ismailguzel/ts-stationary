# scripts/evaluate_models.py
import os
import numpy as np
import pandas as pd
import pickle
import argparse
import sys
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from joblib import load

# Modül yolunu ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import (
    prepare_features_for_training, 
    prepare_lstm_sequences
)
from utils.visualization import (
    plot_confusion_matrix, 
    plot_model_comparison
)

def load_xgboost_model(model_dir):
    """
    XGBoost modelini yükle.
    
    Args:
        model_dir (str): Model klasörü
        
    Returns:
        xgb.XGBClassifier: Yüklenen model
    """
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(model_dir, 'seasonality_model.json'))
    
    return model

def load_lstm_model(model_dir):
    """
    LSTM modelini yükle.
    
    Args:
        model_dir (str): Model klasörü
        
    Returns:
        torch.nn.Module: Yüklenen model
    """
    try:
        import torch
        import torch.nn as nn
        import pickle
        
        # Dosya kontrolü
        model_info_path = os.path.join(model_dir, 'model_info.pkl')
        model_path = os.path.join(model_dir, 'seasonality_model.pth')
        
        if not os.path.exists(model_info_path):
            print(f"UYARI: model_info.pkl dosyası bulunamadı: {model_info_path}")
            return None
            
        if not os.path.exists(model_path):
            print(f"UYARI: seasonality_model.pth dosyası bulunamadı: {model_path}")
            return None
        
        # Model bilgilerini yükle
        with open(model_info_path, 'rb') as f:
            model_info = pickle.load(f)
        
        # LSTM modelini tanımla
        class LSTMModel(nn.Module):
            def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout
                )
                self.fc = nn.Linear(hidden_size, 1)
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                out = self.fc(lstm_out[:, -1, :])
                out = self.sigmoid(out)
                return out
        
        # Modeli oluştur
        model = LSTMModel(
            input_size=model_info.get('input_size', 1),
            hidden_size=model_info.get('hidden_size', 64),
            num_layers=model_info.get('num_layers', 2),
            dropout=model_info.get('dropout_rate', 0.2)
        )
        
        # Modeli yükle
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # Değerlendirme moduna al
        
        return model
    except ImportError:
        print("UYARI: PyTorch yüklü değil. LSTM modeli değerlendirilemeyecek.")
        return None
    except Exception as e:
        print(f"HATA: LSTM modeli yüklenirken bir sorun oluştu: {e}")
        return None

def evaluate_xgboost(model, X_test, y_test):
    """
    XGBoost modelini değerlendir.
    
    Args:
        model: XGBoost modeli
        X_test: Test özellikleri
        y_test: Test etiketleri
        
    Returns:
        dict: Değerlendirme sonuçları
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrikleri hesapla
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Karışıklık matrisi
    cm = confusion_matrix(y_test, y_pred)
    
    # Sınıflandırma raporu
    report = classification_report(y_test, y_pred, target_names=['Non-Seasonal', 'Seasonal'])
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def evaluate_lstm(model, X_test, y_test):
    """
    LSTM modelini değerlendir.
    
    Args:
        model: LSTM modeli (PyTorch)
        X_test: Test sekansları
        y_test: Test etiketleri
        
    Returns:
        dict: Değerlendirme sonuçları
    """
    try:
        import torch
        
        # PyTorch tensor'a dönüştür
        X_test_tensor = torch.FloatTensor(X_test)
        
        # Tahmin
        model.eval()
        with torch.no_grad():
            y_pred_proba = model(X_test_tensor).cpu().numpy().flatten()
        
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Metrikleri hesapla
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Karışıklık matrisi
        cm = confusion_matrix(y_test, y_pred)
        
        # Sınıflandırma raporu
        report = classification_report(y_test, y_pred, target_names=['Non-Seasonal', 'Seasonal'])
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    except Exception as e:
        print(f"HATA: LSTM değerlendirme sırasında bir sorun oluştu: {e}")
        return None

def main(args):
    print("\n" + "="*80)
    print(" "*35 + "MODEL DEĞERLENDİRME")
    print("="*80 + "\n")
    
    # Veri klasörleri
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    raw_dir = os.path.join(data_dir, 'raw')
    processed_dir = os.path.join(data_dir, 'processed')
    
    # Model klasörleri
    xgboost_model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'xgboost_model')
    lstm_model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'lstm_model')
    
    # Karşılaştırma sonuçları için klasör
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'comparison')
    os.makedirs(results_dir, exist_ok=True)
    
    # Zaman serilerini yükle
    print("Veriler yükleniyor...")
    all_series_df = pd.read_csv(os.path.join(raw_dir, 'all_series.csv'))
    
    # Etiketleri yükle
    seasonality_labels = pd.read_csv(os.path.join(processed_dir, 'seasonality_labels.csv'))
    
    # Test indekslerini yükle
    test_indices = np.load(os.path.join(processed_dir, 'test_indices.npy'))
    
    # Tüm serileri bir sözlük olarak grupla
    series_dict = {}
    for series_id, group in all_series_df.groupby('series_id'):
        series_dict[series_id] = pd.Series(group['y'].values, index=pd.to_datetime(group['ds']))
    
    # Test serilerini al
    test_series = [series_dict[i] for i in test_indices if i in series_dict]
    
    # Test etiketleri
    test_seasonality_labels = seasonality_labels[seasonality_labels['series_id'].isin(test_indices)]
    
    print(f"Test seti boyutu: {len(test_series)}")
    
    # XGBoost için veri hazırlığı
    print("\n" + "-"*80)
    print(" "*30 + "XGBOOST DEĞERLENDİRME")
    print("-"*80)
    
    print("\nXGBoost için özellikler hazırlanıyor...")
    test_features = prepare_features_for_training(test_series)
    
    X_test_xgb = test_features.drop(['series_id'], axis=1)
    y_test = test_seasonality_labels['is_seasonal'].values
    
    # Eksik değerleri doldur
    X_test_xgb = X_test_xgb.fillna(0)
    
    # Ölçeklendirme (eğer kaydedilmiş ölçekleyici varsa)
    scaler_path = os.path.join(xgboost_model_dir, 'scaler.joblib')
    if os.path.exists(scaler_path):
        scaler = load(scaler_path)
        X_test_xgb = scaler.transform(X_test_xgb)
    
    # XGBoost modelini yükle
    print("XGBoost modeli yükleniyor...")
    try:
        xgboost_model = load_xgboost_model(xgboost_model_dir)
        print("XGBoost modeli başarıyla yüklendi.")
    except Exception as e:
        print(f"HATA: XGBoost modeli yüklenirken bir sorun oluştu: {e}")
        print("XGBoost değerlendirmesi atlanıyor.")
        xgboost_model = None
        xgboost_results = None
    
    # XGBoost model değerlendirmesi
    if xgboost_model is not None:
        print("XGBoost modeli değerlendiriliyor...")
        xgboost_results = evaluate_xgboost(xgboost_model, X_test_xgb, y_test)
        
        print("\nXGBoost Sonuçları:")
        print(f"Accuracy: {xgboost_results['accuracy']:.4f}")
        print(f"Precision: {xgboost_results['precision']:.4f}")
        print(f"Recall: {xgboost_results['recall']:.4f}")
        print(f"F1 Score: {xgboost_results['f1']:.4f}")
        print(f"AUC: {xgboost_results['auc']:.4f}")
        print("\nKarışıklık Matrisi:")
        print(xgboost_results['confusion_matrix'])
        print("\nSınıflandırma Raporu:")
        print(xgboost_results['classification_report'])
    
    # LSTM için veri hazırlığı
    print("\n" + "-"*80)
    print(" "*30 + "LSTM DEĞERLENDİRME")
    print("-"*80)
    
    # PyTorch'un yüklü olup olmadığını kontrol et
    try:
        import torch
        pytorch_available = True
    except ImportError:
        pytorch_available = False
        print("\nUYARI: PyTorch yüklü değil. LSTM değerlendirmesi atlanıyor.")
    
    if pytorch_available:
        print("\nLSTM için sekanslar hazırlanıyor...")
        
        seq_length = args.seq_length
        
        X_test_lstm = []
        y_test_lstm = []
        
        # StandardScaler import et
        from sklearn.preprocessing import StandardScaler
        
        for i, ts in enumerate(test_series):
            # ID'yi bul
            series_id = test_indices[i]
            
            # Mevsimsellik etiketi
            is_seasonal = test_seasonality_labels[test_seasonality_labels['series_id'] == series_id]['is_seasonal'].values[0]
            
            # Zaman serisini normalize et
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(ts.values.reshape(-1, 1)).flatten()
            
            # Sekansları oluştur
            for j in range(0, len(normalized_data) - seq_length, args.step):
                X_test_lstm.append(normalized_data[j:j+seq_length])
                y_test_lstm.append(is_seasonal)
        
        # Dizilere dönüştür
        X_test_lstm = np.array(X_test_lstm).reshape(-1, seq_length, 1)
        y_test_lstm = np.array(y_test_lstm)
        
        print(f"Test sekansları boyutu: {X_test_lstm.shape}")
        
        # LSTM modelini yükle
        print("LSTM modeli yükleniyor...")
        lstm_model = load_lstm_model(lstm_model_dir)
        
        # LSTM model değerlendirmesi
        if lstm_model is not None:
            print("LSTM modeli değerlendiriliyor...")
            lstm_results = evaluate_lstm(lstm_model, X_test_lstm, y_test_lstm)
            
            if lstm_results is not None:
                print("\nLSTM Sonuçları:")
                print(f"Accuracy: {lstm_results['accuracy']:.4f}")
                print(f"Precision: {lstm_results['precision']:.4f}")
                print(f"Recall: {lstm_results['recall']:.4f}")
                print(f"F1 Score: {lstm_results['f1']:.4f}")
                print(f"AUC: {lstm_results['auc']:.4f}")
                print("\nKarışıklık Matrisi:")
                print(lstm_results['confusion_matrix'])
                print("\nSınıflandırma Raporu:")
                print(lstm_results['classification_report'])
            else:
                print("LSTM değerlendirmesi başarısız oldu.")
                lstm_results = None
        else:
            print("LSTM modeli yüklenemedi. LSTM değerlendirmesi atlanıyor.")
            lstm_results = None
    else:
        lstm_results = None
    
    # Model karşılaştırması (sadece başarılı modeller için)
    print("\n" + "-"*80)
    print(" "*30 + "MODEL KARŞILAŞTIRMA")
    print("-"*80 + "\n")
    
    if xgboost_results is not None and lstm_results is not None:
        # Her iki model de varsa karşılaştır
        model_names = ['XGBoost', 'LSTM']
        metrics = np.array([
            [xgboost_results['accuracy'], xgboost_results['precision'], xgboost_results['recall'], xgboost_results['f1'], xgboost_results['auc']],
            [lstm_results['accuracy'], lstm_results['precision'], lstm_results['recall'], lstm_results['f1'], lstm_results['auc']]
        ])
        
        print("Model Karşılaştırma Tablosu:")
        comparison_df = pd.DataFrame({
            'Metrik': ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'],
            'XGBoost': metrics[0],
            'LSTM': metrics[1]
        })
        print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        
    elif xgboost_results is not None:
        # Sadece XGBoost varsa
        model_names = ['XGBoost']
        metrics = np.array([
            [xgboost_results['accuracy'], xgboost_results['precision'], xgboost_results['recall'], xgboost_results['f1'], xgboost_results['auc']]
        ])
        
        print("Sadece XGBoost modeli değerlendirildi.")
        
    elif lstm_results is not None:
        # Sadece LSTM varsa
        model_names = ['LSTM']
        metrics = np.array([
            [lstm_results['accuracy'], lstm_results['precision'], lstm_results['recall'], lstm_results['f1'], lstm_results['auc']]
        ])
        
        print("Sadece LSTM modeli değerlendirildi.")
        
    else:
        # Hiçbir model değerlendirilemedi
        print("Hiçbir model değerlendirilemedi!")
        return
    
    # Metrik isimleri
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    
    # Grafik çiz
    plt.figure(figsize=(12, 8))
    x = np.arange(len(model_names))
    width = 0.15
    
    for i, metric_name in enumerate(metric_names):
        plt.bar(x + i * width - 0.3, metrics[:, i], width, label=metric_name)
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.ylim(0, 1.0)
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(results_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Sonuçları kaydet
    comparison_results = {
        'xgboost': xgboost_results,
        'lstm': lstm_results,
        'model_names': model_names,
        'metrics': metrics,
        'metric_names': metric_names
    }
    
    with open(os.path.join(results_dir, 'comparison_results.pkl'), 'wb') as f:
        pickle.dump(comparison_results, f)
    
    # Sonuçları CSV olarak da kaydet
    results_df = pd.DataFrame({
        'Model': model_names,
    })
    
    for i, metric in enumerate(metric_names):
        results_df[metric] = metrics[:, i]
    
    results_df.to_csv(os.path.join(results_dir, 'model_comparison.csv'), index=False)
    
    print(f"\nKarşılaştırma sonuçları ve grafikler kaydedildi: {results_dir}")
    print("\n" + "="*80)
    print(" "*30 + "DEĞERLENDİRME TAMAMLANDI")
    print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Modelleri değerlendir ve karşılaştır')
    parser.add_argument('--seq-length', type=int, default=50, help='LSTM sekans uzunluğu')
    parser.add_argument('--step', type=int, default=10, help='LSTM sekans adım boyutu')
    parser.add_argument('--no-lstm', action='store_true', help='LSTM değerlendirmesini atla')
    
    args = parser.parse_args()
    main(args)