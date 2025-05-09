# scripts/train_xgboost_model.py
import os
import numpy as np
import pandas as pd
import pickle
import argparse
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
from joblib import dump

# Modül yolunu ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import prepare_features_for_training
from utils.visualization import (
    plot_confusion_matrix, 
    plot_feature_importance, 
    plot_roc_curve, 
    plot_precision_recall_curve
)

def main(args):
    print("XGBoost modeli eğitiliyor...")
    
    # Veri klasörleri
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    raw_dir = os.path.join(data_dir, 'raw')
    processed_dir = os.path.join(data_dir, 'processed')
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'xgboost_model')
    
    os.makedirs(model_dir, exist_ok=True)
    
    # Zaman serilerini yükle
    print("Veriler yükleniyor...")
    all_series_df = pd.read_csv(os.path.join(raw_dir, 'all_series.csv'))
    
    # Etiketleri yükle
    seasonality_labels = pd.read_csv(os.path.join(processed_dir, 'seasonality_labels.csv'))
    breakpoint_labels = pd.read_csv(os.path.join(processed_dir, 'breakpoint_labels.csv'))
    
    # Eğitim/test indekslerini yükle
    train_indices = np.load(os.path.join(processed_dir, 'train_indices.npy'))
    test_indices = np.load(os.path.join(processed_dir, 'test_indices.npy'))
    
    # Tüm serileri bir sözlük olarak grupla
    series_dict = {}
    for series_id, group in all_series_df.groupby('series_id'):
        series_dict[series_id] = pd.Series(group['y'].values, index=pd.to_datetime(group['ds']))
    
    # Eğitim ve test serilerini ayır
    train_series = [series_dict[i] for i in train_indices if i in series_dict]
    test_series = [series_dict[i] for i in test_indices if i in series_dict]
    
    print(f"Eğitim seti boyutu: {len(train_series)}")
    print(f"Test seti boyutu: {len(test_series)}")
    
    # Özellik çıkarma
    print("Özellikler hazırlanıyor...")
    train_features = prepare_features_for_training(train_series)
    test_features = prepare_features_for_training(test_series)
    
    # Eğitim ve test etiketleri
    train_seasonality_labels = seasonality_labels[seasonality_labels['series_id'].isin(train_indices)]
    test_seasonality_labels = seasonality_labels[seasonality_labels['series_id'].isin(test_indices)]
    
    # Özellik ve hedef sütunları ayarla
    X_train = train_features.drop(['series_id'], axis=1)
    X_test = test_features.drop(['series_id'], axis=1)
    
    y_train_seasonal = train_seasonality_labels['is_seasonal'].values
    y_test_seasonal = test_seasonality_labels['is_seasonal'].values
    
    # Eksik değerleri doldur
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Ölçeklendirme
    if args.scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Ölçekleyiciyi kaydet
        dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    
    # Mevsimsellik tespiti için model
    print("Mevsimsellik tespit modeli eğitiliyor...")
    
    if args.grid_search:
        # Grid Search ile hiperparametre optimizasyonu
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        grid_search = GridSearchCV(
            xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train_seasonal)
        
        print(f"En iyi parametreler: {grid_search.best_params_}")
        seasonality_model = grid_search.best_estimator_
    else:
        # Varsayılan parametrelerle eğit
        seasonality_model = xgb.XGBClassifier(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=200,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        seasonality_model.fit(X_train, y_train_seasonal)
    
    # Modeli değerlendir
    y_pred_seasonal = seasonality_model.predict(X_test)
    y_pred_proba_seasonal = seasonality_model.predict_proba(X_test)[:, 1]
    
    # Metrikleri hesapla
    accuracy = accuracy_score(y_test_seasonal, y_pred_seasonal)
    precision = precision_score(y_test_seasonal, y_pred_seasonal)
    recall = recall_score(y_test_seasonal, y_pred_seasonal)
    f1 = f1_score(y_test_seasonal, y_pred_seasonal)
    auc = roc_auc_score(y_test_seasonal, y_pred_proba_seasonal)
    
    print("\nMevsimsellik Tespit Sonuçları:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Sonuçları kaydet
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }
    
    with open(os.path.join(model_dir, 'seasonality_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Modeli kaydet
    seasonality_model.save_model(os.path.join(model_dir, 'seasonality_model.json'))
    
    # Özellik önemini kaydet
    feature_importance = seasonality_model.feature_importances_
    feature_names = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else [f'feature_{i}' for i in range(X_train.shape[1])]
    
    if not isinstance(X_train, pd.DataFrame):
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        })
    else:
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        })
    
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    feature_importance_df.to_csv(os.path.join(model_dir, 'feature_importance.csv'), index=False)
    
    # Plot'ları oluştur
    print("Görselleştirmeler oluşturuluyor...")
    plot_dir = os.path.join(model_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Karışıklık matrisi
    plot_confusion_matrix(
        y_test_seasonal, 
        y_pred_seasonal, 
        classes=['Non-Seasonal', 'Seasonal'],
        title='Seasonality Detection Confusion Matrix',
        save_path=os.path.join(plot_dir, 'confusion_matrix.png')
    )
    
    # Özellik önemi grafiği
    plot_feature_importance(
        feature_names,
        feature_importance,
        title='Feature Importance for Seasonality Detection',
        save_path=os.path.join(plot_dir, 'feature_importance.png')
    )
    
    # ROC eğrisi
    plot_roc_curve(
        y_test_seasonal,
        y_pred_proba_seasonal,
        title='ROC Curve for Seasonality Detection',
        save_path=os.path.join(plot_dir, 'roc_curve.png')
    )
    
    # Hassasiyet-Duyarlılık eğrisi
    plot_precision_recall_curve(
        y_test_seasonal,
        y_pred_proba_seasonal,
        title='Precision-Recall Curve for Seasonality Detection',
        save_path=os.path.join(plot_dir, 'precision_recall_curve.png')
    )
    
    print("XGBoost modeli eğitimi tamamlandı!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='XGBoost modelini eğit')
    parser.add_argument('--scale', action='store_true', help='Özellikleri ölçeklendir')
    parser.add_argument('--grid-search', action='store_true', help='Grid Search ile hiperparametre optimizasyonu yap')
    
    args = parser.parse_args()
    main(args)