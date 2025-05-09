# test_models.py

import pickle
import numpy as np
import torch
import torch.nn as nn
from xgboost import XGBClassifier
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from train_lstm import LSTMClassifier

def evaluate_classification(y_true, y_pred, model_name):
    """
    Girdi:
      y_true: Gerçek etiketler (0 veya 1)
      y_pred: Tahmin edilen etiketler (0 veya 1)
      model_name: Yazdırma için model adı (örn. "XGBoost" veya "LSTM")
    Çıkış: Metriklerin ekrana yazılması
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # ROC-AUC için y_pred yerine olasılık lazım normalde.
    # Burada sadece 0/1 sınıflandırma var. Sık kullanılan bir hile:
    #   roc_auc_score(y_true, y_pred) -> Bu "binary" tahminle
    # Ama esas doğrusu "predict_proba" skorları ile yapılır.
    # Sadece örnek olarak:
    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        roc_auc = np.nan
    
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"=== {model_name} METRICS ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["non-stationary (0)", "stationary (1)"]))
    print("======================================\n")

def main():
    # 1) Test verisini yükle
    with open("test_data.pkl", "rb") as f:
        test_data = pickle.load(f)
    
    X_test = test_data["X_features"]  # (N,3) [örn. mean, var, lag1Corr]
    y_test = test_data["y"]           # (N,)  [0 veya 1]
    
    seqs_test = test_data["seqs"]     # list of arrays, farklı uzunlukta diziler
    y_test_seq = test_data["y_seq"]   # (N,)  [0 veya 1]
    
    # 2) XGBoost modeli yükle
    xgb_model: XGBClassifier = joblib.load("xgb_model.pkl")
    
    # Tahmin (0/1)
    y_pred_xgb = xgb_model.predict(X_test)
    
    # Ek metrikler:
    evaluate_classification(y_test, y_pred_xgb, model_name="XGBoost")
    
    # 3) LSTM modeli yükle
    #    train_lstm.py'deki model tanımını aynı şekilde alıyoruz.
    lstm_model = LSTMClassifier(input_size=1, hidden_size=16)
    lstm_model.load_state_dict(torch.load("lstm_model.pt"))
    lstm_model.eval()
    
    y_pred_lstm = []
    
    with torch.no_grad():
        for i, seq_array in enumerate(seqs_test):
            label = y_test_seq[i]
            seq_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            out = lstm_model(seq_tensor)  # shape=(1,1)
            prob = torch.sigmoid(out).item()
            pred = 1 if prob >= 0.5 else 0
            y_pred_lstm.append(pred)
    
    y_pred_lstm = np.array(y_pred_lstm)
    
    # Ek metrikler:
    evaluate_classification(y_test_seq, y_pred_lstm, model_name="LSTM")

if __name__ == "__main__":
    main()
