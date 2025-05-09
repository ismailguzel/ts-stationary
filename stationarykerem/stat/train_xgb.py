# train_xgb.py

import pickle
import numpy as np
from xgboost import XGBClassifier
import joblib

def main():
    # 1) Train verisini yükle
    with open("train_data.pkl", "rb") as f:
        train_data = pickle.load(f)
    X_train = train_data["X_features"]  # shape=(N,3)
    y_train = train_data["y"]           # shape=(N,)
    
    # 2) XGBClassifier
    model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    
    # 3) Kaydet
    joblib.dump(model, "xgb_model.pkl")
    print("XGBoost modeli eğitildi ve 'xgb_model.pkl' olarak kaydedildi.")

if __name__ == "__main__":
    main()
