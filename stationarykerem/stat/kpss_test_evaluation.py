# kpss_test_evaluation.py

import os
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as st  # KPSS
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

def kpss_stationary_decision(series_1d, alpha=0.05):
    """
    Uygulama:
      KPSS testi (statsmodels.tsa.stattools.kpss) -> (stat, p_value, used_lags, critical_values)
    Hipotez:
      - Null Hypothesis (H0): Seri stationary (trend-stationary veya level-stationary)
      - p_value < alpha => H0 reddedilir => seri non-stationary
      - p_value >= alpha => H0 kabul edilir => seri stationary

    Dönüş:
      1 => "stationary"
      0 => "non-stationary"
    """
    # KPSS bazen kısa serilerde veya edge case'lerde uyarı/exception verebilir.
    # Yakalamak için try-except kullanalım.
    try:
        kpss_stat, p_value, lags, crit = st.kpss(series_1d, regression='c', nlags="auto")
    except:
        # Herhangi bir hata, default "non-stationary" diyelim (veya 0.5 vs.)
        return 0
    
    if p_value < alpha:
        # Null reddedildi => non-stationary
        return 0
    else:
        # Null reddedilmedi => stationary
        return 1

def evaluate_kpss_on_testdata(infodatasets_path="infodatasets.csv",
                              datasets_dir="datasets",
                              alpha=0.05):
    """
    1) infodatasets.csv yükle
    2) Son 500 satırı al (test set)
    3) Her dosyanın her kolonunda KPSS uygula, tahmini (0/1) elde et
    4) Gerçek label (0/1) ile kıyasla, metrikleri raporla
    """
    info_df = pd.read_csv(infodatasets_path)
    
    # son 500 satırı test olarak alıyoruz
    test_info = info_df.iloc[500:].reset_index(drop=True)
    
    y_true = []
    y_pred = []
    
    for idx, row in test_info.iterrows():
        filename = row["filename"]              # örn. "dataset_600.csv"
        column_types_str = row["column_types"]  # "col_0=stationary; col_1=nonstationary; ..."
        
        col_type_pairs = column_types_str.split(";")
        col_type_pairs = [x.strip() for x in col_type_pairs]
        
        # CSV oku
        csv_path = os.path.join(datasets_dir, filename)
        df = pd.read_csv(csv_path)
        
        for pair in col_type_pairs:
            # "col_0=stationary"
            col_name, label_str = pair.split("=")
            col_name = col_name.strip()
            label_str = label_str.strip()  # "stationary" veya "nonstationary"
            
            # Ground truth: stationary=1, nonstationary=0
            true_label = 1 if label_str=="stationary" else 0
            
            # Kolonu al
            series_1d = df[col_name].values
            
            # KPSS testiyle karar ver
            pred_label = kpss_stationary_decision(series_1d, alpha=alpha)
            
            y_true.append(true_label)
            y_pred.append(pred_label)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Metirk hesaplama
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # AUC -> Yalnızca 0/1 tahminle kabaca hesaplanabilir.
    # Aslında "p_value" veya benzer continuous skor ile hesaplamak daha doğru.
    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        roc_auc = np.nan
    
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"=== KPSS Test (alpha={alpha}) RESULTS ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")
    
    print("Confusion Matrix:")
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["non-stationary (0)", "stationary (1)"]))
    print("======================================")

def main():
    evaluate_kpss_on_testdata(infodatasets_path="infodatasets.csv",
                              datasets_dir="datasets",
                              alpha=0.05)

if __name__ == "__main__":
    main()
