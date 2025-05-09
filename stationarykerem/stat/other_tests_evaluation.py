# other_tests_evaluation.py

import os
import sys
import numpy as np
import pandas as pd

import statsmodels.tsa.stattools as sm_stattools  # adfuller
# PP (Phillips-Perron) testi için 'arch' kütüphanesi lazım:
# pip install arch
from arch.unitroot import PhillipsPerron

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_auc_score
)

def adf_test_decision(series_1d, alpha=0.05):
    """
    ADF Test:
      H0: seri non-stationary (birim kök var)
      p_value < alpha => H0 reddedilir => seri stationary => return 1
      aksi => return 0
    """
    try:
        # adfuller çıktısı: (stat, pvalue, usedlag, nobs, critical values, icbest)
        result = sm_stattools.adfuller(series_1d, regression='c', autolag='AIC')
        p_value = result[1]
    except:
        # hata durumunda non-stationary diyelim
        return 0
    
    if p_value < alpha:
        return 1  # stationary
    else:
        return 0  # non-stationary

def pp_test_decision(series_1d, alpha=0.05):
    """
    Phillips-Perron Test:
      H0: seri non-stationary (birim kök)
      p_value < alpha => seri stationary => return 1
      aksi => return 0
    """
    try:
        # arch.unitroot.PhillipsPerron
        pp = PhillipsPerron(series_1d, trend='c')
        pp_stat = pp.stat
        p_value = pp.pvalue
    except:
        return 0
    
    if p_value < alpha:
        return 1
    else:
        return 0

def evaluate_test(testname, decision_func, alpha=0.05,
                  infodatasets_path="infodatasets.csv",
                  datasets_dir="datasets"):
    """
    testname: "ADF" / "PP"
    decision_func: adf_test_decision veya pp_test_decision
    alpha: sign. level
    infodatasets_path: 'infodatasets.csv'
    datasets_dir: 'datasets/'
    """
    info_df = pd.read_csv(infodatasets_path)
    # son 500 satır => test
    test_info = info_df.iloc[500:].reset_index(drop=True)
    
    y_true = []
    y_pred = []
    
    for idx, row in test_info.iterrows():
        filename = row["filename"]
        col_types_str = row["column_types"]
        
        csv_path = os.path.join(datasets_dir, filename)
        df = pd.read_csv(csv_path)
        
        col_pairs = [x.strip() for x in col_types_str.split(";")]
        for pair in col_pairs:
            # "col_0=stationary"
            col_name, label_str = pair.split("=")
            col_name = col_name.strip()
            label_str = label_str.strip()
            
            true_label = 1 if label_str=="stationary" else 0
            
            series_1d = df[col_name].values
            pred_label = decision_func(series_1d, alpha=alpha)
            
            y_true.append(true_label)
            y_pred.append(pred_label)
    
    # NumPy array
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        roc_auc = np.nan
    
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"=== {testname} Test (alpha={alpha}) RESULTS ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=["non-stationary (0)", "stationary (1)"]
    ))
    print("======================================")

def main():
    # Kullanıcı argümanı ile test seçimi (ADF veya PP) yapılabilir
    # Örn: python other_tests_evaluation.py ADF
    # veya: python other_tests_evaluation.py PP
    # Varsayılan: ADF
    args = sys.argv
    if len(args) > 1:
        testname = args[1].upper()
    else:
        testname = "ADF"
    
    if testname == "ADF":
        evaluate_test("ADF", adf_test_decision, alpha=0.05)
    elif testname == "PP":
        evaluate_test("PP", pp_test_decision, alpha=0.05)
    else:
        print("Geçersiz test adı. Lütfen 'ADF' veya 'PP' giriniz.")

if __name__ == "__main__":
    main()
