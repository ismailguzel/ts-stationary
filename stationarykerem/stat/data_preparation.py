# data_preparation.py
import os
import numpy as np
import pandas as pd

def extract_features(series_1d):
    """
    Basit öznitelikler: mean, var, lag-1 correlation
    """
    mean_val = np.mean(series_1d)
    var_val = np.var(series_1d)
    if len(series_1d) < 2:
        lag1 = 0
    else:
        c = np.corrcoef(series_1d[:-1], series_1d[1:])[0,1]
        lag1 = 0 if np.isnan(c) else c
    return [mean_val, var_val, lag1]

def main():
    # 1) info dosyası
    info_df = pd.read_csv("infodatasets.csv")
    # Toplam 1000 satır: dataset_0.csv ... dataset_999.csv
    
    # 2) Ayırma (ilk 500 -> train, sonraki 500 -> test)
    train_info = info_df.iloc[:500].reset_index(drop=True)
    test_info  = info_df.iloc[500:].reset_index(drop=True)
    
    X_train_features = []
    y_train = []
    seqs_train = []  # LSTM için orijinal zaman serisi
    y_train_seq = []
    
    # 3) Train verisini oluştur
    for idx, row in train_info.iterrows():
        filename = row["filename"]              # ör: dataset_23.csv
        column_types_str = row["column_types"]  # ör: "col_0=stationary; col_1=nonstationary; ..."
        
        col_type_pairs = column_types_str.split(";")  # ["col_0=stationary", " col_1=nonstationary", ...]
        col_type_pairs = [x.strip() for x in col_type_pairs]
        
        csv_path = os.path.join("datasets", filename)
        df = pd.read_csv(csv_path)  # shape=(num_rows, num_cols)
        
        for pair in col_type_pairs:
            # ör: "col_0=stationary"
            col_name, label_str = pair.split("=")
            col_name = col_name.strip()
            label_str = label_str.strip()  # "stationary" veya "nonstationary"
            
            # stationary=1, nonstationary=0
            label = 1 if label_str == "stationary" else 0
            
            # 4) İlgili kolonu al
            series_1d = df[col_name].values  # shape=(num_rows,)
            
            # 4a) Feature-based (XGBoost)
            feats = extract_features(series_1d)
            X_train_features.append(feats)
            y_train.append(label)
            
            # 4b) Sequence-based (LSTM)
            seqs_train.append(series_1d)
            y_train_seq.append(label)
    
    # 5) Aynı işlemi test için
    X_test_features = []
    y_test = []
    seqs_test = []
    y_test_seq = []
    
    for idx, row in test_info.iterrows():
        filename = row["filename"]
        column_types_str = row["column_types"]
        
        col_type_pairs = column_types_str.split(";")
        col_type_pairs = [x.strip() for x in col_type_pairs]
        
        csv_path = os.path.join("datasets", filename)
        df = pd.read_csv(csv_path)
        
        for pair in col_type_pairs:
            col_name, label_str = pair.split("=")
            col_name = col_name.strip()
            label_str = label_str.strip()
            label = 1 if label_str == "stationary" else 0
            
            series_1d = df[col_name].values
            
            # Feature-based
            feats = extract_features(series_1d)
            X_test_features.append(feats)
            y_test.append(label)
            
            # Sequence-based
            seqs_test.append(series_1d)
            y_test_seq.append(label)
    
    # 6) NumPy array'e dönüştür
    X_train_features = np.array(X_train_features, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    X_test_features = np.array(X_test_features, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int64)
    
    # LSTM'de diziler farklı uzunlukta olabilir (bazı CSV 50 satır, bazı 300 satır).
    # Onları liste olarak saklayacağız veya "padding" vb. yapabiliriz.
    # Burada basitçe Python list olarak tutalım:
    seqs_train = [np.array(seq, dtype=np.float32) for seq in seqs_train]
    seqs_test  = [np.array(seq, dtype=np.float32) for seq in seqs_test]
    
    y_train_seq = np.array(y_train_seq, dtype=np.int64)
    y_test_seq  = np.array(y_test_seq, dtype=np.int64)
    
    # 7) Kaydet (pickle veya numpy)
    import pickle
    with open("train_data.pkl", "wb") as f:
        pickle.dump({
            "X_features": X_train_features,  # shape=(N,3)
            "y": y_train,                    # shape=(N,)
            "seqs": seqs_train,             # list of arrays
            "y_seq": y_train_seq
        }, f)
    
    with open("test_data.pkl", "wb") as f:
        pickle.dump({
            "X_features": X_test_features,
            "y": y_test,
            "seqs": seqs_test,
            "y_seq": y_test_seq
        }, f)
    
    print("Train ve test verileri hazır, 'train_data.pkl' ve 'test_data.pkl' kaydedildi.")

if __name__ == "__main__":
    main()
