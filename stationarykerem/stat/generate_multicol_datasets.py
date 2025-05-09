import os
import numpy as np
import pandas as pd

def generate_stationary_series(length=100, phi=0.5):
    """
    AR(1) modeli: X_t = phi * X_{t-1} + e_t
    |phi| < 1 => stationer olma ihtimali yüksek
    """
    series = np.zeros(length)
    series[0] = np.random.normal(0,1)
    for t in range(1, length):
        e = np.random.normal(0,1)
        series[t] = phi*series[t-1] + e
    return series

def generate_nonstationary_series(length=100):
    """
    Basit random walk: X_t = X_{t-1} + e_t
    => Non-stationary
    """
    series = np.zeros(length)
    series[0] = np.random.normal(0,1)
    for t in range(1, length):
        e = np.random.normal(0,1)
        series[t] = series[t-1] + e
    return series

def main():
    # 1) Kaç tane dataset dosyası üretilecek?
    num_datasets = 1000
    
    # 2) Rastgele seçim aralıkları
    possible_num_cols = [5, 10, 20]        # sütun sayısı seçenekleri
    possible_num_rows = [50, 100, 200, 300]  # satır sayısı seçenekleri
    
    # 3) Çıktı klasörü
    output_dir = "datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    # 4) Bilgi listesini tutacağımız yapı
    info_records = []  # her dataset için bir dict
    
    for i in range(num_datasets):
        # 4.1 Rastgele sütun ve satır seç
        num_cols = np.random.choice(possible_num_cols)
        num_rows = np.random.choice(possible_num_rows)
        
        # 4.2 Hangi sütunların stationer, hangi sütunların non-stationer olacağına karar ver
        #    Örneğin tamamen rastgele atayabiliriz (her bir sütunun %50 ihtimal stationer).
        col_types = []
        for _ in range(num_cols):
            is_stationary = np.random.choice([True, False])  # True => stationer, False => non-stationer
            col_types.append(is_stationary)
        
        # 4.3 DataFrame oluşturmak için verileri üret
        data = {}
        for col_idx, is_st in enumerate(col_types):
            col_name = f"col_{col_idx}"
            if is_st:
                # Rastgele phi seçelim
                phi = np.random.uniform(-0.8, 0.8)
                series = generate_stationary_series(length=num_rows, phi=phi)
            else:
                series = generate_nonstationary_series(length=num_rows)
            
            data[col_name] = series
        
        df = pd.DataFrame(data)
        
        # 4.4 Dosya ismi
        csv_filename = f"dataset_{i}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        
        # 4.5 CSV olarak kaydet
        df.to_csv(csv_path, index=False)
        
        # 4.6 Info kaydı
        # col_types listesi True/False => "stationary"/"nonstationary" dizgesi
        column_info = []
        for idx, st in enumerate(col_types):
            col_name = f"col_{idx}"
            label = "stationary" if st else "nonstationary"
            column_info.append(f"{col_name}={label}")
        
        column_info_str = "; ".join(column_info)
        
        info_dict = {
            "filename": csv_filename,
            "num_rows": num_rows,
            "num_cols": num_cols,
            "column_types": column_info_str  # "col_0=stationary; col_1=nonstationary; ..."
        }
        info_records.append(info_dict)
    
    # 5) info_records'u bir DataFrame yapıp infodatasets.csv ye kaydet
    info_df = pd.DataFrame(info_records)
    info_df.to_csv("infodatasets.csv", index=False)
    
    print(f"{num_datasets} adet dataset oluşturuldu, 'infodatasets.csv' dosyasına bilgiler kaydedildi.")

if __name__ == "__main__":
    main()
