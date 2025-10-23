#########DATA FORMAT##########

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from generator import TimeSeriesGenerator

# === Metadata function ===
def create_metadata_record(
    series_id,
    length,
    label,
    is_stationary=1,
    base_series=0,
    order=0,
    base_coefs=0,
    linear_trend_up=0,
    linear_trend_down=0,
    quadratic_trend=0,
    cubic_trend=0,
    exponential_trend=0,
    damped_trend=0,
    stochastic_trend=0,
    difference=0,
    seasonality=0,
    multiple_seasonality=0,
    seasonality_from_base=0,
    seasonality_frequency=0,
    seasonal_difference=0,
    volatility=0,
    mean_shift_increase=0,
    mean_shift_decrease=0,
    multi_mean_shift=0,
    variance_shift_increase=0,
    variance_shift_decrease=0,
    multi_variance_shift=0,
    trend_shift_slope=0,
    trend_shift_intercept=0,
    multi_trend_shift=0,
    point_anomaly=0,
    collective_anomaly=0,
    contextual_anomaly=0,
    multi_point_anomaly=0,
    multi_collective_anomaly=0,
    multi_contextual_anomaly=0,
    location_point=0,
    location_collective=0,
    location_mean_shift=0,
    location_variance_shift=0,
    location_trend_shift=0,
    location_contextual=0,
    location_point_pts=0,
    location_collective_pts=0,
    location_mean_pts=0,
    location_contextual_pts=0,
    location_variance_pts=0,
    location_trend_pts=0,
    ):
    # --- DEĞİŞEN KISIM ---
    # Sözlüğü, fonksiyonun aldığı gerçek parametre değerlerini kullanarak oluştur
    record = {
        # General
        "series_id": series_id,
        "length": length,
        "label": label,
        "is_stationary": is_stationary,

        # Base stochastic process
        "base_series": base_series,
        "order": order,
        "base_coefs": base_coefs,

        # Deterministic trends
        "linear_trend_up": linear_trend_up,
        "linear_trend_down": linear_trend_down,
        "quadratic_trend": quadratic_trend,
        "cubic_trend": cubic_trend,
        "exponential_trend": exponential_trend,
        "damped_trend": damped_trend,

        # Stochastic trend
        "stochastic_trend": stochastic_trend,
        "difference": difference,

        # Seasonality
        "seasonality": seasonality,
        "multiple_seasonality": multiple_seasonality,
        "seasonality_from_base": seasonality_from_base,
        "seasonality_frequency": seasonality_frequency,
        "seasonal_difference": seasonal_difference,

        # Volatility
        "volatility": volatility,

        # Mean shift
        "mean_shift_increase": mean_shift_increase,
        "mean_shift_decrease": mean_shift_decrease,
        "multi_mean_shift": multi_mean_shift,

        # Variance shift
        "variance_shift_increase": variance_shift_increase,
        "variance_shift_decrease": variance_shift_decrease,
        "multi_variance_shift": multi_variance_shift,

        # Trend shift
        "trend_shift_slope": trend_shift_slope,
        "trend_shift_intercept": trend_shift_intercept,
        "multi_trend_shift": multi_trend_shift,

        # Anomaly types
        "point_anomaly": point_anomaly,
        "collective_anomaly": collective_anomaly,
        "contextual_anomaly": contextual_anomaly,
        "multi_point_anomaly": multi_point_anomaly,
        "multi_collective_anomaly": multi_collective_anomaly,
        "multi_contextual_anomaly": multi_contextual_anomaly,

        # Anomaly locations
        "location_point": location_point,
        "location_collective": location_collective,
        "location_mean_shift": location_mean_shift,
        "location_variance_shift": location_variance_shift,
        "location_trend_shift": location_trend_shift,
        "location_contextual": location_contextual,

        # Anomaly location points/magnitude info
        "location_point_pts": location_point_pts,
        "location_collective_pts": location_collective_pts,
        "location_mean_pts": location_mean_pts,
        "location_contextual_pts": location_contextual_pts,
        "location_variance_pts": location_variance_pts,
        "location_trend_pts": location_trend_pts
    }
    return record

def make_json_serializable(obj):
    if isinstance(obj, (np.integer, np.int_, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)
    return str(obj)  

def get_metadata_columns_defaults():
    # 'is_stationary=1' eklendi
    dummy = create_metadata_record(series_id=0, length=0, label="", is_stationary=1)
    return list(dummy.keys()), dummy


# Bu fonksiyon 'is_stationary' sütununu otomatik olarak ekleyecektir
def attach_metadata_columns_to_df(df, metadata_record):
    metadata_cols, default_record = get_metadata_columns_defaults()

    for col in metadata_cols:
        val = metadata_record.get(col, default_record[col])

        if isinstance(val, (int, float, str)):
            df[col] = val
        else:
            df[col] = str(val)

    df['label'] = metadata_record['label']
    
    core_cols = ['series_id', 'time', 'data', 'label']
    meta_cols = [col for col in metadata_cols if col not in core_cols and col in df.columns]
    final_cols_order = ['series_id', 'time', 'data'] + meta_cols + ['label']
    
    final_cols_in_df = [col for col in final_cols_order if col in df.columns]
    
    df = df[final_cols_in_df]
    return df


# --- Helper: Boş klasörü silmek için ---
def _save_and_cleanup(all_dfs, folder, count, label):
    """
    Birleştirilmiş DataFrame'i Parquet olarak kaydeder ve
    artık gereksiz olan boş alt klasörü siler.
    """
    if not all_dfs:
        print(f"No data generated for {folder}")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    category_label = os.path.basename(folder) 
    parent_folder = os.path.dirname(folder)   
    
    output_filename = f"{category_label}.parquet"
    output_path = os.path.join(parent_folder, output_filename)

    combined_df.to_parquet(output_path, index=False)

    try:
        os.rmdir(folder)
    except OSError as e:
        print(f"Warning: Could not remove empty directory {folder}: {e}")

    print(f"{count} '{label}' series saved in ONE file: '{output_path}'")
    

######### STATIONARY SERIES #########
# NOT: Buradan sonraki tüm fonksiyonlar 'is_stationary' bayrağını okuyacak
# ve 'stationary' sütununu silecek.

#### WHITE NOISE #####
def generate_wn_dataset(
    folder,
    count=20,
    length_range=(50, 100)
):
    os.makedirs(folder, exist_ok=True)
    global all_dfs
    all_dfs = [] 
    label = "" 

    if length_range == (50,100):
        l = "short"
    elif length_range == (300,500):
        l = "medium"
    else:
        l = "long"

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = TimeSeriesGenerator(length=length)
        df,info = ts.generate_stationary_base_series('white_noise')

        label = f"white_noise_{l}"
        series_id = i + 1 

        # --- YENİ LOGİK ---
        is_stat_flag = int(df['stationary'].iloc[0]) # Değeri al (1)
        df = df.drop(columns=['stationary'])         # Gereksiz sütunu sil
        # --- BİTTİ ---

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag, # <--- YENİ EKLENDİ
            base_series="white_noise",
        )
        
        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    _save_and_cleanup(all_dfs, folder, count, label)
    

#### AR #####
def generate_ar_dataset(folder, count=20, length_range=(50, 100)):
    os.makedirs(folder, exist_ok=True)
    global all_dfs
    all_dfs = []
    label = ""

    if length_range == (50,100):
        l = "short"
    elif length_range == (300,500):
        l = "medium"
    else:
        l = "long"

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = TimeSeriesGenerator(length=length)
        df, info = ts.generate_stationary_base_series('ar')
        
        base_coefs=f"({info['ar_coefs']})"
        base_order=f"({info['ar_order']})"
        label = f"ar_{l}"
        series_id = i + 1

        # --- YENİ LOGİK ---
        is_stat_flag = int(df['stationary'].iloc[0]) # Değeri al (1)
        df = df.drop(columns=['stationary'])         # Gereksiz sütunu sil
        # --- BİTTİ ---

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag, # <--- YENİ EKLENDİ
            base_series="ar",
            base_coefs=base_coefs,
            order=base_order
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    _save_and_cleanup(all_dfs, folder, count, label)



#### MA #####
def generate_ma_dataset(folder, count=20, length_range=(50, 100)):
    os.makedirs(folder, exist_ok=True)
    global all_dfs
    all_dfs = []
    label = ""

    if length_range == (50,100):
        l = "short"
    elif length_range == (300,500):
        l = "medium"
    else:
        l = "long"

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = TimeSeriesGenerator(length=length)
        df,info = ts.generate_stationary_base_series('ma')

        base_coefs=f"({info['ma_coefs']})"
        base_order = f"({info['ma_order']})"
        label = f"ma_{l}"
        series_id = i + 1

        # --- YENİ LOGİK ---
        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])
        # --- BİTTİ ---

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag, # <--- YENİ EKLENDİ
            base_series="ma",
            base_coefs=base_coefs,
            order=base_order
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    _save_and_cleanup(all_dfs, folder, count, label)


#### ARMA #####
def generate_arma_dataset(folder, count=20, length_range=(50, 100)):
    os.makedirs(folder, exist_ok=True)
    global all_dfs
    all_dfs = []
    label = ""

    if length_range == (50,100):
        l = "short"
    elif length_range == (300,500):
        l = "medium"
    else:
        l = "long"

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = TimeSeriesGenerator(length=length)
        df,info = ts.generate_stationary_base_series('arma')

        base_coefs = f"({info['ar_coefs']},{info['ma_coefs']})"
        base_order = f"({info['ar_order']},{info['ma_order']})"
        label = f"arma_{l}"
        series_id = i + 1 

        # --- YENİ LOGİK ---
        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])
        # --- BİTTİ ---

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag, # <--- YENİ EKLENDİ
            base_series="arma",
            base_coefs=base_coefs,
            order=base_order
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    _save_and_cleanup(all_dfs, folder, count, label)



#########DETERMINISTIC TREND LINEAR############
def generate_linear_trend_dataset(
    folder,
    kind='ar',
    count=5,
    length_range=(300, 500),
    sign=1
):
    os.makedirs(folder, exist_ok=True)
    global all_dfs
    all_dfs = []
    label = ""

    trend_label = "up" if sign == 1 else "down"
    if length_range == (50,100):
        l = "short"
    elif length_range == (300,500):
        l = "medium"
    else:
        l = "long"

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = TimeSeriesGenerator(length=length)

        if kind == 'ar':
            df, info = ts.generate_stationary_base_series('ar')
            base_coefs = f"({info['ar_coefs']})"
            base_order =  f"({info['ar_order']})"
        elif kind == 'ma':
            df,info = ts.generate_stationary_base_series('ma')
            base_coefs = f"({info['ma_coefs']})"
            base_order = f"({info['ma_order']})"
        elif kind == 'arma':
            df,info = ts.generate_stationary_base_series('arma')
            base_coefs = f"({info['ar_coefs']},{info['ma_coefs']})"
            base_order = f"({info['ar_order']},{info['ma_order']})"
        elif kind == 'white_noise':
            df,info = ts.generate_stationary_base_series('white_noise')
            base_coefs = 0
            base_order = 0
        else:
            raise ValueError(f"Unknown kind: {kind}")

        # Bu fonksiyon df['stationary']'yi 0 yapacak
        df,info_trend = ts.generate_deterministic_trend_linear(df, sign=sign)

        label = f"{kind}_linear_trend_{l}_{trend_label}"
        series_id = i + 1

        # --- YENİ LOGİK ---
        is_stat_flag = int(df['stationary'].iloc[0]) # Değeri al (0)
        df = df.drop(columns=['stationary'])         # Sütunu sil
        # --- BİTTİ ---

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag, # <--- YENİ EKLENDİ
            base_series=kind,
            base_coefs=base_coefs,
            order=base_order,
            linear_trend_up=1 if sign == 1 else 0,
            linear_trend_down=1 if sign == -1 else 0
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    _save_and_cleanup(all_dfs, folder, count, label)


##########DETERMINISTIC TREND QUADRATIC###########
def generate_quadratic_trend_dataset(
    folder,
    kind='ar',
    count=5,
    length_range=(300, 500),
    sign=1
):
    os.makedirs(folder, exist_ok=True)
    global all_dfs
    all_dfs = []
    label = ""

    if length_range == (50,100):
        l = "short"
    # ... (diğer l etiketlemeleri) ...
    else:
        l = "long"

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = TimeSeriesGenerator(length=length)
        
        # ... (base series üretimi) ...
        if kind == 'ar':
            df, info = ts.generate_stationary_base_series('ar')
            base_coefs = f"({info['ar_coefs']})"
            base_order = f"({info['ar_order']})"
        elif kind == 'ma':
            df,info = ts.generate_stationary_base_series('ma')
            base_coefs = f"({info['ma_coefs']})"
            base_order = f"({info['ma_order']})"
        elif kind == 'arma':
            df,info = ts.generate_stationary_base_series('arma')
            base_coefs = f"({info['ar_coefs']},{info['ma_coefs']})"
            base_order = f"({info['ar_order']},{info['ma_order']})"
        elif kind == 'white_noise':
            df,info = ts.generate_stationary_base_series('white_noise')
            base_coefs = 0
            base_order = 0
        else:
            raise ValueError(f"Unknown kind: {kind}")

        df,info_trend = ts.generate_deterministic_trend_quadratic(df, sign=sign, location = "center")

        label = f"{kind}_quadratic_trend_{l}"
        series_id = i + 1

        # --- YENİ LOGİK ---
        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])
        # --- BİTTİ ---

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag, # <--- YENİ EKLENDİ
            base_series=kind,
            base_coefs=base_coefs,
            order=base_order,
            quadratic_trend=1
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    _save_and_cleanup(all_dfs, folder, count, label)



#############DETERMINISTIC TREND CUBIC#################
def generate_cubic_trend_dataset(
    folder,
    kind='ar',
    count=5,
    length_range=(300, 500),
    sign=1
):
    os.makedirs(folder, exist_ok=True)
    global all_dfs
    all_dfs = []
    label = ""
    
    # ... (l etiketlemesi) ...
    if length_range == (50,100):
        l = "short"
    elif length_range == (300,500):
        l = "medium"
    else:
        l = "long"

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = TimeSeriesGenerator(length=length)

        # ... (base series üretimi) ...
        if kind == 'ar':
            df, info = ts.generate_stationary_base_series('ar')
            base_coefs = f"({info['ar_coefs']})"
            base_order = f"({info['ar_order']})"
        elif kind == 'ma':
            df,info = ts.generate_stationary_base_series('ma')
            base_coefs = f"({info['ma_coefs']})"
            base_order = f"({info['ma_order']})"
        elif kind == 'arma':
            df,info = ts.generate_stationary_base_series('arma')
            base_coefs = f"({info['ar_coefs']},{info['ma_coefs']})"
            base_order = f"({info['ar_order']},{info['ma_order']})"
        elif kind == 'white_noise':
            df,info = ts.generate_stationary_base_series('white_noise')
            base_coefs = 0
            base_order = 0
        else:
            raise ValueError(f"Unknown kind: {kind}")

        df,info_trend = ts.generate_deterministic_trend_cubic(df, sign=sign, location = "center")

        label = f"{kind}_cubic_trend_{l}"
        series_id = i + 1

        # --- YENİ LOGİK ---
        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])
        # --- BİTTİ ---

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag, # <--- YENİ EKLENDİ
            base_series=kind,
            base_coefs=base_coefs,
            order=base_order,
            cubic_trend=1
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    _save_and_cleanup(all_dfs, folder, count, label)


#########DETERMINISTIC TREND EXPONENTIAL################
def generate_exponential_trend_dataset(
    folder,
    kind='ar',
    count=5,
    length_range=(300, 500),
    sign=1
):
    os.makedirs(folder, exist_ok=True)
    global all_dfs
    all_dfs = []
    label = ""
    # ... (l etiketlemesi) ...
    if length_range == (50,100):
        l = "short"
    elif length_range == (300,500):
        l = "medium"
    else:
        l = "long"

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = TimeSeriesGenerator(length=length)

        # ... (base series üretimi) ...
        if kind == 'ar':
            df, info = ts.generate_stationary_base_series('ar')
            base_coefs = f"({info['ar_coefs']})"
            base_order = f"({info['ar_order']})"
        elif kind == 'ma':
            df,info = ts.generate_stationary_base_series('ma')
            base_coefs = f"({info['ma_coefs']})"
            base_order = f"({info['ma_order']})"
        elif kind == 'arma':
            df,info = ts.generate_stationary_base_series('arma')
            base_coefs = f"({info['ar_coefs']},{info['ma_coefs']})"
            base_order = f"({info['ar_order']},{info['ma_order']})"
        elif kind == 'white_noise':
            df,info = ts.generate_stationary_base_series('white_noise')
            base_coefs = 0
            base_order = 0
        else:
            raise ValueError(f"Unknown kind: {kind}")


        df,info_trend = ts.generate_deterministic_trend_exponential(df, sign=sign)

        label = f"{kind}_exponential_trend_{l}"
        series_id = i + 1

        # --- YENİ LOGİK ---
        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])
        # --- BİTTİ ---

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag, # <--- YENİ EKLENDİ
            base_series=kind,
            base_coefs=base_coefs,
            order=base_order,
            exponential_trend=1
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    _save_and_cleanup(all_dfs, folder, count, label)



##########DETERMINISTIC TREND - DAMPED###############
def generate_damped_trend_dataset(
    folder,
    kind='ar',
    count=5,
    length_range=(300, 500),
    sign=1
):
    os.makedirs(folder, exist_ok=True)
    global all_dfs
    all_dfs = []
    label = ""
    # ... (l etiketlemesi) ...
    if length_range == (50,100):
        l = "short"
    elif length_range == (300,500):
        l = "medium"
    else:
        l = "long"

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = TimeSeriesGenerator(length=length)

        # ... (base series üretimi) ...
        if kind == 'ar':
            df,info = ts.generate_stationary_base_series('ar')
            base_coefs = f"({info['ar_coefs']})"
            base_order = f"({info['ar_order']})"
        elif kind == 'ma':
            df,info = ts.generate_stationary_base_series('ma')
            base_coefs = f"({info['ma_coefs']})"
            base_order = f"({info['ma_order']})"
        elif kind == 'arma':
            df,info = ts.generate_stationary_base_series('arma')
            base_coefs = f"({info['ar_coefs']},{info['ma_coefs']})"
            base_order = f"({info['ar_order']},{info['ma_order']})"
        elif kind == 'white_noise':
            df,info = ts.generate_stationary_base_series('white_noise')
            base_coefs = 0
            base_order = 0
        else:
            raise ValueError(f"Unknown kind: {kind}")


        df,info_trend = ts.generate_deterministic_trend_damped(df, sign=sign)

        label = f"{kind}_damped_trend_{l}"
        series_id = i + 1

        # --- YENİ LOGİK ---
        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])
        # --- BİTTİ ---

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag, # <--- YENİ EKLENDİ
            base_series=kind,
            base_coefs=base_coefs,
            order=base_order,
            damped_trend=1
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    _save_and_cleanup(all_dfs, folder, count, label)


########POINT ANOMALY#########
def generate_point_anomaly_dataset(
    folder,
    kind='ar',
    count=5,
    length_range=(300, 500),
    anomaly_type='single',
    location="middle"
):
    os.makedirs(folder, exist_ok=True)
    global all_dfs
    all_dfs = []
    label = ""
    # ... (l etiketlemesi) ...
    if length_range == (50,100):
        l = "short"
    elif length_range == (300,500):
        l = "medium"
    else:
        l = "long"

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = TimeSeriesGenerator(length=length)

        # ... (base series üretimi) ...
        if kind == 'ar':
            df, info = ts.generate_stationary_base_series('ar')
            base_coefs = f"({info['ar_coefs']})"
            base_order = f"({info['ar_order']})"
        elif kind == 'ma':
            df,info = ts.generate_stationary_base_series('ma')
            base_coefs = f"({info['ma_coefs']})"
            base_order = f"({info['ma_order']})"
        elif kind == 'arma':
            df,info = ts.generate_stationary_base_series('arma')
            base_coefs = f"({info['ar_coefs']},{info['ma_coefs']})"
            base_order = f"({info['ar_order']},{info['ma_order']})"
        elif kind == 'white_noise':
            df,info = ts.generate_stationary_base_series('white_noise')
            base_coefs = 0
            base_order = 0
        else:
            raise ValueError(f"Unknown kind: {kind}")


        if anomaly_type == 'single':
            loc = location if location else np.random.choice(['beginning', 'middle', 'end'])
            df, info_anom = ts.generate_point_anomaly(df, location=loc) # Bu df['stationary']'yi 0 yapar
            label = f"{kind}_single_point_anomaly_{loc}_{l}"
            point_anomaly=1
            multi_point_anomaly = 0
            location_meta = f"{info_anom['location']}"
            anomaly_indices = f"{info_anom['anomaly_indices']}"

        elif anomaly_type == 'multiple':
            df,info_anom = ts.generate_point_anomalies(df) # Bu df['stationary']'yi 0 yapar
            label = f"{kind}_multiple_point_anomalies_{l}"
            multi_point_anomaly = 1
            point_anomaly = 0
            location_meta = "multiple"
            anomaly_indices = f"{info_anom['anomaly_indices']}"
        else:
            raise ValueError("Invalid anomaly_type. Must be 'single' or 'multiple'.")

        series_id = i + 1

        # --- YENİ LOGİK ---
        is_stat_flag = int(df['stationary'].iloc[0]) # Değeri al (0)
        df = df.drop(columns=['stationary'])
        # --- BİTTİ ---

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag, # <--- YENİ EKLENDİ
            base_series=kind,
            base_coefs=base_coefs,
            order=base_order,
            point_anomaly=point_anomaly,
            multi_point_anomaly=multi_point_anomaly,
            location_point=location_meta,
            location_point_pts=anomaly_indices
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    _save_and_cleanup(all_dfs, folder, count, label)


########COLLECTIVE ANOMALY########
def generate_collective_anomaly_dataset(
    folder,
    kind='ar',
    count=5,
    length_range=(300, 500),
    anomaly_type='single',          
    location="middle",              
    num_anomalies=2,
    scale_factor=1
):
    os.makedirs(folder, exist_ok=True)
    global all_dfs
    all_dfs = []
    label = ""
    # ... (l etiketlemesi) ...
    if length_range == (50, 100):
        l = "short"
    elif length_range == (300, 500):
        l = "medium"
    else:
        l = "long"

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = TimeSeriesGenerator(length=length)

        # ... (base series üretimi) ...
        if kind == 'ar':
            df, info = ts.generate_stationary_base_series('ar')
            base_coefs = f"({info['ar_coefs']})"
            base_order = f"({info['ar_order']})"
        elif kind == 'ma':
            df,info = ts.generate_stationary_base_series('ma')
            base_coefs = f"({info['ma_coefs']})"
            base_order = f"({info['ma_order']})"
        elif kind == 'arma':
            df,info = ts.generate_stationary_base_series('arma')
            base_coefs = f"({info['ar_coefs']},{info['ma_coefs']})"
            base_order = f"({info['ar_order']},{info['ma_order']})"
        elif kind == 'white_noise':
            df,info = ts.generate_stationary_base_series('white_noise')
            base_coefs = 0
            base_order = 0
        else:
            raise ValueError(f"Unknown kind: {kind}")

        if anomaly_type == 'single':
            loc = location if location else np.random.choice(['beginning', 'middle', 'end'])
            df, info_anom = ts.generate_collective_anomalies(
                df, num_anomalies=1, location=loc, scale_factor=scale_factor
            ) # Bu df['stationary']'yi 0 yapar
            label = f"{kind}_single_collective_anomaly_{loc}_{l}"
            collective_anomaly = 1
            multi_collective_anomaly = 0
            location_meta = f"{info_anom['location']}"
            anomaly_indices = f"{info_anom['starts']}, {info_anom['ends']}"

        elif anomaly_type == 'multiple':
            k = max(2, int(num_anomalies))
            df, info_anom = ts.generate_collective_anomalies(
                df, num_anomalies=k, location=location, scale_factor=scale_factor
            ) # Bu df['stationary']'yi 0 yapar
            label = f"{kind}_multiple_collective_anomalies_{l}"
            collective_anomaly = 0
            multi_collective_anomaly = 1
            location_meta = "multiple"
            anomaly_indices = f"{info_anom['starts']}, {info_anom['ends']}"
        else:
            raise ValueError("Invalid anomaly_type. Must be 'single' or 'multiple'.")

        series_id = i + 1

        # --- YENİ LOGİK ---
        is_stat_flag = int(df['stationary'].iloc[0]) # Değeri al (0)
        df = df.drop(columns=['stationary'])
        # --- BİTTİ ---

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag, # <--- YENİ EKLENDİ
            base_series=kind,
            base_coefs=base_coefs,
            order=base_order,
            collective_anomaly=collective_anomaly,
            multi_collective_anomaly=multi_collective_anomaly,
            location_collective=location_meta,
            location_collective_pts=anomaly_indices
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    _save_and_cleanup(all_dfs, folder, count, label)


########CONTEXTUAL ANOMALY########
def generate_contextual_anomaly_dataset(
    folder,
    count=5,
    anomaly_type = 'single',
    length_range=(300, 500),        
    location="middle",              
    num_anomalies=2,
    scale_factor=1
):
    os.makedirs(folder, exist_ok=True)
    global all_dfs
    all_dfs = []
    label = ""
    # ... (l etiketlemesi) ...
    if length_range == (50, 100):
        l = "short"
    elif length_range == (300, 500):
        l = "medium"
    else:
        l = "long"

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = TimeSeriesGenerator(length=length)
        df,info1 = ts.generate_seasonality_from_base_series('single') # Bu df['stationary']'yi 0 yapar
        if df is None: 
            print(f"Skipping one contextual anomaly generation for {folder} due to base seasonality error.")
            continue

        if anomaly_type == 'single':
            loc = location if location else np.random.choice(['beginning', 'middle', 'end'])
            df, info2 = ts.generate_contextual_anomalies(
                df, num_anomalies=1, location=loc, scale_factor=scale_factor, seasonal_period = info1['period']
            ) # Bu df['stationary']'yi 0 yapar (zaten 0'dı)
            if info2 is None:
                print(f"Skipping one contextual anomaly generation for {folder} due to anomaly error.")
                continue
            label = f"single_contextual_anomaly_{loc}_{l}"
            contextual_anomaly = 1
            multi_contextual_anomaly = 0
            location_meta = f"{info2['location']}"
            anomaly_indices = f"{info2['starts']}, {info2['ends']}"

        elif anomaly_type == 'multiple':
            k = max(2, int(num_anomalies))
            df, info2 = ts.generate_contextual_anomalies(
                df, num_anomalies=k, location=location, scale_factor=scale_factor, seasonal_period = info1['period']
            ) # Bu df['stationary']'yi 0 yapar
            if info2 is None:
                print(f"Skipping one contextual anomaly generation for {folder} due to anomaly error.")
                continue
            label = f"multiple_contextual_anomalies_{l}"
            contextual_anomaly = 0
            multi_contextual_anomaly = 1
            location_meta = "multiple"
            anomaly_indices = f"{info2['starts']}, {info2['ends']}"
        else:
            raise ValueError("Invalid anomaly_type. Must be 'single' or 'multiple'.")

        series_id = i + 1

        # --- YENİ LOGİK ---
        is_stat_flag = int(df['stationary'].iloc[0]) # Değeri al (0)
        df = df.drop(columns=['stationary'])
        # --- BİTTİ ---

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag, # <--- YENİ EKLENDİ
            contextual_anomaly=contextual_anomaly,
            multi_contextual_anomaly=multi_contextual_anomaly,
            location_contextual=location_meta,
            location_contextual_pts=anomaly_indices,
            seasonality=1,
            seasonality_frequency=info1['period']
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    _save_and_cleanup(all_dfs, folder, len(all_dfs), label)


########## STOCHASTIC TREND ############
# Not: Bu fonksiyonlar (rw, rwd, ima, ari, arima) zaten df['stationary'] = 0 
# olarak üreten generate_stochastic_trend'i çağırır.

#### RANDOM WALK #####
def generate_random_walk_dataset(folder, count=20, length_range=(50, 100)):
    os.makedirs(folder, exist_ok=True)
    global all_dfs
    all_dfs = []
    label = ""
    # ... (l etiketlemesi) ...
    if length_range == (50,100):
        l = "short"
    elif length_range == (300,500):
        l = "medium"
    else:
        l = "long"

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = TimeSeriesGenerator(length=length)
        df, info = ts.generate_stochastic_trend(kind='rw') # stationary = 0
        
        base_coefs = 0
        base_order = 0
        label = f"random_walk_{l}"
        series_id = i + 1

        # --- YENİ LOGİK ---
        is_stat_flag = int(df['stationary'].iloc[0]) # Değeri al (0)
        df = df.drop(columns=['stationary'])
        # --- BİTTİ ---

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag, # <--- YENİ EKLENDİ
            base_series="rw",
            base_coefs=base_coefs,
            order=base_order,
            stochastic_trend=1
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    _save_and_cleanup(all_dfs, folder, count, label)



#### RANDOM WALK WITH DRIFT #####
def generate_random_walk_with_drift_dataset(folder, count=20, length_range=(50, 100)):
    os.makedirs(folder, exist_ok=True)
    global all_dfs
    all_dfs = []
    label = ""
    # ... (l etiketlemesi) ...
    if length_range == (50,100):
        l = "short"
    elif length_range == (300,500):
        l = "medium"
    else:
        l = "long"

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = TimeSeriesGenerator(length=length)
        df, info = ts.generate_stochastic_trend(kind='rwd') # stationary = 0
        
        base_coefs = 0
        base_order = 0
        label = f"random_walk_drift_{l}"
        series_id = i + 1

        # --- YENİ LOGİK ---
        is_stat_flag = int(df['stationary'].iloc[0]) # Değeri al (0)
        df = df.drop(columns=['stationary'])
        # --- BİTTİ ---

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag, # <--- YENİ EKLENDİ
            base_series="rwd",
            order=base_order,
            base_coefs=base_coefs,
            stochastic_trend=1
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    _save_and_cleanup(all_dfs, folder, count, label)



#### IMA #####
def generate_ima_dataset(folder, count=20, length_range=(50, 100)):
    os.makedirs(folder, exist_ok=True)
    global all_dfs
    all_dfs = []
    label = ""
    # ... (l etiketlemesi) ...
    if length_range == (50,100):
        l = "short"
    elif length_range == (300,500):
        l = "medium"
    else:
        l = "long"

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = TimeSeriesGenerator(length=length)
        df, info = ts.generate_stochastic_trend(kind='ima') # stationary = 0

        base_coefs = f"({info['ma_coefs']})"
        base_order = f"({info['ma_order']})"
        diff = f"({info['diff']})"
        label = f"ima_{l}"
        series_id = i + 1

        # --- YENİ LOGİK ---
        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])
        # --- BİTTİ ---

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag, # <--- YENİ EKLENDİ
            base_series="ima",
            order=base_order,
            base_coefs=base_coefs,
            stochastic_trend=1,
            difference = diff
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    _save_and_cleanup(all_dfs, folder, count, label)



#### ARI #####
def generate_ari_dataset(folder, count=20, length_range=(50, 100)):
    os.makedirs(folder, exist_ok=True)
    global all_dfs
    all_dfs = []
    label = ""
    # ... (l etiketlemesi) ...
    if length_range == (50,100):
        l = "short"
    elif length_range == (300,500):
        l = "medium"
    else:
        l = "long"

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = TimeSeriesGenerator(length=length)
        df, info = ts.generate_stochastic_trend(kind='ari') # stationary = 0

        base_coefs = f"({info['ar_coefs']})"
        base_order = f"({info['ar_order']})"
        diff = f"({info['diff']})"
        label = f"ari_{l}"
        series_id = i + 1

        # --- YENİ LOGİK ---
        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])
        # --- BİTTİ ---

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag, # <--- YENİ EKLENDİ
            base_series="ari",
            order=base_order,
            base_coefs=base_coefs,
            stochastic_trend=1,
            difference = diff
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    _save_and_cleanup(all_dfs, folder, count, label)


#### ARIMA #####
def generate_arima_dataset(folder, count=20, length_range=(50, 100)):
    os.makedirs(folder, exist_ok=True)
    global all_dfs
    all_dfs = []
    label = ""
    # ... (l etiketlemesi) ...
    if length_range == (50,100):
        l = "short"
    elif length_range == (300,500):
        l = "medium"
    else:
        l = "long"

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = TimeSeriesGenerator(length=length)
        df, info = ts.generate_stochastic_trend(kind='arima') # stationary = 0

        base_coefs = f"({info['ar_coefs']},{info['ma_coefs']})"
        base_order = f"({info['ar_order']},{info['ma_order']})"
        diff = f"({info['diff']})"
        label = f"arima_{l}"
        series_id = i + 1

        # --- YENİ LOGİK ---
        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])
        # --- BİTTİ ---

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag, # <--- YENİ EKLENDİ
            base_series="arima",
            order=base_order,
            base_coefs=base_coefs,
            stochastic_trend=1,
            difference = diff
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    _save_and_cleanup(all_dfs, folder, count, label)



######## VOLATILITY #########
# Not: Bu fonksiyonlar (arch, garch, vb.) zaten df['stationary'] = 0 
# olarak üreten generate_volatility'yi çağırır.

#### ARCH #####
def generate_arch_dataset(folder, count=20, length_range=(50, 100)):
    os.makedirs(folder, exist_ok=True)
    global all_dfs
    all_dfs = []
    label = ""
    # ... (l etiketlemesi) ...
    if length_range == (50,100):
        l = "short"
    elif length_range == (300,500):
        l = "medium"
    else:
        l = "long"

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = TimeSeriesGenerator(length=length)
        df, info = ts.generate_volatility(kind='arch') # stationary = 0
        
        base_coefs = f"({info['alpha']},{info['omega']})"
        label = f"arch_{l}"
        series_id = i + 1

        # --- YENİ LOGİK ---
        is_stat_flag = int(df['stationary'].iloc[0]) # Değeri al (0)
        df = df.drop(columns=['stationary'])
        # --- BİTTİ ---

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag, # <--- YENİ EKLENDİ
            base_series="arch",
            base_coefs=base_coefs,
            volatility=1
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)
        
    _save_and_cleanup(all_dfs, folder, count, label)


#### GARCH #####
def generate_garch_dataset(folder, count=20, length_range=(50, 100)):
    os.makedirs(folder, exist_ok=True)
    global all_dfs
    all_dfs = []
    label = ""
    # ... (l etiketlemesi) ...
    if length_range == (50,100):
        l = "short"
    elif length_range == (300,500):
        l = "medium"
    else:
        l = "long"

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = TimeSeriesGenerator(length=length)
        df,info = ts.generate_volatility(kind='garch') # stationary = 0

        base_coefs = f"({info['alpha']},{info['beta']},{info['omega']})"
        label = f"garch_{l}"
        series_id = i + 1

        # --- YENİ LOGİK ---
        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])
        # --- BİTTİ ---

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag, # <--- YENİ EKLENDİ
            base_series="garch",
            base_coefs=base_coefs,
            volatility=1
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    _save_and_cleanup(all_dfs, folder, count, label)



#### EGARCH #####
def generate_egarch_dataset(folder, count=20, length_range=(50, 100)):
    os.makedirs(folder, exist_ok=True)
    global all_dfs
    all_dfs = []
    label = ""
    # ... (l etiketlemesi) ...
    if length_range == (50,100):
        l = "short"
    elif length_range == (300,500):
        l = "medium"
    else:
        l = "long"

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = TimeSeriesGenerator(length=length)
        df,info = ts.generate_volatility(kind='egarch') # stationary = 0

        base_coefs = f"({info['alpha']},{info['beta']},{info['theta']},{info['lambda']},{info['omega']})"
        label = f"egarch_{l}"
        series_id = i + 1

        # --- YENİ LOGİK ---
        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])
        # --- BİTTİ ---

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag, # <--- YENİ EKLENDİ
            base_series="egarch",
            base_coefs=base_coefs,
            volatility=1
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)
        
    _save_and_cleanup(all_dfs, folder, count, label)


#### APARCH #####
def generate_aparch_dataset(folder, count=20, length_range=(50, 100)):
    os.makedirs(folder, exist_ok=True)
    global all_dfs
    all_dfs = []
    label = ""
    # ... (l etiketlemesi) ...
    if length_range == (50,100):
        l = "short"
    elif length_range == (300,500):
        l = "medium"
    else:
        l = "long"

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = TimeSeriesGenerator(length=length)
        df,info = ts.generate_volatility(kind='aparch') # stationary = 0

        base_coefs = f"({info['alpha']},{info['beta']},{info['gamma']},{info['delta']},{info['omega']})"
        label = f"aparch_{l}"
        series_id = i + 1

        # --- YENİ LOGİK ---
        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])
        # --- BİTTİ ---

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag, # <--- YENİ EKLENDİ
            base_series="aparch",
            base_coefs=base_coefs,
            volatility=1
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)
        
    _save_and_cleanup(all_dfs, folder, count, label)


#### SINGLE SEASONALITY #####
def generate_single_seasonality_dataset(folder, count=20, length_range=(50, 100)):
    os.makedirs(folder, exist_ok=True)
    global all_dfs
    all_dfs = []
    label = ""
    # ... (l etiketlemesi) ...
    if length_range == (50,100):
        l = "short"
    elif length_range == (300,500):
        l = "medium"
    else:
        l = "long"

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = TimeSeriesGenerator(length=length)
        df, info = ts.generate_seasonality_from_base_series(kind = 'single') # stationary = 0
        if df is None: continue
        
        period = f"({info['period']})"
        label = f"single_seasonality_{l}"
        series_id = i + 1

        # --- YENİ LOGİK ---
        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])
        # --- BİTTİ ---

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag, # <--- YENİ EKLENDİ
            seasonality = 1,
            seasonality_frequency = period
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)
        
    _save_and_cleanup(all_dfs, folder, len(all_dfs), label)


#### MULTIPLE SEASONALITY #####
def generate_multiple_seasonality_dataset(folder, count=20, num_components=2, length_range=(50, 100)):
    os.makedirs(folder, exist_ok=True)
    global all_dfs
    all_dfs = []
    label = ""
    # ... (l etiketlemesi) ...
    if length_range == (50,100):
        l = "short"
    elif length_range == (300,500):
        l = "medium"
    else:
        l = "long"

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = TimeSeriesGenerator(length=length)
        df, info = ts.generate_seasonality_from_base_series(kind = 'multiple', num_components = num_components) # stationary = 0
        if df is None: continue
        
        periods = f"({info['periods']})"
        label = f"multiple_seasonality_{l}"
        series_id = i + 1

        # --- YENİ LOGİK ---
        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])
        # --- BİTTİ ---

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag, # <--- YENİ EKLENDİ
            multiple_seasonality = 1,
            seasonality_frequency = periods
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)
        
    _save_and_cleanup(all_dfs, folder, len(all_dfs), label)


#### SARMA SEASONALITY #####
def generate_sarma_dataset(folder, count=20, length_range=(50, 100)):
    os.makedirs(folder, exist_ok=True)
    global all_dfs
    all_dfs = []
    label = ""
    # ... (l etiketlemesi) ...
    if length_range == (50,100):
        l = "short"
    elif length_range == (300,500):
        l = "medium"
    else:
        l = "long"

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = TimeSeriesGenerator(length=length)
        df, info = ts.generate_seasonality_from_base_series(kind = 'sarma') # stationary = 0
        if df is None or info is None:
             print(f"Skipping one SARMA generation for {folder} due to an error.")
             continue
             
        base_coefs = f"({info['coefs']})"
        base_order = f"({info['ar_order']}, {info['ma_order']}, {info['seasonal_ar_order']}, {info['seasonal_ma_order']})"
        period = f"({info['period']})"
        diff = f"({info['diff']})"
        seasonal_diff = f"({info['seasonal_diff']})"
        label = f"sarma_seasonality_{l}"
        series_id = i + 1

        # --- YENİ LOGİK ---
        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])
        # --- BİTTİ ---

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag, # <--- YENİ EKLENDİ
            base_series = 'sarma',
            seasonality_from_base = 1,
            seasonality_frequency = period,
            base_coefs = base_coefs,
            order= base_order,
            difference = diff,
            seasonal_difference = seasonal_diff
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    _save_and_cleanup(all_dfs, folder, len(all_dfs), label)


#### SARIMA SEASONALITY #####
def generate_sarima_dataset(folder, count=20, length_range=(50, 100)):
    os.makedirs(folder, exist_ok=True)
    global all_dfs
    all_dfs = []
    label = ""
    # ... (l etiketlemesi) ...
    if length_range == (50,100):
        l = "short"
    elif length_range == (300,500):
        l = "medium"
    else:
        l = "long"

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = TimeSeriesGenerator(length=length)
        df, info = ts.generate_seasonality_from_base_series(kind = 'sarima') # stationary = 0
        if df is None or info is None:
             print(f"Skipping one SARIMA generation for {folder} due to an error.")
             continue

        base_coefs = f"({info['coefs']})"
        base_order = f"({info['ar_order']}, {info['ma_order']}, {info['seasonal_ar_order']}, {info['seasonal_ma_order']})"
        period = f"({info['period']})"
        diff = f"({info['diff']})"
        seasonal_diff = f"({info['seasonal_diff']})"
        label = f"sarima_seasonality_{l}"
        series_id = i + 1

        # --- YENİ LOGİK ---
        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])
        # --- BİTTİ ---

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag, # <--- YENİ EKLENDİ
            base_series = 'sarima',
            seasonality_from_base = 1,
            seasonality_frequency = period,
            base_coefs = base_coefs,
            order= base_order,
            difference = diff,
            seasonal_difference = seasonal_diff
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    _save_and_cleanup(all_dfs, folder, len(all_dfs), label)


######## MEAN SHIFT ########
def generate_mean_shift_dataset(
    folder,
    kind='ar',
    count=5,
    length_range=(300, 500),    
    break_type = 'single',
    signs = [1],
    location="middle",              
    num_breaks=2,
    scale_factor=1,
    seasonal_period=None
):
    os.makedirs(folder, exist_ok=True)
    global all_dfs
    all_dfs = []
    label = ""
    # ... (l etiketlemesi) ...
    if length_range == (50, 100):
        l = "short"
    elif length_range == (300, 500):
        l = "medium"
    else:
        l = "long"

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = TimeSeriesGenerator(length=length)

        # ... (base series üretimi) ...
        if kind == 'ar':
            df, info = ts.generate_stationary_base_series('ar')
            base_coefs = f"({info['ar_coefs']})"
            base_order = f"({info['ar_order']})"
        elif kind == 'ma':
            df,info = ts.generate_stationary_base_series('ma')
            base_coefs = f"({info['ma_coefs']})"
            base_order = f"({info['ma_order']})"
        elif kind == 'arma':
            df,info = ts.generate_stationary_base_series('arma')
            base_coefs = f"({info['ar_coefs']},{info['ma_coefs']})"
            base_order = f"({info['ar_order']},{info['ma_order']})"
        elif kind == 'white_noise':
            df,info = ts.generate_stationary_base_series('white_noise')
            base_coefs = 0
            base_order = 0
        else:
            raise ValueError(f"Unknown kind: {kind}")


        if signs[0] < 0:
            mean_shift_increase_meta = 0
            mean_shift_decrease_meta = 1 
        else:
            mean_shift_increase_meta = 1
            mean_shift_decrease_meta = 0

        if break_type == 'single':
            loc = location if location else np.random.choice(['beginning', 'middle', 'end'])
            df, info2 = ts.generate_mean_shift(
                df, signs = signs, num_breaks=1, location=loc,
                scale_factor=scale_factor, seasonal_period = seasonal_period
            ) # Bu df['stationary']'yi 0 yapar
            label = f"{kind}_single_mean_shift_{loc}_{l}"
            location_meta = f"{info2['location']}"
            shift_indices_magnitudes = f"{info2['shift_indices']},{info2['shift_magnitudes']}"
            multi_mean_shift_meta = 0
            
        elif break_type == 'multiple':
            k = max(2, int(num_breaks))
            signs_meta = [np.random.choice([1, -1]) for _ in range(k)]
            df, info2 = ts.generate_mean_shift(
                df, signs = signs_meta, num_breaks=k,
                scale_factor=scale_factor, seasonal_period = seasonal_period
            ) # Bu df['stationary']'yi 0 yapar
            label = f"{kind}_multiple_mean_shifts_{l}"
            multi_mean_shift_meta = 1
            location_meta = "multiple"
            shift_indices_magnitudes = f"{info2['shift_indices']},{info2['shift_magnitudes']}"
            mean_shift_increase_meta = 0
            mean_shift_decrease_meta = 0
        else:
            raise ValueError("Invalid break_type. Must be 'single' or 'multiple'.")

        series_id = i + 1

        # --- YENİ LOGİK ---
        is_stat_flag = int(df['stationary'].iloc[0]) # Değeri al (0)
        df = df.drop(columns=['stationary'])
        # --- BİTTİ ---

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag, # <--- YENİ EKLENDİ
            base_series=kind,
            base_coefs=base_coefs,
            order=base_order,
            mean_shift_increase = mean_shift_increase_meta,
            mean_shift_decrease = mean_shift_decrease_meta,
            multi_mean_shift = multi_mean_shift_meta,
            location_mean_shift=location_meta,
            location_mean_pts=shift_indices_magnitudes
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    _save_and_cleanup(all_dfs, folder, count, label)


######## VARIANCE SHIFT ########
def generate_variance_shift_dataset(
    folder,
    kind='ar',
    count=5,
    length_range=(300, 500),    
    break_type = 'single',
    signs = [1],
    location="middle",              
    num_breaks=2,
    scale_factor=1,
    seasonal_period=None
):
    os.makedirs(folder, exist_ok=True)
    global all_dfs
    all_dfs = []
    label = ""
    # ... (l etiketlemesi) ...
    if length_range == (50, 100):
        l = "short"
    elif length_range == (300, 500):
        l = "medium"
    else:
        l = "long"

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = TimeSeriesGenerator(length=length)

        # ... (base series üretimi) ...
        if kind == 'ar':
            df, info = ts.generate_stationary_base_series('ar')
            base_coefs = f"({info['ar_coefs']})"
            base_order = f"({info['ar_order']})"
        elif kind == 'ma':
            df,info = ts.generate_stationary_base_series('ma')
            base_coefs = f"({info['ma_coefs']})"
            base_order = f"({info['ma_order']})"
        elif kind == 'arma':
            df,info = ts.generate_stationary_base_series('arma')
            base_coefs = f"({info['ar_coefs']},{info['ma_coefs']})"
            base_order = f"({info['ar_order']},{info['ma_order']})"
        elif kind == 'white_noise':
            df,info = ts.generate_stationary_base_series('white_noise')
            base_coefs = 0
            base_order = 0
        else:
            raise ValueError(f"Unknown kind: {kind}")


        if signs[0] < 0:
            variance_shift_increase_meta = 0
            variance_shift_decrease_meta = 1 
        else:
            variance_shift_increase_meta = 1
            variance_shift_decrease_meta = 0

        if break_type == 'single':
            loc = location if location else np.random.choice(['beginning', 'middle', 'end'])
            df, info2 = ts.generate_variance_shift(
                df, signs = signs, num_breaks=1, location=loc,
                scale_factor=scale_factor, seasonal_period = seasonal_period
            ) # Bu df['stationary']'yi 0 yapar
            label = f"{kind}_single_variance_shift_{loc}_{l}"
            location_meta = f"{info2['location']}"
            shift_indices_magnitudes = f"{info2['shift_indices']},{info2['shift_magnitudes']}"
            multi_variance_shift_meta = 0
            
        elif break_type == 'multiple':
            k = max(2, int(num_breaks))
            signs_meta = [np.random.choice([1, -1]) for _ in range(k)]
            df, info2 = ts.generate_variance_shift(
                df, signs = signs_meta, num_breaks=k,
                scale_factor=scale_factor, seasonal_period = seasonal_period
            ) # Bu df['stationary']'yi 0 yapar
            label = f"{kind}_multiple_variance_shifts_{l}"
            multi_variance_shift_meta = 1
            location_meta = "multiple"
            shift_indices_magnitudes = f"{info2['shift_indices']},{info2['shift_magnitudes']}"
            variance_shift_increase_meta = 0
            variance_shift_decrease_meta = 0
        else:
            raise ValueError("Invalid break_type. Must be 'single' or 'multiple'.")

        series_id = i + 1

        # --- YENİ LOGİK ---
        is_stat_flag = int(df['stationary'].iloc[0]) # Değeri al (0)
        df = df.drop(columns=['stationary'])
        # --- BİTTİ ---

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag, # <--- YENİ EKLENDİ
            base_series=kind,
            base_coefs=base_coefs,
            order=base_order,
            variance_shift_increase = variance_shift_increase_meta,
            variance_shift_decrease = variance_shift_decrease_meta,
            multi_variance_shift = multi_variance_shift_meta,
            location_variance_shift=location_meta,
            location_variance_pts=shift_indices_magnitudes
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    _save_and_cleanup(all_dfs, folder, count, label)


######## TREND SHIFT ########
def generate_trend_shift_dataset(
    folder,
    kind='ar',
    count=5,
    length_range=(300, 500),    
    break_type = 'single',
    change_types = ['direction_change'],
    location="middle",              
    num_breaks=2,
    scale_factor=1,
    seasonal_period=None,
    sign = None
):
    os.makedirs(folder, exist_ok=True)
    global all_dfs
    all_dfs = []
    label = ""
    # ... (l etiketlemesi) ...
    if length_range == (50, 100):
        l = "short"
    elif length_range == (300, 500):
        l = "medium"
    else:
        l = "long"

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = TimeSeriesGenerator(length=length)

        # ... (base series üretimi) ...
        if kind == 'ar':
            df, info = ts.generate_stationary_base_series('ar')
            base_coefs = f"({info['ar_coefs']})"
            base_order = f"({info['ar_order']})"
        elif kind == 'ma':
            df,info = ts.generate_stationary_base_series('ma')
            base_coefs = f"({info['ma_coefs']})"
            base_order = f"({info['ma_order']})"
        elif kind == 'arma':
            df,info = ts.generate_stationary_base_series('arma')
            base_coefs = f"({info['ar_coefs']},{info['ma_coefs']})"
            base_order = f"({info['ar_order']},{info['ma_order']})"
        elif kind == 'white_noise':
            df,info = ts.generate_stationary_base_series('white_noise')
            base_coefs = 0
            base_order = 0
        else:
            raise ValueError(f"Unknown kind: {kind}")


        current_change_types = change_types
        if break_type == 'multiple':
             k = max(2, int(num_breaks))
             current_change_types = np.random.choice(['direction_change', 'magnitude_change', 'direction_and_magnitude_change'], k).tolist()
        
        if break_type == 'single':
            if current_change_types[0] == 'direction_change':
                trend_shift_slope_meta = 1
                trend_shift_intercept_meta = 0 
            elif current_change_types[0] == 'magnitude_change':
                trend_shift_slope_meta = 0
                trend_shift_intercept_meta = 1
            elif current_change_types[0] == 'direction_and_magnitude_change':
                trend_shift_slope_meta = 1
                trend_shift_intercept_meta = 1
            else:
                raise ValueError(f"Unknown change type: {current_change_types[0]}")
        
        sign = sign if sign is not None else np.random.choice([-1,1])
        
        if break_type == 'single':
            loc = location if location else np.random.choice(['beginning', 'middle', 'end'])
            df, info2 = ts.generate_trend_shift(
                df, sign = sign, num_breaks=1, change_types= current_change_types,
                location=loc, scale_factor=scale_factor, seasonal_period = seasonal_period
            ) # Bu df['stationary']'yi 0 yapar
            label = f"{kind}_single_trend_shift_{loc}_{l}"
            location_meta = f"{info2['location']}"
            shift_indices_types = f"{info2['shift_indices']},{info2['shift_types']}"
            multi_trend_shift_meta = 0
            
        elif break_type == 'multiple':
            df, info2 = ts.generate_trend_shift(
                df, sign = sign, change_types = current_change_types, num_breaks=k,
                scale_factor=scale_factor, seasonal_period = seasonal_period
            ) # Bu df['stationary']'yi 0 yapar
            label = f"{kind}_multiple_trend_shifts_{l}"
            multi_trend_shift_meta = 1
            location_meta = "multiple"
            shift_indices_types = f"{info2['shift_indices']},{info2['shift_types']}"
            trend_shift_slope_meta = 0
            trend_shift_intercept_meta = 0
        else:
            raise ValueError("Invalid break_type. Must be 'single' or 'multiple'.")

        series_id = i + 1

        # --- YENİ LOGİK ---
        is_stat_flag = int(df['stationary'].iloc[0]) # Değeri al (0)
        df = df.drop(columns=['stationary'])
        # --- BİTTİ ---

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag, # <--- YENİ EKLENDİ
            base_series=kind,
            base_coefs=base_coefs,
            order=base_order,
            trend_shift_slope = trend_shift_slope_meta,
            trend_shift_intercept = trend_shift_intercept_meta,
            multi_trend_shift = multi_trend_shift_meta,
            location_trend_shift=location_meta,
            location_trend_pts=shift_indices_types
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    _save_and_cleanup(all_dfs, folder, count, label)