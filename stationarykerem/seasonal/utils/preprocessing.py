# utils/preprocessing.py
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler

def extract_features(time_series, window_sizes=[7, 30, 90], decompose=True):
    """
    Zaman serisinden özellikler çıkar.
    
    Args:
        time_series (pd.Series): Zaman serisi
        window_sizes (list): Hareketli ortalama pencereleri
        decompose (bool): STL ayrıştırması yapılıp yapılmayacağı
        
    Returns:
        dict: Özellikler sözlüğü
    """
    features = {}
    
    # Temel istatistikler
    features['mean'] = time_series.mean()
    features['std'] = time_series.std()
    features['min'] = time_series.min()
    features['max'] = time_series.max()
    features['median'] = time_series.median()
    features['skew'] = time_series.skew()
    features['kurtosis'] = time_series.kurtosis()
    
    # Fark istatistikleri
    diff = time_series.diff().dropna()
    features['diff_mean'] = diff.mean()
    features['diff_std'] = diff.std()
    features['diff_min'] = diff.min()
    features['diff_max'] = diff.max()
    
    # Otokorelasyon özellikleri
    for lag in [1, 7, 14, 30]:
        if len(time_series) > lag:
            features[f'autocorr_{lag}'] = time_series.autocorr(lag)
    
    # Hareketli ortalama özellikleri
    for window in window_sizes:
        if len(time_series) > window:
            rolling_mean = time_series.rolling(window=window).mean()
            features[f'rolling_mean_{window}'] = rolling_mean.mean()
            features[f'rolling_std_{window}'] = time_series.rolling(window=window).std().mean()
            
            # Orijinal seriden hareketli ortalama farkı
            deviation = time_series - rolling_mean
            features[f'rolling_dev_{window}_mean'] = deviation.mean()
            features[f'rolling_dev_{window}_std'] = deviation.std()
    
    # STL ayrıştırması
    if decompose and len(time_series) >= 24:  # Minimum uzunluk gerekli
        try:
            # En uygun periyot belirleme
            potential_periods = [7, 14, 30, 90, 365]
            best_period = min([p for p in potential_periods if p < len(time_series) * 0.33], 
                              default=min(14, len(time_series) // 3))
            
            # STL ayrıştırması
            stl = STL(time_series, seasonal=best_period, robust=True)
            result = stl.fit()
            
            # Trend, mevsimsel ve residual bileşenler
            trend = result.trend
            seasonal = result.seasonal
            residual = result.resid
            
            # Bileşenlerin istatistikleri
            features['trend_strength'] = 1 - (residual.var() / (seasonal + residual).var())
            features['seasonal_strength'] = 1 - (residual.var() / (trend + residual).var())
            features['seasonal_peak_to_peak'] = seasonal.max() - seasonal.min()
            features['seasonal_mean'] = seasonal.mean()
            features['seasonal_std'] = seasonal.std()
            
            # Mevsimsel indeks hesapla
            seasonal_index = seasonal / time_series.mean()
            features['seasonal_index_std'] = seasonal_index.std()
            
            # Mevsimsel bileşenin otokorelasyonu
            for lag in [1, 7, 14]:
                if len(seasonal) > lag:
                    features[f'seasonal_autocorr_{lag}'] = seasonal.autocorr(lag)
        except:
            # Ayrıştırma başarısız olursa varsayılan değerler
            features['trend_strength'] = 0
            features['seasonal_strength'] = 0
            features['seasonal_peak_to_peak'] = 0
            features['seasonal_mean'] = 0
            features['seasonal_std'] = 0
            features['seasonal_index_std'] = 0
    
    return features

def prepare_features_for_training(time_series_list, window_sizes=[7, 30, 90], decompose=True):
    """
    Zaman serisi listesi için eğitim özellikleri hazırla.
    
    Args:
        time_series_list (list): Zaman serisi listesi
        window_sizes (list): Hareketli ortalama pencereleri
        decompose (bool): STL ayrıştırması yapılıp yapılmayacağı
        
    Returns:
        pd.DataFrame: Hazırlanan özellikler
    """
    features_list = []
    
    for i, ts in enumerate(time_series_list):
        # Özellik çıkarma
        features = extract_features(ts, window_sizes, decompose)
        features['series_id'] = i
        features_list.append(features)
    
    # DataFrame oluştur
    features_df = pd.DataFrame(features_list)
    
    # Eksik değerleri doldur
    features_df = features_df.fillna(0)
    
    return features_df

def prepare_lstm_sequences(time_series_list, seq_length=50, step=1):
    """
    LSTM modeli için sekans verileri hazırla.
    
    Args:
        time_series_list (list): Zaman serisi listesi
        seq_length (int): Sekans uzunluğu
        step (int): Adım boyutu
        
    Returns:
        tuple: X ve y dizileri
    """
    sequences = []
    
    for ts in time_series_list:
        # Zaman serisini normalize et
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(ts.values.reshape(-1, 1)).flatten()
        
        # Sekansları oluştur
        for i in range(0, len(normalized_data) - seq_length, step):
            sequences.append(normalized_data[i:i+seq_length])
    
    # Dizi olarak dönüştür
    X = np.array(sequences)
    
    return X

def create_breakpoint_labels(metadata_list):
    """
    Kırılma noktaları için etiketler oluştur.
    
    Args:
        metadata_list (list): Metadata listesi
        
    Returns:
        pd.DataFrame: Kırılma noktaları etiketi
    """
    breakpoint_labels = []
    
    for i, meta in enumerate(metadata_list):
        seasonality_info = meta['seasonality']
        
        # Mevsimsellik durumu
        is_seasonal = seasonality_info['is_seasonal']
        
        # Kırılma noktaları
        if is_seasonal and 'breakpoints' in seasonality_info and seasonality_info['breakpoints']:
            for bp in seasonality_info['breakpoints']:
                breakpoint_labels.append({
                    'series_id': i,
                    'is_seasonal': is_seasonal,
                    'has_breakpoint': True,
                    'breakpoint_position': bp['position'],
                    'old_period': bp['old_period'],
                    'new_period': bp['new_period'],
                    'period_ratio': bp['new_period'] / bp['old_period'] if bp['old_period'] != 0 else 0,
                    'amplitude_ratio': bp['new_amplitude'] / bp['old_amplitude'] if bp['old_amplitude'] != 0 else 0
                })
        else:
            # Kırılma noktası olmayan seriler
            breakpoint_labels.append({
                'series_id': i,
                'is_seasonal': is_seasonal,
                'has_breakpoint': False,
                'breakpoint_position': 0,
                'old_period': 0,
                'new_period': 0,
                'period_ratio': 0,
                'amplitude_ratio': 0
            })
    
    return pd.DataFrame(breakpoint_labels)

def create_seasonality_labels(metadata_list):
    """
    Mevsimsellik için etiketler oluştur.
    
    Args:
        metadata_list (list): Metadata listesi
        
    Returns:
        pd.DataFrame: Mevsimsellik etiketi
    """
    seasonality_labels = []
    
    for i, meta in enumerate(metadata_list):
        seasonality_info = meta['seasonality']
        
        # Mevsimsellik durumu
        is_seasonal = seasonality_info['is_seasonal']
        seasonal_type = seasonality_info['type']
        
        # Dönemleri string olarak birleştir
        periods_str = ','.join(map(str, seasonality_info['periods'])) if seasonality_info['periods'] else ''
        
        seasonality_labels.append({
            'series_id': i,
            'is_seasonal': is_seasonal,
            'seasonal_type': seasonal_type,
            'periods': periods_str,
            'has_breakpoints': len(seasonality_info.get('breakpoints', [])) > 0
        })
    
    return pd.DataFrame(seasonality_labels)