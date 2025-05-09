# utils/seasonality_detector.py
import os
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
import xgboost as xgb
import pickle
from joblib import load
from sklearn.preprocessing import StandardScaler

class SeasonalityDetector:
    """
    Zaman serilerinde mevsimsellik ve mevsimsel kırılma noktalarını tespit eden sınıf.
    """
    
    def __init__(self, xgboost_model_path=None, lstm_model_path=None, scaler_path=None, 
                 use_xgboost=True, use_lstm=False, seq_length=50, confidence_threshold=0.6):
        """
        Args:
            xgboost_model_path (str, optional): XGBoost modeli dosya yolu. None ise varsayılan konum kullanılır.
            lstm_model_path (str, optional): LSTM modeli dosya yolu. None ise varsayılan konum kullanılır.
            scaler_path (str, optional): Özellik ölçekleyici dosya yolu. None ise ölçekleme yapılmaz.
            use_xgboost (bool): XGBoost modelini kullanıp kullanmamayı belirler.
            use_lstm (bool): LSTM modelini kullanıp kullanmamayı belirler.
            seq_length (int): LSTM sekans uzunluğu.
            confidence_threshold (float): Mevsimsellik kabul etmek için gereken eşik değeri.
        """
        # Proje ana klasörünü belirle
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Varsayılan model yolları
        if xgboost_model_path is None:
            xgboost_model_path = os.path.join(project_dir, 'models', 'xgboost_model', 'seasonality_model.json')
        
        if lstm_model_path is None:
            lstm_model_path = os.path.join(project_dir, 'models', 'lstm_model', 'seasonality_model.pth')
            lstm_info_path = os.path.join(project_dir, 'models', 'lstm_model', 'model_info.pkl')
        else:
            lstm_info_path = os.path.splitext(lstm_model_path)[0] + '_info.pkl'
        
        if scaler_path is None:
            scaler_path = os.path.join(project_dir, 'models', 'xgboost_model', 'scaler.joblib')
        
        self.use_xgboost = use_xgboost
        self.use_lstm = use_lstm
        self.seq_length = seq_length
        self.confidence_threshold = confidence_threshold
        
        # XGBoost modelini yükle
        if use_xgboost and os.path.exists(xgboost_model_path):
            self.xgboost_model = xgb.XGBClassifier()
            self.xgboost_model.load_model(xgboost_model_path)
            print("XGBoost modeli yüklendi.")
        elif use_xgboost:
            raise FileNotFoundError(f"XGBoost modeli bulunamadı: {xgboost_model_path}")
        else:
            self.xgboost_model = None
        
        # LSTM modelini yükle (PyTorch)
        if use_lstm:
            try:
                # PyTorch kullanıldığı için gerekli importları yap
                import torch
                import torch.nn as nn
                
                if os.path.exists(lstm_model_path) and os.path.exists(lstm_info_path):
                    # Model mimarisini yükle
                    with open(lstm_info_path, 'rb') as f:
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
                    self.lstm_model = LSTMModel(
                        input_size=model_info.get('input_size', 1),
                        hidden_size=model_info.get('hidden_size', 64),
                        num_layers=model_info.get('num_layers', 2),
                        dropout=model_info.get('dropout_rate', 0.2)
                    )
                    
                    # Modeli yükle
                    self.lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=torch.device('cpu')))
                    self.lstm_model.eval()  # Değerlendirme moduna al
                    
                    print("LSTM modeli yüklendi (PyTorch).")
                else:
                    print(f"LSTM model dosyaları bulunamadı. LSTM devre dışı bırakılıyor.")
                    self.use_lstm = False
                    self.lstm_model = None
            except ImportError:
                print("PyTorch yüklü değil. LSTM devre dışı bırakılıyor.")
                self.use_lstm = False
                self.lstm_model = None
            except Exception as e:
                print(f"LSTM modeli yüklenirken hata oluştu: {e}")
                self.use_lstm = False
                self.lstm_model = None
        else:
            self.lstm_model = None
            
        # Ölçekleyiciyi yükle
        if os.path.exists(scaler_path):
            self.scaler = load(scaler_path)
            print("Özellik ölçekleyici yüklendi.")
        else:
            self.scaler = None
    
    def _detect_seasonality_with_xgboost(self, time_series):
        """
        XGBoost modeli ile mevsimsellik tespiti yap.
        
        Args:
            time_series (pd.Series): Analiz edilecek zaman serisi
            
        Returns:
            dict: Tespit sonuçları
        """
        # Özellik çıkarma
        features = extract_features(time_series)
        
        # DataFrame'e dönüştür
        features_df = pd.DataFrame([features])
        
        # Eksik değerleri doldur
        features_df = features_df.fillna(0)
        
        # Ölçeklendirme
        if self.scaler is not None:
            features_array = self.scaler.transform(features_df)
        else:
            features_array = features_df.values
        
        # Tahmin
        prediction_proba = self.xgboost_model.predict_proba(features_array)[0, 1]
        prediction = prediction_proba > self.confidence_threshold
        
        return {
            'is_seasonal': bool(prediction),
            'confidence': float(prediction_proba),
            'method': 'xgboost'
        }
    
    def _detect_seasonality_with_lstm(self, time_series):
        """
        LSTM modeli ile mevsimsellik tespiti yap.
        
        Args:
            time_series (pd.Series): Analiz edilecek zaman serisi
            
        Returns:
            dict: Tespit sonuçları
        """
        try:
            # PyTorch importu
            import torch
            
            # Zaman serisini normalize et
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(time_series.values.reshape(-1, 1)).flatten()
            
            # Eğer seri kısa ise
            if len(normalized_data) < self.seq_length:
                return {
                    'is_seasonal': False,
                    'confidence': 0.0,
                    'method': 'lstm',
                    'error': 'Series too short'
                }
            
            # Sekansları oluştur
            sequences = []
            for i in range(0, len(normalized_data) - self.seq_length + 1, self.seq_length // 2):
                sequences.append(normalized_data[i:i+self.seq_length])
            
            # PyTorch tensor'a dönüştür
            X = np.array(sequences).reshape(-1, self.seq_length, 1)
            X_tensor = torch.FloatTensor(X)
            
            # Tahmin
            self.lstm_model.eval()
            with torch.no_grad():
                predictions = self.lstm_model(X_tensor).cpu().numpy().flatten()
            
            avg_prediction = float(np.mean(predictions))
            prediction = avg_prediction > self.confidence_threshold
            
            return {
                'is_seasonal': bool(prediction),
                'confidence': avg_prediction,
                'method': 'lstm'
            }
            
        except Exception as e:
            print(f"LSTM tahmin sırasında hata: {e}")
            return {
                'is_seasonal': False,
                'confidence': 0.0,
                'method': 'lstm',
                'error': str(e)
            }
    
    def _detect_breakpoints(self, time_series, window_size=None, stride=None):
        """
        Zaman serisinde mevsimsel kırılma noktalarını tespit et.
        
        Args:
            time_series (pd.Series): Analiz edilecek zaman serisi
            window_size (int, optional): Analiz penceresi boyutu. None ise serinin %30'u kullanılır.
            stride (int, optional): Adım boyutu. None ise window_size'ın %25'i kullanılır.
            
        Returns:
            list: Tespit edilen kırılma noktaları
        """
        # Varsayılan pencere boyutu (serinin %30'u)
        if window_size is None:
            window_size = max(int(len(time_series) * 0.3), 14)
        
        # Varsayılan adım boyutu (pencere boyutunun %25'i)
        if stride is None:
            stride = max(int(window_size * 0.25), 1)
        
        # Serinin çok kısa olması durumu
        if len(time_series) < window_size * 2:
            return []
        
        breakpoints = []
        periods = []
        strengths = []
        
        # Kayan pencere analizi
        for i in range(0, len(time_series) - window_size, stride):
            segment = time_series.iloc[i:i+window_size]
            
            # STL ayrıştırması için minimum uzunluk kontrolü
            if len(segment) < 10:
                continue
            
            # En uygun periyodu belirle
            best_period = self._find_best_period(segment)
            
            if best_period is not None:
                periods.append(best_period)
                
                # Mevsimsel gücü hesapla
                seasonal_strength = self._calculate_seasonal_strength(segment, best_period)
                strengths.append(seasonal_strength)
            else:
                periods.append(0)
                strengths.append(0)
        
        # Dönem değişimlerini belirle
        period_changes = []
        for i in range(1, len(periods)):
            if periods[i-1] != 0 and periods[i] != 0:
                change_ratio = abs(periods[i] - periods[i-1]) / max(periods[i-1], 1)
                if change_ratio > 0.2:  # %20'den fazla değişim
                    period_changes.append({
                        'position': i * stride,
                        'old_period': periods[i-1],
                        'new_period': periods[i],
                        'change_ratio': change_ratio
                    })
        
        # Mevsimsel güç değişimlerini belirle
        strength_changes = []
        for i in range(1, len(strengths)):
            if strengths[i-1] > 0.3 and strengths[i] > 0.3:  # Anlamlı mevsimsellik olan bölgeler
                change_ratio = abs(strengths[i] - strengths[i-1]) / max(strengths[i-1], 0.1)
                if change_ratio > 0.3:  # %30'dan fazla değişim
                    strength_changes.append({
                        'position': i * stride,
                        'old_strength': strengths[i-1],
                        'new_strength': strengths[i],
                        'change_ratio': change_ratio
                    })
        
        # Değişimleri birleştir
        all_changes = period_changes + strength_changes
        
        # Değişimleri pozisyona göre sırala
        all_changes.sort(key=lambda x: x['position'])
        
        # Yakın değişimleri birleştir
        if len(all_changes) > 0:
            merged_changes = [all_changes[0]]
            min_gap = window_size // 2  # Minimum aralık
            
            for change in all_changes[1:]:
                if change['position'] - merged_changes[-1]['position'] > min_gap:
                    merged_changes.append(change)
            
            # Kırılma noktası bilgilerini düzenle
            for change in merged_changes:
                pos = change['position']
                
                # Orijinal zaman indeksini kullan
                if isinstance(time_series.index, pd.DatetimeIndex):
                    breakpoint_date = time_series.index[min(pos, len(time_series) - 1)]
                    breakpoints.append({
                        'position': pos,
                        'date': breakpoint_date,
                        'old_period': change.get('old_period', 0),
                        'new_period': change.get('new_period', 0),
                        'old_strength': change.get('old_strength', 0),
                        'new_strength': change.get('new_strength', 0)
                    })
                else:
                    breakpoints.append({
                        'position': pos,
                        'old_period': change.get('old_period', 0),
                        'new_period': change.get('new_period', 0),
                        'old_strength': change.get('old_strength', 0),
                        'new_strength': change.get('new_strength', 0)
                    })
        
        return breakpoints
    
    def _find_best_period(self, time_series, max_period=None):
        """
        Zaman serisi için en uygun periyodu bul.
        
        Args:
            time_series (pd.Series): Analiz edilecek zaman serisi
            max_period (int, optional): Maksimum periyot. None ise serinin 1/3'ü kullanılır.
            
        Returns:
            int: Bulunan en uygun periyot, yoksa None
        """
        if max_period is None:
            max_period = len(time_series) // 3
        
        # Çok kısa seriler için periyot arama
        if len(time_series) < 10 or max_period < 2:
            return None
        
        # Potansiyel periyotlar (7=haftalık, 14=iki haftalık, 30=aylık, 90=çeyreklik)
        potential_periods = [p for p in [7, 14, 30, 90, 365] if p < max_period]
        
        if not potential_periods:
            potential_periods = list(range(2, min(10, max_period)))
        
        best_period = None
        best_strength = 0
        
        for period in potential_periods:
            try:
                # STL ayrıştırması
                stl = STL(time_series, seasonal=period, robust=True)
                result = stl.fit()
                
                # Mevsimsel gücü hesapla
                seasonal = result.seasonal
                residual = result.resid
                trend = result.trend
                
                if seasonal.var() == 0:
                    continue
                
                strength = 1 - (residual.var() / (seasonal + residual).var())
                
                # Daha güçlü mevsimsellik
                if strength > best_strength and strength > 0.3:
                    best_strength = strength
                    best_period = period
            except:
                continue
        
        return best_period
    
    def _calculate_seasonal_strength(self, time_series, period):
        """
        Mevsimsel gücü hesapla.
        
        Args:
            time_series (pd.Series): Analiz edilecek zaman serisi
            period (int): Mevsimsel periyot
            
        Returns:
            float: Mevsimsel güç (0-1 arası)
        """
        try:
            # STL ayrıştırması
            stl = STL(time_series, seasonal=period, robust=True)
            result = stl.fit()
            
            # Bileşenleri al
            seasonal = result.seasonal
            residual = result.resid
            
            # Mevsimsel gücü hesapla
            if seasonal.var() + residual.var() == 0:
                return 0
                
            strength = 1 - (residual.var() / (seasonal + residual).var())
            
            return max(0, min(1, strength))  # 0-1 arasına sınırla
        except:
            return 0
    
    def detect_seasonality(self, time_series):
        """
        Zaman serisinde mevsimsellik tespit et.
        Her iki model de etkinse, en yüksek güvenilirliğe sahip modelin sonucu kullanılır.
        
        Args:
            time_series (pd.Series): Analiz edilecek zaman serisi
            
        Returns:
            dict: Tespit sonuçları
        """
        results = {}
        
        # XGBoost ile tespit
        if self.use_xgboost and self.xgboost_model is not None:
            xgboost_result = self._detect_seasonality_with_xgboost(time_series)
            results['xgboost'] = xgboost_result
        
        # LSTM ile tespit
        if self.use_lstm and self.lstm_model is not None:
            lstm_result = self._detect_seasonality_with_lstm(time_series)
            results['lstm'] = lstm_result
        
        # Sonuçları birleştir
        if self.use_xgboost and self.use_lstm and 'xgboost' in results and 'lstm' in results:
            # Her iki modelin sonuçlarını karşılaştır
            if 'error' not in results['lstm']:
                xgb_confidence = results['xgboost']['confidence']
                lstm_confidence = results['lstm']['confidence']
                
                # En yüksek güvenilirliğe sahip modeli seç
                if xgb_confidence >= lstm_confidence:
                    final_result = results['xgboost']
                else:
                    final_result = results['lstm']
            else:
                final_result = results['xgboost']
        elif self.use_xgboost and 'xgboost' in results:
            final_result = results['xgboost']
        elif self.use_lstm and 'lstm' in results:
            final_result = results['lstm']
        else:
            # Hiçbir model etkin değilse
            return {
                'is_seasonal': False,
                'confidence': 0.0,
                'method': 'none',
                'error': 'No model enabled'
            }
        
        return final_result
    
    def analyze_time_series(self, time_series, detect_breakpoints=True):
        """
        Zaman serisini analiz et ve mevsimselliği ve kırılma noktalarını tespit et.
        
        Args:
            time_series (pd.Series): Analiz edilecek zaman serisi
            detect_breakpoints (bool): Kırılma noktalarını tespit edip etmeme
            
        Returns:
            dict: Analiz sonuçları
        """
        # Mevsimsellik tespiti
        seasonality_result = self.detect_seasonality(time_series)
        
        # Mevsimsel serilerde kırılma noktası tespiti
        breakpoints = []
        if seasonality_result['is_seasonal'] and detect_breakpoints:
            breakpoints = self._detect_breakpoints(time_series)
        
        # STL ayrıştırması ile dönem tespiti
        period = None
        strength = 0
        
        if seasonality_result['is_seasonal']:
            period = self._find_best_period(time_series)
            if period is not None:
                strength = self._calculate_seasonal_strength(time_series, period)
        
        # Sonuçları birleştir
        results = {
            'is_seasonal': seasonality_result['is_seasonal'],
            'confidence': seasonality_result['confidence'],
            'method': seasonality_result['method'],
            'period': period,
            'seasonal_strength': strength,
            'breakpoints': breakpoints
        }
        
        return results

# extract_features fonksiyonunu ekleyin (preprocessing.py'dan)
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