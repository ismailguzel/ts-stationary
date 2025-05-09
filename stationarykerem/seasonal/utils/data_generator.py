# utils/data_generator.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class TimeSeriesGenerator:
    """
    Sentetik zaman serisi veri üretimi için sınıf.
    Çeşitli tiplerde mevsimsel ve mevsimsel olmayan zaman serileri üretebilir.
    """
    
    def __init__(self, seed=None):
        """
        Args:
            seed (int, optional): Rastgele sayı üreteci için başlangıç değeri. Varsayılan None.
        """
        if seed is not None:
            np.random.seed(seed)
    
    def generate_time_index(self, start_date, periods, freq='D'):
        """
        Zaman serisi için zaman indeksi oluştur.
        
        Args:
            start_date (str): Başlangıç tarihi ('YYYY-MM-DD' formatında)
            periods (int): Oluşturulacak veri noktası sayısı
            freq (str): Tarih frekansı ('D' günlük, 'H' saatlik, vb.)
            
        Returns:
            pd.DatetimeIndex: Oluşturulan zaman indeksi
        """
        return pd.date_range(start=start_date, periods=periods, freq=freq)
    
    def generate_trend(self, periods, trend_type='linear', trend_strength=1.0):
        """
        Trend bileşeni oluştur.
        
        Args:
            periods (int): Veri noktası sayısı
            trend_type (str): Trend tipi ('linear', 'quadratic', 'exponential', 'logarithmic', 'no_trend')
            trend_strength (float): Trend gücü çarpanı
            
        Returns:
            np.array: Trend bileşeni
        """
        x = np.arange(periods)
        
        if trend_type == 'linear':
            trend = trend_strength * x / periods
        elif trend_type == 'quadratic':
            trend = trend_strength * (x / periods) ** 2
        elif trend_type == 'exponential':
            trend = trend_strength * (np.exp(x / periods) - 1)
        elif trend_type == 'logarithmic':
            trend = trend_strength * np.log(1 + x)
        elif trend_type == 'no_trend':
            trend = np.zeros(periods)
        else:
            raise ValueError(f"Geçersiz trend tipi: {trend_type}")
            
        return trend
    
    def generate_seasonality(self, periods, seasonal_type='single', 
                             seasonal_periods=[365, 7], 
                             seasonal_amplitudes=[1.0, 0.5],
                             seasonal_phases=[0, 0],
                             seasonal_breakpoints=None):
        """
        Mevsimsellik bileşeni oluştur.
        
        Args:
            periods (int): Veri noktası sayısı
            seasonal_type (str): Mevsimsellik tipi ('single', 'multiple', 'changing', 'no_seasonality')
            seasonal_periods (list): Mevsimsel dönemlerin listesi
            seasonal_amplitudes (list): Mevsimsel genliklerin listesi
            seasonal_phases (list): Mevsimsel fazların listesi (radian cinsinden)
            seasonal_breakpoints (list): Mevsimsel değişim noktaları ve yeni özellikler
            
        Returns:
            np.array: Mevsimsellik bileşeni
            dict: Mevsimsellik bilgisi
        """
        x = np.arange(periods)
        seasonality = np.zeros(periods)
        seasonality_info = {
            'is_seasonal': seasonal_type != 'no_seasonality',
            'type': seasonal_type,
            'periods': [],
            'amplitudes': [],
            'phases': [],
            'breakpoints': []
        }
        
        if seasonal_type == 'no_seasonality':
            return seasonality, seasonality_info
            
        if seasonal_type == 'single':
            period = seasonal_periods[0]
            amplitude = seasonal_amplitudes[0]
            phase = seasonal_phases[0]
            
            seasonality = amplitude * np.sin(2 * np.pi * x / period + phase)
            
            seasonality_info['periods'].append(period)
            seasonality_info['amplitudes'].append(amplitude)
            seasonality_info['phases'].append(phase)
            
        elif seasonal_type == 'multiple':
            for period, amplitude, phase in zip(seasonal_periods, seasonal_amplitudes, seasonal_phases):
                seasonality += amplitude * np.sin(2 * np.pi * x / period + phase)
                
                seasonality_info['periods'].append(period)
                seasonality_info['amplitudes'].append(amplitude)
                seasonality_info['phases'].append(phase)
                
        elif seasonal_type == 'changing':
            if seasonal_breakpoints is None:
                # Varsayılan olarak serinin ortasında bir değişim
                seasonal_breakpoints = [{
                    'position': periods // 2,
                    'new_period': seasonal_periods[0] * 1.5,
                    'new_amplitude': seasonal_amplitudes[0] * 0.8,
                    'new_phase': seasonal_phases[0] + np.pi / 4
                }]
            
            current_period = seasonal_periods[0]
            current_amplitude = seasonal_amplitudes[0]
            current_phase = seasonal_phases[0]
            
            # Başlangıçta kullanılan mevsimsellik özelliklerini kaydet
            seasonality_info['periods'].append(current_period)
            seasonality_info['amplitudes'].append(current_amplitude)
            seasonality_info['phases'].append(current_phase)
            
            last_pos = 0
            
            # Her breakpoint için dizi dilimlerini hesapla ve doldur
            for breakpoint in sorted(seasonal_breakpoints, key=lambda x: x['position']):
                pos = breakpoint['position']
                
                # Breakpoint bilgisini kaydet
                seasonality_info['breakpoints'].append({
                    'position': pos,
                    'old_period': current_period,
                    'new_period': breakpoint['new_period'],
                    'old_amplitude': current_amplitude,
                    'new_amplitude': breakpoint['new_amplitude'],
                    'old_phase': current_phase,
                    'new_phase': breakpoint['new_phase']
                })
                
                # Breakpoint'e kadar mevcut mevsimsellik ile doldur
                seasonality[last_pos:pos] = current_amplitude * np.sin(
                    2 * np.pi * np.arange(last_pos, pos) / current_period + current_phase
                )
                
                # Mevsimsellik parametrelerini güncelle
                current_period = breakpoint['new_period']
                current_amplitude = breakpoint['new_amplitude']
                current_phase = breakpoint['new_phase']
                
                # Yeni eklenen mevsimsellik özelliklerini kaydet
                seasonality_info['periods'].append(current_period)
                seasonality_info['amplitudes'].append(current_amplitude)
                seasonality_info['phases'].append(current_phase)
                
                last_pos = pos
            
            # Son breakpoint'ten dizinin sonuna kadar doldur
            seasonality[last_pos:] = current_amplitude * np.sin(
                2 * np.pi * np.arange(last_pos, periods) / current_period + current_phase
            )
            
        else:
            raise ValueError(f"Geçersiz mevsimsellik tipi: {seasonal_type}")
            
        return seasonality, seasonality_info
    
    def generate_noise(self, periods, noise_level=0.1, noise_type='gaussian'):
        """
        Gürültü bileşeni oluştur.
        
        Args:
            periods (int): Veri noktası sayısı
            noise_level (float): Gürültü seviyesi
            noise_type (str): Gürültü tipi ('gaussian', 'uniform')
            
        Returns:
            np.array: Gürültü bileşeni
        """
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_level, periods)
        elif noise_type == 'uniform':
            noise = np.random.uniform(-noise_level, noise_level, periods)
        else:
            raise ValueError(f"Geçersiz gürültü tipi: {noise_type}")
            
        return noise
    
    def generate_time_series(self, periods=730, start_date='2020-01-01', freq='D',
                              trend_type='linear', trend_strength=1.0,
                              seasonal_type='single', seasonal_periods=[365, 7],
                              seasonal_amplitudes=[1.0, 0.5], seasonal_phases=[0, 0],
                              seasonal_breakpoints=None,
                              noise_level=0.1, noise_type='gaussian',
                              add_outliers=False, outlier_fraction=0.01, outlier_scale=3.0):
        """
        Tam bir zaman serisi oluştur.
        
        Args:
            periods (int): Veri noktası sayısı
            start_date (str): Başlangıç tarihi ('YYYY-MM-DD' formatında)
            freq (str): Tarih frekansı ('D' günlük, 'H' saatlik, vb.)
            trend_type (str): Trend tipi
            trend_strength (float): Trend gücü çarpanı
            seasonal_type (str): Mevsimsellik tipi
            seasonal_periods (list): Mevsimsel dönemlerin listesi
            seasonal_amplitudes (list): Mevsimsel genliklerin listesi
            seasonal_phases (list): Mevsimsel fazların listesi
            seasonal_breakpoints (list): Mevsimsel değişim noktaları
            noise_level (float): Gürültü seviyesi
            noise_type (str): Gürültü tipi
            add_outliers (bool): Aykırı değerler eklenip eklenmeyeceği
            outlier_fraction (float): Aykırı değer oranı
            outlier_scale (float): Aykırı değer ölçeği
            
        Returns:
            pd.Series: Oluşturulan zaman serisi
            dict: Zaman serisi özellikleri
        """
        # Zaman indeksi oluştur
        index = self.generate_time_index(start_date, periods, freq)
        
        # Bileşenleri oluştur
        trend = self.generate_trend(periods, trend_type, trend_strength)
        seasonality, seasonality_info = self.generate_seasonality(
            periods, seasonal_type, seasonal_periods, 
            seasonal_amplitudes, seasonal_phases, seasonal_breakpoints
        )
        noise = self.generate_noise(periods, noise_level, noise_type)
        
        # Bileşenleri birleştir
        data = trend + seasonality + noise
        
        # Aykırı değerler ekle
        if add_outliers:
            outlier_count = int(periods * outlier_fraction)
            outlier_indices = np.random.choice(periods, outlier_count, replace=False)
            outlier_signs = np.random.choice([-1, 1], outlier_count)
            data[outlier_indices] += outlier_signs * outlier_scale * np.abs(data[outlier_indices])
        
        # Zaman serisi oluştur
        time_series = pd.Series(data, index=index)
        
        # Metadata sözlüğü oluştur
        metadata = {
            'trend': {
                'type': trend_type,
                'strength': trend_strength
            },
            'seasonality': seasonality_info,
            'noise': {
                'level': noise_level,
                'type': noise_type
            },
            'outliers': {
                'added': add_outliers,
                'fraction': outlier_fraction if add_outliers else 0,
                'scale': outlier_scale if add_outliers else 0
            }
        }
        
        return time_series, metadata

    def generate_batch(self, count=1000, balanced=True, **kwargs):
        """
        Birden fazla zaman serisi oluştur.
        
        Args:
            count (int): Oluşturulacak zaman serisi sayısı
            balanced (bool): Mevsimsel ve mevsimsel olmayan serilerin dengeli olup olmayacağı
            **kwargs: generate_time_series fonksiyonuna geçirilecek diğer parametreler
            
        Returns:
            list: Zaman serilerinin listesi
            list: Metadata listesi
        """
        time_series_list = []
        metadata_list = []
        
        # Mevsimsel ve mevsimsel olmayan serilerin sayısını belirle
        if balanced:
            seasonal_count = count // 2
            non_seasonal_count = count - seasonal_count
        else:
            # Rastgele oran (mevsimsel seriler %30-%70 arasında)
            seasonal_ratio = np.random.uniform(0.3, 0.7)
            seasonal_count = int(count * seasonal_ratio)
            non_seasonal_count = count - seasonal_count
        
        # Mevsimsel zaman serileri oluştur
        for i in range(seasonal_count):
            # Rastgele mevsimsellik parametreleri oluştur
            seasonal_type = np.random.choice(['single', 'multiple', 'changing'])
            
            # Rastgele mevsimsel dönemler (günlük, haftalık, aylık, yıllık gibi)
            potential_periods = [7, 30, 90, 365]  # Haftalık, aylık, çeyreklik, yıllık
            if seasonal_type == 'single':
                periods = [np.random.choice(potential_periods)]
            else:
                # Multiple ya da changing durumunda 1-3 farklı dönem seç
                num_periods = np.random.randint(1, min(4, len(potential_periods) + 1))
                periods = sorted(np.random.choice(potential_periods, num_periods, replace=False))
            
            # Rastgele genlikler
            amplitudes = [np.random.uniform(0.5, 2.0) for _ in range(len(periods))]
            
            # Rastgele fazlar
            phases = [np.random.uniform(0, 2 * np.pi) for _ in range(len(periods))]
            
            # Mevsimsel kırılma noktaları (changing durumunda)
            breakpoints = None
            if seasonal_type == 'changing':
                num_breakpoints = np.random.randint(1, 4)  # 1-3 kırılma noktası
                total_periods = kwargs.get('periods', 730)
                
                # Kırılma noktalarını hesapla (10%-90% aralığında)
                positions = sorted(np.random.choice(
                    range(int(total_periods * 0.1), int(total_periods * 0.9)),
                    num_breakpoints,
                    replace=False
                ))
                
                breakpoints = []
                for pos in positions:
                    # Yeni dönem (eski dönemin %50-%150'si arasında)
                    new_period = periods[0] * np.random.uniform(0.5, 1.5)
                    # Yeni genlik (eski genliğin %50-%150'si arasında)
                    new_amplitude = amplitudes[0] * np.random.uniform(0.5, 1.5)
                    # Yeni faz (0-2π arasında)
                    new_phase = np.random.uniform(0, 2 * np.pi)
                    
                    breakpoints.append({
                        'position': pos,
                        'new_period': new_period,
                        'new_amplitude': new_amplitude,
                        'new_phase': new_phase
                    })
            
            # Zaman serisi oluştur
            ts, meta = self.generate_time_series(
                seasonal_type=seasonal_type,
                seasonal_periods=periods,
                seasonal_amplitudes=amplitudes,
                seasonal_phases=phases,
                seasonal_breakpoints=breakpoints,
                **kwargs
            )
            
            time_series_list.append(ts)
            metadata_list.append(meta)
        
        # Mevsimsel olmayan zaman serileri oluştur
        for i in range(non_seasonal_count):
            # Rastgele trend parametreleri
            trend_type = np.random.choice(['linear', 'quadratic', 'exponential', 'logarithmic', 'no_trend'])
            trend_strength = np.random.uniform(0.0, 2.0)
            
            # Zaman serisi oluştur
            ts, meta = self.generate_time_series(
                seasonal_type='no_seasonality',
                trend_type=trend_type,
                trend_strength=trend_strength,
                **kwargs
            )
            
            time_series_list.append(ts)
            metadata_list.append(meta)
        
        return time_series_list, metadata_list