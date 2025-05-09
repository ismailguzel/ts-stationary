# creator_ex.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse

def create_seasonal_series(start_date='2020-01-01', periods=730, freq='D',
                          trend_type='linear', trend_strength=0.5,
                          seasonal_period1=365, seasonal_amplitude1=1.0,
                          seasonal_period2=None, seasonal_amplitude2=0.0,
                          breakpoint_position=None, new_period=None, new_amplitude=None,
                          noise_level=0.1, output_file='example_seasonal.csv'):
    """
    Bilinen mevsimsellik özelliklerine sahip zaman serisi oluştur.
    
    Args:
        start_date (str): Başlangıç tarihi ('YYYY-MM-DD' formatında)
        periods (int): Oluşturulacak veri noktası sayısı
        freq (str): Zaman frekansı ('D' günlük, 'H' saatlik, vb.)
        trend_type (str): Trend tipi ('linear', 'quadratic', 'exponential', 'no_trend')
        trend_strength (float): Trend gücü çarpanı
        seasonal_period1 (int): İlk mevsimsel dönem
        seasonal_amplitude1 (float): İlk mevsimsel genlik
        seasonal_period2 (int, optional): İkinci mevsimsel dönem
        seasonal_amplitude2 (float): İkinci mevsimsel genlik
        breakpoint_position (int, optional): Kırılma noktası pozisyonu
        new_period (float, optional): Kırılma sonrası yeni dönem
        new_amplitude (float, optional): Kırılma sonrası yeni genlik
        noise_level (float): Gürültü seviyesi
        output_file (str): Çıktı CSV dosyası
        
    Returns:
        pd.Series: Oluşturulan zaman serisi
    """
    # Zaman indeksi oluştur
    date_range = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Trend bileşeni
    x = np.arange(periods)
    
    if trend_type == 'linear':
        trend = trend_strength * x / periods
    elif trend_type == 'quadratic':
        trend = trend_strength * (x / periods) ** 2
    elif trend_type == 'exponential':
        trend = trend_strength * (np.exp(x / periods) - 1)
    elif trend_type == 'no_trend':
        trend = np.zeros(periods)
    else:
        raise ValueError(f"Geçersiz trend tipi: {trend_type}")
    
    # Mevsimsellik bileşeni
    seasonality = np.zeros(periods)
    
    # İlk mevsimsel dönem - herkese uygula
    seasonality += seasonal_amplitude1 * np.sin(2 * np.pi * x / seasonal_period1)
    
    # İkinci mevsimsel dönem (opsiyonel)
    if seasonal_period2 is not None and seasonal_amplitude2 > 0:
        seasonality += seasonal_amplitude2 * np.sin(2 * np.pi * x / seasonal_period2)
    
    # Kırılma noktası (breakpoint) varsa uygula
    if breakpoint_position is not None and new_period is not None:
        # Kırılma noktasından önceki orijinal mevsimsellik bileşeni
        seasonality_before = seasonality[:breakpoint_position].copy()
        
        # Kırılma noktasından sonraki kısım için yeni mevsimsellik parametreleri
        x_after = np.arange(periods - breakpoint_position)
        seasonality_after = np.zeros(periods - breakpoint_position)
        
        # Yeni dönem ve genlik ile mevsimsellik oluştur
        new_amp = new_amplitude if new_amplitude is not None else seasonal_amplitude1
        seasonality_after = new_amp * np.sin(2 * np.pi * x_after / new_period)
        
        # İkinci mevsimsel dönem de varsa ekle
        if seasonal_period2 is not None and seasonal_amplitude2 > 0:
            seasonality_after += seasonal_amplitude2 * np.sin(2 * np.pi * x_after / seasonal_period2)
        
        # Kırılma öncesi ve sonrası birleştir
        seasonality = np.concatenate([seasonality_before, seasonality_after])
    
    # Gürültü bileşeni
    noise = np.random.normal(0, noise_level, periods)
    
    # Tüm bileşenleri birleştir
    data = trend + seasonality + noise
    
    # Zaman serisi oluştur
    time_series = pd.Series(data, index=date_range)
    
    # Veriyi DataFrame olarak hazırla
    df = pd.DataFrame({
        'date': time_series.index,
        'value': time_series.values
    })
    
    # CSV olarak kaydet
    df.to_csv(output_file, index=False)
    print(f"Zaman serisi CSV dosyası oluşturuldu: {output_file}")
    
    # Bilinen özellikleri yazdır
    print("\nZaman Serisi Özellikleri:")
    print(f"Mevsimsellik: VAR")
    print(f"Ana Periyot: {seasonal_period1}")
    if seasonal_period2 is not None and seasonal_amplitude2 > 0:
        print(f"İkincil Periyot: {seasonal_period2}")
    
    if breakpoint_position is not None:
        breakpoint_date = date_range[breakpoint_position]
        print(f"\nKırılma Noktası:")
        print(f"Pozisyon: {breakpoint_position}")
        print(f"Tarih: {breakpoint_date.strftime('%Y-%m-%d')}")
        print(f"Periyot Değişimi: {seasonal_period1} => {new_period}")
        if new_amplitude is not None:
            print(f"Genlik Değişimi: {seasonal_amplitude1} => {new_amplitude}")
    
    return time_series

def create_non_seasonal_series(start_date='2020-01-01', periods=730, freq='D',
                              trend_type='linear', trend_strength=1.0,
                              noise_level=0.3, output_file='example_non_seasonal.csv'):
    """
    Mevsimsel olmayan zaman serisi oluştur.
    
    Args:
        start_date (str): Başlangıç tarihi ('YYYY-MM-DD' formatında)
        periods (int): Oluşturulacak veri noktası sayısı
        freq (str): Zaman frekansı ('D' günlük, 'H' saatlik, vb.)
        trend_type (str): Trend tipi ('linear', 'quadratic', 'exponential', 'no_trend')
        trend_strength (float): Trend gücü çarpanı
        noise_level (float): Gürültü seviyesi
        output_file (str): Çıktı CSV dosyası
        
    Returns:
        pd.Series: Oluşturulan zaman serisi
    """
    # Zaman indeksi oluştur
    date_range = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Trend bileşeni
    x = np.arange(periods)
    
    if trend_type == 'linear':
        trend = trend_strength * x / periods
    elif trend_type == 'quadratic':
        trend = trend_strength * (x / periods) ** 2
    elif trend_type == 'exponential':
        trend = trend_strength * (np.exp(x / periods) - 1)
    elif trend_type == 'no_trend':
        trend = np.zeros(periods)
    else:
        raise ValueError(f"Geçersiz trend tipi: {trend_type}")
    
    # Gürültü bileşeni
    noise = np.random.normal(0, noise_level, periods)
    
    # Tüm bileşenleri birleştir
    data = trend + noise
    
    # Zaman serisi oluştur
    time_series = pd.Series(data, index=date_range)
    
    # Veriyi DataFrame olarak hazırla
    df = pd.DataFrame({
        'date': time_series.index,
        'value': time_series.values
    })
    
    # CSV olarak kaydet
    df.to_csv(output_file, index=False)
    print(f"Zaman serisi CSV dosyası oluşturuldu: {output_file}")
    
    # Bilinen özellikleri yazdır
    print("\nZaman Serisi Özellikleri:")
    print(f"Mevsimsellik: YOK")
    print(f"Trend Tipi: {trend_type}")
    
    return time_series

def plot_time_series(time_series, title=None, breakpoint_position=None):
    """
    Zaman serisini çizdir.
    
    Args:
        time_series (pd.Series): Çizdirilecek zaman serisi
        title (str, optional): Grafik başlığı
        breakpoint_position (int, optional): Kırılma noktası pozisyonu
    """
    plt.figure(figsize=(12, 6))
    plt.plot(time_series.index, time_series.values)
    
    if breakpoint_position is not None:
        plt.axvline(x=time_series.index[breakpoint_position], color='r', linestyle='--', alpha=0.7)
        plt.text(time_series.index[breakpoint_position], time_series.max() * 1.1, 
                 "Breakpoint", rotation=90, verticalalignment='bottom')
    
    if title:
        plt.title(title)
    plt.xlabel('Tarih')
    plt.ylabel('Değer')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test amaçlı zaman serisi CSV dosyası oluştur')
    parser.add_argument('--type', type=str, default='seasonal', choices=['seasonal', 'non_seasonal'],
                      help='Zaman serisi tipi (seasonal veya non_seasonal)')
    
    args = parser.parse_args()
    
    if args.type == 'seasonal':
        # 1. Kırılma noktalı mevsimsel seri
        ts1 = create_seasonal_series(
            start_date='2020-01-01',
            periods=730,  # 2 yıl
            seasonal_period1=365,  # Yıllık dönem
            seasonal_amplitude1=1.0,
            breakpoint_position=300,  # 300. günde kırılma
            new_period=180,  # Yeni dönem 180 gün
            new_amplitude=1.5,  # Yeni genlik
            trend_type='linear',
            trend_strength=0.5,
            noise_level=0.1,
            output_file='example_seasonal_with_breakpoint.csv'
        )
        
        # Grafiği göster
        plot_time_series(ts1, title="Kırılma Noktalı Mevsimsel Seri", breakpoint_position=300)
        
        # 2. Çoklu mevsimsellikli seri
        ts2 = create_seasonal_series(
            start_date='2020-01-01',
            periods=730,
            seasonal_period1=365,  # Yıllık dönem
            seasonal_amplitude1=1.0,
            seasonal_period2=7,  # Haftalık dönem
            seasonal_amplitude2=0.3,
            trend_type='quadratic',
            trend_strength=0.3,
            noise_level=0.15,
            output_file='example_multiple_seasonal.csv'
        )
        
        # Grafiği göster
        plot_time_series(ts2, title="Çoklu Mevsimsellikli Seri")
        
    else:
        # Mevsimsel olmayan seri
        ts3 = create_non_seasonal_series(
            start_date='2020-01-01',
            periods=730,
            trend_type='exponential',
            trend_strength=1.0,
            noise_level=0.3,
            output_file='example_non_seasonal.csv'
        )
        
        # Grafiği göster
        plot_time_series(ts3, title="Mevsimsel Olmayan Seri")
    
    print("\nCSV dosyaları başarıyla oluşturuldu!")
    print("Bu dosyaları 'app.py' uygulamasına yükleyerek mevsimsellik analizi yapabilirsiniz.")