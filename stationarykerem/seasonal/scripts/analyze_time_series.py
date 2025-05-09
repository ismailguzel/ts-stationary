# scripts/analyze_time_series.py
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Modül yolunu ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seasonality_detector import SeasonalityDetector
from utils.visualization import (
    plot_time_series, 
    plot_seasonality_components, 
    plot_time_series_with_breakpoints
)

def load_time_series(file_path, date_col=None, value_col=None, format=None):
    """
    Dosyadan zaman serisi yükle.
    
    Args:
        file_path (str): Veri dosyası yolu
        date_col (str, optional): Tarih sütunu adı. None ise ilk sütun kullanılır.
        value_col (str, optional): Değer sütunu adı. None ise ikinci sütun kullanılır.
        format (str, optional): Tarih biçimi. None ise otomatik tespit.
        
    Returns:
        pd.Series: Yüklenen zaman serisi
    """
    # Dosya uzantısını kontrol et
    _, ext = os.path.splitext(file_path)
    
    if ext.lower() == '.csv':
        df = pd.read_csv(file_path)
    elif ext.lower() in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Desteklenmeyen dosya formatı: {ext}")
    
    # Sütunları belirle
    if date_col is None:
        date_col = df.columns[0]
    
    if value_col is None:
        value_col = df.columns[1]
    
    # Tarih sütununu dönüştür
    if format is not None:
        df[date_col] = pd.to_datetime(df[date_col], format=format)
    else:
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Zaman serisini oluştur
    time_series = pd.Series(df[value_col].values, index=df[date_col])
    
    # Zaman serisini sırala
    time_series = time_series.sort_index()
    
    return time_series

def save_results(results, output_file):
    """
    Analiz sonuçlarını kaydet.
    
    Args:
        results (dict): Analiz sonuçları
        output_file (str): Çıktı dosyası yolu
    """
    # Datetime nesnelerini stringe dönüştür
    for bp in results.get('breakpoints', []):
        if 'date' in bp and isinstance(bp['date'], (pd.Timestamp, datetime)):
            bp['date'] = bp['date'].strftime('%Y-%m-%d')
    
    # JSON olarak kaydet
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

def main(args):
    print(f"Zaman serisi analiz ediliyor: {args.input}")
    
    # Zaman serisini yükle
    time_series = load_time_series(
        args.input, 
        date_col=args.date_column,
        value_col=args.value_column,
        format=args.date_format
    )
    
    print(f"Zaman serisi yüklendi: {len(time_series)} veri noktası")
    
    # Mevsimsellik tespit edicisi oluştur
    detector = SeasonalityDetector(
        use_xgboost=args.use_xgboost,
        use_lstm=args.use_lstm,
        confidence_threshold=args.threshold
    )
    
    # Zaman serisini analiz et
    results = detector.analyze_time_series(time_series, detect_breakpoints=args.detect_breakpoints)
    
    # Sonuçları yazdır
    print("\nAnaliz Sonuçları:")
    print(f"Mevsimsellik: {'Var' if results['is_seasonal'] else 'Yok'}")
    print(f"Güven Skoru: {results['confidence']:.4f}")
    print(f"Kullanılan Model: {results['method']}")
    
    if results['period'] is not None:
        print(f"Tespit Edilen Periyot: {results['period']}")
        print(f"Mevsimsel Güç: {results['seasonal_strength']:.4f}")
    
    if results['breakpoints']:
        print("\nKırılma Noktaları:")
        for i, bp in enumerate(results['breakpoints']):
            print(f"  {i+1}. Nokta:")
            print(f"     Pozisyon: {bp['position']}")
            if 'date' in bp:
                print(f"     Tarih: {bp['date']}")
            if 'old_period' in bp and 'new_period' in bp:
                print(f"     Periyot Değişimi: {bp['old_period']:.1f} → {bp['new_period']:.1f}")
            if 'old_strength' in bp and 'new_strength' in bp:
                print(f"     Güç Değişimi: {bp['old_strength']:.2f} → {bp['new_strength']:.2f}")
    
    # Sonuçları kaydet
    if args.output:
        save_results(results, args.output)
        print(f"\nSonuçlar kaydedildi: {args.output}")
    
    # Grafikleri çiz
    if args.plot:
        print("\nGrafikler oluşturuluyor...")
        
        # Çıktı klasörü
        output_dir = os.path.dirname(args.output) if args.output else '.'
        
        # Dosya adı kökü
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        
        # Zaman serisi grafiği
        plot_time_series(
            time_series,
            title=f"Time Series: {base_name}" + (" (Seasonal)" if results['is_seasonal'] else " (Non-Seasonal)"),
            save_path=os.path.join(output_dir, f"{base_name}_time_series.png") if args.save_plots else None
        )
        
        # Mevsimsellik bileşenleri grafiği
        if results['is_seasonal'] and results['period'] is not None:
            plot_seasonality_components(
                time_series,
                period=results['period'],
                title=f"Seasonal Components (Period={results['period']})",
                save_path=os.path.join(output_dir, f"{base_name}_components.png") if args.save_plots else None
            )
        
        # Kırılma noktaları grafiği
        if results['breakpoints']:
            plot_time_series_with_breakpoints(
                time_series,
                breakpoints=results['breakpoints'],
                title=f"Time Series with Breakpoints",
                save_path=os.path.join(output_dir, f"{base_name}_breakpoints.png") if args.save_plots else None
            )
    
    print("\nAnaliz tamamlandı!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Zaman serisi analizi')
    parser.add_argument('input', type=str, help='Girdi dosyası yolu (CSV veya Excel)')
    parser.add_argument('--date-column', type=str, help='Tarih sütunu adı (varsayılan: ilk sütun)')
    parser.add_argument('--value-column', type=str, help='Değer sütunu adı (varsayılan: ikinci sütun)')
    parser.add_argument('--date-format', type=str, help='Tarih biçimi (örn. %%Y-%%m-%%d)')
    parser.add_argument('--output', type=str, help='Çıktı JSON dosyası yolu')
    parser.add_argument('--threshold', type=float, default=0.6, help='Mevsimsellik güven eşiği (0-1 arası)')
    parser.add_argument('--use-xgboost', action='store_true', help='XGBoost modelini kullan')
    parser.add_argument('--use-lstm', action='store_true', help='LSTM modelini kullan')
    parser.add_argument('--detect-breakpoints', action='store_true', help='Kırılma noktalarını tespit et')
    parser.add_argument('--plot', action='store_true', help='Grafikleri çiz')
    parser.add_argument('--save-plots', action='store_true', help='Grafikleri kaydet')
    
    args = parser.parse_args()
    
    # Varsayılan model seçimi
    if not args.use_xgboost and not args.use_lstm:
        args.use_xgboost = True  # Varsayılan olarak XGBoost kullan
    
    main(args)