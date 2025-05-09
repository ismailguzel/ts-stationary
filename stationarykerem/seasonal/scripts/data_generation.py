# scripts/data_generation.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import argparse
import sys

# Modül yolunu ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_generator import TimeSeriesGenerator
from utils.preprocessing import create_seasonality_labels, create_breakpoint_labels
from utils.visualization import plot_multiple_time_series, plot_time_series_with_breakpoints

def main(args):
    print("Sentetik zaman serisi verileri oluşturuluyor...")
    
    # Çıktı klasörlerini oluştur
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    raw_dir = os.path.join(data_dir, 'raw')
    processed_dir = os.path.join(data_dir, 'processed')
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Sabit sayı üreteci ayarla
    np.random.seed(args.seed)
    
    # Zaman serisi üreteci
    generator = TimeSeriesGenerator(seed=args.seed)
    
    # Veri üret
    time_series_list, metadata_list = generator.generate_batch(
        count=args.count,
        balanced=args.balanced,
        periods=args.periods,
        start_date=args.start_date,
        freq=args.freq,
        noise_level=args.noise_level
    )
    
    print(f"Toplam {len(time_series_list)} zaman serisi üretildi.")
    
    # Etiketleri oluştur
    seasonality_labels = create_seasonality_labels(metadata_list)
    breakpoint_labels = create_breakpoint_labels(metadata_list)
    
    # İstatistikler göster
    print(f"Mevsimsel seriler: {seasonality_labels['is_seasonal'].sum()}")
    print(f"Mevsimsel olmayan seriler: {len(seasonality_labels) - seasonality_labels['is_seasonal'].sum()}")
    print(f"Kırılma noktası içeren seriler: {seasonality_labels['has_breakpoints'].sum()}")
    
    # Etiketleri kaydet
    seasonality_labels.to_csv(os.path.join(processed_dir, 'seasonality_labels.csv'), index=False)
    breakpoint_labels.to_csv(os.path.join(processed_dir, 'breakpoint_labels.csv'), index=False)
    
    # Zaman serilerini kaydet
    print("Zaman serileri kaydediliyor...")
    
    # Tekli CSV olarak kaydet
    all_series_df = pd.DataFrame()
    
    for i, ts in enumerate(tqdm(time_series_list)):
        # Seri ID'sini ekle
        ts_df = pd.DataFrame({
            'ds': ts.index,
            'y': ts.values,
            'series_id': i
        })
        
        # Tüm serilere ekle
        all_series_df = pd.concat([all_series_df, ts_df])
        
        # Ayrı CSV olarak kaydet
        if args.save_individual:
            ts_df.to_csv(os.path.join(raw_dir, f'series_{i}.csv'), index=False)
    
    # Tüm serileri tek CSV'ye kaydet
    all_series_df.to_csv(os.path.join(raw_dir, 'all_series.csv'), index=False)
    
    # Metadata'yı pickle olarak kaydet
    with open(os.path.join(processed_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata_list, f)
    
    # Eğitim/test bölünmesi için indeksleri kaydet
    indices = np.arange(args.count)
    np.random.shuffle(indices)
    
    train_indices = indices[:args.count // 2]
    test_indices = indices[args.count // 2:]
    
    np.save(os.path.join(processed_dir, 'train_indices.npy'), train_indices)
    np.save(os.path.join(processed_dir, 'test_indices.npy'), test_indices)
    
    print(f"Eğitim seti boyutu: {len(train_indices)}")
    print(f"Test seti boyutu: {len(test_indices)}")
    
    # Örnek grafikleri çiz ve kaydet
    if args.plot_samples:
        print("Örnek grafikler oluşturuluyor...")
        
        # Rastgele 12 örnek seri seç
        sample_indices = np.random.choice(len(time_series_list), min(12, len(time_series_list)), replace=False)
        sample_series = [time_series_list[i] for i in sample_indices]
        
        # Başlıkları oluştur
        titles = []
        for i in sample_indices:
            is_seasonal = metadata_list[i]['seasonality']['is_seasonal']
            has_bp = len(metadata_list[i]['seasonality'].get('breakpoints', [])) > 0
            titles.append(f"Series {i}: {'Seasonal' if is_seasonal else 'Non-Seasonal'}{' with BP' if has_bp else ''}")
        
        # Grafik kaydetme yolu
        plot_dir = os.path.join(processed_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Örnek serileri çiz
        plot_multiple_time_series(
            sample_series, 
            titles=titles, 
            max_cols=3, 
            save_path=os.path.join(plot_dir, 'sample_series.png')
        )
        
        # Kırılma noktalarıyla örnekler çiz
        for i, meta in enumerate(metadata_list):
            if meta['seasonality']['is_seasonal'] and meta['seasonality'].get('breakpoints'):
                # Kırılma noktaları olan ilk 5 seriyi çiz
                if i < 5:
                    plot_time_series_with_breakpoints(
                        time_series_list[i],
                        breakpoints=meta['seasonality']['breakpoints'],
                        title=f"Series {i} with Breakpoints",
                        save_path=os.path.join(plot_dir, f'series_{i}_breakpoints.png')
                    )
    
    print("Veri üretimi tamamlandı!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sentetik zaman serisi verisi üret')
    parser.add_argument('--count', type=int, default=1000, help='Oluşturulacak zaman serisi sayısı')
    parser.add_argument('--balanced', action='store_true', help='Mevsimsel/mevsimsel olmayan serilerin dengeli olup olmayacağı')
    parser.add_argument('--periods', type=int, default=730, help='Her zaman serisindeki veri noktası sayısı')
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='Başlangıç tarihi')
    parser.add_argument('--freq', type=str, default='D', help='Zaman serisi frekansı')
    parser.add_argument('--noise-level', type=float, default=0.1, help='Gürültü seviyesi')
    parser.add_argument('--seed', type=int, default=42, help='Rastgele sayı üreteci tohumu')
    parser.add_argument('--save-individual', action='store_true', help='Her zaman serisini ayrı dosya olarak kaydet')
    parser.add_argument('--plot-samples', action='store_true', help='Örnek grafikleri çiz ve kaydet')
    
    args = parser.parse_args()
    main(args)