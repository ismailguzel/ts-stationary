# scripts/main.py
import os
import sys
import argparse
import subprocess
import time

def run_command(command, description=None):
    """
    Komut çalıştır ve çıktıyı göster.
    
    Args:
        command (str): Çalıştırılacak komut
        description (str, optional): Komut açıklaması
    
    Returns:
        int: Komut çıkış kodu
    """
    if description:
        print(f"\n{'=' * 80}")
        print(f"  {description}")
        print(f"{'=' * 80}")
    
    print(f"\n> {command}\n")
    
    return subprocess.call(command, shell=True)

def main(args):
    # Proje ana klasörünü belirle
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Betiklerin yolu
    scripts_dir = os.path.join(project_dir, "scripts")
    
    # Başlangıç zamanı
    start_time = time.time()
    
    # 1. Veri üretimi
    if args.generate_data:
        data_cmd = f"python {os.path.join(scripts_dir, 'data_generation.py')} --count {args.count} --plot-samples"
        if args.balanced:
            data_cmd += " --balanced"
        
        exit_code = run_command(data_cmd, "Veri Üretimi")
        if exit_code != 0:
            print("Hata: Veri üretimi başarısız!")
            return exit_code
    
    # 2. XGBoost modeli eğitimi
    if args.train_xgboost:
        xgb_cmd = f"python {os.path.join(scripts_dir, 'train_xgboost_model.py')} --scale"
        if args.grid_search:
            xgb_cmd += " --grid-search"
            
        exit_code = run_command(xgb_cmd, "XGBoost Modeli Eğitimi")
        if exit_code != 0:
            print("Hata: XGBoost modeli eğitimi başarısız!")
            return exit_code
    
    # 3. LSTM modeli eğitimi
    if args.train_lstm:
        lstm_cmd = f"python {os.path.join(scripts_dir, 'train_lstm_model.py')} --seq-length {args.seq_length} --epochs {args.epochs}"
        
        exit_code = run_command(lstm_cmd, "LSTM Modeli Eğitimi")
        if exit_code != 0:
            print("Hata: LSTM modeli eğitimi başarısız!")
            return exit_code
    
    # 4. Model değerlendirmesi
    if args.evaluate:
        eval_cmd = f"python {os.path.join(scripts_dir, 'evaluate_models.py')} --seq-length {args.seq_length}"
        
        exit_code = run_command(eval_cmd, "Model Değerlendirmesi")
        if exit_code != 0:
            print("Hata: Model değerlendirmesi başarısız!")
            return exit_code
    
    # 5. Örnek analiz
    if args.analyze_example:
        # Örnek veri yolu
        raw_dir = os.path.join(project_dir, "data", "raw")
        example_file = os.path.join(raw_dir, "all_series.csv")
        
        if not os.path.exists(example_file):
            print(f"Hata: Örnek veri dosyası bulunamadı: {example_file}")
            return 1
        
        # Örnek zaman serisi oluştur
        import pandas as pd
        all_series = pd.read_csv(example_file)
        
        # Mevsimsel ve mevsimsel olmayan birer örnek seç
        from utils.preprocessing import create_seasonality_labels
        import pickle
        
        # Metadata'yı yükle
        with open(os.path.join(project_dir, "data", "processed", "metadata.pkl"), 'rb') as f:
            metadata_list = pickle.load(f)
        
        # Etiketleri oluştur
        seasonality_labels = create_seasonality_labels(metadata_list)
        
        # Mevsimsel örnek
        seasonal_id = seasonality_labels[seasonality_labels['is_seasonal'] == True]['series_id'].iloc[0]
        seasonal_series = all_series[all_series['series_id'] == seasonal_id]
        seasonal_file = os.path.join(raw_dir, "example_seasonal.csv")
        seasonal_series[['ds', 'y']].to_csv(seasonal_file, index=False)
        
        # Mevsimsel olmayan örnek
        non_seasonal_id = seasonality_labels[seasonality_labels['is_seasonal'] == False]['series_id'].iloc[0]
        non_seasonal_series = all_series[all_series['series_id'] == non_seasonal_id]
        non_seasonal_file = os.path.join(raw_dir, "example_non_seasonal.csv")
        non_seasonal_series[['ds', 'y']].to_csv(non_seasonal_file, index=False)
        
        # Analiz komutları
        seasonal_cmd = f"python {os.path.join(scripts_dir, 'analyze_time_series.py')} {seasonal_file} --output {os.path.join(raw_dir, 'example_seasonal_results.json')} --use-xgboost --detect-breakpoints --plot --save-plots"
        non_seasonal_cmd = f"python {os.path.join(scripts_dir, 'analyze_time_series.py')} {non_seasonal_file} --output {os.path.join(raw_dir, 'example_non_seasonal_results.json')} --use-xgboost --plot --save-plots"
        
        # Mevsimsel örnek analizi
        exit_code = run_command(seasonal_cmd, "Mevsimsel Örnek Analizi")
        if exit_code != 0:
            print("Hata: Mevsimsel örnek analizi başarısız!")
            return exit_code
        
        # Mevsimsel olmayan örnek analizi
        exit_code = run_command(non_seasonal_cmd, "Mevsimsel Olmayan Örnek Analizi")
        if exit_code != 0:
            print("Hata: Mevsimsel olmayan örnek analizi başarısız!")
            return exit_code
    
    # İşlem süresi
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n{'=' * 80}")
    print(f"  Tüm işlemler tamamlandı!")
    print(f"  Toplam süre: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
    print(f"{'=' * 80}")
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mevsimsellik analizi için tüm işlemleri çalıştır')
    parser.add_argument('--generate-data', action='store_true', help='Sentetik veri üret')
    parser.add_argument('--count', type=int, default=1000, help='Oluşturulacak zaman serisi sayısı')
    parser.add_argument('--balanced', action='store_true', help='Mevsimsel/mevsimsel olmayan serilerin dengeli olup olmayacağı')
    parser.add_argument('--train-xgboost', action='store_true', help='XGBoost modelini eğit')
    parser.add_argument('--train-lstm', action='store_true', help='LSTM modelini eğit')
    parser.add_argument('--evaluate', action='store_true', help='Modelleri değerlendir')
    parser.add_argument('--analyze-example', action='store_true', help='Örnek analiz yap')
    parser.add_argument('--seq-length', type=int, default=50, help='LSTM sekans uzunluğu')
    parser.add_argument('--epochs', type=int, default=50, help='LSTM eğitim epoch sayısı')
    parser.add_argument('--grid-search', action='store_true', help='XGBoost için grid search yap')
    parser.add_argument('--all', action='store_true', help='Tüm adımları çalıştır')
    
    args = parser.parse_args()
    
    # Hiçbir argüman verilmemişse veya --all kullanılmışsa tüm adımları çalıştır
    if len(sys.argv) == 1 or args.all:
        args.generate_data = True
        args.train_xgboost = True
        args.train_lstm = True
        args.evaluate = True
        args.analyze_example = True
    
    sys.exit(main(args))