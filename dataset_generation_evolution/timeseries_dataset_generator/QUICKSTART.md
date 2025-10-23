# Quick Start Guide

Bu kılavuz, `timeseries_dataset_generator` paketini hızlıca kullanmaya başlamanız için hazırlanmıştır.

##  Kurulum

### Geliştirme Modunda Kurulum
```bash
cd timeseries_dataset_generator
pip install -e .
```

Bu komut paketi "editable" modda kurar, böylece kod değişiklikleri hemen etkili olur.

##  İlk Veri Setinizi Oluşturun

### 1. Basit AR Serisi

```python
from timeseries_dataset_generator import TimeSeriesGenerator
from timeseries_dataset_generator.generators.stationary import generate_ar_dataset

# 10 adet AR serisi üret (50-100 uzunluğunda)
generate_ar_dataset(
    TimeSeriesGenerator,
    folder='my_output/ar_short',
    count=10,
    length_range=(50, 100)
)

print(" AR dataset oluşturuldu: my_output/ar_short/")
```

### 2. Trend İçeren Seri

```python
from timeseries_dataset_generator import TimeSeriesGenerator
from timeseries_dataset_generator.generators.trends import generate_linear_trend_dataset

# Yukarı trend içeren 5 adet seri üret
generate_linear_trend_dataset(
    TimeSeriesGenerator,
    folder='my_output/linear_up',
    kind='arma',           # ARMA tabanlı
    count=5,
    length_range=(300, 500),
    sign=1                 # 1=yukarı, -1=aşağı
)

print(" Linear trend dataset oluşturuldu: my_output/linear_up/")
```

### 3. Anomalili Seri

```python
from timeseries_dataset_generator import TimeSeriesGenerator
from timeseries_dataset_generator.generators.anomalies import generate_point_anomaly_dataset

# Ortasında nokta anomalisi olan seriler
generate_point_anomaly_dataset(
    TimeSeriesGenerator,
    folder='my_output/anomalies',
    kind='ar',
    count=5,
    length_range=(200, 300),
    anomaly_type='single',
    location='middle'      # 'beginning', 'middle', 'end'
)

print(" Anomaly dataset oluşturuldu: my_output/anomalies/")
```

##  Veriyi Okuyun

Oluşturulan veriler Parquet formatındadır:

```python
import pandas as pd

# Veriyi oku
df = pd.read_parquet('my_output/ar_short/short.parquet')

# İlk birkaç satıra bak
print(df.head())

# Bir seriyi seç ve görselleştir
import matplotlib.pyplot as plt

series_1 = df[df['series_id'] == 1]
plt.figure(figsize=(12, 4))
plt.plot(series_1['time'], series_1['data'])
plt.title('AR Series - ID: 1')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True)
plt.savefig('my_series_plot.png')
plt.close()

print(" Grafik kaydedildi: my_series_plot.png")
```

##  Çok Kategorili Veri Seti Oluşturma

```python
from timeseries_dataset_generator import TimeSeriesGenerator
from timeseries_dataset_generator.generators import (
    generate_ar_dataset,
    generate_ma_dataset,
    generate_arma_dataset,
    generate_linear_trend_dataset,
    generate_garch_dataset,
    generate_point_anomaly_dataset
)

base_folder = 'comprehensive_dataset'

# 1. Stationary seriler
print("1. Generating stationary series...")
generate_ar_dataset(TimeSeriesGenerator, f'{base_folder}/ar', count=20, length_range=(100, 200))
generate_ma_dataset(TimeSeriesGenerator, f'{base_folder}/ma', count=20, length_range=(100, 200))
generate_arma_dataset(TimeSeriesGenerator, f'{base_folder}/arma', count=20, length_range=(100, 200))

# 2. Trend seriler
print("2. Generating trend series...")
generate_linear_trend_dataset(
    TimeSeriesGenerator, f'{base_folder}/linear_up', 
    kind='ar', count=15, length_range=(200, 400), sign=1
)

# 3. Volatility seriler
print("3. Generating volatility series...")
generate_garch_dataset(TimeSeriesGenerator, f'{base_folder}/garch', count=15, length_range=(200, 400))

# 4. Anomalili seriler
print("4. Generating anomaly series...")
generate_point_anomaly_dataset(
    TimeSeriesGenerator, f'{base_folder}/anomalies',
    kind='ar', count=10, length_range=(150, 250),
    anomaly_type='multiple'
)

print(f"\n Comprehensive dataset oluşturuldu: {base_folder}/")
```

##  İleri Seviye: Manuel Seri Üretimi

```python
from timeseries_dataset_generator import TimeSeriesGenerator

# Generator örneği oluştur
ts = TimeSeriesGenerator(length=500)

# 1. Temel AR serisi
df, info = ts.generate_stationary_base_series('ar')
print(f"AR order: {info['ar_order']}")
print(f"AR coefs: {info['ar_coefs']}")

# 2. Üstüne trend ekle
df, trend_info = ts.generate_deterministic_trend_linear(df, sign=1)

# 3. Mevsimsellik ekle
df_seasonal, season_info = ts.generate_seasonality_from_base_series('single')
if df_seasonal is not None:
    print(f"Seasonal period: {season_info['period']}")

# 4. Anomali ekle
df, anomaly_info = ts.generate_point_anomaly(df, location='end')
print(f"Anomaly location: {anomaly_info['location']}")
print(f"Anomaly indices: {anomaly_info['anomaly_indices']}")

# DataFrame'i kaydet
df.to_csv('custom_series.csv', index=False)
print("\n Custom series kaydedildi: custom_series.csv")
```

##  Tüm Generator'lar

### Stationary
- `generate_wn_dataset` - White Noise
- `generate_ar_dataset` - Autoregressive
- `generate_ma_dataset` - Moving Average
- `generate_arma_dataset` - ARMA

### Trends
- `generate_linear_trend_dataset` - Linear
- `generate_quadratic_trend_dataset` - Quadratic
- `generate_cubic_trend_dataset` - Cubic
- `generate_exponential_trend_dataset` - Exponential
- `generate_damped_trend_dataset` - Damped

### Stochastic
- `generate_random_walk_dataset` - Random Walk
- `generate_random_walk_with_drift_dataset` - RW with Drift
- `generate_ima_dataset` - IMA
- `generate_ari_dataset` - ARI
- `generate_arima_dataset` - ARIMA

### Volatility
- `generate_arch_dataset` - ARCH
- `generate_garch_dataset` - GARCH
- `generate_egarch_dataset` - EGARCH
- `generate_aparch_dataset` - APARCH

### Seasonality
- `generate_single_seasonality_dataset` - Single Seasonal
- `generate_multiple_seasonality_dataset` - Multiple Seasonal
- `generate_sarma_dataset` - SARMA
- `generate_sarima_dataset` - SARIMA

### Anomalies
- `generate_point_anomaly_dataset` - Point Anomalies
- `generate_collective_anomaly_dataset` - Collective Anomalies
- `generate_contextual_anomaly_dataset` - Contextual Anomalies

### Structural Breaks
- `generate_mean_shift_dataset` - Mean Shifts
- `generate_variance_shift_dataset` - Variance Shifts
- `generate_trend_shift_dataset` - Trend Shifts

##  İpuçları

1. **Klasör Oluşturma**: Generator'lar klasörleri otomatik oluşturur
2. **Length Range**: `(min, max)` formatında tuple kullanın
3. **Count**: Kaç seri üretileceğini belirtir
4. **Parquet Format**: Verimli ve hızlı, pandas ile kolayca okunur
5. **Metadata**: Her seri için kapsamlı metadata kaydedilir

##  Sorun Giderme

### Import Hatası
```python
# Hata: ModuleNotFoundError: No module named 'timeseries_dataset_generator'

# Çözüm: Paketi kurun
# cd timeseries_dataset_generator
# pip install -e .
```

### TimeSeriesGenerator Eksik
```python
# Hata: NameError: name 'TimeSeriesGenerator' is not defined

# Çözüm: Import edin
from timeseries_dataset_generator import TimeSeriesGenerator
```

##  Daha Fazla Bilgi

- **README.md**: Detaylı dokümantasyon
- **MIGRATION_GUIDE.md**: Eski koddan geçiş
- **examples/generate_dataset.py**: Kapsamlı örnek
- **GitHub**: Issues ve discussions

##  Hazırsınız!

Artık zaman serisi veri setleri oluşturmaya hazırsınız. Başarılar! 

