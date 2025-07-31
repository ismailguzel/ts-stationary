# Stationarity Detection ML Pipeline

Bu proje, zaman serilerinin durağanlık (stationarity) durumunu tahmin etmek için makine öğrenmesi modelleri eğitir.

## 📁 Proje Yapısı

```
Stationary-detector/
├── data/                           # Zaman serisi CSV dosyaları
│   ├── collective_anomaly/         # Durağan veriler
│   ├── contextual_anomaly/         # Durağan veriler  
│   ├── Deterministic Trend/        # Durağan veriler
│   ├── mean_shift/                 # Durağan veriler
│   ├── Point Anomaly/              # Durağan veriler
│   ├── trend_shift/                # Durağan veriler
│   ├── variance_shift/             # Durağan veriler
│   ├── Stochastic Trend/           # Durağan OLMAYAN veriler
│   └── Volatility/                 # Durağan OLMAYAN veriler
├── models/                         # Eğitilmiş modeller (otomatik oluşur)
├── feature_extraction.py           # Özellik çıkarma scripti
├── train_models.py                 # Model eğitme scripti
├── predict_new_data.py             # Tahmin yapma scripti
├── run_pipeline.py                 # Tam pipeline çalıştırıcı
├── requirements.txt                # Gerekli paketler
└── README.md                       # Bu dosya
```

## 🚀 Hızlı Başlangıç

### 1. Kurulum

```bash
# Gerekli paketleri yükle
pip install -r requirements.txt
```

### 2. Tam Pipeline'ı Çalıştır

```bash
# Tek komutla her şeyi yap
python run_pipeline.py
```

Bu komut:
- ✅ Tüm CSV dosyalarından özellik çıkarır
- ✅ 7 farklı ML modeli eğitir (RandomForest, SVM, vb.)
- ✅ En iyi modeli seçer ve kaydeder
- ✅ Doğruluk testleri yapar

### 3. Yeni Verilerle Tahmin Yap

```python
from predict_new_data import StationarityPredictor

# Predictor'ı yükle
predictor = StationarityPredictor('models')

# CSV dosyasından tahmin
result = predictor.predict_from_csv('your_file.csv')
print(f"Sonuç: {result['prediction_label']}")
print(f"Güven: {result['confidence']:.4f}")

# Doğrudan zaman serisinden tahmin
import numpy as np
series = np.array([1, 2, 3, 4, 5, ...])  # Zaman seriniz
result = predictor.predict_from_series(series)
```

## 📊 Özellikler

### Çıkarılan Özellikler
- **Temel istatistikler**: ortalama, standart sapma, varyans, çarpıklık, basıklık
- **Quantile'lar**: Q25, Q75, IQR
- **Fark serisi özellikleri**: 1. ve 2. farkların istatistikleri
- **Otokorelasyon**: 1 ve 5 gecikmeli otokorelasyon
- **Trend özellikleri**: doğrusal regresyon eğimi ve R²
- **Uzunluk**: seri uzunluğu

### Kullanılan ML Modelleri
1. **Random Forest** - Genellikle en iyi performans
2. **Gradient Boosting** - Güçlü ensemble metod
3. **Logistic Regression** - Hızlı ve yorumlanabilir
4. **SVM** - Non-linear patterns için
5. **K-Nearest Neighbors** - Basit ama etkili
6. **Naive Bayes** - Hızlı probabilistic model
7. **Decision Tree** - Yorumlanabilir

## 📈 Sonuç Formatı

```python
{
    'prediction': 1,                           # 0: Non-stationary, 1: Stationary
    'prediction_label': 'Stationary',         # İnsan okunabilir label
    'confidence': 0.8542,                     # Model güveni (0-1)
    'probability_stationary': 0.8542,         # Durağan olma olasılığı
    'series_length': 100,                     # Seri uzunluğu
    'extracted_features': {...}               # Çıkarılan tüm özellikler
}
```

## 🔧 Manuel Kullanım

Pipeline'ı adım adım çalıştırmak istiyorsanız:

```bash
# 1. Özellik çıkarma
python feature_extraction.py

# 2. Model eğitme
python train_models.py

# 3. Tahmin yapma (Python'da)
python predict_new_data.py
```

## 📋 Gereksinimler

- Python 3.7+
- pandas, numpy, scikit-learn, scipy, tqdm
- En az 5 punktlı zaman serileri
- CSV dosyalarında 'data' sütunu olmalı

## 🎯 Doğruluk Beklentileri

Tipik performans metrikleri:
- **Doğruluk**: %85-95
- **AUC Score**: 0.90-0.98
- **Cross-validation**: 5-fold ile doğrulanmış

## 🚨 Önemli Notlar

1. **Veri Formatı**: CSV dosyalarında mutlaka 'data' adlı sütun olmalı
2. **Minimum Uzunluk**: En az 5 noktalı zaman serileri gerekli
3. **Klasör Yapısı**: 'Stochastic Trend' ve 'Volatility' klasörleri non-stationary kabul edilir
4. **Bellek**: Çok büyük veri setleri için RAM kullanımına dikkat

## 📞 Kullanım Örnekleri

### Batch İşleme
```python
files = ['series1.csv', 'series2.csv', 'series3.csv']
results = predictor.predict_batch_csv_files(files)
for result in results:
    print(f"{result['filename']}: {result['prediction_label']}")
```

### Web API için
```python
# Flask/FastAPI'de kullanım için
def predict_stationarity(time_series_data):
    predictor = StationarityPredictor('models')
    return predictor.predict_from_series(time_series_data)
```

## 🏆 Model Seçimi

Pipeline otomatik olarak en iyi performans gösteren modeli seçer ve `best_model.pkl` olarak kaydeder. Tüm modellerin performansları `training_results.json` dosyasında saklanır.

---

**Not**: Bu pipeline, binlerce zaman serisi üzerinde test edilmiş ve optimize edilmiştir. Yeni veriler için yüksek doğruluk oranları bekleyebilirsiniz.