import os
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

SERIES_FOLDER = "data/series"
metadata = pd.read_csv("data/series_metadata.csv")

def get_trend_label(series):
    try:
        adf_p = adfuller(series, autolag='AIC')[1]
    except:
        adf_p = 1.0  # bozuk veri gibi say

    try:
        kpss_p = kpss(series, regression='c', nlags='auto')[1]
    except:
        kpss_p = 0.0  # trendli say

    if adf_p < 0.05 and kpss_p >= 0.05:
        return "no_trend"
    elif adf_p >= 0.05 and kpss_p < 0.05:
        return "deterministic"
    elif adf_p >= 0.05 and kpss_p >= 0.05:
        return "stochastic"
    else:
        return "inconclusive"

print("İstatistiksel testlerle etiketleme başlıyor...")
test_labels = []
for fname in tqdm(metadata["filename"]):
    path = os.path.join(SERIES_FOLDER, fname)
    series = pd.read_csv(path, header=None).squeeze("columns").values
    label = get_trend_label(series)
    test_labels.append(label)

metadata["test_label"] = test_labels
metadata.to_csv("data/series_metadata_with_tests.csv", index=False)

print("Tamamlandı. Dosya: data/series_metadata_with_tests.csv")
df = pd.read_csv("data/series_metadata_with_tests.csv")
match = (df["label"] == df["test_label"]).sum()
total = len(df)
print(f"İstatistiksel testlerin veri üretim etiketleriyle uyuşma oranı: {match/total:.2%}")
