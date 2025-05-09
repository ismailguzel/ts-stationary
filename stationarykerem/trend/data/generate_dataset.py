import numpy as np
import pandas as pd
import random
import os

os.makedirs("data/series", exist_ok=True)

def generate_series(length, trend_type="none"):
    x = np.arange(length)
    noise = np.random.normal(0, 1, length)
    if trend_type == "deterministic":
        slope = random.uniform(0.3, 3.0)
        return slope * x + noise
    elif trend_type == "stochastic":
        return np.cumsum(noise)
    else:
        return noise

series_metadata = []

for i in range(1000):
    length = random.randint(20, 150)
    trend_type = random.choice(["no_trend", "deterministic", "stochastic"])
    series = generate_series(length=length, trend_type=trend_type)
    
    filename = f"series_{i}.csv"
    filepath = os.path.join("data", "series", filename)
    
    # Kaydet
    pd.Series(series).to_csv(filepath, index=False, header=False)

    # Metadata kaydet
    split = "train" if i < 500 else "test"
    series_metadata.append((filename, trend_type, length, split))

# metadata dosyası
meta_df = pd.DataFrame(series_metadata, columns=["filename", "label", "length", "split"])
meta_df.to_csv("data/series_metadata.csv", index=False)
