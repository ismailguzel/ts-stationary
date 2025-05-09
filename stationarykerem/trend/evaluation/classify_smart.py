import pandas as pd
import numpy as np
import os
import warnings
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

SERIES_FOLDER = "data/series"
metadata = pd.read_csv("data/series_metadata.csv")

# Polinom R^2
def fit_poly(series, degree=2):
    X = np.arange(len(series)).reshape(-1, 1)
    y = series
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    y_pred = model.predict(X_poly)
    return r2_score(y, y_pred)

# Varyans artışı
def increasing_variance(series, segments=3):
    n = len(series)
    split = np.array_split(series, segments)
    variances = [np.var(s) for s in split]
    return variances == sorted(variances)

# Differencing sonrası ADF
def difference_adf(series):
    if len(series) < 3:
        return False
    diff = np.diff(series)
    try:
        pval = adfuller(diff)[1]
    except:
        pval = 1.0
    return pval < 0.05

# ADF & KPSS & smart classify
def smart_classify(series):
    try:
        adf_p = adfuller(series, autolag='AIC')[1]
    except:
        adf_p = 1.0

    try:
        kpss_p = kpss(series, regression='c', nlags='auto')[1]
    except:
        kpss_p = 0.0

    poly_r2 = fit_poly(series)
    var_inc = increasing_variance(series)
    diff_stat = difference_adf(series)

    # Akıllı sınıflandırma kararı
    if adf_p < 0.05 and kpss_p >= 0.05 and poly_r2 < 0.5:
        return "no_trend"
    elif poly_r2 > 0.9 and not var_inc:
        return "deterministic"
    elif diff_stat and var_inc:
        return "stochastic"
    else:
        return "inconclusive"

# Tüm verileri işle
labels = []
adf_ps = []
kpss_ps = []
poly_r2s = []
var_flags = []
diff_flags = []
smart_labels = []

print("Tüm testler uygulanıyor...")
for fname in tqdm(metadata["filename"]):
    series = pd.read_csv(os.path.join(SERIES_FOLDER, fname), header=None).squeeze("columns").values

    try:
        adf_p = adfuller(series, autolag='AIC')[1]
    except:
        adf_p = 1.0

    try:
        kpss_p = kpss(series, regression='c', nlags='auto')[1]
    except:
        kpss_p = 0.0

    r2 = fit_poly(series)
    var_inc = increasing_variance(series)
    diff_stat = difference_adf(series)
    smart_label = smart_classify(series)

    adf_ps.append(adf_p)
    kpss_ps.append(kpss_p)
    poly_r2s.append(r2)
    var_flags.append(var_inc)
    diff_flags.append(diff_stat)
    smart_labels.append(smart_label)

# Sonuçları dataframe'e ekle
metadata["adf_p"] = adf_ps
metadata["kpss_p"] = kpss_ps
metadata["poly_r2"] = poly_r2s
metadata["var_increasing"] = var_flags
metadata["diff_stationary"] = diff_flags
metadata["smart_label"] = smart_labels

# Kaydet
metadata.to_csv("data/series_metadata_with_diagnostics.csv", index=False)

# Karşılaştırma
match = (metadata["label"] == metadata["smart_label"]).sum()
total = len(metadata)
print(f"Smart Classifier ile etiket uyuşması: {match}/{total} ({match/total:.2%})")
