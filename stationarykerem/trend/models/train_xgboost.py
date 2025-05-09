import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Parametreler
SERIES_FOLDER = "data/series"
MAX_LEN = 150  # maksimum uzunluk
metadata = pd.read_csv("data/series_metadata.csv")

# Encode etiketler
label_encoder = LabelEncoder()
metadata["label_encoded"] = label_encoder.fit_transform(metadata["label"])

# Train / Test ayır
train_meta = metadata[metadata["split"] == "train"]
test_meta = metadata[metadata["split"] == "test"]

def load_and_pad(filename):
    series = pd.read_csv(os.path.join(SERIES_FOLDER, filename), header=None).squeeze("columns").values
    padded = np.pad(series, (0, MAX_LEN - len(series)), 'constant')
    return padded

# Veri yükle
X_train = np.stack([load_and_pad(f) for f in train_meta["filename"]])
y_train = train_meta["label_encoded"].values

X_test = np.stack([load_and_pad(f) for f in test_meta["filename"]])
y_test = test_meta["label_encoded"].values

# XGBoost modeli
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)

# Sonuç
print(classification_report(y_test, pred, target_names=label_encoder.classes_))
