import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from xgboost import XGBClassifier

import statsmodels.tsa.stattools as sm_stattools  # adfuller
from arch.unitroot import PhillipsPerron
import statsmodels.tsa.stattools as st_kpss  # kpss


class LSTMClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        out = self.fc(last_hidden)
        return out

########################
# 1) Yardımcı Fonksiyonlar
########################

def convert_to_numeric(series_1d):
    """
    Parametre: 'series_1d' => numpy array veya list
    İşlem:
      1) Pandas Series'e dönüştür
      2) to_numeric(..., errors='coerce') => numeric olmayanları NaN yap
      3) dropna() => NaN satırları at
      4) Kalan satır sayısı < 2 ise None döndür
      5) Yoksa .values ile numpy array döndür
    """
    s = pd.Series(series_1d)
    s = pd.to_numeric(s, errors='coerce')
    s = s.dropna()
    if len(s) < 2:
        return None
    return s.values  # numpy array

def extract_features(series_1d):
    """
    XGBoost vb. ML model için feature: [mean, var, lag-1 correlation]
    Numerik veri elde edemezsek None döndürür (es geçilir).
    """
    numeric_vals = convert_to_numeric(series_1d)
    if numeric_vals is None:
        return None
    
    mean_val = np.mean(numeric_vals)
    var_val = np.var(numeric_vals)
    
    corr_mat = np.corrcoef(numeric_vals[:-1], numeric_vals[1:])
    if np.isnan(corr_mat[0,1]):
        lag1_corr = 0
    else:
        lag1_corr = corr_mat[0,1]
    
    return [mean_val, var_val, lag1_corr]

########################
# 2) ML MODELLERİ
########################

@st.cache_resource
def load_xgb_model():
    return joblib.load("xgb_model.pkl")

@st.cache_resource
def load_lstm_model():
    lstm_model = LSTMClassifier(input_size=1, hidden_size=16)
    lstm_model.load_state_dict(torch.load("lstm_model.pt"))
    lstm_model.eval()
    return lstm_model

def predict_xgb(series_1d, xgb_model):
    """Seri üzerinde feature çıkar, XGBoost tahmini."""
    feats = extract_features(series_1d)
    if feats is None:
        return None  # Yeterli numerik veri yok
    feats_np = np.array(feats).reshape(1, -1)
    pred = xgb_model.predict(feats_np)[0]
    return int(pred)  # 0 veya 1

def predict_lstm(series_1d, lstm_model):
    """Seriyi numeric'e çevirip tensöre dönüştür, LSTM tahmini."""
    numeric_vals = convert_to_numeric(series_1d)
    if numeric_vals is None:
        return None
    seq_tensor = torch.tensor(numeric_vals, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        out = lstm_model(seq_tensor)
        prob = torch.sigmoid(out).item()
        pred = 1 if prob >= 0.5 else 0
    return pred

########################
# 3) İSTATİSTİKSEL TESTLER
########################

def adf_decision(series_1d, alpha=0.05):
    numeric_vals = convert_to_numeric(series_1d)
    if numeric_vals is None:
        return None
    try:
        result = sm_stattools.adfuller(numeric_vals, regression='c', autolag='AIC')
        p_value = result[1]
    except:
        return None
    return 1 if p_value < alpha else 0

def pp_decision(series_1d, alpha=0.05):
    numeric_vals = convert_to_numeric(series_1d)
    if numeric_vals is None:
        return None
    try:
        pp = PhillipsPerron(numeric_vals, trend='c')
        p_value = pp.pvalue
    except:
        return None
    return 1 if p_value < alpha else 0

def kpss_decision(series_1d, alpha=0.05):
    numeric_vals = convert_to_numeric(series_1d)
    if numeric_vals is None:
        return None
    try:
        stat, p_value, lags, crit = st_kpss.kpss(numeric_vals, regression='c', nlags="auto")
    except:
        return None
    # KPSS: H0=stationary, p_value < alpha => reddedildi => non-stationary => 0
    return 1 if p_value >= alpha else 0

########################
# 4) STREAMLIT ARAYÜZ
########################

def main():
    st.title("Stationary vs Non-Stationary Classifier - With Non-Numeric Skip")

    uploaded_file = st.file_uploader("Bir CSV dosyası seçin", type=["csv"])

    st.write("**Yöntem Seçimi**")
    use_xgb = st.checkbox("XGBoost (ML model)")
    use_lstm = st.checkbox("LSTM (ML model)")
    use_kpss = st.checkbox("KPSS Test")
    use_adf  = st.checkbox("ADF Test")
    use_pp   = st.checkbox("PP Test")

    alpha_val = st.number_input("Alpha (istatistiksel testler)", min_value=0.001, max_value=0.2, value=0.05, step=0.01)

    if st.button("Classify Columns"):
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write(f"Yüklenen dosyanın boyutu: {df.shape[0]} satır, {df.shape[1]} sütun.")

            if not (use_xgb or use_lstm or use_kpss or use_adf or use_pp):
                st.warning("Lütfen en az bir yöntem seçiniz.")
                return

            # Modelleri yükleyelim (Streamlit cache_resource)
            xgb_model = load_xgb_model() if use_xgb else None
            lstm_model = load_lstm_model() if use_lstm else None

            results = []
            for col_name in df.columns:
                series = df[col_name].values
                row_res = {"Column": col_name}

                if use_xgb:
                    pred_x = predict_xgb(series, xgb_model)
                    if pred_x is None:
                        row_res["XGB"] = "N/A (not numeric or insufficient data)"
                    else:
                        row_res["XGB"] = f"{pred_x} => Stationary" if pred_x==1 else "0 => Non-Stationary"

                if use_lstm:
                    pred_l = predict_lstm(series, lstm_model)
                    if pred_l is None:
                        row_res["LSTM"] = "N/A"
                    else:
                        row_res["LSTM"] = f"{pred_l} => Stationary" if pred_l==1 else "0 => Non-Stationary"

                if use_kpss:
                    pred_k = kpss_decision(series, alpha_val)
                    if pred_k is None:
                        row_res["KPSS"] = "N/A"
                    else:
                        row_res["KPSS"] = f"{pred_k} => Stationary" if pred_k==1 else "0 => Non-Stationary"

                if use_adf:
                    pred_a = adf_decision(series, alpha_val)
                    if pred_a is None:
                        row_res["ADF"] = "N/A"
                    else:
                        row_res["ADF"] = f"{pred_a} => Stationary" if pred_a==1 else "0 => Non-Stationary"

                if use_pp:
                    pred_p = pp_decision(series, alpha_val)
                    if pred_p is None:
                        row_res["PP"] = "N/A"
                    else:
                        row_res["PP"] = f"{pred_p} => Stationary" if pred_p==1 else "0 => Non-Stationary"

                results.append(row_res)

            if len(results) > 0:
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)
            else:
                st.write("Hiç kolon bulunamadı.")
        else:
            st.error("Lütfen önce bir CSV dosyası yükleyin.")

if __name__ == "__main__":
    main()
