import ipdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
import random
import statsmodels.api as sm
from numpy.polynomial.polynomial import Polynomial
from statsmodels.tsa.filters.hp_filter import hpfilter
import pickle

class TimeSeriesGenerator:
  def __init__(self, length=None):
    self.length = length if length is not None else 400
    self.base_distributions = ['white_noise', 'ar', 'ma', 'arma']
    self.characteristics = {'deterministic_trend_linear' : self.generate_deterministic_trend_linear,
    'deterministic_trend_cubic': self.generate_deterministic_trend_cubic,
    'deterministic_trend_quadratic': self.generate_deterministic_trend_quadratic,
    'stochastic_trend': self.generate_stochastic_trend,
    'seasonality': self.generate_seasonality}
    self.structural_breaks = {'mean_shift': self.generate_mean_shift,
    'variance_shift': self.generate_variance_shift,
    'trend_shift': self.generate_trend_shift,
    'gradual_change_mean': self.generate_gradual_mean_shift,
    'gradual_change_variance': self.generate_gradual_variance_shift,
    'gradual_change_trend': self.generate_gradual_trend_shift}

  def z_normalize(self,series):
    return (series - np.mean(series)) / np.std(series)

  def is_stationary(self, ar_params):
    # Check if AR parameters lead to stationarity
    ar_poly = np.r_[1, -ar_params]
    roots = Polynomial(ar_poly).roots()
    return np.all(np.abs(roots) > 1)

  def is_invertible(self, ma_params):
    # Check if MA parameters lead to invertibility
    ma_poly = np.r_[1, ma_params]
    roots = Polynomial(ma_poly).roots()
    return np.all(np.abs(roots) > 1)

  def extract_seasonal_part(self,series, period):
    decomposition = seasonal_decompose(series, model='additive', period=period)
    seasonal = decomposition.seasonal
    return seasonal

  def generate_ar_params(self, order_range=(1, 3), coef_range=(-0.3, 0.3)):
    while True:
      order = np.random.randint(order_range[0], order_range[1] + 1)
      coefs = np.random.uniform(coef_range[0], coef_range[1], order)
      ar = np.r_[1, -coefs]
      ma = np.array([1])
      arma_process = ArmaProcess(ar, ma)
      if arma_process.isstationary:
        break
    return order, coefs

  def generate_ma_params(self, order_range=(1, 3), coef_range=(-0.3, 0.3)):
    while True:
      order = np.random.randint(order_range[0], order_range[1] + 1)
      coefs = np.random.uniform(coef_range[0], coef_range[1], order)
      ma = np.r_[1, coefs]
      ar = np.array([1])
      arma_process = ArmaProcess(ar, ma)
      if arma_process.isinvertible:
        break
    return order, coefs

  def generate_arma_params(self, order_range=(1, 3), coef_range=(-0.3, 0.3)):
    while True:
      ar_order = np.random.randint(order_range[0], order_range[1] + 1)
      ma_order = np.random.randint(order_range[0], order_range[1] + 1)
      ar_coefs = np.random.uniform(coef_range[0], coef_range[1], ar_order)
      ma_coefs = np.random.uniform(coef_range[0], coef_range[1], ma_order)
      ma = np.r_[1, ma_coefs]
      ar = np.r_[1, -ar_coefs]
      arma_process = ArmaProcess(ar, ma)
      if arma_process.isinvertible and arma_process.isstationary:
        break
    return ar_order, ma_order, ar_coefs, ma_coefs

  def generate_sarima_params(self, p_range=(1, 3), d_range=(0, 1), q_range=(1, 3), seasonal_order=(1, 1, 1, 12), coef_range=(-0.2, 0.2)):
    while True:
      p = np.random.randint(p_range[0], p_range[1] + 1)
      d = np.random.randint(d_range[0], d_range[1] + 1)
      q = np.random.randint(q_range[0], q_range[1] + 1)
      P, D, Q, s = seasonal_order

      ar_params = np.random.uniform(coef_range[0], coef_range[1], p)
      ma_params = np.random.uniform(coef_range[0], coef_range[1], q)

      seasonal_ar_params = np.random.uniform(coef_range[0], coef_range[1], P)
      seasonal_ma_params = np.random.uniform(coef_range[0], coef_range[1], Q)

      if self.is_stationary(ar_params) and self.is_invertible(ma_params) and self.is_stationary(seasonal_ar_params) and self.is_invertible(seasonal_ma_params):
        params = np.concatenate(([0], ar_params, ma_params, seasonal_ar_params, seasonal_ma_params))
        return (p, d, q), (P, D, Q, s), params

  def generate_white_noise(self, length, noise_scale = 0.05):
    series = np.random.normal(0, 0.05, length)
    series = series * noise_scale
    return series

  def generate_ar_series(self, length, noise_scale = 0.5):
    order,coefs = self.generate_ar_params()
    ar = np.r_[1, -np.array(coefs)]  # leading 1 and negate the coefficients
    ma = np.r_[1]  # MA coefficients are just [1] for a pure AR process
    ar_process = ArmaProcess(ar, ma)
    series = ar_process.generate_sample(nsample=length)
    series = series * noise_scale
    return series

  def generate_ma_series(self, length, noise_scale = 0.5):
    order,coefs = self.generate_ma_params()
    ar = np.r_[1]  # AR coefficients are just [1] for a pure MA process
    ma = np.r_[1, np.array(coefs)]  # leading 1 for the MA coefficients
    arma_process = ArmaProcess(ar, ma)
    series = arma_process.generate_sample(nsample=length)
    series = series * noise_scale
    return series

  def generate_arma_series(self, length, noise_scale = 0.5):
    ar_order,ma_order,ar_coefs,ma_coefs = self.generate_arma_params()
    ar = np.r_[1, -np.array(ar_coefs)]
    ma = np.r_[1, np.array(ma_coefs)]
    arma_process = ArmaProcess(ar, ma)
    series = arma_process.generate_sample(nsample=length)
    series = series * noise_scale
    return series

  def generate_sarima_series(self, length, noise_scale=0.5, max_attempts=10):
    attempts = 0
    while attempts < max_attempts:
      try:
        order, seasonal_order, params = self.generate_sarima_params()
        p, d, q = order
        P, D, Q, s = seasonal_order

        sarima_model = SARIMAX(endog=np.zeros(length), order=(p, d, q), seasonal_order=(P, D, Q, s))

        series = sarima_model.simulate(params=params, nsimulations=length)
        if not np.isnan(series).any():
          return series * noise_scale

      except (ValueError, np.linalg.LinAlgError):
        attempts += 1
        print(f"Attempt {attempts}/{max_attempts} failed due to numerical instability. Retrying with new parameters...")

    raise ValueError("Failed to generate a stable SARIMA series after multiple attempts.")

  def generate_base_series(self, distribution=None):
    if distribution is None:
      distribution = np.random.choice(self.base_distributions)
    if distribution == 'white_noise':
      series = self.generate_white_noise(self.length)
    elif distribution == 'ar':
      series = self.generate_ar_series(self.length)
    elif distribution == 'ma':
      series = self.generate_ma_series(self.length)
    elif distribution == 'arma':
      series = self.generate_arma_series(self.length)
    elif distribution == 'sarima':
      series = self.generate_sarima_series(self.length)

    df = pd.DataFrame({
    'data': series,
    'mean_shift': (np.zeros(self.length)).astype(int),
    'variance_shift': (np.zeros(self.length)).astype(int),
    'trend_shift': (np.zeros(self.length)).astype(int),
    'anomaly': (np.zeros(self.length)).astype(int),})
    return df

  def generate_point_anomalies(self, df, scale_factor=1):
    series = df['data'].copy()
    series = self.z_normalize(series)
    num_anomalies = np.random.randint(int(len(series)*0.01), int(len(series)*0.05))
    magnitude = np.random.uniform(3,6)
    anomaly_indices = np.random.choice(len(series), num_anomalies, replace=False)
    series[anomaly_indices] += np.random.choice([-1, 1], num_anomalies) * magnitude * np.std(series) * scale_factor
    df.loc[:,'data'] = series
    df.loc[anomaly_indices,'anomaly'] = 1
    return df

  def generate_collective_anomalies(self, df, change_type = None, scale_factor = 1, seasonal_period = None):
    series = df['data'].copy()
    series = self.z_normalize(series)
    start = np.random.randint(int(len(series)*0.1), int(len(series)*0.9))
    anomaly_length = np.random.randint(int(len(series))*0.05, int(len(series)*0.09))
    end = start + anomaly_length
    magnitude = np.random.uniform(1.5,2.5)
    sign = np.random.choice([-1, 1])
    if change_type == 'mean':
      series[start:end] += sign * np.ones(anomaly_length) * magnitude * np.std(series) * scale_factor
    elif change_type == 'variance':
      if sign == 1:
        series[start:end] *= magnitude * np.std(series) * scale_factor
      elif sign == -1:
        series[start:end] *= (1/magnitude) * np.std(series) * scale_factor
    elif change_type == 'seasonal':
      distortion_factor = np.random.uniform(0.5, 2)
      distorted_seasonality = np.sin(2 * np.pi * np.arange(anomaly_length) / (seasonal_period * distortion_factor))
      series[start:end] += sign * distorted_seasonality * magnitude * scale_factor
    else:
      raise ValueError("Invalid change_type. Expected 'mean', 'variance', or 'seasonal'.")
    df.loc[:,'data'] = series
    df.loc[start:end,'anomaly'] = 1
    return df

  def generate_deterministic_trend_linear(self, df, sign = None, slope= None, noise_std = None, intercept = 1, scale_factor = 1):
    series = df['data'].copy()
    noise_std = noise_std if noise_std is not None else np.random.uniform(0.1, 1.5)
    if len(series) == 100:
        slope_range = [0.4, 0.5, 0.6, 0.7, 0.8]
        scale_factor = 4
    else:
        slope_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        scale_factor = 1
    slope = slope if slope is not None else random.choice(slope_range)*sign
    trend = intercept + slope * np.arange(len(series)) + np.random.normal(0, noise_std, len(series))
    series += trend * scale_factor
    series = self.z_normalize(series)
    df.loc[:,'data'] = series
    return df

  def generate_deterministic_trend_cubic(self, df, sign = None, a=None, b=None, c=None, d=None, noise_std=None, scale_factor = 1):
    series = df['data'].copy()
    noise_std = noise_std if noise_std is not None else np.random.uniform(0.1, 1.5)
    if len(series) == 100:
        scale_factor = 5
    else:
        scale_factor = 1
    a = a if a is not None else sign * random.uniform(0.0000001, 0.000001)
    b = b if b is not None else sign * random.uniform(0.0001, 0.001)
    c = c if c is not None else sign * random.uniform(0.01, 0.1)
    d = d if d is not None else sign * random.uniform(1, 10)
    trend = (a * (np.arange(len(series)))**3 + b * (np.arange(len(series)))**2 + c * (np.arange(len(series))) + d) * scale_factor + np.random.normal(0, noise_std, len(series))
    series += trend
    series = self.z_normalize(series)
    df.loc[:,'data'] = series
    return df

  def generate_deterministic_trend_quadratic(self, df, sign = None, a=None, b=None, c=None, noise_std=None, scale_factor =1):
    series = df['data'].copy()
    noise_std = noise_std if noise_std is not None else np.random.uniform(0.1, 1.5)
    if len(series) == 100:
        scale_factor = 5
    else:
        scale_factor = 1
    a = a if a is not None else sign * random.uniform(0.0001, 0.001)
    b = b if b is not None else sign * random.uniform(0.001, 0.01)
    c = c if c is not None else sign * random.uniform(1, 10)
    trend = (a * (np.arange(len(series)))**2 + b * (np.arange(len(series))) + c) * scale_factor + np.random.normal(0, noise_std, len(series))
    series += trend
    series = self.z_normalize(series)
    df.loc[:,'data'] = series
    return df

  def generate_stochastic_trend(self, df, sign = None, drift=None, noise_std=None, scale_factor=1):
    series = df['data'].copy()
    noise_std = noise_std if noise_std is not None else np.random.uniform(0.1, 1.5)
    if len(series) == 100:
        drift_range = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        scale_factor = 4
    else:
        drift_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        scale_factor = 1
    drift = drift if drift is not None else random.choice(drift_range) * sign
    random_walk = np.cumsum(np.random.normal(0, noise_std, len(series))) * scale_factor
    trend = drift * np.arange(len(series)) * scale_factor + random_walk
    series += trend
    series = self.z_normalize(series)
    df.loc[:,'data'] = series
    return df

  def generate_seasonality(self, df, period=None, amplitude=None, noise_std=None, scale_factor = 3):
    series = df['data'].copy()
    noise_std = noise_std if noise_std is not None else np.random.uniform(0.01, 0.05)
    min_period = 5  
    max_period = len(series) // 6  # Ensure at least 6 cycles
    periods = [p for p in [5, 7, 12, 24, 30, 52, 90, 180] if min_period <= p <= max_period]
    period = period if period is not None else random.choice(periods)
    amplitude = amplitude if amplitude is not None else np.std(series) * np.random.uniform(0.5, 2.5)
    seasonality = (amplitude * np.sin(2 * np.pi * np.arange(len(series)) / period) + np.random.normal(0, noise_std, size = len(series)))
    series += seasonality * scale_factor
    series = self.z_normalize(series)
    df.loc[:,'data'] = series
    return period, df

  def generate_mean_shift(self, df, sign = None, location=None, num_breaks=1, scale_factor=1, seasonal_period = None):
    series = df['data'].copy()
    noise = np.random.uniform(0.1, 1.5)
    n = len(series)
    mean = np.mean(series)
    std = np.std(series)
    residuals, trend = hpfilter(series, lamb=11000)
    min_break_length = int(0.1 * n) 
    max_break_length = int(0.2 * n)

    created_breaks = []

    if num_breaks == 1 and location:
      if location == "beginning":
        start, end = int(0.1 * n), int(0.3 * n)
      elif location == "middle":
        start, end = int(0.3 * n), int(0.7 * n) 
      elif location == "end":
        start, end = int(0.7 * n), int(0.9 * n)
      else:
        raise ValueError("Invalid location. Expected 'beginning', 'middle', or 'end'.")

      break_point = np.random.randint(start, end)

      if seasonal_period:
        phase = break_point % seasonal_period
        break_point -= phase 

      break_length =  np.random.randint(int(0.1 * n), n - break_point)
      break_end_point = break_point + break_length

      if seasonal_period:
        phase_end = break_end_point % seasonal_period
        break_end_point -= phase_end
        break_end_point = min(break_end_point, n)

      magnitude = np.random.uniform(0.5, 2)
      level_shift = sign * magnitude

      if seasonal_period:
        transition_length = max(1, seasonal_period // 2)
        transition_start = max(0, break_point - transition_length)
        transition_end = min(n, break_end_point + transition_length)

        sigmoid_transition = np.linspace(-6, 6, transition_end - transition_start)
        smooth_weights = 1 / (1 + np.exp(-sigmoid_transition))

        trend[transition_start:break_point] += smooth_weights[: break_point - transition_start] * level_shift * scale_factor
        trend[break_point:break_end_point] += level_shift * scale_factor
        trend[break_end_point:transition_end] += smooth_weights[break_end_point - transition_start :] * level_shift * scale_factor

        created_breaks.append((break_point, break_end_point))

      else:
        trend[break_point:break_end_point] += level_shift * scale_factor
        created_breaks.append((break_point, break_end_point))

    else:
      segment_size = n // num_breaks
      for i in range(num_breaks):
        start = i * segment_size
        end = (i + 1) * segment_size if i < num_breaks - 1 else n

        if end - start < min_break_length:
          continue

        break_point = np.random.randint(start, end - min_break_length + 1)
        if seasonal_period:
          phase = break_point % seasonal_period
          break_point -= phase

        max_possible_length = int(n - break_point)
        break_length = np.random.randint(min_break_length, min(max_possible_length, max_break_length) + 1)
        break_end_point = break_point + break_length

        if seasonal_period:
          phase_end = break_end_point % seasonal_period
          break_end_point -= phase_end
          break_end_point = min(break_end_point, n)

        magnitude = np.random.uniform(0.5, 1.5)
        level_shift = sign * magnitude

        if seasonal_period:
          transition_length = max(1, seasonal_period // 2) 
          transition_start = max(0, break_point - transition_length)
          transition_end = min(n, break_end_point + transition_length)

          sigmoid_transition = np.linspace(-6, 6, transition_end - transition_start)
          smooth_weights = 1 / (1 + np.exp(-sigmoid_transition))

          trend[transition_start:break_point] += smooth_weights[: break_point - transition_start] * level_shift * scale_factor
          trend[break_point:break_end_point] += level_shift * scale_factor
          trend[break_end_point:transition_end] += smooth_weights[break_end_point - transition_start :] * level_shift * scale_factor

          created_breaks.append((break_point, break_end_point))

        else:
          trend[break_point:break_end_point] += level_shift * scale_factor
          created_breaks.append((break_point, break_end_point))

    series = residuals + trend
    series += noise
    series = self.z_normalize(series)

    for i, (break_start, break_end) in enumerate(created_breaks):
      df.loc[break_start:break_end, 'mean_shift'] = 1

    df.loc[:,'data'] = series
    return df

  def generate_variance_shift(self, df, sign = None, location=None, num_breaks=1, scale_factor=1, seasonal_period = None):
    series = df['data'].copy()
    n = len(series)
    mean = np.mean(series)
    std = np.std(series)
    residuals, trend = hpfilter(series, lamb=11000)

    min_break_length = int(0.1 * n)
    max_break_length = int(0.2 * n)

    created_breaks = []

    if num_breaks == 1 and location:
      if location == "beginning":
        start, end = int(0.1 * n ), int(0.3 * n)
      elif location == "middle":
        start, end = int(0.3 * n), int(0.7 * n)
      elif location == "end":
        start, end = int(0.7 * n), int(0.9 * n)
      else:
        raise ValueError("Invalid location. Expected 'beginning', 'middle', or 'end'.")

      break_point = np.random.randint(start, end)

      if seasonal_period:
        phase = break_point % seasonal_period
        break_point -= phase 

      break_length =  np.random.randint(int(0.1 * n), n - break_point)
      break_end_point = break_point + break_length

      if seasonal_period:
        phase_end = break_end_point % seasonal_period
        break_end_point -= phase_end
        break_end_point = min(break_end_point, n)

      variance_factor = np.random.uniform(1.5, 3)
      if sign == 1:
        residuals[break_point:break_end_point] *= variance_factor * scale_factor
      elif sign == -1:
        residuals[:break_point] *= variance_factor * scale_factor
        residuals[break_end_point:] *= variance_factor * scale_factor
      else:
        raise ValueError("Invalid sign. Expected 1 or -1.")

      created_breaks.append((break_point, break_end_point))

    else:
      segment_size = n // num_breaks
      for i in range(num_breaks):
        start = i * segment_size
        end = (i + 1) * segment_size if i < num_breaks - 1 else n

        if end - start < min_break_length:
          continue 

        break_point = np.random.randint(start, end - min_break_length + 1)

        if seasonal_period:
          phase = break_point % seasonal_period
          break_point -= phase

        break_length = np.random.randint(min_break_length, min(max_break_length, end - break_point + 1))
        break_end_point = break_point + break_length

        if seasonal_period:
          phase_end = break_end_point % seasonal_period
          break_end_point -= phase_end
          break_end_point = min(break_end_point, n)

        variance_factor = np.random.uniform(1.5, 3)

        if sign == 1:
          residuals[break_point:break_end_point] *= variance_factor * scale_factor
        elif sign == -1:
          residuals[:break_point] *= variance_factor * scale_factor
          residuals[break_end_point:] *= variance_factor * scale_factor
        else:
          raise ValueError("Invalid sign. Expected 1 or -1.")

        created_breaks.append((break_point, break_end_point))

    series = residuals + trend
    series = self.z_normalize(series)

    for i, (break_start, break_end) in enumerate(created_breaks):
      df.loc[break_start:break_end, 'variance_shift'] = 1

    df.loc[:,'data'] = series
    return df

  def generate_trend_shift(self, df, sign=None, location=None, num_breaks=1, change_type=None, seasonal_period = None, scale_factor = 1):
    series = df['data'].copy()
    noise = np.random.uniform(0.1, 1.5)
    n = len(series)
    mean = np.mean(series)
    if seasonal_period:
      seasonality = self.extract_seasonal_part(series, seasonal_period)
    positive_slopes = [0.1, 0.2, 0.3, 0.4, 0.5]
    negative_slopes = [-0.1, -0.2, -0.3, -0.4, -0.5]
    min_break_length = int(0.1 * n)
    max_break_length = int(0.2 * n)
    slope_change_factor = np.random.uniform(3, 6)
    created_breaks = []

    if num_breaks == 1 and location:
      if location == "beginning":
        start, end = int(0.1 * n), int(0.3 * n)
      elif location == "middle":
        start, end = int(0.3 * n), int(0.7 * n)
      elif location == "end":
        start, end = int(0.7 * n), int(0.9 * n)
      else:
        raise ValueError("Invalid location. Expected 'beginning', 'middle', or 'end'.")

      break_point = np.random.randint(start, end)
      if seasonal_period:
        phase = break_point % seasonal_period
        break_point -= phase

      break_length =  np.random.randint(int(0.1 * n), n - break_point)
      break_end_point = break_point + break_length

      if seasonal_period:
        phase_end = break_end_point % seasonal_period
        break_end_point -= phase_end
        break_end_point += seasonal_period
        break_end_point = min(break_end_point, n)

      if change_type == 'direction_change':
        if sign == 1:
          slope = np.random.choice(positive_slopes)
          series += slope * np.arange(n)
          residuals, trend = hpfilter(series, lamb=11000)
          offset = series[break_point]
          series[break_point:break_end_point] = offset + (-slope * np.arange(break_end_point-break_point)) + residuals[break_point:break_end_point]
          post_break_offset = series[break_end_point - 1]
          series[break_end_point:] = post_break_offset + (slope * np.arange(1, n - break_end_point + 1)) + residuals[break_end_point:]
        elif sign == -1:
          slope = np.random.choice(negative_slopes)
          series += slope * np.arange(n)
          residuals, trend = hpfilter(series, lamb=11000)
          offset = series[break_point]
          series[break_point:break_end_point] = offset + (-slope * np.arange(break_end_point-break_point) + residuals[break_point:break_end_point] )
          post_break_offset = series[break_end_point - 1]
          series[break_end_point:] = post_break_offset + (slope * np.arange(1, n - break_end_point + 1) + residuals[break_end_point:])
        else:
          raise ValueError("Invalid sign. Expected '1' or '-1'.")

      elif change_type == 'magnitude_change':
        if sign == 1:
          slope = np.random.choice(positive_slopes)
          series += slope * np.arange(n)
          residuals, trend = hpfilter(series, lamb=11000)
          offset = series[break_point]
          series[break_point:break_end_point] = offset + (slope * slope_change_factor * np.arange(break_end_point-break_point) * scale_factor + residuals[break_point:break_end_point])
          post_break_offset = series[break_end_point - 1]
          series[break_end_point:] = post_break_offset + (slope * np.arange(1, n - break_end_point + 1) + residuals[break_end_point:])
        elif sign == -1:
          slope = np.random.choice(negative_slopes)
          series += slope * np.arange(n)
          residuals, trend = hpfilter(series, lamb=11000)
          offset = series[break_point]
          series[break_point:break_end_point] = offset + (slope * slope_change_factor * np.arange(break_end_point-break_point) * scale_factor + residuals[break_point:break_end_point])
          post_break_offset = series[break_end_point - 1]
          series[break_end_point:] = post_break_offset + (slope * np.arange(1, n - break_end_point + 1) + residuals[break_end_point:])
        else:
          raise ValueError("Invalid sign. Expected '1' or '-1'.")

      elif change_type == 'direction_and_magnitude_change':
        if sign == 1:
          slope = np.random.choice(positive_slopes)
          series += slope * np.arange(n)
          residuals, trend = hpfilter(series, lamb=11000)
          offset = series[break_point]
          series[break_point:break_end_point] = offset + (-slope * slope_change_factor * np.arange(break_end_point-break_point) * scale_factor + residuals[break_point:break_end_point])
          post_break_offset = series[break_end_point - 1]
          series[break_end_point:] = post_break_offset + (slope * np.arange(1, n - break_end_point + 1) + residuals[break_end_point:])
        elif sign == -1:
          slope = np.random.choice(negative_slopes)
          series += slope * np.arange(n)
          residuals, trend = hpfilter(series, lamb=11000)
          offset = series[break_point]
          series[break_point:break_end_point] = offset + (-slope * slope_change_factor * np.arange(break_end_point-break_point) * scale_factor + residuals[break_point:break_end_point])
          post_break_offset = series[break_end_point - 1]
          series[break_end_point:] = post_break_offset + (slope * np.arange(1, n - break_end_point + 1) + residuals[break_end_point:])
        else:
          raise ValueError("Invalid sign. Expected '1' or '-1'.")

      else:
        raise ValueError("Invalid change_type. Expected 'direction_change', 'magnitude_change' or 'direction_and_magnitude_change.")

      created_breaks.append((break_point, break_end_point))

    else:
      segment_size = n // num_breaks
      positive_slope = np.random.choice(positive_slopes)  
      negative_slope = np.random.choice(negative_slopes)
        
      for i in range(num_breaks):
        start = i * segment_size
        end = (i + 1) * segment_size if i < num_breaks - 1 else n

        if end - start < min_break_length:
          continue

        break_point = np.random.randint(start, end - min_break_length + 1)
        if seasonal_period:
          phase = break_point % seasonal_period
          break_point -= phase

        break_length = np.random.randint(min_break_length, min(max_break_length, end - break_point + 1))
        break_end_point = break_point + break_length

        if seasonal_period:
          phase_end = break_end_point % seasonal_period
          break_end_point -= phase_end
          break_end_point += seasonal_period
          break_end_point = min(break_end_point, n)

        if change_type == 'direction_change':
          if sign == 1:
            slope = positive_slope
            series += slope * np.arange(n)
            residuals, trend = hpfilter(series, lamb=11000)
            offset = series[break_point]
            series[break_point:break_end_point] = offset + (-slope * slope_change_factor * np.arange(break_end_point-break_point)) + residuals[break_point:break_end_point]
            post_break_offset = series[break_end_point - 1]
            series[break_end_point:] = post_break_offset + (slope * np.arange(1, n - break_end_point + 1)) + residuals[break_end_point:]
          elif sign == -1:
            slope = negative_slope
            series += slope * np.arange(n)
            residuals, trend = hpfilter(series, lamb=11000)
            offset = series[break_point]
            series[break_point:break_end_point] = offset + (-slope * slope_change_factor * np.arange(break_end_point-break_point)) + residuals[break_point:break_end_point]
            post_break_offset = series[break_end_point - 1]
            series[break_end_point:] = post_break_offset + (slope * np.arange(1, n - break_end_point + 1)) + residuals[break_end_point:]
          else:
            raise ValueError("Invalid sign. Expected '1' or '-1'.")

        elif change_type == 'magnitude_change':
          if sign == 1:
            slope = positive_slope
            series += slope * np.arange(n)
            residuals, trend = hpfilter(series, lamb=11000)
            offset = series[break_point]
            series[break_point:break_end_point] = offset + (slope * slope_change_factor * np.arange(break_end_point-break_point) * scale_factor) + residuals[break_point:break_end_point]
            post_break_offset = series[break_end_point - 1]
            series[break_end_point:] = post_break_offset + (slope * np.arange(1, n - break_end_point + 1)) + residuals[break_end_point:]
          elif sign == -1:
            slope = negative_slope
            series += slope * np.arange(n)
            residuals, trend = hpfilter(series, lamb=11000)
            offset = series[break_point]
            series[break_point:break_end_point] = offset + (slope * slope_change_factor * np.arange(break_end_point-break_point) * scale_factor) + residuals[break_point:break_end_point]
            post_break_offset = series[break_end_point - 1]
            series[break_end_point:] = post_break_offset + (slope * np.arange(1, n - break_end_point + 1)) + residuals[break_end_point:]
          else:
            raise ValueError("Invalid sign. Expected '1' or '-1'.")

        elif change_type == 'direction_and_magnitude_change':
          if sign == 1:
            slope = positive_slope
            series += slope * np.arange(n)
            residuals, trend = hpfilter(series, lamb=11000)
            offset = series[break_point]
            series[break_point:break_end_point] = offset + (-slope * slope_change_factor * np.arange(break_end_point-break_point) * scale_factor) + residuals[break_point:break_end_point]
            post_break_offset = series[break_end_point - 1]
            series[break_end_point:] = post_break_offset + (slope * np.arange(1, n - break_end_point + 1)) + residuals[break_end_point:]
          elif sign == -1:
            slope = negative_slope
            series += slope * np.arange(n)
            residuals, trend = hpfilter(series, lamb=11000)
            offset = series[break_point]
            series[break_point:break_end_point] = offset + (-slope * slope_change_factor * np.arange(break_end_point-break_point) * scale_factor) + residuals[break_point:break_end_point]
            post_break_offset = series[break_end_point - 1]
            series[break_end_point:] = post_break_offset + (slope * np.arange(1, n - break_end_point + 1)) + residuals[break_end_point:]
          else:
            raise ValueError("Invalid sign. Expected '1' or '-1'.")

        else:
          raise ValueError("Invalid change_type. Expected 'direction_change', 'magnitude_change' or 'direction_and_magnitude_change'.")

        created_breaks.append((break_point, break_end_point))

    if seasonal_period:
      series[:break_point] += seasonality[:break_point] * 3
      series[break_point:] += seasonality[break_point:] * 4
      series += noise
      series = self.z_normalize(series)
    else:
      series += noise
      series = self.z_normalize(series)

    for i, (break_start, break_end) in enumerate(created_breaks):
      df.loc[break_start:break_end, 'trend_shift'] = 1

    df.loc[:,'data'] = series
    return df

  def generate_gradual_mean_shift(self, df, sign=None, location=None, num_breaks=1, scale_factor=2):
    series = df['data'].copy()
    noise = np.random.uniform(0.1, 1.5)
    n = len(series)
    mean = np.mean(series)
    std = np.std(series)
    residuals, trend = hpfilter(series, lamb=11000)

    min_break_length = int(0.1 * n)
    max_break_length = int(0.2 * n)

    created_breaks = []

    if num_breaks == 1 and location:
      if location == "beginning":
        start, end = int(0.1 * n), int(0.3 * n)
      elif location == "middle":
        start, end = int(0.3 * n), int(0.7 * n)
      elif location == "end":
        start, end = int(0.7 * n), int(0.9 * n) 
      else:
        raise ValueError("Invalid location. Expected 'beginning', 'middle', or 'end'.")

      break_point = np.random.randint(start, end)
      break_length = np.random.randint(int(0.1 * n), n - break_point)
      break_end_point = break_point + break_length

      transition_length = int(0.2 * break_length)
      main_shift_length = break_length - 2 * transition_length

      if main_shift_length <= 0:
        raise ValueError("Break length is too small for meaningful transitions.")

      magnitude = np.random.uniform(0.5, 2)
      level_shift = sign * magnitude * scale_factor

      start_transition = np.linspace(0, level_shift, transition_length)
      main_shift = np.ones(main_shift_length) * level_shift
      end_transition = np.linspace(level_shift, 0, transition_length)

      total_shift = np.concatenate([start_transition, main_shift, end_transition])

      trend[break_point:break_point + len(total_shift)] += total_shift
      created_breaks.append((break_point, break_point + len(total_shift)))

    else:
      segment_size = n // num_breaks
      for i in range(num_breaks):
        start = i * segment_size
        end = (i + 1) * segment_size if i < num_breaks - 1 else n

        if end - start < min_break_length:
          continue

        break_point = np.random.randint(start, end - min_break_length + 1)
        max_possible_length = int(n - break_point)
        break_length = np.random.randint(min_break_length, min(max_possible_length, max_break_length) + 1)
        break_end_point = break_point + break_length

        transition_length = int(0.1 * break_length)
        main_shift_length = break_length - 2 * transition_length

        if main_shift_length <= 0:
          continue

        magnitude = np.random.uniform(0.5, 2)
        level_shift = sign * magnitude * scale_factor

        start_transition = np.linspace(0, level_shift, transition_length)
        main_shift = np.ones(main_shift_length) * level_shift
        end_transition = np.linspace(level_shift, 0, transition_length)

        total_shift = np.concatenate([start_transition, main_shift, end_transition])

        trend[break_point:break_point + len(total_shift)] += total_shift
        created_breaks.append((break_point, break_point + len(total_shift)))

    series = residuals + trend
    series += noise
    series = self.z_normalize(series)

    for i, (break_start, break_end) in enumerate(created_breaks):
      df.loc[break_start:break_end, 'mean_shift'] = 1

    df.loc[:,'data'] = series
    return df

  def generate_gradual_variance_shift(self, df, sign=None, location=None, num_breaks=1, scale_factor=1):
    series = df['data'].copy()
    n = len(series)
    mean = np.mean(series)
    std = np.std(series)
    residuals, trend = hpfilter(series, lamb=11000)

    min_break_length = int(0.1 * n)
    max_break_length = int(0.2 * n)

    created_breaks = []

    if num_breaks == 1 and location:
      if location == "beginning":
        start, end = int(0.1 * n), int(0.3 * n)
      elif location == "middle":
        start, end = int(0.3 * n), int(0.7 * n)
      elif location == "end":
        start, end = int(0.7 * n), int(0.9 * n)
      else:
        raise ValueError("Invalid location. Expected 'beginning', 'middle', or 'end'.")

      break_point = np.random.randint(start, end)
      break_length = np.random.randint(int(0.1 * n), n - break_point)
      break_end_point = break_point + break_length

      transition_length = int(0.2 * break_length)
      main_shift_length = break_length - 2 * transition_length

      if main_shift_length <= 0:
        raise ValueError("Break length is too small for meaningful transitions.")

      target_variance_factor = np.random.uniform(1.5, 3)

      if sign == 1:
        start_transition = np.linspace(1, target_variance_factor, transition_length)
        main_shift = np.ones(main_shift_length) * target_variance_factor
        end_transition = np.linspace(target_variance_factor, 1, transition_length)

        total_shift = np.concatenate([start_transition, main_shift, end_transition])

        residuals[break_point:break_point + len(total_shift)] *= total_shift
      elif sign == -1:
        start_transition = np.linspace(1/target_variance_factor, 1, transition_length)
        main_shift = np.ones(main_shift_length) * 1/target_variance_factor
        end_transition = np.linspace(1, 1/target_variance_factor, transition_length)

        total_shift = np.concatenate([start_transition, main_shift, end_transition])

        residuals[break_point:break_point + len(total_shift)] *= total_shift
      else:
        raise ValueError("Invalid sign. Expected 1 or -1.")

      created_breaks.append((break_point, break_point + len(total_shift)))

    else:
      segment_size = n // num_breaks
      for i in range(num_breaks):
        start = i * segment_size
        end = (i + 1) * segment_size if i < num_breaks - 1 else n

        if end - start < min_break_length:
          continue 

        break_point = np.random.randint(start, end - min_break_length + 1)
        max_possible_length = int(n - break_point)
        break_length = np.random.randint(min_break_length, min(max_possible_length, max_break_length) + 1)
        break_end_point = break_point + break_length

        transition_length = int(0.2 * break_length)
        main_shift_length = break_length - 2 * transition_length

        if main_shift_length <= 0:
          raise ValueError("Break length is too small for meaningful transitions.")

        target_variance_factor = np.random.uniform(1.5, 3)

        if sign == 1:
          start_transition = np.linspace(1, target_variance_factor, transition_length)
          main_shift = np.ones(main_shift_length) * target_variance_factor
          end_transition = np.linspace(target_variance_factor, 1, transition_length)

          total_shift = np.concatenate([start_transition, main_shift, end_transition])

          residuals[break_point:break_point + len(total_shift)] *= total_shift
        elif sign == -1:
          start_transition = np.linspace(1/target_variance_factor, 1, transition_length)
          main_shift = np.ones(main_shift_length) * 1/target_variance_factor
          end_transition = np.linspace(1, 1/target_variance_factor, transition_length)

          total_shift = np.concatenate([start_transition, main_shift, end_transition])

          residuals[break_point:break_point + len(total_shift)] *= total_shift
        else:
          raise ValueError("Invalid sign. Expected 1 or -1.")

        created_breaks.append((break_point, break_point + len(total_shift)))

    series = residuals + trend
    series = self.z_normalize(series)

    for i, (break_start, break_end) in enumerate(created_breaks):
      df.loc[break_start:break_end, 'variance_shift'] = 1

    df.loc[:,'data'] = series
    return df

  def generate_gradual_trend_shift(self, df, sign=None, location=None, num_breaks=1, change_type=None, scale_factor=1):
    series = df['data'].copy()
    n = len(series)
    noise = np.random.uniform(0.1, 1.5)
    positive_slopes = [0.1, 0.2, 0.3, 0.4, 0.5]
    negative_slopes = [-0.1, -0.2, -0.3, -0.4, -0.5]
    min_break_length = int(0.1 * n) 
    max_break_length = int(0.2 * n)
    slope_change_factor = np.random.uniform(5, 10)
    created_breaks = []

    if num_breaks == 1 and location:
      if location == "beginning":
        start, end = int(0.1 * n), int(0.3 * n)
      elif location == "middle":
        start, end = int(0.3 * n), int(0.7 * n)
      elif location == "end":
        start, end = int(0.7 * n), int(0.9 * n)
      else:
        raise ValueError("Invalid location. Expected 'beginning', 'middle', or 'end'.")

      break_point = np.random.randint(start, end)
      break_length = np.random.randint(int(0.1 * n), n - break_point)
      break_end_point = break_point + break_length

      transition_length = int(0.2 * break_length)
      main_shift_length = break_length - 2 * transition_length

      if main_shift_length <= 0:
        raise ValueError("Break length is too small for meaningful transitions.")

      if change_type == 'direction_change':
        if sign == 1:
          slope = np.random.choice(positive_slopes)
          series += slope * np.arange(n)
          residuals, trend = hpfilter(series, lamb=11000)
          offset = series[break_point]
          transition_slope = np.linspace(slope, -slope, break_length)
          trend[break_point:break_end_point] = offset + (transition_slope * np.arange(break_end_point-break_point)) * scale_factor + residuals[break_point:break_end_point]
          post_break_offset = series[break_end_point - 1]
          transition_slope_2 = np.linspace(-slope, slope, n - break_end_point)
          trend[break_end_point:] = post_break_offset + np.cumsum(transition_slope_2) * scale_factor + residuals[break_end_point:]
        elif sign == -1:
          slope = np.random.choice(negative_slopes)
          series += slope * np.arange(n)
          residuals, trend = hpfilter(series, lamb=11000)
          offset = series[break_point]
          transition_slope = np.linspace(slope, -slope, break_length)
          trend[break_point:break_end_point] = offset + (transition_slope * np.arange(break_end_point-break_point)) * scale_factor + residuals[break_point:break_end_point]
          post_break_offset = series[break_end_point - 1]
          transition_slope_2 = np.linspace(-slope, slope, n - break_end_point)
          trend[break_end_point:] = post_break_offset + np.cumsum(transition_slope_2) * scale_factor + residuals[break_end_point:]
        else:
          raise ValueError("Invalid sign. Expected '1' or '-1'.")

      elif change_type == 'magnitude_change':
        if sign == 1:
          slope = np.random.choice(positive_slopes)
          series += slope * np.arange(n)
          residuals, trend = hpfilter(series, lamb=11000)
          offset = series[break_point]
          transition_slope = np.linspace(slope, slope * slope_change_factor, break_length)
          trend[break_point:break_end_point] = offset + (transition_slope * np.arange(break_end_point-break_point)) * scale_factor + residuals[break_point:break_end_point]
          post_break_offset = series[break_end_point - 1]
          transition_slope_2 = np.linspace(slope * slope_change_factor, slope, n - break_end_point)
          trend[break_end_point:] = post_break_offset + np.cumsum(transition_slope_2) * scale_factor + residuals[break_end_point:]
        elif sign == -1:
          slope = np.random.choice(negative_slopes)
          series += slope * np.arange(n)
          residuals, trend = hpfilter(series, lamb=11000)
          offset = series[break_point]
          transition_slope = np.linspace(slope, slope * slope_change_factor, break_length)
          trend[break_point:break_end_point] = offset + (transition_slope * np.arange(break_end_point-break_point)) * scale_factor + residuals[break_point:break_end_point]
          post_break_offset = series[break_end_point - 1]
          transition_slope_2 = np.linspace(slope * slope_change_factor, slope, n - break_end_point)
          trend[break_end_point:] = post_break_offset + np.cumsum(transition_slope_2) * scale_factor + residuals[break_end_point:]
        else:
          raise ValueError("Invalid sign. Expected '1' or '-1'.")

      elif change_type == 'direction_and_magnitude_change':
        if sign == 1:
          slope = np.random.choice(positive_slopes)
          series += slope * np.arange(n)
          residuals, trend = hpfilter(series, lamb=11000)
          offset = series[break_point]
          transition_slope = np.linspace(slope, -slope * slope_change_factor, break_length)
          trend[break_point:break_end_point] = offset + (transition_slope * np.arange(break_end_point-break_point)) * scale_factor + residuals[break_point:break_end_point]
          post_break_offset = series[break_end_point - 1]
          transition_slope_2 = np.linspace(-slope * slope_change_factor, slope, n - break_end_point)
          trend[break_end_point:] = post_break_offset + np.cumsum(transition_slope_2) * scale_factor + residuals[break_end_point:]
        elif sign == -1:
          slope = np.random.choice(negative_slopes)
          series += slope * np.arange(n)
          residuals, trend = hpfilter(series, lamb=11000)
          offset = series[break_point]
          transition_slope = np.linspace(slope, -slope * slope_change_factor, break_length)
          trend[break_point:break_end_point] = offset + (transition_slope * np.arange(break_end_point-break_point)) * scale_factor + residuals[break_point:break_end_point]
          post_break_offset = series[break_end_point - 1]
          transition_slope_2 = np.linspace(-slope * slope_change_factor, slope, n - break_end_point)
          trend[break_end_point:] = post_break_offset + np.cumsum(transition_slope_2) * scale_factor + residuals[break_end_point:]
        else:
          raise ValueError("Invalid sign. Expected '1' or '-1'.")

      else:
        raise ValueError("Invalid change_type. Expected 'direction_change' or 'magnitude_change' or 'direction_and_magnitude_change.")

      created_breaks.append((break_point, break_end_point))

    else:
      segment_size = n // num_breaks

      for i in range(num_breaks):
        start = i * segment_size
        end = (i + 1) * segment_size if i < num_breaks - 1 else n

        if end - start < min_break_length:
          continue

        break_point = np.random.randint(start, end - min_break_length + 1)
        break_length = np.random.randint(min_break_length, min(max_break_length, end - break_point + 1))
        break_end_point = break_point + break_length

        if change_type == 'direction_change':
          if sign == 1:
            slope = np.random.choice(positive_slopes)
            series += slope * np.arange(n)
            offset = series[break_point]
            transition_slope = np.linspace(slope, -slope, break_length)
            series[break_point:break_end_point] = offset + (transition_slope * np.arange(break_end_point-break_point)) * scale_factor
            post_break_offset = series[break_end_point - 1]
            transition_slope_2 = np.linspace(-slope, slope, n - break_end_point)
            series[break_end_point:] = post_break_offset + np.cumsum(transition_slope_2) * scale_factor
          elif sign == -1:
            slope = np.random.choice(negative_slopes)
            series += slope * np.arange(n)
            offset = series[break_point]
            transition_slope = np.linspace(slope, -slope, break_length)
            series[break_point:break_end_point] = offset + (transition_slope * np.arange(break_end_point-break_point)) * scale_factor
            post_break_offset = series[break_end_point - 1]
            transition_slope_2 = np.linspace(-slope, slope, n - break_end_point)
            series[break_end_point:] = post_break_offset + np.cumsum(transition_slope_2) * scale_factor
          else:
            raise ValueError("Invalid sign. Expected '1' or '-1'.")

        elif change_type == 'magnitude_change':
          if sign == 1:
            slope = np.random.choice(positive_slopes)
            series += slope * np.arange(n)
            offset = series[break_point]
            transition_slope = np.linspace(slope, slope * slope_change_factor, break_length)
            series[break_point:break_end_point] = offset + (transition_slope * np.arange(break_end_point-break_point)) * scale_factor
            post_break_offset = series[break_end_point - 1]
            transition_slope_2 = np.linspace(slope * slope_change_factor, slope, n - break_end_point)
            series[break_end_point:] = post_break_offset + np.cumsum(transition_slope_2) * scale_factor
          elif sign == -1:
            slope = np.random.choice(negative_slopes)
            series += slope * np.arange(n)
            offset = series[break_point]
            transition_slope = np.linspace(slope, slope * slope_change_factor, break_length)
            series[break_point:break_end_point] = offset + (transition_slope * np.arange(break_end_point-break_point)) * scale_factor
            post_break_offset = series[break_end_point - 1]
            transition_slope_2 = np.linspace(slope * slope_change_factor, slope, n - break_end_point)
            series[break_end_point:] = post_break_offset + np.cumsum(transition_slope_2) * scale_factor
          else:
            raise ValueError("Invalid sign. Expected '1' or '-1'.")

        elif change_type == 'direction_and_magnitude_change':
          if sign == 1:
            slope = np.random.choice(positive_slopes)
            series += slope * np.arange(n)
            offset = series[break_point]
            transition_slope = np.linspace(slope, -slope * slope_change_factor, break_length)
            series[break_point:break_end_point] = offset + (transition_slope * np.arange(break_end_point-break_point)) * scale_factor
            post_break_offset = series[break_end_point - 1]
            transition_slope_2 = np.linspace(-slope * slope_change_factor, slope, n - break_end_point)
            series[break_end_point:] = post_break_offset + np.cumsum(transition_slope_2) * scale_factor
          elif sign == -1:
            slope = np.random.choice(positive_slopes)
            series += slope * np.arange(n)
            offset = series[break_point]
            transition_slope = np.linspace(slope, -slope * slope_change_factor, break_length)
            series[break_point:break_end_point] = offset + (transition_slope * np.arange(break_end_point-break_point)) * scale_factor
            post_break_offset = series[break_end_point - 1]
            transition_slope_2 = np.linspace(-slope * slope_change_factor, slope, n - break_end_point)
            series[break_end_point:] = post_break_offset + np.cumsum(transition_slope_2) * scale_factor
          else:
            raise ValueError("Invalid sign. Expected '1' or '-1'.")

        else:
          raise ValueError("Invalid change_type. Expected 'direction_change' or 'magnitude_change'.")

        created_breaks.append((break_point, break_end_point))

    series += noise
    series = self.z_normalize(series)

    for i, (break_start, break_end) in enumerate(created_breaks):
      df.loc[break_start:break_end, 'trend_shift'] = 1

    df.loc[:,'data'] = series
    return df
      
# def convert_to_tensor(data):
#   series_list = []
#   labels_list = []

#   for _, _, _, df in data:
#     df['data'] = (df['data'].values.reshape(-1, 1)).flatten()
#     series_list.append(torch.tensor(df['data'].values, dtype=torch.float32))
#     labels_list.append(torch.tensor(df.drop('data', axis=1).values, dtype=torch.float32))

#   return series_list, labels_list

def save_data(dataframes, filename):
    saved_data = [a.astype({
        'data': np.float32,
        "mean_shift": bool,
        "variance_shift": bool,
        "trend_shift": bool,
        "anomaly": bool,
    }) for a in dataframes]
    
    with open(filename, "wb") as f:
        pickle.dump(saved_data, f)