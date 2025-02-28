"""
Main TimeSeriesGenerator class implementation.

This module provides the TimeSeriesGenerator class which can generate synthetic time series
with various characteristics including trends, seasonality, and structural breaks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.filters.hp_filter import hpfilter
from numpy.polynomial.polynomial import Polynomial
import random

class TimeSeriesGenerator:
    def __init__(self, length=None, random_seed=None):
        """
        Initialize the TimeSeriesGenerator class.
        
        Parameters:
            length (int, optional): The length of the time series to generate. Defaults to 400 if None is provided.
            random_seed (int, optional): Seed for the random number generator. Defaults to None.
        
        Description:
            Constructor that sets up the generator with the specified length and initializes dictionaries for various
            time series characteristics and structural breaks, mapping them to their respective generator methods.
        """
        self.length = length if length is not None else 400
        self.random_seed = random_seed
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
        self.base_distributions = ['white_noise', 'ar', 'ma', 'arma']
        self.characteristics = {
            'deterministic_trend_linear': self.generate_deterministic_trend_linear,
            'deterministic_trend_cubic': self.generate_deterministic_trend_cubic,
            'deterministic_trend_quadratic': self.generate_deterministic_trend_quadratic,
            'stochastic_trend': self.generate_stochastic_trend,
            'seasonality': self.generate_seasonality
        }
        self.structural_breaks = {
            'mean_shift': self.generate_mean_shift,
            'variance_shift': self.generate_variance_shift,
            'trend_shift': self.generate_trend_shift,
            'gradual_change_mean': self.generate_gradual_mean_shift,
            'gradual_change_variance': self.generate_gradual_variance_shift,
            'gradual_change_trend': self.generate_gradual_trend_shift
        }



    def z_normalize(self, series):
        """
        Normalize a time series using z-score normalization.
        
        Parameters:
            series (numpy.ndarray): The time series to normalize.
        
        Returns:
            numpy.ndarray: Z-normalized time series with mean 0 and standard deviation 1.
        
        Description:
            Standardizes the input series by subtracting the mean and dividing by the standard deviation.
        """
        return (series - np.mean(series)) / np.std(series)

    def is_stationary(self, ar_params):
        """
        Check if AR parameters lead to a stationary process.
        
        Parameters:
            ar_params (numpy.ndarray): Autoregressive parameters to check.
        
        Returns:
            bool: True if the AR process is stationary, False otherwise.
        
        Description:
            A time series is stationary if all roots of the AR polynomial lie outside the unit circle.
            This function verifies this condition for the given AR parameters.
        """
        ar_poly = np.r_[1, -ar_params]
        roots = Polynomial(ar_poly).roots()
        return np.all(np.abs(roots) > 1)

    def is_invertible(self, ma_params):
        """
        Check if MA parameters lead to an invertible process.
        
        Parameters:
            ma_params (numpy.ndarray): Moving average parameters to check.
        
        Returns:
            bool: True if the MA process is invertible, False otherwise.
        
        Description:
            A moving average process is invertible if all roots of the MA polynomial lie outside the unit circle.
            This function verifies this condition for the given MA parameters.
        """
        ma_poly = np.r_[1, ma_params]
        roots = Polynomial(ma_poly).roots()
        return np.all(np.abs(roots) > 1)

    def extract_seasonal_part(self, series, period):
        """
        Extract the seasonal component from a time series.
        
        Parameters:
            series (numpy.ndarray): The time series to decompose.
            period (int): The periodicity of the seasonal component.
        
        Returns:
            numpy.ndarray: The extracted seasonal component.
        
        Description:
            Uses seasonal decomposition to separate the seasonal component from the time series.
        """
        decomposition = seasonal_decompose(series, model='additive', period=period)
        seasonal = decomposition.seasonal
        return seasonal

    def generate_ar_params(self, order_range=(1, 3), coef_range=(-0.3, 0.3)):
        """
        Generate random parameters for an autoregressive (AR) process.
        
        Parameters:
            order_range (tuple, optional): Range of possible AR orders (min, max). Defaults to (1, 3).
            coef_range (tuple, optional): Range of possible coefficient values. Defaults to (-0.3, 0.3).
        
        Returns:
            tuple: (order, coefficients) where order is the AR order and coefficients are the AR parameters.
        
        Description:
            Generates random AR parameters that ensure the resulting process is stationary.
            Continues generating parameters until a stationary set is found.
        """
        
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
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
        """
        Generate random parameters for a moving average (MA) process.
        
        Parameters:
            order_range (tuple, optional): Range of possible MA orders (min, max). Defaults to (1, 3).
            coef_range (tuple, optional): Range of possible coefficient values. Defaults to (-0.3, 0.3).
        
        Returns:
            tuple: (order, coefficients) where order is the MA order and coefficients are the MA parameters.
        
        Description:
            Generates random MA parameters that ensure the resulting process is invertible.
            Continues generating parameters until an invertible set is found.
        """

        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
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
        """
        Generate random parameters for an ARMA (autoregressive moving average) process.
        
        Parameters:
            order_range (tuple, optional): Range of possible AR and MA orders (min, max). Defaults to (1, 3).
            coef_range (tuple, optional): Range of possible coefficient values. Defaults to (-0.3, 0.3).
        
        Returns:
            tuple: (ar_order, ma_order, ar_coefs, ma_coefs) containing the orders and coefficients for both components.
        
        Description:
            Generates random ARMA parameters that ensure the resulting process is both stationary and invertible.
            Continues generating parameters until a valid set is found.
        """
                
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
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
        """
        Generate random parameters for a SARIMA (Seasonal ARIMA) process.
        
        Parameters:
            p_range (tuple, optional): Range for the AR order (min, max). Defaults to (1, 3).
            d_range (tuple, optional): Range for the differencing order (min, max). Defaults to (0, 1).
            q_range (tuple, optional): Range for the MA order (min, max). Defaults to (1, 3).
            seasonal_order (tuple, optional): The seasonal parameters (P, D, Q, s). Defaults to (1, 1, 1, 12).
            coef_range (tuple, optional): Range of possible coefficient values. Defaults to (-0.2, 0.2).
        
        Returns:
            tuple: ((p, d, q), (P, D, Q, s), params) containing the non-seasonal orders, seasonal orders, and all parameters.
        
        Description:
            Generates random SARIMA parameters that ensure stationarity and invertibility conditions are met
            for both the regular and seasonal components.
        """

        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
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

    def generate_white_noise(self, noise_scale=0.05):
        """
        Generate Gaussian white noise.
        
        Parameters:
            length (int): The length of the time series to generate.
            noise_scale (float, optional): The standard deviation of the noise. Defaults to 0.05.
        
        Returns:
            numpy.ndarray: The generated white noise series.
        
        Description:
            Creates a series of independent and identically distributed random variables
            with mean 0 and standard deviation determined by noise_scale.
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
        series = np.random.normal(0, 0.05, self.length)
        series = series * noise_scale
        return series

    def generate_ar_series(self, noise_scale=0.5):
        """
        Generate a time series from an autoregressive (AR) process.
        
        Parameters:
            length (int): The length of the time series to generate.
            noise_scale (float, optional): Scale factor for the variance of the process. Defaults to 0.5.
        
        Returns:
            numpy.ndarray: The generated AR time series.
        
        Description:
            Creates a stationary AR time series using randomly generated parameters within specified ranges.
            The series is scaled by noise_scale to control its variance.
        """
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
        order, coefs = self.generate_ar_params()
        ar = np.r_[1, -np.array(coefs)]  # leading 1 and negate the coefficients
        ma = np.r_[1]  # MA coefficients are just [1] for a pure AR process
        ar_process = ArmaProcess(ar, ma)
        series = ar_process.generate_sample(nsample=self.length)
        series = series * noise_scale
        return series

    def generate_ma_series(self, noise_scale=0.5):
        """
        Generate a time series from a moving average (MA) process.
        
        Parameters:
            length (int): The length of the time series to generate.
            noise_scale (float, optional): Scale factor for the variance of the process. Defaults to 0.5.
        
        Returns:
            numpy.ndarray: The generated MA time series.
        
        Description:
            Creates an invertible MA time series using randomly generated parameters within specified ranges.
            The series is scaled by noise_scale to control its variance.
        """
        order, coefs = self.generate_ma_params()
        ar = np.r_[1]  # AR coefficients are just [1] for a pure MA process
        ma = np.r_[1, np.array(coefs)]  # leading 1 for the MA coefficients
        arma_process = ArmaProcess(ar, ma)
        series = arma_process.generate_sample(nsample=self.length)
        series = series * noise_scale
        return series

    def generate_arma_series(self, noise_scale=0.5):
        """
        Generate a time series from an ARMA process.
        
        Parameters:
            length (int): The length of the time series to generate.
            noise_scale (float, optional): Scale factor for the variance of the process. Defaults to 0.5.
        
        Returns:
            numpy.ndarray: The generated ARMA time series.
        
        Description:
            Creates a stationary and invertible ARMA time series using randomly generated parameters.
            The series is scaled by noise_scale to control its variance.
        """
        ar_order, ma_order, ar_coefs, ma_coefs = self.generate_arma_params()
        ar = np.r_[1, -np.array(ar_coefs)]
        ma = np.r_[1, np.array(ma_coefs)]
        arma_process = ArmaProcess(ar, ma)
        series = arma_process.generate_sample(nsample=self.length)
        series = series * noise_scale
        return series
    
    def generate_sarima_series(self, noise_scale=0.5, max_attempts=10):
        """
        Generate a time series from a SARIMA process.
        
        Parameters:
            length (int): The length of the time series to generate.
            noise_scale (float, optional): Scale factor for the variance of the process. Defaults to 0.5.
            max_attempts (int, optional): Maximum number of attempts to generate a stable series. Defaults to 10.
        
        Returns:
            numpy.ndarray: The generated SARIMA time series.
        
        Raises:
            ValueError: If unable to generate a stable SARIMA series after max_attempts.
        
        Description:
            Creates a seasonal ARIMA time series using randomly generated parameters.
            Handles potential numerical instability issues by retrying with new parameters if necessary.
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        attempts = 0
        while attempts < max_attempts:
          try:     
                
            order, seasonal_order, params = self.generate_sarima_params()
            p, d, q = order
            P, D, Q, s = seasonal_order
    
            sarima_model = SARIMAX(endog=np.zeros(self.length), order=(p, d, q), seasonal_order=(P, D, Q, s))
    
            series = sarima_model.simulate(params=params, nsimulations=self.length)
            if not np.isnan(series).any():
              return series * noise_scale
    
          except (ValueError, np.linalg.LinAlgError):
            attempts += 1
            print(f"Attempt {attempts}/{max_attempts} failed due to numerical instability. Retrying with new parameters...")
    
        raise ValueError("Failed to generate a stable SARIMA series after multiple attempts.")
    
    def generate_base_series(self, distribution=None):
        """
        Generate a base time series with a specified distribution.
        
        Parameters:
            distribution (str, optional): The type of time series to generate ('white_noise', 'ar', 'ma', 'arma', 'sarima').
                                          If None, one is randomly chosen from available distributions.
        
        Returns:
            pandas.DataFrame: DataFrame containing the generated time series and columns for tracking structural breaks.
        
        Description:
            Creates a basic time series with the specified stochastic process.
            Initializes a DataFrame with the series data and columns for tracking different types of structural breaks.
        """
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
        """
        Add point anomalies (outliers) to a time series.
        
        Parameters:
            df (pandas.DataFrame): DataFrame containing the time series.
            scale_factor (float, optional): Multiplier to control the magnitude of anomalies. Defaults to 1.
        
        Returns:
            pandas.DataFrame: The DataFrame with added point anomalies and updated anomaly indicators.
        
        Description:
            Adds random point anomalies (1-5% of the series length) with magnitudes between 3-6 standard deviations.
            Updates the 'anomaly' column to indicate where anomalies were introduced.
        """
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
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
        """
        Add collective anomalies (segments of unusual behavior) to a time series.
        
        Parameters:
            df (pandas.DataFrame): DataFrame containing the time series.
            change_type (str, optional): Type of anomaly: 'mean', 'variance', or 'seasonal'.
            scale_factor (float, optional): Multiplier to control the magnitude of anomalies. Defaults to 1.
            seasonal_period (int, optional): Period for seasonal anomalies. Required if change_type is 'seasonal'.
        
        Returns:
            pandas.DataFrame: The DataFrame with added collective anomalies and updated anomaly indicators.
        
        Raises:
            ValueError: If change_type is not one of the expected values.
        
        Description:
            Adds a segment of unusual behavior (5-9% of series length) by modifying mean, variance,
            or seasonal patterns. Updates the 'anomaly' column to indicate where anomalies were introduced.
        """
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
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
        """
        Add a linear deterministic trend to a time series.
        
        Parameters:
            df (pandas.DataFrame): DataFrame containing the time series.
            sign (int, optional): Direction of the trend (positive or negative).
            slope (float, optional): Slope of the linear trend. If None, randomly chosen from predefined ranges.
            noise_std (float, optional): Standard deviation of the noise added to the trend. If None, random value between 0.1-1.5.
            intercept (float, optional): Y-intercept of the trend line. Defaults to 1.
            scale_factor (float, optional): Multiplier to control the magnitude of the trend. Defaults to 1.
        
        Returns:
            pandas.DataFrame: The DataFrame with the added linear trend.
        
        Description:
            Adds a linear trend (y = intercept + slope*x + noise) to the time series.
            Adjusts parameters based on series length for appropriate scaling.
            Z-normalizes the result to maintain consistent scale.
        """
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
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
        """
        Add a cubic deterministic trend to a time series.
        
        Parameters:
            df (pandas.DataFrame): DataFrame containing the time series.
            sign (int, optional): Direction of the trend (positive or negative).
            a (float, optional): Coefficient of the cubic term. If None, random value is generated.
            b (float, optional): Coefficient of the quadratic term. If None, random value is generated.
            c (float, optional): Coefficient of the linear term. If None, random value is generated.
            d (float, optional): Constant term. If None, random value is generated.
            noise_std (float, optional): Standard deviation of the noise added to the trend. If None, random value between 0.1-1.5.
            scale_factor (float, optional): Multiplier to control the magnitude of the trend. Defaults to 1.
        
        Returns:
            pandas.DataFrame: The DataFrame with the added cubic trend.
        
        Description:
            Adds a cubic trend (y = a*x³ + b*x² + c*x + d + noise) to the time series.
            Adjusts parameters based on series length for appropriate scaling.
            Z-normalizes the result to maintain consistent scale.
        """
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
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
        """
        Add a quadratic deterministic trend to a time series.
        
        Parameters:
            df (pandas.DataFrame): DataFrame containing the time series.
            sign (int, optional): Direction of the trend (positive or negative).
            a (float, optional): Coefficient of the quadratic term. If None, random value is generated.
            b (float, optional): Coefficient of the linear term. If None, random value is generated.
            c (float, optional): Constant term. If None, random value is generated.
            noise_std (float, optional): Standard deviation of the noise added to the trend. If None, random value between 0.1-1.5.
            scale_factor (float, optional): Multiplier to control the magnitude of the trend. Defaults to 1.
        
        Returns:
            pandas.DataFrame: The DataFrame with the added quadratic trend.
        
        Description:
            Adds a quadratic trend (y = a*x² + b*x + c + noise) to the time series.
            Adjusts parameters based on series length for appropriate scaling.
            Z-normalizes the result to maintain consistent scale.
        """
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
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
        """
        Generates a stochastic trend component for time series data.
        
        This method adds a random walk with drift to the existing time series data.
        The stochastic trend combines a deterministic linear trend (drift component)
        with a random walk component to create realistic non-stationary behavior.
        
        Parameters:
        -----------
        df : DataFrame
            DataFrame containing the time series data in a column named 'data'
        sign : int, optional
            Direction of the trend (positive or negative)
        drift : float, optional
            Slope of the deterministic trend component
        noise_std : float, optional
            Standard deviation of the random noise used in the random walk
        scale_factor : int, default=1
            Multiplier to adjust the magnitude of the trend
            
        Returns:
        --------
        DataFrame
            The input DataFrame with the 'data' column modified to include the stochastic trend
        """
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
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
        """
        Adds a seasonal component to the time series data.
        
        This method creates a sinusoidal pattern with optional noise to simulate
        seasonal patterns in the data. The seasonality is controlled by the period
        (cycle length) and amplitude (strength of the seasonal effect).
        
        Parameters:
        -----------
        df : DataFrame
            DataFrame containing the time series data in a column named 'data'
        period : int, optional
            Number of time steps in one complete seasonal cycle
        amplitude : float, optional
            Magnitude of the seasonal effect
        noise_std : float, optional
            Standard deviation of random noise added to the seasonal pattern
        scale_factor : int, default=3
            Multiplier to adjust the overall magnitude of the seasonal component
            
        Returns:
        --------
        tuple
            (period, DataFrame) - The seasonal period used and the modified DataFrame
        """
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
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
    
    def generate_mean_shift(self, df, sign=None, location=None, num_breaks=1, scale_factor=1, seasonal_period=None):
        """
        Introduces level shifts (structural breaks) in the time series.
        
        This method creates sudden or gradual changes in the mean level of the time series,
        which can represent regime changes, policy interventions, or other structural breaks.
        The shifts can be placed at specific locations or distributed throughout the series.
        
        Parameters:
        -----------
        df : DataFrame
            DataFrame containing the time series data in a column named 'data'
        sign : int, optional
            Direction of the mean shift (positive or negative)
        location : str, optional
            Where to place the shift: 'beginning', 'middle', or 'end'
        num_breaks : int, default=1
            Number of level shifts to introduce
        scale_factor : int, default=1
            Multiplier to adjust the magnitude of the shifts
        seasonal_period : int, optional
            If provided, aligns shifts with seasonal patterns and creates smooth transitions
            
        Returns:
        --------
        DataFrame
            The modified DataFrame with the 'data' column updated and a 'mean_shift' column
            marking the locations of the shifts
        """
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
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
    
            break_length = np.random.randint(int(0.1 * n), n - break_point)
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
        """
        Introduces variance shifts in the time series data.
        
        This method creates periods where the volatility (variance) of the time series changes,
        which can represent regime changes, market turbulence, or other structural changes in
        the data generating process. The method can create higher volatility in specific segments
        or in the surrounding periods, depending on the sign parameter.
        
        Parameters:
        -----------
        df : DataFrame
            DataFrame containing the time series data in a column named 'data'
        sign : int, optional
            Direction of the variance shift:
            1: Increases variance within the break period
            -1: Increases variance outside the break period
        location : str, optional
            Where to place the shift: 'beginning', 'middle', or 'end'
        num_breaks : int, default=1
            Number of variance shifts to introduce
        scale_factor : int, default=1
            Multiplier to adjust the magnitude of the variance shifts
        seasonal_period : int, optional
            If provided, aligns shifts with seasonal patterns
            
        Returns:
        --------
        DataFrame
            The modified DataFrame with the 'data' column updated and a 'variance_shift' column
            marking the locations of the shifts
        """
        
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
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
            # Increase variance within the break period
            residuals[break_point:break_end_point] *= variance_factor * scale_factor
          elif sign == -1:
            # Increase variance outside the break period
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
              # Increase variance within the break period
              residuals[break_point:break_end_point] *= variance_factor * scale_factor
            elif sign == -1:
              # Increase variance outside the break period
              residuals[:break_point] *= variance_factor * scale_factor
              residuals[break_end_point:] *= variance_factor * scale_factor
            else:
              raise ValueError("Invalid sign. Expected 1 or -1.")
    
            created_breaks.append((break_point, break_end_point))
    
        series = residuals + trend
        series = self.z_normalize(series)
    
        # Mark the variance shift periods in the DataFrame
        for i, (break_start, break_end) in enumerate(created_breaks):
          df.loc[break_start:break_end, 'variance_shift'] = 1
    
        df.loc[:,'data'] = series
        return df

    def generate_trend_shift(self, df, sign=None, location=None, num_breaks=1, change_type=None, seasonal_period=None, scale_factor=1):
        """
        Introduce trend shifts into a time series dataset.
    
        Parameters:
            df (pd.DataFrame): Input DataFrame containing the time series data in the 'data' column.
            sign (int, optional): Direction of the trend shift. 
                                  1 for a positive trend shift, -1 for a negative trend shift. Default is None.
            location (str, optional): Location of the trend shift in the time series. 
                                      Options are 'beginning', 'middle', or 'end'. Default is None.
            num_breaks (int, optional): Number of trend shifts to introduce. Default is 1.
            change_type (str, optional): Type of trend shift to apply. 
                                         Options are 'direction_change', 'magnitude_change', or 'direction_and_magnitude_change'. Default is None.
            seasonal_period (int, optional): Periodicity of the seasonality in the time series. 
                                             If provided, the function accounts for seasonality when applying trend shifts. Default is None.
            scale_factor (float, optional): Scaling factor for the intensity of the trend shift. 
                                            Larger values result in more pronounced shifts. Default is 1.
    
        Returns:
            pd.DataFrame: Modified DataFrame with trend shifts applied to the 'data' column and a new 'trend_shift' column indicating the regions of trend shifts.
        """
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
    
            break_length = np.random.randint(int(0.1 * n), n - break_point)
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
                    series[break_point:break_end_point] = offset + (-slope * np.arange(break_end_point - break_point)) + residuals[break_point:break_end_point]
                    post_break_offset = series[break_end_point - 1]
                    series[break_end_point:] = post_break_offset + (slope * np.arange(1, n - break_end_point + 1)) + residuals[break_end_point:]
                elif sign == -1:
                    slope = np.random.choice(negative_slopes)
                    series += slope * np.arange(n)
                    residuals, trend = hpfilter(series, lamb=11000)
                    offset = series[break_point]
                    series[break_point:break_end_point] = offset + (-slope * np.arange(break_end_point - break_point) + residuals[break_point:break_end_point])
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
                    series[break_point:break_end_point] = offset + (slope * slope_change_factor * np.arange(break_end_point - break_point) * scale_factor + residuals[break_point:break_end_point])
                    post_break_offset = series[break_end_point - 1]
                    series[break_end_point:] = post_break_offset + (slope * np.arange(1, n - break_end_point + 1) + residuals[break_end_point:])
                elif sign == -1:
                    slope = np.random.choice(negative_slopes)
                    series += slope * np.arange(n)
                    residuals, trend = hpfilter(series, lamb=11000)
                    offset = series[break_point]
                    series[break_point:break_end_point] = offset + (slope * slope_change_factor * np.arange(break_end_point - break_point) * scale_factor + residuals[break_point:break_end_point])
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
                    series[break_point:break_end_point] = offset + (-slope * slope_change_factor * np.arange(break_end_point - break_point) * scale_factor + residuals[break_point:break_end_point])
                    post_break_offset = series[break_end_point - 1]
                    series[break_end_point:] = post_break_offset + (slope * np.arange(1, n - break_end_point + 1) + residuals[break_end_point:])
                elif sign == -1:
                    slope = np.random.choice(negative_slopes)
                    series += slope * np.arange(n)
                    residuals, trend = hpfilter(series, lamb=11000)
                    offset = series[break_point]
                    series[break_point:break_end_point] = offset + (-slope * slope_change_factor * np.arange(break_end_point - break_point) * scale_factor + residuals[break_point:break_end_point])
                    post_break_offset = series[break_end_point - 1]
                    series[break_end_point:] = post_break_offset + (slope * np.arange(1, n - break_end_point + 1) + residuals[break_end_point:])
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
    
        df.loc[:, 'data'] = series
        return df

    
    def generate_gradual_mean_shift(self, df, sign=None, location=None, num_breaks=1, scale_factor=2):
        """
        Introduces gradual mean shifts (level changes) in the time series data.
        
        Unlike abrupt mean shifts, this method creates smooth transitions between different
        mean levels. The shift consists of three phases: a gradual increase/decrease phase,
        a stable shifted phase, and a gradual return phase. This creates a temporary but
        smooth deviation in the level of the time series.
        
        Parameters:
        -----------
        df : DataFrame
            DataFrame containing the time series data in a column named 'data'
        sign : int, optional
            Direction of the mean shift:
            1: Positive shift (increasing level)
            -1: Negative shift (decreasing level)
        location : str, optional
            Where to place the shift: 'beginning', 'middle', or 'end'
        num_breaks : int, default=1
            Number of gradual shifts to introduce
        scale_factor : int, default=2
            Multiplier to adjust the magnitude of the shifts
            
        Returns:
        --------
        DataFrame
            The modified DataFrame with the 'data' column updated and a 'mean_shift' column
            marking the locations of the shifts
        """
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
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
    
            start_transition = np.linspace(0, level_shift, transition_length)  # Gradual increase/decrease
            main_shift = np.ones(main_shift_length) * level_shift             # Stable shifted period
            end_transition = np.linspace(level_shift, 0, transition_length)   # Gradual return
    
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
        """
        Introduces gradual variance shifts in the time series data.
        
        This method creates smooth transitions in the volatility (variance) of the time series.
        Unlike abrupt variance shifts, this creates a three-phase pattern with gradual increase
        or decrease in volatility. The method can create higher volatility within specific segments
        or lower volatility, depending on the sign parameter.
        
        Parameters:
        -----------
        df : DataFrame
            DataFrame containing the time series data in a column named 'data'
        sign : int, optional
            Direction of the variance shift:
            1: Gradually increases variance within the break period
            -1: Gradually decreases variance within the break period
        location : str, optional
            Where to place the shift: 'beginning', 'middle', or 'end'
        num_breaks : int, default=1
            Number of variance shifts to introduce
        scale_factor : int, default=1
            Multiplier to adjust the magnitude of the variance shifts
            
        Returns:
        --------
        DataFrame
            The modified DataFrame with the 'data' column updated and a 'variance_shift' column
            marking the locations of the shifts
        """
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
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
        """
        Introduce gradual trend shifts into a time series dataset.
    
        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing the time series data in the 'data' column.
        sign : int, optional
            Direction of the trend shift. 
            - 1 for a positive trend shift.
            - -1 for a negative trend shift.
            Default is None.
        location : str, optional
            Location of the trend shift in the time series. 
            Options are:
              - 'beginning'
              - 'middle'
              - 'end'
            Default is None.
        num_breaks : int, optional
            Number of trend shifts to introduce. Default is 1.
        change_type : str, optional
            Type of trend shift to apply. Options are:
              - 'direction_change': Changes the direction of the trend.
              - 'magnitude_change': Changes the magnitude of the trend.
              - 'direction_and_magnitude_change': Changes both the direction and magnitude of the trend.
            Default is None.
        scale_factor : float, optional
            Scaling factor for the intensity of the trend shift. Larger values result in more pronounced shifts.
            Default is 1.
    
        Returns
        -------
        pandas.DataFrame
            Modified DataFrame with trend shifts applied to the 'data' column and a new 'trend_shift' column
            indicating the regions of trend shifts.
    
        Raises
        ------
        ValueError
            If an invalid `location`, `sign`, or `change_type` is provided.
            If the break length is too small for meaningful transitions.
        """
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
                    trend[break_point:break_end_point] = (
                        offset +
                        (transition_slope * np.arange(break_end_point - break_point)) * scale_factor +
                        residuals[break_point:break_end_point]
                    )
                    post_break_offset = series[break_end_point - 1]
                    transition_slope_2 = np.linspace(-slope, slope, n - break_end_point)
                    trend[break_end_point:] = (
                        post_break_offset +
                        np.cumsum(transition_slope_2) * scale_factor +
                        residuals[break_end_point:]
                    )
                elif sign == -1:
                    slope = np.random.choice(negative_slopes)
                    series += slope * np.arange(n)
                    residuals, trend = hpfilter(series, lamb=11000)
                    offset = series[break_point]
                    transition_slope = np.linspace(slope, -slope, break_length)
                    trend[break_point:break_end_point] = (
                        offset +
                        (transition_slope * np.arange(break_end_point - break_point)) * scale_factor +
                        residuals[break_point:break_end_point]
                    )
                    post_break_offset = series[break_end_point - 1]
                    transition_slope_2 = np.linspace(-slope, slope, n - break_end_point)
                    trend[break_end_point:] = (
                        post_break_offset +
                        np.cumsum(transition_slope_2) * scale_factor +
                        residuals[break_end_point:]
                    )
                else:
                    raise ValueError("Invalid sign. Expected '1' or '-1'.")
            elif change_type == 'magnitude_change':
                if sign == 1:
                    slope = np.random.choice(positive_slopes)
                    series += slope * np.arange(n)
                    residuals, trend = hpfilter(series, lamb=11000)
                    offset = series[break_point]
                    transition_slope = np.linspace(slope, slope * slope_change_factor, break_length)
                    trend[break_point:break_end_point] = (
                        offset +
                        (transition_slope * np.arange(break_end_point - break_point)) * scale_factor +
                        residuals[break_point:break_end_point]
                    )
                    post_break_offset = series[break_end_point - 1]
                    transition_slope_2 = np.linspace(slope * slope_change_factor, slope, n - break_end_point)
                    trend[break_end_point:] = (
                        post_break_offset +
                        np.cumsum(transition_slope_2) * scale_factor +
                        residuals[break_end_point:]
                    )
                elif sign == -1:
                    slope = np.random.choice(negative_slopes)
                    series += slope * np.arange(n)
                    residuals, trend = hpfilter(series, lamb=11000)
                    offset = series[break_point]
                    transition_slope = np.linspace(slope, slope * slope_change_factor, break_length)
                    trend[break_point:break_end_point] = (
                        offset +
                        (transition_slope * np.arange(break_end_point - break_point)) * scale_factor +
                        residuals[break_point:break_end_point]
                    )
                    post_break_offset = series[break_end_point - 1]
                    transition_slope_2 = np.linspace(slope * slope_change_factor, slope, n - break_end_point)
                    trend[break_end_point:] = (
                        post_break_offset +
                        np.cumsum(transition_slope_2) * scale_factor +
                        residuals[break_end_point:]
                    )
                else:
                    raise ValueError("Invalid sign. Expected '1' or '-1'.")
            elif change_type == 'direction_and_magnitude_change':
                if sign == 1:
                    slope = np.random.choice(positive_slopes)
                    series += slope * np.arange(n)
                    residuals, trend = hpfilter(series, lamb=11000)
                    offset = series[break_point]
                    transition_slope = np.linspace(slope, -slope * slope_change_factor, break_length)
                    trend[break_point:break_end_point] = (
                        offset +
                        (transition_slope * np.arange(break_end_point - break_point)) * scale_factor +
                        residuals[break_point:break_end_point]
                    )
                    post_break_offset = series[break_end_point - 1]
                    transition_slope_2 = np.linspace(-slope * slope_change_factor, slope, n - break_end_point)
                    trend[break_end_point:] = (
                        post_break_offset +
                        np.cumsum(transition_slope_2) * scale_factor +
                        residuals[break_end_point:]
                    )
                elif sign == -1:
                    slope = np.random.choice(negative_slopes)
                    series += slope * np.arange(n)
                    residuals, trend = hpfilter(series, lamb=11000)
                    offset = series[break_point]
                    transition_slope = np.linspace(slope, -slope * slope_change_factor, break_length)
                    trend[break_point:break_end_point] = (
                        offset +
                        (transition_slope * np.arange(break_end_point - break_point)) * scale_factor +
                        residuals[break_point:break_end_point]
                    )
                    post_break_offset = series[break_end_point - 1]
                    transition_slope_2 = np.linspace(-slope * slope_change_factor, slope, n - break_end_point)
                    trend[break_end_point:] = (
                        post_break_offset +
                        np.cumsum(transition_slope_2) * scale_factor +
                        residuals[break_end_point:]
                    )
                else:
                    raise ValueError("Invalid sign. Expected '1' or '-1'.")
            else:
                raise ValueError("Invalid change_type. Expected 'direction_change', 'magnitude_change', or 'direction_and_magnitude_change'.")
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
                        series[break_point:break_end_point] = (
                            offset +
                            (transition_slope * np.arange(break_end_point - break_point)) * scale_factor
                        )
                        post_break_offset = series[break_end_point - 1]
                        transition_slope_2 = np.linspace(-slope, slope, n - break_end_point)
                        series[break_end_point:] = post_break_offset + np.cumsum(transition_slope_2) * scale_factor
                    elif sign == -1:
                        slope = np.random.choice(negative_slopes)
                        series += slope * np.arange(n)
                        offset = series[break_point]
                        transition_slope = np.linspace(slope, -slope, break_length)
                        series[break_point:break_end_point] = (
                            offset +
                            (transition_slope * np.arange(break_end_point - break_point)) * scale_factor
                        )
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
                        series[break_point:break_end_point] = (
                            offset +
                            (transition_slope * np.arange(break_end_point - break_point)) * scale_factor
                        )
                        post_break_offset = series[break_end_point - 1]
                        transition_slope_2 = np.linspace(slope * slope_change_factor, slope, n - break_end_point)
                        series[break_end_point:] = post_break_offset + np.cumsum(transition_slope_2) * scale_factor
                    elif sign == -1:
                        slope = np.random.choice(negative_slopes)
                        series += slope * np.arange(n)
                        offset = series[break_point]
                        transition_slope = np.linspace(slope, slope * slope_change_factor, break_length)
                        series[break_point:break_end_point] = (
                            offset +
                            (transition_slope * np.arange(break_end_point - break_point)) * scale_factor
                        )
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
                        series[break_point:break_end_point] = (
                            offset +
                            (transition_slope * np.arange(break_end_point - break_point)) * scale_factor
                        )
                        post_break_offset = series[break_end_point - 1]
                        transition_slope_2 = np.linspace(-slope * slope_change_factor, slope, n - break_end_point)
                        series[break_end_point:] = post_break_offset + np.cumsum(transition_slope_2) * scale_factor
                    elif sign == -1:
                        slope = np.random.choice(negative_slopes)
                        series += slope * np.arange(n)
                        offset = series[break_point]
                        transition_slope = np.linspace(slope, -slope * slope_change_factor, break_length)
                        series[break_point:break_end_point] = (
                            offset +
                            (transition_slope * np.arange(break_end_point - break_point)) * scale_factor
                        )
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

        df.loc[:, 'data'] = series
        return df

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