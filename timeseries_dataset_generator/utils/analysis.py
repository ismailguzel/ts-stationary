"""
Analysis utilities for time series datasets.

This module provides helper functions for analyzing time series data,
including statistical tests, feature extraction, and anomaly detection evaluation.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Union
import warnings


def compute_basic_statistics(
    df: pd.DataFrame,
    series_id: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute basic statistical measures for a time series.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data
    series_id : int, optional
        ID of the series to analyze. If None, uses first series
        
    Returns
    -------
    dict
        Dictionary containing statistical measures
    """
    if series_id is None:
        series_id = df['series_id'].iloc[0]
    
    series_data = df[df['series_id'] == series_id]['data']
    
    stats = {
        'mean': float(np.mean(series_data)),
        'median': float(np.median(series_data)),
        'std': float(np.std(series_data)),
        'var': float(np.var(series_data)),
        'min': float(np.min(series_data)),
        'max': float(np.max(series_data)),
        'range': float(np.max(series_data) - np.min(series_data)),
        'q25': float(np.percentile(series_data, 25)),
        'q75': float(np.percentile(series_data, 75)),
        'iqr': float(np.percentile(series_data, 75) - np.percentile(series_data, 25)),
        'skewness': float(pd.Series(series_data).skew()),
        'kurtosis': float(pd.Series(series_data).kurtosis()),
        'cv': float(np.std(series_data) / np.mean(series_data)) if np.mean(series_data) != 0 else np.nan
    }
    
    return stats


def test_stationarity(
    df: pd.DataFrame,
    series_id: Optional[int] = None,
    alpha: float = 0.05
) -> Dict[str, Union[float, bool, str]]:
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data
    series_id : int, optional
        ID of the series to analyze. If None, uses first series
    alpha : float, default=0.05
        Significance level for the test
        
    Returns
    -------
    dict
        Dictionary containing test results
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return {'error': 'statsmodels required for stationarity test'}
    
    if series_id is None:
        series_id = df['series_id'].iloc[0]
    
    series_data = df[df['series_id'] == series_id]['data']
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        result = adfuller(series_data, autolag='AIC')
    
    test_result = {
        'test_statistic': float(result[0]),
        'p_value': float(result[1]),
        'n_lags': int(result[2]),
        'n_obs': int(result[3]),
        'critical_values': {k: float(v) for k, v in result[4].items()},
        'is_stationary': result[1] < alpha,
        'conclusion': 'Stationary' if result[1] < alpha else 'Non-stationary',
        'alpha': alpha
    }
    
    return test_result


def test_normality(
    df: pd.DataFrame,
    series_id: Optional[int] = None,
    alpha: float = 0.05
) -> Dict[str, Union[float, bool, str]]:
    """
    Perform Shapiro-Wilk test for normality.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data
    series_id : int, optional
        ID of the series to analyze. If None, uses first series
    alpha : float, default=0.05
        Significance level for the test
        
    Returns
    -------
    dict
        Dictionary containing test results
    """
    try:
        from scipy.stats import shapiro
    except ImportError:
        return {'error': 'scipy required for normality test'}
    
    if series_id is None:
        series_id = df['series_id'].iloc[0]
    
    series_data = df[df['series_id'] == series_id]['data']
    
    # Sample if too large (Shapiro-Wilk has limits)
    if len(series_data) > 5000:
        series_data = np.random.choice(series_data, 5000, replace=False)
    
    statistic, p_value = shapiro(series_data)
    
    test_result = {
        'test_statistic': float(statistic),
        'p_value': float(p_value),
        'is_normal': p_value > alpha,
        'conclusion': 'Normal' if p_value > alpha else 'Non-normal',
        'alpha': alpha
    }
    
    return test_result


def detect_seasonality(
    df: pd.DataFrame,
    series_id: Optional[int] = None,
    max_period: Optional[int] = None
) -> Dict[str, Union[float, int, bool]]:
    """
    Detect seasonality using autocorrelation.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data
    series_id : int, optional
        ID of the series to analyze. If None, uses first series
    max_period : int, optional
        Maximum period to check. If None, uses len(series) // 3
        
    Returns
    -------
    dict
        Dictionary containing seasonality detection results
    """
    try:
        from statsmodels.tsa.stattools import acf
    except ImportError:
        return {'error': 'statsmodels required for seasonality detection'}
    
    if series_id is None:
        series_id = df['series_id'].iloc[0]
    
    series_data = df[df['series_id'] == series_id]['data']
    
    if max_period is None:
        max_period = min(len(series_data) // 3, 200)
    
    # Compute ACF
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        acf_values = acf(series_data, nlags=max_period, fft=True)
    
    # Find peaks in ACF (excluding lag 0)
    acf_values = acf_values[1:]  # Remove lag 0
    
    # Simple peak detection
    peaks = []
    for i in range(1, len(acf_values) - 1):
        if acf_values[i] > acf_values[i-1] and acf_values[i] > acf_values[i+1]:
            if acf_values[i] > 0.3:  # Threshold for significant peak
                peaks.append((i + 1, acf_values[i]))  # +1 because we removed lag 0
    
    # Sort by ACF value
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    has_seasonality = len(peaks) > 0
    dominant_period = peaks[0][0] if peaks else None
    dominant_acf = peaks[0][1] if peaks else None
    
    result = {
        'has_seasonality': has_seasonality,
        'dominant_period': dominant_period,
        'dominant_acf': float(dominant_acf) if dominant_acf is not None else None,
        'all_periods': [p[0] for p in peaks],
        'all_acf_values': [float(p[1]) for p in peaks],
        'max_acf': float(np.max(acf_values)),
        'mean_acf': float(np.mean(acf_values))
    }
    
    return result


def compute_autocorrelation_stats(
    df: pd.DataFrame,
    series_id: Optional[int] = None,
    nlags: int = 40
) -> Dict[str, Union[np.ndarray, float, int]]:
    """
    Compute autocorrelation statistics.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data
    series_id : int, optional
        ID of the series to analyze. If None, uses first series
    nlags : int, default=40
        Number of lags to compute
        
    Returns
    -------
    dict
        Dictionary containing ACF statistics
    """
    try:
        from statsmodels.tsa.stattools import acf, pacf
    except ImportError:
        return {'error': 'statsmodels required for autocorrelation analysis'}
    
    if series_id is None:
        series_id = df['series_id'].iloc[0]
    
    series_data = df[df['series_id'] == series_id]['data']
    
    nlags = min(nlags, len(series_data) // 2 - 1)
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        acf_values = acf(series_data, nlags=nlags, fft=True)
        pacf_values = pacf(series_data, nlags=nlags)
    
    result = {
        'acf': acf_values,
        'pacf': pacf_values,
        'acf_lag1': float(acf_values[1]),
        'acf_sum': float(np.sum(np.abs(acf_values[1:]))),
        'acf_mean': float(np.mean(np.abs(acf_values[1:]))),
        'n_significant_lags_acf': int(np.sum(np.abs(acf_values[1:]) > 2/np.sqrt(len(series_data)))),
        'n_significant_lags_pacf': int(np.sum(np.abs(pacf_values[1:]) > 2/np.sqrt(len(series_data))))
    }
    
    return result


def detect_trend(
    df: pd.DataFrame,
    series_id: Optional[int] = None,
    alpha: float = 0.05
) -> Dict[str, Union[float, bool, str]]:
    """
    Detect trend using Mann-Kendall test.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data
    series_id : int, optional
        ID of the series to analyze. If None, uses first series
    alpha : float, default=0.05
        Significance level for the test
        
    Returns
    -------
    dict
        Dictionary containing trend detection results
    """
    if series_id is None:
        series_id = df['series_id'].iloc[0]
    
    series_data = df[df['series_id'] == series_id]['data'].values
    n = len(series_data)
    
    # Mann-Kendall test
    s = 0
    for i in range(n-1):
        for j in range(i+1, n):
            s += np.sign(series_data[j] - series_data[i])
    
    # Variance
    var_s = n * (n - 1) * (2 * n + 5) / 18
    
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    
    # P-value (two-tailed test)
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(z)))
    
    # Linear regression slope for trend magnitude
    x = np.arange(n)
    slope = np.polyfit(x, series_data, 1)[0]
    
    has_trend = p_value < alpha
    
    if has_trend:
        trend_direction = 'increasing' if z > 0 else 'decreasing'
    else:
        trend_direction = 'no trend'
    
    result = {
        'has_trend': has_trend,
        'trend_direction': trend_direction,
        'z_score': float(z),
        'p_value': float(p_value),
        'slope': float(slope),
        'alpha': alpha
    }
    
    return result


def detect_changepoints(
    df: pd.DataFrame,
    series_id: Optional[int] = None,
    window_size: int = 20,
    threshold: float = 2.0
) -> Dict[str, Union[List[int], int]]:
    """
    Detect changepoints using moving window approach.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data
    series_id : int, optional
        ID of the series to analyze. If None, uses first series
    window_size : int, default=20
        Size of the moving window
    threshold : float, default=2.0
        Threshold for changepoint detection (in standard deviations)
        
    Returns
    -------
    dict
        Dictionary containing changepoint detection results
    """
    if series_id is None:
        series_id = df['series_id'].iloc[0]
    
    series_data = df[df['series_id'] == series_id]['data'].values
    n = len(series_data)
    
    # Compute rolling statistics
    changepoints_mean = []
    changepoints_var = []
    
    for i in range(window_size, n - window_size):
        before = series_data[i-window_size:i]
        after = series_data[i:i+window_size]
        
        # Test for mean shift
        mean_diff = abs(np.mean(after) - np.mean(before))
        pooled_std = np.sqrt((np.var(before) + np.var(after)) / 2)
        
        if pooled_std > 0:
            mean_test = mean_diff / pooled_std
            if mean_test > threshold:
                changepoints_mean.append(i)
        
        # Test for variance shift
        var_ratio = np.var(after) / (np.var(before) + 1e-10)
        if var_ratio > threshold or var_ratio < 1/threshold:
            changepoints_var.append(i)
    
    result = {
        'mean_changepoints': changepoints_mean,
        'variance_changepoints': changepoints_var,
        'n_mean_changepoints': len(changepoints_mean),
        'n_variance_changepoints': len(changepoints_var),
        'total_changepoints': len(set(changepoints_mean + changepoints_var))
    }
    
    return result


def compute_entropy(
    df: pd.DataFrame,
    series_id: Optional[int] = None,
    bins: int = 50
) -> Dict[str, float]:
    """
    Compute various entropy measures.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data
    series_id : int, optional
        ID of the series to analyze. If None, uses first series
    bins : int, default=50
        Number of bins for histogram
        
    Returns
    -------
    dict
        Dictionary containing entropy measures
    """
    if series_id is None:
        series_id = df['series_id'].iloc[0]
    
    series_data = df[df['series_id'] == series_id]['data'].values
    
    # Shannon entropy
    hist, _ = np.histogram(series_data, bins=bins)
    hist = hist / np.sum(hist)  # Normalize
    hist = hist[hist > 0]  # Remove zeros
    shannon_entropy = -np.sum(hist * np.log2(hist))
    
    # Sample entropy (simplified version)
    def sample_entropy_calc(data, m=2, r=None):
        if r is None:
            r = 0.2 * np.std(data)
        
        n = len(data)
        
        def _maxdist(xi, xj, m):
            return max([abs(xi[k] - xj[k]) for k in range(m)])
        
        def _phi(m):
            patterns = np.array([data[i:i+m] for i in range(n - m)])
            count = 0
            for i in range(len(patterns)):
                for j in range(len(patterns)):
                    if i != j:
                        if _maxdist(patterns[i], patterns[j], m) <= r:
                            count += 1
            return count / (n - m)
        
        return -np.log(_phi(m+1) / _phi(m)) if _phi(m) > 0 else 0
    
    try:
        samp_entropy = sample_entropy_calc(series_data)
    except:
        samp_entropy = np.nan
    
    result = {
        'shannon_entropy': float(shannon_entropy),
        'sample_entropy': float(samp_entropy)
    }
    
    return result


def analyze_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary analysis for all series in a dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data
        
    Returns
    -------
    pd.DataFrame
        Summary DataFrame with statistics for each series
    """
    series_ids = df['series_id'].unique()
    
    summaries = []
    for series_id in series_ids:
        series_data = df[df['series_id'] == series_id]
        
        summary = {
            'series_id': series_id,
            'length': len(series_data),
        }
        
        # Add label if available
        if 'label' in series_data.columns:
            summary['label'] = series_data['label'].iloc[0]
        
        # Basic statistics
        stats = compute_basic_statistics(df, series_id)
        summary.update(stats)
        
        # Stationarity test
        stationarity = test_stationarity(df, series_id)
        if 'error' not in stationarity:
            summary['is_stationary'] = stationarity['is_stationary']
            summary['adf_pvalue'] = stationarity['p_value']
        
        # Trend detection
        trend = detect_trend(df, series_id)
        summary['has_trend'] = trend['has_trend']
        summary['trend_direction'] = trend['trend_direction']
        summary['trend_slope'] = trend['slope']
        
        # Seasonality detection
        seasonality = detect_seasonality(df, series_id)
        if 'error' not in seasonality:
            summary['has_seasonality'] = seasonality['has_seasonality']
            summary['dominant_period'] = seasonality['dominant_period']
        
        summaries.append(summary)
    
    return pd.DataFrame(summaries)


def compare_series(
    df: pd.DataFrame,
    series_id1: int,
    series_id2: int
) -> Dict[str, float]:
    """
    Compare two time series.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data
    series_id1 : int
        ID of the first series
    series_id2 : int
        ID of the second series
        
    Returns
    -------
    dict
        Dictionary containing comparison metrics
    """
    series1 = df[df['series_id'] == series_id1]['data'].values
    series2 = df[df['series_id'] == series_id2]['data'].values
    
    # Ensure same length
    min_len = min(len(series1), len(series2))
    series1 = series1[:min_len]
    series2 = series2[:min_len]
    
    # Correlation
    correlation = np.corrcoef(series1, series2)[0, 1]
    
    # Euclidean distance
    euclidean_dist = np.sqrt(np.sum((series1 - series2) ** 2))
    
    # Mean absolute error
    mae = np.mean(np.abs(series1 - series2))
    
    # Root mean squared error
    rmse = np.sqrt(np.mean((series1 - series2) ** 2))
    
    # Dynamic time warping (simplified)
    def dtw_distance(s1, s2):
        n, m = len(s1), len(s2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(s1[i-1] - s2[j-1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],
                                             dtw_matrix[i, j-1],
                                             dtw_matrix[i-1, j-1])
        
        return dtw_matrix[n, m]
    
    # Subsample for DTW if series is too long
    if len(series1) > 500:
        dtw_dist = dtw_distance(series1[::5], series2[::5])
    else:
        dtw_dist = dtw_distance(series1, series2)
    
    result = {
        'correlation': float(correlation),
        'euclidean_distance': float(euclidean_dist),
        'mae': float(mae),
        'rmse': float(rmse),
        'dtw_distance': float(dtw_dist)
    }
    
    return result

