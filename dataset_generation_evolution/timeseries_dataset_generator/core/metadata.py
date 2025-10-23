"""
Metadata management for time series datasets.

This module contains functions for creating and managing metadata records
that describe the characteristics of generated time series.
"""

import numpy as np
import pandas as pd


def create_metadata_record(
    series_id,
    length,
    label,
    is_stationary=1,
    base_series=0,
    order=0,
    base_coefs=0,
    linear_trend_up=0,
    linear_trend_down=0,
    quadratic_trend=0,
    cubic_trend=0,
    exponential_trend=0,
    damped_trend=0,
    stochastic_trend=0,
    difference=0,
    seasonality=0,
    multiple_seasonality=0,
    seasonality_from_base=0,
    seasonality_frequency=0,
    seasonal_difference=0,
    volatility=0,
    mean_shift_increase=0,
    mean_shift_decrease=0,
    multi_mean_shift=0,
    variance_shift_increase=0,
    variance_shift_decrease=0,
    multi_variance_shift=0,
    trend_shift_slope=0,
    trend_shift_intercept=0,
    multi_trend_shift=0,
    point_anomaly=0,
    collective_anomaly=0,
    contextual_anomaly=0,
    multi_point_anomaly=0,
    multi_collective_anomaly=0,
    multi_contextual_anomaly=0,
    location_point=0,
    location_collective=0,
    location_mean_shift=0,
    location_variance_shift=0,
    location_trend_shift=0,
    location_contextual=0,
    location_point_pts=0,
    location_collective_pts=0,
    location_mean_pts=0,
    location_contextual_pts=0,
    location_variance_pts=0,
    location_trend_pts=0,
):
    """
    Create a metadata record for a time series.

    Parameters
    ----------
    series_id : int
        Unique identifier for the series
    length : int
        Length of the time series
    label : str
        Label/category of the time series
    is_stationary : int, default=1
        Whether the series is stationary (1) or not (0)
    base_series : str or int, default=0
        Base stochastic process type (e.g., 'ar', 'ma', 'arma', 'white_noise')
    order : int or str, default=0
        Order of the base process
    base_coefs : int or str, default=0
        Coefficients of the base process
    linear_trend_up : int, default=0
        Presence of upward linear trend
    linear_trend_down : int, default=0
        Presence of downward linear trend
    quadratic_trend : int, default=0
        Presence of quadratic trend
    cubic_trend : int, default=0
        Presence of cubic trend
    exponential_trend : int, default=0
        Presence of exponential trend
    damped_trend : int, default=0
        Presence of damped trend
    stochastic_trend : int, default=0
        Presence of stochastic trend
    difference : int or str, default=0
        Order of differencing
    seasonality : int, default=0
        Presence of seasonality
    multiple_seasonality : int, default=0
        Presence of multiple seasonal components
    seasonality_from_base : int, default=0
        Whether seasonality comes from base series (SARMA/SARIMA)
    seasonality_frequency : int or str, default=0
        Seasonal period(s)
    seasonal_difference : int or str, default=0
        Order of seasonal differencing
    volatility : int, default=0
        Presence of volatility clustering
    mean_shift_increase : int, default=0
        Presence of mean increase shift
    mean_shift_decrease : int, default=0
        Presence of mean decrease shift
    multi_mean_shift : int, default=0
        Presence of multiple mean shifts
    variance_shift_increase : int, default=0
        Presence of variance increase shift
    variance_shift_decrease : int, default=0
        Presence of variance decrease shift
    multi_variance_shift : int, default=0
        Presence of multiple variance shifts
    trend_shift_slope : int, default=0
        Presence of trend slope shift
    trend_shift_intercept : int, default=0
        Presence of trend intercept shift
    multi_trend_shift : int, default=0
        Presence of multiple trend shifts
    point_anomaly : int, default=0
        Presence of single point anomaly
    collective_anomaly : int, default=0
        Presence of single collective anomaly
    contextual_anomaly : int, default=0
        Presence of single contextual anomaly
    multi_point_anomaly : int, default=0
        Presence of multiple point anomalies
    multi_collective_anomaly : int, default=0
        Presence of multiple collective anomalies
    multi_contextual_anomaly : int, default=0
        Presence of multiple contextual anomalies
    location_point : str or int, default=0
        Location of point anomaly
    location_collective : str or int, default=0
        Location of collective anomaly
    location_mean_shift : str or int, default=0
        Location of mean shift
    location_variance_shift : str or int, default=0
        Location of variance shift
    location_trend_shift : str or int, default=0
        Location of trend shift
    location_contextual : str or int, default=0
        Location of contextual anomaly
    location_point_pts : str or int, default=0
        Specific points/magnitudes for point anomaly
    location_collective_pts : str or int, default=0
        Specific points/magnitudes for collective anomaly
    location_mean_pts : str or int, default=0
        Specific points/magnitudes for mean shift
    location_contextual_pts : str or int, default=0
        Specific points/magnitudes for contextual anomaly
    location_variance_pts : str or int, default=0
        Specific points/magnitudes for variance shift
    location_trend_pts : str or int, default=0
        Specific points/types for trend shift

    Returns
    -------
    dict
        Dictionary containing all metadata fields
    """
    record = {
        # General
        "series_id": series_id,
        "length": length,
        "label": label,
        "is_stationary": is_stationary,

        # Base stochastic process
        "base_series": base_series,
        "order": order,
        "base_coefs": base_coefs,

        # Deterministic trends
        "linear_trend_up": linear_trend_up,
        "linear_trend_down": linear_trend_down,
        "quadratic_trend": quadratic_trend,
        "cubic_trend": cubic_trend,
        "exponential_trend": exponential_trend,
        "damped_trend": damped_trend,

        # Stochastic trend
        "stochastic_trend": stochastic_trend,
        "difference": difference,

        # Seasonality
        "seasonality": seasonality,
        "multiple_seasonality": multiple_seasonality,
        "seasonality_from_base": seasonality_from_base,
        "seasonality_frequency": seasonality_frequency,
        "seasonal_difference": seasonal_difference,

        # Volatility
        "volatility": volatility,

        # Mean shift
        "mean_shift_increase": mean_shift_increase,
        "mean_shift_decrease": mean_shift_decrease,
        "multi_mean_shift": multi_mean_shift,

        # Variance shift
        "variance_shift_increase": variance_shift_increase,
        "variance_shift_decrease": variance_shift_decrease,
        "multi_variance_shift": multi_variance_shift,

        # Trend shift
        "trend_shift_slope": trend_shift_slope,
        "trend_shift_intercept": trend_shift_intercept,
        "multi_trend_shift": multi_trend_shift,

        # Anomaly types
        "point_anomaly": point_anomaly,
        "collective_anomaly": collective_anomaly,
        "contextual_anomaly": contextual_anomaly,
        "multi_point_anomaly": multi_point_anomaly,
        "multi_collective_anomaly": multi_collective_anomaly,
        "multi_contextual_anomaly": multi_contextual_anomaly,

        # Anomaly locations
        "location_point": location_point,
        "location_collective": location_collective,
        "location_mean_shift": location_mean_shift,
        "location_variance_shift": location_variance_shift,
        "location_trend_shift": location_trend_shift,
        "location_contextual": location_contextual,

        # Anomaly location points/magnitude info
        "location_point_pts": location_point_pts,
        "location_collective_pts": location_collective_pts,
        "location_mean_pts": location_mean_pts,
        "location_contextual_pts": location_contextual_pts,
        "location_variance_pts": location_variance_pts,
        "location_trend_pts": location_trend_pts
    }
    return record


def make_json_serializable(obj):
    """
    Convert numpy types to Python native types for JSON serialization.

    Parameters
    ----------
    obj : any
        Object to convert

    Returns
    -------
    any
        JSON-serializable version of the object
    """
    if isinstance(obj, (np.integer, np.int_, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)
    return str(obj)


def get_metadata_columns_defaults():
    """
    Get metadata column names and default values.

    Returns
    -------
    tuple
        (column_names, default_record)
    """
    dummy = create_metadata_record(series_id=0, length=0, label="", is_stationary=1)
    return list(dummy.keys()), dummy


def attach_metadata_columns_to_df(df, metadata_record):
    """
    Attach metadata columns to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time series data
    metadata_record : dict
        Metadata record to attach

    Returns
    -------
    pd.DataFrame
        DataFrame with metadata columns attached
    """
    metadata_cols, default_record = get_metadata_columns_defaults()

    for col in metadata_cols:
        val = metadata_record.get(col, default_record[col])

        if isinstance(val, (int, float, str)):
            df[col] = val
        else:
            df[col] = str(val)

    df['label'] = metadata_record['label']
    
    core_cols = ['series_id', 'time', 'data', 'label']
    meta_cols = [col for col in metadata_cols if col not in core_cols and col in df.columns]
    final_cols_order = ['series_id', 'time', 'data'] + meta_cols + ['label']
    
    final_cols_in_df = [col for col in final_cols_order if col in df.columns]
    
    df = df[final_cols_in_df]
    return df

