"""
Generators for time series datasets with anomalies.

This module contains functions to generate datasets with various anomaly types:
- Point Anomalies (single and multiple)
- Collective Anomalies (single and multiple)
- Contextual Anomalies (single and multiple)
"""

import os
import numpy as np
from ..core.metadata import create_metadata_record, attach_metadata_columns_to_df
from ..utils.helpers import save_and_cleanup, get_length_label


def _get_base_series(ts, kind):
    """
    Generate base series based on kind.

    Parameters
    ----------
    ts : TimeSeriesGenerator
        Time series generator instance
    kind : str
        Type of base series ('ar', 'ma', 'arma', 'white_noise')

    Returns
    -------
    tuple
        (df, base_coefs, base_order)
    """
    if kind == 'ar':
        df, info = ts.generate_stationary_base_series('ar')
        base_coefs = f"({info['ar_coefs']})"
        base_order = f"({info['ar_order']})"
    elif kind == 'ma':
        df, info = ts.generate_stationary_base_series('ma')
        base_coefs = f"({info['ma_coefs']})"
        base_order = f"({info['ma_order']})"
    elif kind == 'arma':
        df, info = ts.generate_stationary_base_series('arma')
        base_coefs = f"({info['ar_coefs']},{info['ma_coefs']})"
        base_order = f"({info['ar_order']},{info['ma_order']})"
    elif kind == 'white_noise':
        df, info = ts.generate_stationary_base_series('white_noise')
        base_coefs = 0
        base_order = 0
    else:
        raise ValueError(f"Unknown kind: {kind}")
    
    return df, base_coefs, base_order


def generate_point_anomaly_dataset(
    ts_generator_class,
    folder,
    kind='ar',
    count=5,
    length_range=(300, 500),
    anomaly_type='single',
    location="middle"
):
    """
    Generate point anomaly dataset.

    Parameters
    ----------
    ts_generator_class : class
        TimeSeriesGenerator class
    folder : str
        Output folder path
    kind : str, default='ar'
        Base series type ('ar', 'ma', 'arma', 'white_noise')
    count : int, default=5
        Number of series to generate
    length_range : tuple, default=(300, 500)
        (min, max) length for generated series
    anomaly_type : str, default='single'
        'single' or 'multiple'
    location : str, default='middle'
        Location of anomaly ('beginning', 'middle', 'end')

    Returns
    -------
    None
    """
    os.makedirs(folder, exist_ok=True)
    all_dfs = []
    label = ""
    l = get_length_label(length_range)

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = ts_generator_class(length=length)

        df, base_coefs, base_order = _get_base_series(ts, kind)

        if anomaly_type == 'single':
            loc = location if location else np.random.choice(['beginning', 'middle', 'end'])
            df, info_anom = ts.generate_point_anomaly(df, location=loc)
            label = f"{kind}_single_point_anomaly_{loc}_{l}"
            point_anomaly = 1
            multi_point_anomaly = 0
            location_meta = f"{info_anom['location']}"
            anomaly_indices = f"{info_anom['anomaly_indices']}"

        elif anomaly_type == 'multiple':
            df, info_anom = ts.generate_point_anomalies(df)
            label = f"{kind}_multiple_point_anomalies_{l}"
            multi_point_anomaly = 1
            point_anomaly = 0
            location_meta = "multiple"
            anomaly_indices = f"{info_anom['anomaly_indices']}"
        else:
            raise ValueError("Invalid anomaly_type. Must be 'single' or 'multiple'.")

        series_id = i + 1

        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag,
            base_series=kind,
            base_coefs=base_coefs,
            order=base_order,
            point_anomaly=point_anomaly,
            multi_point_anomaly=multi_point_anomaly,
            location_point=location_meta,
            location_point_pts=anomaly_indices
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    save_and_cleanup(all_dfs, folder, count, label)


def generate_collective_anomaly_dataset(
    ts_generator_class,
    folder,
    kind='ar',
    count=5,
    length_range=(300, 500),
    anomaly_type='single',          
    location="middle",              
    num_anomalies=2,
    scale_factor=1
):
    """
    Generate collective anomaly dataset.

    Parameters
    ----------
    ts_generator_class : class
        TimeSeriesGenerator class
    folder : str
        Output folder path
    kind : str, default='ar'
        Base series type ('ar', 'ma', 'arma', 'white_noise')
    count : int, default=5
        Number of series to generate
    length_range : tuple, default=(300, 500)
        (min, max) length for generated series
    anomaly_type : str, default='single'
        'single' or 'multiple'
    location : str, default='middle'
        Location of anomaly ('beginning', 'middle', 'end')
    num_anomalies : int, default=2
        Number of anomalies (for multiple type)
    scale_factor : float, default=1
        Scale factor for anomaly magnitude

    Returns
    -------
    None
    """
    os.makedirs(folder, exist_ok=True)
    all_dfs = []
    label = ""
    l = get_length_label(length_range)

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = ts_generator_class(length=length)

        df, base_coefs, base_order = _get_base_series(ts, kind)

        if anomaly_type == 'single':
            loc = location if location else np.random.choice(['beginning', 'middle', 'end'])
            df, info_anom = ts.generate_collective_anomalies(
                df, num_anomalies=1, location=loc, scale_factor=scale_factor
            )
            label = f"{kind}_single_collective_anomaly_{loc}_{l}"
            collective_anomaly = 1
            multi_collective_anomaly = 0
            location_meta = f"{info_anom['location']}"
            anomaly_indices = f"{info_anom['starts']}, {info_anom['ends']}"

        elif anomaly_type == 'multiple':
            k = max(2, int(num_anomalies))
            df, info_anom = ts.generate_collective_anomalies(
                df, num_anomalies=k, location=location, scale_factor=scale_factor
            )
            label = f"{kind}_multiple_collective_anomalies_{l}"
            collective_anomaly = 0
            multi_collective_anomaly = 1
            location_meta = "multiple"
            anomaly_indices = f"{info_anom['starts']}, {info_anom['ends']}"
        else:
            raise ValueError("Invalid anomaly_type. Must be 'single' or 'multiple'.")

        series_id = i + 1

        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag,
            base_series=kind,
            base_coefs=base_coefs,
            order=base_order,
            collective_anomaly=collective_anomaly,
            multi_collective_anomaly=multi_collective_anomaly,
            location_collective=location_meta,
            location_collective_pts=anomaly_indices
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    save_and_cleanup(all_dfs, folder, count, label)


def generate_contextual_anomaly_dataset(
    ts_generator_class,
    folder,
    count=5,
    anomaly_type='single',
    length_range=(300, 500),        
    location="middle",              
    num_anomalies=2,
    scale_factor=1
):
    """
    Generate contextual anomaly dataset.

    Parameters
    ----------
    ts_generator_class : class
        TimeSeriesGenerator class
    folder : str
        Output folder path
    count : int, default=5
        Number of series to generate
    anomaly_type : str, default='single'
        'single' or 'multiple'
    length_range : tuple, default=(300, 500)
        (min, max) length for generated series
    location : str, default='middle'
        Location of anomaly ('beginning', 'middle', 'end')
    num_anomalies : int, default=2
        Number of anomalies (for multiple type)
    scale_factor : float, default=1
        Scale factor for anomaly magnitude

    Returns
    -------
    None
    """
    os.makedirs(folder, exist_ok=True)
    all_dfs = []
    label = ""
    l = get_length_label(length_range)

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = ts_generator_class(length=length)
        df, info1 = ts.generate_seasonality_from_base_series('single')
        if df is None: 
            print(f"Skipping one contextual anomaly generation for {folder} due to base seasonality error.")
            continue

        if anomaly_type == 'single':
            loc = location if location else np.random.choice(['beginning', 'middle', 'end'])
            df, info2 = ts.generate_contextual_anomalies(
                df, num_anomalies=1, location=loc, scale_factor=scale_factor, seasonal_period=info1['period']
            )
            if info2 is None:
                print(f"Skipping one contextual anomaly generation for {folder} due to anomaly error.")
                continue
            label = f"single_contextual_anomaly_{loc}_{l}"
            contextual_anomaly = 1
            multi_contextual_anomaly = 0
            location_meta = f"{info2['location']}"
            anomaly_indices = f"{info2['starts']}, {info2['ends']}"

        elif anomaly_type == 'multiple':
            k = max(2, int(num_anomalies))
            df, info2 = ts.generate_contextual_anomalies(
                df, num_anomalies=k, location=location, scale_factor=scale_factor, seasonal_period=info1['period']
            )
            if info2 is None:
                print(f"Skipping one contextual anomaly generation for {folder} due to anomaly error.")
                continue
            label = f"multiple_contextual_anomalies_{l}"
            contextual_anomaly = 0
            multi_contextual_anomaly = 1
            location_meta = "multiple"
            anomaly_indices = f"{info2['starts']}, {info2['ends']}"
        else:
            raise ValueError("Invalid anomaly_type. Must be 'single' or 'multiple'.")

        series_id = i + 1

        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag,
            contextual_anomaly=contextual_anomaly,
            multi_contextual_anomaly=multi_contextual_anomaly,
            location_contextual=location_meta,
            location_contextual_pts=anomaly_indices,
            seasonality=1,
            seasonality_frequency=info1['period']
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    save_and_cleanup(all_dfs, folder, len(all_dfs), label)

