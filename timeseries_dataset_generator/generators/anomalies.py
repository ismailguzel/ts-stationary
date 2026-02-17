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
from ..utils.helpers import save_and_cleanup, get_length_label, add_indices_column


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
    location="middle",
    num_anomalies=1,
    start_id=1,
    is_loc = None,
):
    """
    Generate point anomaly dataset.
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
            subcat = "point_single"
            location_meta = f"{info_anom['location']}"
            anomaly_indices = info_anom['anomaly_indices'].tolist() if hasattr(info_anom['anomaly_indices'], 'tolist') else info_anom['anomaly_indices']
            anomaly_count = 1
        elif anomaly_type == 'multiple':
            df, info_anom = ts.generate_point_anomalies(df)
            label = f"{kind}_multiple_point_anomalies_{l}"
            subcat = "point_multiple"
            location_meta = "multiple"
            anomaly_indices = info_anom['anomaly_indices'].tolist() if hasattr(info_anom['anomaly_indices'], 'tolist') else info_anom['anomaly_indices']
            anomaly_count = len(anomaly_indices)
        else:
            raise ValueError("Invalid anomaly_type. Must be 'single' or 'multiple'.")

        series_id = start_id + i

        is_stat_flag = int(df['stationary'].iloc[0])
        is_seasonal_flag = int(df['seasonal'].iloc[0])
        df = df.drop(columns=['seasonal'])
        df = df.drop(columns=['stationary'])

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag,
            is_seasonal=is_seasonal_flag,
            primary_category="anomaly",
            primary_label=1,
            sub_category=subcat,
            sub_label=0,
            base_series=kind,
            base_coefs=base_coefs,
            order=base_order,
            anomaly_type=subcat,
            anomaly_count=anomaly_count,
            anomaly_indices=anomaly_indices,
            location_point=location_meta
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        if is_loc:
            df_with_meta_and_indices = add_indices_column(df_with_meta)
            all_dfs.append(df_with_meta_and_indices)
        else:
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
    scale_factor=1,
    start_id=1,
    is_loc = None
):
    """
    Generate collective anomaly dataset.
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
            subcat = "collective_single"
            location_meta = f"{info_anom['location']}"
            anomaly_starts = info_anom['starts']
            anomaly_ends = info_anom['ends']
            anomaly_indices = [anomaly_starts.tolist(),anomaly_ends.tolist()]
            anomaly_count = 1
        elif anomaly_type == 'multiple':
            k = max(2, int(num_anomalies))
            df, info_anom = ts.generate_collective_anomalies(
                df, num_anomalies=k, location=location, scale_factor=scale_factor
            )
            label = f"{kind}_multiple_collective_anomalies_{l}"
            subcat = "collective_multiple"
            location_meta = "multiple"
            anomaly_starts = info_anom['starts']
            anomaly_ends = info_anom['ends']
            anomaly_indices = [anomaly_starts.tolist(),anomaly_ends.tolist()]
            anomaly_count = k
        else:
            raise ValueError("Invalid anomaly_type. Must be 'single' or 'multiple'.")

        series_id = start_id + i

        is_stat_flag = int(df['stationary'].iloc[0])
        is_seasonal_flag = int(df['seasonal'].iloc[0])
        df = df.drop(columns=['seasonal'])
        df = df.drop(columns=['stationary'])

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag,
            is_seasonal=is_seasonal_flag,
            primary_category="anomaly",
            primary_label=1,
            sub_category=subcat,
            sub_label=1,
            base_series=kind,
            base_coefs=base_coefs,
            order=base_order,
            anomaly_type=subcat,
            anomaly_count=anomaly_count,
            anomaly_indices=anomaly_indices,
            location_collective=location_meta
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        if is_loc:
            df_with_meta_and_indices = add_indices_column(df_with_meta)
            all_dfs.append(df_with_meta_and_indices)
        else:
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
    scale_factor=1,
    start_id=1,
    is_loc = None
):
    """
    Generate contextual anomaly dataset.
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
            subcat = "contextual_single"
            location_meta = f"{info2['location']}"
            anomaly_starts = info2['starts']
            anomaly_ends = info2['ends']
            anomaly_indices = [anomaly_starts.tolist(),anomaly_ends.tolist()]
            anomaly_count = 1
        elif anomaly_type == 'multiple':
            k = max(2, int(num_anomalies))
            df, info2 = ts.generate_contextual_anomalies(
                df, num_anomalies=k, location=location, scale_factor=scale_factor, seasonal_period=info1['period']
            )
            if info2 is None:
                print(f"Skipping one contextual anomaly generation for {folder} due to anomaly error.")
                continue
            label = f"multiple_contextual_anomalies_{l}"
            subcat = "contextual_multiple"
            location_meta = "multiple"
            anomaly_starts = info2['starts']
            anomaly_ends = info2['ends']
            anomaly_indices = [anomaly_starts.tolist(),anomaly_ends.tolist()]
            anomaly_count = 1
            anomaly_count = k
        else:
            raise ValueError("Invalid anomaly_type. Must be 'single' or 'multiple'.")

        series_id = start_id + i

        is_stat_flag = int(df['stationary'].iloc[0])
        is_seasonal_flag = int(df['seasonal'].iloc[0])
        df = df.drop(columns=['seasonal'])
        df = df.drop(columns=['stationary'])

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag,
            is_seasonal=is_seasonal_flag,
            primary_category="anomaly",
            primary_label=1,
            sub_category=subcat,
            sub_label=2,
            anomaly_type=subcat,
            anomaly_count=anomaly_count,
            anomaly_indices=anomaly_indices,  
            location_contextual=location_meta,
            seasonality_periods=[info1['period']]
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        if is_loc:
            df_with_meta_and_indices = add_indices_column(df_with_meta)
            all_dfs.append(df_with_meta_and_indices)
        else:
            all_dfs.append(df_with_meta)

    save_and_cleanup(all_dfs, folder, len(all_dfs), label)
