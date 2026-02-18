"""
Generators for seasonal time series datasets.

This module contains functions to generate datasets with seasonality:
- Single Seasonality
- Multiple Seasonality
- SARMA (Seasonal ARMA)
- SARIMA (Seasonal ARIMA)
"""

import os
import numpy as np
from ..core.metadata import create_metadata_record, attach_metadata_columns_to_df
from ..utils.helpers import save_and_cleanup, get_length_label, add_indices_column


def generate_single_seasonality_dataset(
    ts_generator_class,
    folder,
    count=20,
    length_range=(50, 100),
    start_id=1,
    is_loc=None
):
    """
    Generate single seasonality dataset.
    """
    os.makedirs(folder, exist_ok=True)
    all_dfs = []
    label = ""
    l = get_length_label(length_range)

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = ts_generator_class(length=length)
        df, info = ts.generate_seasonality_from_base_series(kind='single')
        if df is None: 
            continue
        
        period = info['period']
        label = f"single_seasonality_{l}"
        series_id = start_id + i

        is_stat_flag = int(df['stationary'].iloc[0])
        is_seasonal_flag = int(df['seasonal'].iloc[0])
        df = df.drop(columns=['stationary'])
        df = df.drop(columns=['seasonal'])

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag,
            is_seasonal=is_seasonal_flag,
            primary_category="seasonality",
            primary_label=4,
            sub_category="single",
            sub_label=0,
            seasonality_periods=[period]
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        if is_loc:
            df_with_meta_and_indices = add_indices_column(df_with_meta)
            all_dfs.append(df_with_meta_and_indices)
        else:
            all_dfs.append(df_with_meta)
        
    save_and_cleanup(all_dfs, folder, len(all_dfs), label)


def generate_multiple_seasonality_dataset(
    ts_generator_class,
    folder,
    count=20,
    num_components=2,
    length_range=(50, 100),
    start_id=1,
    is_loc=None
):
    """
    Generate multiple seasonality dataset.
    """
    os.makedirs(folder, exist_ok=True)
    all_dfs = []
    label = ""
    l = get_length_label(length_range)

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = ts_generator_class(length=length)
        df, info = ts.generate_seasonality_from_base_series(kind='multiple', num_components=num_components)
        if df is None: 
            continue
        
        periods = info['periods']
        label = f"multiple_seasonality_{l}"
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
            primary_category="seasonality",
            primary_label=4,
            sub_category="multiple",
            sub_label=1,
            seasonality_periods=periods
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        if is_loc:
            df_with_meta_and_indices = add_indices_column(df_with_meta)
            all_dfs.append(df_with_meta_and_indices)
        else:
            all_dfs.append(df_with_meta)
        
    save_and_cleanup(all_dfs, folder, len(all_dfs), label)


def generate_sarma_dataset(
    ts_generator_class,
    folder,
    count=20,
    length_range=(50, 100),
    start_id=1,
    is_loc=None
):
    """
    Generate SARMA (Seasonal ARMA) dataset.
    """
    os.makedirs(folder, exist_ok=True)
    all_dfs = []
    label = ""
    l = get_length_label(length_range)

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = ts_generator_class(length=length)
        df, info = ts.generate_seasonality_from_base_series(kind='sarma')
        if df is None or info is None:
            print(f"Skipping one SARMA generation for {folder} due to an error.")
            continue
             
        base_coefs = f"({info['coefs']})"
        base_order = f"({info['ar_order']}, {info['ma_order']}, {info['seasonal_ar_order']}, {info['seasonal_ma_order']})"
        period = info['period']
        diff = info['diff']
        seasonal_diff = info['seasonal_diff']
        label = f"sarma_seasonality_{l}"
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
            primary_category="seasonality",
            primary_label=4,
            sub_category="sarma",
            sub_label=2,
            base_series='sarma',
            seasonality_from_base=1,
            seasonality_periods=[period],
            base_coefs=base_coefs,
            order=base_order,
            difference=diff,
            seasonal_difference=seasonal_diff,
            seasonal_ar_order=info['seasonal_ar_order'],
            seasonal_ma_order=info['seasonal_ma_order']
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        if is_loc:
            df_with_meta_and_indices = add_indices_column(df_with_meta)
            all_dfs.append(df_with_meta_and_indices)
        else:
            all_dfs.append(df_with_meta)

    save_and_cleanup(all_dfs, folder, len(all_dfs), label)


def generate_sarima_dataset(
    ts_generator_class,
    folder,
    count=20,
    length_range=(50, 100),
    start_id=1,
    is_loc=None
):
    """
    Generate SARIMA (Seasonal ARIMA) dataset.
    """
    os.makedirs(folder, exist_ok=True)
    all_dfs = []
    label = ""
    l = get_length_label(length_range)

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = ts_generator_class(length=length)
        df, info = ts.generate_seasonality_from_base_series(kind='sarima')
        if df is None or info is None:
            print(f"Skipping one SARIMA generation for {folder} due to an error.")
            continue

        base_coefs = f"({info['coefs']})"
        base_order = f"({info['ar_order']}, {info['ma_order']}, {info['seasonal_ar_order']}, {info['seasonal_ma_order']})"
        period = info['period']
        diff = info['diff']
        seasonal_diff = info['seasonal_diff']
        label = f"sarima_seasonality_{l}"
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
            primary_category="seasonality",
            primary_label=4,
            sub_category="sarima",
            sub_label=3,
            base_series='sarima',
            seasonality_from_base=1,
            seasonality_periods=[period],
            base_coefs=base_coefs,
            order=base_order,
            difference=diff,
            seasonal_difference=seasonal_diff,
            seasonal_ar_order=info['seasonal_ar_order'],
            seasonal_ma_order=info['seasonal_ma_order']
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        if is_loc:
            df_with_meta_and_indices = add_indices_column(df_with_meta)
            all_dfs.append(df_with_meta_and_indices)
        else:
            all_dfs.append(df_with_meta)

    save_and_cleanup(all_dfs, folder, len(all_dfs), label)

