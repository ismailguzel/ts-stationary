"""
Generators for stationary time series datasets.

This module contains functions to generate datasets with stationary base processes:
- White Noise
- AR (Autoregressive)
- MA (Moving Average)
- ARMA (Autoregressive Moving Average)
"""

import os
import numpy as np
from ..core.metadata import create_metadata_record, attach_metadata_columns_to_df
from ..utils.helpers import save_and_cleanup, get_length_label, add_indices_column


def generate_wn_dataset(
    ts_generator_class,
    folder,
    count=20,
    length_range=(50, 100),
    start_id=1,
    is_loc=None
):
    """
    Generate white noise dataset.

    Parameters
    ----------
    ts_generator_class : class
        TimeSeriesGenerator class
    folder : str
        Output folder path
    count : int, default=20
        Number of series to generate
    length_range : tuple, default=(50, 100)
        (min, max) length for generated series
    start_id : int, default=1
        Starting ID for series_id

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
        df, info = ts.generate_stationary_base_series('white_noise')

        label = f"white_noise_{l}"
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
            primary_category="stationary",
            primary_label=0,
            sub_category="white_noise",
            sub_label=0,
            base_series="white_noise",
        )
        
        df_with_meta = attach_metadata_columns_to_df(df, record)
        if is_loc:
            df_with_meta_and_indices = add_indices_column(df_with_meta)
            all_dfs.append(df_with_meta_and_indices)
        else:
            all_dfs.append(df_with_meta)

    save_and_cleanup(all_dfs, folder, count, label)


def generate_ar_dataset(
    ts_generator_class,
    folder,
    count=20,
    length_range=(50, 100),
    start_id=1,
    is_loc=None
):
    """
    Generate AR (Autoregressive) dataset.

    Parameters
    ----------
    ts_generator_class : class
        TimeSeriesGenerator class
    folder : str
        Output folder path
    count : int, default=20
        Number of series to generate
    length_range : tuple, default=(50, 100)
        (min, max) length for generated series
    start_id : int, default=1
        Starting ID for series_id

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
        df, info = ts.generate_stationary_base_series('ar')
        
        base_coefs = f"({info['ar_coefs']})"
        base_order = f"({info['ar_order']})"
        label = f"ar_{l}"
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
            primary_category="stationary",
            primary_label=0,
            sub_category="ar",
            sub_label=0,
            base_series="ar",
            base_coefs=base_coefs,
            order=base_order
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        if is_loc:
            df_with_meta_and_indices = add_indices_column(df_with_meta)
            all_dfs.append(df_with_meta_and_indices)
        else:
            all_dfs.append(df_with_meta)

    save_and_cleanup(all_dfs, folder, count, label)


def generate_ma_dataset(
    ts_generator_class,
    folder,
    count=20,
    length_range=(50, 100),
    start_id=1,
    is_loc=None
):
    """
    Generate MA (Moving Average) dataset.

    Parameters
    ----------
    ts_generator_class : class
        TimeSeriesGenerator class
    folder : str
        Output folder path
    count : int, default=20
        Number of series to generate
    length_range : tuple, default=(50, 100)
        (min, max) length for generated series
    start_id : int, default=1
        Starting ID for series_id

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
        df, info = ts.generate_stationary_base_series('ma')

        base_coefs = f"({info['ma_coefs']})"
        base_order = f"({info['ma_order']})"
        label = f"ma_{l}"
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
            primary_category="stationary",
            primary_label=0,
            sub_category="ma",
            sub_label=0,
            base_series="ma",
            base_coefs=base_coefs,
            order=base_order
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        if is_loc:
            df_with_meta_and_indices = add_indices_column(df_with_meta)
            all_dfs.append(df_with_meta_and_indices)
        else:
            all_dfs.append(df_with_meta)

    save_and_cleanup(all_dfs, folder, count, label)


def generate_arma_dataset(
    ts_generator_class,
    folder,
    count=20,
    length_range=(50, 100),
    start_id=1,
    is_loc=None
):
    """
    Generate ARMA (Autoregressive Moving Average) dataset.

    Parameters
    ----------
    ts_generator_class : class
        TimeSeriesGenerator class
    folder : str
        Output folder path
    count : int, default=20
        Number of series to generate
    length_range : tuple, default=(50, 100)
        (min, max) length for generated series
    start_id : int, default=1
        Starting ID for series_id

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
        df, info = ts.generate_stationary_base_series('arma')

        base_coefs = f"({info['ar_coefs']},{info['ma_coefs']})"
        base_order = f"({info['ar_order']},{info['ma_order']})"
        label = f"arma_{l}"
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
            primary_category="stationary",
            primary_label=0,
            sub_category="arma",
            sub_label=0,
            base_series="arma",
            base_coefs=base_coefs,
            order=base_order
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        if is_loc:
            df_with_meta_and_indices = add_indices_column(df_with_meta)
            all_dfs.append(df_with_meta_and_indices)
        else:
            all_dfs.append(df_with_meta)

    save_and_cleanup(all_dfs, folder, count, label)

