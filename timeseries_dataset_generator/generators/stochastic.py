"""
Generators for stochastic trend time series datasets.

This module contains functions to generate datasets with stochastic trends:
- Random Walk
- Random Walk with Drift
- IMA (Integrated Moving Average)
- ARI (Autoregressive Integrated)
- ARIMA (Autoregressive Integrated Moving Average)
"""

import os
import numpy as np
from ..core.metadata import create_metadata_record, attach_metadata_columns_to_df
from ..utils.helpers import save_and_cleanup, get_length_label, add_indices_column


def generate_random_walk_dataset(
    ts_generator_class,
    folder,
    count=20,
    length_range=(50, 100),
    start_id=1,
    is_loc=None
):
    """
    Generate random walk dataset.

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
        df, info = ts.generate_stochastic_trend(kind='rw')
        
        base_coefs = 0
        base_order = 0
        label = f"random_walk_{l}"
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
            primary_category="stochastic",
            primary_label=3,
            sub_category="rw",
            sub_label=0,
            base_series="rw",
            base_coefs=base_coefs,
            order=base_order,
            stochastic_type="rw"
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        if is_loc:
            df_with_meta_and_indices = add_indices_column(df_with_meta)
            all_dfs.append(df_with_meta_and_indices)
        else:
            all_dfs.append(df_with_meta)

    save_and_cleanup(all_dfs, folder, count, label)


def generate_random_walk_with_drift_dataset(
    ts_generator_class,
    folder,
    count=20,
    length_range=(50, 100),
    start_id=1,
    is_loc=None
):
    """
    Generate random walk with drift dataset.

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
        df, info = ts.generate_stochastic_trend(kind='rwd')
        
        base_coefs = 0
        base_order = 0
        label = f"random_walk_drift_{l}"
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
            primary_category="stochastic",
            primary_label=3,
            sub_category="rwd",
            sub_label=0,
            base_series="rwd",
            base_coefs=base_coefs,
            order=base_order,
            stochastic_type="rwd",
            drift_value=info.get('drift')
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        if is_loc:
            df_with_meta_and_indices = add_indices_column(df_with_meta)
            all_dfs.append(df_with_meta_and_indices)
        else:
            all_dfs.append(df_with_meta)

    save_and_cleanup(all_dfs, folder, count, label)


def generate_ima_dataset(
    ts_generator_class,
    folder,
    count=20,
    length_range=(50, 100),
    start_id=1,
    is_loc=None
):
    """
    Generate IMA (Integrated Moving Average) dataset.

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
        df, info = ts.generate_stochastic_trend(kind='ima')

        base_coefs = f"({info['ma_coefs']})"
        base_order = f"({info['ma_order']})"
        diff = info['diff']
        label = f"ima_{l}"
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
            primary_category="stochastic",
            primary_label=3,
            sub_category="ima",
            sub_label=0,
            base_series="ima",
            order=base_order,
            base_coefs=base_coefs,
            stochastic_type="ima",
            difference=diff
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        if is_loc:
            df_with_meta_and_indices = add_indices_column(df_with_meta)
            all_dfs.append(df_with_meta_and_indices)
        else:
            all_dfs.append(df_with_meta)

    save_and_cleanup(all_dfs, folder, count, label)


def generate_ari_dataset(
    ts_generator_class,
    folder,
    count=20,
    length_range=(50, 100),
    start_id=1,
    is_loc=None
):
    """
    Generate ARI (Autoregressive Integrated) dataset.

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
        df, info = ts.generate_stochastic_trend(kind='ari')

        base_coefs = f"({info['ar_coefs']})"
        base_order = f"({info['ar_order']})"
        diff = info['diff']
        label = f"ari_{l}"
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
            primary_category="stochastic",
            primary_label=3,
            sub_label=0,
            sub_category="ari",
            base_series="ari",
            order=base_order,
            base_coefs=base_coefs,
            stochastic_type="ari",
            difference=diff
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        if is_loc:
            df_with_meta_and_indices = add_indices_column(df_with_meta)
            all_dfs.append(df_with_meta_and_indices)
        else:
            all_dfs.append(df_with_meta)

    save_and_cleanup(all_dfs, folder, count, label)


def generate_arima_dataset(
    ts_generator_class,
    folder,
    count=20,
    length_range=(50, 100),
    start_id=1,
    is_loc=None
):
    """
    Generate ARIMA (Autoregressive Integrated Moving Average) dataset.

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
        df, info = ts.generate_stochastic_trend(kind='arima')

        base_coefs = f"({info['ar_coefs']},{info['ma_coefs']})"
        base_order = f"({info['ar_order']},{info['ma_order']})"
        diff = info['diff']
        label = f"arima_{l}"
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
            primary_category="stochastic",
            primary_label=3,
            sub_label=0,
            sub_category="arima",
            base_series="arima",
            order=base_order,
            base_coefs=base_coefs,
            stochastic_type="arima",
            difference=diff
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        if is_loc:
            df_with_meta_and_indices = add_indices_column(df_with_meta)
            all_dfs.append(df_with_meta_and_indices)
        else:
            all_dfs.append(df_with_meta)

    save_and_cleanup(all_dfs, folder, count, label)

