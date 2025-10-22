"""
Generators for volatility clustering time series datasets.

This module contains functions to generate datasets with volatility models:
- ARCH (Autoregressive Conditional Heteroscedasticity)
- GARCH (Generalized ARCH)
- EGARCH (Exponential GARCH)
- APARCH (Asymmetric Power ARCH)
"""

import os
import numpy as np
from ..core.metadata import create_metadata_record, attach_metadata_columns_to_df
from ..utils.helpers import save_and_cleanup, get_length_label


def generate_arch_dataset(
    ts_generator_class,
    folder,
    count=20,
    length_range=(50, 100)
):
    """
    Generate ARCH dataset.

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
        df, info = ts.generate_volatility(kind='arch')
        
        base_coefs = f"({info['alpha']},{info['omega']})"
        label = f"arch_{l}"
        series_id = i + 1

        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag,
            base_series="arch",
            base_coefs=base_coefs,
            volatility=1
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)
        
    save_and_cleanup(all_dfs, folder, count, label)


def generate_garch_dataset(
    ts_generator_class,
    folder,
    count=20,
    length_range=(50, 100)
):
    """
    Generate GARCH dataset.

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
        df, info = ts.generate_volatility(kind='garch')

        base_coefs = f"({info['alpha']},{info['beta']},{info['omega']})"
        label = f"garch_{l}"
        series_id = i + 1

        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag,
            base_series="garch",
            base_coefs=base_coefs,
            volatility=1
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    save_and_cleanup(all_dfs, folder, count, label)


def generate_egarch_dataset(
    ts_generator_class,
    folder,
    count=20,
    length_range=(50, 100)
):
    """
    Generate EGARCH dataset.

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
        df, info = ts.generate_volatility(kind='egarch')

        base_coefs = f"({info['alpha']},{info['beta']},{info['theta']},{info['lambda']},{info['omega']})"
        label = f"egarch_{l}"
        series_id = i + 1

        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag,
            base_series="egarch",
            base_coefs=base_coefs,
            volatility=1
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)
        
    save_and_cleanup(all_dfs, folder, count, label)


def generate_aparch_dataset(
    ts_generator_class,
    folder,
    count=20,
    length_range=(50, 100)
):
    """
    Generate APARCH dataset.

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
        df, info = ts.generate_volatility(kind='aparch')

        base_coefs = f"({info['alpha']},{info['beta']},{info['gamma']},{info['delta']},{info['omega']})"
        label = f"aparch_{l}"
        series_id = i + 1

        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag,
            base_series="aparch",
            base_coefs=base_coefs,
            volatility=1
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)
        
    save_and_cleanup(all_dfs, folder, count, label)

