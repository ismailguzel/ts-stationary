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
    length_range=(50, 100),
    start_id=1
):
    """
    Generate ARCH dataset.
    """
    os.makedirs(folder, exist_ok=True)
    all_dfs = []
    label = ""
    l = get_length_label(length_range)

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = ts_generator_class(length=length)
        df, info = ts.generate_volatility(kind='arch')

        # Yeni metadata
        series_id = start_id + i
        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])
        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=f"arch_{l}",
            is_stationary=is_stat_flag,
            primary_category="volatility",
            sub_category="arch",
            base_series="arch",
            base_coefs=None,
            volatility_type="arch",
            volatility_alpha=info.get("alpha"),
            volatility_omega=info.get("omega"),
        )
        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)
    save_and_cleanup(all_dfs, folder, count, f"arch_{l}")


def generate_garch_dataset(
    ts_generator_class,
    folder,
    count=20,
    length_range=(50, 100),
    start_id=1
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
        df, info = ts.generate_volatility(kind='garch')

        # Yeni metadata
        series_id = start_id + i
        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])
        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=f"garch_{l}",
            is_stationary=is_stat_flag,
            primary_category="volatility",
            sub_category="garch",
            base_series="garch",
            base_coefs=None,
            volatility_type="garch",
            volatility_alpha=info.get("alpha"),
            volatility_beta=info.get("beta"),
            volatility_omega=info.get("omega"),
        )
        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)
    save_and_cleanup(all_dfs, folder, count, f"garch_{l}")


def generate_egarch_dataset(
    ts_generator_class,
    folder,
    count=20,
    length_range=(50, 100),
    start_id=1
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
        df, info = ts.generate_volatility(kind='egarch')

        # Yeni metadata
        series_id = start_id + i
        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])
        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=f"egarch_{l}",
            is_stationary=is_stat_flag,
            primary_category="volatility",
            sub_category="egarch",
            base_series="egarch",
            base_coefs=None,
            volatility_type="egarch",
            volatility_alpha=info.get("alpha"),
            volatility_beta=info.get("beta"),
            volatility_omega=info.get("omega"),
            volatility_theta=info.get("theta"),
            volatility_lambda=info.get("lambda"),
        )
        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)
    save_and_cleanup(all_dfs, folder, count, f"egarch_{l}")


def generate_aparch_dataset(
    ts_generator_class,
    folder,
    count=20,
    length_range=(50, 100),
    start_id=1
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
        df, info = ts.generate_volatility(kind='aparch')

        # Yeni metadata
        series_id = start_id + i
        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])
        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=f"aparch_{l}",
            is_stationary=is_stat_flag,
            primary_category="volatility",
            sub_category="aparch",
            base_series="aparch",
            base_coefs=None,
            volatility_type="aparch",
            volatility_alpha=info.get("alpha"),
            volatility_beta=info.get("beta"),
            volatility_omega=info.get("omega"),
            volatility_gamma=info.get("gamma"),
            volatility_delta=info.get("delta"),
        )
        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)
    save_and_cleanup(all_dfs, folder, count, f"aparch_{l}")

