"""
Generators for deterministic trend time series datasets.

This module contains functions to generate datasets with deterministic trends:
- Linear trend
- Quadratic trend
- Cubic trend
- Exponential trend
- Damped trend
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


def generate_linear_trend_dataset(
    ts_generator_class,
    folder,
    kind='ar',
    count=5,
    length_range=(300, 500),
    sign=1,
    start_id=1
):
    """
    Generate linear trend dataset.
    """
    os.makedirs(folder, exist_ok=True)
    all_dfs = []
    label = ""

    trend_label = "up" if sign == 1 else "down"
    l = get_length_label(length_range)

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = ts_generator_class(length=length)

        df, base_coefs, base_order = _get_base_series(ts, kind)
        df, info_trend = ts.generate_deterministic_trend_linear(df, sign=sign)

        label = f"{kind}_linear_trend_{l}_{trend_label}"
        series_id = start_id + i

        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag,
            primary_category="trend",
            sub_category="linear_up" if sign == 1 else "linear_down",
            base_series=kind,
            base_coefs=base_coefs,
            order=base_order,
            trend_type="linear",
            trend_slope=info_trend.get('slope'),
            trend_intercept=info_trend.get('intercept')
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    save_and_cleanup(all_dfs, folder, count, label)


def generate_quadratic_trend_dataset(
    ts_generator_class,
    folder,
    kind='ar',
    count=5,
    length_range=(300, 500),
    sign=1,
    start_id=1
):
    """
    Generate quadratic trend dataset.
    """
    os.makedirs(folder, exist_ok=True)
    all_dfs = []
    label = ""
    l = get_length_label(length_range)

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = ts_generator_class(length=length)
        
        df, base_coefs, base_order = _get_base_series(ts, kind)
        df, info_trend = ts.generate_deterministic_trend_quadratic(df, sign=sign, location="center")

        label = f"{kind}_quadratic_trend_{l}"
        series_id = start_id + i

        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag,
            primary_category="trend",
            sub_category="quadratic",
            base_series=kind,
            base_coefs=base_coefs,
            order=base_order,
            trend_type="quadratic",
            trend_coef_a=info_trend.get('a'),
            trend_coef_b=info_trend.get('b'),
            trend_coef_c=info_trend.get('c')
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    save_and_cleanup(all_dfs, folder, count, label)


def generate_cubic_trend_dataset(
    ts_generator_class,
    folder,
    kind='ar',
    count=5,
    length_range=(300, 500),
    sign=1,
    start_id=1
):
    """
    Generate cubic trend dataset.
    """
    os.makedirs(folder, exist_ok=True)
    all_dfs = []
    label = ""
    l = get_length_label(length_range)

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = ts_generator_class(length=length)

        df, base_coefs, base_order = _get_base_series(ts, kind)
        df, info_trend = ts.generate_deterministic_trend_cubic(df, sign=sign, location="center")

        label = f"{kind}_cubic_trend_{l}"
        series_id = start_id + i

        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag,
            primary_category="trend",
            sub_category="cubic",
            base_series=kind,
            base_coefs=base_coefs,
            order=base_order,
            trend_type="cubic",
            trend_coef_a=info_trend.get('a'),
            trend_coef_b=info_trend.get('b')
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    save_and_cleanup(all_dfs, folder, count, label)


def generate_exponential_trend_dataset(
    ts_generator_class,
    folder,
    kind='ar',
    count=5,
    length_range=(300, 500),
    sign=1,
    start_id=1
):
    """
    Generate exponential trend dataset.
    """
    os.makedirs(folder, exist_ok=True)
    all_dfs = []
    label = ""
    l = get_length_label(length_range)

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = ts_generator_class(length=length)

        df, base_coefs, base_order = _get_base_series(ts, kind)
        df, info_trend = ts.generate_deterministic_trend_exponential(df, sign=sign)

        label = f"{kind}_exponential_trend_{l}"
        series_id = start_id + i

        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag,
            primary_category="trend",
            sub_category="exponential",
            base_series=kind,
            base_coefs=base_coefs,
            order=base_order,
            trend_type="exponential",
            trend_coef_a=info_trend.get('a'),
            trend_coef_b=info_trend.get('b')
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    save_and_cleanup(all_dfs, folder, count, label)


def generate_damped_trend_dataset(
    ts_generator_class,
    folder,
    kind='ar',
    count=5,
    length_range=(300, 500),
    sign=1,
    start_id=1
):
    """
    Generate damped trend dataset.
    """
    os.makedirs(folder, exist_ok=True)
    all_dfs = []
    label = ""
    l = get_length_label(length_range)

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = ts_generator_class(length=length)

        df, base_coefs, base_order = _get_base_series(ts, kind)
        df, info_trend = ts.generate_deterministic_trend_damped(df, sign=sign)

        label = f"{kind}_damped_trend_{l}"
        series_id = start_id + i

        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag,
            primary_category="trend",
            sub_category="damped",
            base_series=kind,
            base_coefs=base_coefs,
            order=base_order,
            trend_type="damped",
            trend_damping_rate=info_trend.get('damping_rate'),
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    save_and_cleanup(all_dfs, folder, count, label)

