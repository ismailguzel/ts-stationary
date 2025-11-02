"""
Generators for time series datasets with structural breaks.

This module contains functions to generate datasets with structural breaks:
- Mean Shifts (single and multiple)
- Variance Shifts (single and multiple)
- Trend Shifts (single and multiple)
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


def generate_mean_shift_dataset(
    ts_generator_class,
    folder,
    kind='ar',
    count=5,
    length_range=(300, 500),    
    break_type='single',
    signs=[1],
    location="middle",              
    num_breaks=2,
    scale_factor=1,
    seasonal_period=None,
    start_id=1
):
    """
    Generate mean shift dataset.
    """
    os.makedirs(folder, exist_ok=True)
    all_dfs = []
    label = ""
    l = get_length_label(length_range)

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = ts_generator_class(length=length)

        df, base_coefs, base_order = _get_base_series(ts, kind)

        if break_type == 'single':
            loc = location if location else np.random.choice(['beginning', 'middle', 'end'])
            df, info2 = ts.generate_mean_shift(
                df, signs=signs, num_breaks=1, location=loc,
                scale_factor=scale_factor, seasonal_period=seasonal_period
            )
            label = f"{kind}_single_mean_shift_{loc}_{l}"
            location_meta = f"{info2['location']}"
        elif break_type == 'multiple':
            k = max(2, int(num_breaks))
            signs_meta = [np.random.choice([1, -1]) for _ in range(k)]
            df, info2 = ts.generate_mean_shift(
                df, signs=signs_meta, num_breaks=k,
                scale_factor=scale_factor, seasonal_period=seasonal_period
            )
            label = f"{kind}_multiple_mean_shifts_{l}"
            location_meta = "multiple"
        else:
            raise ValueError("Invalid break_type. Must be 'single' or 'multiple'.")

        shift_indices = info2['shift_indices']
        shift_magnitudes = info2['shift_magnitudes']
        break_count_val = len(shift_indices)

        series_id = start_id + i

        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag,
            primary_category="structural_break",
            sub_category="mean_shift",
            base_series=kind,
            base_coefs=base_coefs,
            order=base_order,
            break_type="mean_shift",
            break_count=break_count_val,
            break_indices=shift_indices,
            break_magnitudes=shift_magnitudes,
            break_directions=signs,
            location_mean_shift=location_meta
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    save_and_cleanup(all_dfs, folder, count, label)


def generate_variance_shift_dataset(
    ts_generator_class,
    folder,
    kind='ar',
    count=5,
    length_range=(300, 500),    
    break_type='single',
    signs=[1],
    location="middle",              
    num_breaks=2,
    scale_factor=1,
    seasonal_period=None,
    start_id=1
):
    """
    Generate variance shift dataset.
    """
    os.makedirs(folder, exist_ok=True)
    all_dfs = []
    label = ""
    l = get_length_label(length_range)

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = ts_generator_class(length=length)

        df, base_coefs, base_order = _get_base_series(ts, kind)

        if break_type == 'single':
            loc = location if location else np.random.choice(['beginning', 'middle', 'end'])
            df, info2 = ts.generate_variance_shift(
                df, signs=signs, num_breaks=1, location=loc,
                scale_factor=scale_factor, seasonal_period=seasonal_period
            )
            label = f"{kind}_single_variance_shift_{loc}_{l}"
            location_meta = f"{info2['location']}"
        elif break_type == 'multiple':
            k = max(2, int(num_breaks))
            signs_meta = [np.random.choice([1, -1]) for _ in range(k)]
            df, info2 = ts.generate_variance_shift(
                df, signs=signs_meta, num_breaks=k,
                scale_factor=scale_factor, seasonal_period=seasonal_period
            )
            label = f"{kind}_multiple_variance_shifts_{l}"
            location_meta = "multiple"
        else:
            raise ValueError("Invalid break_type. Must be 'single' or 'multiple'.")

        shift_indices = info2['shift_indices']
        shift_magnitudes = info2['shift_magnitudes']
        break_count_val = len(shift_indices)

        series_id = start_id + i

        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag,
            primary_category="structural_break",
            sub_category="variance_shift",
            base_series=kind,
            base_coefs=base_coefs,
            order=base_order,
            break_type="variance_shift",
            break_count=break_count_val,
            break_indices=shift_indices,
            break_magnitudes=shift_magnitudes,
            break_directions=signs,
            location_variance_shift=location_meta
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    save_and_cleanup(all_dfs, folder, count, label)


def generate_trend_shift_dataset(
    ts_generator_class,
    folder,
    kind='ar',
    count=5,
    length_range=(300, 500),    
    break_type='single',
    change_types=['direction_change'],
    location="middle",              
    num_breaks=2,
    scale_factor=1,
    seasonal_period=None,
    sign=None,
    start_id=1
):
    """
    Generate trend shift dataset.
    """
    os.makedirs(folder, exist_ok=True)
    all_dfs = []
    label = ""
    l = get_length_label(length_range)

    for i in range(count):
        length = np.random.randint(*length_range)
        ts = ts_generator_class(length=length)

        df, base_coefs, base_order = _get_base_series(ts, kind)

        current_change_types = change_types
        if break_type == 'multiple':
            k = max(2, int(num_breaks))
            current_change_types = np.random.choice(
                ['direction_change', 'magnitude_change', 'direction_and_magnitude_change'], 
                k
            ).tolist()
        
        sign = sign if sign is not None else np.random.choice([-1, 1])
        
        if break_type == 'single':
            loc = location if location else np.random.choice(['beginning', 'middle', 'end'])
            df, info2 = ts.generate_trend_shift(
                df, sign=sign, num_breaks=1, change_types=current_change_types,
                location=loc, scale_factor=scale_factor, seasonal_period=seasonal_period
            )
            label = f"{kind}_single_trend_shift_{loc}_{l}"
            location_meta = f"{info2['location']}"
        elif break_type == 'multiple':
            df, info2 = ts.generate_trend_shift(
                df, sign=sign, change_types=current_change_types, num_breaks=k,
                scale_factor=scale_factor, seasonal_period=seasonal_period
            )
            label = f"{kind}_multiple_trend_shifts_{l}"
            location_meta = "multiple"
        else:
            raise ValueError("Invalid break_type. Must be 'single' or 'multiple'.")

        shift_indices = info2['shift_indices']
        shift_types = info2['shift_types']
        break_count_val = len(shift_indices)

        series_id = start_id + i

        is_stat_flag = int(df['stationary'].iloc[0])
        df = df.drop(columns=['stationary'])

        record = create_metadata_record(
            series_id=series_id,
            length=length,
            label=label,
            is_stationary=is_stat_flag,
            primary_category="structural_break",
            sub_category="trend_shift",
            base_series=kind,
            base_coefs=base_coefs,
            order=base_order,
            break_type="trend_shift",
            break_count=break_count_val,
            break_indices=shift_indices,
            trend_shift_change_types=shift_types,
            location_trend_shift=location_meta
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    save_and_cleanup(all_dfs, folder, count, label)

