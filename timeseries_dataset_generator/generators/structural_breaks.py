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
    seasonal_period=None
):
    """
    Generate mean shift dataset.

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
    break_type : str, default='single'
        'single' or 'multiple'
    signs : list, default=[1]
        Signs of shifts (1 for increase, -1 for decrease)
    location : str, default='middle'
        Location of shift ('beginning', 'middle', 'end')
    num_breaks : int, default=2
        Number of breaks (for multiple type)
    scale_factor : float, default=1
        Scale factor for shift magnitude
    seasonal_period : int, optional
        Seasonal period if applicable

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

        if signs[0] < 0:
            mean_shift_increase_meta = 0
            mean_shift_decrease_meta = 1 
        else:
            mean_shift_increase_meta = 1
            mean_shift_decrease_meta = 0

        if break_type == 'single':
            loc = location if location else np.random.choice(['beginning', 'middle', 'end'])
            df, info2 = ts.generate_mean_shift(
                df, signs=signs, num_breaks=1, location=loc,
                scale_factor=scale_factor, seasonal_period=seasonal_period
            )
            label = f"{kind}_single_mean_shift_{loc}_{l}"
            location_meta = f"{info2['location']}"
            shift_indices_magnitudes = f"{info2['shift_indices']},{info2['shift_magnitudes']}"
            multi_mean_shift_meta = 0
            
        elif break_type == 'multiple':
            k = max(2, int(num_breaks))
            signs_meta = [np.random.choice([1, -1]) for _ in range(k)]
            df, info2 = ts.generate_mean_shift(
                df, signs=signs_meta, num_breaks=k,
                scale_factor=scale_factor, seasonal_period=seasonal_period
            )
            label = f"{kind}_multiple_mean_shifts_{l}"
            multi_mean_shift_meta = 1
            location_meta = "multiple"
            shift_indices_magnitudes = f"{info2['shift_indices']},{info2['shift_magnitudes']}"
            mean_shift_increase_meta = 0
            mean_shift_decrease_meta = 0
        else:
            raise ValueError("Invalid break_type. Must be 'single' or 'multiple'.")

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
            mean_shift_increase=mean_shift_increase_meta,
            mean_shift_decrease=mean_shift_decrease_meta,
            multi_mean_shift=multi_mean_shift_meta,
            location_mean_shift=location_meta,
            location_mean_pts=shift_indices_magnitudes
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
    seasonal_period=None
):
    """
    Generate variance shift dataset.

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
    break_type : str, default='single'
        'single' or 'multiple'
    signs : list, default=[1]
        Signs of shifts (1 for increase, -1 for decrease)
    location : str, default='middle'
        Location of shift ('beginning', 'middle', 'end')
    num_breaks : int, default=2
        Number of breaks (for multiple type)
    scale_factor : float, default=1
        Scale factor for shift magnitude
    seasonal_period : int, optional
        Seasonal period if applicable

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

        if signs[0] < 0:
            variance_shift_increase_meta = 0
            variance_shift_decrease_meta = 1 
        else:
            variance_shift_increase_meta = 1
            variance_shift_decrease_meta = 0

        if break_type == 'single':
            loc = location if location else np.random.choice(['beginning', 'middle', 'end'])
            df, info2 = ts.generate_variance_shift(
                df, signs=signs, num_breaks=1, location=loc,
                scale_factor=scale_factor, seasonal_period=seasonal_period
            )
            label = f"{kind}_single_variance_shift_{loc}_{l}"
            location_meta = f"{info2['location']}"
            shift_indices_magnitudes = f"{info2['shift_indices']},{info2['shift_magnitudes']}"
            multi_variance_shift_meta = 0
            
        elif break_type == 'multiple':
            k = max(2, int(num_breaks))
            signs_meta = [np.random.choice([1, -1]) for _ in range(k)]
            df, info2 = ts.generate_variance_shift(
                df, signs=signs_meta, num_breaks=k,
                scale_factor=scale_factor, seasonal_period=seasonal_period
            )
            label = f"{kind}_multiple_variance_shifts_{l}"
            multi_variance_shift_meta = 1
            location_meta = "multiple"
            shift_indices_magnitudes = f"{info2['shift_indices']},{info2['shift_magnitudes']}"
            variance_shift_increase_meta = 0
            variance_shift_decrease_meta = 0
        else:
            raise ValueError("Invalid break_type. Must be 'single' or 'multiple'.")

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
            variance_shift_increase=variance_shift_increase_meta,
            variance_shift_decrease=variance_shift_decrease_meta,
            multi_variance_shift=multi_variance_shift_meta,
            location_variance_shift=location_meta,
            location_variance_pts=shift_indices_magnitudes
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
    sign=None
):
    """
    Generate trend shift dataset.

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
    break_type : str, default='single'
        'single' or 'multiple'
    change_types : list, default=['direction_change']
        Types of changes ('direction_change', 'magnitude_change', 'direction_and_magnitude_change')
    location : str, default='middle'
        Location of shift ('beginning', 'middle', 'end')
    num_breaks : int, default=2
        Number of breaks (for multiple type)
    scale_factor : float, default=1
        Scale factor for shift magnitude
    seasonal_period : int, optional
        Seasonal period if applicable
    sign : int, optional
        Sign of trend shift

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

        current_change_types = change_types
        if break_type == 'multiple':
            k = max(2, int(num_breaks))
            current_change_types = np.random.choice(
                ['direction_change', 'magnitude_change', 'direction_and_magnitude_change'], 
                k
            ).tolist()
        
        if break_type == 'single':
            if current_change_types[0] == 'direction_change':
                trend_shift_slope_meta = 1
                trend_shift_intercept_meta = 0 
            elif current_change_types[0] == 'magnitude_change':
                trend_shift_slope_meta = 0
                trend_shift_intercept_meta = 1
            elif current_change_types[0] == 'direction_and_magnitude_change':
                trend_shift_slope_meta = 1
                trend_shift_intercept_meta = 1
            else:
                raise ValueError(f"Unknown change type: {current_change_types[0]}")
        
        sign = sign if sign is not None else np.random.choice([-1, 1])
        
        if break_type == 'single':
            loc = location if location else np.random.choice(['beginning', 'middle', 'end'])
            df, info2 = ts.generate_trend_shift(
                df, sign=sign, num_breaks=1, change_types=current_change_types,
                location=loc, scale_factor=scale_factor, seasonal_period=seasonal_period
            )
            label = f"{kind}_single_trend_shift_{loc}_{l}"
            location_meta = f"{info2['location']}"
            shift_indices_types = f"{info2['shift_indices']},{info2['shift_types']}"
            multi_trend_shift_meta = 0
            
        elif break_type == 'multiple':
            df, info2 = ts.generate_trend_shift(
                df, sign=sign, change_types=current_change_types, num_breaks=k,
                scale_factor=scale_factor, seasonal_period=seasonal_period
            )
            label = f"{kind}_multiple_trend_shifts_{l}"
            multi_trend_shift_meta = 1
            location_meta = "multiple"
            shift_indices_types = f"{info2['shift_indices']},{info2['shift_types']}"
            trend_shift_slope_meta = 0
            trend_shift_intercept_meta = 0
        else:
            raise ValueError("Invalid break_type. Must be 'single' or 'multiple'.")

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
            trend_shift_slope=trend_shift_slope_meta,
            trend_shift_intercept=trend_shift_intercept_meta,
            multi_trend_shift=multi_trend_shift_meta,
            location_trend_shift=location_meta,
            location_trend_pts=shift_indices_types
        )

        df_with_meta = attach_metadata_columns_to_df(df, record)
        all_dfs.append(df_with_meta)

    save_and_cleanup(all_dfs, folder, count, label)

