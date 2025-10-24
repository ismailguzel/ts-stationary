"""
Utility functions for dataset generation.

This module contains helper functions used across the package.
"""

from .helpers import save_and_cleanup, get_length_label

# Import visualization functions
from .visualization import (
    plot_single_series,
    plot_multiple_series,
    plot_series_comparison,
    plot_distribution,
    plot_acf_pacf,
    plot_rolling_statistics,
    plot_category_overview,
    create_dashboard
)

# Import analysis functions
from .analysis import (
    compute_basic_statistics,
    test_stationarity,
    test_normality,
    detect_seasonality,
    detect_trend,
    detect_changepoints,
    compute_autocorrelation_stats,
    compute_entropy,
    analyze_dataset_summary,
    compare_series
)

__all__ = [
    # Helper functions
    'save_and_cleanup',
    'get_length_label',
    # Visualization functions
    'plot_single_series',
    'plot_multiple_series',
    'plot_series_comparison',
    'plot_distribution',
    'plot_acf_pacf',
    'plot_rolling_statistics',
    'plot_category_overview',
    'create_dashboard',
    # Analysis functions
    'compute_basic_statistics',
    'test_stationarity',
    'test_normality',
    'detect_seasonality',
    'detect_trend',
    'detect_changepoints',
    'compute_autocorrelation_stats',
    'compute_entropy',
    'analyze_dataset_summary',
    'compare_series'
]

