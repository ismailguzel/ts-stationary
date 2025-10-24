"""
Time Series Dataset Generator

A comprehensive toolkit for generating synthetic time series datasets with various
characteristics including trends, seasonality, anomalies, and structural breaks.

Main Components:
- TimeSeriesGenerator: Core class for generating time series
- Stationary generators: AR, MA, ARMA, White Noise
- Trend generators: Linear, Quadratic, Cubic, Exponential, Damped
- Stochastic generators: Random Walk, ARIMA family
- Volatility generators: ARCH, GARCH, EGARCH, APARCH
- Seasonality generators: Single, Multiple, SARMA, SARIMA
- Anomaly generators: Point, Collective, Contextual
- Structural break generators: Mean, Variance, Trend shifts
- Visualization utilities: Plot single/multiple series, distributions, ACF/PACF, dashboards
- Analysis utilities: Statistical tests, trend/seasonality detection, changepoint analysis

Example:
    >>> from timeseries_dataset_generator import TimeSeriesGenerator
    >>> from timeseries_dataset_generator.generators.stationary import generate_ar_dataset
    >>> from timeseries_dataset_generator.utils import plot_single_series, analyze_dataset_summary
    >>> 
    >>> # Generate AR dataset
    >>> generate_ar_dataset(TimeSeriesGenerator, folder='output/ar', count=10)
    >>> 
    >>> # Load and visualize
    >>> import pandas as pd
    >>> df = pd.read_parquet('output/ar.parquet')
    >>> plot_single_series(df, series_id=0, save_path='plot.png')
"""

__version__ = "0.1.0"
__author__ = "TS-Stationary Project Team, TUBITAK 1001 Project No. 124F095"
__license__ = "MIT"

# Import core components
from .core.generator import TimeSeriesGenerator
from .core.metadata import (
    create_metadata_record,
    attach_metadata_columns_to_df,
    get_metadata_columns_defaults,
    make_json_serializable
)

# Import all generator functions
from .generators.stationary import (
    generate_wn_dataset,
    generate_ar_dataset,
    generate_ma_dataset,
    generate_arma_dataset
)

from .generators.trends import (
    generate_linear_trend_dataset,
    generate_quadratic_trend_dataset,
    generate_cubic_trend_dataset,
    generate_exponential_trend_dataset,
    generate_damped_trend_dataset
)

from .generators.stochastic import (
    generate_random_walk_dataset,
    generate_random_walk_with_drift_dataset,
    generate_ima_dataset,
    generate_ari_dataset,
    generate_arima_dataset
)

from .generators.volatility import (
    generate_arch_dataset,
    generate_garch_dataset,
    generate_egarch_dataset,
    generate_aparch_dataset
)

from .generators.seasonality import (
    generate_single_seasonality_dataset,
    generate_multiple_seasonality_dataset,
    generate_sarma_dataset,
    generate_sarima_dataset
)

from .generators.anomalies import (
    generate_point_anomaly_dataset,
    generate_collective_anomaly_dataset,
    generate_contextual_anomaly_dataset
)

from .generators.structural_breaks import (
    generate_mean_shift_dataset,
    generate_variance_shift_dataset,
    generate_trend_shift_dataset
)

# Import utility functions (optional - users can also import from utils directly)
from .utils import (
    save_and_cleanup,
    get_length_label,
    plot_single_series,
    plot_multiple_series,
    plot_series_comparison,
    plot_distribution,
    plot_acf_pacf,
    plot_rolling_statistics,
    plot_category_overview,
    create_dashboard,
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

# Define what's available when using "from timeseries_dataset_generator import *"
__all__ = [
    # Core
    'TimeSeriesGenerator',
    'create_metadata_record',
    'attach_metadata_columns_to_df',
    'get_metadata_columns_defaults',
    'make_json_serializable',
    
    # Stationary
    'generate_wn_dataset',
    'generate_ar_dataset',
    'generate_ma_dataset',
    'generate_arma_dataset',
    
    # Trends
    'generate_linear_trend_dataset',
    'generate_quadratic_trend_dataset',
    'generate_cubic_trend_dataset',
    'generate_exponential_trend_dataset',
    'generate_damped_trend_dataset',
    
    # Stochastic
    'generate_random_walk_dataset',
    'generate_random_walk_with_drift_dataset',
    'generate_ima_dataset',
    'generate_ari_dataset',
    'generate_arima_dataset',
    
    # Volatility
    'generate_arch_dataset',
    'generate_garch_dataset',
    'generate_egarch_dataset',
    'generate_aparch_dataset',
    
    # Seasonality
    'generate_single_seasonality_dataset',
    'generate_multiple_seasonality_dataset',
    'generate_sarma_dataset',
    'generate_sarima_dataset',
    
    # Anomalies
    'generate_point_anomaly_dataset',
    'generate_collective_anomaly_dataset',
    'generate_contextual_anomaly_dataset',
    
    # Structural Breaks
    'generate_mean_shift_dataset',
    'generate_variance_shift_dataset',
    'generate_trend_shift_dataset',
    
    # Utilities - Helpers
    'save_and_cleanup',
    'get_length_label',
    
    # Utilities - Visualization
    'plot_single_series',
    'plot_multiple_series',
    'plot_series_comparison',
    'plot_distribution',
    'plot_acf_pacf',
    'plot_rolling_statistics',
    'plot_category_overview',
    'create_dashboard',
    
    # Utilities - Analysis
    'compute_basic_statistics',
    'test_stationarity',
    'test_normality',
    'detect_seasonality',
    'detect_trend',
    'detect_changepoints',
    'compute_autocorrelation_stats',
    'compute_entropy',
    'analyze_dataset_summary',
    'compare_series',
]

