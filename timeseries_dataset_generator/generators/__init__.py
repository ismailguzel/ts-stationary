"""
Dataset generators for various time series patterns.

This module contains specialized generators for different types of time series:
- Stationary processes
- Deterministic trends
- Stochastic trends
- Volatility clustering
- Seasonality
- Anomalies
- Structural breaks
"""

from .stationary import (
    generate_wn_dataset,
    generate_ar_dataset,
    generate_ma_dataset,
    generate_arma_dataset
)

from .trends import (
    generate_linear_trend_dataset,
    generate_quadratic_trend_dataset,
    generate_cubic_trend_dataset,
    generate_exponential_trend_dataset,
    generate_damped_trend_dataset
)

from .stochastic import (
    generate_random_walk_dataset,
    generate_random_walk_with_drift_dataset,
    generate_ima_dataset,
    generate_ari_dataset,
    generate_arima_dataset
)

from .volatility import (
    generate_arch_dataset,
    generate_garch_dataset,
    generate_egarch_dataset,
    generate_aparch_dataset
)

from .seasonality import (
    generate_single_seasonality_dataset,
    generate_multiple_seasonality_dataset,
    generate_sarma_dataset,
    generate_sarima_dataset
)

from .anomalies import (
    generate_point_anomaly_dataset,
    generate_collective_anomaly_dataset,
    generate_contextual_anomaly_dataset
)

from .structural_breaks import (
    generate_mean_shift_dataset,
    generate_variance_shift_dataset,
    generate_trend_shift_dataset
)

__all__ = [
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
]

