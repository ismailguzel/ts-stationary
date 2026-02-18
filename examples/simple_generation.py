"""
Simple Time Series Generation Example
Using the new timeseries_dataset_generator library
"""

import sys
from pathlib import Path
import numpy as np
import random
import pandas as pd

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Add library to path
sys.path.insert(0, str(Path(__file__).parent))

from timeseries_dataset_generator.core.generator import TimeSeriesGenerator
from timeseries_dataset_generator.generators import (
    generate_ar_dataset,
    generate_linear_trend_dataset,
    generate_point_anomaly_dataset,
    generate_collective_anomaly_dataset,
    generate_contextual_anomaly_dataset,
    generate_garch_dataset,
    generate_single_seasonality_dataset,
    generate_mean_shift_dataset,
    generate_trend_shift_dataset,
    generate_multiple_seasonality_dataset,
    generate_cubic_trend_dataset,
    generate_variance_shift_dataset,
    generate_arima_dataset
)

print("=" * 80)
print("Time Series Generation with New Library")
print("=" * 80)

# Output directory
output_dir = Path("simple_output")
output_dir.mkdir(exist_ok=True)

# 1. Generate AR Dataset
print("\n1. Generating AR (Autoregressive) Dataset...")
generate_ar_dataset(
    TimeSeriesGenerator,
    folder=str(output_dir / "ar_short"),
    count=5,
    length_range=(50, 100)
)
print(f"   Generated 5 AR series (50-100 length)")
print(f"   Output: {output_dir / 'ar_short.parquet'}")

# 2. Generate Linear Trend Dataset
print("\n2. Generating Linear Trend Dataset...")
generate_linear_trend_dataset(
    TimeSeriesGenerator,
    folder=str(output_dir / "linear_trend_up"),
    kind='ar',
    count=5,
    length_range=(100, 200),
    sign=1  # upward trend
)
print(f"   Generated 5 AR series with upward linear trend")
print(f"   Output: {output_dir / 'linear_trend_up.parquet'}")

# 3. Generate Point Anomaly Dataset
print("\n3. Generating Point Anomaly Dataset...")
generate_point_anomaly_dataset(
    TimeSeriesGenerator,
    folder=str(output_dir / "point_anomalies"),
    kind='arma',
    count=5,
    length_range=(300, 500),
    anomaly_type='multiple',
    num_anomalies=5
)
print(f"   Generated 5 ARMA series with point anomalies")
print(f"   Output: {output_dir / 'point_anomalies.parquet'}")

# 4. Generate GARCH Volatility Dataset
print("\n4. Generating GARCH Volatility Dataset...")
generate_garch_dataset(
    TimeSeriesGenerator,
    folder=str(output_dir / "garch"),
    count=5,
    length_range=(300, 500)
)
print(f"   Generated 5 GARCH series with volatility clustering")
print(f"   Output: {output_dir / 'garch.parquet'}")

# 5. Generate Single Seasonal Dataset
print("\n5. Generating Single Seasonal Dataset...")
generate_single_seasonality_dataset(
    TimeSeriesGenerator,
    folder=str(output_dir / "single_seasonal"),
    count=5,
    length_range=(300, 500)
)
print(f"   Generated 5 series with single seasonality")
print(f"   Output: {output_dir / 'single_seasonal.parquet'}")

# 6. Generate Mean Shift Dataset
print("\n6. Generating Mean Shift Dataset...")
generate_mean_shift_dataset(
    TimeSeriesGenerator,
    folder=str(output_dir / "mean_shift"),
    count=15,
    num_breaks=1,
    break_type= 'single',
    location= 'middle',
    length_range=(300, 500),
    is_loc=False
)
print(f"   Generated 5 series with mean shifts")
print(f"   Output: {output_dir / 'mean_shift.parquet'}")

# 7. Generate Collective Anomaly Dataset
print("\n7. Generating Collective Anomaly Dataset...")
generate_collective_anomaly_dataset(
    TimeSeriesGenerator,
    folder=str(output_dir / "collective_anomalies"),
    kind='arma',
    count=10,
    length_range=(300, 500),
    anomaly_type='multiple',
    num_anomalies=2,
)
print(f"   Generated 5 ARMA series with collective anomaly")
print(f"   Output: {output_dir / 'collective_anomalies.parquet'}")

# 8. Generate Cubic Trend Dataset
print("\n8. Generating Cubic Trend Dataset...")
generate_cubic_trend_dataset(
    TimeSeriesGenerator,
    folder=str(output_dir / "cubic_trend"),
    count=15,
    length_range=(300, 500),
    sign=1  # upward trend
)
print(f"   Generated 5 AR series with upward cubic trend")
print(f"   Output: {output_dir / 'cubic_trend.parquet'}")

# 9. Generate Trend Shift Dataset
print("\n9. Generating Trend Shift Dataset...")
generate_trend_shift_dataset(
    TimeSeriesGenerator,
    folder=str(output_dir / "trend_shift"),
    count=15,
    change_types = ['direction_and_magnitude_change',],
    length_range=(300, 500)
)

print(f"   Generated 5 series with trend shifts")
print(f"   Output: {output_dir / 'trend_shift.parquet'}")

# 11. Generate Variance Shift Dataset
print("\n11. Generating Variance Shift Dataset...")
generate_variance_shift_dataset(
    TimeSeriesGenerator,
    folder=str(output_dir / "variance_shift"),
    count=15,
    num_breaks=2,
    break_type= 'multiple',
    length_range=(300, 500)
)
print(f"   Generated 5 series with variance shifts")
print(f"   Output: {output_dir / 'variance_shift.parquet'}")

# 10. Generate Multiple Seasonal Dataset
print("\n10. Generating Multiple Seasonal Dataset...")
generate_multiple_seasonality_dataset(
    TimeSeriesGenerator,
    folder=str(output_dir / "multiple_seasonal"),
    count=15,
    length_range=(300, 500)
)
print(f"   Generated 5 series with multiple seasonality")
print(f"   Output: {output_dir / 'multiple_seasonal.parquet'}")

# 11. Generate ARIMA Dataset
print("\n11. Generating ARIMA (Autoregressive) Dataset...")
generate_arima_dataset(
    TimeSeriesGenerator,
    folder=str(output_dir / "arima"),
    count=5,
    length_range=(300, 500)
)
print(f"   Generated 5 ARIMA series (300-500 length)")
print(f"   Output: {output_dir / 'arima.parquet'}")

# 12. Generate Contextual Anomaly Dataset
print("\n12. Generating Contextual Anomaly Dataset...")
generate_contextual_anomaly_dataset(
    TimeSeriesGenerator,
    folder=str(output_dir / "contextual_anomalies"),
    count=10,
    length_range=(300, 500),
    anomaly_type='multiple',
    num_anomalies=2,
)
print(f"   Generated 5 ARMA series with contextual anomaly")
print(f"   Output: {output_dir / 'contextual_anomalies.parquet'}")

# Summary
print("\n" + "=" * 80)
print("Generation Summary")
print("=" * 80)
print(f"Total datasets generated: 6")
print(f"Total series: 30")
print(f"Output directory: {output_dir.absolute()}")
print("\nAll datasets generated successfully!")
print("=" * 80)

print("\nSample from mean shift dataset:")
df_sample = pd.read_parquet(output_dir / 'mean_shift.parquet')
print(f"\nDataFrame shape: {df_sample.shape}")
print(f"Columns: {list(df_sample.columns)}")
print(f"\nFirst 5 rows:")
print(df_sample.head())
print(f"\nNumber of unique series: {df_sample['series_id'].nunique()}")




