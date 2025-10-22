"""
Simple Time Series Generation Example
Using the new timeseries_dataset_generator library
"""

import sys
from pathlib import Path
import numpy as np
import random

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Add library to path
sys.path.insert(0, str(Path(__file__).parent))

from timeseries_dataset_generator import TimeSeriesGenerator
from timeseries_dataset_generator.generators import (
    generate_ar_dataset,
    generate_linear_trend_dataset,
    generate_point_anomaly_dataset,
    generate_garch_dataset,
    generate_single_seasonality_dataset
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
    folder=str(output_dir / "anomalies"),
    kind='arma',
    count=5,
    length_range=(150, 250),
    anomaly_type='single',
    location='middle'
)
print(f"   Generated 5 ARMA series with point anomaly in middle")
print(f"   Output: {output_dir / 'anomalies.parquet'}")

# 4. Generate GARCH Volatility Dataset
print("\n4. Generating GARCH Volatility Dataset...")
generate_garch_dataset(
    TimeSeriesGenerator,
    folder=str(output_dir / "garch"),
    count=5,
    length_range=(200, 300)
)
print(f"   Generated 5 GARCH series with volatility clustering")
print(f"   Output: {output_dir / 'garch.parquet'}")

# 5. Generate Seasonal Dataset
print("\n5. Generating Seasonal Dataset...")
generate_single_seasonality_dataset(
    TimeSeriesGenerator,
    folder=str(output_dir / "seasonal"),
    count=5,
    length_range=(150, 250)
)
print(f"   Generated 5 series with seasonality")
print(f"   Output: {output_dir / 'seasonal.parquet'}")

# Summary
print("\n" + "=" * 80)
print("Generation Summary")
print("=" * 80)
print(f"Total datasets generated: 5")
print(f"Total series: 25")
print(f"Output directory: {output_dir.absolute()}")
print("\nAll datasets generated successfully!")
print("=" * 80)

# Display one dataset
print("\nSample from AR dataset:")
import pandas as pd
df_sample = pd.read_parquet(output_dir / 'ar_short.parquet')
print(f"\nDataFrame shape: {df_sample.shape}")
print(f"Columns: {list(df_sample.columns)}")
print(f"\nFirst 5 rows:")
print(df_sample.head())
print(f"\nNumber of unique series: {df_sample['series_id'].nunique()}")

