"""
Example script for generating various time series datasets.

This script demonstrates how to use the timeseries_dataset_generator package
to create datasets with different characteristics.
"""

from pathlib import Path
import sys

# Add parent directory to path to import the package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from timeseries_dataset_generator import TimeSeriesGenerator
from timeseries_dataset_generator.generators import (
    generate_ar_dataset,
    generate_ma_dataset,
    generate_arma_dataset,
    generate_wn_dataset,
    generate_linear_trend_dataset,
    generate_quadratic_trend_dataset,
    generate_random_walk_dataset,
    generate_arima_dataset,
    generate_garch_dataset,
    generate_single_seasonality_dataset,
    generate_point_anomaly_dataset,
    generate_collective_anomaly_dataset,
    generate_mean_shift_dataset,
)


def ensure_base_dir(base_dir: Path) -> Path:
    """Create base directory if it doesn't exist."""
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def folder_path(base_dir: Path, *parts: str) -> str:
    """Create folder path and ensure it exists."""
    path = ensure_base_dir(base_dir.joinpath(*parts))
    return str(path)


def main():
    """Generate example datasets."""
    
    # Base output directory
    BASE_OUTPUT_DIR = ensure_base_dir(Path("example-output"))
    
    # Configuration
    bases = ["ar", "ma", "arma", "white_noise"]
    length_ranges = [(50, 100), (300, 500)]
    
    print("=" * 80)
    print("Time Series Dataset Generator - Example")
    print("=" * 80)
    
    # 1. Stationary Series
    print("\n1. Generating Stationary Series...")
    for base in bases:
        for length_range in length_ranges:
            l = "short" if length_range == (50, 100) else "medium"
            
            if base == "ar":
                generate_ar_dataset(
                    TimeSeriesGenerator,
                    folder=folder_path(BASE_OUTPUT_DIR, "stationary", base, l),
                    count=5,
                    length_range=length_range
                )
            elif base == "ma":
                generate_ma_dataset(
                    TimeSeriesGenerator,
                    folder=folder_path(BASE_OUTPUT_DIR, "stationary", base, l),
                    count=5,
                    length_range=length_range
                )
            elif base == "arma":
                generate_arma_dataset(
                    TimeSeriesGenerator,
                    folder=folder_path(BASE_OUTPUT_DIR, "stationary", base, l),
                    count=5,
                    length_range=length_range
                )
            elif base == "white_noise":
                generate_wn_dataset(
                    TimeSeriesGenerator,
                    folder=folder_path(BASE_OUTPUT_DIR, "stationary", base, l),
                    count=5,
                    length_range=length_range
                )
    
    # 2. Linear Trends
    print("\n2. Generating Linear Trend Series...")
    for sign in [1, -1]:
        direction = "up" if sign == 1 else "down"
        generate_linear_trend_dataset(
            TimeSeriesGenerator,
            folder=folder_path(BASE_OUTPUT_DIR, "linear_trend", direction, "ar", "medium"),
            kind="ar",
            count=5,
            length_range=(300, 500),
            sign=sign
        )
    
    # 3. Quadratic Trends
    print("\n3. Generating Quadratic Trend Series...")
    generate_quadratic_trend_dataset(
        TimeSeriesGenerator,
        folder=folder_path(BASE_OUTPUT_DIR, "quadratic_trend", "arma", "medium"),
        kind="arma",
        count=5,
        length_range=(300, 500),
        sign=1
    )
    
    # 4. Stochastic Trends
    print("\n4. Generating Stochastic Trend Series...")
    generate_random_walk_dataset(
        TimeSeriesGenerator,
        folder=folder_path(BASE_OUTPUT_DIR, "stochastic", "random_walk", "medium"),
        count=5,
        length_range=(300, 500)
    )
    
    generate_arima_dataset(
        TimeSeriesGenerator,
        folder=folder_path(BASE_OUTPUT_DIR, "stochastic", "arima", "medium"),
        count=5,
        length_range=(300, 500)
    )
    
    # 5. Volatility
    print("\n5. Generating Volatility Series...")
    generate_garch_dataset(
        TimeSeriesGenerator,
        folder=folder_path(BASE_OUTPUT_DIR, "volatility", "garch", "medium"),
        count=5,
        length_range=(300, 500)
    )
    
    # 6. Seasonality
    print("\n6. Generating Seasonal Series...")
    generate_single_seasonality_dataset(
        TimeSeriesGenerator,
        folder=folder_path(BASE_OUTPUT_DIR, "seasonality", "single", "medium"),
        count=5,
        length_range=(300, 500)
    )
    
    # 7. Anomalies
    print("\n7. Generating Series with Anomalies...")
    generate_point_anomaly_dataset(
        TimeSeriesGenerator,
        folder=folder_path(BASE_OUTPUT_DIR, "anomalies", "point", "ar", "medium"),
        kind="ar",
        count=5,
        length_range=(300, 500),
        anomaly_type='single',
        location='middle'
    )
    
    generate_collective_anomaly_dataset(
        TimeSeriesGenerator,
        folder=folder_path(BASE_OUTPUT_DIR, "anomalies", "collective", "ma", "medium"),
        kind="ma",
        count=5,
        length_range=(300, 500),
        anomaly_type='single',
        location='middle'
    )
    
    # 8. Structural Breaks
    print("\n8. Generating Series with Structural Breaks...")
    generate_mean_shift_dataset(
        TimeSeriesGenerator,
        folder=folder_path(BASE_OUTPUT_DIR, "structural_breaks", "mean_shift", "arma", "medium"),
        kind="arma",
        count=5,
        length_range=(300, 500),
        break_type='single',
        signs=[1],
        location='middle'
    )
    
    print("\n" + "=" * 80)
    print("Dataset generation complete!")
    print(f"Output directory: {BASE_OUTPUT_DIR.absolute()}")
    print("=" * 80)


if __name__ == "__main__":
    main()

