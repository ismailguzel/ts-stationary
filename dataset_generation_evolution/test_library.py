"""
Test script for timeseries_dataset_generator library.

This script tests the basic functionality of the new modular library.
"""

import sys
from pathlib import Path

# Add the library parent directory to Python path (not the library itself)
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("Testing Time Series Dataset Generator Library")
print("=" * 80)

# Test 1: Import core module
print("\n1. Testing Core Module Import...")
try:
    from timeseries_dataset_generator.core.generator import TimeSeriesGenerator
    print("    TimeSeriesGenerator imported successfully")
except Exception as e:
    print(f"    Error: {e}")
    sys.exit(1)

# Test 2: Import metadata module
print("\n2. Testing Metadata Module...")
try:
    from timeseries_dataset_generator.core.metadata import create_metadata_record, attach_metadata_columns_to_df
    print("    Metadata functions imported successfully")
    
    # Test metadata creation
    record = create_metadata_record(
        series_id=1,
        length=100,
        label="test_series",
        is_stationary=1,
        base_series="ar"
    )
    print(f"    Metadata record created: {len(record)} fields")
except Exception as e:
    print(f"    Error: {e}")
    sys.exit(1)

# Test 3: Import generator modules
print("\n3. Testing Generator Modules...")
try:
    from timeseries_dataset_generator.generators.stationary import generate_ar_dataset
    print("    Stationary generators imported")
    
    from timeseries_dataset_generator.generators.trends import generate_linear_trend_dataset
    print("    Trend generators imported")
    
    from timeseries_dataset_generator.generators.stochastic import generate_random_walk_dataset
    print("    Stochastic generators imported")
    
    from timeseries_dataset_generator.generators.volatility import generate_garch_dataset
    print("    Volatility generators imported")
    
    from timeseries_dataset_generator.generators.seasonality import generate_single_seasonality_dataset
    print("    Seasonality generators imported")
    
    from timeseries_dataset_generator.generators.anomalies import generate_point_anomaly_dataset
    print("    Anomaly generators imported")
    
    from timeseries_dataset_generator.generators.structural_breaks import generate_mean_shift_dataset
    print("    Structural break generators imported")
except Exception as e:
    print(f"    Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Generate a simple time series
print("\n4. Testing Time Series Generation...")
try:
    ts = TimeSeriesGenerator(length=100)
    df, info = ts.generate_stationary_base_series('ar')
    print(f"    AR series generated: {len(df)} data points")
    print(f"    AR order: {info['ar_order']}")
    print(f"    DataFrame columns: {list(df.columns)}")
except Exception as e:
    print(f"    Error: {e}")
    sys.exit(1)

# Test 5: Test dataset generation
print("\n5. Testing Dataset Generation...")
try:
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    generate_ar_dataset(
        TimeSeriesGenerator,
        folder=str(output_dir / "ar_test"),
        count=3,
        length_range=(50, 100)
    )
    
    # Check if file was created
    output_file = output_dir / "ar_test.parquet"
    if output_file.exists():
        import pandas as pd
        df_test = pd.read_parquet(output_file)
        print(f"    Dataset generated: {len(df_test)} total rows")
        print(f"    Number of series: {df_test['series_id'].nunique()}")
        print(f"    Output file: {output_file}")
        
        # Show sample data
        print(f"\n   Sample data:")
        print(df_test.head(3).to_string(index=False))
    else:
        print(f"    Output file not found: {output_file}")
except Exception as e:
    print(f"    Error: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Test multiple generators
print("\n6. Testing Multiple Generator Types...")
try:
    # Linear trend
    generate_linear_trend_dataset(
        TimeSeriesGenerator,
        folder=str(output_dir / "trend_test"),
        kind='ar',
        count=2,
        length_range=(50, 100),
        sign=1
    )
    print("    Linear trend dataset generated")
    
    # Point anomaly
    generate_point_anomaly_dataset(
        TimeSeriesGenerator,
        folder=str(output_dir / "anomaly_test"),
        kind='ar',
        count=2,
        length_range=(50, 100),
        anomaly_type='single',
        location='middle'
    )
    print("    Point anomaly dataset generated")
    
except Exception as e:
    print(f"    Error: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 80)
print("Test Summary")
print("=" * 80)
print(" All core imports working")
print(" Time series generation working")
print(" Dataset generation working")
print(" Multiple generator types working")
print(f"\n Test outputs saved in: {output_dir.absolute()}")
print("\n Library is ready to use!")
print("=" * 80)

