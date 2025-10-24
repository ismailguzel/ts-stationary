"""
Example: Analyzing Generated Time Series Datasets

This script demonstrates how to:
1. Load generated datasets
2. Compute basic statistics
3. Test for stationarity
4. Detect trends and seasonality
5. Identify changepoints
6. Compare multiple series
7. Generate comprehensive analysis reports
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Add library to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from timeseries_dataset_generator.utils.analysis import (
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

print("=" * 80)
print("Time Series Dataset Analysis Example")
print("=" * 80)

# ============================================================================
# 1. LOAD DATASET
# ============================================================================
print("\n[1] Loading Dataset...")
print("-" * 80)

# Check if simple_output directory exists
data_dir = Path("simple_output")
if not data_dir.exists():
    print(f"Error: {data_dir} not found!")
    print("Please run 'python examples/simple_generation.py' first to generate sample data.")
    sys.exit(1)

# List available parquet files
parquet_files = list(data_dir.glob("*.parquet"))
if not parquet_files:
    print(f"Error: No parquet files found in {data_dir}")
    sys.exit(1)

print(f"Found {len(parquet_files)} parquet file(s):")
for f in parquet_files:
    file_size = f.stat().st_size / 1024  # KB
    print(f"  - {f.name} ({file_size:.2f} KB)")

# Load the first dataset
example_file = parquet_files[0]
print(f"\nLoading: {example_file.name}")
df = pd.read_parquet(example_file)

print(f"\nDataset loaded successfully!")
print(f"  Shape: {df.shape}")
print(f"  Number of series: {df['series_id'].nunique()}")
print(f"  Series lengths: {df.groupby('series_id').size().describe()}")

# ============================================================================
# 2. BASIC STATISTICS
# ============================================================================
print("\n[2] Computing Basic Statistics...")
print("-" * 80)

first_series_id = df['series_id'].iloc[0]
print(f"\nAnalyzing series ID: {first_series_id}")

basic_stats = compute_basic_statistics(df, first_series_id)

print("\nBasic Statistics:")
print("-" * 40)
for key, value in basic_stats.items():
    if isinstance(value, float):
        print(f"  {key:20s}: {value:12.4f}")
    else:
        print(f"  {key:20s}: {value}")

# ============================================================================
# 3. STATIONARITY TEST
# ============================================================================
print("\n[3] Testing for Stationarity...")
print("-" * 80)

stationarity_result = test_stationarity(df, first_series_id, alpha=0.05)

if 'error' in stationarity_result:
    print(f"  ⚠ {stationarity_result['error']}")
else:
    print("\nAugmented Dickey-Fuller Test:")
    print("-" * 40)
    print(f"  Test Statistic     : {stationarity_result['test_statistic']:.4f}")
    print(f"  P-value            : {stationarity_result['p_value']:.4f}")
    print(f"  Number of lags     : {stationarity_result['n_lags']}")
    print(f"  Number of obs      : {stationarity_result['n_obs']}")
    print(f"\n  Critical Values:")
    for key, value in stationarity_result['critical_values'].items():
        print(f"    {key:5s}: {value:.4f}")
    print(f"\n  Conclusion         : {stationarity_result['conclusion']}")
    print(f"  Is Stationary      : {stationarity_result['is_stationary']}")

# ============================================================================
# 4. NORMALITY TEST
# ============================================================================
print("\n[4] Testing for Normality...")
print("-" * 80)

normality_result = test_normality(df, first_series_id, alpha=0.05)

if 'error' in normality_result:
    print(f"  ⚠ {normality_result['error']}")
else:
    print("\nShapiro-Wilk Test:")
    print("-" * 40)
    print(f"  Test Statistic     : {normality_result['test_statistic']:.4f}")
    print(f"  P-value            : {normality_result['p_value']:.4f}")
    print(f"  Conclusion         : {normality_result['conclusion']}")
    print(f"  Is Normal          : {normality_result['is_normal']}")

# ============================================================================
# 5. TREND DETECTION
# ============================================================================
print("\n[5] Detecting Trends...")
print("-" * 80)

trend_result = detect_trend(df, first_series_id, alpha=0.05)

print("\nMann-Kendall Trend Test:")
print("-" * 40)
print(f"  Z-score            : {trend_result['z_score']:.4f}")
print(f"  P-value            : {trend_result['p_value']:.4f}")
print(f"  Slope              : {trend_result['slope']:.6f}")
print(f"  Has Trend          : {trend_result['has_trend']}")
print(f"  Trend Direction    : {trend_result['trend_direction']}")

# ============================================================================
# 6. SEASONALITY DETECTION
# ============================================================================
print("\n[6] Detecting Seasonality...")
print("-" * 80)

seasonality_result = detect_seasonality(df, first_series_id)

if 'error' in seasonality_result:
    print(f"  ⚠ {seasonality_result['error']}")
else:
    print("\nSeasonality Detection (ACF-based):")
    print("-" * 40)
    print(f"  Has Seasonality    : {seasonality_result['has_seasonality']}")
    
    if seasonality_result['has_seasonality']:
        print(f"  Dominant Period    : {seasonality_result['dominant_period']}")
        print(f"  Dominant ACF       : {seasonality_result['dominant_acf']:.4f}")
        
        if len(seasonality_result['all_periods']) > 1:
            print(f"\n  Additional Periods:")
            for period, acf in zip(seasonality_result['all_periods'][1:4], 
                                   seasonality_result['all_acf_values'][1:4]):
                print(f"    Period {period:3d}: ACF = {acf:.4f}")
    
    print(f"\n  Max ACF            : {seasonality_result['max_acf']:.4f}")
    print(f"  Mean ACF           : {seasonality_result['mean_acf']:.4f}")

# ============================================================================
# 7. CHANGEPOINT DETECTION
# ============================================================================
print("\n[7] Detecting Changepoints...")
print("-" * 80)

series_length = len(df[df['series_id'] == first_series_id])
window_size = max(10, series_length // 20)

changepoint_result = detect_changepoints(df, first_series_id, 
                                        window_size=window_size, 
                                        threshold=2.0)

print(f"\nChangepoint Detection (window={window_size}, threshold=2.0):")
print("-" * 40)
print(f"  Mean Changepoints     : {changepoint_result['n_mean_changepoints']}")
print(f"  Variance Changepoints : {changepoint_result['n_variance_changepoints']}")
print(f"  Total Changepoints    : {changepoint_result['total_changepoints']}")

if changepoint_result['n_mean_changepoints'] > 0:
    print(f"\n  Mean Shift Locations:")
    for idx, loc in enumerate(changepoint_result['mean_changepoints'][:5], 1):
        print(f"    {idx}. Time step {loc}")
    if len(changepoint_result['mean_changepoints']) > 5:
        print(f"    ... and {len(changepoint_result['mean_changepoints']) - 5} more")

if changepoint_result['n_variance_changepoints'] > 0:
    print(f"\n  Variance Shift Locations:")
    for idx, loc in enumerate(changepoint_result['variance_changepoints'][:5], 1):
        print(f"    {idx}. Time step {loc}")
    if len(changepoint_result['variance_changepoints']) > 5:
        print(f"    ... and {len(changepoint_result['variance_changepoints']) - 5} more")

# ============================================================================
# 8. AUTOCORRELATION ANALYSIS
# ============================================================================
print("\n[8] Analyzing Autocorrelation...")
print("-" * 80)

acf_result = compute_autocorrelation_stats(df, first_series_id, nlags=min(40, series_length // 2 - 1))

if 'error' in acf_result:
    print(f"  ⚠ {acf_result['error']}")
else:
    print("\nAutocorrelation Statistics:")
    print("-" * 40)
    print(f"  ACF at lag 1       : {acf_result['acf_lag1']:.4f}")
    print(f"  ACF sum (abs)      : {acf_result['acf_sum']:.4f}")
    print(f"  ACF mean (abs)     : {acf_result['acf_mean']:.4f}")
    print(f"  Significant lags   :")
    print(f"    ACF              : {acf_result['n_significant_lags_acf']}")
    print(f"    PACF             : {acf_result['n_significant_lags_pacf']}")

# ============================================================================
# 9. ENTROPY MEASURES
# ============================================================================
print("\n[9] Computing Entropy Measures...")
print("-" * 80)

entropy_result = compute_entropy(df, first_series_id, bins=50)

print("\nEntropy Measures:")
print("-" * 40)
print(f"  Shannon Entropy    : {entropy_result['shannon_entropy']:.4f}")
if not np.isnan(entropy_result['sample_entropy']):
    print(f"  Sample Entropy     : {entropy_result['sample_entropy']:.4f}")
else:
    print(f"  Sample Entropy     : N/A")

# ============================================================================
# 10. COMPARE MULTIPLE SERIES
# ============================================================================
print("\n[10] Comparing Multiple Series...")
print("-" * 80)

if df['series_id'].nunique() >= 2:
    series_ids = df['series_id'].unique()[:2]
    print(f"\nComparing Series {series_ids[0]} and {series_ids[1]}:")
    
    comparison_result = compare_series(df, series_ids[0], series_ids[1])
    
    print("-" * 40)
    print(f"  Correlation        : {comparison_result['correlation']:.4f}")
    print(f"  Euclidean Distance : {comparison_result['euclidean_distance']:.4f}")
    print(f"  MAE                : {comparison_result['mae']:.4f}")
    print(f"  RMSE               : {comparison_result['rmse']:.4f}")
    print(f"  DTW Distance       : {comparison_result['dtw_distance']:.4f}")
else:
    print("\n  ⚠ Need at least 2 series for comparison")

# ============================================================================
# 11. DATASET SUMMARY
# ============================================================================
print("\n[11] Generating Dataset Summary...")
print("-" * 80)

print("\nComputing summary for all series in dataset...")
summary_df = analyze_dataset_summary(df)

print(f"\nSummary DataFrame shape: {summary_df.shape}")
print("\nFirst few rows of summary:")
print(summary_df.head().to_string())

# ============================================================================
# 12. SAVE ANALYSIS RESULTS
# ============================================================================
print("\n[12] Saving Analysis Results...")
print("-" * 80)

# Create output directory
output_dir = Path("analysis_output")
output_dir.mkdir(exist_ok=True)

# Save summary as CSV
summary_path = output_dir / f"{example_file.stem}_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\n  ✓ Saved summary to: {summary_path.name}")

# Save detailed analysis as JSON
analysis_report = {
    'dataset_name': example_file.name,
    'series_id': int(first_series_id),
    'basic_statistics': basic_stats,
    'stationarity_test': stationarity_result if 'error' not in stationarity_result else None,
    'normality_test': normality_result if 'error' not in normality_result else None,
    'trend_detection': trend_result,
    'seasonality_detection': seasonality_result if 'error' not in seasonality_result else None,
    'changepoint_detection': {
        'n_mean_changepoints': changepoint_result['n_mean_changepoints'],
        'n_variance_changepoints': changepoint_result['n_variance_changepoints'],
        'total_changepoints': changepoint_result['total_changepoints']
    },
    'autocorrelation_stats': {
        'acf_lag1': acf_result.get('acf_lag1'),
        'acf_sum': acf_result.get('acf_sum'),
        'n_significant_lags_acf': acf_result.get('n_significant_lags_acf'),
        'n_significant_lags_pacf': acf_result.get('n_significant_lags_pacf')
    } if 'error' not in acf_result else None,
    'entropy_measures': entropy_result
}

report_path = output_dir / f"{example_file.stem}_analysis.json"

# Convert numpy/pandas types to JSON-serializable types
def convert_to_json_serializable(obj):
    """Convert numpy/pandas types to JSON-serializable types."""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif obj is None or isinstance(obj, (str, int, float)):
        return obj
    return obj

analysis_report_json = convert_to_json_serializable(analysis_report)

with open(report_path, 'w') as f:
    json.dump(analysis_report_json, f, indent=2)
print(f"  ✓ Saved detailed analysis to: {report_path.name}")

# ============================================================================
# 13. ANALYZE MULTIPLE DATASETS (if available)
# ============================================================================
print("\n[13] Multi-Dataset Analysis...")
print("-" * 80)

if len(parquet_files) > 1:
    print(f"\nAnalyzing {len(parquet_files)} datasets...")
    
    all_summaries = []
    
    for pq_file in parquet_files:
        temp_df = pd.read_parquet(pq_file)
        temp_summary = analyze_dataset_summary(temp_df)
        temp_summary['dataset'] = pq_file.stem
        all_summaries.append(temp_summary)
        print(f"  ✓ Analyzed: {pq_file.name}")
    
    # Combine all summaries
    combined_summary = pd.concat(all_summaries, ignore_index=True)
    
    # Save combined summary
    combined_path = output_dir / "all_datasets_summary.csv"
    combined_summary.to_csv(combined_path, index=False)
    print(f"\n  ✓ Saved combined summary to: {combined_path.name}")
    
    # Print aggregate statistics
    print("\n  Aggregate Statistics Across All Datasets:")
    print("  " + "-" * 40)
    
    numeric_cols = combined_summary.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"    Mean series length : {combined_summary['length'].mean():.0f}")
        print(f"    Total series       : {len(combined_summary)}")
        
        if 'is_stationary' in combined_summary.columns:
            stationary_pct = combined_summary['is_stationary'].mean() * 100
            print(f"    Stationary series  : {stationary_pct:.1f}%")
        
        if 'has_trend' in combined_summary.columns:
            trend_pct = combined_summary['has_trend'].mean() * 100
            print(f"    Series with trend  : {trend_pct:.1f}%")
        
        if 'has_seasonality' in combined_summary.columns:
            seasonal_pct = combined_summary['has_seasonality'].mean() * 100
            print(f"    Seasonal series    : {seasonal_pct:.1f}%")

else:
    print("\n  Only one dataset available. Load more datasets for multi-dataset analysis.")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("Analysis Summary")
print("=" * 80)
print(f"\nAll analysis results saved to: {output_dir.absolute()}")
print("\nGenerated files:")
output_files = list(output_dir.glob("*"))
for idx, out_file in enumerate(output_files, 1):
    file_size = out_file.stat().st_size / 1024  # KB
    print(f"  {idx}. {out_file.name} ({file_size:.2f} KB)")

print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)
print("\nKey Findings:")
print(f"  • Series ID {first_series_id}:")
print(f"    - Stationarity: {stationarity_result.get('conclusion', 'N/A')}")
print(f"    - Trend: {trend_result['trend_direction']}")
print(f"    - Seasonality: {'Yes' if seasonality_result.get('has_seasonality', False) else 'No'}")
print(f"    - Changepoints: {changepoint_result['total_changepoints']}")

print("\nNext steps:")
print("  1. Review the analysis_output folder for detailed results")
print("  2. Run 'python examples/visualize_datasets.py' for visualizations")
print("  3. Modify this script to analyze your own datasets")
print("=" * 80)

