# Visualization and Analysis Guide

This guide provides comprehensive documentation for the visualization and analysis utilities added to the Time Series Dataset Generator library.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Visualization Functions](#visualization-functions)
4. [Analysis Functions](#analysis-functions)
5. [Example Workflows](#example-workflows)
6. [API Reference](#api-reference)

## Overview

The library now includes two powerful utility modules:

- **`timeseries_dataset_generator.utils.visualization`**: Rich plotting functions for time series visualization
- **`timeseries_dataset_generator.utils.analysis`**: Statistical tests and analysis tools

These modules integrate seamlessly with the dataset generation capabilities, allowing you to:
- Generate synthetic time series datasets
- Visualize the generated data with professional plots
- Perform comprehensive statistical analysis
- Export results for further research

## Installation

### Basic Installation

```bash
pip install -r requirements.txt
```

### Required Dependencies

```
# Core dependencies
numpy>=1.20.0
pandas>=1.3.0
statsmodels>=0.13.0
arch>=5.0.0
pyarrow>=6.0.0

# Visualization and analysis
matplotlib>=3.3.0
seaborn>=0.11.0
scipy>=1.7.0
```

## Visualization Functions

### 1. plot_single_series()

Plot an individual time series with optional anomaly and structural break highlighting.

**Parameters:**
- `df`: DataFrame containing the time series data
- `series_id`: ID of the series to plot (optional)
- `title`: Custom title (optional)
- `highlight_anomalies`: Whether to highlight anomalies (default: True)
- `highlight_breaks`: Whether to highlight structural breaks (default: True)
- `figsize`: Figure size tuple (default: (14, 6))
- `save_path`: Path to save the figure (optional)

**Example:**
```python
from timeseries_dataset_generator.utils import plot_single_series
import pandas as pd

df = pd.read_parquet('output/ar.parquet')
plot_single_series(
    df,
    series_id=0,
    highlight_anomalies=True,
    save_path='single_series.png'
)
```

### 2. plot_multiple_series()

Plot multiple time series in a grid layout for easy comparison.

**Parameters:**
- `df`: DataFrame containing the time series data
- `series_ids`: List of series IDs to plot (optional)
- `n_series`: Number of series to plot if series_ids not provided (default: 4)
- `figsize`: Figure size tuple (default: (14, 10))
- `title`: Overall title (optional)
- `save_path`: Path to save the figure (optional)

**Example:**
```python
from timeseries_dataset_generator.utils import plot_multiple_series

plot_multiple_series(
    df,
    n_series=6,
    title='Multiple Time Series Comparison',
    save_path='multiple_series.png'
)
```

### 3. plot_series_comparison()

Compare multiple time series on the same plot.

**Example:**
```python
from timeseries_dataset_generator.utils import plot_series_comparison

plot_series_comparison(
    df,
    series_ids=[0, 1, 2],
    labels=['Series A', 'Series B', 'Series C'],
    save_path='comparison.png'
)
```

### 4. plot_distribution()

Plot the distribution of a time series with histogram and KDE.

**Example:**
```python
from timeseries_dataset_generator.utils import plot_distribution

plot_distribution(
    df,
    series_id=0,
    bins=50,
    save_path='distribution.png'
)
```

### 5. plot_acf_pacf()

Plot Autocorrelation (ACF) and Partial Autocorrelation (PACF) functions.

**Example:**
```python
from timeseries_dataset_generator.utils import plot_acf_pacf

plot_acf_pacf(
    df,
    series_id=0,
    lags=40,
    save_path='acf_pacf.png'
)
```

### 6. plot_rolling_statistics()

Plot rolling mean and standard deviation.

**Example:**
```python
from timeseries_dataset_generator.utils import plot_rolling_statistics

plot_rolling_statistics(
    df,
    series_id=0,
    window=20,
    save_path='rolling_stats.png'
)
```

### 7. plot_category_overview()

Create an overview plot showing examples from different categories.

**Example:**
```python
from timeseries_dataset_generator.utils import plot_category_overview

plot_category_overview(
    df,
    max_series=9,
    save_path='category_overview.png'
)
```

### 8. create_dashboard()

Create a comprehensive dashboard for a single time series.

**Example:**
```python
from timeseries_dataset_generator.utils import create_dashboard

create_dashboard(
    df,
    series_id=0,
    figsize=(16, 12),
    save_path='dashboard.png'
)
```

## Analysis Functions

### 1. compute_basic_statistics()

Compute basic statistical measures for a time series.

**Returns:** Dictionary containing:
- mean, median, std, var
- min, max, range
- q25, q75, iqr (interquartile range)
- skewness, kurtosis
- cv (coefficient of variation)

**Example:**
```python
from timeseries_dataset_generator.utils import compute_basic_statistics

stats = compute_basic_statistics(df, series_id=0)
print(f"Mean: {stats['mean']:.4f}")
print(f"Std Dev: {stats['std']:.4f}")
print(f"Skewness: {stats['skewness']:.4f}")
```

### 2. test_stationarity()

Perform Augmented Dickey-Fuller test for stationarity.

**Returns:** Dictionary containing:
- test_statistic
- p_value
- n_lags
- critical_values
- is_stationary (boolean)
- conclusion (string)

**Example:**
```python
from timeseries_dataset_generator.utils import test_stationarity

result = test_stationarity(df, series_id=0, alpha=0.05)
print(f"Is Stationary: {result['is_stationary']}")
print(f"P-value: {result['p_value']:.4f}")
```

### 3. test_normality()

Perform Shapiro-Wilk test for normality.

**Returns:** Dictionary containing:
- test_statistic
- p_value
- is_normal (boolean)
- conclusion (string)

**Example:**
```python
from timeseries_dataset_generator.utils import test_normality

result = test_normality(df, series_id=0)
print(f"Is Normal: {result['is_normal']}")
```

### 4. detect_seasonality()

Detect seasonality using autocorrelation analysis.

**Returns:** Dictionary containing:
- has_seasonality (boolean)
- dominant_period
- dominant_acf
- all_periods (list)
- all_acf_values (list)

**Example:**
```python
from timeseries_dataset_generator.utils import detect_seasonality

result = detect_seasonality(df, series_id=0)
if result['has_seasonality']:
    print(f"Dominant Period: {result['dominant_period']}")
    print(f"ACF Value: {result['dominant_acf']:.4f}")
```

### 5. detect_trend()

Detect trend using Mann-Kendall test.

**Returns:** Dictionary containing:
- has_trend (boolean)
- trend_direction (string)
- z_score
- p_value
- slope

**Example:**
```python
from timeseries_dataset_generator.utils import detect_trend

result = detect_trend(df, series_id=0)
print(f"Has Trend: {result['has_trend']}")
print(f"Direction: {result['trend_direction']}")
print(f"Slope: {result['slope']:.6f}")
```

### 6. detect_changepoints()

Detect changepoints using moving window approach.

**Returns:** Dictionary containing:
- mean_changepoints (list)
- variance_changepoints (list)
- n_mean_changepoints
- n_variance_changepoints
- total_changepoints

**Example:**
```python
from timeseries_dataset_generator.utils import detect_changepoints

result = detect_changepoints(df, series_id=0, window_size=20)
print(f"Total Changepoints: {result['total_changepoints']}")
print(f"Mean Shifts: {result['n_mean_changepoints']}")
```

### 7. compute_autocorrelation_stats()

Compute autocorrelation statistics including ACF and PACF.

**Example:**
```python
from timeseries_dataset_generator.utils import compute_autocorrelation_stats

result = compute_autocorrelation_stats(df, series_id=0, nlags=40)
print(f"ACF at lag 1: {result['acf_lag1']:.4f}")
print(f"Significant lags (ACF): {result['n_significant_lags_acf']}")
```

### 8. compute_entropy()

Compute Shannon and sample entropy measures.

**Example:**
```python
from timeseries_dataset_generator.utils import compute_entropy

result = compute_entropy(df, series_id=0, bins=50)
print(f"Shannon Entropy: {result['shannon_entropy']:.4f}")
```

### 9. analyze_dataset_summary()

Generate a summary analysis for all series in a dataset.

**Returns:** DataFrame with statistics for each series

**Example:**
```python
from timeseries_dataset_generator.utils import analyze_dataset_summary

summary_df = analyze_dataset_summary(df)
print(summary_df)
summary_df.to_csv('dataset_summary.csv', index=False)
```

### 10. compare_series()

Compare two time series using various distance metrics.

**Returns:** Dictionary containing:
- correlation
- euclidean_distance
- mae (Mean Absolute Error)
- rmse (Root Mean Squared Error)
- dtw_distance (Dynamic Time Warping)

**Example:**
```python
from timeseries_dataset_generator.utils import compare_series

result = compare_series(df, series_id1=0, series_id2=1)
print(f"Correlation: {result['correlation']:.4f}")
print(f"RMSE: {result['rmse']:.4f}")
```

## Example Workflows

### Workflow 1: Generate, Visualize, and Analyze

```python
import pandas as pd
from timeseries_dataset_generator import TimeSeriesGenerator
from timeseries_dataset_generator.generators import generate_ar_dataset
from timeseries_dataset_generator.utils import (
    plot_single_series,
    create_dashboard,
    test_stationarity,
    detect_seasonality,
    analyze_dataset_summary
)

# 1. Generate dataset
generate_ar_dataset(
    TimeSeriesGenerator,
    folder='output/ar',
    count=10,
    length_range=(200, 500)
)

# 2. Load data
df = pd.read_parquet('output/ar.parquet')

# 3. Visualize
plot_single_series(df, series_id=0, save_path='series.png')
create_dashboard(df, series_id=0, save_path='dashboard.png')

# 4. Analyze
stationarity = test_stationarity(df, series_id=0)
seasonality = detect_seasonality(df, series_id=0)

print(f"Stationary: {stationarity['is_stationary']}")
print(f"Seasonal: {seasonality['has_seasonality']}")

# 5. Generate summary report
summary = analyze_dataset_summary(df)
summary.to_csv('analysis_report.csv', index=False)
```

### Workflow 2: Batch Analysis of Multiple Datasets

```python
from pathlib import Path
import pandas as pd
from timeseries_dataset_generator.utils import analyze_dataset_summary

# Analyze all datasets in a directory
data_dir = Path('generated-dataset')
all_summaries = []

for parquet_file in data_dir.rglob('*.parquet'):
    df = pd.read_parquet(parquet_file)
    summary = analyze_dataset_summary(df)
    summary['dataset'] = parquet_file.stem
    all_summaries.append(summary)

# Combine all summaries
combined = pd.concat(all_summaries, ignore_index=True)
combined.to_csv('all_datasets_analysis.csv', index=False)

print(f"Analyzed {len(all_summaries)} datasets")
print(f"Total series: {len(combined)}")
```

### Workflow 3: Visual Report Generation

```python
from pathlib import Path
import pandas as pd
from timeseries_dataset_generator.utils import (
    plot_multiple_series,
    plot_category_overview,
    create_dashboard
)

# Create visual report
df = pd.read_parquet('output/dataset.parquet')
output_dir = Path('visual_report')
output_dir.mkdir(exist_ok=True)

# Overview of all categories
plot_category_overview(
    df,
    max_series=9,
    save_path=output_dir / 'overview.png'
)

# Grid of multiple series
plot_multiple_series(
    df,
    n_series=6,
    save_path=output_dir / 'comparison.png'
)

# Detailed dashboards for first 3 series
for i in range(min(3, df['series_id'].nunique())):
    create_dashboard(
        df,
        series_id=i,
        save_path=output_dir / f'dashboard_{i}.png'
    )

print(f"Visual report saved to: {output_dir.absolute()}")
```

## API Reference

### Visualization Module

```python
from timeseries_dataset_generator.utils.visualization import (
    plot_single_series,
    plot_multiple_series,
    plot_series_comparison,
    plot_distribution,
    plot_acf_pacf,
    plot_rolling_statistics,
    plot_category_overview,
    create_dashboard
)
```

### Analysis Module

```python
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
```

### Convenience Import

```python
# Import directly from utils
from timeseries_dataset_generator.utils import (
    plot_single_series,
    test_stationarity,
    analyze_dataset_summary,
    # ... all other functions
)
```

## Best Practices

1. **Always save plots**: Use the `save_path` parameter to save figures for documentation
2. **Use summary analysis**: For large datasets, use `analyze_dataset_summary()` first
3. **Batch processing**: Process multiple datasets in loops for efficiency
4. **Error handling**: Check for 'error' keys in analysis results
5. **Documentation**: Save analysis results as CSV/JSON for reproducibility

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'statsmodels'`
**Solution**: Install required dependencies:
```bash
pip install -r requirements.txt
```

**Issue**: Plots not displaying
**Solution**: Use `save_path` parameter to save plots to disk

**Issue**: Memory errors with large datasets
**Solution**: Process series individually or in batches

## Contributing

To add new visualization or analysis functions:

1. Add function to appropriate module (`visualization.py` or `analysis.py`)
2. Update `__init__.py` in utils folder
3. Add documentation and examples
4. Test with various time series types

## Support

For questions or issues:
- Open an issue on GitHub
- Check the documentation in `docs/` folder
- Review example scripts in `examples/` folder

