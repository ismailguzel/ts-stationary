# Quick Start Guide - Visualization & Analysis

## 🚀 3 Steps to Get Started

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Generate Sample Data

```bash
python examples/simple_generation.py
```

### Step 3: Visualize & Analyze

```bash
# Create visualizations
python examples/visualize_datasets.py

# Run statistical analysis
python examples/analyze_datasets.py
```

## 📊 Common Tasks

### Task 1: Create a Dashboard

```python
import pandas as pd
from timeseries_dataset_generator.utils import create_dashboard

df = pd.read_parquet('simple_output/ar_short.parquet')
create_dashboard(df, series_id=0, save_path='my_dashboard.png')
```

### Task 2: Test for Stationarity

```python
from timeseries_dataset_generator.utils import test_stationarity

result = test_stationarity(df, series_id=0)
print(f"Is stationary: {result['is_stationary']}")
print(f"P-value: {result['p_value']:.4f}")
```

### Task 3: Detect Seasonality

```python
from timeseries_dataset_generator.utils import detect_seasonality

result = detect_seasonality(df, series_id=0)
if result['has_seasonality']:
    print(f"Period: {result['dominant_period']}")
```

### Task 4: Analyze Entire Dataset

```python
from timeseries_dataset_generator.utils import analyze_dataset_summary

summary = analyze_dataset_summary(df)
summary.to_csv('dataset_analysis.csv', index=False)
print(summary)
```

### Task 5: Compare Two Series

```python
from timeseries_dataset_generator.utils import compare_series

comparison = compare_series(df, series_id1=0, series_id2=1)
print(f"Correlation: {comparison['correlation']:.4f}")
print(f"RMSE: {comparison['rmse']:.4f}")
```

### Task 6: Plot Multiple Series

```python
from timeseries_dataset_generator.utils import plot_multiple_series

plot_multiple_series(df, n_series=4, save_path='grid.png')
```

### Task 7: Check for Trends

```python
from timeseries_dataset_generator.utils import detect_trend

result = detect_trend(df, series_id=0)
print(f"Has trend: {result['has_trend']}")
print(f"Direction: {result['trend_direction']}")
```

## 📖 All Available Functions

### Visualization (8 functions)
- `plot_single_series()` - Single series plot
- `plot_multiple_series()` - Grid layout
- `plot_series_comparison()` - Overlay plot
- `plot_distribution()` - Histogram + KDE
- `plot_acf_pacf()` - Autocorrelation plots
- `plot_rolling_statistics()` - Rolling stats
- `plot_category_overview()` - Category overview
- `create_dashboard()` - Comprehensive dashboard

### Analysis (10 functions)
- `compute_basic_statistics()` - Basic stats
- `test_stationarity()` - ADF test
- `test_normality()` - Shapiro-Wilk test
- `detect_seasonality()` - Seasonality detection
- `detect_trend()` - Mann-Kendall test
- `detect_changepoints()` - Changepoint detection
- `compute_autocorrelation_stats()` - ACF/PACF stats
- `compute_entropy()` - Entropy measures
- `analyze_dataset_summary()` - Batch analysis
- `compare_series()` - Series comparison

## 🎯 Import Patterns

### Pattern 1: Import from utils

```python
from timeseries_dataset_generator.utils import (
    plot_single_series,
    test_stationarity,
    analyze_dataset_summary
)
```

### Pattern 2: Import modules

```python
from timeseries_dataset_generator.utils import visualization, analysis

visualization.create_dashboard(df, series_id=0)
analysis.test_stationarity(df, series_id=0)
```

### Pattern 3: Import main package

```python
from timeseries_dataset_generator import (
    TimeSeriesGenerator,
    plot_single_series,
    test_stationarity
)
```

## 🔥 Power User Tips

### Tip 1: Batch Processing

```python
from pathlib import Path
import pandas as pd
from timeseries_dataset_generator.utils import analyze_dataset_summary

# Analyze all datasets
for file in Path('output').glob('*.parquet'):
    df = pd.read_parquet(file)
    summary = analyze_dataset_summary(df)
    summary.to_csv(f'{file.stem}_analysis.csv')
```

### Tip 2: Custom Analysis Pipeline

```python
def analyze_series(df, series_id):
    """Complete analysis of a single series"""
    from timeseries_dataset_generator.utils import (
        compute_basic_statistics,
        test_stationarity,
        detect_seasonality,
        detect_trend
    )
    
    return {
        'stats': compute_basic_statistics(df, series_id),
        'stationarity': test_stationarity(df, series_id),
        'seasonality': detect_seasonality(df, series_id),
        'trend': detect_trend(df, series_id)
    }

# Use it
results = analyze_series(df, series_id=0)
```

### Tip 3: Visual Report Generation

```python
from timeseries_dataset_generator.utils import (
    plot_single_series,
    plot_distribution,
    plot_acf_pacf,
    create_dashboard
)

# Create comprehensive report
series_id = 0
plot_single_series(df, series_id, save_path='01_series.png')
plot_distribution(df, series_id, save_path='02_distribution.png')
plot_acf_pacf(df, series_id, save_path='03_acf_pacf.png')
create_dashboard(df, series_id, save_path='04_dashboard.png')
```

## 📚 Where to Learn More

- **Quick Overview**: `../README.md`
- **Detailed Guide**: `VISUALIZATION_AND_ANALYSIS_GUIDE.md`
- **Dataset Generation Guide**: `DATASET_GENERATION_GUIDE.md`
- **Example Scripts**: `../examples/visualize_datasets.py` and `../examples/analyze_datasets.py`
- **Interactive Tutorial**: `../examples/quick_start.ipynb`

## ⚡ Troubleshooting

### Issue: Module not found
**Solution**: `pip install -r requirements.txt`

### Issue: No plots showing
**Solution**: Use `save_path` parameter to save plots to files

### Issue: statsmodels error
**Solution**: `pip install statsmodels>=0.13.0`

### Issue: Out of memory
**Solution**: Process series individually instead of whole dataset

## 🎓 Learning Path

1. ✅ Run `simple_generation.py` to create sample data
2. ✅ Run `visualize_datasets.py` to see visualization examples
3. ✅ Run `analyze_datasets.py` to see analysis examples
4. ✅ Try modifying the examples with your own parameters
5. ✅ Read the detailed guide in `docs/` folder
6. ✅ Build your own analysis pipeline

## 💡 Quick Reference

| Task | Function | Example |
|------|----------|---------|
| Single plot | `plot_single_series()` | `plot_single_series(df, 0)` |
| Dashboard | `create_dashboard()` | `create_dashboard(df, 0)` |
| Stationarity | `test_stationarity()` | `test_stationarity(df, 0)` |
| Seasonality | `detect_seasonality()` | `detect_seasonality(df, 0)` |
| Summary | `analyze_dataset_summary()` | `analyze_dataset_summary(df)` |

## 🎉 You're Ready!

Start exploring your time series data with powerful visualization and analysis tools!

```python
# Your first complete workflow
import pandas as pd
from timeseries_dataset_generator.utils import (
    create_dashboard,
    analyze_dataset_summary
)

# Load data
df = pd.read_parquet('simple_output/ar_short.parquet')

# Visualize
create_dashboard(df, series_id=0, save_path='dashboard.png')

# Analyze
summary = analyze_dataset_summary(df)
print(summary)

# Done! 🚀
```

---

**Happy Analyzing! 📊✨**

