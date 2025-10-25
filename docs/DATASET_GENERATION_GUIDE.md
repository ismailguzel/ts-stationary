# Dataset Generation Guide

## Overview

The `dataset_generation.py` script provides a flexible, configuration-based system for generating comprehensive time series datasets. You can easily enable/disable different dataset types by modifying a simple configuration dictionary.

## Quick Start

1. Open `examples/dataset_generation.py`
2. Modify the `CONFIG` dictionary at the top (lines 70-168)
3. Run: `python examples/dataset_generation.py`
4. Find generated datasets in the `generated-dataset/` directory

## Configuration Structure

The `CONFIG` dictionary controls:
- Which dataset types to generate
- Which length ranges to use (short, medium, long)
- How many series to generate per configuration
- Output directory location

```python
CONFIG = {
    'random_seed': 42,           # For reproducibility
    'output_dir': 'generated-dataset',
    'count': 10,                 # Number of series per config
    
    'stationary': {
        'enabled': True/False,
        'length_ranges': ['short', 'medium', 'long']
    },
    # ... more categories
}
```

## Pre-defined Scenarios

### Scenario 1: Research Focused (Current Default)

**Use Case:** Research focused on structural breaks without seasonality

**Configuration:**
- Stationary series (all lengths)
- Deterministic trends (all lengths)
- Point anomalies (all lengths, single & multiple)
- Collective anomalies (only long series, multiple only)
- Contextual anomalies (disabled)
- Stochastic series (all lengths)
- Volatility models (all lengths)
- Seasonality (disabled)
- Structural breaks (only long series, multiple only)

**Already configured by default!** Just run the script.

---

### Scenario 2: Full Dataset Generation

**Use Case:** Generate all possible dataset types

**Changes needed in CONFIG:**

```python
'seasonality': {
    'enabled': True,
    'types': ['single', 'multiple', 'sarma', 'sarima'],
    'length_ranges': ['short', 'medium']  # Avoid long for computational reasons
}

'contextual_anomalies': {
    'enabled': True,
    'single': {
        'enabled': True,
        'length_ranges': ['short', 'medium', 'long']
    },
    'multiple': {
        'enabled': True,
        'length_ranges': ['long']
    }
}

# Also add 'single' sections to structural breaks:
'structural_breaks': {
    'mean_shift': {
        'enabled': True,
        'single': {
            'enabled': True,
            'length_ranges': ['short', 'medium', 'long']
        },
        'multiple': {
            'enabled': True,
            'length_ranges': ['medium', 'long']
        }
    },
    # ... similar for variance_shift and trend_shift
}
```

---

### Scenario 3: Quick Test (Stationary & Trends Only)

**Use Case:** Fast generation for testing

**Changes needed:**

```python
# Set 'enabled': False for all except:
'stationary': {'enabled': True, ...}
'deterministic_trends': {'enabled': True, ...}

# All others:
'enabled': False
```

**Estimated time:** ~5-10 minutes

---

### Scenario 4: Anomaly Detection Focus

**Use Case:** Datasets for anomaly detection research

**Changes needed:**

```python
# Keep enabled:
'stationary': True  # As base series
'point_anomalies': True
'collective_anomalies': True
'contextual_anomalies': True

# Disable:
'deterministic_trends': False
'stochastic': False
'volatility': False
'seasonality': False
'structural_breaks': False (set all sub-categories to False)
```

---

## Length Ranges

- **short**: 50-100 data points
- **medium**: 300-500 data points
- **long**: 1000-10000 data points

## Dataset Categories

### 1. Stationary Series
- AR (Autoregressive)
- MA (Moving Average)
- ARMA (Autoregressive Moving Average)
- White Noise

### 2. Deterministic Trends
- Linear
- Quadratic
- Cubic
- Exponential
- Damped

### 3. Point Anomalies
- Single: One outlier at specified location
- Multiple: Random number of outliers

### 4. Collective Anomalies
- Multiple: Group anomalies in long series

### 5. Contextual Anomalies
- Single: One contextual anomaly
- Multiple: Multiple contextual anomalies

### 6. Stochastic Series
- Random Walk
- Random Walk with Drift
- ARI (Autoregressive Integrated)
- IMA (Integrated Moving Average)
- ARIMA

### 7. Volatility Models
- ARCH
- GARCH
- EGARCH
- APARCH

### 8. Seasonality
- Single Seasonality
- Multiple Seasonality
- SARMA
- SARIMA

### 9. Structural Breaks
- Mean Shift
- Variance Shift
- Trend Shift

## Output Structure

```
generated-dataset/
├── stationary/
│   ├── ar/
│   │   ├── short/
│   │   ├── medium/
│   │   └── long/
│   ├── ma/
│   └── ...
├── deterministic_trend_linear/
│   ├── up/
│   └── down/
├── point_anomaly_single/
├── collective_anomaly/
└── ...
```

Each folder contains `.parquet` files with:
- Time series data
- Metadata (parameters, properties)
- Labels (for anomalies/breaks)

## Tips

1. **Start small**: Test with a quick scenario first
2. **Disk space**: Full generation can create several GB of data
3. **Time**: Full generation may take hours depending on your system
4. **Memory**: Long series (especially GARCH/SARIMA) can be memory intensive
5. **Customize counts**: Reduce `'count'` in CONFIG for faster generation

## Troubleshooting

**Problem**: Generation is very slow
- **Solution**: Reduce length ranges (use only 'short' or 'medium')
- **Solution**: Reduce 'count' to 5 or fewer
- **Solution**: Disable heavy models (EGARCH, APARCH, SARIMA)

**Problem**: Out of memory errors
- **Solution**: Generate 'long' series separately
- **Solution**: Reduce 'count' parameter
- **Solution**: Close other applications

**Problem**: Want to regenerate specific categories
- **Solution**: Delete the corresponding folder in `generated-dataset/`
- **Solution**: Disable all other categories in CONFIG

## Advanced Customization

You can also modify the script to:
- Add new length ranges in the `get_length_range()` function
- Change parameter distributions in the generator functions
- Add custom folder structures
- Implement your own dataset types

## Example: Custom Configuration

```python
CONFIG = {
    'random_seed': 123,
    'output_dir': 'my-custom-dataset',
    'count': 5,
    
    # Only what I need for my research:
    'stationary': {
        'enabled': True,
        'length_ranges': ['medium']  # Only medium length
    },
    'collective_anomalies': {
        'enabled': True,
        'multiple': {
            'enabled': True,
            'length_ranges': ['long']
        }
    },
    # Everything else: enabled=False
}
```

## Support

For issues or questions:
1. Check the docstrings in `examples/dataset_generation.py`
2. Review the main project [README](../README.md)
3. Check the [Visualization & Analysis Guide](VISUALIZATION_AND_ANALYSIS_GUIDE.md)
4. See the [Quick Start Guide](QUICK_START_GUIDE.md)

---

**Current Default Configuration**: Research Focused Scenario (no seasonality, no contextual anomalies, long series for multiple structural breaks)

