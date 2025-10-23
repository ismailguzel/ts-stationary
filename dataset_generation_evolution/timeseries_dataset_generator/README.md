# Time Series Dataset Generator

A comprehensive Python toolkit for generating synthetic time series datasets with various characteristics including trends, seasonality, anomalies, and structural breaks.

## Features

###  Stationary Processes
- **White Noise**: Pure random series
- **AR (Autoregressive)**: Series depending on past values
- **MA (Moving Average)**: Series depending on past errors
- **ARMA**: Combined autoregressive and moving average

###  Deterministic Trends
- Linear trends (upward/downward)
- Quadratic trends
- Cubic trends
- Exponential trends
- Damped trends

###  Stochastic Trends
- Random Walk
- Random Walk with Drift
- IMA (Integrated Moving Average)
- ARI (Autoregressive Integrated)
- ARIMA (Autoregressive Integrated Moving Average)

###  Volatility Clustering
- ARCH (Autoregressive Conditional Heteroscedasticity)
- GARCH (Generalized ARCH)
- EGARCH (Exponential GARCH)
- APARCH (Asymmetric Power ARCH)

###  Seasonality
- Single seasonal patterns
- Multiple seasonal patterns
- SARMA (Seasonal ARMA)
- SARIMA (Seasonal ARIMA)

###  Anomalies
- **Point Anomalies**: Single outlier points
- **Collective Anomalies**: Sequences of unusual behavior
- **Contextual Anomalies**: Context-dependent anomalies

###  Structural Breaks
- **Mean Shifts**: Level changes
- **Variance Shifts**: Volatility changes
- **Trend Shifts**: Slope/direction changes

## Installation

### From PyPI (when published)
```bash
pip install timeseries-dataset-generator
```

### From Source
```bash
git clone https://github.com/yourusername/timeseries-dataset-generator.git
cd timeseries-dataset-generator
pip install -e .
```

## Quick Start

```python
from timeseries_dataset_generator import TimeSeriesGenerator
from timeseries_dataset_generator.generators.stationary import generate_ar_dataset
from timeseries_dataset_generator.generators.trends import generate_linear_trend_dataset

# Generate AR dataset
generate_ar_dataset(
    ts_generator_class=TimeSeriesGenerator,
    folder='output/ar_data',
    count=100,
    length_range=(100, 500)
)

# Generate linear trend dataset
generate_linear_trend_dataset(
    ts_generator_class=TimeSeriesGenerator,
    folder='output/linear_trend',
    kind='ar',
    count=50,
    length_range=(300, 500),
    sign=1  # upward trend
)
```

## Usage Examples

### Generating Different Types of Series

```python
from timeseries_dataset_generator import TimeSeriesGenerator
from timeseries_dataset_generator.generators import (
    generate_arma_dataset,
    generate_garch_dataset,
    generate_single_seasonality_dataset,
    generate_point_anomaly_dataset
)

# ARMA series
generate_arma_dataset(
    TimeSeriesGenerator,
    folder='output/arma',
    count=50,
    length_range=(100, 200)
)

# GARCH volatility
generate_garch_dataset(
    TimeSeriesGenerator,
    folder='output/garch',
    count=30,
    length_range=(500, 1000)
)

# Seasonal series
generate_single_seasonality_dataset(
    TimeSeriesGenerator,
    folder='output/seasonal',
    count=40,
    length_range=(200, 400)
)

# Series with point anomalies
generate_point_anomaly_dataset(
    TimeSeriesGenerator,
    folder='output/anomalies',
    kind='ar',
    count=25,
    length_range=(300, 500),
    anomaly_type='single',
    location='middle'
)
```

### Using the Core Generator Directly

```python
from timeseries_dataset_generator import TimeSeriesGenerator

# Create a generator instance
ts = TimeSeriesGenerator(length=500)

# Generate a stationary AR series
df, info = ts.generate_stationary_base_series('ar')
print(f"Generated AR({info['ar_order']}) series")

# Add a linear trend
df, trend_info = ts.generate_deterministic_trend_linear(df, sign=1)

# Add seasonality
df, season_info = ts.generate_seasonality_from_base_series('single')

# Introduce anomalies
df, anomaly_info = ts.generate_point_anomaly(df, location='middle')
```

## Output Format

All generators save data in Parquet format with the following structure:

| Column | Description |
|--------|-------------|
| `series_id` | Unique identifier for each series |
| `time` | Time index |
| `data` | Time series values |
| `label` | Category/type label |
| `is_stationary` | Stationarity flag (1/0) |
| `base_series` | Base process type |
| `order` | Process order |
| `base_coefs` | Process coefficients |
| ... | Additional metadata fields |

### Metadata Fields

Each generated series includes comprehensive metadata:
- Process parameters (AR/MA coefficients, orders)
- Trend characteristics
- Seasonality information
- Anomaly locations and types
- Structural break points
- Stationarity indicators

## Project Structure

```
timeseries_dataset_generator/
 __init__.py
 core/
    __init__.py
    generator.py       # Core TimeSeriesGenerator class
    metadata.py         # Metadata management
 generators/
    __init__.py
    stationary.py       # Stationary process generators
    trends.py           # Trend generators
    stochastic.py       # Stochastic trend generators
    volatility.py       # Volatility model generators
    seasonality.py      # Seasonality generators
    anomalies.py        # Anomaly generators
    structural_breaks.py # Structural break generators
 utils/
    __init__.py
    helpers.py          # Helper utilities
 examples/
     generate_dataset.py # Example usage script
```

## Requirements

- Python >= 3.8
- numpy >= 1.20.0
- pandas >= 1.3.0
- matplotlib >= 3.3.0
- statsmodels >= 0.13.0
- arch >= 5.0.0
- tqdm >= 4.60.0
- pyarrow >= 6.0.0

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black timeseries_dataset_generator/
```

### Type Checking
```bash
mypy timeseries_dataset_generator/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{timeseries_dataset_generator,
  title = {Time Series Dataset Generator},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/timeseries-dataset-generator}
}
```

## Acknowledgments

- Built with [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), and [statsmodels](https://www.statsmodels.org/)
- Volatility models powered by [ARCH](https://arch.readthedocs.io/)

## Support

For questions and support:
-  Email: your.email@example.com
-  Issues: [GitHub Issues](https://github.com/yourusername/timeseries-dataset-generator/issues)
-  Documentation: [Read the Docs](https://timeseries-dataset-generator.readthedocs.io)

