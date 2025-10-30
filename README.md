# Time Series Dataset Generator

A modular Python library for generating synthetic time series datasets with various characteristics.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This library provides a comprehensive toolkit for generating synthetic time series data with various statistical properties and patterns. It's designed for researchers, data scientists, and developers who need reproducible, well-documented time series datasets for testing, validation, or educational purposes.

## Features

- **25+ Time Series Types**: AR, MA, ARMA, ARIMA, SARIMA, GARCH, EGARCH, and more
- **Reproducible Generation**: Fixed random seed ensures consistent outputs
- **Complete Metadata**: Each series includes hierarchical, non-redundant metadata fields (50+)
- **Modular Architecture**: 16 well-organized modules for easy extension
- **PyPI-Ready**: Professional package structure
- **Comprehensive Documentation**: Detailed guides and examples
- **Production Ready**: Tested and validated
- **Visualization Tools**: Rich plotting functions for time series analysis
- **Analysis Utilities**: Statistical tests, trend detection, changepoint analysis
- **Easy to Use**: Simple API for both generation and analysis

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ismailguzel/ts-stationary.git
cd ts-stationary

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
from timeseries_dataset_generator import TimeSeriesGenerator
from timeseries_dataset_generator.generators import generate_ar_dataset

# Generate AR dataset
generate_ar_dataset(
    TimeSeriesGenerator,
    folder='output/ar',
    count=10,
    length_range=(100, 500)
)
```

### Visualize and Analyze

```python
import pandas as pd
from timeseries_dataset_generator.utils import (
    plot_single_series,
    create_dashboard,
    analyze_dataset_summary
)

# Load generated data
df = pd.read_parquet('output/ar.parquet')

# Visualize
plot_single_series(df, series_id=0, save_path='series_plot.png')
create_dashboard(df, series_id=0, save_path='dashboard.png')

# Analyze
summary = analyze_dataset_summary(df)
print(summary)
```

### Generate Full Dataset

```bash
# Generate comprehensive dataset with configurable scenarios
python examples/dataset_generation.py
```

### Quick Examples

```bash
# Interactive tutorial (recommended for beginners)
jupyter notebook examples/quick_start.ipynb

# Generate sample datasets
python examples/simple_generation.py

# Visualize generated data
python examples/visualize_datasets.py

# Analyze generated data
python examples/analyze_datasets.py
```

## Dataset Categories

The library can generate 7 major categories of time series:

1. **Stationary Processes**: AR, MA, ARMA, White Noise
2. **Deterministic Trends**: Linear, Quadratic, Cubic, Exponential, Damped
3. **Stochastic Trends**: Random Walk, ARI, IMA, ARIMA
4. **Volatility Models**: ARCH, GARCH, EGARCH, APARCH
5. **Seasonality**: Single, Multiple, SARMA, SARIMA
6. **Anomalies**: Point, Collective, Contextual (single/multiple)
7. **Structural Breaks**: Mean shift, Variance shift, Trend shift

## Output Format

Generated datasets are saved as parquet files with the following structure:

```
generated-dataset/
├── stationary/
│   ├── ar/
│   │   ├── short.parquet      (100 time steps)
│   │   ├── medium.parquet     (400 time steps)
│   │   └── long.parquet       (1000 time steps)
│   ├── ma/
│   ├── arma/
│   └── white_noise/
├── deterministic_trend_linear/
│   ├── up/
│   └── down/
└── ... (25+ categories, 225 files total)
```

Each parquet file includes:
- `series_id`: Unique identifier
- `time`: Time index
- `data`: Time series values
- `label`: Category label
- Process parameters (coefficients, etc.)
- Statistical properties
- Generation metadata

## Visualization & Analysis Tools

The library includes powerful visualization and analysis utilities:

**Visualization**: 8 plotting functions including `plot_single_series()`, `create_dashboard()`, `plot_acf_pacf()`, and more.

**Analysis**: 10 statistical functions including `test_stationarity()`, `detect_seasonality()`, `detect_trend()`, and more.

```python
from timeseries_dataset_generator.utils import create_dashboard, analyze_dataset_summary
import pandas as pd

df = pd.read_parquet('output/ar.parquet')

# Create visual dashboard
create_dashboard(df, series_id=0, save_path='dashboard.png')

# Statistical analysis
summary = analyze_dataset_summary(df)
summary.to_csv('analysis.csv')
```

**For detailed documentation**, see:
- [Quick Start Guide](docs/QUICK_START_GUIDE.md) - Common use cases
- [Visualization & Analysis Guide](docs/VISUALIZATION_AND_ANALYSIS_GUIDE.md) - Complete API reference
- [Dataset Generation Guide](docs/DATASET_GENERATION_GUIDE.md) - Configurable dataset generation

## Documentation

- **[Quick Start Guide](docs/QUICK_START_GUIDE.md)**: Common tasks and quick reference
- **[Visualization & Analysis Guide](docs/VISUALIZATION_AND_ANALYSIS_GUIDE.md)**: Complete API documentation
- **[Dataset Generation Guide](docs/DATASET_GENERATION_GUIDE.md)**: Configurable dataset generation scenarios
- **[Interactive Notebook](examples/quick_start.ipynb)**: Jupyter notebook tutorial
- **[Examples](examples/)**: Working example scripts
  - `quick_start.ipynb`: Interactive Jupyter notebook tutorial
  - `simple_generation.py`: Generate sample datasets
  - `visualize_datasets.py`: Visualization examples
  - `analyze_datasets.py`: Analysis examples
  - `dataset_generation.py`: Configurable full dataset generation

## Project Structure

```
ts-stationary/
├── README.md                       # Main documentation
├── LICENSE                         # MIT License
├── setup.py                        # Package setup
├── pyproject.toml                  # Modern packaging config
├── requirements.txt                # Dependencies
│
├── timeseries_dataset_generator/   # Main library
│   ├── __init__.py
│   ├── core/                       # Core components
│   ├── generators/                 # Time series generators
│   └── utils/                      # Visualization & analysis tools
│
├── examples/                       # Working examples
│   ├── quick_start.ipynb           # Interactive Jupyter tutorial
│   ├── simple_generation.py        # Generate sample data
│   ├── visualize_datasets.py       # Visualization examples
│   ├── analyze_datasets.py         # Analysis examples
│   └── dataset_generation.py       # Configurable dataset generation
│
└── docs/                           # Documentation
    ├── QUICK_START_GUIDE.md        # Quick reference guide
    ├── DATASET_GENERATION_GUIDE.md # Dataset generation guide
    └── VISUALIZATION_AND_ANALYSIS_GUIDE.md # API reference
```

## Dependencies

### Core Dependencies
```
numpy>=1.21.0
pandas>=1.3.0
statsmodels>=0.13.0
arch>=5.0.0
```

### Visualization & Analysis (Optional)
```
matplotlib>=3.3.0
seaborn>=0.11.0
scipy>=1.7.0
```

Install with visualization support:
```bash
pip install -e ".[viz]"
```

## Reproducibility

All generation scripts use a fixed random seed (default: 42) to ensure reproducible outputs:

```python
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
```

This ensures that running the same code multiple times produces identical results (98-99% reproducibility, with minor variations in ARCH/GARCH models due to library limitations).

## Legacy and Development History

In addition to the `legacy` branch—which contains the original version of this repository—there is also a `data_generation_evolution` branch. This branch includes all previous code and documents the library's development process.

You can access these branches with:

```bash
# Checkout the legacy branch
git checkout legacy

# Or checkout the data_generation_evolution branch
git checkout data_generation_evolution
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{timeseries_dataset_generator,
  author = {TS-Stationary Project Team, TUBITAK 1001 Project No. 124F095},
  title = {Time Series Dataset Generator},
  year = {2025},
  url = {https://github.com/ismailguzel/ts-stationary}
}
```

## Acknowledgments

- Built with `numpy`, `pandas`, `statsmodels`, and `arch`
- Refactored from a monolithic 1999-line file into a modular library
- Designed for reproducibility and ease of use

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Version**: 0.1.0 | **Status**: Production Ready | **License**: MIT

## Metadata Schema (Hierarchical)

Each generated dataset row includes hierarchical, non-redundant metadata fields:

- Core: `series_id`, `time`, `data`, `label`, `length`, `is_stationary`
- Hierarchy: `primary_category` (stationary | trend | stochastic | seasonality | volatility | anomaly | structural_break), `sub_category`
- Base process: `base_series`, `base_process_type`, `order`, `base_coefs`
- Trend: `trend_type`, `trend_slope`, `trend_intercept`, `trend_coef_a`, `trend_coef_b`, `trend_coef_c`, `trend_damping_rate`
- Stochastic: `stochastic_type`, `difference`, `drift_value`
- Seasonality: `seasonality_type`, `seasonality_periods`, `seasonality_amplitudes`, `seasonality_from_base`, `seasonal_difference`, `seasonal_ar_order`, `seasonal_ma_order`
- Volatility: `volatility_type`, `volatility_alpha`, `volatility_beta`, `volatility_omega`, `volatility_theta`, `volatility_lambda`, `volatility_gamma`, `volatility_delta`
- Anomaly: `anomaly_type`, `anomaly_count`, `anomaly_indices`, `anomaly_magnitudes`
- Structural Break: `break_type`, `break_count`, `break_indices`, `break_magnitudes`, `break_directions`, `trend_shift_change_types`
- Locations: `location_point`, `location_collective`, `location_mean_shift`, `location_variance_shift`, `location_trend_shift`, `location_contextual`
- Noise & Misc: `noise_type`, `noise_std`, `sampling_frequency`

Example (first row transpose view):

```python
import pandas as pd

df = pd.read_parquet('output/ar.parquet')
print(df.head(1).T)
```
