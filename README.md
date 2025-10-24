# Time Series Dataset Generator

A modular Python library for generating synthetic time series datasets with various characteristics.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This library provides a comprehensive toolkit for generating synthetic time series data with various statistical properties and patterns. It's designed for researchers, data scientists, and developers who need reproducible, well-documented time series datasets for testing, validation, or educational purposes.

## Features

- **25+ Time Series Types**: AR, MA, ARMA, ARIMA, SARIMA, GARCH, EGARCH, and more
- **Reproducible Generation**: Fixed random seed ensures consistent outputs
- **Complete Metadata**: Each series includes 50+ metadata fields
- **Modular Architecture**: 16 well-organized modules for easy extension
- **PyPI-Ready**: Professional package structure
- **Comprehensive Documentation**: Detailed guides and examples
- **Production Ready**: Tested and validated
- **📊 Visualization Tools**: Rich plotting functions for time series analysis
- **📈 Analysis Utilities**: Statistical tests, trend detection, changepoint analysis
- **🎯 Easy to Use**: Simple API for both generation and analysis

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
# Generate 225 parquet files (~2GB)
python examples/dataset_updated.py
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
- [Visualization & Analysis Guide](docs/VISUALIZATION_AND_ANALYSIS_GUIDE.md) - Complete API reference
- [Quick Start Guide](QUICK_START_GUIDE.md) - Common use cases

## Documentation

- **[Quick Start Guide](QUICK_START_GUIDE.md)**: Common tasks and quick reference
- **[Visualization & Analysis Guide](docs/VISUALIZATION_AND_ANALYSIS_GUIDE.md)**: Complete API documentation
- **[Interactive Notebook](examples/quick_start.ipynb)**: Jupyter notebook tutorial
- **[Examples](examples/)**: Working example scripts
  - `quick_start.ipynb`: Interactive Jupyter notebook tutorial
  - `simple_generation.py`: Generate sample datasets
  - `visualize_datasets.py`: Visualization examples
  - `analyze_datasets.py`: Analysis examples
  - `dataset_updated.py`: Full dataset generation

## Project Structure

```
ts-stationary/
├── README.md                       # Main documentation
├── QUICK_START_GUIDE.md            # Quick reference
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
│   └── dataset_updated.py          # Full dataset generation
│
└── docs/                           # Additional documentation
    └── VISUALIZATION_AND_ANALYSIS_GUIDE.md
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
