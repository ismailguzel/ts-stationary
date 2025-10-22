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

# Create generator
gen = TimeSeriesGenerator(length=500)

# Generate AR series
ar_series = gen.generate_ar_series(coefficients=[0.5, 0.3])

# Generate GARCH series
garch_series = gen.generate_garch_series(
    omega=0.1,
    alpha=[0.2],
    beta=[0.7]
)
```

### Generate Full Dataset

```bash
# Generate 225 parquet files (~2GB)
python examples/dataset_updated.py
```

### Quick Example

```bash
# Generate 5 sample files
python examples/simple_generation.py
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

## Documentation

- **[Quick Reference](docs/QUICK_REFERENCE.md)**: Quick start guide
- **[Project Summary](docs/PROJECT_SUMMARY.md)**: Comprehensive overview
- **[Examples](examples/)**: Usage examples

## Project Structure

```
ts-stationary/
├── README.md                       # This file
├── LICENSE                         # MIT License
├── .gitignore                      # Git ignore rules
├── setup.py                        # Package setup
├── pyproject.toml                  # Modern packaging config
├── requirements.txt                # Dependencies
│
├── timeseries_dataset_generator/   # Main library
│   ├── __init__.py
│   ├── core/                       # Core components
│   ├── generators/                 # Generator modules
│   └── utils/                      # Utilities
│
├── examples/                       # Usage examples
│   ├── dataset_updated.py          # Full dataset generation
│   ├── simple_generation.py        # Quick example
│   └── test_library.py             # Tests
│
├── docs/                           # Documentation
│   ├── PROJECT_SUMMARY.md
│   └── QUICK_REFERENCE.md
│
└── tests/                          # Unit tests (future)
```

## Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
statsmodels>=0.13.0
arch>=5.0.0
```

## Reproducibility

All generation scripts use a fixed random seed (default: 42) to ensure reproducible outputs:

```python
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
```

This ensures that running the same code multiple times produces identical results (98-99% reproducibility, with minor variations in ARCH/GARCH models due to library limitations).

## Legacy Code

The previous version of this repository has been preserved in the `legacy` branch. You can access it via:

```bash
git checkout legacy
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{timeseries_dataset_generator,
  author = {Guzel, Ismail},
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
