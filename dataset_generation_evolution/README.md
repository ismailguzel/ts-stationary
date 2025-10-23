# Time Series Dataset Generator

A modular Python library for generating synthetic time series datasets with various characteristics.

**Status:** Production Ready | **Version:** 0.1.0 | **License:** MIT

## Project Structure

```
dataset_generation/
├── timeseries_dataset_generator/    # Main library (modular)
│   ├── core/                        # Core components
│   ├── generators/                  # Dataset generators
│   ├── utils/                       # Utilities
│   ├── examples/                    # Example scripts
│   └── README.md                    # Library documentation
│
├── dataset_updated.py               # Main dataset generation script
├── simple_generation.py             # Simple example
├── test_library.py                  # Functional tests
├── quick_test.py                    # Structure tests
│
└── _old_files/                      # Archived old files
    ├── dataset.py                   # Old version
    ├── dataset_generator.py         # Old monolithic version
    └── generator.py                 # Old generator
```

## Quick Start

### 1. Install Dependencies

```bash
pip install numpy pandas matplotlib statsmodels arch tqdm pyarrow
```

### 2. Install Library

```bash
cd timeseries_dataset_generator
pip install -e .
cd ..
```

### 3. Generate Datasets

**Option A: Full dataset (recommended)**
```bash
python dataset_updated.py
```
This generates comprehensive datasets in `generated-dataset/` directory.

**Option B: Simple example**
```bash
python simple_generation.py
```
This generates a small sample in `simple_output/` directory.

## Reproducibility

All scripts use fixed random seeds for reproducibility:
- Default seed: `42`
- Same code = Same output every time
- Modify `RANDOM_SEED` variable to change seed

## Dataset Categories

The library generates the following types of time series:

- **Stationary**: AR, MA, ARMA, White Noise
- **Trends**: Linear, Quadratic, Cubic, Exponential, Damped
- **Stochastic**: Random Walk, ARIMA family
- **Volatility**: ARCH, GARCH, EGARCH, APARCH
- **Seasonality**: Single, Multiple, SARMA, SARIMA
- **Anomalies**: Point, Collective, Contextual
- **Structural Breaks**: Mean shift, Variance shift, Trend shift

## Output Format

Generated datasets are saved as Parquet files with structure:
```
generated-dataset/
├── stationary/
│   ├── ar/
│   │   ├── short.parquet
│   │   ├── medium.parquet
│   │   └── long.parquet
│   ├── ma/
│   └── ...
├── deterministic_trend_linear/
│   ├── up/
│   │   ├── ar/
│   │   └── ...
│   └── down/
└── ...
```

Each parquet file contains:
- `series_id`: Unique series identifier
- `time`: Time index
- `data`: Time series values
- Metadata columns (base_series, order, coefficients, etc.)
- `label`: Category label

## Testing

```bash
# Structure test (no dependencies needed)
python quick_test.py

# Functional test (requires dependencies)
python test_library.py
```

## Files Description

### Active Files
- `dataset_updated.py` - Main script using new library
- `simple_generation.py` - Quick example with 5 dataset types
- `test_library.py` - Comprehensive functionality tests
- `quick_test.py` - Structure and import validation

### Old Files (Archived)
- `_old_files/dataset_generator.py` - Old monolithic version (1999 lines)
- `_old_files/dataset.py` - Old generation script
- `_old_files/generator.py` - Old core generator

## Library Documentation

See `timeseries_dataset_generator/README.md` for detailed library documentation.

## Usage Example

```python
from timeseries_dataset_generator import TimeSeriesGenerator
from timeseries_dataset_generator.generators import generate_ar_dataset

# Generate AR dataset
generate_ar_dataset(
    TimeSeriesGenerator,
    folder='output/ar',
    count=100,
    length_range=(100, 500)
)
```

## Notes

- All generation uses fixed random seeds for reproducibility
- Old files are in `_old_files/` directory (can be safely deleted)
- Library is installable via pip: `pip install -e timeseries_dataset_generator/`
- Generated datasets are in Parquet format for efficiency

