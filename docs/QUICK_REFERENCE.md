# Quick Reference Guide

## Project Status
**PRODUCTION READY** - Version 0.1.0 - MIT License

## Quick Start

### 1. Installation
```bash
cd timeseries_dataset_generator
pip install -r requirements.txt
pip install -e .
```

### 2. Generate Dataset
```bash
# Full dataset (225 files, ~2GB)
python dataset_updated.py

# Sample dataset (5 files)
python simple_generation.py
```

### 3. Validate
```bash
python validate_project.py
```

## File Guide

### Use These Files

| File | Purpose | When to Use |
|------|---------|-------------|
| `dataset_updated.py` | Generate full dataset | Production |
| `simple_generation.py` | Quick example | Testing |
| `test_library.py` | Run tests | After changes |
| `validate_project.py` | Check project | Setup validation |
| `compare_datasets.py` | Check reproducibility | Verify consistency |
| `cleanup.py` | Clean outputs | Cleanup |

### Documentation Files

| File | Content |
|------|---------|
| `README.md` | Main guide |
| `PROJECT_SUMMARY.md` | Complete overview |
| `timeseries_dataset_generator/README.md` | Library docs |
| `timeseries_dataset_generator/QUICKSTART.md` | Quick tutorial |

### Don't Touch These

- `_old_files/` - Archived code
- `timeseries_dataset_generator/` - Library (unless developing)
- `__pycache__/` - Python cache

## Common Tasks

### Change Random Seed
Edit `RANDOM_SEED = 42` in:
- `dataset_updated.py`
- `simple_generation.py`

### Add New Generator
1. Add function to `timeseries_dataset_generator/generators/`
2. Import in `timeseries_dataset_generator/__init__.py`
3. Add call in `dataset_updated.py`

### Check Reproducibility
```bash
python dataset_updated.py
mv generated-dataset test1
python dataset_updated.py
mv generated-dataset test2
python compare_datasets.py  # Edit to compare test1 and test2
```

### Clean Up
```bash
python cleanup.py  # Remove test outputs
```

## Output Structure
```
generated-dataset/
├── stationary/
│   ├── ar/
│   │   ├── short.parquet    (100 steps)
│   │   ├── medium.parquet   (400 steps)
│   │   └── long.parquet     (1000 steps)
│   ├── ma/
│   ├── arma/
│   └── white_noise/
├── deterministic_trend_linear/
│   ├── up/
│   └── down/
└── ... (25+ categories, 225 files total)
```

## Dataset Categories

1. **Stationary**: AR, MA, ARMA, White Noise
2. **Trends**: Linear, Quadratic, Cubic, Exponential, Damped
3. **Stochastic**: Random Walk, ARI, IMA, ARIMA
4. **Volatility**: ARCH, GARCH, EGARCH, APARCH
5. **Seasonality**: Single, Multiple, SARMA, SARIMA
6. **Anomalies**: Point, Collective, Contextual
7. **Breaks**: Mean shift, Variance shift, Trend shift

## Troubleshooting

### Import Error
```bash
cd timeseries_dataset_generator
pip install -r requirements.txt
pip install -e .
```

### No Output
- Check `RANDOM_SEED` is set
- Check output directory permissions
- Run with `python3` instead of `python`

### Reproducibility Issues
- ARCH/GARCH models: ~99% reproducible (library limitation)
- Other models: 100% reproducible
- Ensure same Python/numpy/statsmodels versions

## Key Features

- **Modularity**: 16 organized modules
- **Reproducibility**: Fixed seed (42)
- **Documentation**: 4 comprehensive guides
- **Testing**: 4 validation scripts
- **Output**: 225 parquet files with metadata
- **Quality**: No emojis, clean code, PyPI-ready

## Dependencies
```
numpy>=1.21.0
pandas>=1.3.0
statsmodels>=0.13.0
arch>=5.0.0
```

## Next Steps

1. Generate dataset: `python dataset_updated.py`
2. Read documentation: `PROJECT_SUMMARY.md`
3. Run tests: `python validate_project.py`
4. Optional: Publish to PyPI

---
**Version**: 0.1.0 | **Status**: Production Ready | **License**: MIT
