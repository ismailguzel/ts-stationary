"""Setup script for timeseries_dataset_generator package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="timeseries-dataset-generator",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive toolkit for generating synthetic time series datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/timeseries-dataset-generator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.3.0",
        "statsmodels>=0.13.0",
        "arch>=5.0.0",
        "tqdm>=4.60.0",
        "pyarrow>=6.0.0",  # for parquet support
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "sphinx>=4.5.0",
        ],
    },
    keywords=[
        "time-series",
        "dataset-generation",
        "synthetic-data",
        "arima",
        "garch",
        "anomaly-detection",
        "forecasting",
        "machine-learning",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/timeseries-dataset-generator/issues",
        "Source": "https://github.com/yourusername/timeseries-dataset-generator",
        "Documentation": "https://timeseries-dataset-generator.readthedocs.io",
    },
)

