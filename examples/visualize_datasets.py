"""
Example: Reading and Visualizing Generated Time Series Datasets

This script demonstrates how to:
1. Read generated Parquet files
2. Explore the dataset structure
3. Visualize individual series
4. Create comparative plots
5. Generate comprehensive dashboards
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")  # veya kuruluysa 'TkAgg'

# Add library to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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

print("=" * 80)
print("Time Series Dataset Visualization Example")
print("=" * 80)

# ============================================================================
# 1. LOAD DATASET
# ============================================================================
print("\n[1] Loading Dataset...")
print("-" * 80)

# Check if simple_output directory exists
data_dir = Path("simple_output")
if not data_dir.exists():
    print(f"Error: {data_dir} not found!")
    print("Please run 'python examples/simple_generation.py' first to generate sample data.")
    sys.exit(1)

# List available parquet files
parquet_files = list(data_dir.glob("*.parquet"))
if not parquet_files:
    print(f"Error: No parquet files found in {data_dir}")
    sys.exit(1)

print(f"Found {len(parquet_files)} parquet file(s):")
for f in parquet_files:
    print(f"  - {f.name}")

# Load the first dataset as an example
example_file = parquet_files[0]
print(f"\nLoading: {example_file.name}")
df = pd.read_parquet(example_file)

print(f"\nDataset loaded successfully!")
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")
print(f"  Number of unique series: {df['series_id'].nunique()}")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================================
# 2. EXPLORE DATASET STRUCTURE
# ============================================================================
print("\n[2] Exploring Dataset Structure...")
print("-" * 80)

print("\nFirst few rows:")
print(df.head(10))

print("\nDataset info:")
print(df.info())

print("\nBasic statistics:")
print(df.describe())

if 'label' in df.columns:
    print("\nSeries labels:")
    print(df.groupby('label')['series_id'].nunique())

# ============================================================================
# 3. VISUALIZE SINGLE SERIES
# ============================================================================
print("\n[3] Visualizing Single Series...")
print("-" * 80)

# Create output directory for plots
output_dir = Path("visualization_output")
output_dir.mkdir(exist_ok=True)

# Get first series ID
first_series_id = df['series_id'].iloc[0]
print(f"\nPlotting series ID: {first_series_id}")

# Plot single series
fig1 = plot_single_series(
    df,
    series_id=first_series_id,
    highlight_anomalies=True,
    highlight_breaks=True,
    save_path=output_dir / "single_series.png"
)
print(f"  ✓ Saved: single_series.png")

# ============================================================================
# 4. VISUALIZE MULTIPLE SERIES
# ============================================================================
print("\n[4] Visualizing Multiple Series...")
print("-" * 80)

# Plot multiple series in grid
n_series = min(4, df['series_id'].nunique())
print(f"\nPlotting {n_series} series in a grid layout...")

fig2 = plot_multiple_series(
    df,
    n_series=n_series,
    save_path=output_dir / "multiple_series.png"
)
print(f"  ✓ Saved: multiple_series.png")

# ============================================================================
# 5. COMPARE SERIES
# ============================================================================
print("\n[5] Comparing Series...")
print("-" * 80)

if df['series_id'].nunique() >= 2:
    series_ids = df['series_id'].unique()[:2]
    print(f"\nComparing series: {series_ids[0]} and {series_ids[1]}")
    
    fig3 = plot_series_comparison(
        df,
        series_ids=list(series_ids),
        title="Series Comparison",
        save_path=output_dir / "series_comparison.png"
    )
    print(f"  ✓ Saved: series_comparison.png")
else:
    print("\nSkipping comparison (need at least 2 series)")

# ============================================================================
# 6. DISTRIBUTION ANALYSIS
# ============================================================================
print("\n[6] Analyzing Distribution...")
print("-" * 80)

print(f"\nPlotting distribution for series ID: {first_series_id}")

fig4 = plot_distribution(
    df,
    series_id=first_series_id,
    bins=30,
    save_path=output_dir / "distribution.png"
)
print(f"  ✓ Saved: distribution.png")

# ============================================================================
# 7. ACF/PACF PLOTS
# ============================================================================
print("\n[7] Computing ACF/PACF...")
print("-" * 80)

print(f"\nPlotting ACF/PACF for series ID: {first_series_id}")

try:
    fig5 = plot_acf_pacf(
        df,
        series_id=first_series_id,
        lags=min(40, len(df[df['series_id'] == first_series_id]) // 2 - 1),
        save_path=output_dir / "acf_pacf.png"
    )
    print(f"  ✓ Saved: acf_pacf.png")
except ImportError:
    print("  ⚠ statsmodels not installed, skipping ACF/PACF plots")

# ============================================================================
# 8. ROLLING STATISTICS
# ============================================================================
print("\n[8] Computing Rolling Statistics...")
print("-" * 80)

series_length = len(df[df['series_id'] == first_series_id])
window = max(10, series_length // 20)
print(f"\nPlotting rolling statistics (window={window}) for series ID: {first_series_id}")

fig6 = plot_rolling_statistics(
    df,
    series_id=first_series_id,
    window=window,
    save_path=output_dir / "rolling_stats.png"
)
print(f"  ✓ Saved: rolling_stats.png")

# ============================================================================
# 9. COMPREHENSIVE DASHBOARD
# ============================================================================
print("\n[9] Creating Comprehensive Dashboard...")
print("-" * 80)

print(f"\nGenerating dashboard for series ID: {first_series_id}")

fig7 = create_dashboard(
    df,
    series_id=first_series_id,
    save_path=output_dir / "dashboard.png"
)
print(f"  ✓ Saved: dashboard.png")

# ============================================================================
# 10. CATEGORY OVERVIEW (if multiple labels exist)
# ============================================================================
print("\n[10] Category Overview...")
print("-" * 80)

if 'label' in df.columns and df['label'].nunique() > 1:
    print(f"\nCreating overview of {df['label'].nunique()} categories...")
    
    fig8 = plot_category_overview(
        df,
        max_series=min(9, df['label'].nunique()),
        save_path=output_dir / "category_overview.png"
    )
    print(f"  ✓ Saved: category_overview.png")
else:
    print("\nSkipping category overview (single label or no label column)")

# ============================================================================
# 11. LOAD AND VISUALIZE MULTIPLE DATASETS
# ============================================================================
print("\n[11] Multi-Dataset Visualization...")
print("-" * 80)

if len(parquet_files) > 1:
    print(f"\nVisualizing samples from {min(3, len(parquet_files))} datasets...")
    
    all_series = []
    all_labels = []
    
    for idx, pq_file in enumerate(parquet_files[:3]):
        temp_df = pd.read_parquet(pq_file)
        first_id = temp_df['series_id'].iloc[0]
        
        all_series.append(first_id)
        
        # Store df reference for comparison
        if idx == 0:
            multi_df = temp_df.copy()
        
        label = temp_df['label'].iloc[0] if 'label' in temp_df.columns else pq_file.stem
        all_labels.append(label)
        
        print(f"  - {pq_file.name}: {label}")
    
    # Note: For actual multi-dataset comparison, you'd need to combine DataFrames
    # Here we just demonstrate the concept
    print("\n  Note: Full multi-dataset comparison requires combining DataFrames")
    print("        with unique series IDs. See documentation for details.")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("Visualization Summary")
print("=" * 80)
print(f"\nAll visualizations saved to: {output_dir.absolute()}")
print("\nGenerated plots:")
plot_files = list(output_dir.glob("*.png"))
for idx, plot_file in enumerate(plot_files, 1):
    print(f"  {idx}. {plot_file.name}")

print("\n" + "=" * 80)
print("Example Complete!")
print("=" * 80)
print("\nNext steps:")
print("  1. Open the visualization_output folder to view all plots")
print("  2. Run 'python examples/analyze_datasets.py' for statistical analysis")
print("  3. Modify this script to visualize your own generated datasets")
print("=" * 80)

