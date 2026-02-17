"""
Configurable Dataset Generation Script
========================================

This script generates time series datasets based on a flexible configuration system.

USAGE:
    1. Modify the CONFIG dictionary below to enable/disable dataset types
    2. Run: python dataset_generation.py
    3. Generated datasets will be saved in the 'generated-dataset' directory

QUICK START SCENARIOS:
    
    Scenario 1: Research Focused (Current Default)
    -----------------------------------------------
    - No seasonality
    - No contextual anomalies
    - Only long series for multiple structural breaks
    
    Already configured below!
    
    
    Scenario 2: Full Dataset Generation
    ------------------------------------
    To generate ALL dataset types, change these in CONFIG:
    
    'seasonality': {
        'enabled': True,
        'types': ['single', 'multiple', 'sarma', 'sarima'],
        'length_ranges': ['short', 'medium']  # Exclude 'long' for sarma/sarima
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
    
    Add 'single' sections to structural breaks with all length ranges
    
    
    Scenario 3: Only Stationary & Trends (Quick Test)
    --------------------------------------------------
    Set 'enabled': False for all categories except:
    - 'stationary'
    - 'deterministic_trends'
    
    
    Scenario 4: Anomaly Detection Focus
    ------------------------------------
    Set 'enabled': True only for:
    - 'stationary' (as base)
    - 'point_anomalies'
    - 'collective_anomalies'
    - 'contextual_anomalies'
    
CONFIGURATION:
    Modify the CONFIG dictionary below to control what gets generated.
"""

from pathlib import Path
import random
import numpy as np

# ============================================================================
# CONFIGURATION - Modify these settings to control dataset generation
# ============================================================================

CONFIG = {
    # Random seed for reproducibility
    'random_seed': 42,
    
    # Output directory
    'output_dir': 'generated-dataset',
    
    # Number of series to generate per configuration
    'count': 1,

    # Whether to include indices of anomalies/structural breaks in the output DataFrames
    'include_indices': True,  # Set to False to exclude indices columns
    
    # Dataset types to generate
    'stationary': {
        'enabled': True,
        'length_ranges': ['short', 'medium', 'long']  # Options: 'short', 'medium', 'long'
    },
    
    'deterministic_trends': {
        'enabled': True,
        'length_ranges': ['short', 'medium', 'long']
    },
    
    'point_anomalies': {
        'enabled': True,
        'single': {
            'enabled': True,
            'length_ranges': ['short', 'medium', 'long']
        },
        'multiple': {
            'enabled': True,
            'length_ranges': ['short', 'medium', 'long']
        }
    },
    
    'collective_anomalies': {
        'enabled': True,
        'single': {
            'enabled': True,
            'length_ranges': ['short', 'medium', 'long']
        },
        'multiple': {
            'enabled': True,
            'length_ranges': ['long']  # Only long series for multiple cases
        }
    },
    
    'contextual_anomalies': {
        'enabled': True,  # Disabled for research focused scenario
        'single': {
            'enabled': True,
            'length_ranges': ['short', 'medium', 'long']
        },
        'multiple': {
            'enabled': True,
            'length_ranges': ['long']
        }
    },
    
    'stochastic': {
        'enabled': True,
        'length_ranges': ['short', 'medium', 'long']
    },
    
    'volatility': {
        'enabled': True,
        'length_ranges': ['short', 'medium', 'long']
    },
    
    'seasonality': {
        'enabled': True,  # Disabled for research focused scenario
        'types': ['single', 'multiple', 'sarma', 'sarima'],  # Options: 'single', 'multiple', 'sarma', 'sarima'
        'length_ranges': ['short', 'medium']
    },
    
    'structural_breaks': {
        'mean_shift': {
            'enabled': True,
            'multiple': { 
                'enabled': True,
                'length_ranges': ['long']  # Only long series
            },
            'single': {
                'enabled': True,
                'length_ranges': ['short', 'medium', 'long']
            }
        },
        'variance_shift': {
            'enabled': True,
            'multiple': { 
                'enabled': True,
                'length_ranges': ['long']  # Only long series
            },
            'single': {
                'enabled': True,
                'length_ranges': ['short', 'medium', 'long']
            }
        },
        'trend_shift': {
            'enabled': True,
            'multiple': { 
                'enabled': True,
                'length_ranges': ['long']  # Only long series
            },
            'single': {
                'enabled': True,
                'length_ranges': ['short', 'medium', 'long']
            }
        }
    }
}

# ============================================================================
# Helper functions
# ============================================================================

def get_length_range(length_key):
    """Convert length key to actual range tuple."""
    ranges = {
        'short': (50, 100),
        'medium': (300, 500),
        'long': (1000, 10000)
    }
    return ranges.get(length_key, (50, 100))

def is_enabled(*path):
    """Check if a configuration path is enabled."""
    config = CONFIG
    for key in path:
        if key not in config:
            return False
        config = config[key]
        if not isinstance(config, dict):
            return False
    return config.get('enabled', False)

def get_length_ranges(*path):
    """Get enabled length ranges for a configuration path."""
    config = CONFIG
    for key in path:
        if key not in config:
            return []
        config = config[key]
    
    ranges = config.get('length_ranges', [])
    return [(key, get_length_range(key)) for key in ranges]

# ============================================================================
# Setup
# ============================================================================

# Set random seeds for reproducibility
RANDOM_SEED = CONFIG['random_seed']
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Modular library imports
from timeseries_dataset_generator import TimeSeriesGenerator
from timeseries_dataset_generator.generators import (
    # Stationary
    generate_ar_dataset,
    generate_ma_dataset,
    generate_arma_dataset,
    generate_wn_dataset,
    # Trends
    generate_linear_trend_dataset,
    generate_quadratic_trend_dataset,
    generate_cubic_trend_dataset,
    generate_exponential_trend_dataset,
    generate_damped_trend_dataset,
    # Stochastic
    generate_random_walk_dataset,
    generate_random_walk_with_drift_dataset,
    generate_ari_dataset,
    generate_ima_dataset,
    generate_arima_dataset,
    # Volatility
    generate_arch_dataset,
    generate_garch_dataset,
    generate_egarch_dataset,
    generate_aparch_dataset,
    # Seasonality
    generate_single_seasonality_dataset,
    generate_multiple_seasonality_dataset,
    generate_sarma_dataset,
    generate_sarima_dataset,
    # Anomalies
    generate_point_anomaly_dataset,
    generate_collective_anomaly_dataset,
    generate_contextual_anomaly_dataset,
    # Structural Breaks
    generate_mean_shift_dataset,
    generate_variance_shift_dataset,
    generate_trend_shift_dataset,
)

# ============================================================================
# Path and constant setup
# ============================================================================

def ensure_base_dir(base_dir: Path) -> Path:
    """Create base directory if it doesn't exist."""
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


BASE_OUTPUT_DIR = ensure_base_dir(Path(CONFIG['output_dir']))


def folder_path(*parts: str) -> str:
    """Create folder path and ensure it exists."""
    path = ensure_base_dir(BASE_OUTPUT_DIR.joinpath(*parts))
    return str(path)


# Constants
bases = ["ar", "ma", "arma", "white_noise"]
signs = [1, -1]
locations = ["beginning", "middle", "end"]
change_types = ['direction_change', 'magnitude_change', 'direction_and_magnitude_change']

# ============================================================================
# Dataset Generation Functions
# ============================================================================

print("="*70)
print("DATASET GENERATION STARTING")
print("="*70)
print(f"Configuration: Research Focused Scenario")
print(f"  - Seasonality: {'✓' if is_enabled('seasonality') else '✗ (disabled)'}")
print(f"  - Contextual Anomalies: {'✓' if is_enabled('contextual_anomalies') else '✗ (disabled)'}")
print(f"  - Multiple Structural Breaks: Only long series")
print("="*70)
print() 

#### STATIONARY SERIES

if is_enabled('stationary'):
    print("✓ Generating: Stationary series")
    length_ranges_list = get_length_ranges('stationary')
    
    generators = {
        "ar": generate_ar_dataset,
        "ma": generate_ma_dataset,
        "arma": generate_arma_dataset,
        "white_noise": generate_wn_dataset
    }
    
    for base in bases:
        for length_label, length_range in length_ranges_list:
            generators[base](
                TimeSeriesGenerator,
                folder=folder_path("stationary", base, length_label), 
                count=CONFIG['count'], 
                length_range=length_range,
                is_loc=CONFIG['include_indices']
            )
    print()
else:
    print("⊗ Skipping: Stationary series (disabled)\n")




#### DETERMINISTIC TRENDS

if is_enabled('deterministic_trends'):
    print("✓ Generating: Deterministic trends")
    length_ranges_list = get_length_ranges('deterministic_trends')
    
    trend_generators = {
        'linear': generate_linear_trend_dataset,
        'quadratic': generate_quadratic_trend_dataset,
        'cubic': generate_cubic_trend_dataset,
        'exponential': generate_exponential_trend_dataset,
        'damped': generate_damped_trend_dataset
    }
    
    for trend_type, generator_func in trend_generators.items():
        for base in bases:
            for length_label, length_range in length_ranges_list:
                for sign in signs:
                    direction = "up" if sign == 1 else "down"
                    generator_func(
                        TimeSeriesGenerator,
                        folder=folder_path(f"deterministic_trend_{trend_type}", direction, base, length_label),
                        kind=base,
                        count=CONFIG['count'],
                        length_range=length_range,
                        sign=sign,
                        is_loc=CONFIG['include_indices']
                    )
    print()
else:
    print("⊗ Skipping: Deterministic trends (disabled)\n")



#### POINT ANOMALIES

if is_enabled('point_anomalies'):
    print("✓ Generating: Point anomalies")
    
    # Single point anomalies
    if is_enabled('point_anomalies', 'single'):
        length_ranges_list = get_length_ranges('point_anomalies', 'single')
        for base in bases:
            for length_label, length_range in length_ranges_list:
                for loc in locations:
                    generate_point_anomaly_dataset(
                        TimeSeriesGenerator,
                        folder=folder_path("point_anomaly_single", base, length_label),
                        kind=base,
                        count=CONFIG['count'],
                        length_range=length_range,
                        anomaly_type='single',
                        location=loc,
                        is_loc=CONFIG['include_indices']
                    )
    
    # Multiple point anomalies
    if is_enabled('point_anomalies', 'multiple'):
        length_ranges_list = get_length_ranges('point_anomalies', 'multiple')
        for base in bases:
            for length_label, length_range in length_ranges_list:
                generate_point_anomaly_dataset(
                    TimeSeriesGenerator,
                    folder=folder_path("point_anomaly_multiple", base, length_label),
                    kind=base,
                    count=CONFIG['count'],
                    length_range=length_range,
                    anomaly_type='multiple',
                    is_loc=CONFIG['include_indices']
                )
    print()
else:
    print("⊗ Skipping: Point anomalies (disabled)\n")


#### COLLECTIVE ANOMALIES

if is_enabled('collective_anomalies'):
    print("✓ Generating: Collective anomalies")

    # Single collective anomalies
    if is_enabled('collective_anomalies', 'single'):
        length_ranges_list = get_length_ranges('collective_anomalies', 'single')
        for base in bases:
            for length_label, length_range in length_ranges_list:
                for loc in locations:
                    generate_collective_anomaly_dataset(
                        TimeSeriesGenerator,
                        folder=folder_path("collective_anomaly_single", base, length_label),
                        kind=base,
                        count=CONFIG['count'],
                        length_range=length_range,
                        anomaly_type='single',
                        location=loc,
                        is_loc=CONFIG['include_indices']
                    )
    
    # Multiple collective anomalies
    if is_enabled('collective_anomalies', 'multiple'):
        length_ranges_list = get_length_ranges('collective_anomalies', 'multiple')
        for base in bases:
            for length_label, length_range in length_ranges_list:
                n = random.randint(2, 4)
                generate_collective_anomaly_dataset(
                    TimeSeriesGenerator,
                    folder=folder_path("multi_collective_anomaly", base, length_label),
                    kind=base,
                    count=CONFIG['count'],
                    num_anomalies=n,
                    anomaly_type='multiple',
                    length_range=length_range,
                    is_loc=CONFIG['include_indices']
                )
    print()
else:
    print("⊗ Skipping: Collective anomalies (disabled)\n")


#### CONTEXTUAL ANOMALIES

if is_enabled('contextual_anomalies'):
    print("✓ Generating: Contextual anomalies")
    
    # Single contextual anomalies
    if is_enabled('contextual_anomalies', 'single'):
        length_ranges_list = get_length_ranges('contextual_anomalies', 'single')
        for length_label, length_range in length_ranges_list:
            for loc in locations:
                generate_contextual_anomaly_dataset(
                    TimeSeriesGenerator,
                    folder=folder_path("contextual_anomaly", length_label),
                    count=CONFIG['count'],
                    length_range=length_range,
                    location=loc,
                    is_loc=CONFIG['include_indices']
                )
    
    # Multiple contextual anomalies
    if is_enabled('contextual_anomalies', 'multiple'):
        length_ranges_list = get_length_ranges('contextual_anomalies', 'multiple')
        for length_label, length_range in length_ranges_list:
            n = random.randint(2, 4)
            generate_contextual_anomaly_dataset(
                TimeSeriesGenerator,
                folder=folder_path("multi_contextual_anomaly", length_label),
                count=CONFIG['count'],
                num_anomalies=n,
                anomaly_type='multiple',
                length_range=length_range,
                is_loc=CONFIG['include_indices']
            )
    print()
else:
    print("⊗ Skipping: Contextual anomalies (disabled)\n")






#### STOCHASTIC SERIES

if is_enabled('stochastic'):
    print("✓ Generating: Stochastic series")
    length_ranges_list = get_length_ranges('stochastic')
    
    stochastic_bases = ["random_walk", "random_walk_drift", "ari", "ima", "arima"]
    stochastic_generators = {
        "random_walk": generate_random_walk_dataset,
        "random_walk_drift": generate_random_walk_with_drift_dataset,
        "ari": generate_ari_dataset,
        "ima": generate_ima_dataset,
        "arima": generate_arima_dataset
    }
    
    for base in stochastic_bases:
        for length_label, length_range in length_ranges_list:
            stochastic_generators[base](
                TimeSeriesGenerator,
                folder=folder_path("stochastic", base, length_label), 
                count=CONFIG['count'], 
                length_range=length_range,
                is_loc=CONFIG['include_indices']
            )
    print()
else:
    print("⊗ Skipping: Stochastic series (disabled)\n")


#### VOLATILITY SERIES

if is_enabled('volatility'):
    print("✓ Generating: Volatility series")
    length_ranges_list = get_length_ranges('volatility')
    
    volatility_bases = ["arch", "garch", "egarch", "aparch"]
    volatility_generators = {
        "arch": generate_arch_dataset,
        "garch": generate_garch_dataset,
        "egarch": generate_egarch_dataset,
        "aparch": generate_aparch_dataset
    }
    
    for base in volatility_bases:
        for length_label, length_range in length_ranges_list:
            volatility_generators[base](
                TimeSeriesGenerator,
                folder=folder_path("volatility", base, length_label), 
                count=CONFIG['count'], 
                length_range=length_range,
                is_loc=CONFIG['include_indices']
            )
    print()
else:
    print("⊗ Skipping: Volatility series (disabled)\n")


#### SEASONALITY

if is_enabled('seasonality'):
    print("✓ Generating: Seasonality")
    length_ranges_list = get_length_ranges('seasonality')
    seasonality_types_to_gen = CONFIG['seasonality'].get('types', [])
    
    for seasonality_type in seasonality_types_to_gen:
        for length_label, length_range in length_ranges_list:
            if seasonality_type == "single":
                generate_single_seasonality_dataset(
                    TimeSeriesGenerator,
                    folder=folder_path("single_seasonality", length_label),
                    count=CONFIG['count'],
                    length_range=length_range,
                    is_loc=CONFIG['include_indices']
                )
            elif seasonality_type == "multiple":
                generate_multiple_seasonality_dataset(
                    TimeSeriesGenerator,
                    folder=folder_path("multiple_seasonality", length_label),
                    count=CONFIG['count'],
                    length_range=length_range,
                    is_loc=CONFIG['include_indices']
                )
            elif seasonality_type == "sarma":
                generate_sarma_dataset(
                    TimeSeriesGenerator,
                    folder=folder_path("sarma_seasonality", length_label),
                    count=CONFIG['count'],
                    length_range=length_range,
                    is_loc=CONFIG['include_indices']
                )
            elif seasonality_type == "sarima":
                generate_sarima_dataset(
                    TimeSeriesGenerator,
                    folder=folder_path("sarima_seasonality", length_label),
                    count=CONFIG['count'],
                    length_range=length_range,
                    is_loc=CONFIG['include_indices']
                )
    print()
else:
    print("⊗ Skipping: Seasonality (disabled)\n")



#### STRUCTURAL BREAKS

print("✓ Generating: Structural breaks")

# Mean Shift
if is_enabled('structural_breaks', 'mean_shift'):
    print("  - Mean shift")
    if is_enabled('structural_breaks', 'mean_shift', 'single'):
        length_ranges_list = get_length_ranges('structural_breaks', 'mean_shift', 'single')
        for base in bases:
            for length_label, length_range in length_ranges_list:
                n = 1
                generate_mean_shift_dataset(
                    TimeSeriesGenerator,
                    folder=folder_path("mean_shift", base, length_label),
                    kind=base,
                    count=CONFIG['count'],
                    break_type='single',
                    length_range=length_range,
                    is_loc=CONFIG['include_indices']
                )

    if is_enabled('structural_breaks', 'mean_shift', 'multiple'):
        length_ranges_list = get_length_ranges('structural_breaks', 'mean_shift', 'multiple')
        for base in bases:
            for length_label, length_range in length_ranges_list:
                n = random.randint(2, 4)
                generate_mean_shift_dataset(
                    TimeSeriesGenerator,
                    folder=folder_path("multi_mean_shift", base, length_label),
                    kind=base,
                    count=CONFIG['count'],
                    num_breaks=n,
                    break_type='multiple',
                    length_range=length_range,
                    is_loc=CONFIG['include_indices']
                )

# Variance Shift
if is_enabled('structural_breaks', 'variance_shift'):
    print("  - Variance shift")
    if is_enabled('structural_breaks', 'variance_shift', 'single'):
        length_ranges_list = get_length_ranges('structural_breaks', 'variance_shift', 'single')
        for base in bases:
            for length_label, length_range in length_ranges_list:
                n = 1
                generate_variance_shift_dataset(
                    TimeSeriesGenerator,
                    folder=folder_path("variance_shift", base, length_label),
                    kind=base,
                    count=CONFIG['count'],
                    break_type='single',
                    length_range=length_range,
                    is_loc=CONFIG['include_indices']
                )

    if is_enabled('structural_breaks', 'variance_shift', 'multiple'):
        length_ranges_list = get_length_ranges('structural_breaks', 'variance_shift', 'multiple')
        for base in bases:
            for length_label, length_range in length_ranges_list:
                n = random.randint(2, 4)
                generate_variance_shift_dataset(
                    TimeSeriesGenerator,
                    folder=folder_path("multi_variance_shift", base, length_label),
                    kind=base,
                    count=CONFIG['count'],
                    num_breaks=n,
                    break_type='multiple',
                    length_range=length_range,
                    is_loc=CONFIG['include_indices']
                )

# Trend Shift
if is_enabled('structural_breaks', 'trend_shift'):
    print("  - Trend shift")
    if is_enabled('structural_breaks', 'trend_shift', 'single'):
        length_ranges_list = get_length_ranges('structural_breaks', 'trend_shift', 'single')
        for base in bases:
            for length_label, length_range in length_ranges_list:
                for sign in signs:
                    for change_type in change_types:
                        for loc in locations:
                            generate_trend_shift_dataset(
                                TimeSeriesGenerator,
                                folder=folder_path("trend_shift", base, length_label),
                                kind=base,
                                count=CONFIG['count'],
                                change_types=[change_type],
                                sign=sign,
                                break_type='single',
                                length_range=length_range,
                                location=loc,
                                is_loc=CONFIG['include_indices']
                            )

    if is_enabled('structural_breaks', 'trend_shift', 'multiple'):
        length_ranges_list = get_length_ranges('structural_breaks', 'trend_shift', 'multiple')
        for base in bases:
            for length_label, length_range in length_ranges_list:
                for sign in signs:
                    n = random.randint(2, 4)
                    change_type_samples = random.choices(change_types, k=n)
                    generate_trend_shift_dataset(
                        TimeSeriesGenerator,
                        folder=folder_path("multi_trend_shift", base, length_label),
                        kind=base,
                        count=CONFIG['count'],
                        num_breaks=n,
                        change_types=change_type_samples,
                        sign=sign,
                        break_type='multiple',
                        length_range=length_range,
                        is_loc=CONFIG['include_indices']
                    )

print()
print("="*70)
print("DATASET GENERATION COMPLETE!")
print(f"Output directory: {BASE_OUTPUT_DIR.absolute()}")
print("="*70)

