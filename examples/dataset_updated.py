"""
Updated dataset generation script using the new modular library.

This script demonstrates how to use the new timeseries_dataset_generator
library to generate comprehensive time series datasets.
"""

from pathlib import Path
import random
import numpy as np

# Set random seeds for reproducibility
RANDOM_SEED = 42
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


def ensure_base_dir(base_dir: Path) -> Path:
    """Create base directory if it doesn't exist."""
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


BASE_OUTPUT_DIR = ensure_base_dir(Path("generated-dataset"))


def folder_path(*parts: str) -> str:
    """Create folder path and ensure it exists."""
    path = ensure_base_dir(BASE_OUTPUT_DIR.joinpath(*parts))
    return str(path)


bases = ["ar", "ma", "arma", "white_noise"]
length_ranges = [(50, 100), (300, 500), (1000, 10000)]
signs = [1, -1]
locations = ["beginning", "middle", "end"] 

#### STATIONARY SERIES

for base in bases:
    for length_range in length_ranges:
        
        if length_range == (50,100):
            l = "short"
        elif length_range == (300,500):
            l = "medium"
        else:
            l = "long"
        
        if base == "ar":
            generate_ar_dataset(
                TimeSeriesGenerator,  # YENİ: TimeSeriesGenerator ekledik
                folder=folder_path("stationary", base, l), 
                count=10, 
                length_range=length_range
            )
        elif base == "ma":
            generate_ma_dataset(
                TimeSeriesGenerator,
                folder=folder_path("stationary", base, l), 
                count=10, 
                length_range=length_range
            )
        elif base == "arma":
            generate_arma_dataset(
                TimeSeriesGenerator,
                folder=folder_path("stationary", base, l), 
                count=10, 
                length_range=length_range
            )
        elif base == "white_noise":
            generate_wn_dataset(
                TimeSeriesGenerator,
                folder=folder_path("stationary", base, l), 
                count=10, 
                length_range=length_range
            )




######linear trend
for base in bases:
    for length_range in length_ranges:
        for sign in signs:
            d = "up" if sign == 1 else "down"
            if length_range == (50,100):
              l = "short"
            elif length_range == (300,500):
              l = "medium"
            else:
              l = "long"

            generate_linear_trend_dataset(
                TimeSeriesGenerator,
                folder=folder_path("deterministic_trend_linear", d, base, l),
                kind=base,
                count=10,
                length_range=length_range,
                sign=sign
            )

######quadratic trend
for base in bases:
    for length_range in length_ranges:
        for sign in signs:
            d = "up" if sign == 1 else "down"
            if length_range == (50,100):
              l = "short"
            elif length_range == (300,500):
              l = "medium"
            else:
              l = "long"

            generate_quadratic_trend_dataset(
                TimeSeriesGenerator,
                folder=folder_path("deterministic_trend_quadratic", base, l),
                kind=base,
                count=10,
                length_range=length_range,
                sign=sign
            )


#######cubic trend
for base in bases:
    for length_range in length_ranges:
        for sign in signs:
            d = "up" if sign == 1 else "down"
            if length_range == (50,100):
              l = "short"
            elif length_range == (300,500):
              l = "medium"
            else:
              l = "long"

            generate_cubic_trend_dataset(
                TimeSeriesGenerator,
                folder=folder_path("deterministic_trend_cubic", base, l),
                kind=base,
                count=10,
                length_range=length_range,
                sign=sign
            )




########exponential trend
for base in bases:
    for length_range in length_ranges:
        for sign in signs:
            d = "up" if sign == 1 else "down"
            if length_range == (50,100):
              l = "short"
            elif length_range == (300,500):
              l = "medium"
            else:
              l = "long"

            generate_exponential_trend_dataset(
                TimeSeriesGenerator,
                folder=folder_path("deterministic_trend_exponential", base, l),
                kind=base,
                count=10,
                length_range=length_range,
                sign=sign
            )




#######damped trend
for base in bases:
    for length_range in length_ranges:
        for sign in signs:
            d = "up" if sign == 1 else "down"
            if length_range == (50,100):
              l = "short"
            elif length_range == (300,500):
              l = "medium"
            else:
              l = "long"

            generate_damped_trend_dataset(
                TimeSeriesGenerator,
                folder=folder_path("deterministic_trend_damped", base, l),
                kind=base,
                count=10,
                length_range=length_range,
                sign=sign
            )



#######point anomaly
#single point anomalies
for base in bases:
    for length_range in length_ranges:
        if length_range == (50,100):
            l = "short"
        elif length_range == (300,500):
            l = "medium"
        else:
            l = "long"
        for loc in locations:
            generate_point_anomaly_dataset(
                TimeSeriesGenerator,
                folder=folder_path("point_anomaly_single", base, l),
                kind=base,
                count=10,
                length_range=length_range,
                anomaly_type='single',
                location=loc
            )

#multi point anomalies
for base in bases:
    for length_range in length_ranges:
        if length_range == (50,100):
            l = "short"
        elif length_range == (300,500):
            l = "medium"
        else:
            l = "long"
        generate_point_anomaly_dataset(
            TimeSeriesGenerator,
            folder=folder_path("point_anomaly_multiple", base, l),
            kind=base,
            count=10,
            length_range=length_range,
            anomaly_type='multiple'
        )




######collective anomaly
for base in bases:
    for length_range in length_ranges:
        if length_range == (50,100):
            l = "short"
        elif length_range == (300,500):
            l = "medium"
        else:
            l = "long"
        for loc in locations:
            generate_collective_anomaly_dataset(
                TimeSeriesGenerator,
                folder=folder_path("collective_anomaly", base, l),
                kind=base,
                count=10,
                length_range=length_range,
                location=loc
            )


for base in bases:
    for length_range in [(1000, 10000)]:
        if length_range == (50,100):
            l = "short"
        elif length_range == (300,500):
            l = "medium"
        else:
            l = "long"
        n = random.randint(2,4)
        generate_collective_anomaly_dataset(
            TimeSeriesGenerator,
            folder=folder_path("multi_collective_anomaly", base, l),
            kind=base,
            count=10,
            num_anomalies=n,
            anomaly_type='multiple',
            length_range=length_range,
        )



#####contextual anomaly
for length_range in length_ranges:
    if length_range == (50,100):
        l = "short"
    elif length_range == (300,500):
        l = "medium"
    else:
        l = "long"
    for loc in locations:
        generate_contextual_anomaly_dataset(
            TimeSeriesGenerator,
            folder=folder_path("contextual_anomaly", l),
            count=10,
            length_range=length_range,
            location=loc
        )


for length_range in [(1000, 10000)]:
    if length_range == (50,100):
        l = "short"
    elif length_range == (300,500):
        l = "medium"
    else:
        l = "long"
    n = random.randint(2,4)
    generate_contextual_anomaly_dataset(
        TimeSeriesGenerator,
        folder=folder_path("multi_contextual_anomaly", l),
        count=10,
        num_anomalies=n,
        anomaly_type='multiple',
        length_range=length_range,
    )



#### STOCHASTIC SERIES ####

stochastic_bases = ["random_walk", "random_walk_drift", "ari", "ima", "arima"]

for base in stochastic_bases:
    for length_range in length_ranges:

        if length_range == (50,100):
            l = "short"
        elif length_range == (300,500):
            l = "medium"
        else:
            l = "long"

        if base == "random_walk":
            generate_random_walk_dataset(
                TimeSeriesGenerator,
                folder=folder_path("stochastic", base, l), 
                count=10, 
                length_range=length_range
            )
        elif base == "random_walk_drift":
            generate_random_walk_with_drift_dataset(
                TimeSeriesGenerator,
                folder=folder_path("stochastic", base, l), 
                count=10, 
                length_range=length_range
            )
        elif base == "ari":
            generate_ari_dataset(
                TimeSeriesGenerator,
                folder=folder_path("stochastic", base, l), 
                count=10, 
                length_range=length_range
            )
        elif base == "ima":
            generate_ima_dataset(
                TimeSeriesGenerator,
                folder=folder_path("stochastic", base, l), 
                count=10, 
                length_range=length_range
            )
        elif base == "arima":
            generate_arima_dataset(
                TimeSeriesGenerator,
                folder=folder_path("stochastic", base, l), 
                count=10, 
                length_range=length_range
            )

#### VOLATILITY SERIES ####

volatility_bases = ["arch", "garch", "egarch", "aparch"]

for base in volatility_bases:
    for length_range in length_ranges:

        if length_range == (50,100):
            l = "short"
        elif length_range == (300,500):
            l = "medium"
        else:
            l = "long"

        if base == "arch":
            generate_arch_dataset(
                TimeSeriesGenerator,
                folder=folder_path("volatility", base, l), 
                count=10, 
                length_range=length_range
            )
        elif base == "garch":
            generate_garch_dataset(
                TimeSeriesGenerator,
                folder=folder_path("volatility", base, l), 
                count=10, 
                length_range=length_range
            )
        elif base == "egarch":
            generate_egarch_dataset(
                TimeSeriesGenerator,
                folder=folder_path("volatility", base, l), 
                count=10, 
                length_range=length_range
            )
        elif base == "aparch":
            generate_aparch_dataset(
                TimeSeriesGenerator,
                folder=folder_path("volatility", base, l), 
                count=10, 
                length_range=length_range
            )



###### seasonality ######

seasonality_types = ['single', 'multiple', 'sarma', 'sarima']

for seasonality in seasonality_types:
    for length_range in length_ranges:
        if length_range == (50,100):
            l = "short"
        elif length_range == (300,500):
            l = "medium"
        else:
            l = "long"

        if seasonality == 'single':
            generate_single_seasonality_dataset(
                TimeSeriesGenerator,
                folder=folder_path("single_seasonality", l), 
                count = 10, 
                length_range=length_range
            )
        if seasonality == 'multiple':
            generate_multiple_seasonality_dataset(
                TimeSeriesGenerator,
                folder=folder_path("multiple_seasonality", l), 
                count = 10, 
                length_range=length_range
            )
        if seasonality == 'sarma':
            if l == 'long':
                pass
            else:
                generate_sarma_dataset(
                    TimeSeriesGenerator,
                    folder=folder_path("sarma_seasonality", l), 
                    count = 10, 
                    length_range=length_range
                )
        if seasonality == 'sarima':
            if l == 'long':
                pass
            else:
                generate_sarima_dataset(
                    TimeSeriesGenerator,
                    folder=folder_path("sarima_seasonality", l), 
                    count = 10, 
                    length_range=length_range
                )



###### structural breaks ######

### mean shift ###

for base in bases:
    for length_range in length_ranges:
        if length_range == (50,100):
            l = "short"
        elif length_range == (300,500):
            l = "medium"
        else:
            l = "long"
        for loc in locations:
            for sign in signs:
                generate_mean_shift_dataset(
                    TimeSeriesGenerator,
                    folder=folder_path("mean_shift", base, l),
                    kind=base,
                    count=10,
                    signs = [sign],
                    length_range=length_range,
                    location=loc,
                    break_type = 'single',
                    num_breaks = 1)


for base in bases:
    for length_range in [(300,500), (1000, 10000)]:
        if length_range == (50,100):
            l = "short"
        elif length_range == (300,500):
            l = "medium"
        else:
            l = "long"
        n = random.randint(2,4)
        generate_mean_shift_dataset(
            TimeSeriesGenerator,
            folder=folder_path("multi_mean_shift", base, l),
            kind=base,
            count=10,
            num_breaks=n,
            break_type='multiple',
            length_range=length_range)
        

### variance shift ###

for base in bases:
    for length_range in length_ranges:
        if length_range == (50,100):
            l = "short"
        elif length_range == (300,500):
            l = "medium"
        else:
            l = "long"
        for loc in locations:
            for sign in signs:
                generate_variance_shift_dataset(
                    TimeSeriesGenerator,
                    folder=folder_path("variance_shift", base, l),
                    kind=base,
                    count=10,
                    signs = [sign],
                    length_range=length_range,
                    location=loc,
                    break_type = 'single',
                    num_breaks = 1)


for base in bases:
    for length_range in [(300,500), (1000, 10000)]:
        if length_range == (50,100):
            l = "short"
        elif length_range == (300,500):
            l = "medium"
        else:
            l = "long"
        n = random.randint(2,4)
        generate_variance_shift_dataset(
            TimeSeriesGenerator,
            folder=folder_path("multi_variance_shift", base, l),
            kind=base,
            count=10,
            num_breaks=n,
            break_type='multiple',
            length_range=length_range)


### trend shift ###

change_types = ['direction_change', 'magnitude_change', 'direction_and_magnitude_change'] 

for base in bases:
    for length_range in length_ranges:
        if length_range == (50,100):
            l = "short"
        elif length_range == (300,500):
            l = "medium"
        else:
            l = "long"
        for loc in locations:
            for sign in signs:
                for change_type in change_types:
                    generate_trend_shift_dataset(
                        TimeSeriesGenerator,
                        folder=folder_path("trend_shift", base, l),
                        kind=base,
                        count=10,
                        sign = sign,
                        change_types = [change_type],
                        length_range=length_range,
                        location=loc,
                        break_type = 'single',
                        num_breaks = 1)


for base in bases:
    for length_range in [(300,500), (1000, 10000)]:
        if length_range == (50,100):
            l = "short"
        elif length_range == (300,500):
            l = "medium"
        else:
            l = "long"
        for sign in signs:
            n = random.randint(2,4)
            change_type_samples = random.choices(change_types, k=n)
            generate_trend_shift_dataset(
                TimeSeriesGenerator,
                folder=folder_path("multi_trend_shift", base, l),
                kind=base,
                count=10,
                num_breaks=n,
                change_types = change_type_samples, 
                sign = sign,
                break_type='multiple',
                length_range=length_range)

