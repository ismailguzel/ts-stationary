"""
Helper utilities for dataset generation.

This module contains utility functions used across the dataset generation package.
"""

import os
import pandas as pd


def save_and_cleanup(all_dfs, folder, count, label):
    """
    Save combined DataFrame to Parquet and cleanup empty directory.

    Parameters
    ----------
    all_dfs : list of pd.DataFrame
        List of DataFrames to combine
    folder : str
        Folder path where data was temporarily stored
    count : int
        Number of series generated
    label : str
        Label for the dataset

    Returns
    -------
    None
    """
    if not all_dfs:
        print(f"No data generated for {folder}")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    category_label = os.path.basename(folder) 
    parent_folder = os.path.dirname(folder)   
    
    output_filename = f"{category_label}.parquet"
    output_path = os.path.join(parent_folder, output_filename)

    combined_df.to_parquet(output_path, index=False)

    try:
        os.rmdir(folder)
    except OSError as e:
        print(f"Warning: Could not remove empty directory {folder}: {e}")

    print(f"{count} '{label}' series saved in ONE file: '{output_path}'")


def get_length_label(length_range):
    """
    Get a label for the length range.

    Parameters
    ----------
    length_range : tuple
        (min, max) length range

    Returns
    -------
    str
        'short', 'medium', or 'long'
    """
    if length_range == (50, 100):
        return "short"
    elif length_range == (300, 500):
        return "medium"
    else:
        return "long"

