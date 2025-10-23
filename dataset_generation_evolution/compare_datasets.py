#!/usr/bin/env python3
"""
Compare two generated dataset directories for reproducibility check.
"""

import pandas as pd
from pathlib import Path
import numpy as np

print("=" * 80)
print("Dataset Reproducibility Comparison")
print("=" * 80)

dir1 = Path("generated-dataset-test1")
dir2 = Path("generated-dataset-test2")

# Get all parquet files
files1 = sorted(dir1.rglob("*.parquet"))
files2 = sorted(dir2.rglob("*.parquet"))

print(f"\nFiles in {dir1.name}: {len(files1)}")
print(f"Files in {dir2.name}: {len(files2)}")

if len(files1) != len(files2):
    print("\nWARNING: Different number of files!")
else:
    print("\nFile count: MATCH")

# Compare file structure
print("\n" + "=" * 80)
print("File Structure Comparison")
print("=" * 80)

rel_paths1 = {f.relative_to(dir1) for f in files1}
rel_paths2 = {f.relative_to(dir2) for f in files2}

if rel_paths1 == rel_paths2:
    print("File structure: IDENTICAL")
else:
    print("File structure: DIFFERENT")
    only_in_1 = rel_paths1 - rel_paths2
    only_in_2 = rel_paths2 - rel_paths1
    if only_in_1:
        print(f"\nOnly in {dir1.name}:")
        for p in sorted(only_in_1):
            print(f"  {p}")
    if only_in_2:
        print(f"\nOnly in {dir2.name}:")
        for p in sorted(only_in_2):
            print(f"  {p}")

# Compare file contents
print("\n" + "=" * 80)
print("Data Comparison (sampling 10 random files)")
print("=" * 80)

# Sample some files to compare
import random
random.seed(42)
sample_files = random.sample(list(rel_paths1 & rel_paths2), min(10, len(rel_paths1)))

identical_files = 0
different_files = 0
differences = []

for rel_path in sorted(sample_files):
    file1 = dir1 / rel_path
    file2 = dir2 / rel_path
    
    try:
        df1 = pd.read_parquet(file1)
        df2 = pd.read_parquet(file2)
        
        # Compare shapes
        if df1.shape != df2.shape:
            differences.append({
                'file': str(rel_path),
                'issue': 'Different shapes',
                'details': f"{df1.shape} vs {df2.shape}"
            })
            different_files += 1
            continue
        
        # Compare data values
        numerical_cols = df1.select_dtypes(include=[np.number]).columns
        are_equal = True
        diff_details = []
        
        for col in numerical_cols:
            if not np.allclose(df1[col], df2[col], rtol=1e-10, atol=1e-10, equal_nan=True):
                are_equal = False
                max_diff = np.abs(df1[col] - df2[col]).max()
                diff_details.append(f"{col}: max_diff={max_diff:.2e}")
        
        # Compare non-numerical columns
        for col in df1.columns:
            if col not in numerical_cols:
                if not df1[col].equals(df2[col]):
                    are_equal = False
                    diff_details.append(f"{col}: values differ")
        
        if are_equal:
            print(f"  {rel_path}: IDENTICAL")
            identical_files += 1
        else:
            print(f"  {rel_path}: DIFFERENT")
            differences.append({
                'file': str(rel_path),
                'issue': 'Data values differ',
                'details': '; '.join(diff_details)
            })
            different_files += 1
            
    except Exception as e:
        print(f"  {rel_path}: ERROR - {e}")
        differences.append({
            'file': str(rel_path),
            'issue': 'Comparison error',
            'details': str(e)
        })
        different_files += 1

# Full comparison of a few files
print("\n" + "=" * 80)
print("Detailed Comparison (first 3 files)")
print("=" * 80)

for rel_path in list(sorted(rel_paths1 & rel_paths2))[:3]:
    file1 = dir1 / rel_path
    file2 = dir2 / rel_path
    
    print(f"\n{rel_path}:")
    df1 = pd.read_parquet(file1)
    df2 = pd.read_parquet(file2)
    
    print(f"  Shape: {df1.shape} vs {df2.shape}")
    print(f"  Columns: {list(df1.columns)[:5]}...")
    
    # Check first few data values
    if 'data' in df1.columns:
        data1 = df1['data'].head(5).values
        data2 = df2['data'].head(5).values
        print(f"  First 5 data values (file1): {data1}")
        print(f"  First 5 data values (file2): {data2}")
        print(f"  Are equal: {np.allclose(data1, data2)}")

# Summary
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(f"Total files compared: {identical_files + different_files}")
print(f"Identical files: {identical_files}")
print(f"Different files: {different_files}")

if different_files == 0:
    print("\nRESULT: PERFECT REPRODUCIBILITY!")
    print("Both runs produced identical datasets.")
else:
    print(f"\nRESULT: REPRODUCIBILITY ISSUES FOUND")
    print(f"\nDifferences found in {len(differences)} files:")
    for diff in differences[:5]:  # Show first 5
        print(f"\n  File: {diff['file']}")
        print(f"  Issue: {diff['issue']}")
        print(f"  Details: {diff['details']}")
    if len(differences) > 5:
        print(f"\n  ... and {len(differences) - 5} more")

print("=" * 80)

