"""
Quick test script - checks if library structure is correct.
This doesn't require dependencies.
"""

import sys
from pathlib import Path

print("=" * 80)
print("Quick Structure Test - Time Series Dataset Generator")
print("=" * 80)

# Test 1: Check directory structure
print("\n1. Checking Directory Structure...")
base_dir = Path(__file__).parent / 'timeseries_dataset_generator'

required_dirs = [
    'core',
    'generators',
    'utils',
    'examples'
]

all_good = True
for dir_name in required_dirs:
    dir_path = base_dir / dir_name
    if dir_path.exists():
        print(f"    {dir_name}/ directory exists")
    else:
        print(f"    {dir_name}/ directory missing")
        all_good = False

# Test 2: Check core files
print("\n2. Checking Core Files...")
core_files = [
    'core/__init__.py',
    'core/generator.py',
    'core/metadata.py'
]

for file_path in core_files:
    full_path = base_dir / file_path
    if full_path.exists():
        size = full_path.stat().st_size
        print(f"    {file_path} ({size:,} bytes)")
    else:
        print(f"    {file_path} missing")
        all_good = False

# Test 3: Check generator modules
print("\n3. Checking Generator Modules...")
generator_files = [
    'generators/__init__.py',
    'generators/stationary.py',
    'generators/trends.py',
    'generators/stochastic.py',
    'generators/volatility.py',
    'generators/seasonality.py',
    'generators/anomalies.py',
    'generators/structural_breaks.py'
]

for file_path in generator_files:
    full_path = base_dir / file_path
    if full_path.exists():
        size = full_path.stat().st_size
        print(f"    {file_path} ({size:,} bytes)")
    else:
        print(f"    {file_path} missing")
        all_good = False

# Test 4: Check package files
print("\n4. Checking Package Files...")
package_files = [
    '__init__.py',
    'setup.py',
    'pyproject.toml',
    'requirements.txt',
    'README.md',
    'QUICKSTART.md',
    'MIGRATION_GUIDE.md'
]

for file_path in package_files:
    full_path = base_dir / file_path
    if full_path.exists():
        size = full_path.stat().st_size
        print(f"    {file_path} ({size:,} bytes)")
    else:
        print(f"    {file_path} missing")
        all_good = False

# Test 5: Check import structure (without dependencies)
print("\n5. Checking Python Import Structure...")
sys.path.insert(0, str(base_dir))

try:
    # Check if __init__.py exports are defined
    with open(base_dir / '__init__.py', 'r') as f:
        content = f.read()
        if '__all__' in content:
            print("    __init__.py has __all__ defined")
        if 'TimeSeriesGenerator' in content:
            print("    __init__.py exports TimeSeriesGenerator")
        if 'generate_ar_dataset' in content:
            print("    __init__.py exports generator functions")
except Exception as e:
    print(f"    Error reading __init__.py: {e}")
    all_good = False

# Test 6: Count lines of code
print("\n6. Code Statistics...")
try:
    total_lines = 0
    python_files = list(base_dir.rglob('*.py'))
    python_files = [f for f in python_files if '__pycache__' not in str(f)]
    
    for py_file in python_files:
        with open(py_file, 'r') as f:
            lines = len(f.readlines())
            total_lines += lines
    
    print(f"    Total Python files: {len(python_files)}")
    print(f"    Total lines of code: {total_lines:,}")
    print(f"    Average lines/file: {total_lines // len(python_files):,}")
except Exception as e:
    print(f"     Could not count lines: {e}")

# Summary
print("\n" + "=" * 80)
if all_good:
    print(" SUCCESS: Library structure is correct!")
    print("\nNext Steps:")
    print("1. Install dependencies: pip install -r timeseries_dataset_generator/requirements.txt")
    print("2. Install library: cd timeseries_dataset_generator && pip install -e .")
    print("3. Run full test: python3 test_library.py")
    print("4. Or use directly: python3 dataset_updated.py")
else:
    print(" FAILED: Some files are missing")

print("=" * 80)

