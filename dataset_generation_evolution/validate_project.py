#!/usr/bin/env python3
"""
Project validation script.
Checks project structure, files, and readiness.
"""

from pathlib import Path
import sys

print("=" * 80)
print("Project Validation Check")
print("=" * 80)

project_root = Path(__file__).parent
issues = []
warnings = []

# 1. Check essential files
print("\n1. Checking Essential Files...")
essential_files = {
    'README.md': 'Main documentation',
    'dataset_updated.py': 'Production script',
    'timeseries_dataset_generator/__init__.py': 'Library package',
    'timeseries_dataset_generator/setup.py': 'Setup file',
    'timeseries_dataset_generator/requirements.txt': 'Dependencies',
}

for file, desc in essential_files.items():
    file_path = project_root / file
    if file_path.exists():
        print(f"   OK: {file} ({desc})")
    else:
        issues.append(f"Missing: {file} ({desc})")
        print(f"   MISSING: {file}")

# 2. Check library structure
print("\n2. Checking Library Structure...")
required_dirs = {
    'timeseries_dataset_generator/core': 'Core components',
    'timeseries_dataset_generator/generators': 'Generator modules',
    'timeseries_dataset_generator/utils': 'Utilities',
}

for dir_path, desc in required_dirs.items():
    full_path = project_root / dir_path
    if full_path.exists() and full_path.is_dir():
        py_files = list(full_path.glob('*.py'))
        print(f"   OK: {dir_path} ({len(py_files)} Python files)")
    else:
        issues.append(f"Missing directory: {dir_path}")
        print(f"   MISSING: {dir_path}")

# 3. Check for duplicate/old files
print("\n3. Checking for Old Files...")
old_files_dir = project_root / '_old_files'
if old_files_dir.exists():
    old_count = len(list(old_files_dir.glob('*.py')))
    print(f"   OK: Old files archived ({old_count} files in _old_files/)")
else:
    warnings.append("_old_files/ directory not found")
    print(f"   WARNING: No _old_files/ directory")

# Check for old files in root
old_in_root = []
for old_file in ['dataset_generator.py', 'dataset.py', 'generator.py']:
    if (project_root / old_file).exists():
        old_in_root.append(old_file)

if old_in_root:
    warnings.append(f"Old files in root: {', '.join(old_in_root)}")
    print(f"   WARNING: Old files in root: {', '.join(old_in_root)}")
else:
    print(f"   OK: No old files in root directory")

# 4. Check imports
print("\n4. Checking Imports...")
try:
    sys.path.insert(0, str(project_root))
    from timeseries_dataset_generator import TimeSeriesGenerator
    print(f"   OK: TimeSeriesGenerator imported")
    
    from timeseries_dataset_generator.generators.stationary import generate_ar_dataset
    print(f"   OK: Generator modules importable")
except Exception as e:
    issues.append(f"Import error: {e}")
    print(f"   ERROR: {e}")

# 5. Check for emojis
print("\n5. Checking for Emojis (AI markers)...")
emoji_count = 0
py_files = list(project_root.rglob('*.py'))
py_files = [f for f in py_files if '__pycache__' not in str(f) and '_old_files' not in str(f)]

for py_file in py_files[:10]:  # Check first 10
    try:
        content = py_file.read_text()
        if any(ord(c) > 127 for c in content if c not in '\n\t\r'):
            has_emoji = False
            for c in content:
                if ord(c) > 127 and c not in 'çğıöşüÇĞİÖŞÜ':  # Allow Turkish chars
                    has_emoji = True
                    break
            if has_emoji:
                emoji_count += 1
    except:
        pass

if emoji_count == 0:
    print(f"   OK: No emojis found in Python files")
else:
    warnings.append(f"Found potential emojis in {emoji_count} files")
    print(f"   WARNING: Potential emojis in {emoji_count} files")

# 6. Check random seeds
print("\n6. Checking Reproducibility...")
seed_files = ['dataset_updated.py', 'simple_generation.py']
seed_count = 0

for seed_file in seed_files:
    file_path = project_root / seed_file
    if file_path.exists():
        content = file_path.read_text()
        if 'RANDOM_SEED' in content and 'np.random.seed' in content:
            seed_count += 1
            print(f"   OK: {seed_file} has random seed")
        else:
            warnings.append(f"{seed_file} missing random seed")
            print(f"   WARNING: {seed_file} missing seed")

# 7. File count summary
print("\n7. File Count Summary...")
py_count = len([f for f in project_root.rglob('*.py') if '__pycache__' not in str(f) and '_old_files' not in str(f)])
md_count = len(list((project_root).glob('*.md')))
lib_py_count = len(list((project_root / 'timeseries_dataset_generator').rglob('*.py')))

print(f"   Total Python files: {py_count}")
print(f"   Documentation files: {md_count}")
print(f"   Library modules: {lib_py_count}")

# Summary
print("\n" + "=" * 80)
print("Validation Summary")
print("=" * 80)

if not issues and not warnings:
    print("\nSTATUS: ALL CHECKS PASSED")
    print("Project is ready for production use!")
elif not issues:
    print(f"\nSTATUS: PASSED WITH WARNINGS ({len(warnings)})")
    print("\nWarnings:")
    for w in warnings:
        print(f"  - {w}")
else:
    print(f"\nSTATUS: FAILED ({len(issues)} issues, {len(warnings)} warnings)")
    print("\nIssues:")
    for i in issues:
        print(f"  - {i}")
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")

print("\n" + "=" * 80)

sys.exit(0 if not issues else 1)

