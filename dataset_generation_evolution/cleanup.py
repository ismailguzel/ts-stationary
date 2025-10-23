#!/usr/bin/env python3
"""
Cleanup script for the project.
Removes old files, test outputs, and temporary files.
"""

import shutil
from pathlib import Path

print("=" * 80)
print("Project Cleanup")
print("=" * 80)

project_root = Path(__file__).parent

# Directories to clean
cleanup_targets = [
    ("simple_output", "Test output from simple_generation.py"),
    ("test_output", "Test output from test_library.py"),
    ("__pycache__", "Python cache files"),
    ("timeseries_dataset_generator/__pycache__", "Library cache files"),
]

# Optional: completely remove old files
remove_old_files = False  # Set to True to delete _old_files/

print("\nCleaning up temporary files...")
cleaned = 0

for target, description in cleanup_targets:
    target_path = project_root / target
    if target_path.exists():
        try:
            if target_path.is_dir():
                shutil.rmtree(target_path)
            else:
                target_path.unlink()
            print(f"  Removed: {target} ({description})")
            cleaned += 1
        except Exception as e:
            print(f"  Warning: Could not remove {target}: {e}")

# Clean .pyc files
print("\nRemoving .pyc files...")
pyc_count = 0
for pyc_file in project_root.rglob("*.pyc"):
    try:
        pyc_file.unlink()
        pyc_count += 1
    except Exception as e:
        print(f"  Warning: Could not remove {pyc_file}: {e}")

if pyc_count > 0:
    print(f"  Removed {pyc_count} .pyc files")

# Handle old files
if remove_old_files:
    old_files_path = project_root / "_old_files"
    if old_files_path.exists():
        shutil.rmtree(old_files_path)
        print("\n  Removed: _old_files/ (old project files)")
        cleaned += 1
else:
    print("\nNote: _old_files/ kept (contains archived files)")
    print("      Set remove_old_files=True in script to delete")

print("\n" + "=" * 80)
print(f"Cleanup complete! Removed {cleaned} items + {pyc_count} .pyc files")
print("=" * 80)

