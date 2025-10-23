# Git Migration Guide

## Mevcut Durum
- GitHub Repo: https://github.com/ismailguzel/ts-stationary
- Main branch'te eski kod var
- Yeni modüler kütüphaneyi main'e koymak istiyoruz

## Strateji
1. Eski kodu `legacy` branch'ine taşı
2. Main branch'i temizle
3. Yeni kütüphaneyi main'e push et

## Adım Adım Talimatlar

### 1. Mevcut Repo'yu Clone Et (Henüz yapmadıysan)
```bash
cd /Users/iguzel/Downloads
git clone https://github.com/ismailguzel/ts-stationary.git
cd ts-stationary
```

### 2. Eski Kodu Legacy Branch'ine Al
```bash
# Legacy branch oluştur ve eski kodu koru
git checkout -b legacy
git push origin legacy

# Main'e geri dön
git checkout main
```

### 3. Main Branch'i Temizle (Dikkatli!)
```bash
# Tüm dosyaları sil (git history korunur)
git rm -rf .
git commit -m "Clean main branch for new modular library"
```

### 4. Yeni Kütüphaneyi Kopyala
```bash
# Yeni kütüphaneyi buraya kopyala
cp -r /Users/iguzel/Downloads/dataset_generation/timeseries_dataset_generator/* .
cp /Users/iguzel/Downloads/dataset_generation/timeseries_dataset_generator/.gitignore .

# Root'a gerekli dosyaları da ekle
cp /Users/iguzel/Downloads/dataset_generation/README.md .
cp /Users/iguzel/Downloads/dataset_generation/dataset_updated.py ./examples/
cp /Users/iguzel/Downloads/dataset_generation/simple_generation.py ./examples/
```

### 5. Git için Hazırla
```bash
# Yeni dosyaları ekle
git add .
git commit -m "Add modular timeseries dataset generator library

- Refactored from monolithic 1999-line file to 16 modules
- Added reproducibility (seed=42)
- PyPI-ready structure
- Comprehensive documentation
- 98-99% reproducible output
- Production ready

Legacy code moved to 'legacy' branch for reference."
```

### 6. GitHub'a Push Et
```bash
# Main branch'i push et
git push origin main

# Legacy branch zaten push edildi
```

## Alternatif Strateji (Daha Güvenli)

Eğer main'i temizlemek riskli geliyorsa:

### Opsiyon 2: Yeni Repo Oluştur
```bash
cd /Users/iguzel/Downloads/dataset_generation/timeseries_dataset_generator

# Git başlat
git init
git add .
git commit -m "Initial commit: Modular timeseries dataset generator"

# GitHub'da yeni repo oluştur: timeseries-dataset-generator
# Sonra:
git remote add origin https://github.com/ismailguzel/timeseries-dataset-generator.git
git branch -M main
git push -u origin main
```

## Önerilen Repo Yapısı (Main Branch)

```
ts-stationary/                      (veya yeni repo)
├── README.md                       # Ana kılavuz
├── LICENSE                         # MIT
├── .gitignore                      # Python gitignore
├── setup.py                        # PyPI setup
├── pyproject.toml                  # Modern packaging
├── requirements.txt                # Dependencies
│
├── timeseries_dataset_generator/   # Library package
│   ├── __init__.py
│   ├── core/
│   ├── generators/
│   ├── utils/
│   └── ...
│
├── examples/                       # Usage examples
│   ├── dataset_updated.py
│   ├── simple_generation.py
│   └── ...
│
├── tests/                          # Tests (ileride)
│   └── ...
│
└── docs/                           # Documentation
    ├── PROJECT_SUMMARY.md
    ├── QUICK_REFERENCE.md
    └── ...
```

## .gitignore İçeriği

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific
generated-dataset/
generated-dataset*/
simple_output/
test_output/
_old_files/
*.parquet

# Jupyter
.ipynb_checkpoints/
*.ipynb
```

## README.md için Öneriler

```markdown
# Time Series Dataset Generator

A modular Python library for generating synthetic time series datasets with various characteristics.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Features

- 25+ time series types (AR, MA, GARCH, SARIMA, etc.)
- Reproducible generation (fixed seed)
- 225 parquet files with complete metadata
- PyPI-ready structure
- Comprehensive documentation

## Quick Start

\`\`\`bash
pip install -r requirements.txt
pip install -e .
python examples/simple_generation.py
\`\`\`

## Documentation

- [Quick Reference](docs/QUICK_REFERENCE.md)
- [Project Summary](docs/PROJECT_SUMMARY.md)
- [Examples](examples/)

## Legacy Code

Previous monolithic version is available in the `legacy` branch.

## License

MIT License - see [LICENSE](LICENSE)
```

## Komutları Çalıştırma Sırası

### Seçenek 1: Mevcut Repo'yu Güncelle
```bash
# 1. Clone
cd /Users/iguzel/Downloads
git clone https://github.com/ismailguzel/ts-stationary.git
cd ts-stationary

# 2. Legacy branch oluştur
git checkout -b legacy
git push origin legacy
git checkout main

# 3. Main'i temizle
git rm -rf .
git commit -m "Clean for new library"

# 4. Yeni dosyaları organize et
mkdir -p timeseries_dataset_generator examples docs tests

# 5. Kopyala
cp -r /Users/iguzel/Downloads/dataset_generation/timeseries_dataset_generator/* ./timeseries_dataset_generator/
cp /Users/iguzel/Downloads/dataset_generation/README.md .
cp /Users/iguzel/Downloads/dataset_generation/PROJECT_SUMMARY.md ./docs/
cp /Users/iguzel/Downloads/dataset_generation/QUICK_REFERENCE.md ./docs/
cp /Users/iguzel/Downloads/dataset_generation/dataset_updated.py ./examples/
cp /Users/iguzel/Downloads/dataset_generation/simple_generation.py ./examples/
cp /Users/iguzel/Downloads/dataset_generation/timeseries_dataset_generator/.gitignore .
cp /Users/iguzel/Downloads/dataset_generation/timeseries_dataset_generator/LICENSE .

# 6. Commit & Push
git add .
git commit -m "Add modular library (see docs/PROJECT_SUMMARY.md)"
git push origin main -f  # Force push (dikkatli!)
```

### Seçenek 2: Yeni Repo Oluştur (Önerilen)
```bash
# Daha güvenli - yeni repo
cd /Users/iguzel/Downloads
mkdir timeseries-dataset-generator
cd timeseries-dataset-generator

# Organize et
mkdir -p timeseries_dataset_generator examples docs tests

# Kopyala
cp -r /Users/iguzel/Downloads/dataset_generation/timeseries_dataset_generator/* ./timeseries_dataset_generator/
# ... (yukarıdaki gibi)

# Git başlat
git init
git add .
git commit -m "Initial commit"

# GitHub'da yeni repo oluştur, sonra:
git remote add origin https://github.com/ismailguzel/timeseries-dataset-generator.git
git branch -M main
git push -u origin main
```

## Tavsiyem

**Seçenek 2'yi öneririm** (yeni repo):
- Daha güvenli
- Temiz history
- Eski repo bozulmaz
- ts-stationary legacy kalabilir
- Yeni repo: `timeseries-dataset-generator`

Hangisini tercih ediyorsun?
