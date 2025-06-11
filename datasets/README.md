# KSE Memory SDK - Synthetic Datasets

## Overview

This directory contains synthetic datasets for reproducible research and benchmarking. All datasets are generated using privacy-preserving techniques and are freely available under MIT license.

## Available Datasets

### 1. Synthetic Retail Dataset (`synthetic_retail.json`)
- **Size**: 10,000 products across 50 categories
- **Domains**: Fashion, electronics, home goods, sports, beauty
- **Features**: Product descriptions, prices, categories, attributes, reviews
- **Use Case**: E-commerce search and recommendation benchmarking

### 2. Synthetic Finance Dataset (`synthetic_finance.json`)
- **Size**: 5,000 financial products
- **Domains**: Investment products, insurance, loans, credit cards
- **Features**: Risk profiles, returns, fees, regulatory information
- **Use Case**: Financial product matching and risk assessment

### 3. Synthetic Healthcare Dataset (`synthetic_healthcare.json`)
- **Size**: 3,000 medical devices and treatments
- **Domains**: Diagnostic equipment, therapeutic devices, pharmaceuticals
- **Features**: Clinical efficacy, safety profiles, regulatory status
- **Use Case**: Medical device selection and clinical decision support

## Dataset Generation

All datasets are generated using:
- **Faker library**: For realistic but synthetic data
- **Domain-specific templates**: Industry-appropriate attributes
- **Statistical distributions**: Realistic price and rating distributions
- **Privacy preservation**: No real-world data used

## Licensing

All synthetic datasets are released under MIT License for maximum reproducibility and academic use.

## Usage

```python
import json
from kse_memory import KSEMemory

# Load synthetic retail dataset
with open('datasets/synthetic_retail.json', 'r') as f:
    retail_data = json.load(f)

# Initialize KSE with synthetic data
memory = KSEMemory()
for product in retail_data:
    await memory.add_product(product)
```

## Reproducibility

These datasets enable complete reproduction of all empirical results in our academic publications without requiring proprietary data access.