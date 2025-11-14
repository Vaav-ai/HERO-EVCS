# Demand Model Validation Framework

## Overview

This module provides comprehensive validation of demand prediction models using ground truth data from the Vehicle Energy Dataset (VED). It validates the geographic portability of ML models and assesses model performance across different cities and grid configurations.

## Table of Contents

1. [Module Structure](#module-structure)
2. [Purpose](#purpose)
3. [Usage](#usage)
4. [Validation Metrics](#validation-metrics)
5. [Integration](#integration)

## Module Structure

```
evaluation/demand_validation/
├── run_validation.py          # Main validation script
└── README.md                  # This file
```

## Purpose

The demand validation framework addresses key research questions:

1. **Geographic Portability**: Can models trained on one city work on another?
2. **Grid Size Sensitivity**: How does grid size affect model performance?
3. **Model Robustness**: How do different model configurations perform?
4. **Rank-Order Preservation**: Do models preserve relative demand rankings?

## Usage

### Basic Validation

```bash
# Validate with default settings (1.0 km grid)
python evaluation/demand_validation/run_validation.py

# Validate with custom grid size
python evaluation/demand_validation/run_validation.py --grid-size 1.5

# Validate specific model
python evaluation/demand_validation/run_validation.py \
    --model-path models/demand_score_kwh_only_original_quantile_xgboost.pkl
```

### Advanced Options

```bash
# Sensitivity analysis
python evaluation/demand_validation/run_validation.py --sensitivity-analysis

# Custom validation data
python evaluation/demand_validation/run_validation.py \
    --validation-data data/validation/ved_processed_with_grids.parquet

# Output directory
python evaluation/demand_validation/run_validation.py \
    --output-dir results/validation_results
```

### Key Arguments

- `--grid-size`: Grid cell size in kilometers (default: 1.0)
- `--model-path`: Path to model file (default: best model)
- `--validation-data`: Path to validation dataset
- `--output-dir`: Output directory for results
- `--sensitivity-analysis`: Run sensitivity analysis across grid sizes

## Validation Metrics

### Primary Metrics

1. **Spearman's Rank Correlation (ρ)**
   - Measures rank-order preservation
   - Target: ρ > 0.6 for geographic portability
   - **Paper Result**: ρ = 0.626

2. **Pearson Correlation (r)**
   - Measures linear relationship
   - Additional validation metric

3. **R² Score**
   - Coefficient of determination
   - Measures explained variance

4. **RMSE / MAE**
   - Error metrics
   - Absolute and relative errors

### Output Files

- `validation_report.txt`: Human-readable validation report
- `validation_metrics.json`: Detailed metrics in JSON format
- `correlation_plots.png`: Visualization of correlations
- `grid_comparison.png`: Grid size sensitivity analysis

## Integration

### With ML Pipeline

The validation framework uses:
- Pre-trained models from `models/` directory
- VED processed data from `data/validation/`
- OSM feature extraction from ML pipeline
- Grid generation from utils module

### Validation Workflow

```
VED Data → Grid Assignment → OSM Feature Extraction → 
Model Prediction → Ground Truth Comparison → Metrics Calculation
```

## Research Context

This validation framework validates the **geographic portability** claim of the HERO framework:

- Models trained on multiple cities (UrbanEV, ST-EVCDP)
- Validated on Ann Arbor VED data
- Strong rank-order correlation (ρ = 0.626) confirms portability
- Enables deployment to any city with OSM data

## Example Output

```
========================================
Demand Model Validation Report
========================================

Model: demand_score_kwh_only_original_quantile_xgboost.pkl
Grid Size: 1.0 km
Validation Dataset: VED (Ann Arbor)

Metrics:
--------
Spearman's ρ: 0.626
Pearson's r: 0.589
R² Score: 0.347
RMSE: 0.234
MAE: 0.189

Conclusion:
-----------
✅ Geographic portability validated (ρ > 0.6)
✅ Model preserves rank-order across cities
```

## Troubleshooting

### Missing Validation Data

The VED processed data should be located at `data/validation/ved_processed_with_grids.parquet`. If missing, you can process raw VED data using the data preparation utilities in `evaluation/data_prep/`.

### Model Not Found

Pre-trained models are included in the `models/` directory. If models are missing, you can retrain them using the ML pipeline (see `modules/MLPrediction/README.md`).

### Grid Generation Issues

```python
# Check grid generation
from modules.utils.gridding import CityGridding
gridder = CityGridding(primary_grid_size_km=1.0)
grids = gridder.create_city_grid("Ann Arbor, Michigan, USA")
```

## License

This module is part of the HERO framework and is licensed under the same terms as the main project.

---



