# Machine Learning Pipeline for EV Charging Demand Prediction

## Overview

This module implements a comprehensive machine learning pipeline for predicting EV charging demand at the grid-cell level using exclusively OpenStreetMap (OSM) features. The pipeline operates as **Stage 1** of the HERO framework, providing demand predictions that drive proportional station allocation across urban grids.

**Key Achievement**: Trained on 613,108 grid samples extracted from OSM data, enabling portable deployment to any city worldwide without proprietary data dependencies.

## Table of Contents

1. [Architecture](#architecture)
2. [Pipeline Components](#pipeline-components)
3. [Feature Engineering](#feature-engineering)
4. [Model Training](#model-training)
5. [Usage](#usage)
6. [Pre-trained Models](#pre-trained-models)
7. [Performance Metrics](#performance-metrics)
8. [Research Contributions](#research-contributions)
9. [References](#references)

## Architecture

### Pipeline Flow

```
OSM Data → Feature Extraction → Feature Engineering → Model Training → Demand Prediction → Grid Allocation
```

### Module Structure

```
modules/MLPrediction/
├── __init__.py                      # Module initialization and exports
├── run_pipeline.py                  # Main pipeline runner (CLI interface)
├── demand_prediction_pipeline.py   # Orchestrator class (EVDemandPredictionPipeline)
├── feature_engineering.py          # Urban feature engineering (UrbanFeatureEngineer)
├── model_training.py               # Model training and evaluation (ModelTrainer)
├── demand_predictor.py             # Inference interface (DemandPredictor)
├── demand_score_calculator.py      # Demand score computation utilities
├── portable_osm_extractor.py       # OSM feature extraction (PortableOSMExtractor)
├── preprocess_data.py              # Data preprocessing and preparation utilities
├── target_transformation.py        # Target variable transformations (DemandTargetTransformer)
├── hyperparameter_tuning.py        # Hyperparameter optimization (HyperparameterTuner)
├── run_feature_selection.py        # Feature selection CLI and utilities
├── run_hyperparameter_tuning.py    # Standalone hyperparameter tuning CLI
├── synthetic_data_generator.py     # Synthetic data augmentation utilities
└── README.md                       # This file
```

### File Descriptions

- **`__init__.py`**: Module exports and initialization. Exports main classes and functions for easy importing.
- **`run_pipeline.py`**: Command-line interface for running the complete ML pipeline. Handles argument parsing and orchestrates the full training/evaluation workflow.
- **`demand_prediction_pipeline.py`**: Main orchestrator class (`EVDemandPredictionPipeline`) that coordinates all pipeline steps from data loading to model evaluation.
- **`feature_engineering.py`**: Feature engineering utilities and `UrbanFeatureEngineer` class. Transforms raw OSM features into domain-specific features.
- **`model_training.py`**: Model training, evaluation, and comparison utilities. Includes `ModelTrainer` class for training multiple algorithms and comparing performance.
- **`demand_predictor.py`**: Inference interface (`DemandPredictor` class) for loading and using trained models to make predictions on new data.
- **`demand_score_calculator.py`**: Utilities for calculating demand scores from raw charging data (volume, duration, etc.). Handles multiple target formulations.
- **`portable_osm_extractor.py`**: OSM feature extraction (`PortableOSMExtractor` class) for any city worldwide. Extracts road network, POI, land use, and geographic features.
- **`preprocess_data.py`**: Data preprocessing and preparation utilities. Handles data loading, cleaning, and preparation for training.
- **`target_transformation.py`**: Target variable transformation and normalization (`DemandTargetTransformer` class). Creates multiple target formulations and applies transformations.
- **`hyperparameter_tuning.py`**: Optuna-based hyperparameter optimization (`HyperparameterTuner` class). Performs efficient hyperparameter search for ML models.
- **`run_feature_selection.py`**: Feature selection CLI and utilities. Implements RFE, mutual information, and other feature selection methods.
- **`run_hyperparameter_tuning.py`**: Standalone hyperparameter tuning script. CLI interface for running hyperparameter optimization independently.
- **`synthetic_data_generator.py`**: Synthetic data augmentation utilities. Generates synthetic training samples for data augmentation and balancing.

## Pipeline Components

### 1. OSM Feature Extraction

**Module**: `portable_osm_extractor.py`

Extracts geospatial features from OpenStreetMap data for any city:

**Feature Categories:**
- **Road Network**: Road density, road types, intersection density
- **Points of Interest (POI)**: Commercial, residential, industrial zones
- **Land Use**: Urban, residential, commercial, industrial areas
- **Transportation**: Public transit stops, parking facilities
- **Geographic**: Elevation, water bodies, green spaces

**Key Features:**
- Works with any city worldwide
- No proprietary data required
- Automatic feature extraction
- Caching for efficiency

### 2. Feature Engineering

**Module**: `feature_engineering.py`

**Class**: `UrbanFeatureEngineer`

Transforms raw OSM features into domain-specific features:

**Engineering Techniques:**
- **Spatial Aggregation**: Multi-scale feature aggregation
- **Distance Features**: Proximity to key locations
- **Density Features**: Normalized density metrics
- **Interaction Features**: Feature interactions and ratios
- **Temporal Features**: Time-based patterns (if available)

**Example Features:**
```python
# Road network features
- road_density_1km
- intersection_density_500m
- highway_proximity

# POI features
- commercial_poi_density
- residential_density
- industrial_zone_coverage

# Spatial features
- distance_to_city_center
- urban_land_use_ratio
- green_space_coverage
```

### 3. Target Transformation

**Module**: `target_transformation.py`

**Class**: `DemandTargetTransformer`

Creates multiple target variable formulations:

**Target Variants:**
- `demand_score_kwh_only`: Energy-based demand
- `demand_score_hours_only`: Time-based demand
- `demand_score_kwh75_hrs25`: Weighted combination (75% kWh, 25% hours)
- `demand_score_kwh25_hrs75`: Weighted combination (25% kWh, 75% hours)
- `demand_score_balanced`: Equal weighting

**Transformation Options:**
- Original scale
- Log transformation
- Quantile transformation
- Robust scaling

### 4. Model Training

**Module**: `model_training.py`

**Class**: `ModelTrainer`

Trains and evaluates multiple regression models:

**Supported Algorithms:**
- **Gradient Boosting**: XGBoost, LightGBM, CatBoost
- **Random Forests**: RandomForest, ExtraTrees
- **Linear Models**: Ridge, Lasso, ElasticNet, BayesianRidge
- **Neural Networks**: MLPRegressor

**Training Features:**
- Cross-validation with multiple metrics
- Hyperparameter optimization (Optuna)
- Feature importance analysis
- Model comparison and selection
- Imbalanced data handling (SMOTE)

### 5. Hyperparameter Tuning

**Module**: `hyperparameter_tuning.py`

**Class**: `HyperparameterTuner`

Uses Optuna for efficient hyperparameter optimization:

**Optimization Strategy:**
- Tree-structured Parzen Estimator (TPE)
- Multi-objective optimization
- Early stopping
- Pruning of unpromising trials

## Feature Engineering

### OSM Feature Categories

#### 1. Road Network Features
- Road length by type (primary, secondary, residential)
- Road density at multiple scales (500m, 1km, 2km)
- Intersection density
- Highway proximity
- Road network connectivity

#### 2. POI Features
- Commercial POI density (restaurants, shops, malls)
- Residential building density
- Industrial zone coverage
- Educational facilities
- Healthcare facilities
- Entertainment venues

#### 3. Land Use Features
- Urban land use ratio
- Residential land use ratio
- Commercial land use ratio
- Industrial land use ratio
- Green space coverage
- Water body coverage

#### 4. Transportation Features
- Public transit stop density
- Parking facility density
- Distance to major transit hubs
- Bike lane density

#### 5. Geographic Features
- Elevation statistics
- Distance to city center
- Distance to water bodies
- Urban development index

### Feature Selection

**Module**: `run_feature_selection.py`

Selects optimal feature subsets using **Fast Mutual Information (MI) based selection**:

**Primary Method:**
- **Fast Mutual Information (MI) Selection**: Uses `SelectKBest` with `mutual_info_regression`
  - Automatically removes zero-variance features
  - Removes highly correlated features (>0.95)
  - Selects top features based on mutual information scores with target variable
  - Fast and efficient for large datasets

**Data Cleaning Steps:**
- Variance Threshold: Removes constant/zero-variance features
- Correlation Filtering: Removes highly correlated feature pairs (>0.95)
- Smart Feature Count: Auto-determines optimal number of features based on dataset size

**Output**: JSON file with selected features for reproducibility

## Model Training

### Training Pipeline

```python
from modules.MLPrediction.demand_prediction_pipeline import EVDemandPredictionPipeline

# Initialize pipeline
pipeline = EVDemandPredictionPipeline(
    random_state=42,
    skip_synthetic_data=False,
    enable_feature_caching=True
)

# Run complete pipeline
results = pipeline.run_complete_pipeline(
    data_path="data/training_data.csv",
    target_column="demand_score_kwh75_hrs25",
    output_dir="results/ml_output"
)
```

### Training Configuration

**Key Parameters:**
- `random_state`: Seed for reproducibility
- `skip_synthetic_data`: Disable synthetic data augmentation
- `enable_feature_caching`: Cache feature engineering results
- `skip_feature_selection`: Use all features (faster)
- `pre_selected_features_path`: Path to pre-selected features JSON
- `limit_transformations`: Limit target transformations

### Model Evaluation

**Metrics:**
- R² Score (coefficient of determination)
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Spearman's rank correlation

**Validation:**
- K-fold cross-validation
- Train/validation/test split
- Statistical significance testing

## Usage

### Command-Line Interface

**Main Pipeline Runner:**

```bash
python -m modules.MLPrediction.run_pipeline \
    --data-path results/HERO_dataset.csv \
    --output-dir results/ml_pipeline_output \
    --target-column demand_score_kwh75_hrs25 \
    --pre-selected-features results/feature_selection/selected_features.json \
    --skip-feature-selection \
    --random-state 42 \
    --skip-synthetic-data \
    --skip-spatial-features \
    --limit-target-transformations \
    --force-retrain
```

**Note:** You can use the same pre-selected features file to train models for different target formulations. Simply change the `--target-column` parameter while keeping the same `--pre-selected-features` path. This is computationally efficient and scientifically valid.

**Key Arguments:**
- `--data-path`: Path to training data CSV
- `--output-dir`: Output directory for models and results
- `--target-column`: Target variable column name
- `--pre-selected-features`: Path to JSON with pre-selected features
- `--skip-feature-selection`: Skip feature selection step
- `--random-state`: Random seed for reproducibility
- `--skip-synthetic-data`: Disable synthetic data generation
- `--skip-spatial-features`: Skip spatial feature engineering
- `--limit-target-transformations`: Use only best target transformations
- `--force-retrain`: Force retraining even if model exists

### Python API

```python
from modules.MLPrediction.demand_prediction_pipeline import EVDemandPredictionPipeline

# Initialize
pipeline = EVDemandPredictionPipeline(random_state=42)

# Run pipeline
results = pipeline.run_complete_pipeline(
    data_path="data/training.csv",
    target_column="demand_score",
    output_dir="results/"
)

# Access results
best_model = results['evaluation']['overall_best']
print(f"Best model: {best_model['configuration']}")
print(f"Best score: {best_model['score']:.4f}")
```

### Inference (Prediction)

```python
from modules.MLPrediction.demand_predictor import DemandPredictor

# Load trained model
predictor = DemandPredictor(model_path="models/demand_score_model.pkl")

# Prepare features (OSM features for new city)
features = extract_osm_features(city_name="New City")

# Predict demand
demand_scores = predictor.predict(features)
```

## Pre-trained Models

Pre-trained models are available in the `models/` directory:

**Model Naming Convention:**
```
demand_score_{target_variant}_{transformation}_{algorithm}.pkl
```

**Example Models:**
- `demand_score_kwh75_hrs25_original_quantile_xgboost.pkl`
- `demand_score_kwh_only_original_robust_xgboost.pkl`
- `demand_score_hours_only_log_quantile_xgboost.pkl`

**Model Selection:**
- Best overall: `demand_score_kwh_only_original_quantile_xgboost.pkl`
- For time-focused: `demand_score_hours_only_*`
- For balanced: `demand_score_kwh75_hrs25_*`

## Performance Metrics

### Training Performance

**Best Model Performance** (on validation set):
- R² Score: ~0.65-0.75
- RMSE: Varies by target transformation
- Spearman's ρ: ~0.70-0.80

### Validation Performance

**Geographic Portability Validation** (VED dataset):
- Spearman's rank correlation: **ρ = 0.626**
- Strong rank-order preservation across cities
- Confirms geographic portability

### Model Comparison

| Model Type | R² Score | RMSE | Training Time |
|------------|----------|------|---------------|
| XGBoost | 0.72 | Low | Medium |
| LightGBM | 0.70 | Low | Fast |
| Random Forest | 0.68 | Medium | Slow |
| Neural Network | 0.65 | Medium | Slow |

## Research Contributions

1. **Portable Demand Prediction**: Models trained exclusively on OSM features enable deployment to any city
2. **Large-Scale Training**: 613,108 grid samples from multiple cities
3. **Comprehensive Feature Engineering**: Domain-specific features from urban planning principles
4. **Multiple Target Formulations**: Robust to different demand definitions
5. **Reproducible Pipeline**: Complete open-source implementation

## Data Requirements

### Training Data

**Required Format:**
- CSV file with OSM features and demand scores
- Grid-level data (one row per grid cell)
- Features extracted from OSM data
- Target variable (demand score)

**Example Columns:**
```csv
grid_id,road_density,poi_density,demand_score_kwh75_hrs25,...
grid_1,0.45,0.32,0.78,...
grid_2,0.23,0.15,0.45,...
```

### Data Sources

- **UrbanEV Dataset**: Urban electric vehicle charging data
- **ST-EVCDP Dataset**: Spatio-temporal EV charging demand prediction data
- **VED Dataset**: Vehicle Energy Dataset (for validation)

See the main README.md for detailed data management information.

## Hyperparameter Tuning

### Using Optuna

```python
from modules.MLPrediction.hyperparameter_tuning import HyperparameterTuner

tuner = HyperparameterTuner(
    model_type="xgboost",
    n_trials=100,
    timeout=3600  # 1 hour
)

best_params = tuner.optimize(
    X_train, y_train,
    X_val, y_val
)
```

## Feature Selection

### Running Feature Selection

```bash
python -m modules.MLPrediction.run_feature_selection \
    --data-path results/HERO_dataset.csv \
    --output-dir results/feature_selection \
    --random-state 42
    # Note: --max-features is optional (auto-determined based on dataset size)
    # Method: Fast Mutual Information (MI) based selection (not RFE)
```

**Output**: JSON file with selected features for reproducibility

## Troubleshooting

### Common Issues

**1. Missing Features Error**
```
ValueError: Missing required OSM feature columns: ['feature_x', 'feature_y']
```
**Solution**: Ensure OSM feature extraction includes all required features

**2. Model Loading Error**
```
FileNotFoundError: Model file not found
```
**Solution**: Check model path or retrain model

**3. Memory Issues**
```
MemoryError: Unable to allocate array
```
**Solution**: 
- Use `--skip-synthetic-data`
- Reduce feature set
- Use feature selection

**4. Slow Training**
**Solution**:
- Use `--skip-feature-selection`
- Enable feature caching
- Use GPU acceleration (if available)

## References

1. Li, H., et al. (2025). UrbanEV: An Open Benchmark Dataset for Urban Electric Vehicle Charging Demand Prediction. *Scientific Data*, 12, 523.

2. Qu, H., et al. (2024). A Physics-Informed and Attention-Based Graph Learning Approach for Regional Electric Vehicle Charging Demand Prediction. *IEEE Transactions on Intelligent Transportation Systems*.

3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference*.

4. Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems*, 30.

5. Prokhorenkova, L., et al. (2018). CatBoost: Unbiased Boosting with Categorical Features. *Advances in Neural Information Processing Systems*, 31.

## License

This module is part of the HERO framework and is licensed under the same terms as the main project.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback about the ML prediction module, please open an issue in the repository.

---

