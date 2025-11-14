# Reproducibility Guide

This document provides exact commands and steps to reproduce all results presented in the HERO paper.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Original Experimental Environments](#original-experimental-environments)
3. [Data Setup](#data-setup)
4. [Reproducing ML Pipeline Results](#reproducing-ml-pipeline-results)
5. [Reproducing RL Evaluation Results](#reproducing-rl-evaluation-results)
6. [Reproducing Main Pipeline Results](#reproducing-main-pipeline-results)
7. [Reproducing Validation Results](#reproducing-validation-results)
8. [Expected Outputs](#expected-outputs)
9. [Reproducibility Notes](#reproducibility-notes)

## Prerequisites

### System Requirements

- Python 3.8 or higher (3.10 recommended)
- SUMO 1.15.0 or higher
- OSMnx 1.9.4 
- 16GB+ RAM (32GB recommended for parallel processing)
- SSD storage recommended
- Linux/macOS (Windows with WSL or Docker recommended)

### Installation Options

#### Option 1: Docker (Recommended - Easiest)

The Docker image includes all dependencies pre-configured:

```bash
# Pull the pre-built image
docker pull vatsal1729/hero-evcs:latest

# Verify installation
docker run --rm vatsal1729/hero-evcs:latest python3 -c "import traci, sumolib; print('SUMO OK')"
docker run --rm vatsal1729/hero-evcs:latest python3 -c "import osmnx as ox; print(f'OSMnx {ox.__version__}')"
```

**Docker Image Specifications:**
- Base: Ubuntu 24.04 (matches paper experimental environments)
- Python: 3.10
- SUMO: Latest stable (from PPA)
- OSMnx: 1.9.4
- All dependencies: Pre-installed

#### Option 2: Native Installation

```bash
# 1. Clone repository
git clone git@github.com:Vaav-ai/HERO-EVCS.git
cd HERO-EVCS

# 2. Create virtual environment
python -m venv ev_placement_env
source ev_placement_env/bin/activate  # On Windows: ev_placement_env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set SUMO environment variables
export SUMO_HOME=/usr/share/sumo  # Adjust path as needed
export PATH=$PATH:$SUMO_HOME/bin
export PYTHONPATH=$PYTHONPATH:$SUMO_HOME/tools

# 5. Verify installation
python -c "import traci, sumolib; print('SUMO OK')"
python -c "import osmnx as ox; print(f'OSMnx {ox.__version__}')"  # Should be 1.9.4
```

### Seed Configuration

All commands use `--seed 42` or `--random-state 42` for reproducibility. **Do not change the seed** if you want to reproduce exact results.

### Docker Usage for All Evaluation Steps

**Yes, Docker can be used for all evaluation steps!** The Docker image includes all dependencies (SUMO, Python packages, etc.) and can run:
- ✅ ML Pipeline (feature selection, model training)
- ✅ RL Evaluation (multi-grid and single-grid)
- ✅ Main Pipeline (end-to-end placement)
- ✅ Validation (demand validation, geographic portability)

**Required Volume Mounts:**

For different use cases, mount the following directories:

| Use Case | Required Mounts | Description |
|----------|----------------|-------------|
| **ML Training** | `results/`, `models/` | For training data and model outputs |
| **RL Evaluation** | `results/`, `models/`, `data/`, `generated_files/` | For checkpoints, models, validation data, and SUMO networks |
| **Main Pipeline** | `results/`, `models/`, `generated_files/` | For outputs, models, and network files |
| **Validation** | `results/`, `models/`, `data/` | For validation data and model outputs |

**Example Docker Command Template:**
```bash
docker run --rm \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/generated_files:/app/generated_files \
    vatsal1729/hero-evcs:latest python3 <script> <arguments>
```

**Important:** Always mount `results/` to persist outputs. Mount other directories (`models/`, `data/`, `generated_files/`) only if needed for the specific task.

## Original Experimental Environments

**Important:** The results in the paper were generated using specific cloud environments. The Docker image (Ubuntu 24.04) matches the paper environment and should reproduce results with minimal variations. Minor variations may still occur due to:
- Different Python/NumPy random number generator implementations
- Different OS-level random number generators
- Floating-point precision differences across platforms
- Different library versions (if not using Docker)

### Environments Used for Paper Results

| Component | Environment | Specifications |
|-----------|-------------|----------------|
| **ML Pipeline** | Lightning AI L4 Machine | Ubuntu 24.04, Python 3.10, Conda/Venv environment |
| **Main Pipeline** | Lightning AI L4 Machine | Ubuntu 24.04, Python 3.10, Conda/Venv environment |
| **RL Evaluation** | Google Cloud VM | Ubuntu 24.04 (noble), n2-custom-12-49152 (12 vCPUs, 48 GB RAM) |

**Note:** The Docker image now uses Ubuntu 24.04 to match the paper experimental environments, which should reduce numerical variations compared to results generated on Ubuntu 24.04 systems.

### Reproducibility Considerations

- **Seed Consistency**: All commands use `--seed 42` or `--random-state 42` for reproducibility
- **Environment Differences**: Results may vary slightly across different Python environments (conda vs venv vs Docker), but Docker (Ubuntu 24.04) matches the paper environment
- **Random Number Generators**: NumPy, Python's `random`, and OS-level RNGs may produce slightly different sequences across platforms
- **Floating-Point Precision**: Minor differences in floating-point operations are expected and acceptable

**Expected Variation Range:**
- ML metrics: ±0.01-0.02 in R², MAE, RMSE
- RL rewards: ±0.01-0.03 in composite rewards
- Statistical significance: Should remain consistent (p < 0.05)

## Data Setup

### Option 1: Use Pre-trained Models (Recommended for Quick Start)

Pre-trained models are already included in the `models/` directory. No download needed!

**Verify models exist:**
```bash
ls models/demand_score_*.pkl
```

**Expected models:**
- `demand_score_balanced_*.pkl`
- `demand_score_hours_only_*.pkl`
- `demand_score_kwh_only_*.pkl`
- `demand_score_kwh25_hrs75_*.pkl`
- `demand_score_kwh75_hrs25_*.pkl`

### Option 2: Download Processed Training Dataset (For ML Model Retraining)

The processed dataset from `preprocess_data.py` is available on Kaggle. This is the output of the preprocessing pipeline and can be used directly for training ML models.

**Native Installation:**
```bash
# Install Kaggle CLI (if not already installed)
pip install kaggle

# Download the processed dataset
kaggle datasets download -d vatsal1729/hero-geospatial-ev-charging-demand-dataset
unzip hero-geospatial-ev-charging-demand-dataset.zip

# Extract HERO_dataset.csv to results/ directory
cp HERO_dataset.csv results/HERO_dataset.csv
# This file is ready to use with run_pipeline.py
```

**Docker:**
```bash
# Download dataset locally first, then mount it
kaggle datasets download -d vatsal1729/hero-geospatial-ev-charging-demand-dataset
unzip hero-geospatial-ev-charging-demand-dataset.zip

# Run with data mounted
docker run --rm \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/data:/app/data \
    vatsal1729/hero-evcs:latest python3 -m modules.MLPrediction.run_pipeline \
    --data-path /app/results/HERO_dataset.csv \
    --output-dir /app/results/ml_pipeline_output \
    --target-column demand_score_kwh75_hrs25 \
    --random-state 42 \
    --force-retrain
```

### Option 3: Full Data Setup (For Complete Reproduction)

**Core datasets are already included:**
- `data/training/UrbanEV-main/` - UrbanEV dataset
- `data/training/ST-EVCDP-main/` - ST-EVCDP dataset

**Additional validation data (optional):**
- `data/validation/ved_processed_with_grids.parquet` - VED validation data

**Expected Data Structure:**
```
data/
├── training/
│   ├── UrbanEV-main/          # UrbanEV dataset (included)
│   └── ST-EVCDP-main/         # ST-EVCDP dataset (included)
└── validation/
    └── ved_processed_with_grids.parquet  # VED validation data (optional)

models/
└── demand_score_*.pkl         # Pre-trained models (included)

results/
└── HERO_dataset.csv            # Processed dataset (download from Kaggle)
```

**Note:** The `download_data.py` script is not included. Core datasets are already in the repository. For the processed training dataset, download from Kaggle as shown above.

## Reproducing ML Pipeline Results

### Step 1: Feature Selection (If Needed)

**Note:** Feature selection uses **Fast Mutual Information (MI) based selection** with SelectKBest, not RFE. The method automatically:
- Removes zero-variance features
- Removes highly correlated features (>0.95)
- Selects top features based on mutual information scores

**Exact Command Used (Lightning AI L4 - Ubuntu 24.04):**
```bash
python -m modules.MLPrediction.run_feature_selection \
    --data-path results/HERO_dataset.csv \
    --output-dir results/feature_selection \
    --random-state 42
```

**Note:** The `--max-features` parameter is optional (auto-determined based on dataset size). The method uses mutual information regression, not RFE.

**Native Installation (Alternative):**
```bash
python -m modules.MLPrediction.run_feature_selection \
    --data-path results/HERO_dataset.csv \
    --output-dir results/feature_selection \
    --random-state 42
```

**Docker (Alternative - May Show Minor Variations):**
```bash
docker run --rm \
    -v $(pwd)/results:/app/results \
    vatsal1729/hero-evcs:latest python3 -m modules.MLPrediction.run_feature_selection \
    --data-path /app/results/HERO_dataset.csv \
    --output-dir /app/results/feature_selection \
    --random-state 42
```

**Output**: `results/feature_selection/selected_features.json`

**Note:** The output directory name (`feature_selection`) is the standard default. You can use any folder name, but ensure consistency when referencing it in subsequent commands.

**Feature Selection Method Details:**
- **Method**: Fast Mutual Information (MI) based selection using `SelectKBest` with `mutual_info_regression`
- **Data Cleaning**: Automatic removal of zero-variance and highly correlated features
- **Selection**: Top features based on mutual information scores with target variable
- **Reproducibility**: Fixed random state (42) ensures consistent results

### Step 2: Train/Retrain Models

**Exact Command Used (Lightning AI L4 - Ubuntu 24.04):**
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

**Native Installation (Alternative):**
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

**Docker (Alternative - May Show Minor Variations):**
```bash
docker run --rm \
    -v $(pwd)/results:/app/results \
    vatsal1729/hero-evcs:latest python3 -m modules.MLPrediction.run_pipeline \
    --data-path /app/results/HERO_dataset.csv \
    --output-dir /app/results/ml_pipeline_output \
    --target-column demand_score_kwh75_hrs25 \
    --pre-selected-features /app/results/feature_selection/selected_features.json \
    --skip-feature-selection \
    --random-state 42 \
    --skip-synthetic-data \
    --skip-spatial-features \
    --limit-target-transformations \
    --force-retrain
```

**Key Parameters:**
- `--target-column demand_score_kwh75_hrs25`: Target variable (75% kWh, 25% hours)
- `--pre-selected-features`: Path to pre-selected features JSON file (same features can be reused for different targets)
- `--skip-feature-selection`: Skip feature selection (use pre-selected features)
- `--random-state 42`: Reproducibility seed (required)
- `--skip-synthetic-data`: Disable synthetic data augmentation
- `--skip-spatial-features`: Skip spatial feature generation (faster for large datasets)
- `--limit-target-transformations`: Use only best transformations (original, log)

**Note:** Feature selection uses **Fast Mutual Information (MI) based method**, not RFE. The pre-selected features file contains features selected using mutual information regression.

**Training Different Target Formulations:**
You can use the same pre-selected features file to train models for different target formulations. Simply change the `--target-column` parameter while keeping the same `--pre-selected-features` path:

```bash
# Train for balanced target (50% kWh, 50% hours)
python -m modules.MLPrediction.run_pipeline \
    --data-path results/HERO_dataset.csv \
    --target-column demand_score_balanced \
    --pre-selected-features results/feature_selection/selected_features.json \
    --skip-feature-selection \
    --random-state 42 \
    --skip-synthetic-data \
    --skip-spatial-features \
    --limit-target-transformations \
    --force-retrain

# Train for kWh-only target
python -m modules.MLPrediction.run_pipeline \
    --data-path results/HERO_dataset.csv \
    --target-column demand_score_kwh_only \
    --pre-selected-features results/feature_selection/selected_features.json \
    --skip-feature-selection \
    --random-state 42 \
    --skip-synthetic-data \
    --skip-spatial-features \
    --limit-target-transformations \
    --force-retrain

# Train for hours-only target
python -m modules.MLPrediction.run_pipeline \
    --data-path results/HERO_dataset.csv \
    --target-column demand_score_hours_only \
    --pre-selected-features results/feature_selection/selected_features.json \
    --skip-feature-selection \
    --random-state 42 \
    --skip-synthetic-data \
    --skip-spatial-features \
    --limit-target-transformations \
    --force-retrain
```

**Available Target Formulations:**
- `demand_score_balanced` (50% kWh, 50% hours)
- `demand_score_kwh_only` (100% kWh, 0% hours)
- `demand_score_hours_only` (0% kWh, 100% hours)
- `demand_score_kwh25_hrs75` (25% kWh, 75% hours)
- `demand_score_kwh75_hrs25` (75% kWh, 25% hours)

**Expected Output:**
- Trained models in `results/ml_pipeline_output/models/`
- Evaluation metrics in `results/ml_pipeline_output/evaluation/`
- Feature importance plots

### Step 3: Validate Model Performance

#### Option A: Validate Single Model

**Native Installation:**
```bash
python evaluation/demand_validation/run_validation.py \
    --grid-size 1.0 \
    --model-path models/demand_score_kwh_only_original_quantile_xgboost.pkl \
    --output-dir results/validation_results
```

**Docker:**
```bash
docker run --rm \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    vatsal1729/hero-evcs:latest python3 evaluation/demand_validation/run_validation.py \
    --grid-size 1.0 \
    --model-path /app/models/demand_score_kwh_only_original_quantile_xgboost.pkl \
    --output-dir /app/results/validation_results
```

#### Option B: Validate All Models (Recommended for Comprehensive Validation)

**Native Installation:**
```bash
python evaluation/demand_validation/run_validation.py \
    --validate-all-models \
    --grid-size 1.0 \
    --models-dir results/models \
    --output-dir results/validation_results
```

**With Target Filter (Optional):**
```bash
# Validate only models for specific target formulation
python evaluation/demand_validation/run_validation.py \
    --validate-all-models \
    --grid-size 1.0 \
    --target-filter demand_score_kwh_only \
    --models-dir results/models \
    --output-dir results/validation_results
```

**Docker:**
```bash
docker run --rm \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    vatsal1729/hero-evcs:latest python3 evaluation/demand_validation/run_validation.py \
    --validate-all-models \
    --grid-size 1.0 \
    --models-dir /app/models \
    --output-dir /app/results/validation_results
```

**Key Parameters:**
- `--validate-all-models`: Validate all models in the models directory (required flag)
- `--grid-size 1.0`: Grid size in kilometers for validation (default: 1.0)
- `--models-dir results/models`: Directory containing model PKL files (default: `results/models`)
- `--target-filter`: Optional filter by target formulation (e.g., `demand_score_kwh_only`, `demand_score_balanced`)
- `--limit`: Optional limit on number of models to validate (for testing)
- `--output-dir`: Output directory for validation results (default: `validation_results`)

**Expected Results:**
- **Single Model**: Spearman's ρ ≈ 0.626 (geographic portability validation), R² Score ≈ 0.35-0.40
- **All Models**: Comprehensive validation report with metrics for all models
- Detailed validation reports saved to `results/validation_results/`
- Summary CSV with all model performance metrics

## Reproducing RL Evaluation Results

### Comprehensive Multi-Grid Evaluation

**Exact Command Used (Google Cloud VM - Ubuntu 24.04, n2-custom-12-49152):**
```bash
python evaluation/microplacement_validation/run_all_grids_evaluation.py \
    --seed 42 \
    --station-budget 150 \
    --adaptive-mode \
    --confidence-threshold 0.95 \
    --resume \
    --verbose
```

**Native Installation (Alternative):**
```bash
python evaluation/microplacement_validation/run_all_grids_evaluation.py \
    --seed 42 \
    --station-budget 150 \
    --adaptive-mode \
    --confidence-threshold 0.95 \
    --resume \
    --verbose
```

**Docker:**
```bash
docker run --rm \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/generated_files:/app/generated_files \
    vatsal1729/hero-evcs:latest python3 evaluation/microplacement_validation/run_all_grids_evaluation.py \
    --seed 42 \
    --station-budget 150 \
    --adaptive-mode \
    --confidence-threshold 0.95 \
    --resume \
    --verbose
```

**Key Parameters:**
- `--seed 42`: Reproducibility seed (required)
- `--station-budget 150`: Total stations to allocate (matches paper)
- `--adaptive-mode`: Enable adaptive episode calculation
- `--confidence-threshold 0.95`: Statistical significance level
- `--resume`: Resume from checkpoints if available
- `--verbose`: Detailed logging

**Docker Data Mounting Notes:**
- `results/`: For outputs and checkpoints
- `models/`: For pre-trained demand prediction models
- `data/`: For validation data (VED dataset)
- `generated_files/`: For SUMO network files (if pre-generated)

**Expected Results:**
- Hybrid methods (UCB, Thompson Sampling, ε-Greedy): Composite reward ~0.72
- Baseline methods (K-Means, Random, Uniform): Composite reward ~0.36-0.41
- Statistical significance: p < 0.05 for hybrid vs. baselines
- Results in `results/evaluation_results/`

### Single Grid Evaluation (For Testing)

**Native Installation:**
```bash
python evaluation/microplacement_validation/run_evaluation.py \
    --grid-id grid_1 \
    --station-budget 10 \
    --methods ucb thompson_sampling kmeans \
    --seed 42
```

**Docker:**
```bash
docker run --rm \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/generated_files:/app/generated_files \
    vatsal1729/hero-evcs:latest python3 evaluation/microplacement_validation/run_evaluation.py \
    --grid-id grid_1 \
    --station-budget 10 \
    --methods ucb thompson_sampling kmeans \
    --seed 42
```

## Reproducing Main Pipeline Results

### Case Study 1: São Paulo, Brazil

**Exact Command Used (Lightning AI L4 - Ubuntu 24.04):**
```bash
python main.py \
    --city "Sao Paulo" \
    --total-stations 25 \
    --optimization-method ucb \
    --max-episodes 10
```

**Full Command with All Options (Alternative):**
```bash
python main.py \
    --city "Sao Paulo" \
    --total-stations 25 \
    --optimization-method ucb \
    --max-episodes 10 \
    --random-seed 42 \
    --output-dir results/sao_paulo \
    --visualize \
    --export-json
```

**Docker:**
```bash
docker run --rm \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/generated_files:/app/generated_files \
    vatsal1729/hero-evcs:latest python3 main.py \
    --city "Sao Paulo, Brazil" \
    --total-stations 10 \
    --optimization-method ucb \
    --max-episodes 10 \
    --random-seed 42 \
    --output-dir /app/results/sao_paulo \
    --visualize \
    --export-json
```

**Expected Output:**
- Station locations in `results/sao_paulo/station_coordinates.json`
- Visualization maps in `results/sao_paulo/`
- Performance metrics in `results/sao_paulo/performance_metrics.json`

### Case Study 2: Mumbai, India

**Exact Command Used (Lightning AI L4 - Ubuntu 24.04):**
```bash
python main.py \
    --city "Mumbai" \
    --total-stations 25 \
    --optimization-method ucb \
    --max-episodes 10
```

**Full Command with All Options (Alternative):**
```bash
python main.py \
    --city "Mumbai" \
    --total-stations 25 \
    --optimization-method ucb \
    --max-episodes 10 \
    --random-seed 42 \
    --output-dir results/mumbai \
    --visualize \
    --export-json
```

**Docker:**
```bash
docker run --rm \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/generated_files:/app/generated_files \
    vatsal1729/hero-evcs:latest python3 main.py \
    --city "Mumbai, India" \
    --total-stations 50 \
    --optimization-method thompson_sampling \
    --max-episodes 15 \
    --random-seed 42 \
    --output-dir /app/results/mumbai \
    --visualize \
    --export-json
```

### Case Study 3: Ann Arbor, USA (Validation)

**Native Installation:**
```bash
python main.py \
    --city "Ann Arbor, Michigan, USA" \
    --total-stations 75 \
    --optimization-method epsilon_greedy \
    --adaptive-mode \
    --max-episodes 20 \
    --random-seed 42 \
    --output-dir results/ann_arbor \
    --visualize \
    --export-json
```

**Docker:**
```bash
docker run --rm \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/generated_files:/app/generated_files \
    vatsal1729/hero-evcs:latest python3 main.py \
    --city "Ann Arbor, Michigan, USA" \
    --total-stations 75 \
    --optimization-method epsilon_greedy \
    --adaptive-mode \
    --max-episodes 20 \
    --random-seed 42 \
    --output-dir /app/results/ann_arbor \
    --visualize \
    --export-json
```

**Note:** For Ann Arbor validation, the `data/` mount includes VED validation data for comparison. Ann Arbor results are used for validation against VED ground truth data.

## Reproducing Validation Results

### Geographic Portability Validation

```bash
python evaluation/demand_validation/run_validation.py \
    --grid-size 1.0 \
    --model-path models/demand_score_kwh_only_original_quantile_xgboost.pkl \
    --validation-data data/validation/ved_processed_with_grids.parquet \
    --output-dir results/portability_validation
```

**Expected Result:**
- **Spearman's ρ = 0.626** (validates geographic portability)
- Correlation plots and validation report

## Expected Outputs

### ML Pipeline Outputs

```
results/ml_pipeline_output/
├── models/
│   └── demand_score_*.pkl          # Trained models
├── evaluation/
│   ├── metrics.json                # Evaluation metrics
│   ├── feature_importance.png     # Feature importance plot
│   └── model_comparison.png        # Model comparison
└── logs/
    └── pipeline.log                # Execution log
```

### RL Evaluation Outputs

```
results/evaluation_results/
├── comparison_report.txt           # Human-readable comparison
├── metrics_summary.json            # Comprehensive metrics
├── statistical_analysis.json       # Statistical tests
├── grid_1/
│   ├── ucb_results.json
│   ├── thompson_sampling_results.json
│   └── ...
└── checkpoints/                    # Resumable checkpoints
```

### Main Pipeline Outputs

```
results/{city_name}/
├── station_coordinates.json        # GPS coordinates
├── performance_metrics.json       # Performance data
├── policy_summary.json            # Policy-maker summary
├── visualization_map.html          # Interactive map
└── summary_report.txt             # Technical report
```

## Reproducing Paper Figures

### Figure 7: Case Study Maps

```bash
# Generate maps for all case studies (exact commands used in paper)
python main.py --city "Sao Paulo" --total-stations 25 --optimization-method ucb --max-episodes 10
python main.py --city "Mumbai" --total-stations 25 --optimization-method ucb --max-episodes 10
python main.py --city "Ann Arbor, Michigan, USA" --total-stations 75 --optimization-method epsilon_greedy --adaptive-mode --max-episodes 20 --random-seed 42
```

**Output**: `results/{city}/visualization_map.html`

## Troubleshooting

### Issue: SUMO Not Found

```bash
# Verify SUMO installation
which sumo
sumo --version

# Set SUMO_HOME
export SUMO_HOME=/usr/share/sumo
export PATH=$PATH:$SUMO_HOME/bin
```

### Issue: Model File Not Found

```bash

# Or retrain models
python -m modules.MLPrediction.run_pipeline --force-retrain ...
```

### Issue: Memory Errors

```bash
# Reduce parallel workers
python evaluation/microplacement_validation/run_all_grids_evaluation.py \
    --workers 2 \
    ...

# Or process grids sequentially
python evaluation/microplacement_validation/run_evaluation.py \
    --grid-id grid_1 \
    ...
```

### Issue: Different Results

1. **Check Seed**: Ensure `--seed 42` or `--random-state 42` is used
2. **Check Data**: Verify data files match expected format
3. **Check Versions**: Ensure Python and library versions match
4. **Clear Cache**: Clear cache if using cached results
   ```bash
   python -m modules.utils.manage_cache --clear-all
   ```

## Verification Checklist

- [ ] All commands executed with seed 42
- [ ] Pre-trained models available or models retrained
- [ ] SUMO properly installed and configured
- [ ] Data files in correct locations
- [ ] Results match expected metrics:
  - [ ] Spearman's ρ ≈ 0.626 (validation)
  - [ ] Hybrid RL reward ~0.72
  - [ ] Baseline reward ~0.36-0.41
- [ ] All output files generated
- [ ] Statistical significance confirmed (p < 0.05)

## Citation

## Reproducibility Notes

### Key Points for Reproducing Results

1. **Exact Commands**: The commands marked as "Exact Command Used" were the ones used to generate paper results on specific cloud environments.

2. **Environment Differences**:
   - **Paper Results**: Generated on Ubuntu 24.04 (Lightning AI L4 for ML/Main, Google Cloud VM for RL)
   - **Docker Image**: Ubuntu 24.04 (matches paper environment, should minimize variations)
   - **Local Environments**: May vary based on OS version and Python environment (conda/venv)

3. **Expected Variations**:
   - **ML Metrics**: ±0.01-0.02 in R², MAE, RMSE due to different random number generators
   - **RL Rewards**: ±0.01-0.03 in composite rewards
   - **Statistical Significance**: Should remain consistent (p < 0.05)
   - **Random Operators**: NumPy, Python's `random`, and OS-level RNGs may produce slightly different sequences

4. **Why Variations Occur**:
   - Different Python/NumPy random number generator implementations across platforms
   - Different OS-level random number generators (Linux vs different Ubuntu versions)
   - Floating-point precision differences
   - Different library versions (if not using Docker)
   - Conda vs venv vs Docker environment differences

5. **Reproducibility Best Practices**:
   - Always use `--seed 42` or `--random-state 42` (as shown in all commands)
   - Use the same Python version (3.10 recommended)
   - Use the same library versions (Docker ensures this)
   - Note that exact bit-for-bit reproducibility may not be possible across different platforms

6. **Docker vs Native**:
   - Docker (Ubuntu 24.04) now matches the paper environment and should produce very similar results
   - Native installation on Ubuntu 24.04 (matching paper environment) should produce identical results
   - Both are acceptable for reproduction; focus on statistical significance rather than exact numerical matches

### Data File Names

**Standard Dataset Name:** `results/HERO_dataset.csv`

This is the processed training dataset available on Kaggle. It's the output of `preprocess_data.py` and contains all the features and demand score variations needed for ML model training.

## Contact

For issues with reproducibility:
- Open a GitHub issue
- Contact the paper authors
- Check the main README.md for general troubleshooting

---



