# HERO: Hybrid Framework for Portable EV Charging Station Placement

[![License](https://img.shields.io/badge/license-Proprietary-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![SUMO](https://img.shields.io/badge/SUMO-1.15%2B-orange.svg)](https://sumo.dlr.de/)
[![Docker](https://img.shields.io/badge/docker-available-blue.svg)](https://hub.docker.com/r/vatsal1729/hero-evcs)

**HERO** (Hybrid EV RL Optimization) is a comprehensive, research-grade framework for optimizing Electric Vehicle (EV) charging station placement in urban environments. The framework combines geospatial machine learning for demand prediction with reinforcement learning optimization to deliver portable, data-driven placement recommendations for any city worldwide.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Research Contributions](#research-contributions)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Data Management](#data-management)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

## Overview

HERO addresses the critical challenge of EV charging infrastructure planning through a two-stage, data-driven approach:

1. **Stage 1 - Demand Prediction**: Geospatial ML models trained on 613,108 grid samples from OpenStreetMap (OSM) data predict grid-level charging demand across urban landscapes. Normalized proportional allocation distributes city-wide station budgets across grid cells.

2. **Stage 2 - Micro-Placement Optimization**: A reinforcement learning controller structured as a multi-armed bandit executes tactical micro-placement within allocated grids. POI- and network-based heuristics generate candidate locations, evaluated through high-fidelity SUMO traffic simulations.

### Research Validation

- **Geographic Portability**: Validated against Ann Arbor Vehicle Energy Dataset (VED) with strong rank-order correlation (Spearman's ρ = 0.626)
- **Performance**: Hybrid RL approach consistently outperforms heuristic baselines, achieving composite rewards ~0.72 vs. 0.36-0.41 for traditional methods
- **Case Studies**: End-to-end evaluation in Ann Arbor, Mumbai, and São Paulo
- **Data Independence**: Operates exclusively on ubiquitous OSM features, eliminating proprietary data dependencies

## Key Features

### 🤖 Intelligent Placement Algorithms
- **Hybrid Bandit Framework**: Combines heuristic knowledge with reinforcement learning
- **Multi-Armed Bandit Methods**: UCB, ε-greedy, and Thompson sampling algorithms
- **Adaptive Exploration**: Dynamic episode calculation based on action space
- **Baseline Comparisons**: K-Means clustering, Random placement, Uniform spacing

### 📊 Comprehensive Evaluation
- **Simulation-Based Validation**: SUMO traffic simulation integration for realistic evaluation
- **Statistical Analysis**: Confidence intervals, significance testing, robust metrics
- **Multi-Grid Evaluation**: Parallel processing across multiple urban grids
- **Reproducible Results**: Deterministic seeding and consistent random states

### 🔧 Production-Ready Features
- **Parallel Processing**: Multi-grid evaluation with worker pools
- **Checkpoint System**: Resumable evaluation runs with optimized storage (90% reduction)
- **Portable Design**: Works with any city worldwide using only OpenStreetMap data
- **Comprehensive Logging**: Detailed execution logs and performance metrics

## Research Contributions

1. **Portable Demand Prediction**: ML models trained exclusively on OSM features enable deployment to any city without proprietary data
2. **Hybrid RL Framework**: Novel combination of heuristic initialization with bandit-based optimization
3. **Simulation-Based Validation**: High-fidelity SUMO integration for realistic performance evaluation
4. **Reproducible Pipeline**: Complete open-source framework for research reproducibility

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HERO Framework                            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Stage 1: Demand Prediction (ML Pipeline)                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  OSM Feature Extraction → ML Model → Demand Scores │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ↓                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Proportional Allocation → Grid Budget Distribution │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ↓                                    │
│  Stage 2: Micro-Placement (RL Optimization)                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Candidate Generation → Bandit Selection → SUMO     │    │
│  │  Simulation → Reward Calculation → Policy Update    │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ↓                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Final Station Locations + Performance Metrics       │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8 or higher (for native installation)
- Docker Desktop (for Docker installation - recommended for Windows/macOS)
- 16GB+ RAM recommended for parallel processing
- SSD storage recommended for better I/O performance

### Installation

#### Docker (Cross-Platform - Recommended for Windows/macOS)

Docker provides a consistent environment across Windows, macOS, and Linux. **This is the easiest option for Windows users!**

**Prerequisites:**
- Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) for your OS

**Option 1: Use Pre-built Image (Recommended - Fastest)**

```bash
# Clone the repository
git clone git@github.com:Vaav-ai/HERO-EVCS.git
cd HERO-EVCS

# Pull the pre-built image from Docker Hub (no build time!)
docker pull vatsal1729/hero-evcs:latest

# Run the container with volume mounts
docker run -it --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/generated_files:/app/generated_files \
    -v $(pwd)/models:/app/models \
    vatsal1729/hero-evcs:latest

# Run commands directly (no need to enter container)
docker run --rm \
    -v $(pwd)/results:/app/results \
    vatsal1729/hero-evcs:latest python3 main.py \
    --city "Singapore" \
    --total-stations 50

# Or enter container interactively
docker run -it --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/results:/app/results \
    vatsal1729/hero-evcs:latest /bin/bash
```

**Windows PowerShell (Pre-built):**
```powershell
# Clone the repository
git clone git@github.com:Vaav-ai/HERO-EVCS.git
cd HERO-EVCS

# Pull the pre-built image
docker pull vatsal1729/hero-evcs:latest

# Run commands directly (PowerShell syntax)
docker run --rm `
    -v ${PWD}/results:/app/results `
    vatsal1729/hero-evcs:latest python3 main.py `
    --city "Singapore" `
    --total-stations 50
```

**Option 2: Build Locally (For Latest Code or Customization)**

If you want the latest code from the repository or need to customize the image:

```bash
# Clone the repository
git clone git@github.com:Vaav-ai/HERO-EVCS.git
cd HERO-EVCS

# Build the Docker image (this may take 10-15 minutes)
docker build -t hero-evcs .

# Run the container with volume mounts
docker run -it --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/generated_files:/app/generated_files \
    -v $(pwd)/models:/app/models \
    hero-evcs
```

**Option 3: Using Docker Compose (Alternative)**

For easier volume management and service configuration:

```bash
# Build and run with docker-compose
docker-compose build
docker-compose run --rm hero-evcs python3 main.py \
    --city "Singapore" \
    --total-stations 50

# Or use the test service
docker-compose run --rm hero-evcs-test
```

**Docker Compose Configuration:**
- Pre-configured volume mounts for `data/`, `results/`, `generated_files/`, and `cache/`
- Environment variables set automatically (SUMO_HOME, PYTHONPATH)
- Test service included for quick validation

> **Note:** 
> - Docker containers include SUMO and all dependencies pre-installed. No manual setup required!
> - The Docker image includes: Ubuntu 24.04, Python 3.10, SUMO (latest stable), OSMnx 1.9.4, and all required Python packages
> - The `data/` folder is excluded from the Docker image (to keep it small) and should be mounted as a volume if you have data
> - Pre-trained models from `models/` are bundled inside the image so the pipeline works out-of-the-box; mount `models/` only if you want to override them
> - If you don't have data, the pipeline can generate synthetic data automatically
> - To run commands, use: `docker run --rm -v $(pwd)/results:/app/results vatsal1729/hero-evcs:latest python3 main.py --city "Singapore" --total-stations 50`

#### Ubuntu/Debian Linux (Automated Setup)

The `setup.sh` script automatically installs SUMO, all system dependencies, Python packages, and sets up the environment:

```bash
# Clone the repository
git clone git@github.com:Vaav-ai/HERO-EVCS.git
cd HERO-EVCS

# Run the setup script (installs everything automatically)
chmod +x setup.sh
./setup.sh

# Activate the virtual environment (created by setup.sh)
source venv/bin/activate

# Verify SUMO installation
python -c "import traci, sumolib; print('SUMO OK')"
```

#### macOS (Manual Setup)

```bash
# Clone the repository
git clone git@github.com:Vaav-ai/HERO-EVCS.git
cd HERO-EVCS

# Install SUMO and dependencies via Homebrew
brew install sumo
brew install geos proj gdal gcc

# Set SUMO environment variables
export SUMO_HOME=$(brew --prefix sumo)
export PATH=$PATH:$SUMO_HOME/bin
export PYTHONPATH=$PYTHONPATH:$SUMO_HOME/tools

# Add to ~/.zshrc or ~/.bash_profile for persistence
echo 'export SUMO_HOME=$(brew --prefix sumo)' >> ~/.zshrc
echo 'export PATH=$PATH:$SUMO_HOME/bin' >> ~/.zshrc
echo 'export PYTHONPATH=$PYTHONPATH:$SUMO_HOME/tools' >> ~/.zshrc

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify SUMO installation
python -c "import traci, sumolib; print('SUMO OK')"
```

#### Windows (Manual Setup)

**Option 1: Using Docker (Easiest - Recommended)**
```powershell
# Follow the Docker installation instructions above
# This is the simplest way to get started on Windows!
```

**Option 2: Using WSL2**
```bash
# Install WSL2 with Ubuntu, then follow Ubuntu/Debian instructions above
# Use setup.sh in WSL2 environment
```

**Option 3: Native Windows Installation**
```powershell
# Clone the repository
git clone git@github.com:Vaav-ai/HERO-EVCS.git
cd HERO-EVCS

# Download and install SUMO from: https://sumo.dlr.de/docs/Installing/index.html
# Add SUMO to PATH environment variable

# Set SUMO environment variables (adjust path to your SUMO installation)
$env:SUMO_HOME = "C:\Program Files\sumo"
$env:PATH += ";$env:SUMO_HOME\bin"
$env:PYTHONPATH += ";$env:SUMO_HOME\tools"

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify SUMO installation
python -c "import traci, sumolib; print('SUMO OK')"
```

> **Note:** For Windows, Docker is the easiest option (no WSL2 or manual SUMO installation needed). WSL2 is also a good alternative. Native Windows installation may require additional configuration.

## Quick Start

### Basic Usage (No Data Required!)

The framework works out-of-the-box with pre-trained models and synthetic data generation:

```bash
# Quick start with default settings
python main.py --city "Singapore" --total-stations 50

# Specify optimization method
python main.py --city "Mumbai, India" --total-stations 100 --optimization-method ucb --max-episodes 10

# Full customization
python main.py \
    --city "Sao Paulo, Brazil" \
    --total-stations 150 \
    --optimization-method thompson_sampling \
    --max-episodes 20 \
    --output-dir results/sao_paulo \
    --visualize \
    --export-json
```

## Usage Examples

### 1. Main Pipeline (`main.py`)

The main pipeline orchestrates the complete workflow from demand prediction to station placement:

```bash
# Example 1: São Paulo with UCB optimization
python main.py \
    --city "Sao Paulo, Brazil" \
    --total-stations 25 \
    --optimization-method ucb \
    --max-episodes 10 \
    --output-dir results/sao_paulo \
    --visualize

# Example 2: Mumbai with Thompson Sampling
python main.py \
    --city "Mumbai, India" \
    --total-stations 25 \
    --optimization-method thompson_sampling \
    --max-episodes 10 \
    --output-dir results/mumbai \
    --visualize \
    --export-json

# Example 3: Ann Arbor with adaptive mode
python main.py \
    --city "Ann Arbor, Michigan, USA" \
    --total-stations 75 \
    --optimization-method epsilon_greedy \
    --adaptive-mode \
    --random-seed 42 \
    --output-dir results/ann_arbor
```

**Key Arguments:**
- `--city`: City name (e.g., "Mumbai, India", "Sao Paulo, Brazil")
- `--total-stations`: Total number of charging stations to place
- `--optimization-method`: Bandit algorithm (`ucb`, `epsilon_greedy`, `thompson_sampling`)
- `--max-episodes`: Maximum number of RL episodes
- `--adaptive-mode`: Enable adaptive episode calculation
- `--output-dir`: Output directory for results
- `--visualize`: Generate visualization maps
- `--export-json`: Export station coordinates as JSON

### 2. ML Pipeline (`modules/MLPrediction/run_pipeline.py`)

Train or retrain demand prediction models:

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

**Key Arguments:**
- `--data-path`: Path to training data CSV
- `--output-dir`: Output directory for models and results
- `--target-column`: Target variable column name (see available targets below)
- `--pre-selected-features`: Path to JSON file with pre-selected features (same features can be reused for different targets)
- `--skip-feature-selection`: Skip feature selection (use pre-selected features)
- `--random-state`: Random seed for reproducibility
- `--force-retrain`: Force retraining even if model exists

**Training Different Target Formulations:**
You can use the same pre-selected features file to train models for different target formulations. Simply change the `--target-column` parameter:

```bash
# Train for different target using same features
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
```

**Available Target Formulations:**
- `demand_score_balanced` (50% kWh, 50% hours)
- `demand_score_kwh_only` (100% kWh, 0% hours)
- `demand_score_hours_only` (0% kWh, 100% hours)
- `demand_score_kwh25_hrs75` (25% kWh, 75% hours)
- `demand_score_kwh75_hrs25` (75% kWh, 25% hours)

### 3. RL Evaluation (`evaluation/microplacement_validation/run_all_grids_evaluation.py`)

Comprehensive multi-grid evaluation with statistical analysis:

```bash
python evaluation/microplacement_validation/run_all_grids_evaluation.py \
    --seed 42 \
    --station-budget 150 \
    --adaptive-mode \
    --confidence-threshold 0.95 \
    --resume \
    --verbose
```

**Key Arguments:**
- `--seed`: Random seed for reproducibility
- `--station-budget`: Total number of stations to allocate
- `--adaptive-mode`: Enable adaptive episode calculation
- `--confidence-threshold`: Confidence level for statistical tests (default: 0.95)
- `--resume`: Resume from checkpoints if available
- `--verbose`: Enable verbose logging
- `--workers`: Number of parallel workers (default: CPU count - 1)

**Additional Options:**
```bash
# Override episode count for testing
--override-episodes 1000

# Clear existing checkpoints
--clear-checkpoints

# Optimize checkpoint storage
--optimize-checkpoints
```

## Data Management

### Quick Start (No Data Required!)

For basic usage with `main.py`, **no datasets are required**:
- ✅ Pre-trained models (included or auto-downloaded)
- ✅ Synthetic trajectory generation (automatic)
- ✅ OpenStreetMap data (downloaded on-the-fly)

### Data Requirements by Use Case

| Use Case | Required Data | Location |
|----------|--------------|----------|
| **Basic Planning** (`main.py`) | None (uses pre-trained models) | N/A |
| **Model Retraining** | Training datasets | `data/training/` |
| **Paper Reproduction** | All datasets | See data/training/ and data/validation/ directories |

### Data Sources

- **Training Data**: UrbanEV and ST-EVCDP datasets (core datasets included in `data/training/`)
- **Validation Data**: VED (Vehicle Energy Dataset) processed data
- **Pre-trained Models**: Available in `models/` directory (included in repository)
- **Processed Dataset**: Pre-processed training dataset (`HERO_dataset.csv`) available on [Kaggle](https://www.kaggle.com/datasets/vatsal1729/hero-geospatial-ev-charging-demand-dataset)

### Download Options

**Processed Training Dataset (Recommended for ML Model Training):**

The processed dataset from `preprocess_data.py` can be downloaded directly from Kaggle:

```bash
# Install Kaggle CLI (if not already installed)
pip install kaggle

# Download the processed dataset
kaggle datasets download -d vatsal1729/hero-geospatial-ev-charging-demand-dataset
unzip hero-geospatial-ev-charging-demand-dataset.zip

# Extract HERO_dataset.csv to results/ directory
cp HERO_dataset.csv results/HERO_dataset.csv

# The dataset is ready to use with run_pipeline.py
```

**Note:** Core training datasets (UrbanEV and ST-EVCDP) are already included in `data/training/`. The processed dataset on Kaggle is the output of `preprocess_data.py` and can be used directly for training ML models without running preprocessing.

## Project Structure

```
HERO-EVCS/
├── main.py                          # Main entry point for complete pipeline
├── modules/                         # Core implementation modules
│   ├── __init__.py                 # Module initialization
│   ├── MLPrediction/               # Stage 1: Demand prediction pipeline
│   │   ├── __init__.py            # Package initialization
│   │   ├── run_pipeline.py        # ML pipeline CLI runner
│   │   ├── demand_prediction_pipeline.py  # Main orchestrator
│   │   ├── demand_predictor.py    # Inference interface
│   │   ├── demand_score_calculator.py  # Demand score computation
│   │   ├── feature_engineering.py  # Urban feature engineering
│   │   ├── hyperparameter_tuning.py  # Hyperparameter optimization
│   │   ├── model_training.py      # Model training and evaluation
│   │   ├── portable_osm_extractor.py  # OSM feature extraction
│   │   ├── preprocess_data.py     # Data preprocessing utilities
│   │   ├── run_feature_selection.py  # Feature selection runner
│   │   ├── run_hyperparameter_tuning.py  # Hyperparameter tuning runner
│   │   ├── synthetic_data_generator.py  # Synthetic trajectory generation
│   │   ├── target_transformation.py  # Target variable transformations
│   │   └── README.md              # ML pipeline documentation
│   ├── RLOptimization/            # Stage 2: Reinforcement learning optimization
│   │   ├── __init__.py            # Package initialization
│   │   ├── HybridBanditFramework.py  # Main RL framework
│   │   ├── BanditOptimizationFramework.py  # Base bandit framework
│   │   ├── SimulationManager.py   # SUMO simulation management
│   │   ├── ChargingStationManager.py  # Station operations
│   │   ├── RouteGenerator.py      # Vehicle route generation
│   │   ├── SimulationAnalyzer.py  # Simulation analysis utilities
│   │   ├── SumoNetwork.py         # SUMO network utilities
│   │   ├── DomainRandomization.py  # Domain randomization for robustness
│   │   ├── HeuristicBaselines.py  # Heuristic baseline methods
│   │   ├── ValidationFramework.py  # Validation framework
│   │   ├── metrics.py             # Performance metrics calculation
│   │   ├── models/                # Bandit algorithm implementations
│   │   │   ├── __init__.py        # Models package initialization
│   │   │   ├── BaseBandit.py      # Base bandit class
│   │   │   ├── UCB.py             # Upper Confidence Bound algorithm
│   │   │   ├── ThompsonSampling.py  # Thompson Sampling algorithm
│   │   │   └── EpsilonGreedy.py   # ε-Greedy algorithm
│   │   └── README.md              # RL framework documentation
│   ├── config/                     # Configuration management
│   │   ├── __init__.py            # Package initialization
│   │   ├── SimulationConfig.py   # SUMO simulation config
│   │   ├── EVConfig.py           # EV fleet config
│   │   └── README.md              # Config documentation
│   └── utils/                      # Utility functions
│       ├── __init__.py            # Package initialization
│       ├── gridding.py            # City gridding utilities
│       ├── seed_utils.py          # Reproducibility utilities
│       ├── cache_utils.py         # Caching utilities
│       ├── manage_cache.py        # Cache management utilities
│       ├── gpu_utils.py           # GPU utility functions
│       ├── log_configs.py        # Logging configuration
│       └── README.md              # Utils documentation
├── evaluation/                     # Evaluation framework
│   ├── microplacement_validation/  # Multi-grid RL evaluation
│   │   ├── run_all_grids_evaluation.py  # Main evaluation runner
│   │   ├── run_evaluation.py     # Single grid evaluation runner
│   │   ├── data_loader.py         # Data loading utilities
│   │   ├── execution_engine.py    # Method execution engine
│   │   ├── metrics_analyzer.py    # Metrics calculation and analysis
│   │   ├── visualization.py      # Visualization utilities
│   │   └── README.md              # Evaluation documentation
│   ├── demand_validation/         # ML demand validation
│   │   ├── run_validation.py     # Validation script
│   │   └── README.md             # Validation documentation
│   └── data_prep/                 # Data preparation utilities
│       ├── data_preparation.py    # Data prep functions
│       └── README.md              # Data prep documentation
├── data/                           # Data directory (training and validation datasets)
│   ├── training/                  # Training datasets (optional)
│   │   ├── UrbanEV-main/         # UrbanEV dataset
│   │   │   ├── adj.csv           # Adjacency matrix
│   │   │   ├── distance.csv      # Distance matrix
│   │   │   ├── duration.csv      # Duration matrix
│   │   │   ├── e_price.csv       # Electricity price data
│   │   │   ├── inf.csv           # Information data
│   │   │   ├── occupancy.csv     # Occupancy data
│   │   │   ├── poi.csv           # Points of interest data
│   │   │   ├── s_price.csv       # Service price data
│   │   │   ├── volume.csv        # Volume data
│   │   │   ├── volume-11kW.csv   # 11kW volume data
│   │   │   └── weather_*.csv      # Weather data files
│   │   └── ST-EVCDP-main/         # ST-EVCDP dataset
│   │       ├── adj.csv           # Adjacency matrix
│   │       ├── distance.csv      # Distance matrix
│   │       ├── duration.csv      # Duration matrix
│   │       ├── information.csv   # Information data
│   │       ├── occupancy.csv     # Occupancy data
│   │       ├── price.csv         # Price data
│   │       ├── time.csv          # Time data
│   │       ├── volume.csv        # Volume data
│   │       ├── Shenzhen.qgz      # Shenzhen geospatial data
│   │       ├── SZ_districts/     # Shenzhen district shapefiles
│   │       └── SZweather*.xls    # Weather data
│   └── validation/                # Validation datasets (optional)
│       └── ved_processed_with_grids.parquet  # VED validation data
├── models/                         # Pre-trained ML models
│   ├── demand_score_balanced_*.pkl  # Balanced demand score models
│   ├── demand_score_hours_only_*.pkl  # Hours-only demand score models
│   ├── demand_score_kwh_only_*.pkl  # kWh-only demand score models
│   └── demand_score_kwh*_hrs*_*.pkl  # Weighted demand score models
├── results/                        # Output directory (generated)
│   ├── evaluation_results/        # RL evaluation outputs
│   ├── ml_pipeline_output/        # ML pipeline outputs
│   └── {city_name}/               # City-specific results
│       └── checkpoints/           # Evaluation checkpoints
├── generated_files/                # Generated network files
│   └── city_network/              # SUMO network files
│       ├── {city}.osm.xml         # OSM network file
│       ├── {city}.osm.net.xml     # SUMO network file
│       └── {city}.osm_grid_constrained.net.xml  # Grid-constrained network
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker image definition
├── docker-compose.yml             # Docker Compose configuration
├── setup.sh                       # Automated setup script (Linux)
├── README.md                      # This file
└── REPRODUCIBILITY.md             # Reproducibility guide
```

### Key Directories

- **`modules/`**: Core implementation (ML prediction, RL optimization, config, utils)
- **`evaluation/`**: Evaluation and validation frameworks
- **`data/`**: Training and validation datasets (see data/training/ and data/validation/ directories)
- **`models/`**: Pre-trained ML models for demand prediction
- **`results/`**: Generated outputs (created during execution)
- **`generated_files/`**: Generated SUMO network files

## Contact & Support

- **Repository**: [GitHub Repository](https://github.com/Vaav-ai/HERO-EVCS)
- **Docker Hub**: [vatsal1729/hero-evcs](https://hub.docker.com/r/vatsal1729/hero-evcs)
- **Issues**: Please use GitHub Issues for bug reports and feature requests
- **Paper**: Under review - contact authors for preprints

## Acknowledgments

We acknowledge the open-source communities behind SUMO, OpenStreetMap, and the various ML libraries that made this work possible.

---

