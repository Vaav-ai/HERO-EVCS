# Multi-Grid Evaluation Framework for EV Charging Station Placement

## Overview

This module provides a comprehensive evaluation framework for assessing EV charging station placement methods across multiple urban grid cells. The framework supports parallel evaluation of hybrid RL methods (UCB, Thompson Sampling, ε-Greedy) and baseline methods (K-Means, Random, Uniform) with statistical significance testing and robust metrics analysis.

## Table of Contents

1. [Architecture](#architecture)
2. [Components](#components)
3. [Usage](#usage)
4. [Configuration](#configuration)
5. [Output and Results](#output-and-results)
6. [Performance Metrics](#performance-metrics)
7. [Troubleshooting](#troubleshooting)

## Architecture

### Module Structure

```
evaluation/microplacement_validation/
├── run_all_grids_evaluation.py    # Main evaluation runner (multi-grid)
├── run_evaluation.py              # Single-grid evaluation
├── data_loader.py                 # Data loading and preparation (DataLoader)
├── execution_engine.py            # Method execution and simulation (ExecutionEngine)
├── metrics_analyzer.py            # Metrics calculation and analysis (MetricsAnalyzer)
├── visualization.py               # Result visualization (ChargingStationVisualizer)
└── README.md                      # This file
```

### File Descriptions

- **`run_all_grids_evaluation.py`**: Main script for comprehensive multi-grid evaluation with parallel processing
- **`run_evaluation.py`**: Single-grid evaluation script for testing and debugging
- **`data_loader.py`**: DataLoader class for loading VED data, creating grids, and preparing trajectories
- **`execution_engine.py`**: ExecutionEngine class for running hybrid and baseline methods with SUMO simulation
- **`metrics_analyzer.py`**: MetricsAnalyzer class for calculating comprehensive metrics and statistical analysis
- **`visualization.py`**: Visualization utilities for generating maps and charts

### Evaluation Flow

```
Data Loading → Grid Preparation → Method Execution → SUMO Simulation → Metrics Analysis → Results Export
```

## Components

### 1. Data Loader (`data_loader.py`)

**Purpose**: Handles all data loading, preparation, and validation

**Key Methods:**
- `load_test_data()`: Loads VED data, creates city grids, prepares trajectory data
- `_create_enhanced_trajectory_data()`: Creates enhanced trajectory data for simulation
- `validate_data()`: Validates that all required data is available and properly formatted

**Features:**
- VED dataset integration
- Grid cell generation
- Trajectory data preparation
- Data validation and error handling

### 2. Execution Engine (`execution_engine.py`)

**Purpose**: Handles execution of hybrid and baseline placement methods

**Key Methods:**
- `run_hybrid_methods()`: Runs all 3 hybrid methods (UCB, Epsilon-Greedy, Thompson Sampling)
- `run_baseline_methods()`: Runs all 3 baseline methods (K-Means, Random, Uniform)
- `_validate_placements()`: Validates placement data and ensures proper edge information
- `_evaluate_placements_with_simulation()`: Evaluates placements using SUMO simulation
- `_fallback_evaluation()`: Fallback evaluation when simulation fails

**Features:**
- Parallel method execution
- SUMO simulation integration
- Checkpoint/resume capability
- Error handling and recovery

### 3. Metrics Analyzer (`metrics_analyzer.py`)

**Purpose**: Handles all metrics calculation, comparison, and reporting

**Key Methods:**
- `calculate_comparison_metrics()`: Calculates comprehensive comparison metrics for all methods
- `generate_comparison_report()`: Generates detailed comparison reports
- `save_results()`: Saves all results to files
- Various `_calculate_*` methods for specific metrics calculations

**Features:**
- Statistical significance testing
- Confidence intervals
- Comprehensive metric suite
- Publication-ready reports

### 4. Visualization (`visualization.py`)

**Purpose**: Generates visualizations of results

**Features:**
- Station placement maps
- Performance comparison charts
- Statistical analysis plots
- Interactive visualizations

## Usage

### Main Evaluation Runner

**Comprehensive Multi-Grid Evaluation:**

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
- `--seed`: Random seed for reproducibility (default: 42)
- `--station-budget`: Total number of stations to allocate across grids (default: 150)
- `--adaptive-mode`: Enable adaptive episode calculation based on action space
- `--confidence-threshold`: Confidence level for statistical tests (default: 0.95)
- `--resume`: Resume from checkpoints if available
- `--verbose`: Enable verbose logging

**Additional Options:**
```bash
# Override episode count for testing
--override-episodes 1000

# Number of parallel workers
--workers 8

# Clear existing checkpoints
--clear-checkpoints

# Optimize checkpoint storage (reduces size by 90%)
--optimize-checkpoints

# Specific grid IDs to evaluate
--grid-ids grid_1 grid_2 grid_3
```

### Single-Grid Evaluation

**Evaluate a single grid:**

```bash
python evaluation/microplacement_validation/run_evaluation.py \
    --grid-id grid_1 \
    --station-budget 10 \
    --methods ucb thompson_sampling kmeans
```

### Example Usage Scenarios

#### Scenario 1: Full Evaluation with Adaptive Episodes

```bash
python evaluation/microplacement_validation/run_all_grids_evaluation.py \
    --seed 42 \
    --station-budget 150 \
    --adaptive-mode \
    --confidence-threshold 0.95 \
    --resume \
    --workers 4 \
    --verbose
```

**Use Case**: Comprehensive evaluation for paper results

#### Scenario 2: Quick Testing with Fixed Episodes

```bash
python evaluation/microplacement_validation/run_all_grids_evaluation.py \
    --seed 42 \
    --station-budget 50 \
    --override-episodes 100 \
    --clear-checkpoints \
    --workers 2
```

**Use Case**: Quick testing during development

#### Scenario 3: Resume from Checkpoint

```bash
python evaluation/microplacement_validation/run_all_grids_evaluation.py \
    --seed 42 \
    --station-budget 150 \
    --resume \
    --verbose
```

**Use Case**: Continue interrupted evaluation

## Configuration

### Adaptive Episode Calculation

When `--adaptive-mode` is enabled, the framework calculates the number of episodes based on:

\[ \text{episodes} = \max\left(100, \min\left(5000, \frac{\text{action_space_size} \times \log(\text{action_space_size})}{10}\right)\right) \]

This ensures:
- Sufficient exploration for large action spaces
- Efficient evaluation for small action spaces
- Bounded computation time

### Confidence Threshold

The `--confidence-threshold` parameter controls:
- Statistical significance testing (t-tests, Mann-Whitney U)
- Confidence intervals for metrics
- Hypothesis testing for method comparison

**Recommended Values:**
- `0.95`: Standard research confidence level
- `0.99`: Higher confidence for critical comparisons
- `0.90`: Lower confidence for exploratory analysis

### Station Budget Allocation

Stations are allocated across grids using proportional allocation based on ML demand predictions:

\[ \text{budget}_i = \frac{\text{demand}_i}{\sum_j \text{demand}_j} \times \text{total_budget} \]

## Output and Results

### Output Structure

```
results/
├── evaluation_results/
│   ├── grid_1/
│   │   ├── ucb_results.json
│   │   ├── thompson_sampling_results.json
│   │   ├── epsilon_greedy_results.json
│   │   ├── kmeans_results.json
│   │   ├── random_results.json
│   │   └── uniform_results.json
│   ├── grid_2/
│   │   └── ...
│   └── ...
├── comparison_report.txt
├── metrics_summary.json
├── statistical_analysis.json
└── checkpoints/
    └── grid_*/checkpoint_*.pkl
```

### Key Output Files

1. **comparison_report.txt**: Human-readable comparison of all methods
2. **metrics_summary.json**: Comprehensive metrics for all methods
3. **statistical_analysis.json**: Statistical significance tests
4. **checkpoints/**: Resumable evaluation checkpoints

### Metrics Included

- **Simulation Rewards**: Primary performance metric
- **Coverage Metrics**: Service area, population, network coverage
- **Utilization Metrics**: Station utilization, queue length, waiting time
- **Network Impact**: Traffic congestion, travel time changes
- **Fairness Metrics**: Gini coefficient, spatial equity
- **Convergence Metrics**: Algorithm convergence rates

## Performance Metrics

### Primary Metrics

**Simulation Reward**: Composite reward from SUMO simulation
- **Range**: 0.0 to 1.0
- **Higher is better**
- **Typical Values**: 
  - Hybrid methods: ~0.72
  - Baseline methods: ~0.36-0.41

**Coverage**: Percentage of network/population covered
- **Service Area Coverage**: % of network within service radius
- **Population Coverage**: % of population within service radius

**Utilization**: Station usage efficiency
- **Station Utilization Rate**: Average charging port usage
- **Queue Length**: Average number of waiting vehicles

### Statistical Analysis

The framework performs:
- **T-tests**: Compare means between methods
- **Mann-Whitney U**: Non-parametric comparison
- **Confidence Intervals**: 95% CI for all metrics
- **Effect Size**: Cohen's d for practical significance

## Checkpoint System

### Checkpoint Features

- **Resumable Evaluation**: Resume from last checkpoint
- **Optimized Storage**: 90% size reduction (35KB → 5KB per checkpoint)
- **Essential Data Only**: Rewards, convergence, key metrics
- **Backward Compatible**: Automatic conversion of old checkpoints

### Checkpoint Management

```bash
# Resume from checkpoint
--resume

# Clear all checkpoints
--clear-checkpoints

# Optimize existing checkpoints
--optimize-checkpoints
```

## Parallel Processing

### Worker Configuration

```bash
# Use all available CPUs (minus 1)
--workers auto

# Specify number of workers
--workers 8

# Single-threaded (for debugging)
--workers 1
```

### Performance Considerations

- **Memory**: ~2-4GB per worker
- **CPU**: Multi-core recommended
- **Storage**: SSD recommended for checkpoint I/O
- **Network**: Local execution recommended (no network overhead)

## Troubleshooting

### Common Issues

**1. SUMO Simulation Fails**
```
Error: SUMO simulation failed for grid_1
```
**Solution**: 
- Check SUMO installation: `sumo --version`
- Verify SUMO_HOME environment variable
- Check network file validity
- Review trajectory data format

**2. Memory Issues**
```
MemoryError: Unable to allocate array
```
**Solution**:
- Reduce number of workers: `--workers 2`
- Process grids sequentially
- Clear checkpoints: `--clear-checkpoints`

**3. Checkpoint Errors**
```
Error: Unable to load checkpoint
```
**Solution**:
- Clear corrupted checkpoints: `--clear-checkpoints`
- Use `--optimize-checkpoints` to convert old format
- Resume from scratch if necessary

**4. Slow Evaluation**
**Solution**:
- Increase workers: `--workers 8`
- Use SSD storage
- Reduce episode count for testing: `--override-episodes 100`
- Disable verbose logging

### Debug Mode

Enable verbose logging for detailed debugging:

```bash
python evaluation/microplacement_validation/run_all_grids_evaluation.py \
    --seed 42 \
    --station-budget 150 \
    --verbose \
    --workers 1  # Single-threaded for easier debugging
```

## Integration with Main Pipeline

The evaluation framework integrates with the main HERO pipeline:

1. **ML Demand Prediction** → Grid-level demand scores
2. **Proportional Allocation** → Station budgets per grid
3. **RL Optimization** → Micro-placement within grids
4. **SUMO Simulation** → Performance evaluation
5. **Metrics Analysis** → Final results

## Best Practices

### ✅ Recommended Practices

- Use `--resume` for long-running evaluations
- Set appropriate `--confidence-threshold` for your analysis
- Use `--adaptive-mode` for optimal episode counts
- Save checkpoints regularly
- Use parallel processing for faster evaluation
- Review `comparison_report.txt` for quick insights

### ❌ Avoid

- Don't clear checkpoints unless necessary
- Don't use too many workers (memory constraints)
- Don't skip statistical analysis (important for research)
- Don't ignore convergence metrics

## References

For detailed information about:
- **RL Methods**: See [modules/RLOptimization/README.md](../../modules/RLOptimization/README.md)
- **ML Pipeline**: See [modules/MLPrediction/README.md](../../modules/MLPrediction/README.md)
- **Main Pipeline**: See [README.md](../../README.md)

## License

This module is part of the HERO framework and is licensed under the same terms as the main project.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback about the evaluation framework, please open an issue in the repository.

---
