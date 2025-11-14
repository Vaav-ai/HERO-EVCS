# Reinforcement Learning Framework for EV Charging Station Placement

## Overview

This module implements a comprehensive reinforcement learning framework for optimizing electric vehicle (EV) charging station placement in urban environments. The framework combines multi-armed bandit algorithms with SUMO (Simulation of Urban MObility) to evaluate and optimize charging station locations based on real-world traffic patterns and EV charging demand.

The RL framework operates as **Stage 2** of the HERO pipeline, executing tactical micro-placement within grid cells allocated by the ML demand prediction module (Stage 1).

## Table of Contents

1. [Architecture](#architecture)
2. [Mathematical Framework](#mathematical-framework)
3. [Components](#components)
4. [Algorithms](#algorithms)
5. [Usage](#usage)
6. [Performance Metrics](#performance-metrics)
7. [Integration with Main Pipeline](#integration-with-main-pipeline)
8. [References](#references)

## Architecture

### Core Components

```
modules/RLOptimization/
├── __init__.py                    # Module initialization and exports
├── HybridBanditFramework.py      # Main hybrid bandit framework orchestrator
├── BanditOptimizationFramework.py # Alternative bandit optimization framework
├── SimulationManager.py          # SUMO simulation execution and coordination
├── ChargingStationManager.py     # Station operations, state, and queue management
├── RouteGenerator.py             # Vehicle route generation from GPS trajectories
├── SimulationAnalyzer.py         # SUMO simulation result analysis and metrics
├── SumoNetwork.py               # SUMO network interface and operations
├── ValidationFramework.py        # Validation utilities for placement evaluation
├── DomainRandomization.py        # Domain randomization for robust evaluation
├── metrics.py                    # Performance metrics calculation
├── HeuristicBaselines.py        # Baseline methods (K-Means, Random, Uniform)
├── models/                      # Bandit algorithm implementations
│   ├── __init__.py              # Models module initialization
│   ├── BaseBandit.py           # Base class for all bandit algorithms
│   ├── UCB.py                  # Upper Confidence Bound algorithm
│   ├── ThompsonSampling.py     # Thompson Sampling algorithm
│   └── EpsilonGreedy.py        # ε-Greedy algorithm
└── README.md                    # This file
```

### File Descriptions

- **`__init__.py`**: Module exports and initialization
- **`HybridBanditFramework.py`**: Main hybrid bandit framework combining heuristics with reinforcement learning. Orchestrates the complete RL optimization process.
- **`BanditOptimizationFramework.py`**: Alternative bandit optimization framework implementation
- **`SimulationManager.py`**: Manages SUMO simulation execution, coordination, and result aggregation
- **`ChargingStationManager.py`**: Handles charging station operations, state management, capacity tracking, and queue handling
- **`RouteGenerator.py`**: Generates realistic vehicle routes from GPS trajectories, converts to SUMO trips, and handles map matching
- **`SimulationAnalyzer.py`**: Analyzes SUMO simulation results, computes performance metrics (coverage, utilization, network impact, etc.)
- **`SumoNetwork.py`**: Interface for SUMO network operations, network loading, and manipulation
- **`ValidationFramework.py`**: Validation utilities for placement evaluation and performance assessment
- **`DomainRandomization.py`**: Domain randomization utilities for robust evaluation across different conditions
- **`metrics.py`**: Performance metrics calculation including coverage, utilization, network impact, fairness, and economic metrics
- **`HeuristicBaselines.py`**: Baseline placement methods including K-Means clustering, Random placement, and Uniform spacing for comparison
- **`models/`**: Bandit algorithm implementations
  - **`BaseBandit.py`**: Abstract base class defining the interface for all bandit algorithms
  - **`UCB.py`**: Upper Confidence Bound algorithm implementation
  - **`ThompsonSampling.py`**: Thompson Sampling (Bayesian) algorithm implementation
  - **`EpsilonGreedy.py`**: ε-Greedy exploration-exploitation algorithm implementation

### Data Flow

1. **Input**: Grid-level station budgets from ML demand prediction
2. **Candidate Generation**: POI- and network-based heuristics generate candidate locations
3. **Bandit Selection**: Multi-armed bandit algorithm selects placement action
4. **Simulation**: SUMO traffic simulation evaluates placement
5. **Reward Calculation**: Composite reward computed from multiple objectives
6. **Policy Update**: Bandit algorithm updates based on observed rewards
7. **Output**: Optimized station locations with performance metrics

## Mathematical Framework

### 1. State Space

The state space S represents the urban environment within a grid cell:

\[ S = \{s_1, s_2, ..., s_n\} \]

where each state \(s_i\) contains:
- Traffic density patterns
- EV charging demand (from ML prediction)
- Network topology (road network structure)
- Existing charging infrastructure
- POI (Points of Interest) distribution

### 2. Action Space

The action space A represents possible charging station placements:

\[ A = \{a_1, a_2, ..., a_k\} \]

where each action \(a_i\) specifies:
- Location coordinates \((x, y)\) on road network
- Station capacity (number of charging ports)
- Charging rate (kW)

Actions are generated using:
- **POI-based heuristics**: High-traffic areas, commercial zones
- **Network-based heuristics**: Road intersections, highway exits
- **Demand-based heuristics**: Areas with high predicted demand

### 3. Reward Function

The reward function \(R(s, a)\) combines multiple objectives:

\[ R(s, a) = w_1 R_{coverage}(s, a) + w_2 R_{utilization}(s, a) + w_3 R_{network}(s, a) + w_4 R_{fairness}(s, a) + w_5 R_{battery}(s, a) \]

where:

**Coverage Reward:**
\[ R_{coverage}(s, a) = \frac{\text{Number of EVs within service radius}}{\text{Total EVs in network}} \]

**Utilization Reward:**
\[ R_{utilization}(s, a) = \frac{\text{Total charging time}}{\text{Total simulation time}} \]

**Network Impact Reward:**
\[ R_{network}(s, a) = 1 - \frac{\text{Traffic congestion with station}}{\text{Base traffic congestion}} \]

**Fairness Reward:**
\[ R_{fairness}(s, a) = 1 - \text{Gini coefficient of waiting times} \]

**Battery Health Reward:**
\[ R_{battery}(s, a) = \frac{\text{EVs avoiding deep discharge}}{\text{Total EVs}} \]

### 4. Optimization Objective

Maximize the expected cumulative reward:

\[ \max_{\pi} \mathbb{E}\left[\sum_{t=1}^T R(s_t, \pi(s_t))\right] \]

where \(\pi\) is the policy (bandit algorithm) and \(T\) is the number of episodes.

## Components

### 1. Hybrid Bandit Framework

```python
class HybridBanditFramework:
    """
    Main framework for EV charging station placement optimization.
    
    Combines heuristic initialization with bandit-based exploration.
    
    Parameters:
        osm_file_path: Path to OpenStreetMap data
        num_stations: Number of charging stations to place
        sim_config: Simulation configuration
        ev_config: EV fleet configuration
        bandit_algorithm: Bandit algorithm instance (UCB, Thompson, etc.)
    """
```

**Key Features:**
- Heuristic candidate generation
- Bandit-based action selection
- SUMO simulation integration
- Reward calculation and policy updates
- Checkpoint/resume capability

### 2. Simulation Manager

```python
class SimulationManager:
    """
    Manages SUMO simulation execution and coordination.
    
    Features:
        - Parallel simulation execution
        - Result aggregation
        - Performance monitoring
        - Error handling and recovery
    """
```

**Responsibilities:**
- SUMO process management
- Route file generation
- Simulation execution
- Result parsing and aggregation

### 3. Charging Station Manager

```python
class ChargingStationManager:
    """
    Manages charging station operations and state.
    
    Features:
        - Station placement validation
        - Capacity management
        - Queue handling
        - Charging event tracking
    """
```

### 4. Route Generator

```python
class RouteGenerator:
    """
    Generates realistic vehicle routes for simulation.
    
    Features:
        - Origin-destination matrix generation
        - Traffic demand modeling
        - Route optimization
        - Real trajectory integration (VED data)
    """
```

**Route Generation:**
- Converts GPS trajectories to SUMO trips
- Uses `duarouter` for path computation
- Map matching with waypoints
- Handles coordinate-based and edge-based routing

## Algorithms

The framework implements three multi-armed bandit algorithms:

### 1. Upper Confidence Bound (UCB)

**Algorithm:**
\[ \text{UCB}_a(t) = \hat{\mu}_a(t) + c\sqrt{\frac{\log(t)}{N_a(t)}} \]

where:
- \(\hat{\mu}_a(t)\): Empirical mean reward for action \(a\)
- \(N_a(t)\): Number of times action \(a\) was selected
- \(c\): Exploration parameter (typically \(\sqrt{2}\))

**Implementation:** `models/UCB.py`

### 2. Thompson Sampling

**Bayesian Approach:**
\[ P(\theta_a|D) \propto P(D|\theta_a)P(\theta_a) \]

**Selection Rule:**
- Sample \(\theta_a \sim P(\theta_a|D)\) for each action
- Select action with highest sampled value

**Implementation:** `models/ThompsonSampling.py`

### 3. ε-Greedy

**Selection Rule:**
\[ a_t = \begin{cases}
\text{argmax}_a \hat{\mu}_a(t) & \text{with probability } 1-\epsilon \\
\text{random action} & \text{with probability } \epsilon
\end{cases} \]

**Implementation:** `models/EpsilonGreedy.py`

### Algorithm Comparison

| Algorithm | Exploration Strategy | Best For |
|-----------|---------------------|----------|
| **UCB** | Optimistic exploration | Balanced exploration-exploitation |
| **Thompson Sampling** | Bayesian sampling | Fast convergence, uncertainty handling |
| **ε-Greedy** | Random exploration | Simple, interpretable |

## Usage

### Basic Usage

```python
from modules.RLOptimization import HybridBanditFramework
from modules.RLOptimization.models import ThompsonSampling
from modules.config import SimulationConfig, EVConfig

# Initialize framework
framework = HybridBanditFramework(
    osm_file_path="city.osm",
    num_stations=10,
    sim_config=SimulationConfig(),
    ev_config=EVConfig(),
    bandit_algorithm=ThompsonSampling()
)

# Run optimization
best_placement, rewards = framework.run_optimization(
    episodes=1000
)
```

### Advanced Configuration

```python
# Custom simulation settings
sim_config = SimulationConfig(
    time_step=1.0,
    end_time=3600,
    warmup_period=300
)

# EV fleet configuration
ev_config = EVConfig(
    battery_capacity=50,      # kWh
    charging_rate=7.4,         # kW
    consumption_rate=0.2      # kWh/km
)

# Custom bandit parameters
bandit = UCB(exploration_param=1.5)
```

### Integration with Main Pipeline

The RL framework is typically called from `main.py`:

```bash
python main.py \
    --city "Mumbai, India" \
    --total-stations 50 \
    --optimization-method ucb \
    --max-episodes 20
```

**Optimization Methods:**
- `ucb`: Upper Confidence Bound
- `thompson_sampling`: Thompson Sampling
- `epsilon_greedy`: ε-Greedy

### Multi-Grid Evaluation

For comprehensive evaluation across multiple grids:

```bash
python evaluation/microplacement_validation/run_all_grids_evaluation.py \
    --seed 42 \
    --station-budget 150 \
    --adaptive-mode \
    --confidence-threshold 0.95 \
    --resume \
    --verbose
```

## Performance Metrics

### 1. Coverage Metrics
- **Service Area Coverage**: Percentage of network covered
- **Population Coverage**: Percentage of population within service radius
- **Network Coverage**: Percentage of road network accessible

### 2. Utilization Metrics
- **Station Utilization Rate**: Average charging port usage
- **Queue Length**: Average number of waiting vehicles
- **Waiting Time**: Average time before charging starts

### 3. Network Impact
- **Traffic Congestion**: Change in average travel time
- **Network Efficiency**: Overall network performance
- **Route Changes**: Impact on vehicle routing

### 4. Economic Metrics
- **Installation Cost**: Based on location and infrastructure
- **Operational Cost**: Energy consumption and maintenance
- **Revenue Potential**: Based on utilization

### 5. Fairness Metrics
- **Gini Coefficient**: Distribution of waiting times
- **Spatial Equity**: Distribution across geographic areas
- **Accessibility**: Coverage of underserved areas

## Integration with Main Pipeline

### Workflow

1. **ML Demand Prediction** (Stage 1) → Grid-level demand scores
2. **Proportional Allocation** → Station budgets per grid
3. **RL Optimization** (Stage 2) → Micro-placement within grids
4. **SUMO Simulation** → Performance evaluation
5. **Metrics Analysis** → Final results

### Data Flow

```
ML Prediction → Demand Scores → Grid Budgets → RL Framework → Station Locations → SUMO → Metrics
```

## Route Generation from Real Data

The pipeline converts VED GPS trajectories into SUMO `<trip>`s and runs `duarouter` to compute routes.

**Key Features:**
- Trips sorted by `depart` to avoid SUMO warnings
- Coordinate-based trips use `fromLonLat`, `toLonLat`, and `viaLonLat` waypoints
- Map matching with configurable distance threshold
- Robust routing with fallback mechanisms

**Routing Parameters:**
```bash
duarouter \
    --unsorted-input \
    --mapmatch.distance 1000 \
    --mapmatch.junctions \
    --repair \
    --remove-loops \
    --routing-threads 4 \
    --routing-algorithm CH
```

**Troubleshooting:**
- Check generated `temp_trips_*.xml` files
- Verify lon,lat order and reasonable depart times
- Adjust `mapmatch.distance` if mismatches occur
- Increase neighbor search radius (default: 400m) for edge-based fallback

## License

This module is part of the HERO framework and is licensed under the same terms as the main project.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback about the RL optimization module, please open an issue in the repository.

---