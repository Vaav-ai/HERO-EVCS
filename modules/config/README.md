# Configuration Management for EV Charging Station Placement

## Overview
This module provides a robust configuration management system for the EV Charging Station Placement project, handling simulation parameters and EV-specific settings. It implements a flexible, hierarchical configuration system that supports both research experimentation and production deployment.

## Table of Contents
1. [Research Context](#research-context)
2. [Architecture](#architecture)
3. [Configuration Components](#configuration-components)
4. [Implementation](#implementation)
5. [Usage](#usage)
6. [Validation](#validation)
7. [Contributing](#contributing)

## Research Context

### Problem Statement
The configuration of EV charging station placement simulations requires careful management of multiple interdependent parameters across different domains:
- Simulation parameters for traffic modeling
- EV-specific parameters for charging behavior
- System-wide parameters for optimization

### Research Contributions
1. **Hierarchical Configuration System**
   - Modular parameter management
   - Cross-domain parameter validation
   - Dynamic parameter adjustment

2. **Validation Framework**
   - Parameter range validation
   - Dependency checking
   - Consistency verification

3. **Research Experimentation Support**
   - Parameter sweep capabilities
   - Configuration versioning
   - Experiment reproducibility

## Architecture

### Module Structure

```
modules/config/
├── __init__.py              # Module initialization and exports
├── SimulationConfig.py      # SUMO simulation configuration
├── EVConfig.py             # EV fleet configuration
└── README.md               # This file
```

### File Descriptions

- **`__init__.py`**: Module exports and initialization
- **`SimulationConfig.py`**: SimulationConfig class for SUMO simulation parameters
- **`EVConfig.py`**: EVConfig class for EV fleet parameters (battery, charging, consumption)

### Core Components

1. **Simulation Configuration**
   ```python
   class SimulationConfig:
       """
       Manages SUMO simulation parameters and settings.
       
       Attributes:
           time_step: Simulation time step (seconds)
           end_time: Simulation end time (seconds)
           warmup_period: Warmup period before data collection
           network_file: Path to SUMO network file
       """
   ```

2. **EV Configuration**
   ```python
   class EVConfig:
       """
       Manages EV-specific parameters and charging behavior settings.
       
       Attributes:
           battery_capacity: Battery capacity (kWh)
           charging_rate: Charging rate (kW)
           consumption_rate: Energy consumption rate (kWh/km)
       """
   ```

## Configuration Components

### 1. Simulation Parameters
```python
class SimulationConfig:
    """
    Simulation configuration parameters:
    - Time step duration
    - Simulation duration
    - Network parameters
    - Traffic flow settings
    """
```

### 2. EV Parameters
```python
class EVConfig:
    """
    EV configuration parameters:
    - Battery capacity
    - Charging rates
    - Range parameters
    - Behavior models
    """
```

## Implementation

### 1. Parameter Management
```python
def validate_parameters(self):
    """
    Validates configuration parameters:
    1. Range checking
    2. Dependency validation
    3. Consistency verification
    """
```

### 2. Configuration Loading
```python
def load_config(self, config_path):
    """
    Loads configuration from file:
    1. File parsing
    2. Parameter validation
    3. Default value handling
    """
```

### 3. Parameter Export
```python
def export_config(self, format='json'):
    """
    Exports configuration in specified format:
    1. Parameter serialization
    2. Format conversion
    3. File writing
    """
```

## Usage

### Basic Usage
```python
from modules.config import SimulationConfig, EVConfig

# Initialize configurations
sim_config = SimulationConfig()
ev_config = EVConfig()

# Load from file
sim_config.load_config('simulation_config.json')
```

### Advanced Features
```python
# Custom configuration with validation
ev_config = EVConfig(
    battery_capacity=75.0,
    charging_rate=50.0,
    consumption_rate=0.2
)

# Parameter sweep configuration
sweep_config = {
    'battery_capacity': [50.0, 75.0, 100.0],
    'charging_rate': [25.0, 50.0, 75.0]
}
```

## Validation

### 1. Parameter Validation
- Range checking
- Type verification
- Dependency validation

### 2. Configuration Consistency
- Cross-parameter validation
- Default value handling
- Error reporting

### 3. Research Validation
- Experiment reproducibility
- Parameter sweep validation
- Configuration versioning

## Research Applications

### 1. Experiment Design
- Parameter sweep studies
- Sensitivity analysis
- Robustness testing

### 2. Model Calibration
- Parameter optimization
- Model validation
- Performance tuning

### 3. System Integration
- Component configuration
- Interface settings
- System-wide parameters

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This module is part of the HERO project and is licensed under the MIT License.

## Contact
For questions or feedback about the configuration module, please open an issue in the repository. 