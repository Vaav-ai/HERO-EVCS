# Utilities Module

## Overview

The `utils` module provides essential utility functions and classes used throughout the HERO framework. These utilities ensure reproducibility, efficient caching, consistent logging, and robust geospatial operations across all components of the system.

## Table of Contents

1. [Module Structure](#module-structure)
2. [Components](#components)
3. [Usage](#usage)
4. [Integration](#integration)
5. [Best Practices](#best-practices)

## Module Structure

```
modules/utils/
├── __init__.py              # Module initialization
├── cache_utils.py          # Cache management utilities
├── gpu_utils.py            # GPU detection and configuration
├── gridding.py             # City gridding and spatial operations
├── log_configs.py         # Logging configuration
├── manage_cache.py         # Cache management CLI tool
├── seed_utils.py           # Reproducibility and seed management
└── README.md              # This file
```

## Components

### 1. Seed Utilities (`seed_utils.py`)

**Purpose**: Ensures complete reproducibility across all modules by setting consistent random seeds.

**Key Functions:**
- `set_global_seeds(seed)`: Sets seeds for Python, NumPy, TensorFlow, PyTorch, SUMO, and more
- `validate_seed_consistency(seed, logger)`: Validates that seeds are set correctly

**Usage:**
```python
from modules.utils.seed_utils import set_global_seeds, validate_seed_consistency

# Set global seeds at the start of your script
set_global_seeds(42)

# Validate consistency
validate_seed_consistency(42, logger)
```

**Supported Libraries:**
- Python `random`
- NumPy
- TensorFlow
- PyTorch
- SUMO (via environment variables)
- scikit-learn
- scipy.stats

### 2. Cache Utilities (`cache_utils.py`)

**Purpose**: Manages caching of expensive operations (OSM feature extraction, grid generation, etc.)

**Key Classes:**
- `CacheManager`: Main cache management class

**Key Methods:**
- `get_cache_path(cache_type, cache_key)`: Get path for cache file
- `exists(cache_type, cache_key)`: Check if cache exists
- `load(cache_type, cache_key)`: Load from cache
- `save(cache_type, cache_key, data)`: Save to cache
- `clear_cache(cache_type)`: Clear specific or all cache

**Usage:**
```python
from modules.utils.cache_utils import CacheManager, create_cache_key

cache = CacheManager(cache_dir="cache")

# Check if cached
if cache.exists("osm_features", "singapore"):
    features = cache.load("osm_features", "singapore")
else:
    features = extract_osm_features("singapore")
    cache.save("osm_features", "singapore", features)
```

**Cache Types:**
- `osm_features`: OpenStreetMap feature extractions
- `grid_cells`: Generated grid cells
- `demand_predictions`: ML model predictions
- `simulation_results`: SUMO simulation outputs

### 3. City Gridding (`gridding.py`)

**Purpose**: Robust, adaptive algorithm for dividing urban areas into optimized grid cells.

**Key Class:**
- `CityGridding`: Main gridding class

**Key Features:**
- Adaptive grid creation (primary and flexible secondary grids)
- Geographic feature awareness (water bodies, urban density)
- Multi-scale support
- Comprehensive caching
- Quality validation

**Usage:**
```python
from modules.utils.gridding import CityGridding

# Create gridder
gridder = CityGridding(
    primary_grid_size_km=1.0,
    water_threshold=0.7,
    urban_threshold=0.3
)

# Generate grid for a city
grid_cells = gridder.create_city_grid("Mumbai, India")

# Visualize
gridder.visualize_grid(grid_cells, "mumbai_grid.html")
```

**Standard Grid Format:**
All grid cells follow a consistent structure:
```python
{
    'grid_id': 'grid_1',           # Unique identifier
    'cell_id': 'grid_1',            # Legacy compatibility
    'min_lat': 19.0, 'max_lat': 19.1,
    'min_lon': 72.8, 'max_lon': 72.9,
    'center_lat': 19.05, 'center_lon': 72.85,
    'area_km2': 1.0,
    'corners': [(lat, lon), ...]    # Corner coordinates
}
```

### 4. Logging Configuration (`log_configs.py`)

**Purpose**: Centralized logging configuration for consistent log formatting across the framework.

**Key Functions:**
- `setup_logging(log_file_name)`: Configure logging with file and console handlers

**Usage:**
```python
from modules.utils.log_configs import setup_logging

# Setup logging at the start of your script
setup_logging(log_file_name="pipeline.log")

# Use standard logging
import logging
logger = logging.getLogger(__name__)
logger.info("Pipeline started")
```

**Features:**
- Color-coded console output
- File logging with rotation
- Consistent format across modules
- Log level configuration

### 5. GPU Utilities (`gpu_utils.py`)

**Purpose**: GPU detection and configuration for accelerated ML operations.

**Key Functions:**
- `detect_gpu_availability()`: Detect available GPU resources
- `configure_gpu_for_ml()`: Configure ML libraries for GPU usage

**Usage:**
```python
from modules.utils.gpu_utils import detect_gpu_availability, configure_gpu_for_ml

# Detect GPU
gpu_info = detect_gpu_availability()
if gpu_info['cuda_available']:
    print(f"GPU detected: {gpu_info['gpu_names']}")
    configure_gpu_for_ml()
```

**Supported Libraries:**
- XGBoost (GPU)
- LightGBM (GPU)
- CatBoost (GPU)
- TensorFlow (GPU)
- PyTorch (CUDA)

### 6. Cache Management CLI (`manage_cache.py`)

**Purpose**: Command-line tool for managing cache files.

**Usage:**
```bash
# Show cache status
python -m modules.utils.manage_cache --status

# Clear all cache
python -m modules.utils.manage_cache --clear-all

# Clear specific cache type
python -m modules.utils.manage_cache --clear-type osm_features

# List all cached items
python -m modules.utils.manage_cache --list
```

## Usage

### Basic Setup Pattern

```python
import logging
from modules.utils.seed_utils import set_global_seeds, validate_seed_consistency
from modules.utils.log_configs import setup_logging
from modules.utils.cache_utils import CacheManager

# 1. Setup logging first
setup_logging(log_file_name="my_script.log")
logger = logging.getLogger(__name__)

# 2. Set seeds for reproducibility
set_global_seeds(42)
validate_seed_consistency(42, logger)

# 3. Initialize cache manager
cache = CacheManager(cache_dir="cache")

# 4. Your code here...
```

### Gridding Example

```python
from modules.utils.gridding import CityGridding

# Initialize gridder
gridder = CityGridding(
    primary_grid_size_km=1.0,
    water_threshold=0.7,
    urban_threshold=0.3,
    cache_dir="cache"
)

# Generate grids (with automatic caching)
grids = gridder.create_city_grid("Ann Arbor, Michigan, USA")

# Access grid properties
for grid in grids:
    print(f"Grid {grid['grid_id']}: {grid['area_km2']:.2f} km²")
```

### Caching Example

```python
from modules.utils.cache_utils import CacheManager, create_cache_key

cache = CacheManager()

# Create cache key
city_name = "Mumbai"
cache_key = create_cache_key(city_name, grid_size=1.0)

# Check and use cache
if cache.exists("grid_cells", cache_key):
    grids = cache.load("grid_cells", cache_key)
    logger.info("Loaded grids from cache")
else:
    grids = generate_grids(city_name)
    cache.save("grid_cells", cache_key, grids)
    logger.info("Saved grids to cache")
```

## Integration

### With ML Pipeline

```python
from modules.utils.seed_utils import set_global_seeds
from modules.utils.gpu_utils import detect_gpu_availability

# Set seeds before ML training
set_global_seeds(42)

# Check GPU for acceleration
gpu_info = detect_gpu_availability()
if gpu_info['cuda_available']:
    # Use GPU-accelerated models
    pass
```

### With RL Pipeline

```python
from modules.utils.seed_utils import set_global_seeds
from modules.utils.log_configs import setup_logging

# Setup logging
setup_logging("rl_training.log")

# Set seeds for SUMO reproducibility
set_global_seeds(42)  # Also sets SUMO_RANDOM_SEED
```

### With Evaluation Framework

```python
from modules.utils.cache_utils import CacheManager
from modules.utils.gridding import CityGridding

# Use cached grids for evaluation
gridder = CityGridding(cache_dir="cache")
grids = gridder.create_city_grid("Ann Arbor")  # Uses cache if available
```

## Best Practices

### 1. Always Set Seeds

```python
# At the start of every script
set_global_seeds(42)
```

### 2. Use Caching for Expensive Operations

```python
# Cache OSM feature extraction
# Cache grid generation
# Cache demand predictions
```

### 3. Consistent Logging

```python
# Use setup_logging() at the start
# Use consistent logger names
# Log important operations
```

### 4. Grid Format Consistency

```python
# Always use CityGridding for grid generation
# Use standard grid format fields
# Validate grid structure before use
```

### 5. GPU Detection

```python
# Check GPU availability before using GPU features
# Fall back to CPU if GPU unavailable
# Log GPU status for debugging
```

## Dependencies

- `osmnx`: OpenStreetMap data access
- `geopandas`: Geospatial data operations
- `shapely`: Geometric operations
- `numpy`: Numerical operations
- `pandas`: Data manipulation
- `folium`: Map visualization
- `tqdm`: Progress bars
- `retry`: Retry mechanisms

## Troubleshooting

### Cache Issues

```bash
# Clear corrupted cache
python -m modules.utils.manage_cache --clear-all
```

### Seed Inconsistency

```python
# Validate seeds are set correctly
validate_seed_consistency(42, logger)
```

### Grid Generation Failures

```python
# Enable debug mode
gridder = CityGridding(debug_mode=True)

# Check logs for specific errors
# Verify city name format
```

## License

This module is part of the HERO framework and is licensed under the same terms as the main project.

## Contributing

Contributions are welcome! Please ensure:
- All functions have proper docstrings
- Seeds are set for reproducibility
- Caching is used for expensive operations
- Logging is consistent

---
