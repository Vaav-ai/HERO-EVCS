# Data Preparation Utilities

## Overview

This module provides utilities for preparing and processing raw datasets for use in the HERO framework. It includes functions for loading, combining, and preprocessing Vehicle Energy Dataset (VED) data and assigning trajectories to grid cells.

## Module Structure

```
evaluation/data_prep/
├── data_preparation.py      # Data preparation utilities
└── README.md               # This file
```

## Purpose

The data preparation module handles:

1. **VED Data Loading**: Load and combine multiple VED CSV files
2. **Grid Assignment**: Assign GPS trajectories to grid cells
3. **Data Aggregation**: Aggregate trajectories by grid and time
4. **Visualization**: Generate heatmaps and coverage plots

## Key Functions

### 1. Load and Combine VED Data

```python
from evaluation.data_prep.data_preparation import load_and_combine_ved

# Load all VED CSV files from directory
df = load_and_combine_ved(data_dir="data/ved_raw/")
```

**Features:**
- Combines multiple VED CSV files
- Adds week column from filename
- Handles coordinate conversion
- Validates data integrity

### 2. Assign Points to Grids

```python
from evaluation.data_prep.data_preparation import assign_points_to_grid
from modules.utils.gridding import CityGridding

# Generate grids
gridder = CityGridding(primary_grid_size_km=1.0)
grids = gridder.create_city_grid("Ann Arbor, Michigan, USA")

# Assign trajectories to grids
df_with_grids = assign_points_to_grid(df, grids)
```

**Features:**
- Spatial point-in-polygon assignment
- Grid ID assignment
- Coordinate validation
- Efficient spatial indexing

### 3. Activity Heatmap

```python
from evaluation.data_prep.data_preparation import plot_activity_heatmap

# Generate activity heatmap
plot_activity_heatmap(df_with_grids, output_file="activity_heatmap.html")
```

### 4. Weekly Grid Coverage

```python
from evaluation.data_prep.data_preparation import plot_weekly_grid_coverage

# Plot weekly coverage
plot_weekly_grid_coverage(df_with_grids, output_file="weekly_coverage.png")
```

## Usage

### Complete Data Preparation Pipeline

```python
from evaluation.data_prep.data_preparation import (
    load_and_combine_ved,
    assign_points_to_grid
)
from modules.utils.gridding import CityGridding

# 1. Load VED data
ved_data = load_and_combine_ved("data/ved_raw/")

# 2. Generate grids
gridder = CityGridding(primary_grid_size_km=1.0)
grids = gridder.create_city_grid("Ann Arbor, Michigan, USA")

# 3. Assign to grids
ved_with_grids = assign_points_to_grid(ved_data, grids)

# 4. Save processed data
ved_with_grids.to_parquet("data/validation/ved_processed_with_grids.parquet")
```

## Integration

### With Validation Framework

The processed data from this module is used by:
- `evaluation/demand_validation/run_validation.py`: For model validation
- `evaluation/microplacement_validation/`: For RL evaluation

### Data Flow

```
Raw VED CSVs → Load & Combine → Grid Assignment → 
Aggregation → Processed Parquet → Validation/Evaluation
```

## Output Format

### Processed VED Data Structure

```python
{
    'timestamp': datetime,
    'lat': float,
    'lon': float,
    'VehId': str,
    'week': datetime,
    'grid_id': str,          # Assigned grid ID
    'grid_center_lat': float,
    'grid_center_lon': float
}
```

## Dependencies

- `pandas`: Data manipulation
- `geopandas`: Geospatial operations
- `shapely`: Geometric operations
- `folium`: Interactive maps
- `matplotlib`: Static plots
- `seaborn`: Statistical plots
- `modules.utils.gridding`: Grid generation

## Troubleshooting

### Coordinate Issues

```python
# Ensure coordinates are numeric
df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
df['lon'] = pd.to_numeric(df['lon'], errors='coerce')

# Drop invalid coordinates
df = df.dropna(subset=['lat', 'lon'])
```

### Grid Assignment Failures

```python
# Verify grid format
assert 'grid_id' in grids[0]
assert 'geometry' in grids[0] or 'corners' in grids[0]

# Check coordinate system
# Ensure lat/lon are in WGS84 (EPSG:4326)
```

## License

This module is part of the HERO framework and is licensed under the same terms as the main project.

---

**Last Updated**: January 2025



