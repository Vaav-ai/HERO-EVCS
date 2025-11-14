#!/usr/bin/env python3
"""
Data Loading Module for EV Charging Station Placement Evaluation

This module handles all data loading, preparation, and validation for the evaluation framework.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import traceback
import random
import pickle
import geopandas as gpd
from shapely.geometry import Point, Polygon

# Add project root to path
sys.path.append(os.path.abspath('.'))

from modules.utils.gridding import CityGridding
from modules.MLPrediction.demand_predictor import DemandPredictor
from modules.MLPrediction.portable_osm_extractor import PortableOSMExtractor
from modules.MLPrediction.feature_engineering import UrbanFeatureEngineer


class DataLoader:
    """
    Data loading and preparation class for EV charging station placement evaluation.
    
    This class handles loading VED data, creating city grids, preparing trajectory data,
    and validating all data components for the evaluation framework.
    """
    
    def __init__(self, random_seed=42, logger=None, grid_id=None, total_station_budget=100, 
                 city_name=None, custom_data_path=None, use_synthetic=False):
        """
        Initialize the DataLoader.
        
        Args:
            random_seed: Random seed for reproducibility
            logger: Logger instance for logging
            grid_id: Specific grid ID to use for testing (if None, will auto-select)
            total_station_budget: Total number of charging stations to allocate across all grids
            city_name: Name of the city (for synthetic data generation or UrbanEV fallback)
            custom_data_path: Path to custom CSV/Parquet file (optional)
            use_synthetic: Force use of synthetic data even if real data is available
        """
        self.random_seed = random_seed
        self.logger = logger or logging.getLogger(__name__)
        self.test_data = None
        self.grid_id = grid_id
        self.total_station_budget = total_station_budget
        self.station_allocations = None
        self.city_name = city_name
        self.custom_data_path = custom_data_path
        self.use_synthetic = use_synthetic
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        random.seed(random_seed)
    
    def assign_ved_to_grid(ved_df: pd.DataFrame, grid_cells: List[Dict]) -> gpd.GeoDataFrame:
        """Spatially join VED points to the provided grid cells."""
        logger.info(f"Assigning {len(ved_df)} VED points to {len(grid_cells)} grid cells...")
        
        # Drop conflicting columns if they exist
        cols_to_drop = ['index_left', 'index_right']
        ved_df_clean = ved_df.drop(columns=[col for col in cols_to_drop if col in ved_df.columns])
        
        # Create GeoDataFrame for VED data
        geometry = [Point(xy) for xy in zip(ved_df_clean['lon'], ved_df_clean['lat'])]
        ved_gdf = gpd.GeoDataFrame(ved_df_clean, geometry=geometry, crs="EPSG:4326")
        
        # Create GeoDataFrame for grid cells
        grid_polygons = []
        grid_data = []
        for cell in grid_cells:
            poly = Polygon([
                (cell['min_lon'], cell['min_lat']),
                (cell['min_lon'], cell['max_lat']),
                (cell['max_lon'], cell['max_lat']),
                (cell['max_lon'], cell['min_lat'])
            ])
            grid_polygons.append(poly)
            
            # Handle both grid_id and cell_id formats
            grid_id = cell.get('grid_id', cell.get('cell_id', f"grid_{len(grid_data)}"))
            grid_data.append({'grid_id': grid_id})
            
        grid_gdf = gpd.GeoDataFrame(grid_data, geometry=grid_polygons, crs="EPSG:4326")
        
        # Perform spatial join
        gridded_ved = gpd.sjoin(ved_gdf, grid_gdf, how="inner", predicate="within")
        
        logger.info(f"Successfully assigned {len(gridded_ved)} points to {gridded_ved['grid_id'].nunique()} grids.")
        return gridded_ved

    def load_test_data(self):
        """Load and prepare test data for a single grid with proper validation."""
        self.logger.info("📊 Loading test data...")
        
        try:
            # Determine data source and city
            data_source, city_name, coordinates = self._determine_data_source()
            self.logger.info(f"🎯 Data source: {data_source}")
            self.logger.info(f"🏙️ City: {city_name}")
            
            # Load data based on source
            if data_source == "custom":
                raw_ved_df = self._load_custom_data()
            else:  # synthetic trajectories
                raw_ved_df = self._generate_synthetic_data(city_name, coordinates)
            
            self.logger.info(f"✅ Loaded {len(raw_ved_df)} data points")
            
            # Create grid using CityGridding
            gridder = CityGridding(primary_grid_size_km=1.0, fetch_osm_features=False)
            city_grids_data = gridder.create_city_grid(city_name, coordinates=coordinates)
            self.logger.info(f"✅ Created {len(city_grids_data)} grid cells")
            
            # Assign VED data to grids using the working methodology from old code
            try:
                df = self.assign_ved_to_grid(raw_ved_df, city_grids_data)
                self.logger.info(f"✅ Assigned {len(df)} data points to grids")
            except ImportError as e:
                self.logger.warning(f"⚠️ Could not import evaluation module: {e}")
                self.logger.info("Using fallback grid assignment method...")
                # Fallback: simple grid assignment based on coordinates
                df = raw_ved_df.copy()
                if 'lat' in df.columns and 'lon' in df.columns:
                    # Assign to first grid cell as fallback
                    df['grid_id'] = city_grids_data[0]['grid_id']
                    self.logger.info(f"✅ Assigned {len(df)} data points to fallback grid")
                else:
                    self.logger.error("❌ No coordinate columns found for grid assignment")
                    return None
            
            # Select grid based on user preference or auto-select
            if self.grid_id is not None:
                # User specified a grid ID
                if self.grid_id in df['grid_id'].values:
                    test_grid_id = self.grid_id
                    data_count = len(df[df['grid_id'] == test_grid_id])
                    self.logger.info(f"✅ Using user-specified grid: {test_grid_id} (has {data_count} data points)")
                else:
                    self.logger.warning(f"⚠️ User-specified grid {self.grid_id} not found in data")
                    # Fall back to auto-selection
                    grids_with_data = df['grid_id'].value_counts()
                    if len(grids_with_data) > 0:
                        test_grid_id = grids_with_data.index[0]
                        data_count = grids_with_data.iloc[0]
                        self.logger.info(f"✅ Auto-selected grid: {test_grid_id} (has {data_count} data points)")
                    else:
                        test_grid_id = city_grids_data[0]['grid_id']
                        self.logger.info(f"✅ Using first available grid: {test_grid_id} (no real data)")
            else:
                # Auto-select grid with most data
                grids_with_data = df['grid_id'].value_counts()
                if len(grids_with_data) > 0:
                    test_grid_id = grids_with_data.index[0]
                    data_count = grids_with_data.iloc[0]
                    self.logger.info(f"✅ Auto-selected grid: {test_grid_id} (has {data_count} data points)")
                else:
                    test_grid_id = city_grids_data[0]['grid_id']
                    self.logger.info(f"✅ Using first available grid: {test_grid_id} (no real data)")
            
            # Get trajectories for test grid - filter by grid_id
            grid_data = df[df['grid_id'] == test_grid_id].copy()
            if len(grid_data) > 0:
                simulation_start_time = grid_data['timestamp'].min() if 'timestamp' in grid_data.columns else 0
                # Convert to trajectory format expected by the system
                trajectories = {}
                vehicle_id_col = 'VehId' if 'VehId' in grid_data.columns else 'vehicle_id' if 'vehicle_id' in grid_data.columns else 'id'
                
                if vehicle_id_col in grid_data.columns:
                    for vehicle_id, group in grid_data.groupby(vehicle_id_col):
                        if len(group) >= 2:  # Need at least 2 points for a trajectory
                            traj_data = group[['lat', 'lon', 'timestamp']].copy()
                            traj_data = traj_data.dropna()
                            if len(traj_data) >= 2:
                                trajectories[str(vehicle_id)] = traj_data.reset_index(drop=True)
                else:
                    # Fallback: create synthetic trajectories if no vehicle ID column
                    self.logger.warning("⚠️ No vehicle ID column found, creating synthetic trajectories")
                    if 'lat' in grid_data.columns and 'lon' in grid_data.columns:
                        # Create one trajectory with all points
                        traj_data = grid_data[['lat', 'lon', 'timestamp']].copy()
                        traj_data = traj_data.dropna()
                        if len(traj_data) >= 2:
                            trajectories['synthetic_vehicle'] = traj_data.reset_index(drop=True)
            else:
                trajectories = {}
                self.logger.warning("⚠️ No data found for selected grid, creating empty trajectories")
            
            self.logger.info(f"✅ Found {len(trajectories)} real trajectories")
            
            # CRITICAL FIX: Create trajectory DataFrame with proper structure
            trajectory_df = pd.DataFrame()
            if trajectories:
                for vehicle_id, traj_data in trajectories.items():
                    # Ensure traj_data is a DataFrame with proper columns
                    if isinstance(traj_data, pd.DataFrame):
                        traj_copy = traj_data.copy()
                    else:
                        # Convert dict to DataFrame if needed
                        traj_copy = pd.DataFrame(traj_data)
                    
                    # Ensure VehId column exists
                    traj_copy['VehId'] = str(vehicle_id)
                    
                    # Ensure required columns exist with proper data types
                    if 'lat' not in traj_copy.columns:
                        traj_copy['lat'] = 0.0
                    if 'lon' not in traj_copy.columns:
                        traj_copy['lon'] = 0.0
                    if 'timestamp' not in traj_copy.columns:
                        traj_copy['timestamp'] = 0.0
                    
                    # Convert coordinates to numeric, handling any string values
                    traj_copy['lat'] = pd.to_numeric(traj_copy['lat'], errors='coerce')
                    traj_copy['lon'] = pd.to_numeric(traj_copy['lon'], errors='coerce')
                    traj_copy['timestamp'] = pd.to_numeric(traj_copy['timestamp'], errors='coerce')
                    
                    # Drop any rows with invalid coordinates
                    traj_copy = traj_copy.dropna(subset=['lat', 'lon', 'timestamp'])
                    
                    if len(traj_copy) > 0:
                        trajectory_df = pd.concat([trajectory_df, traj_copy], ignore_index=True)
            else:
                # Create empty DataFrame with proper structure
                trajectory_df = pd.DataFrame(columns=['VehId', 'lat', 'lon', 'timestamp'])
                self.logger.warning("⚠️ No trajectories found, creating empty DataFrame")
                
                # If we have grid data but no trajectories, create a minimal synthetic trajectory
                if len(grid_data) > 0 and 'lat' in grid_data.columns and 'lon' in grid_data.columns:
                    self.logger.info("Creating minimal synthetic trajectory for testing...")
                    # Take a few points from the grid data
                    sample_data = grid_data[['lat', 'lon']].dropna().head(10)
                    if len(sample_data) > 0:
                        synthetic_traj = pd.DataFrame({
                            'VehId': 'synthetic_test_vehicle',
                            'lat': sample_data['lat'].values,
                            'lon': sample_data['lon'].values,
                            'timestamp': np.linspace(0, 3600, len(sample_data))
                        })
                        trajectory_df = synthetic_traj
                        self.logger.info(f"✅ Created synthetic trajectory with {len(synthetic_traj)} points")
            
            # Ensure proper index and data types
            trajectory_df = trajectory_df.reset_index(drop=True)
            
            # CRITICAL FIX: Validate and fix trajectory data structure
            if len(trajectory_df) > 0:
                # Check for required columns
                required_cols = ['VehId', 'lat', 'lon', 'timestamp']
                missing_cols = [col for col in required_cols if col not in trajectory_df.columns]
                if missing_cols:
                    self.logger.warning(f"⚠️ Missing columns {missing_cols} in trajectory data")
                    # Add missing columns with default values
                    for col in missing_cols:
                        if col == 'VehId':
                            trajectory_df[col] = 'unknown_vehicle'
                        elif col == 'timestamp':
                            trajectory_df[col] = np.linspace(0, 3600, len(trajectory_df))
                        else:
                            trajectory_df[col] = 0.0
                
                # CRITICAL FIX: Ensure numeric data types for coordinates with proper error handling
                trajectory_df['lat'] = pd.to_numeric(trajectory_df['lat'], errors='coerce')
                trajectory_df['lon'] = pd.to_numeric(trajectory_df['lon'], errors='coerce')
                
                # Drop any rows with invalid coordinates
                initial_count = len(trajectory_df)
                trajectory_df = trajectory_df.dropna(subset=['lat', 'lon'])
                final_count = len(trajectory_df)
                if initial_count != final_count:
                    self.logger.warning(f"⚠️ Removed {initial_count - final_count} rows with invalid coordinates")
                
                # Additional validation: ensure coordinates are within reasonable bounds
                trajectory_df = trajectory_df[
                    (trajectory_df['lat'] >= -90) & (trajectory_df['lat'] <= 90) &
                    (trajectory_df['lon'] >= -180) & (trajectory_df['lon'] <= 180)
                ]
                
                # Ensure timestamps are properly formatted
                if 'timestamp' in trajectory_df.columns:
                    trajectory_df['timestamp'] = pd.to_numeric(trajectory_df['timestamp'], errors='coerce')
                    trajectory_df = trajectory_df.dropna(subset=['timestamp'])
                    if len(trajectory_df) > 0:
                        # Normalize timestamps to 0-3600 range
                        min_ts = trajectory_df['timestamp'].min()
                        max_ts = trajectory_df['timestamp'].max()
                        if max_ts > min_ts:
                            trajectory_df['timestamp'] = (trajectory_df['timestamp'] - min_ts) / (max_ts - min_ts) * 3600
                        else:
                            trajectory_df['timestamp'] = 0
            else:
                self.logger.warning("⚠️ Empty trajectory DataFrame after processing")
            
            # Store both DataFrame and dict formats for different components
            trajectory_data = {
                'dataframe': trajectory_df,
                'dict': trajectories
            }
            
            self.logger.info(f"✅ Created trajectory DataFrame with {len(trajectory_df)} points")
            
            # Calculate grid bounds from actual grid data
            target_grid = next((cell for cell in city_grids_data if cell.get('grid_id') == test_grid_id), None)
            if target_grid:
                grid_bounds = {
                    'min_lat': target_grid['min_lat'],
                    'max_lat': target_grid['max_lat'],
                    'min_lon': target_grid['min_lon'],
                    'max_lon': target_grid['max_lon']
                }
                self.logger.info(f"✅ Using actual grid bounds: lat=[{grid_bounds['min_lat']:.4f}, {grid_bounds['max_lat']:.4f}], lon=[{grid_bounds['min_lon']:.4f}, {grid_bounds['max_lon']:.4f}]")
            else:
                # Fallback bounds
                grid_bounds = {
                    'min_lat': 42.25, 'max_lat': 42.31,
                    'min_lon': -83.75, 'max_lon': -83.73
                }
                self.logger.warning("⚠️ Using fallback grid bounds")
            
            # Generate proper network file for the actual city
            network_file = self._generate_city_network_file(city_name, coordinates)
            if not network_file or not os.path.exists(network_file):
                self.logger.error(f"❌ Failed to generate network file for {city_name}")
                return None
            
            # Use actual grid bounds from the city data
            if target_grid:
                grid_bounds = {
                    'min_lat': target_grid['min_lat'],
                    'max_lat': target_grid['max_lat'],
                    'min_lon': target_grid['min_lon'],
                    'max_lon': target_grid['max_lon']
                }
                self.logger.info(f"✅ Using actual city grid bounds: lat=[{grid_bounds['min_lat']:.4f}, {grid_bounds['max_lat']:.4f}], lon=[{grid_bounds['min_lon']:.4f}, {grid_bounds['max_lon']:.4f}]")
            else:
                # Fallback bounds based on city coordinates
                if coordinates:
                    center_lat, center_lon = coordinates
                    grid_bounds = {
                        'min_lat': center_lat - 0.05, 'max_lat': center_lat + 0.05,
                        'min_lon': center_lon - 0.05, 'max_lon': center_lon + 0.05
                    }
                else:
                    grid_bounds = {
                        'min_lat': 42.25, 'max_lat': 42.31,
                        'min_lon': -83.75, 'max_lon': -83.73
                    }
                self.logger.warning("⚠️ Using fallback grid bounds")
            
            # Validate network file
            try:
                try:
                    import sumolib
                    test_net = sumolib.net.readNet(network_file)
                    edge_count = len(test_net.getEdges())
                    self.logger.info(f"✅ Network validation: {edge_count} edges found")
                except ImportError:
                    self.logger.warning("⚠️ SUMO not available for network validation, skipping")
                    edge_count = 15791  # Known value from previous runs
            except Exception as e:
                self.logger.error(f"❌ Network file validation failed: {e}")
                return None
            
            self.test_data = {
                'test_grid_id': test_grid_id,
                'trajectory_df': trajectory_data['dataframe'],
                'trajectory_dict': trajectory_data['dict'],
                'grid_bounds': grid_bounds,
                'network_file': network_file,
                'city_grids_data': city_grids_data
            }
            
            return self.test_data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load test data: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _create_enhanced_trajectory_data(self, method_name):
        """Create enhanced trajectory data for better simulation evaluation."""
        try:
            # Start with existing trajectory data
            if self.test_data and 'trajectory_df' in self.test_data:
                trajectory_df = self.test_data['trajectory_df'].copy()
            else:
                trajectory_df = pd.DataFrame()
            
            # If we have real data, use it
            if len(trajectory_df) > 0 and 'VehId' in trajectory_df.columns:
                # Clean and validate existing data
                trajectory_df = trajectory_df.dropna(subset=['lat', 'lon'])
                trajectory_df['lat'] = pd.to_numeric(trajectory_df['lat'], errors='coerce')
                trajectory_df['lon'] = pd.to_numeric(trajectory_df['lon'], errors='coerce')
                trajectory_df['timestamp'] = pd.to_numeric(trajectory_df['timestamp'], errors='coerce')
                trajectory_df = trajectory_df.dropna(subset=['lat', 'lon', 'timestamp'])
                
                if len(trajectory_df) > 0:
                    # Normalize timestamps
                    min_ts = trajectory_df['timestamp'].min()
                    max_ts = trajectory_df['timestamp'].max()
                    if max_ts > min_ts:
                        trajectory_df['timestamp'] = (trajectory_df['timestamp'] - min_ts) / (max_ts - min_ts) * 3600
                    else:
                        trajectory_df['timestamp'] = 0
                    
                    self.logger.info(f"Using {len(trajectory_df)} real trajectory points for {method_name}")
                    return trajectory_df
            
            # Create synthetic trajectory data for better simulation
            self.logger.info(f"Creating synthetic trajectory data for {method_name}")
            
            # Generate realistic trajectories within grid bounds
            grid_bounds = self.test_data.get('grid_bounds', {
                'min_lat': 42.25, 'max_lat': 42.31,
                'min_lon': -83.75, 'max_lon': -83.73
            })
            
            # Create multiple vehicle trajectories
            trajectories = []
            num_vehicles = 3  # Same number of vehicles for all methods
            
            for i in range(num_vehicles):
                vehicle_id = f"vehicle_{i}"
                
                # Create a realistic trajectory path
                num_points = np.random.randint(10, 20)
                
                # Start point within grid bounds
                start_lat = np.random.uniform(grid_bounds['min_lat'], grid_bounds['max_lat'])
                start_lon = np.random.uniform(grid_bounds['min_lon'], grid_bounds['max_lon'])
                
                # Create a path that moves through the grid
                lats = [start_lat]
                lons = [start_lon]
                timestamps = [0]
                
                for j in range(1, num_points):
                    # Move in a realistic pattern
                    lat_delta = np.random.normal(0, 0.005)  # Small movements
                    lon_delta = np.random.normal(0, 0.005)
                    
                    new_lat = max(grid_bounds['min_lat'], 
                                min(grid_bounds['max_lat'], lats[-1] + lat_delta))
                    new_lon = max(grid_bounds['min_lon'], 
                                min(grid_bounds['max_lon'], lons[-1] + lon_delta))
                    
                    lats.append(new_lat)
                    lons.append(new_lon)
                    timestamps.append(j * 60)  # 1 minute intervals
                
                # Create DataFrame for this vehicle
                vehicle_df = pd.DataFrame({
                    'VehId': vehicle_id,
                    'lat': lats,
                    'lon': lons,
                    'timestamp': timestamps
                })
                trajectories.append(vehicle_df)
            
            # Combine all trajectories
            if trajectories:
                trajectory_df = pd.concat(trajectories, ignore_index=True)
                self.logger.info(f"Created synthetic trajectory data with {len(trajectory_df)} points for {method_name}")
                return trajectory_df
            else:
                # Fallback: create minimal trajectory
                trajectory_df = pd.DataFrame({
                    'VehId': 'fallback_vehicle',
                    'lat': [grid_bounds['min_lat'] + (grid_bounds['max_lat'] - grid_bounds['min_lat']) / 2],
                    'lon': [grid_bounds['min_lon'] + (grid_bounds['max_lon'] - grid_bounds['min_lon']) / 2],
                    'timestamp': [0]
                })
                self.logger.info(f"Created minimal fallback trajectory for {method_name}")
                return trajectory_df
                
        except Exception as e:
            self.logger.error(f"Error creating enhanced trajectory data: {e}")
            return pd.DataFrame()

    def get_test_data(self):
        """Get the loaded test data."""
        return self.test_data
    
    def list_available_grids(self):
        """List all available grids with their data counts."""
        self.logger.info("📋 Listing available grids...")
        
        try:
            # Load VED data
            raw_ved_df = pd.read_parquet("./data/validation/ved_processed_with_grids.parquet")
            
            # Create grid using CityGridding
            gridder = CityGridding(primary_grid_size_km=1.0, fetch_osm_features=False)
            city_grids_data = gridder.create_city_grid("Ann Arbor, Michigan, USA", coordinates=(42.2808, -83.7430))
            
            # Assign VED data to grids
            try:
                df = self.assign_ved_to_grid(raw_ved_df, city_grids_data)
            except ImportError:
                # Fallback: simple grid assignment
                df = raw_ved_df.copy()
                if 'lat' in df.columns and 'lon' in df.columns:
                    df['grid_id'] = city_grids_data[0]['grid_id']
                else:
                    self.logger.error("❌ No coordinate columns found for grid assignment")
                    return []
            
            # Count data points per grid
            grids_with_data = df['grid_id'].value_counts().sort_values(ascending=False)
            
            self.logger.info("📊 Available grids:")
            self.logger.info("=" * 60)
            self.logger.info(f"{'Grid ID':<30} {'Data Points':<15} {'Status'}")
            self.logger.info("-" * 60)
            
            for grid_id, count in grids_with_data.items():
                status = "✅ Recommended" if count > 100 else "⚠️ Low data" if count > 10 else "❌ Very low data"
                self.logger.info(f"{grid_id:<30} {count:<15} {status}")
            
            self.logger.info("=" * 60)
            self.logger.info(f"Total grids: {len(grids_with_data)}")
            self.logger.info(f"Total data points: {len(df)}")
            
            return grids_with_data.to_dict()
            
        except Exception as e:
            self.logger.error(f"❌ Error listing grids: {e}")
            return {}

    def validate_data(self):
        """Validate that all required data is available and properly formatted."""
        if not self.test_data:
            self.logger.error("❌ No test data loaded")
            return False
        
        required_keys = ['test_grid_id', 'trajectory_df', 'grid_bounds', 'network_file']
        missing_keys = [key for key in required_keys if key not in self.test_data]
        
        if missing_keys:
            self.logger.error(f"❌ Missing required data keys: {missing_keys}")
            return False
        
        # Validate trajectory data
        if len(self.test_data['trajectory_df']) == 0:
            self.logger.warning("⚠️ Empty trajectory data")
        
        # Validate network file
        if not os.path.exists(self.test_data['network_file']):
            self.logger.error(f"❌ Network file not found: {self.test_data['network_file']}")
            return False
        
        self.logger.info("✅ Data validation passed")
        return True
    
    def _determine_data_source(self):
        """Determine the data source and city information."""
        # Priority: custom data > synthetic trajectories (no UrbanEV fallback)
        if self.custom_data_path and os.path.exists(self.custom_data_path):
            self.logger.info("📁 Using custom VED data file")
            return "custom", self.city_name or "Custom City", None
        
        # Always use synthetic trajectories for any city (no UrbanEV fallback)
        if self.city_name:
            self.logger.info(f"🎲 Generating synthetic trajectories for {self.city_name}")
            # Try to get coordinates for the city
            coordinates = self._get_city_coordinates(self.city_name)
            return "synthetic_trajectories", self.city_name, coordinates
        else:
            self.logger.info("🎲 Generating synthetic trajectories for default city")
            return "synthetic_trajectories", "Singapore", (1.3521, 103.8198)
    
    def _load_custom_data(self):
        """Load custom data from CSV or Parquet file."""
        try:
            if self.custom_data_path.endswith('.parquet'):
                df = pd.read_parquet(self.custom_data_path)
            elif self.custom_data_path.endswith('.csv'):
                df = pd.read_csv(self.custom_data_path)
            else:
                raise ValueError(f"Unsupported file format: {self.custom_data_path}")
            
            # Validate required columns
            required_cols = ['lat', 'lon', 'kwh', 'hours']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            self.logger.info(f"✅ Loaded custom data: {len(df)} records")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load custom data: {e}")
            raise
    
    
    def _generate_synthetic_data(self, city_name, coordinates):
        """Generate synthetic trajectory data for SUMO simulation."""
        try:
            self.logger.info(f"🎲 Generating synthetic trajectories for {city_name}")
            
            # Generate synthetic trajectories for SUMO simulation
            synthetic_df = self._create_synthetic_trajectories(city_name, coordinates)
            
            self.logger.info(f"✅ Generated synthetic trajectories: {len(synthetic_df)} records")
            return synthetic_df
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate synthetic trajectories: {e}")
            # Fallback: create minimal synthetic data
            return self._create_minimal_synthetic_data(city_name, coordinates)
    
    def _create_minimal_synthetic_data(self, city_name, coordinates):
        """Create minimal synthetic data as fallback."""
        self.logger.warning("⚠️ Creating minimal synthetic data as fallback")
        
        # Generate random data points around the city center
        if coordinates:
            center_lat, center_lon = coordinates
        else:
            center_lat, center_lon = 0.0, 0.0
        
        # Create random data points
        np.random.seed(self.random_seed)
        n_points = 500
        
        # Random points within ~10km radius
        lat_offset = np.random.normal(0, 0.05, n_points)  # ~5km
        lon_offset = np.random.normal(0, 0.05, n_points)
        
        synthetic_data = pd.DataFrame({
            'lat': center_lat + lat_offset,
            'lon': center_lon + lon_offset,
            'kwh': np.random.exponential(50, n_points),  # kWh charging
            'hours': np.random.exponential(2, n_points),  # Hours charging
            'timestamp': pd.date_range('2023-01-01', periods=n_points, freq='H'),
            'station_id': [f"synthetic_{i}" for i in range(n_points)]
        })
        
        self.logger.info(f"✅ Created minimal synthetic data: {len(synthetic_data)} records")
        return synthetic_data
    
    def _get_city_coordinates(self, city_name):
        """Get coordinates for a city name."""
        try:
            import osmnx as ox
            # Try to geocode the city
            gdf = ox.geocode_to_gdf(city_name)
            if not gdf.empty:
                centroid = gdf.geometry.iloc[0].centroid
                return (centroid.y, centroid.x)  # (lat, lon)
        except Exception as e:
            self.logger.warning(f"Could not geocode {city_name}: {e}")
        
        # Return None if geocoding fails
        return None
    
    def _create_synthetic_trajectories(self, city_name, coordinates):
        """Create synthetic trajectories for SUMO simulation using robust randomTrips.py approach."""
        self.logger.info("🎲 Creating synthetic trajectories for SUMO simulation...")
        
        # Generate random data points around the city center
        if coordinates:
            center_lat, center_lon = coordinates
        else:
            center_lat, center_lon = 0.0, 0.0
        
        # Create robust synthetic trajectory data inspired by SUMO randomTrips.py
        np.random.seed(self.random_seed)
        n_vehicles = 2000  # Large number of vehicles for realistic simulation
        
        trajectories = []
        
        # Define realistic trajectory patterns inspired by randomTrips.py
        trajectory_patterns = [
            'commuter',      # Home to work patterns
            'shopping',      # Shopping center trips
            'leisure',       # Recreational trips
            'business',      # Business district trips
            'residential'    # Local neighborhood trips
        ]
        
        for i in range(n_vehicles):
            vehicle_id = f"vehicle_{i}"
            
            # Select trajectory pattern
            pattern = np.random.choice(trajectory_patterns)
            
            # Generate realistic trajectory based on pattern
            if pattern == 'commuter':
                # Long-distance commuter trips
                n_points = np.random.randint(15, 30)
                lat_spread = np.random.uniform(0.01, 0.05)  # 1-5km spread
                lon_spread = np.random.uniform(0.01, 0.05)
            elif pattern == 'shopping':
                # Medium-distance shopping trips
                n_points = np.random.randint(10, 20)
                lat_spread = np.random.uniform(0.005, 0.02)  # 0.5-2km spread
                lon_spread = np.random.uniform(0.005, 0.02)
            elif pattern == 'leisure':
                # Variable leisure trips
                n_points = np.random.randint(8, 25)
                lat_spread = np.random.uniform(0.003, 0.03)  # 0.3-3km spread
                lon_spread = np.random.uniform(0.003, 0.03)
            elif pattern == 'business':
                # Business district trips
                n_points = np.random.randint(12, 22)
                lat_spread = np.random.uniform(0.008, 0.025)  # 0.8-2.5km spread
                lon_spread = np.random.uniform(0.008, 0.025)
            else:  # residential
                # Short local trips
                n_points = np.random.randint(6, 15)
                lat_spread = np.random.uniform(0.002, 0.01)  # 0.2-1km spread
                lon_spread = np.random.uniform(0.002, 0.01)
            
            # Generate start and end points with realistic distribution
            start_lat = center_lat + np.random.normal(0, lat_spread/2)
            start_lon = center_lon + np.random.normal(0, lon_spread/2)
            end_lat = center_lat + np.random.normal(0, lat_spread/2)
            end_lon = center_lon + np.random.normal(0, lon_spread/2)
            
            # Ensure points are within reasonable bounds
            start_lat = np.clip(start_lat, center_lat - lat_spread, center_lat + lat_spread)
            start_lon = np.clip(start_lon, center_lon - lon_spread, center_lon + lon_spread)
            end_lat = np.clip(end_lat, center_lat - lat_spread, center_lat + lat_spread)
            end_lon = np.clip(end_lon, center_lon - lon_spread, center_lon + lon_spread)
            
            # Generate trajectory points with realistic movement patterns
            trajectory_points = []
            
            for j in range(n_points):
                # Use different interpolation methods for different patterns
                if pattern == 'commuter':
                    # Linear interpolation for commuter trips
                    t = j / (n_points - 1) if n_points > 1 else 0
                    lat = start_lat + t * (end_lat - start_lat)
                    lon = start_lon + t * (end_lon - start_lon)
                elif pattern == 'shopping':
                    # Slightly curved path for shopping trips
                    t = j / (n_points - 1) if n_points > 1 else 0
                    # Add slight curve
                    curve_factor = 0.1 * np.sin(np.pi * t)
                    lat = start_lat + t * (end_lat - start_lat) + curve_factor * lat_spread * 0.1
                    lon = start_lon + t * (end_lon - start_lon) + curve_factor * lon_spread * 0.1
                else:
                    # Random walk for other patterns
                    if j == 0:
                        lat, lon = start_lat, start_lon
                    else:
                        # Move towards destination with some randomness
                        progress = j / (n_points - 1) if n_points > 1 else 0
                        target_lat = start_lat + progress * (end_lat - start_lat)
                        target_lon = start_lon + progress * (end_lon - start_lon)
                        
                        # Add realistic movement noise
                        noise_lat = np.random.normal(0, lat_spread * 0.05)
                        noise_lon = np.random.normal(0, lon_spread * 0.05)
                        
                        lat = target_lat + noise_lat
                        lon = target_lon + noise_lon
                
                # Add realistic timestamp progression
                if pattern == 'commuter':
                    # Commuter trips: 30-60 minutes
                    duration = np.random.uniform(30, 60)  # minutes
                elif pattern == 'shopping':
                    # Shopping trips: 15-45 minutes
                    duration = np.random.uniform(15, 45)
                elif pattern == 'leisure':
                    # Leisure trips: 20-90 minutes
                    duration = np.random.uniform(20, 90)
                elif pattern == 'business':
                    # Business trips: 10-40 minutes
                    duration = np.random.uniform(10, 40)
                else:  # residential
                    # Residential trips: 5-25 minutes
                    duration = np.random.uniform(5, 25)
                
                timestamp = pd.Timestamp.now() + pd.Timedelta(minutes=j * duration / n_points)
                
                trajectory_points.append({
                    'lat': lat,
                    'lon': lon,
                    'timestamp': timestamp,
                    'vehicle_id': vehicle_id,
                    'point_order': j,
                    'pattern': pattern
                })
            
            trajectories.extend(trajectory_points)
        
        # Create DataFrame
        synthetic_df = pd.DataFrame(trajectories)
        
        # Add realistic charging-related columns
        # Different charging patterns based on trip type
        charging_data = []
        for _, row in synthetic_df.iterrows():
            pattern = row['pattern']
            
            if pattern == 'commuter':
                # Commuters: higher energy consumption, longer charging
                kwh = np.random.exponential(60)  # Higher energy consumption
                hours = np.random.exponential(2.5)  # Longer charging time
            elif pattern == 'shopping':
                # Shopping: medium energy consumption
                kwh = np.random.exponential(35)
                hours = np.random.exponential(1.8)
            elif pattern == 'leisure':
                # Leisure: variable energy consumption
                kwh = np.random.exponential(45)
                hours = np.random.exponential(2.0)
            elif pattern == 'business':
                # Business: quick charging
                kwh = np.random.exponential(25)
                hours = np.random.exponential(1.2)
            else:  # residential
                # Residential: lower energy consumption
                kwh = np.random.exponential(20)
                hours = np.random.exponential(1.0)
            
            charging_data.append({
                'kwh': kwh,
                'hours': hours
            })
        
        charging_df = pd.DataFrame(charging_data)
        synthetic_df = pd.concat([synthetic_df, charging_df], axis=1)
        
        # Add station IDs with realistic distribution
        n_stations = 50  # Realistic number of stations
        station_ids = [f"synthetic_station_{i%n_stations}" for i in range(len(synthetic_df))]
        synthetic_df['station_id'] = station_ids
        
        # Add vehicle type information
        synthetic_df['vehicle_type'] = synthetic_df['pattern'].map({
            'commuter': 'sedan',
            'shopping': 'suv', 
            'leisure': 'crossover',
            'business': 'sedan',
            'residential': 'compact'
        })
        
        self.logger.info(f"✅ Created {len(trajectories)} trajectory points for {n_vehicles} vehicles")
        self.logger.info(f"   Pattern distribution: {synthetic_df['pattern'].value_counts().to_dict()}")
        self.logger.info(f"   Vehicle types: {synthetic_df['vehicle_type'].value_counts().to_dict()}")
        
        return synthetic_df
    
    def get_city_grids_data(self):
        """Get the city grids data."""
        if hasattr(self, 'test_data') and self.test_data:
            return self.test_data.get('city_grids_data', [])
        return []
    
    def get_test_data(self):
        """Get the test data."""
        return self.test_data
    
    def load_best_model_path(self) -> str:
        """
        Load the path to the best performing demand model.
        
        Returns:
            Path to the best model (demand_score_kwh_only_original_quantile_xgboost.pkl)
        """
        # Use the specific model with highest correlation as requested
        best_model_path = 'models/demand_score_kwh_only_original_quantile_xgboost.pkl'
        
        if os.path.exists(best_model_path):
            self.logger.info(f"🎯 Using best performing model: {best_model_path}")
            self.logger.info(f"   Model type: demand_score_kwh_only_original_quantile_xgboost")
            return best_model_path
        else:
            self.logger.error(f"❌ Best model not found at {best_model_path}")
            raise FileNotFoundError(f"Required model file not found: {best_model_path}")
    
    def assign_ved_to_grid(self, ved_df: pd.DataFrame, grid_cells: List[Dict]) -> gpd.GeoDataFrame:
        """Spatially join VED points to the provided grid cells."""
        self.logger.info(f"Assigning {len(ved_df)} VED points to {len(grid_cells)} grid cells...")
        
        # Drop conflicting columns if they exist
        cols_to_drop = ['index_left', 'index_right']
        ved_df_clean = ved_df.drop(columns=[col for col in cols_to_drop if col in ved_df.columns])
        
        # Create GeoDataFrame for VED data
        geometry = [Point(xy) for xy in zip(ved_df_clean['lon'], ved_df_clean['lat'])]
        ved_gdf = gpd.GeoDataFrame(ved_df_clean, geometry=geometry, crs="EPSG:4326")
        
        # Create GeoDataFrame for grid cells
        grid_polygons = []
        grid_data = []
        for cell in grid_cells:
            poly = Polygon([
                (cell['min_lon'], cell['min_lat']),
                (cell['min_lon'], cell['max_lat']),
                (cell['max_lon'], cell['max_lat']),
                (cell['max_lon'], cell['min_lat'])
            ])
            grid_polygons.append(poly)
            
            # Handle both grid_id and cell_id formats
            grid_id = cell.get('grid_id', cell.get('cell_id', f"grid_{len(grid_data)}"))
            grid_data.append({'grid_id': grid_id})
            
        grid_gdf = gpd.GeoDataFrame(grid_data, geometry=grid_polygons, crs="EPSG:4326")
        
        # Perform spatial join
        gridded_ved = gpd.sjoin(ved_gdf, grid_gdf, how="inner", predicate="within")
        
        self.logger.info(f"Successfully assigned {len(gridded_ved)} points to {gridded_ved['grid_id'].nunique()} grids.")
        return gridded_ved
    
    def allocate_stations_from_demand(self, city_grids_df: pd.DataFrame, total_station_budget: int) -> Dict[str, int]:
        """
        Allocates a total budget of EV charging stations across city grid cells
        based on predicted demand scores using the best performing model.

        Args:
            city_grids_df (pd.DataFrame): DataFrame where each row represents a grid cell
                                       and includes its 'grid_id' and OSM features required
                                       for demand prediction.
            total_station_budget (int): The total number of charging stations to allocate
                                        across the entire city.

        Returns:
            Dict[str, int]: A dictionary mapping 'grid_id' to the final allocated
                            number of charging stations for that grid.
        """
        self.logger.info(f"🎯 Allocating {total_station_budget} stations using ML demand prediction...")
        
        if city_grids_df.empty:
            self.logger.warning("City grids DataFrame is empty, returning no allocations.")
            return {}

        # Step 1: Predict Demand (OSM features should already be in city_grids_df)
        try:
            # Use the best validated model
            best_model_path = self.load_best_model_path()
            predictor = DemandPredictor(model_path=best_model_path)
            
            # Verify that necessary feature columns are present
            if not all(col in city_grids_df.columns for col in predictor.feature_columns):
                missing_cols = [col for col in predictor.feature_columns if col not in city_grids_df.columns]
                self.logger.error(f"❌ Missing required feature columns for prediction: {missing_cols}")
                raise ValueError("Input DataFrame is missing required feature columns for demand prediction.")

            predicted_scores = predictor.predict(city_grids_df)
            city_grids_df['demand_score'] = predicted_scores
            self.logger.info(f"✅ Predicted demand scores for {len(predicted_scores)} grid cells")
        except Exception as e:
            self.logger.error(f"❌ Failed to predict demand scores: {e}. Aborting allocation.")
            return {}
        
        # Step 2: Normalize and Calculate Share
        total_predicted_demand = city_grids_df['demand_score'].sum()

        if total_predicted_demand <= 0:
            self.logger.warning("Total predicted demand is zero or negative. Using uniform allocation as a fallback.")
            num_grids = len(city_grids_df)
            city_grids_df['share'] = 1 / num_grids if num_grids > 0 else 0
        else:
            city_grids_df['share'] = city_grids_df['demand_score'] / total_predicted_demand
        
        # Validate normalization: sum of shares should equal 1
        share_sum = city_grids_df['share'].sum()
        self.logger.info(f"📊 Demand normalization: sum of shares = {share_sum:.6f} (should be 1.0)")
        if abs(share_sum - 1.0) > 1e-6:
            self.logger.warning(f"⚠️ Normalization error: sum = {share_sum}, expected 1.0")

        # Step 3: Base Allocation
        city_grids_df['base_allocation'] = np.floor(city_grids_df['share'] * total_station_budget)
        
        # Step 4: Allocate Remainder via Priority
        allocated_total = city_grids_df['base_allocation'].sum()
        remaining_stations = int(total_station_budget - allocated_total)
        
        if remaining_stations > 0:
            city_grids_df['priority'] = (city_grids_df['share'] * total_station_budget) - city_grids_df['base_allocation']
            
            # Step 5: Final Assignment
            # Sort by priority (desc) and then by grid_id (asc) to ensure deterministic tie-breaking
            sorted_for_remainder = city_grids_df.sort_values(by=['priority', 'grid_id'], ascending=[False, True])
            top_priority_indices = sorted_for_remainder.head(remaining_stations).index
            
            city_grids_df.loc[top_priority_indices, 'base_allocation'] += 1

        # Create final allocation dictionary
        city_grids_df['final_allocation'] = city_grids_df['base_allocation'].astype(int)
        
        allocation = {
            str(row['grid_id']): int(row['final_allocation'])
            for _, row in city_grids_df.iterrows() if row['final_allocation'] > 0
        }
        
        final_allocated_count = sum(allocation.values())
        self.logger.info(f"✅ Successfully allocated {final_allocated_count}/{total_station_budget} stations to {len(allocation)} grids.")
        
        if final_allocated_count != total_station_budget:
            self.logger.warning(f"⚠️ Final allocation count {final_allocated_count} does not match budget {total_station_budget}.")

        return allocation
    
    def allocate_stations_simple(self, city_grids_df: pd.DataFrame, stations_per_grid: int = 2) -> Dict[str, int]:
        """
        Simple allocation: assign the same number of stations to each grid for testing.
        
        Args:
            city_grids_df (pd.DataFrame): DataFrame where each row represents a grid cell
            stations_per_grid (int): Number of stations to assign to each grid
            
        Returns:
            Dict[str, int]: A dictionary mapping grid_id to number of stations
        """
        self.logger.info(f"🔧 Simple allocation: {stations_per_grid} stations per grid for {len(city_grids_df)} grids")
        
        if city_grids_df.empty:
            self.logger.warning("City grids DataFrame is empty, returning no allocations.")
            return {}
        
        # Use grid_id if available, otherwise use cell_id
        grid_id_col = 'grid_id' if 'grid_id' in city_grids_df.columns else 'cell_id'
        
        allocation = {}
        for _, row in city_grids_df.iterrows():
            grid_id = str(row[grid_id_col])
            allocation[grid_id] = stations_per_grid
        
        total_stations = sum(allocation.values())
        self.logger.info(f"✅ Allocated {total_stations} stations ({stations_per_grid} each) to {len(allocation)} grids")
        
        return allocation
    
    def load_all_grids_with_station_allocation(self):
        """
        Load all grids and perform station allocation using ML demand prediction.
        
        Returns:
            Dict containing all grids data and station allocations
        """
        self.logger.info("📊 Loading all grids with station allocation...")
        
        try:
            # Load VED data
            raw_ved_df = pd.read_parquet("./data/validation/ved_processed_with_grids.parquet")
            self.logger.info(f"✅ Loaded {len(raw_ved_df)} VED data points")
            
            # Create grid using CityGridding
            gridder = CityGridding(primary_grid_size_km=1.0, fetch_osm_features=False)
            city_grids_data = gridder.create_city_grid("Ann Arbor, Michigan, USA", coordinates=(42.2808, -83.7430))
            self.logger.info(f"✅ Created {len(city_grids_data)} grid cells")
            
            # Assign VED data to grids
            df = self.assign_ved_to_grid(raw_ved_df, city_grids_data)
            self.logger.info(f"✅ Assigned {len(df)} data points to grids")
            
            # Extract OSM Features and Apply MLPrediction Pipeline
            self.logger.info("🗺️ Extracting OSM features and applying MLPrediction pipeline...")
            try:
                # Step 1: Extract basic OSM features
                osm_extractor = PortableOSMExtractor(use_cache=True)
                osm_features_df = osm_extractor.extract_features_for_grids(
                    grid_cells=city_grids_data,
                    city_name="Ann Arbor, Michigan, USA"
                )
                self.logger.info(f"✅ Extracted basic OSM features for {len(osm_features_df)} grid cells")
                
                # Step 2: Load the model to check feature requirements
                best_model_path = self.load_best_model_path()
                
                # Load model to check what features it expects
                with open(best_model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                expected_features = model_data.get('feature_columns', [])
                basic_osm_features = set(osm_extractor.get_portable_feature_columns())
                expected_features_set = set(expected_features)
                
                # Check if model needs engineered features (features not in basic OSM set)
                engineered_features_needed = expected_features_set - basic_osm_features
                
                if engineered_features_needed:
                    self.logger.info(f"🔧 Model requires engineered features. Applying MLPrediction feature engineering pipeline...")
                    self.logger.info(f"Engineered features needed: {len(engineered_features_needed)} additional features")
                    
                    # Add lat/lon coordinates (required for spatial features) - following MLPrediction pattern
                    features_with_coords = osm_features_df.copy()
                    city_grids_df = pd.DataFrame(city_grids_data)
                    features_with_coords['latitude'] = [
                        (row['min_lat'] + row['max_lat']) / 2 
                        for _, row in city_grids_df.iterrows()
                    ]
                    features_with_coords['longitude'] = [
                        (row['min_lon'] + row['max_lon']) / 2 
                        for _, row in city_grids_df.iterrows()
                    ]
                    
                    # Apply complete feature engineering pipeline following MLPrediction structure
                    feature_engineer = UrbanFeatureEngineer()
                    engineered_features = feature_engineer.apply_complete_feature_engineering(
                        features_with_coords, 
                        skip_spatial_features=False  # Enable spatial features
                    )
                    
                    # Use engineered features for prediction
                    prediction_features = engineered_features
                    self.logger.info(f"✅ MLPrediction feature engineering complete: {len(osm_features_df.columns)} → {len(engineered_features.columns)} features")
                else:
                    self.logger.info("✅ Model uses basic OSM features only")
                    prediction_features = osm_features_df
                
                # Step 3: Merge features with grid data for allocation
                city_grids_df = city_grids_df.merge(prediction_features, on='grid_id', how='left')
                
                # Step 4: Use demand-based allocation
                station_allocations = self.allocate_stations_from_demand(city_grids_df, self.total_station_budget)
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to extract OSM features or apply MLPrediction pipeline: {e}")
                self.logger.info("Using simple allocation as fallback...")
                city_grids_df = pd.DataFrame(city_grids_data)
                station_allocations = self.allocate_stations_simple(city_grids_df, stations_per_grid=2)
            
            if not station_allocations:
                self.logger.error("🔴 FATAL: Station allocation failed. Cannot proceed.")
                return None

            self.logger.info(f"✅ Station allocation: {sum(station_allocations.values())} stations across {len(station_allocations)} grids")
            
            # Print detailed allocation breakdown
            self.logger.info(f"\n📊 DETAILED STATION ALLOCATION BREAKDOWN:")
            self.logger.info(f"{'Grid ID':<15} {'Stations':<10} {'Percentage':<12} {'Demand Score':<15}")
            self.logger.info("-" * 60)
            
            # Calculate percentages for display
            total_stations = sum(station_allocations.values())
            for grid_id, num_stations in sorted(station_allocations.items(), key=lambda x: x[1], reverse=True):
                percentage = (num_stations / total_stations) * 100 if total_stations > 0 else 0
                # Get demand score if available
                demand_score = city_grids_df[city_grids_df['grid_id'] == grid_id]['demand_score'].iloc[0] if 'demand_score' in city_grids_df.columns else 'N/A'
                demand_str = f"{demand_score:.4f}" if demand_score != 'N/A' else "N/A"
                self.logger.info(f"{grid_id:<15} {num_stations:<10} {percentage:>8.1f}%     {demand_str:<15}")
            
            self.logger.info(f"\n📈 ALLOCATION STATISTICS:")
            self.logger.info(f"   • Total stations allocated: {total_stations}")
            self.logger.info(f"   • Grids with stations: {len(station_allocations)}")
            self.logger.info(f"   • Average stations per grid: {total_stations/len(station_allocations):.2f}")
            self.logger.info(f"   • Max stations in single grid: {max(station_allocations.values())}")
            self.logger.info(f"   • Min stations in single grid: {min(station_allocations.values())}")
            
            # Store allocations for later use
            self.station_allocations = station_allocations
            
            return {
                'city_grids_data': city_grids_data,
                'station_allocations': station_allocations,
                'ved_data': df,
                'total_stations': total_stations,
                'allocation_df': pd.DataFrame([
                    {'grid_id': grid_id, 'allocated_stations': num_stations}
                    for grid_id, num_stations in station_allocations.items()
                ])
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load grids with station allocation: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def get_station_allocations(self):
        """Get the station allocations."""
        return self.station_allocations
    
    def _generate_city_network_file(self, city_name, coordinates):
        """Generate SUMO network file for the specific city."""
        try:
            self.logger.info(f"🗺️ Generating network file for {city_name}")
            
            # Create output directory
            network_dir = os.path.abspath("generated_files/city_network")
            os.makedirs(network_dir, exist_ok=True)
            
            # Create safe filename from city name
            safe_city_name = "".join(c for c in city_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_city_name = safe_city_name.replace(' ', '_').lower()
            
            osm_file = os.path.join(network_dir, f"{safe_city_name}.osm")
            net_file = os.path.join(network_dir, f"{safe_city_name}.osm.net.xml")
            
            # Check if network file already exists
            if os.path.exists(net_file):
                self.logger.info(f"✅ Using existing network file: {net_file}")
                return net_file
            
            # Download OSM data for the city
            if not self._download_city_osm_data(city_name, coordinates, osm_file):
                self.logger.error(f"❌ Failed to download OSM data for {city_name}")
                return None
            
            # Convert OSM to SUMO network
            if not self._convert_osm_to_sumo_network(osm_file, net_file):
                self.logger.error(f"❌ Failed to convert OSM to SUMO network for {city_name}")
                return None
            
            self.logger.info(f"✅ Generated network file: {net_file}")
            return net_file
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate network file for {city_name}: {e}")
            return None
    
    def _download_city_osm_data(self, city_name, coordinates, osm_file):
        """Download OSM data for the city."""
        try:
            import osmnx as ox
            
            # Configure OSMnx
            ox.settings.use_cache = True
            ox.settings.log_console = False
            ox.settings.requests_timeout = 60
            
            self.logger.info(f"📡 Downloading OSM data for {city_name}")
            
            if coordinates:
                center_lat, center_lon = coordinates
                # Create a bounding box around the city center
                bbox = (
                    center_lat + 0.05,  # north
                    center_lat - 0.05,  # south  
                    center_lon + 0.05,  # east
                    center_lon - 0.05   # west
                )
                
                # Download graph from bounding box
                graph = ox.graph_from_bbox(
                    *bbox,
                    network_type='drive',
                    simplify=True,
                    truncate_by_edge=True,
                    clean_periphery=True
                )
            else:
                # Download graph by place name
                graph = ox.graph_from_place(
                    city_name,
                    network_type='drive',
                    simplify=True,
                    truncate_by_edge=True,
                    clean_periphery=True
                )
            
            # Save as OSM XML
            ox.save_graph_xml(graph, filepath=osm_file)
            
            # Verify file was created
            if os.path.exists(osm_file) and os.path.getsize(osm_file) > 0:
                self.logger.info(f"✅ Downloaded OSM data: {osm_file}")
                return True
            else:
                self.logger.error(f"❌ OSM file is empty or missing: {osm_file}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to download OSM data: {e}")
            return False
    
    def _convert_osm_to_sumo_network(self, osm_file, net_file):
        """Convert OSM file to SUMO network file."""
        try:
            import subprocess
            
            self.logger.info(f"🔧 Converting OSM to SUMO network: {osm_file} -> {net_file}")
            
            # Use netconvert to convert OSM to SUMO network
            cmd = [
                'netconvert',
                '--osm-files', osm_file,
                '-o', net_file,
                '--geometry.remove',
                '--roundabouts.guess',
                '--ramps.guess',
                '--junctions.join',
                '--tls.guess-signals',
                '--tls.discard-simple',
                '--remove-edges.by-vclass', 'rail_slow,rail_fast,bicycle,pedestrian',
                '--keep-edges.by-vclass', 'passenger,delivery,taxi,bus,coach,motorcycle',
                '--remove-edges.isolated',
                '--junctions.corner-detail', '5',
                '--output.street-names',
                '--output.original-names',
                '--proj.utm',
                '--ignore-errors.edge-type'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                if os.path.exists(net_file) and os.path.getsize(net_file) > 0:
                    self.logger.info(f"✅ Successfully converted to SUMO network: {net_file}")
                    return True
                else:
                    self.logger.error(f"❌ SUMO network file is empty: {net_file}")
                    return False
            else:
                self.logger.error(f"❌ netconvert failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to convert OSM to SUMO: {e}")
            return False
