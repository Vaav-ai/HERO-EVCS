import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import geopandas as gpd
import hashlib
from shapely.geometry import Polygon, Point

logger = logging.getLogger(__name__)

class DemandScoreCalculator:
    """
    Calculates demand scores from raw EV charging data.
    
    This class processes charging station utilization data (kWh and hours) 
    to create normalized demand scores that serve as ground truth for training
    the demand prediction model.
    """
    
    def __init__(self, kwh_weight: float = 0.7, hours_weight: float = 0.3):
        """
        Initialize the demand score calculator.
        
        Args:
            kwh_weight (float): Weight for kWh component in demand score calculation
            hours_weight (float): Weight for hours component in demand score calculation
        """
        if not np.isclose(kwh_weight + hours_weight, 1.0):
            raise ValueError("kwh_weight and hours_weight must sum to 1.0")
        
        self.kwh_weight = kwh_weight
        self.hours_weight = hours_weight
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized DemandScoreCalculator with weights: kWh={kwh_weight}, hours={hours_weight}")
    
    def calculate_demand_scores(self, 
                              charging_data: pd.DataFrame,
                              grid_id_col: str = 'grid_id',
                              kwh_col: str = 'kwh_dispensed',
                              hours_col: str = 'hours_connected',
                              charger_count_col: str = 'num_chargers') -> pd.DataFrame:
        """
        Calculate demand scores from raw charging data.
        
        Args:
            charging_data (pd.DataFrame): Raw charging station data
            grid_id_col (str): Column name for grid identifiers
            kwh_col (str): Column name for energy dispensed (kWh)
            hours_col (str): Column name for connection time (hours)
            charger_count_col (str): Column name for number of chargers
            
        Returns:
            pd.DataFrame: DataFrame with grid_id and demand_score
        """
        logger.info(f"Calculating demand scores for {len(charging_data)} records")
        
        # Validate input data
        required_cols = [grid_id_col, kwh_col, hours_col, charger_count_col]
        missing_cols = [col for col in required_cols if col not in charging_data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Group by grid and aggregate
        grid_aggregates = charging_data.groupby(grid_id_col).agg({
            kwh_col: 'sum',
            hours_col: 'sum', 
            charger_count_col: 'first'  # Assuming constant per grid
        }).reset_index()
        
        # Calculate per-charger metrics
        grid_aggregates['kwh_per_charger'] = (
            grid_aggregates[kwh_col] / grid_aggregates[charger_count_col].replace(0, 1)
        )
        grid_aggregates['hours_per_charger'] = (
            grid_aggregates[hours_col] / grid_aggregates[charger_count_col].replace(0, 1)
        )
        
        # Normalize components to 0-1 scale
        normalized_scores = self._normalize_components(
            grid_aggregates['kwh_per_charger'].values,
            grid_aggregates['hours_per_charger'].values
        )
        
        # Calculate weighted demand score
        demand_scores = (
            self.kwh_weight * normalized_scores['kwh_normalized'] +
            self.hours_weight * normalized_scores['hours_normalized']
        )
        
        # Create output DataFrame
        result_df = pd.DataFrame({
            grid_id_col: grid_aggregates[grid_id_col],
            'kwh_per_charger': grid_aggregates['kwh_per_charger'],
            'hours_per_charger': grid_aggregates['hours_per_charger'],
            'kwh_normalized': normalized_scores['kwh_normalized'],
            'hours_normalized': normalized_scores['hours_normalized'],
            'demand_score': demand_scores
        })
        
        logger.info(f"✅ Calculated demand scores for {len(result_df)} grids")
        logger.info(f"   Mean demand score: {demand_scores.mean():.3f}")
        logger.info(f"   Std demand score: {demand_scores.std():.3f}")
        
        return result_df
    
    def calculate_demand_scores_from_shenzhen(self, 
                                            occupancy_file: str,
                                            volume_file: str,
                                            grid_mapping_file: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate demand scores from Shenzhen UrbanEV dataset format.
        
        Args:
            occupancy_file (str): Path to occupancy.csv file
            volume_file (str): Path to volume.csv file  
            grid_mapping_file (str, optional): Path to grid mapping file
            
        Returns:
            pd.DataFrame: DataFrame with grid_id and demand_score
        """
        logger.info("Calculating demand scores from Shenzhen UrbanEV dataset")
        
        try:
            # Load the datasets
            occupancy_df = pd.read_csv(occupancy_file)
            volume_df = pd.read_csv(volume_file)
            
            logger.info(f"Loaded occupancy data: {len(occupancy_df)} records")
            logger.info(f"Loaded volume data: {len(volume_df)} records")
            
            # Merge occupancy and volume data
            # Assuming both have station_id and time period columns
            merged_df = self._merge_shenzhen_data(occupancy_df, volume_df)
            
            # Require real grid mapping
            if not grid_mapping_file or not Path(grid_mapping_file).exists():
                raise FileNotFoundError(f"Grid mapping file required but not found: {grid_mapping_file}")
            
            grid_mapping = pd.read_csv(grid_mapping_file)
            merged_df = merged_df.merge(grid_mapping, on='station_id', how='left')
            
            # Check for stations without grid mapping
            unmapped_stations = merged_df[merged_df['grid_id'].isna()]
            if len(unmapped_stations) > 0:
                logger.warning(f"Found {len(unmapped_stations)} stations without grid mapping")
                # Remove unmapped stations rather than creating synthetic mappings
                merged_df = merged_df.dropna(subset=['grid_id'])
            
            # Calculate demand scores
            demand_scores = self._calculate_shenzhen_demand_scores(merged_df)
            
            return demand_scores
            
        except Exception as e:
            logger.error(f"Failed to calculate Shenzhen demand scores: {e}")
            raise
    
    def calculate_demand_scores_from_urban_ev(self, 
                                            data_dir: str,
                                            time_period: str = "monthly",
                                            kwh_weight: Optional[float] = None,
                                            hrs_weight: Optional[float] = None) -> pd.DataFrame:
        """
        Calculate demand scores from Shenzhen UrbanEV dataset.
        
        Based on research: UrbanEV has station-based data with:
        - inf.csv: Station information with TAZID, longitude, latitude, charge_count, area, perimeter
        - volume.csv: Time series data with stations as columns (indexed by time)
        - duration.csv: Time series duration data (indexed by time)
        - occupancy.csv: Time series occupancy data (indexed by time)
        
        Args:
            data_dir (str): Path to UrbanEV data directory
            time_period (str): Aggregation period ('daily', 'weekly', 'monthly')
            kwh_weight (Optional[float]): Override for kWh weight.
            hrs_weight (Optional[float]): Override for hours weight.
            
        Returns:
            pd.DataFrame: DataFrame with station_id and demand metrics
        """
        logger.info(f"Processing Shenzhen UrbanEV data from {data_dir}")
        
        data_path = Path(data_dir)
        
        # Load required files from UrbanEV dataset
        required_files = {
            'volume': 'volume.csv',          # Energy data (kWh) - time series with stations as columns
            'duration': 'duration.csv',      # Duration data (hours) - time series with stations as columns  
            'inf': 'inf.csv',               # Station information - TAZID, coordinates, charge_count
            'occupancy': 'occupancy.csv'     # Occupancy data - time series with stations as columns
        }
        
        data_files = {}
        for file_key, filename in required_files.items():
            file_path = data_path / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Required UrbanEV file not found: {file_path}")
            
            logger.info(f"Loading {filename}...")
            try:
                if filename == 'inf.csv':
                    # Station info file - CSV with headers: TAZID, longitude, latitude, charge_count, area, perimeter
                    data_files[file_key] = pd.read_csv(file_path)
                    logger.info(f"Loaded station info: {len(data_files[file_key])} stations")
                else:
                    # Time series data files - first column is time, rest are station IDs as columns
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    data_files[file_key] = df
                    logger.info(f"Loaded {filename}: {len(df)} time points, {len(df.columns)} stations")
                    
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
                raise
        
        logger.info("Successfully loaded all UrbanEV data files")
        
        # Process the data to calculate demand scores
        demand_scores = self._process_urban_ev_data(data_files, time_period, kwh_weight=kwh_weight, hrs_weight=hrs_weight)
        
        logger.info(f"Calculated demand scores for {len(demand_scores)} stations")
        return demand_scores
    
    def _process_urban_ev_data(self, data_files: Dict, time_period: str, kwh_weight: Optional[float] = None,
                               hrs_weight: Optional[float] = None) -> pd.DataFrame:
        """
        Process UrbanEV data files to extract demand metrics.
        
        Args:
            data_files (Dict): Dictionary of loaded data files
            time_period (str): Aggregation period
            kwh_weight (Optional[float]): Override for kWh weight.
            hrs_weight (Optional[float]): Override for hours weight.
            
        Returns:
            pd.DataFrame: Processed demand data
        """
        volume_df = data_files['volume']      # Time series: columns are station TAZIDs
        duration_df = data_files['duration']  # Time series: columns are station TAZIDs
        inf_df = data_files['inf']           # Station info: TAZID, longitude, latitude, charge_count, etc.
        occupancy_df = data_files['occupancy'] # Time series: columns are station TAZIDs
        
        # Get station information - UrbanEV uses TAZID as station identifier
        station_info = inf_df.copy()
        station_info = station_info.rename(columns={'TAZID': 'station_id'})
        
        # Handle missing or infinite values in time series data
        volume_df = volume_df.fillna(0).replace([np.inf, -np.inf], 0)
        duration_df = duration_df.fillna(0).replace([np.inf, -np.inf], 0)
        occupancy_df = occupancy_df.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Aggregate energy and duration data by time period
        if time_period == 'monthly':
            volume_agg = volume_df.resample('M').sum()
            duration_agg = duration_df.resample('M').sum()
            occupancy_agg = occupancy_df.resample('M').mean()
        elif time_period == 'weekly':
            volume_agg = volume_df.resample('W').sum()
            duration_agg = duration_df.resample('W').sum()
            occupancy_agg = occupancy_df.resample('W').mean()
        elif time_period == 'daily':
            volume_agg = volume_df.resample('D').sum()
            duration_agg = duration_df.resample('D').sum()
            occupancy_agg = occupancy_df.resample('D').mean()
        else:
            raise ValueError(f"Unsupported time period: {time_period}")
        
        # Calculate metrics per station
        station_metrics = []
        
        # Get common stations across all datasets
        common_stations = set(volume_agg.columns) & set(duration_agg.columns) & set(station_info['station_id'].astype(str))
        logger.info(f"Found {len(common_stations)} common stations across all datasets")
        
        for station_id in common_stations:
            try:
                # Convert station_id to appropriate type for lookup
                station_id_int = int(float(station_id))
                
                # Get station info
                station_row = station_info[station_info['station_id'] == station_id_int]
                if station_row.empty:
                    logger.warning(f"No info found for station {station_id}")
                    continue
                
                # Extract station details from UrbanEV format
                charge_count = station_row.iloc[0].get('charge_count', 1)  # Number of charging piles
                lat = station_row.iloc[0].get('latitude', None)
                lon = station_row.iloc[0].get('longitude', None)
                
                # Check for missing or invalid coordinates (including 0,0 which indicates missing data)
                if pd.isna(lat) or pd.isna(lon) or lat == 0 or lon == 0:
                    logger.warning(f"Missing or invalid coordinates for station {station_id}: lat={lat}, lon={lon}")
                    continue
                
                # Calculate total energy and duration over the period
                # UrbanEV volume is already in kWh, duration needs conversion
                total_kwh = volume_agg[station_id].sum()
                total_hours = duration_agg[station_id].sum()  # Already in hours based on file examination
                
                # Calculate per-charger metrics
                num_chargers = max(charge_count, 1)  # Ensure we don't divide by zero
                kwh_per_charger = total_kwh / num_chargers
                hours_per_charger = total_hours / num_chargers
                
                # Calculate average occupancy if available
                avg_occupancy = occupancy_agg[station_id].mean() if station_id in occupancy_agg.columns else 0
                
                station_metrics.append({
                    'station_id': station_id,
                    'latitude': lat,
                    'longitude': lon,
                    'num_chargers': num_chargers,
                    'total_kwh': total_kwh,
                    'total_hours': total_hours,
                    'kwh_per_charger': kwh_per_charger,
                    'hours_per_charger': hours_per_charger,
                    'avg_occupancy': avg_occupancy
                })
                
            except Exception as e:
                logger.warning(f"Error processing station {station_id}: {e}")
                continue
        
        if not station_metrics:
            raise ValueError("No valid station metrics could be calculated")
        
        # Create DataFrame
        metrics_df = pd.DataFrame(station_metrics)
        
        # Remove any rows with invalid data
        metrics_df = metrics_df.dropna(subset=['kwh_per_charger', 'hours_per_charger'])
        metrics_df = metrics_df[metrics_df['kwh_per_charger'] >= 0]
        metrics_df = metrics_df[metrics_df['hours_per_charger'] >= 0]
        
        if len(metrics_df) == 0:
            raise ValueError("No valid metrics after cleaning")
        
        # Normalize components to 0-1 scale
        normalized_scores = self._normalize_components(
            metrics_df['kwh_per_charger'].values,
            metrics_df['hours_per_charger'].values
        )
        
        # Use provided weights or fall back to instance weights
        kwh_w = self.kwh_weight if kwh_weight is None else kwh_weight
        hrs_w = self.hours_weight if hrs_weight is None else hrs_weight
        
        if kwh_weight is not None or hrs_weight is not None:
            if not np.isclose(kwh_w + hrs_w, 1.0):
                raise ValueError(f"kwh_weight and hrs_weight must sum to 1.0. Got {kwh_w} and {hrs_w}.")

        # Calculate weighted demand score
        demand_scores = (
            kwh_w * normalized_scores['kwh_normalized'] +
            hrs_w * normalized_scores['hours_normalized']
        )
        
        # Add normalized scores and demand score to DataFrame
        metrics_df['kwh_normalized'] = normalized_scores['kwh_normalized']
        metrics_df['hours_normalized'] = normalized_scores['hours_normalized']
        metrics_df['demand_score'] = demand_scores
        
        logger.info(f"Processed {len(metrics_df)} stations successfully")
        logger.info(f"Mean demand score: {demand_scores.mean():.3f}")
        logger.info(f"Demand score range: {demand_scores.min():.3f} - {demand_scores.max():.3f}")
        
        return metrics_df

    def calculate_demand_scores_from_st_evcdp(self, 
                                            data_dir: str,
                                            time_period: str = "monthly",
                                            kwh_weight: Optional[float] = None,
                                            hrs_weight: Optional[float] = None) -> pd.DataFrame:
        """
        Calculate demand scores from ST-EVCDP dataset.
        
        Based on research: ST-EVCDP has traffic zone-based data with:
        - information.csv: Zone info with num, grid, count, fast_count, slow_count, area, lon, la, CBD, dynamic_pricing
        - volume.csv: Time series data indexed by time (similar to UrbanEV)
        - duration.csv: Time series duration data indexed by time
        - occupancy.csv: Time series occupancy data indexed by time
        
        Args:
            data_dir (str): Path to ST-EVCDP data directory
            time_period (str): Aggregation period ('daily', 'weekly', 'monthly')
            kwh_weight (Optional[float]): Override for kWh weight.
            hrs_weight (Optional[float]): Override for hours weight.
            
        Returns:
            pd.DataFrame: DataFrame with zone_id and demand metrics
        """
        logger.info(f"Processing ST-EVCDP data from {data_dir}")
        
        data_path = Path(data_dir)
        
        # Load required files from ST-EVCDP dataset
        required_files = {
            'volume': 'volume.csv',          # Energy data (kWh) - time series with zones as columns
            'duration': 'duration.csv',      # Duration data - time series with zones as columns  
            'information': 'information.csv', # Zone information
            'occupancy': 'occupancy.csv'     # Occupancy data - time series with zones as columns
        }
        
        data_files = {}
        for file_key, filename in required_files.items():
            file_path = data_path / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Required ST-EVCDP file not found: {file_path}")
            
            logger.info(f"Loading {filename}...")
            try:
                if filename == 'information.csv':
                    # Zone info file - CSV with headers: num, grid, count, fast_count, slow_count, area, lon, la, CBD, dynamic_pricing
                    data_files[file_key] = pd.read_csv(file_path)
                    logger.info(f"Loaded zone info: {len(data_files[file_key])} zones")
                else:
                    # Time series data files
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    data_files[file_key] = df
                    logger.info(f"Loaded {filename}: {len(df)} time points, {len(df.columns)} zones")
                    
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
                raise
        
        logger.info("Successfully loaded all ST-EVCDP data files")
        
        # Process the data to calculate demand scores
        demand_scores = self._process_st_evcdp_data(data_files, time_period, kwh_weight=kwh_weight, hrs_weight=hrs_weight)
        
        logger.info(f"Calculated demand scores for {len(demand_scores)} zones")
        return demand_scores
        
    def _process_st_evcdp_data(self, data_files: Dict, time_period: str,
                               kwh_weight: Optional[float] = None,
                               hrs_weight: Optional[float] = None) -> pd.DataFrame:
        """
        Process ST-EVCDP data files to extract demand metrics.
        
        Args:
            data_files (Dict): Dictionary of loaded data files
            time_period (str): Aggregation period
            kwh_weight (Optional[float]): Override for kWh weight.
            hrs_weight (Optional[float]): Override for hours weight.
            
        Returns:
            pd.DataFrame: Processed demand data
        """
        volume_df = data_files['volume']        # Time series: columns are zone IDs
        duration_df = data_files['duration']    # Time series: columns are zone IDs
        info_df = data_files['information']     # Zone info: num, grid, count, coordinates, etc.
        occupancy_df = data_files['occupancy']  # Time series: columns are zone IDs
        
        # Get zone information - ST-EVCDP uses 'grid' as zone identifier
        zone_info = info_df.copy()
        zone_info = zone_info.rename(columns={'grid': 'zone_id', 'la': 'latitude', 'lon': 'longitude'})
        
        # Handle missing or infinite values in time series data
        volume_df = volume_df.fillna(0).replace([np.inf, -np.inf], 0)
        duration_df = duration_df.fillna(0).replace([np.inf, -np.inf], 0)
        occupancy_df = occupancy_df.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Aggregate energy and duration data by time period
        if time_period == 'monthly':
            volume_agg = volume_df.resample('M').sum()
            duration_agg = duration_df.resample('M').sum()
            occupancy_agg = occupancy_df.resample('M').mean()
        elif time_period == 'weekly':
            volume_agg = volume_df.resample('W').sum()
            duration_agg = duration_df.resample('W').sum()
            occupancy_agg = occupancy_df.resample('W').mean()
        elif time_period == 'daily':
            volume_agg = volume_df.resample('D').sum()
            duration_agg = duration_df.resample('D').sum()
            occupancy_agg = occupancy_df.resample('D').mean()
        else:
            raise ValueError(f"Unsupported time period: {time_period}")
        
        # Calculate metrics per zone
        zone_metrics = []
        
        # Get common zones across all datasets
        common_zones = set(volume_agg.columns) & set(duration_agg.columns) & set(zone_info['zone_id'].astype(str))
        logger.info(f"Found {len(common_zones)} common zones across all datasets")
        
        for zone_id in common_zones:
            try:
                # Convert zone_id to appropriate type for lookup
                zone_id_int = int(float(zone_id))
                
                # Get zone info
                zone_row = zone_info[zone_info['zone_id'] == zone_id_int]
                if zone_row.empty:
                    logger.warning(f"No info found for zone {zone_id}")
                    continue
                
                # Extract zone details from ST-EVCDP format
                total_count = zone_row.iloc[0].get('count', 1)  # Total charging piles
                fast_count = zone_row.iloc[0].get('fast_count', 0)  # Fast charging piles
                slow_count = zone_row.iloc[0].get('slow_count', 0)  # Slow charging piles
                lat = zone_row.iloc[0].get('latitude', None)
                lon = zone_row.iloc[0].get('longitude', None)
                
                if pd.isna(lat) or pd.isna(lon):
                    logger.warning(f"Missing coordinates for zone {zone_id}")
                    continue
                
                # Calculate total energy and duration over the period
                total_kwh = volume_agg[zone_id].sum()
                total_hours = duration_agg[zone_id].sum()
                
                # Calculate per-charger metrics
                num_chargers = max(total_count, 1)  # Ensure we don't divide by zero
                kwh_per_charger = total_kwh / num_chargers
                hours_per_charger = total_hours / num_chargers
                
                # Calculate average occupancy if available
                avg_occupancy = occupancy_agg[zone_id].mean() if zone_id in occupancy_agg.columns else 0
                
                zone_metrics.append({
                    'zone_id': zone_id,
                    'latitude': lat,
                    'longitude': lon,
                    'num_chargers': num_chargers,
                    'fast_chargers': fast_count,
                    'slow_chargers': slow_count,
                    'total_kwh': total_kwh,
                    'total_hours': total_hours,
                    'kwh_per_charger': kwh_per_charger,
                    'hours_per_charger': hours_per_charger,
                    'avg_occupancy': avg_occupancy
                })
                
            except Exception as e:
                logger.warning(f"Error processing zone {zone_id}: {e}")
                continue
        
        if not zone_metrics:
            raise ValueError("No valid zone metrics could be calculated")
        
        # Create DataFrame
        metrics_df = pd.DataFrame(zone_metrics)
        
        # Remove any rows with invalid data
        metrics_df = metrics_df.dropna(subset=['kwh_per_charger', 'hours_per_charger'])
        metrics_df = metrics_df[metrics_df['kwh_per_charger'] >= 0]
        metrics_df = metrics_df[metrics_df['hours_per_charger'] >= 0]
        
        if len(metrics_df) == 0:
            raise ValueError("No valid metrics after cleaning")
        
        # Normalize components to 0-1 scale
        normalized_scores = self._normalize_components(
            metrics_df['kwh_per_charger'].values,
            metrics_df['hours_per_charger'].values
        )
        
        # Calculate weighted demand score
        kwh_w = self.kwh_weight if kwh_weight is None else kwh_weight
        hrs_w = self.hours_weight if hrs_weight is None else hrs_weight
        
        if kwh_weight is not None or hrs_weight is not None:
            if not np.isclose(kwh_w + hrs_w, 1.0):
                raise ValueError(f"kwh_weight and hrs_weight must sum to 1.0. Got {kwh_w} and {hrs_w}.")

        demand_scores = (
            kwh_w * normalized_scores['kwh_normalized'] +
            hrs_w * normalized_scores['hours_normalized']
        )
        
        # Add normalized scores and demand score to DataFrame
        metrics_df['kwh_normalized'] = normalized_scores['kwh_normalized']
        metrics_df['hours_normalized'] = normalized_scores['hours_normalized']
        metrics_df['demand_score'] = demand_scores
        
        logger.info(f"Processed {len(metrics_df)} zones successfully")
        logger.info(f"Mean demand score: {demand_scores.mean():.3f}")
        logger.info(f"Demand score range: {demand_scores.min():.3f} - {demand_scores.max():.3f}")
        
        return metrics_df
    
    def _merge_shenzhen_data(self, occupancy_df: pd.DataFrame, volume_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge Shenzhen occupancy and volume data.
        
        Args:
            occupancy_df: Occupancy data (hours connected)
            volume_df: Volume data (kWh dispensed)
            
        Returns:
            pd.DataFrame: Merged data
        """
        # This is a simplified merge - actual implementation would depend on
        # the exact structure of the Shenzhen datasets
        
        # Assuming both datasets have similar structure with station identifiers
        occupancy_agg = occupancy_df.groupby('station_id').agg({
            'duration_hours': 'sum',  # Total connection time
            'sessions': 'sum'         # Total charging sessions
        }).reset_index()
        
        volume_agg = volume_df.groupby('station_id').agg({
            'kwh_dispensed': 'sum',   # Total energy dispensed
            'num_chargers': 'first'   # Number of chargers per station
        }).reset_index()
        
        # Merge the aggregated data
        merged_df = occupancy_agg.merge(volume_agg, on='station_id', how='inner')
        
        logger.info(f"Merged data for {len(merged_df)} stations")
        return merged_df
    

    
    def _calculate_shenzhen_demand_scores(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate demand scores from merged Shenzhen data.
        
        Args:
            merged_df: Merged occupancy and volume data
            
        Returns:
            pd.DataFrame: Demand scores by grid
        """
        # Group by grid and aggregate
        grid_data = merged_df.groupby('grid_id').agg({
            'kwh_dispensed': 'sum',
            'duration_hours': 'sum',
            'num_chargers': 'sum'
        }).reset_index()
        
        # Calculate per-charger metrics
        grid_data['kwh_per_charger'] = (
            grid_data['kwh_dispensed'] / grid_data['num_chargers'].replace(0, 1)
        )
        grid_data['hours_per_charger'] = (
            grid_data['duration_hours'] / grid_data['num_chargers'].replace(0, 1)
        )
        
        # Normalize and calculate demand scores
        normalized_scores = self._normalize_components(
            grid_data['kwh_per_charger'].values,
            grid_data['hours_per_charger'].values
        )
        
        demand_scores = (
            self.kwh_weight * normalized_scores['kwh_normalized'] +
            self.hours_weight * normalized_scores['hours_normalized']
        )
        
        result_df = pd.DataFrame({
            'grid_id': grid_data['grid_id'],
            'kwh_per_charger': grid_data['kwh_per_charger'],
            'hours_per_charger': grid_data['hours_per_charger'],
            'kwh_normalized': normalized_scores['kwh_normalized'],
            'hours_normalized': normalized_scores['hours_normalized'],
            'demand_score': demand_scores
        })
        
        return result_df
    
    def _normalize_components(self, 
                            kwh_values: np.ndarray, 
                            hours_values: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Normalize kWh and hours components to 0-1 scale using min-max normalization.
        
        Args:
            kwh_values: Array of kWh per charger values
            hours_values: Array of hours per charger values
            
        Returns:
            Dict containing normalized arrays
        """
        # Min-max normalization with small epsilon to handle edge cases
        eps = 1e-8
        
        kwh_min, kwh_max = kwh_values.min(), kwh_values.max()
        kwh_normalized = (kwh_values - kwh_min) / max(kwh_max - kwh_min, eps)
        
        hours_min, hours_max = hours_values.min(), hours_values.max()
        hours_normalized = (hours_values - hours_min) / max(hours_max - hours_min, eps)
        
        # Ensure values are in [0, 1] range
        kwh_normalized = np.clip(kwh_normalized, 0, 1)
        hours_normalized = np.clip(hours_normalized, 0, 1)
        
        logger.debug(f"kWh normalization: [{kwh_min:.2f}, {kwh_max:.2f}] -> [0, 1]")
        logger.debug(f"Hours normalization: [{hours_min:.2f}, {hours_max:.2f}] -> [0, 1]")
        
        return {
            'kwh_normalized': kwh_normalized,
            'hours_normalized': hours_normalized,
            'kwh_min': kwh_min,
            'kwh_max': kwh_max,
            'hours_min': hours_min,
            'hours_max': hours_max
        }
    
    def save_demand_scores(self, 
                          demand_scores: pd.DataFrame, 
                          output_path: str,
                          include_metadata: bool = True) -> None:
        """
        Save demand scores to CSV file.
        
        Args:
            demand_scores: DataFrame with demand scores
            output_path: Output file path
            include_metadata: Whether to include calculation metadata
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save main data
        demand_scores.to_csv(output_file, index=False)
        
        # Save metadata if requested
        if include_metadata:
            metadata = {
                'kwh_weight': self.kwh_weight,
                'hours_weight': self.hours_weight,
                'n_grids': len(demand_scores),
                'mean_demand_score': demand_scores['demand_score'].mean(),
                'std_demand_score': demand_scores['demand_score'].std(),
                'min_demand_score': demand_scores['demand_score'].min(),
                'max_demand_score': demand_scores['demand_score'].max()
            }
            
            metadata_file = output_file.with_suffix('.metadata.json')
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved metadata to {metadata_file}")
        
        logger.info(f"💾 Saved demand scores to {output_file}")
    
    def load_demand_scores(self, input_path: str) -> pd.DataFrame:
        """
        Load previously calculated demand scores.
        
        Args:
            input_path: Input file path
            
        Returns:
            pd.DataFrame: Loaded demand scores
        """
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Demand scores file not found: {input_file}")
        
        demand_scores = pd.read_csv(input_file)
        logger.info(f"📂 Loaded demand scores from {input_file}")
        
        return demand_scores
    
    def validate_demand_scores(self, demand_scores: pd.DataFrame) -> Dict[str, float]:
        """
        Validate calculated demand scores and return quality metrics.
        
        Args:
            demand_scores: DataFrame with demand scores
            
        Returns:
            Dict: Validation metrics
        """
        if 'demand_score' not in demand_scores.columns:
            raise ValueError("demand_score column not found")
        
        scores = demand_scores['demand_score']
        
        # Check for issues
        n_nan = scores.isna().sum()
        n_negative = (scores < 0).sum()
        n_above_one = (scores > 1).sum()
        
        # Calculate distribution metrics
        metrics = {
            'n_grids': len(scores),
            'n_nan': int(n_nan),
            'n_negative': int(n_negative), 
            'n_above_one': int(n_above_one),
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'min': float(scores.min()),
            'max': float(scores.max()),
            'q25': float(scores.quantile(0.25)),
            'q50': float(scores.quantile(0.50)),
            'q75': float(scores.quantile(0.75))
        }
        
        # Log validation results
        logger.info("📊 Demand Score Validation Results:")
        logger.info(f"  • Total grids: {metrics['n_grids']}")
        logger.info(f"  • Mean score: {metrics['mean']:.3f}")
        logger.info(f"  • Std score: {metrics['std']:.3f}")
        logger.info(f"  • Range: [{metrics['min']:.3f}, {metrics['max']:.3f}]")
        
        if n_nan > 0:
            logger.warning(f"  ⚠️ Found {n_nan} NaN values")
        if n_negative > 0:
            logger.warning(f"  ⚠️ Found {n_negative} negative values")
        if n_above_one > 0:
            logger.warning(f"  ⚠️ Found {n_above_one} values above 1.0")
        
        return metrics 
    
    def _map_stations_to_grids(self, station_info: pd.DataFrame, grid_cells: List[Dict]) -> Dict[str, str]:
        """
        Maps station coordinates to grid cells using an efficient spatial join.

        Args:
            station_info: DataFrame with station information, including latitude and longitude.
            grid_cells: A list of grid cell dictionaries.

        Returns:
            A dictionary mapping station_id to grid_id.
        """
        if not grid_cells:
            self.logger.warning("No grid cells provided for station mapping.")
            return {}

        # Create a GeoDataFrame for the grid cells
        grid_df = pd.DataFrame(grid_cells)
        if 'corners' not in grid_df.columns:
            self.logger.error("Grid cells are missing 'corners' data for geometry creation.")
            return {}
            
        grid_gdf = gpd.GeoDataFrame(
            grid_df,
            geometry=grid_df['corners'].apply(Polygon),
            crs="EPSG:4326"
        )

        # Create a GeoDataFrame for the stations
        stations_with_coords = station_info.dropna(subset=['latitude', 'longitude']).copy()
        if stations_with_coords.empty:
            self.logger.warning("No stations with valid coordinates found for mapping.")
            return {}
            
        stations_gdf = gpd.GeoDataFrame(
            stations_with_coords,
            geometry=gpd.points_from_xy(stations_with_coords.longitude, stations_with_coords.latitude),
            crs="EPSG:4326"
        )

        # Perform the spatial join
        try:
            # The op='within' predicate is generally more accurate for point-in-polygon tests
            joined_gdf = gpd.sjoin(stations_gdf, grid_gdf, how="inner", predicate="within")
            
            # Create the mapping dictionary
            # Ensure station_id is string type to match the demand_scores DataFrame
            station_grid_mapping = pd.Series(
                joined_gdf.grid_id.values,
                index=joined_gdf.station_id.astype(str)
            ).to_dict()
            
            self.logger.info(f"Successfully mapped {len(station_grid_mapping)} stations to grid cells.")
            return station_grid_mapping
            
        except Exception as e:
            self.logger.error(f"An error occurred during spatial join for station mapping: {e}")
            return {}

    def aggregate_to_grids(self, 
                       demand_scores: pd.DataFrame, 
                       station_info: pd.DataFrame, 
                       grid_cells: List[Dict],
                       include_zero_demand_grids: bool = False) -> pd.DataFrame:
        """
        Aggregates station-level demand scores to a grid level.
        
        Args:
            demand_scores (pd.DataFrame): Station-level demand scores
            station_info (pd.DataFrame): Station information with coordinates
            grid_cells (List[Dict]): List of grid cell dictionaries
            
        Returns:
            pd.DataFrame: Grid-level aggregated demand scores
        """
        logger.info("Aggregating station demand scores to grid level")

        # Step 1: Map stations to grid cells
        station_grid_mapping = self._map_stations_to_grids(station_info, grid_cells)

        # Step 2: Merge grid mapping with demand scores
        station_scores_with_grid = demand_scores.copy()
        # Ensure station_id is string type to match the mapping keys
        station_scores_with_grid['grid_id'] = station_scores_with_grid['station_id'].astype(str).map(station_grid_mapping)
        
        # Log successful mapping
        mapped_count = station_scores_with_grid['grid_id'].notna().sum()
        logger.debug(f"Mapped {mapped_count} stations to grids")
        
        # Remove stations without grid mapping
        valid_stations = station_scores_with_grid.dropna(subset=['grid_id'])
        
        if len(valid_stations) == 0:
            logger.error(f"No stations mapped! Total stations: {len(station_scores_with_grid)}, Mapping size: {len(station_grid_mapping)}")
            raise ValueError("No stations could be mapped to grids")
        
        logger.info(f"Mapped {len(valid_stations)} out of {len(station_scores_with_grid)} stations to grids")
        
        # Step 3: Aggregate by grid for grids with stations
        # Aggregate by grid
        grid_aggregates = valid_stations.groupby('grid_id').agg({
            'demand_score': 'mean',
            'latitude': 'mean',
            'longitude': 'mean',
            'num_chargers': 'sum',
            'total_kwh': 'sum',
            'total_hours': 'sum',
            'kwh_per_charger': 'mean',  # Average across stations in grid
            'hours_per_charger': 'mean',
            'avg_occupancy': 'mean',
            'station_id': 'count'  # Count of stations per grid
        }).reset_index()
        
        # Rename station count column
        grid_aggregates = grid_aggregates.rename(columns={'station_id': 'num_stations'})
        
        # Recalculate grid-level per-charger metrics
        grid_aggregates['grid_kwh_per_charger'] = (
            grid_aggregates['total_kwh'] / grid_aggregates['num_chargers']
        )
        grid_aggregates['grid_hours_per_charger'] = (
            grid_aggregates['total_hours'] / grid_aggregates['num_chargers']
        )
        
        # Normalize grid-level components
        grid_normalized = self._normalize_components(
            grid_aggregates['grid_kwh_per_charger'].values,
            grid_aggregates['grid_hours_per_charger'].values
        )
        
        # Calculate grid-level demand score
        grid_demand_scores = (
            self.kwh_weight * grid_normalized['kwh_normalized'] +
            self.hours_weight * grid_normalized['hours_normalized']
        )
        
        # Create final grid DataFrame
        grid_demand_df = pd.DataFrame({
            'grid_id': grid_aggregates['grid_id'],
            'num_stations': grid_aggregates['num_stations'],
            'num_chargers': grid_aggregates['num_chargers'],
            'total_kwh': grid_aggregates['total_kwh'],
            'total_hours': grid_aggregates['total_hours'],
            'kwh_per_charger': grid_aggregates['grid_kwh_per_charger'],
            'hours_per_charger': grid_aggregates['grid_hours_per_charger'],
            'kwh_normalized': grid_normalized['kwh_normalized'],
            'hours_normalized': grid_normalized['hours_normalized'],
            'demand_score': grid_demand_scores,
            'avg_occupancy': grid_aggregates['avg_occupancy']
        })
        
        # Step 4: Create synthetic demand for all grids using spatial interpolation
        if include_zero_demand_grids:
            all_grids_df = self._create_synthetic_demand_for_all_grids(
                grid_demand_df, grid_cells, valid_stations
            )
            logger.info(f"✅ Aggregated demand scores for {len(all_grids_df)} grids (including synthetic)")
            logger.info(f"   Grid demand score range: {all_grids_df['demand_score'].min():.3f} - {all_grids_df['demand_score'].max():.3f}")
            return all_grids_df
        else:
            logger.info(f"✅ Aggregated demand scores for {len(grid_demand_df)} grids")
            logger.info(f"   Grid demand score range: {grid_demand_scores.min():.3f} - {grid_demand_scores.max():.3f}")
            return grid_demand_df
    
    def _create_synthetic_demand_for_all_grids(self, 
                                             actual_demand_df: pd.DataFrame,
                                             all_grid_cells: List[Dict],
                                             station_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create synthetic demand scores for all grid cells using spatial interpolation
        and distance-based decay from real charging stations.
        
        Args:
            actual_demand_df: DataFrame with actual demand scores for grids with stations
            all_grid_cells: List of all grid cell dictionaries
            station_data: DataFrame with station locations and demand scores
            
        Returns:
            DataFrame with demand scores for all grid cells
        """
        logger.info("🔮 Creating synthetic demand scores for all grid cells...")
        
        # Create DataFrame for all grids
        all_grids_data = []
        
        for cell in all_grid_cells:
            grid_id = cell['grid_id']
            
            # Get grid center coordinates
            center_lat = cell.get('center_lat', (cell.get('min_lat', 0) + cell.get('max_lat', 0)) / 2)
            center_lon = cell.get('center_lon', (cell.get('min_lon', 0) + cell.get('max_lon', 0)) / 2)
            grid_scale = cell.get('grid_scale', 1.0)
            
            # Check if this grid already has actual demand data
            if grid_id in actual_demand_df['grid_id'].values:
                # Use actual demand data
                actual_row = actual_demand_df[actual_demand_df['grid_id'] == grid_id].iloc[0]
                demand_score = actual_row['demand_score']
                is_synthetic = False
            else:
                # Generate synthetic demand based on nearby stations
                demand_score = self._calculate_synthetic_demand(
                    center_lat, center_lon, grid_scale, station_data
                )
                is_synthetic = True
            
            all_grids_data.append({
                'grid_id': grid_id,
                'demand_score': demand_score,
                'is_synthetic': is_synthetic,
                'num_stations': 0 if is_synthetic else actual_demand_df[actual_demand_df['grid_id'] == grid_id]['num_stations'].iloc[0],
                'num_chargers': 0 if is_synthetic else actual_demand_df[actual_demand_df['grid_id'] == grid_id]['num_chargers'].iloc[0],
                'total_kwh': 0.0 if is_synthetic else actual_demand_df[actual_demand_df['grid_id'] == grid_id]['total_kwh'].iloc[0],
                'total_hours': 0.0 if is_synthetic else actual_demand_df[actual_demand_df['grid_id'] == grid_id]['total_hours'].iloc[0],
                'kwh_per_charger': 0.0 if is_synthetic else actual_demand_df[actual_demand_df['grid_id'] == grid_id]['kwh_per_charger'].iloc[0],
                'hours_per_charger': 0.0 if is_synthetic else actual_demand_df[actual_demand_df['grid_id'] == grid_id]['hours_per_charger'].iloc[0],
                'kwh_normalized': demand_score * 0.7 if is_synthetic else actual_demand_df[actual_demand_df['grid_id'] == grid_id]['kwh_normalized'].iloc[0],
                'hours_normalized': demand_score * 0.3 if is_synthetic else actual_demand_df[actual_demand_df['grid_id'] == grid_id]['hours_normalized'].iloc[0],
                'avg_occupancy': demand_score * 0.5 if is_synthetic else actual_demand_df[actual_demand_df['grid_id'] == grid_id]['avg_occupancy'].iloc[0]
            })
        
        synthetic_df = pd.DataFrame(all_grids_data)
        
        # Log statistics
        real_count = (~synthetic_df['is_synthetic']).sum()
        synthetic_count = synthetic_df['is_synthetic'].sum()
        
        logger.info(f"   📊 Created {len(synthetic_df)} total grid records:")
        logger.info(f"      • Real demand: {real_count} grids")
        logger.info(f"      • Synthetic demand: {synthetic_count} grids")
        logger.info(f"      • Synthetic demand range: {synthetic_df[synthetic_df['is_synthetic']]['demand_score'].min():.3f} - {synthetic_df[synthetic_df['is_synthetic']]['demand_score'].max():.3f}")
        
        return synthetic_df
    
    def _calculate_synthetic_demand(self, 
                                  grid_lat: float, 
                                  grid_lon: float, 
                                  grid_scale: float,
                                  station_data: pd.DataFrame) -> float:
        """
        Calculate synthetic demand for a grid cell based on nearby charging stations
        using distance-weighted interpolation.
        
        Args:
            grid_lat: Grid cell center latitude
            grid_lon: Grid cell center longitude  
            grid_scale: Grid scale factor (affects influence radius)
            station_data: DataFrame with station locations and demand scores
            
        Returns:
            Synthetic demand score (0.0 to 1.0)
        """
        if station_data.empty:
            return 0.0
        
        # Calculate distances to all stations (in km using Haversine formula)
        distances = []
        weights = []
        
        for _, station in station_data.iterrows():
            station_lat = station['latitude']
            station_lon = station['longitude']
            station_demand = station['demand_score']
            
            # Haversine distance formula
            lat_diff = np.radians(station_lat - grid_lat)
            lon_diff = np.radians(station_lon - grid_lon)
            a = (np.sin(lat_diff/2)**2 + 
                 np.cos(np.radians(grid_lat)) * np.cos(np.radians(station_lat)) * 
                 np.sin(lon_diff/2)**2)
            distance_km = 2 * 6371 * np.arcsin(np.sqrt(a))  # 6371 km = Earth radius
            
            # Scale-aware influence radius (larger grids have wider influence)
            max_influence_radius = grid_scale * 5.0  # 5km base radius, scaled by grid size
            
            if distance_km <= max_influence_radius:
                # Distance-based weight with exponential decay
                weight = np.exp(-distance_km / (grid_scale * 2.0)) * station_demand
                distances.append(distance_km)
                weights.append(weight)
        
        if not weights:
            return 0.0
        
        # Calculate weighted average demand with distance decay
        synthetic_demand = np.sum(weights) / len(weights) if weights else 0.0
        
        # Apply scale-based dampening (smaller grids get more precise, larger grids get smoothed)
        scale_factor = 1.0 / (1.0 + grid_scale * 0.2)  # Dampen larger scales
        synthetic_demand *= scale_factor
        
        # Ensure reasonable bounds
        synthetic_demand = np.clip(synthetic_demand, 0.0, 1.0)
        
        return synthetic_demand
    
    def interpolate_demand_to_all_grids(self, 
                                      actual_demand_df: pd.DataFrame,
                                      all_grid_cells: List[Dict],
                                      station_data: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate demand scores for ALL grid cells using spatial interpolation
        from existing charging stations.
        
        This creates a comprehensive dataset by:
        1. Using actual demand for grids with stations
        2. Interpolating demand for grids without stations using spatial patterns
        3. Creating realistic demand gradients across the urban area
        
        Args:
            actual_demand_df: DataFrame with actual demand scores for grids with stations
            all_grid_cells: List of all grid cell dictionaries
            station_data: DataFrame with station locations and demand scores
            
        Returns:
            DataFrame with demand scores for ALL grid cells
        """
        from scipy.spatial.distance import cdist
        import numpy as np
        
        logger.info("🔮 Creating comprehensive demand dataset using spatial interpolation...")
        
        if len(actual_demand_df) < 3:
            logger.warning("Need at least 3 stations for meaningful interpolation")
            return actual_demand_df
        
        # Handle merged column names (they might have _x, _y suffixes)
        lat_col = 'latitude'
        lon_col = 'longitude'
        
        if 'latitude' not in station_data.columns:
            if 'latitude_x' in station_data.columns:
                lat_col = 'latitude_x'
            elif 'latitude_y' in station_data.columns:
                lat_col = 'latitude_y'
            else:
                logger.error("No latitude column found in station data")
                return actual_demand_df
                
        if 'longitude' not in station_data.columns:
            if 'longitude_x' in station_data.columns:
                lon_col = 'longitude_x'
            elif 'longitude_y' in station_data.columns:
                lon_col = 'longitude_y' 
            else:
                logger.error("No longitude column found in station data")
                return actual_demand_df
        
        logger.info(f"Using coordinates: {lat_col}, {lon_col}")
        
        # Prepare station coordinates and demand values for interpolation
        station_coords = station_data[[lat_col, lon_col]].values
        station_demands = station_data['demand_score'].values
        
        logger.info(f"Using {len(station_data)} stations for spatial interpolation")
        
        # Try to create RBF interpolator for smooth spatial interpolation
        rbf_interpolator = None
        try:
            from scipy.interpolate import Rbf
            # Use multiquadric RBF for smooth interpolation
            rbf_interpolator = Rbf(
                station_coords[:, 0], station_coords[:, 1], station_demands,
                function='multiquadric', smooth=0.1
            )
            logger.info("✅ RBF interpolator created successfully")
        except Exception as e:
            logger.warning(f"RBF interpolation not available: {e}. Using distance-based interpolation.")
        
        # Process all grid cells
        all_grid_data = []
        grids_with_stations = set(actual_demand_df['grid_id'])
        
        for cell in all_grid_cells:
            grid_id = cell['grid_id']
            
            # Get grid center coordinates
            center_lat = cell.get('center_lat', (cell.get('min_lat', 0) + cell.get('max_lat', 0)) / 2)
            center_lon = cell.get('center_lon', (cell.get('min_lon', 0) + cell.get('max_lon', 0)) / 2)
            grid_scale = cell.get('grid_scale', 1.0)
            
            if grid_id in grids_with_stations:
                # Use actual demand data
                actual_row = actual_demand_df[actual_demand_df['grid_id'] == grid_id].iloc[0]
                demand_score = actual_row['demand_score']
                is_interpolated = False
                
                # Copy all actual data fields
                grid_data = actual_row.to_dict()
                grid_data['is_interpolated'] = is_interpolated
                
            else:
                # Interpolate demand for this location
                if rbf_interpolator is not None:
                    try:
                        # Use RBF interpolation
                        interpolated_demand = rbf_interpolator(center_lat, center_lon)
                        # Ensure positive values and reasonable bounds
                        interpolated_demand = max(0.0, min(1.0, float(interpolated_demand)))
                    except:
                        # Fallback to distance-based interpolation
                        interpolated_demand = self._distance_based_interpolation(
                            center_lat, center_lon, station_coords, station_demands, grid_scale
                        )
                else:
                    # Use distance-based interpolation
                    interpolated_demand = self._distance_based_interpolation(
                        center_lat, center_lon, station_coords, station_demands, grid_scale
                    )
                
                # Create synthetic grid data
                grid_data = {
                    'grid_id': grid_id,
                    'demand_score': interpolated_demand,
                    'is_interpolated': True,
                    'grid_scale': grid_scale,
                    'num_stations': 0,
                    'num_chargers': 0,
                    'total_kwh': 0,
                    'total_hours': 0,
                    'kwh_per_charger': 0,
                    'hours_per_charger': 0,
                    'kwh_normalized': interpolated_demand,
                    'hours_normalized': interpolated_demand,
                    'avg_occupancy': interpolated_demand * 0.6  # Estimated occupancy
                }
            
            all_grid_data.append(grid_data)
        
        # Create final DataFrame
        all_grids_df = pd.DataFrame(all_grid_data)
        
        # Calculate statistics
        actual_count = len(all_grids_df[~all_grids_df['is_interpolated']])
        interpolated_count = len(all_grids_df[all_grids_df['is_interpolated']])
        
        # Quality statistics
        actual_demand_range = all_grids_df[~all_grids_df['is_interpolated']]['demand_score']
        interpolated_demand_range = all_grids_df[all_grids_df['is_interpolated']]['demand_score']
        
        logger.info(f"✅ Created comprehensive demand dataset:")
        logger.info(f"   📊 {actual_count} grids with actual station data")
        logger.info(f"   🔮 {interpolated_count} grids with interpolated demand")
        logger.info(f"   📈 Total expansion: {(interpolated_count + actual_count) / actual_count:.1f}x")
        
        if len(actual_demand_range) > 0:
            logger.info(f"   📊 Actual demand range: {actual_demand_range.min():.3f} - {actual_demand_range.max():.3f}")
        if len(interpolated_demand_range) > 0:
            logger.info(f"   🔮 Interpolated demand range: {interpolated_demand_range.min():.3f} - {interpolated_demand_range.max():.3f}")
        
        return all_grids_df
    
    def _distance_based_interpolation(self, lat: float, lon: float, 
                                    station_coords: np.ndarray, station_demands: np.ndarray, 
                                    grid_scale: float) -> float:
        """
        Perform distance-based interpolation for a single point.
        """
        import numpy as np
        from scipy.spatial.distance import cdist
        
        grid_coord = np.array([[lat, lon]])
        
        # Calculate distances in km
        distances = cdist(grid_coord, station_coords, metric='euclidean')[0] * 111.32
        
        # Dynamic influence radius based on grid scale and station density
        base_radius = 3.0 * grid_scale  # Larger influence radius for interpolation
        
        # Find nearby stations
        nearby_mask = distances <= base_radius
        
        if not np.any(nearby_mask):
            # If no stations within base radius, use the 3 closest stations
            closest_indices = np.argsort(distances)[:3]
            nearby_distances = distances[closest_indices]
            nearby_demands = station_demands[closest_indices]
        else:
            nearby_distances = distances[nearby_mask]
            nearby_demands = station_demands[nearby_mask]
        
        # Inverse distance weighting with smooth decay
        weights = 1.0 / (nearby_distances + 0.1) ** 1.5  # Smooth distance decay
        weights = weights / np.sum(weights)  # Normalize
        
        # Calculate weighted average
        interpolated_demand = np.sum(nearby_demands * weights)
        
        # Apply distance decay (farther from stations = lower demand)
        min_distance = np.min(nearby_distances)
        distance_factor = np.exp(-min_distance / (2.0 * grid_scale))  # Exponential decay
        
        interpolated_demand *= distance_factor
        
        # Add some spatial variation
        variation = np.random.normal(1.0, 0.1 * grid_scale)
        interpolated_demand *= max(0.1, variation)
        
        return max(0.0, min(1.0, interpolated_demand))
    
    def interpolate_all_demand_variations_at_once(self,
                                                all_demand_variations: Dict[str, pd.DataFrame],
                                                grid_cells: List[Dict],
                                                station_data_with_demand: pd.DataFrame) -> pd.DataFrame:
        """
        OPTIMIZED: Interpolate ALL demand variations simultaneously to avoid redundant spatial calculations.
        
        This method calculates spatial distances and interpolation weights ONCE and applies them
        to all demand variations, dramatically reducing computation time for multiscale grids.
        
        Args:
            all_demand_variations: Dict mapping variation names to demand DataFrames
            grid_cells: List of all grid cell dictionaries
            station_data_with_demand: Station data merged with all demand variations
            
        Returns:
            DataFrame with all demand score variations for all grid cells
        """
        from scipy.spatial.distance import cdist
        import numpy as np
        
        logger.info(f"🚀 OPTIMIZED: Interpolating {len(all_demand_variations)} demand variations for {len(grid_cells)} grids simultaneously")
        
        if len(station_data_with_demand) < 3:
            logger.warning("Need at least 3 stations for meaningful interpolation")
            # Return simple fallback: zero demand scores for all grids
            logger.info("Using fallback: assigning zero demand scores to all grids")
            
            # Create basic grid structure
            grid_metadata = []
            for cell in grid_cells:
                grid_metadata.append({
                    'grid_id': cell['grid_id'],
                    'latitude': cell.get('center_lat', (cell.get('min_lat', 0) + cell.get('max_lat', 0)) / 2),
                    'longitude': cell.get('center_lon', (cell.get('min_lon', 0) + cell.get('max_lon', 0)) / 2),
                    'grid_type': cell.get('grid_type', 'unknown'),
                    'grid_scale': cell.get('grid_scale', 1.0),
                    'grid_variation': cell.get('grid_variation', 0)
                })
            
            fallback_result = pd.DataFrame(grid_metadata)
            
            # Add zero demand scores for all variations
            for variation_name in all_demand_variations.keys():
                fallback_result[f'demand_score_{variation_name}'] = 0.0
            
            logger.info(f"✅ Fallback complete: {len(fallback_result)} grids with zero demand scores")
            return fallback_result
        
        # Extract station locations and all demand values
        station_coords = station_data_with_demand[['latitude', 'longitude']].values
        
        # Extract all grid coordinates
        grid_coords = []
        grid_ids = []
        grid_metadata = []
        
        for cell in grid_cells:
            grid_id = cell['grid_id']
            # Get center coordinates
            center_lat = cell.get('center_lat', (cell.get('min_lat', 0) + cell.get('max_lat', 0)) / 2)
            center_lon = cell.get('center_lon', (cell.get('min_lon', 0) + cell.get('max_lon', 0)) / 2)
            
            grid_coords.append([center_lat, center_lon])
            grid_ids.append(grid_id)
            grid_metadata.append({
                'grid_id': grid_id,
                'latitude': center_lat,
                'longitude': center_lon,
                'grid_type': cell.get('grid_type', 'unknown'),
                'grid_scale': cell.get('grid_scale', 1.0),
                'grid_variation': cell.get('grid_variation', 0)
            })
        
        grid_coords = np.array(grid_coords)
        
        # OPTIMIZATION: Calculate distances once for all variations
        logger.info("📐 Calculating spatial distances (once for all variations)...")
        distances = cdist(grid_coords, station_coords, metric='euclidean')
        
        # Use inverse distance weighting with distance decay
        epsilon = 1e-10  # Avoid division by zero
        weights = 1.0 / (distances + epsilon)
        
        # Apply distance decay (closer stations have exponentially more influence)
        decay_factor = 2.0
        weights = np.power(weights, decay_factor)
        
        # Normalize weights so each grid's weights sum to 1
        weight_sums = weights.sum(axis=1, keepdims=True)
        normalized_weights = weights / (weight_sums + epsilon)
        
        # OPTIMIZATION: Apply interpolation to all variations using same weights
        logger.info("🔮 Applying interpolation weights to all demand variations...")
        
        all_results = []
        for variation_name, demand_scores in all_demand_variations.items():
            logger.info(f"  → Processing '{variation_name}' variation...")
            
            # Get demand values for stations in same order as coordinates
            demand_column = f'demand_score_{variation_name}'
            if demand_column not in station_data_with_demand.columns:
                logger.warning(f"Column {demand_column} not found, skipping variation")
                continue
                
            station_demands = station_data_with_demand[demand_column].values
            
            # Apply interpolation using pre-calculated weights
            interpolated_demands = normalized_weights.dot(station_demands)
            
            # Create result DataFrame for this variation
            variation_results = []
            for i, (grid_id, demand) in enumerate(zip(grid_ids, interpolated_demands)):
                result_row = grid_metadata[i].copy()
                result_row[demand_column] = demand
                variation_results.append(result_row)
            
            variation_df = pd.DataFrame(variation_results)
            all_results.append(variation_df)
        
        # OPTIMIZATION: Merge all variations efficiently
        logger.info("🔗 Merging all demand variations...")
        if not all_results:
            raise ValueError("No valid demand variations processed")
        
        # Start with grid metadata from first result
        final_result = all_results[0][['grid_id', 'latitude', 'longitude', 'grid_type', 'grid_scale', 'grid_variation']].copy()
        
        # Add demand columns from all variations
        for result_df in all_results:
            demand_cols = [col for col in result_df.columns if col.startswith('demand_score_')]
            for col in demand_cols:
                final_result[col] = result_df[col]
        
        logger.info(f"✅ OPTIMIZED interpolation complete: {len(final_result)} grids × {len(all_demand_variations)} variations")
        
        return final_result