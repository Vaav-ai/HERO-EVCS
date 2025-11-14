"""
Feature Engineering for EV Charging Demand Prediction
====================================================

This module implements domain-specific feature engineering based on urban planning
theory and EV charging behavior patterns.

Key Components:
---------------
- `UrbanFeatureEngineer`: Creates features related to urban context, accessibility,
  economic activity, and spatial relationships.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from scipy.spatial.distance import pdist, squareform
import hashlib
import pickle
from pathlib import Path
from joblib import Parallel, delayed
import multiprocessing as mp
# import holidays  # Removed - not needed for geospatial demand prediction

logger = logging.getLogger(__name__)

class UrbanFeatureEngineer:
    """
    Creates domain-specific features for urban EV charging demand prediction
    based on established urban planning principles and EV infrastructure research.
    """
    
    def __init__(self, enable_caching: bool = True, use_parallel: bool = True):
        """Initialize the feature engineer with standard configurations."""
        self.enable_caching = enable_caching
        self.use_parallel = use_parallel
        self.cache_dir = Path("cache/feature_engineering")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Weights based on EV charging behavior research
        self.poi_weights = {
            'poi_density_commercial': 0.30,     # High correlation with charging demand
            'poi_density_residential': 0.20,    # Home/workplace charging context
            'poi_density_transport_hub': 0.25,  # Transit-oriented development
            'poi_density_healthcare': 0.10,     # Essential services
            'poi_density_education': 0.10,      # Regular trip destinations
            'poi_density_government': 0.05      # Occasional destinations
        }
        
        # Infrastructure weights for accessibility calculation
        self.accessibility_weights = {
            'road_component': 0.60,      # Primary accessibility factor
            'transit_component': 0.40    # Secondary accessibility factor
        }
        
        # Economic activity composition
        self.economic_weights = {
            'commercial_weight': 0.70,   # Primary economic driver
            'industrial_weight': 0.30    # Secondary economic factor
        }
    
    def _get_cache_key(self, data: pd.DataFrame, operation_name: str) -> str:
        """Generate a unique cache key for the data and operation."""
        # Create hash based on data shape, column names, and first/last few numeric values
        numeric_data = data.select_dtypes(include=[np.number])
        data_fingerprint = f"{data.shape}_{list(data.columns)}_{numeric_data.iloc[0].sum() if len(numeric_data) > 0 else 0}_{numeric_data.iloc[-1].sum() if len(numeric_data) > 0 else 0}"
        combined = f"{operation_name}_{data_fingerprint}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load cached results if available."""
        if not self.enable_caching:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame) -> None:
        """Save results to cache."""
        if not self.enable_caching:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
    
    # Removed engineer_demand_based_spatial_features method to prevent data leakage

    def create_spatial_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create spatial features based on urban planning theory"""
        logger.info("Creating spatial features...")
        
        result = data.copy()
        
        lat_col = next((col for col in result.columns if col.startswith('latitude')), None)
        lon_col = next((col for col in result.columns if col.startswith('longitude')), None)

        if not lat_col or not lon_col:
            logger.warning("No coordinates available for spatial features")
            return result
        
        logger.info(f"Using '{lat_col}' and '{lon_col}' for spatial features.")
        coordinates = result[[lat_col, lon_col]].values
        
        try:
            # 1. Distance to city center (CBD proxy)
            center_lat = np.mean(coordinates[:, 0])
            center_lon = np.mean(coordinates[:, 1])
            
            distances_to_center = np.sqrt(
                (coordinates[:, 0] - center_lat)**2 + 
                (coordinates[:, 1] - center_lon)**2
            )
            
            result['distance_to_center'] = distances_to_center
            result['inverse_distance_center'] = 1 / (1 + distances_to_center)
            
            # 2. Spatial density features (local neighborhood effects)
            # Skip expensive pairwise distance calculation for very large datasets
            if len(coordinates) > 1 and len(coordinates) <= 50000:
                try:
                    pairwise_distances = squareform(pdist(coordinates))
                    
                    # For each point, count neighbors within different radii
                    radii = [0.01, 0.02, 0.05]  # degrees (roughly 1km, 2km, 5km)
                    
                    for radius in radii:
                        neighbor_counts = (pairwise_distances <= radius).sum(axis=1) - 1  # Exclude self
                        result[f'neighbors_within_{int(radius*100)}km'] = neighbor_counts
                        
                        # Weighted neighbor density (closer neighbors have more weight)
                        weights = np.exp(-pairwise_distances / radius)
                        np.fill_diagonal(weights, 0)  # Remove self-weight
                        result[f'weighted_density_{int(radius*100)}km'] = weights.sum(axis=1)
                except MemoryError:
                    logger.warning("Memory limit exceeded for spatial density features, skipping...")
            elif len(coordinates) > 50000:
                logger.info(f"Dataset too large ({len(coordinates)} points) for pairwise spatial features, using simplified approach...")
                # Use a simpler approach for large datasets - grid-based density estimation
                for radius_deg in [0.01, 0.02, 0.05]:
                    # Simple grid-based neighbor counting
                    grid_size = radius_deg / 2
                    x_bins = np.arange(coordinates[:, 1].min(), coordinates[:, 1].max() + grid_size, grid_size)
                    y_bins = np.arange(coordinates[:, 0].min(), coordinates[:, 0].max() + grid_size, grid_size)
                    
                    # Count points in each grid cell
                    hist, _, _ = np.histogram2d(coordinates[:, 0], coordinates[:, 1], bins=[y_bins, x_bins])
                    
                    # Assign density values to points based on their grid cell
                    x_indices = np.digitize(coordinates[:, 1], x_bins) - 1
                    y_indices = np.digitize(coordinates[:, 0], y_bins) - 1
                    
                    # Clip indices to valid range
                    x_indices = np.clip(x_indices, 0, hist.shape[1] - 1)
                    y_indices = np.clip(y_indices, 0, hist.shape[0] - 1)
                    
                    density_values = hist[y_indices, x_indices]
                    result[f'neighbors_within_{int(radius_deg*100)}km'] = density_values
                    result[f'weighted_density_{int(radius_deg*100)}km'] = density_values
            
            logger.info(f"Created {len(result.columns) - len(data.columns)} spatial features")
            
        except Exception as e:
            logger.warning(f"Spatial feature creation failed: {e}")
        
        return result
    
    def engineer_urban_context_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create urban context features from POI density data.
        
        Args:
            data: DataFrame with POI density columns
            
        Returns:
            DataFrame with additional urban context features
        """
        logger.info("Engineering urban context features from POI data...")
        
        # Check cache first
        cache_key = self._get_cache_key(data, "urban_context")
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            logger.info("✅ Loaded urban context features from cache")
            return cached_result
        
        result = data.copy()
        
        # 1. Urban Intensity Index - vectorized weighted combination
        poi_features = [f for f in self.poi_weights.keys() if f in result.columns]
        if poi_features:
            weights = np.array([self.poi_weights[f] for f in poi_features])
            poi_values = result[poi_features].fillna(0).values
            result['urban_intensity_index'] = np.dot(poi_values, weights)
        else:
            result['urban_intensity_index'] = 0.0
        
        # 2. Land Use Diversity - vectorized Shannon entropy
        poi_columns = [col for col in result.columns if col.startswith('poi_density_')]
        if len(poi_columns) > 1:
            diversity_scores = self._calculate_land_use_diversity_vectorized(result, poi_columns)
            result['land_use_diversity'] = diversity_scores
        
        # 3. Infrastructure Service Level - vectorized mean
        infrastructure_cols = [
            'poi_density_healthcare', 
            'poi_density_education', 
            'poi_density_government'
        ]
        available_infra_cols = [col for col in infrastructure_cols if col in result.columns]
        
        if available_infra_cols:
            infra_data = result[available_infra_cols].fillna(0).values
            result['infrastructure_service_level'] = np.mean(infra_data, axis=1)
        else:
            result['infrastructure_service_level'] = 0.0
        
        # Cache the result
        self._save_to_cache(cache_key, result)
        
        logger.info(f"Created {len(result.columns) - len(data.columns)} urban context features")
        return result
    
    def engineer_accessibility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create accessibility features from transportation data.
        
        Args:
            data: DataFrame with road and transit data
            
        Returns:
            DataFrame with accessibility features
        """
        logger.info("Engineering accessibility features from transportation data...")
        
        result = data.copy()
        
        # 1. Multimodal Accessibility Score
        if 'road_density_total' in result.columns and 'public_transport_access' in result.columns:
            road_max = result['road_density_total'].max()
            transit_max = result['public_transport_access'].max()
            
            if road_max > 0:
                road_norm = result['road_density_total'] / road_max
            else:
                road_norm = pd.Series(0, index=result.index)
                
            if transit_max > 0:
                transit_norm = result['public_transport_access'] / transit_max
            else:
                transit_norm = pd.Series(0, index=result.index)
            
            result['multimodal_accessibility'] = (
                self.accessibility_weights['road_component'] * road_norm + 
                self.accessibility_weights['transit_component'] * transit_norm
            )
        
        # 2. Transport Hub Effectiveness (composite measure)
        if 'poi_density_transport_hub' in result.columns and 'intersection_density' in result.columns:
            hub_component = result['poi_density_transport_hub'].fillna(0)
            intersection_component = result['intersection_density'].fillna(0)
            
            # Normalize both components to 0-1 range to prevent perfect correlation
            hub_max = hub_component.max()
            intersection_max = intersection_component.max()
            
            if hub_max > 0:
                hub_norm = hub_component / hub_max
            else:
                hub_norm = pd.Series(0, index=result.index)
                
            if intersection_max > 0:
                intersection_norm = intersection_component / intersection_max
            else:
                intersection_norm = pd.Series(0, index=result.index)
            
            # Create composite measure that's distinct from raw hub density
            result['transport_hub_effectiveness'] = (
                hub_norm * 0.4 + intersection_norm * 0.6  # Weighted toward intersection density
            )
        
        logger.info(f"Created {len(result.columns) - len(data.columns)} accessibility features")
        return result
    
    def engineer_economic_activity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create economic activity features from commercial and industrial data.
        
        Args:
            data: DataFrame with commercial and industrial POI data
            
        Returns:
            DataFrame with economic activity features
        """
        logger.info("Engineering economic activity features...")
        
        result = data.copy()
        
        # 1. Economic Activity Index
        if 'poi_density_commercial' in result.columns and 'poi_density_industrial' in result.columns:
            result['economic_activity_index'] = (
                result['poi_density_commercial'].fillna(0) * self.economic_weights['commercial_weight'] + 
                result['poi_density_industrial'].fillna(0) * self.economic_weights['industrial_weight']
            )
        
        # 2. Commercial to Residential Ratio (mixed-use indicator)
        if 'poi_density_commercial' in result.columns and 'poi_density_residential' in result.columns:
            # Add small constant to avoid division by zero
            denominator = result['poi_density_residential'].fillna(0) + 1e-6
            result['commercial_residential_ratio'] = result['poi_density_commercial'].fillna(0) / denominator
        
        logger.info(f"Created {len(result.columns) - len(data.columns)} economic activity features")
        return result
    
    def engineer_spatial_context_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create spatial context features from grid positioning data.
        
        Args:
            data: DataFrame with grid_id information
            
        Returns:
            DataFrame with spatial context features
        """
        logger.info("Engineering spatial context features from grid positioning...")
        
        result = data.copy()
        
        if 'grid_id' in result.columns and pd.api.types.is_string_dtype(result['grid_id']):
            # Extract numerical components from grid IDs for spatial analysis
            grid_numeric = result['grid_id'].str.extract(r'(\d+)').astype(float).fillna(0)
            result['grid_position_x'] = grid_numeric % 100  # Proxy for x-coordinate
            result['grid_position_y'] = grid_numeric // 100  # Proxy for y-coordinate
            
            # Calculate distance from center (urbanity proxy)
            center_x = result['grid_position_x'].mean()
            center_y = result['grid_position_y'].mean()
            
            result['distance_from_center'] = np.sqrt(
                (result['grid_position_x'] - center_x)**2 + 
                (result['grid_position_y'] - center_y)**2
            )
            
            # Note: urban_centrality_score removed to prevent redundancy with distance_from_center
            # distance_from_center already captures spatial centrality information
        
        logger.info(f"Created {len(result.columns) - len(data.columns)} spatial context features")
        return result
    
    def engineer_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between key urban characteristics.
        
        Args:
            data: DataFrame with base features
            
        Returns:
            DataFrame with interaction features
        """
        logger.info("Engineering interaction features between urban characteristics...")
        
        result = data.copy()
        
        # Key interaction pairs based on urban planning theory
        interaction_pairs = [
            ('urban_intensity_index', 'multimodal_accessibility', 'urban_accessibility_synergy'),
            ('economic_activity_index', 'transport_hub_effectiveness', 'economic_transport_synergy'),
            ('land_use_diversity', 'infrastructure_service_level', 'service_diversity_interaction')
        ]
        
        for feature1, feature2, interaction_name in interaction_pairs:
            if feature1 in result.columns and feature2 in result.columns:
                result[interaction_name] = result[feature1].fillna(0) * result[feature2].fillna(0)
        
        logger.info(f"Created {len(result.columns) - len(data.columns)} interaction features")
        return result
    
    def engineer_nonlinear_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create nonlinear transformations of key features.
        
        NOTE: Polynomial features (squared/sqrt) are disabled to prevent redundancy.
        These features create high correlation (0.8-0.84) with original features,
        causing multicollinearity issues. The original features already capture
        the necessary information for ML models.
        
        Args:
            data: DataFrame with base features
            
        Returns:
            DataFrame unchanged (polynomial features disabled)
        """
        logger.info("Polynomial features disabled to prevent redundancy and multicollinearity")
        logger.info("Original features already capture necessary nonlinear relationships")
        
        # Return data unchanged - polynomial features are redundant
        return data.copy()
    
    def normalize_feature_set(self, data: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
        """
        Normalize specified features to [0, 1] range.
        
        Args:
            data: DataFrame with features to normalize
            feature_list: List of feature names to normalize
            
        Returns:
            DataFrame with normalized features
        """
        result = data.copy()
        
        for feature in feature_list:
            if feature in result.columns:
                feat_min, feat_max = result[feature].min(), result[feature].max()
                if feat_max > feat_min:
                    result[f'{feature}_normalized'] = (result[feature].fillna(0) - feat_min) / (feat_max - feat_min)
                else:
                    result[f'{feature}_normalized'] = 0.0
        
        return result
    
    def _calculate_land_use_diversity_vectorized(self, data: pd.DataFrame, poi_columns: List[str]) -> np.ndarray:
        """
        OPTIMIZED: Vectorized Shannon entropy calculation for land use diversity.
        
        Args:
            data: DataFrame with POI columns
            poi_columns: List of POI density column names
            
        Returns:
            NumPy array of diversity scores
        """
        poi_data = data[poi_columns].fillna(0).values
        
        # Vectorized computation
        row_sums = poi_data.sum(axis=1)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1e-8
        
        # Calculate proportions for all rows at once
        proportions = poi_data / row_sums.reshape(-1, 1)
        
        # Handle zeros in log calculation
        proportions = np.where(proportions == 0, 1e-8, proportions)
        
        # Vectorized Shannon entropy
        log_proportions = np.log(proportions)
        diversity_scores = -np.sum(proportions * log_proportions, axis=1)
        
        # Set diversity to 0 where all POI values were 0
        mask = (poi_data.sum(axis=1) == 0)
        diversity_scores[mask] = 0.0
        
        return diversity_scores
    
    def _calculate_land_use_diversity(self, data: pd.DataFrame, poi_columns: List[str]) -> List[float]:
        """
        Calculate Shannon entropy for land use diversity (wrapper for backward compatibility).
        """
        return self._calculate_land_use_diversity_vectorized(data, poi_columns).tolist()
    
    def apply_complete_feature_engineering(self, data: pd.DataFrame, skip_spatial_features: bool = False) -> pd.DataFrame:
        """
        Apply complete feature engineering pipeline.
        
        Args:
            data: Raw training data with OSM features
            skip_spatial_features: If True, skip spatial feature generation (useful for large datasets)
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("=== APPLYING COMPLETE FEATURE ENGINEERING PIPELINE ===")
        
        # Step 1: Urban context features
        result = self.engineer_urban_context_features(data)
        
        # Step 2: Accessibility features
        result = self.engineer_accessibility_features(result)
        
        # Step 3: Economic activity features
        result = self.engineer_economic_activity_features(result)
        
        # Step 4: Spatial context features (from lat/lon and grid)
        if skip_spatial_features:
            logger.info("⏭️  Skipping spatial feature generation (disabled via parameter)")
        else:
            result = self.create_spatial_features(result)
        result = self.engineer_spatial_context_features(result)
        
        # Step 5: Interaction features
        result = self.engineer_interaction_features(result)
        
        # Step 6: Nonlinear features
        result = self.engineer_nonlinear_features(result)
        
        # Step 7: Normalization disabled to prevent redundant features
        # Normalized features create perfect correlation (1.0) with original features
        # ML models can handle unnormalized features, and we avoid multicollinearity
        logger.info("Feature normalization disabled to prevent redundant features")
        
        logger.info(f"Feature engineering complete: {len(data.columns)} → {len(result.columns)} features")
        return result

