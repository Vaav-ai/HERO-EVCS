#!/usr/bin/env python3
"""
Comprehensive Demand Model Validation Script

This script provides rigorous validation of demand prediction models using
Ann Arbor VED data as ground truth. It addresses the core methodology questions
about grid size and model performance.

Usage:
    python evaluation/run_validation.py --grid-size 1.0
    python evaluation/run_validation.py --grid-size 1.5 --model-path models/custom_model.pkl
    python evaluation/run_validation.py --sensitivity-analysis
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
import pickle
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from modules.utils.gridding import CityGridding
from modules.MLPrediction.demand_predictor import DemandPredictor
from modules.MLPrediction.portable_osm_extractor import PortableOSMExtractor
from modules.MLPrediction.feature_engineering import UrbanFeatureEngineer
# Simple model loading from directory
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DemandModelValidator:
    """
    Comprehensive validation framework for demand prediction models.
    
    This class provides tools to evaluate model performance across different
    grid sizes and generate detailed validation reports.
    """
    
    def __init__(self, validation_data_path: str = "data/validation/ved_processed_with_grids.parquet",
                 output_dir: str = "validation_results", models_dir: str = "results/models"):
        """
        Initialize the validator.
        
        Args:
            validation_data_path: Path to Ann Arbor VED validation data
            output_dir: Directory to save validation results
            models_dir: Directory containing model PKL files
        """
        self.validation_data_path = Path(validation_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.osm_extractor = PortableOSMExtractor()
        self.feature_engineer = UrbanFeatureEngineer()
        
        # Simple model directory
        self.models_dir = Path(models_dir)
        
        # Validation results storage
        self.results = {}
        
    def validate_model(self, grid_size_km: float, model_path: Optional[str] = None, model_data: Optional[Dict] = None) -> Dict:
        """
        Validate a demand prediction model using Ann Arbor VED data.
        
        Args:
            grid_size_km: Grid size in kilometers to use for validation
            model_path: Optional path to specific model. If None, uses default trained model
            model_data: Optional pre-loaded model data dictionary (for individual models)
            
        Returns:
            Dictionary containing validation metrics and analysis
        """
        logger.info(f"🔬 Starting validation with {grid_size_km}km grid size")
        
        # Step 1: Load and prepare validation data
        if not self.validation_data_path.exists():
            raise FileNotFoundError(f"Validation data not found: {self.validation_data_path}")
            
        logger.info("📊 Loading Ann Arbor VED validation data...")
        ved_data = pd.read_parquet(self.validation_data_path)
        logger.info(f"Loaded {len(ved_data)} validation data points")
        
        # Step 2: Create grids for Ann Arbor with specified size
        logger.info(f"🏗️ Creating {grid_size_km}km grids for Ann Arbor...")
        gridder = CityGridding(
            primary_grid_size_km=grid_size_km,
            fetch_osm_features=False,
            debug_mode=False
        )
        
        ann_arbor_coords = (42.2808, -83.7430)
        grid_cells = gridder.create_city_grid("Ann Arbor, Michigan, USA", coordinates=ann_arbor_coords)
        
        logger.info(f"Created {len(grid_cells)} grid cells for Ann Arbor")
        
        # Step 3: Assign VED data to grids
        logger.info("📍 Assigning VED data points to grid cells...")
        ved_with_grids = self._assign_data_to_grids(ved_data, grid_cells)
        
        # Step 4: Calculate actual demand scores from VED data
        logger.info("⚡ Calculating actual demand scores from VED data...")
        actual_demand = self._calculate_actual_demand_scores(ved_with_grids)
        
        # Step 5: Extract OSM features for grids with demand data
        logger.info("🗺️ Extracting OSM features for validation grids...")
        grid_cells_for_prediction = []
        for _, row in actual_demand.iterrows():
            grid_cells_for_prediction.append({
                'grid_id': row['grid_id'],
                'min_lat': row['min_lat'],
                'max_lat': row['max_lat'],
                'min_lon': row['min_lon'],
                'max_lon': row['max_lon']
            })
        
        # Only extract features for grids that have demand data
        relevant_grid_ids = list(actual_demand['grid_id'].unique())
        osm_features = self.osm_extractor.extract_features_for_grids(
            grid_cells_for_prediction,
            city_name="Ann Arbor (Validation)",
            relevant_grid_ids=relevant_grid_ids
        )
        
        # Step 6: Load demand predictor and check feature requirements
        logger.info("🧠 Loading demand model and checking feature requirements...")
        
        if model_data is not None:
            # Use pre-loaded model data (for individual models)
            predictor = self._create_predictor_from_model_data(model_data)
        else:
            # Check if this is an individual model file
            if model_path and model_path.endswith('.pkl'):
                try:
                    # Try to load as individual model first
                    with open(model_path, 'rb') as f:
                        individual_model_data = pickle.load(f)
                    
                    # Check if it's an individual model format (MLPrediction structure)
                    if isinstance(individual_model_data, dict) and 'model_object' in individual_model_data:
                        logger.info(f"✅ Loaded individual model: {individual_model_data.get('model_name', 'unknown')} "
                                  f"({individual_model_data.get('target_formulation', 'unknown')} - "
                                  f"{individual_model_data.get('target_transformation', 'unknown')})")
                        predictor = self._create_predictor_from_model_data(individual_model_data)
                    else:
                        # Fall back to traditional DemandPredictor
                        logger.info("Using traditional DemandPredictor for model loading")
                        predictor = DemandPredictor(model_path=model_path)
                except Exception as e:
                    logger.warning(f"Failed to load as individual model, falling back to DemandPredictor: {e}")
                    predictor = DemandPredictor(model_path=model_path)
            else:
                # Use model path (for traditional model files)
                predictor = DemandPredictor(model_path=model_path)
        
        # Determine what features the model expects
        expected_features = predictor._get_expected_features()
        logger.info(f"Model expects {len(expected_features)} features")
        
        # Check if we need feature engineering by examining the feature names
        basic_osm_features = set(self.osm_extractor.get_portable_feature_columns())
        expected_features_set = set(expected_features)
        
        # Check if model needs engineered features (features not in basic OSM set)
        engineered_features_needed = expected_features_set - basic_osm_features
        
        if engineered_features_needed:
            logger.info("🔧 Model requires engineered features. Applying MLPrediction feature engineering pipeline...")
            logger.info(f"Engineered features needed: {sorted(list(engineered_features_needed))[:5]}...")
            
            # Add required columns for feature engineering following MLPrediction pattern
            features_with_coords = osm_features.copy()
            
            # Add lat/lon coordinates (required for spatial features) - following MLPrediction pattern
            features_with_coords['latitude'] = [
                (row['min_lat'] + row['max_lat']) / 2 
                for _, row in actual_demand.iterrows()
            ]
            features_with_coords['longitude'] = [
                (row['min_lon'] + row['max_lon']) / 2 
                for _, row in actual_demand.iterrows()
            ]
            
            # Apply complete feature engineering pipeline following MLPrediction structure
            engineered_features = self.feature_engineer.apply_complete_feature_engineering(
                features_with_coords, 
                skip_spatial_features=False  # Enable spatial features for validation
            )
            
            # Use engineered features for prediction
            prediction_features = engineered_features
            logger.info(f"MLPrediction feature engineering complete: {len(osm_features.columns)} → {len(engineered_features.columns)} features")
        else:
            logger.info("✅ Model uses basic OSM features only")
            prediction_features = osm_features
        
        # Step 7: Make predictions
        logger.info("🎯 Making demand predictions...")
        try:
            predicted_scores = predictor.predict(prediction_features)
        except ValueError as e:
            if "Missing required OSM feature columns" in str(e):
                logger.error(f"❌ Feature mismatch: {e}")
                self._debug_feature_mismatch(
                    list(prediction_features.columns), 
                    expected_features
                )
            raise e
        
        # Step 8: Combine predictions with actual demand
        validation_results = prediction_features[['grid_id']].copy()
        validation_results['predicted_demand'] = predicted_scores
        validation_results = validation_results.merge(
            actual_demand[['grid_id', 'actual_demand']], 
            on='grid_id', 
            how='inner'
        )
        
        logger.info(f"Successfully matched {len(validation_results)} grids with both predicted and actual demand")
        
        # Step 9: Calculate validation metrics
        metrics = self._calculate_validation_metrics(validation_results)
        
        # Create unique filename based on model info following MLPrediction naming convention
        if model_data is not None:
            model_name = model_data.get('model_name', 'unknown')
            target_formulation = model_data.get('target_formulation', 'unknown')
            target_transformation = model_data.get('target_transformation', 'unknown')
            scaler_name = model_data.get('scaler_name', 'unknown')
            sample_size = model_data.get('training_configuration', {}).get('sample_size', '')
            sample_suffix = f"_sample{sample_size}" if sample_size else ""
            model_identifier = f"{target_formulation}_{target_transformation}_{scaler_name}_{model_name}{sample_suffix}"
            
            # Log model information following MLPrediction patterns
            logger.info(f"📊 Validating MLPrediction model: {model_name}")
            logger.info(f"   • Target: {target_formulation} ({target_transformation})")
            logger.info(f"   • Scaler: {scaler_name}")
            logger.info(f"   • Sample Size: {sample_size if sample_size else 'Full dataset'}")
        else:
            model_identifier = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save metrics to CSV with unique filename
        metrics_df = pd.DataFrame([metrics])
        metrics_path = self.output_dir / f"validation_metrics_{grid_size_km}km_{model_identifier}.csv"
        metrics_df.to_csv(metrics_path, index=False)
        
        # Step 10: Create comprehensive report (no plots)
        report = {
            'grid_size_km': grid_size_km,
            'model_path': model_path or 'default',
            'model_identifier': model_identifier,
            'timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_ved_points': len(ved_data),
                'total_grids_created': len(grid_cells),
                'grids_with_demand': len(actual_demand),
                'grids_for_validation': len(validation_results)
            },
            'metrics': metrics,
            'validation_data': validation_results.to_dict('records')
        }
        
        # Save detailed results with unique filename
        report_path = self.output_dir / f"validation_report_{grid_size_km}km_{model_identifier}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        csv_path = self.output_dir / f"validation_data_{grid_size_km}km_{model_identifier}.csv"
        validation_results.to_csv(csv_path, index=False)
        
        # Log validation results following MLPrediction patterns
        logger.info(f"📊 MLPrediction Validation Results for {grid_size_km}km grids:")
        logger.info(f"  • Pearson Correlation: {metrics['pearson_correlation']:.4f}")
        logger.info(f"  • Spearman Correlation: {metrics['spearman_correlation']:.4f}")
        logger.info(f"  • R² Score: {metrics['r2_score']:.4f}")
        logger.info(f"  • RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  • MAE: {metrics['mae']:.4f}")
        logger.info(f"  • Sample Size: {metrics['n_samples']}")
        
        # Log file paths following MLPrediction output patterns
        logger.info(f"💾 MLPrediction validation results saved to:")
        logger.info(f"  • Report: {report_path}")
        logger.info(f"  • Data: {csv_path}")
        logger.info(f"  • Metrics: {metrics_path}")
        logger.info(f"  • Model: {model_identifier}")
        
        return report
    
    def sensitivity_analysis(self, grid_sizes: List[float] = None, model_path: Optional[str] = None) -> Dict:
        """
        Perform grid size sensitivity analysis.
        
        Args:
            grid_sizes: List of grid sizes to test. Defaults to [1.0, 1.5, 2.0, 2.5]
            model_path: Optional path to specific model
            
        Returns:
            Dictionary containing sensitivity analysis results
        """
        if grid_sizes is None:
            grid_sizes = [0.5, 0.75, 1.0, 1.5, 2.0]  # Match training scales exactly
            
        logger.info(f"🔬 Starting grid size sensitivity analysis for sizes: {grid_sizes}")
        
        sensitivity_results = []
        
        for grid_size in grid_sizes:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing Grid Size: {grid_size} km")
            logger.info(f"{'='*60}")
            
            try:
                result = self.validate_model(grid_size, model_path)
                sensitivity_results.append({
                    'grid_size_km': grid_size,
                    'pearson_correlation': result['metrics']['pearson_correlation'],
                    'spearman_correlation': result['metrics']['spearman_correlation'],
                    'r2_score': result['metrics']['r2_score'],
                    'rmse': result['metrics']['rmse'],
                    'mae': result['metrics']['mae'],
                    'n_samples': result['metrics']['n_samples']
                })
            except Exception as e:
                logger.error(f"❌ Failed validation for {grid_size}km: {e}")
                sensitivity_results.append({
                    'grid_size_km': grid_size,
                    'error': str(e)
                })
        
        # Save sensitivity analysis results to CSV
        sensitivity_df = pd.DataFrame(sensitivity_results)
        sensitivity_path = self.output_dir / "sensitivity_analysis.csv"
        sensitivity_df.to_csv(sensitivity_path, index=False)
        
        summary = {
            'analysis_type': 'grid_size_sensitivity',
            'timestamp': datetime.now().isoformat(),
            'tested_grid_sizes': grid_sizes,
            'results': sensitivity_results,
            'best_grid_size': self._find_best_grid_size(sensitivity_results)
        }
        
        summary_path = self.output_dir / "sensitivity_analysis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n🎯 Sensitivity Analysis Complete!")
        logger.info(f"📊 Results saved to: {sensitivity_path}")
        logger.info(f"📋 Summary saved to: {summary_path}")
        
        if summary['best_grid_size']:
            logger.info(f"🏆 Best performing grid size: {summary['best_grid_size']['grid_size_km']}km")
            logger.info(f"    Spearman Correlation: {summary['best_grid_size']['spearman_correlation']:.4f}")
        
        return summary
    
    def validate_all_models(self, grid_size_km: float = 1.0, target_filter: str = None, limit: int = None) -> Dict:
        """
        Validate all models in the models directory against Ann Arbor VED data.
        
        Args:
            grid_size_km: Grid size in kilometers to use for validation
            target_filter: Optional target formulation to filter models (e.g., 'demand_score_balanced')
            limit: Optional limit on number of models to validate (for testing)
            
        Returns:
            Dictionary containing validation results for all models
        """
        logger.info(f"🔬 Validating ALL models with {grid_size_km}km grid size")
        if target_filter:
            logger.info(f"🎯 Filtering for target: {target_filter}")
        
        # Get all model files from directory
        model_files = list(self.models_dir.glob("*.pkl"))
        
        if not model_files:
            logger.warning("No model files found in models directory")
            return {}
        
        # Filter models if target_filter is specified
        if target_filter:
            model_files = [f for f in model_files if target_filter in f.name]
            logger.info(f"📊 Found {len(model_files)} models for target: {target_filter}")
        
        if not model_files:
            logger.warning(f"No models found for target: {target_filter}")
            return {}
        
        # Apply limit if specified
        if limit is not None:
            model_files = model_files[:limit]
            logger.info(f"🔢 Limited to first {limit} models for testing")
        
        validation_results = {}
        successful_validations = 0
        failed_validations = 0
        
        for i, model_file in enumerate(model_files, 1):
            logger.info(f"📊 Validating ({i}/{len(model_files)}): {model_file.name}")
            
            try:
                # Load model data with better error handling
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Validate model data structure
                if not isinstance(model_data, dict):
                    raise ValueError(f"Model file {model_file.name} does not contain expected dictionary format")
                
                required_keys = ['model_object', 'feature_columns']
                missing_keys = [key for key in required_keys if key not in model_data]
                if missing_keys:
                    raise ValueError(f"Model file {model_file.name} missing required keys: {missing_keys}")
                
                # Validate this specific model using pre-loaded data
                result = self.validate_model(
                    grid_size_km=grid_size_km,
                    model_data=model_data
                )
                
                # Add model metadata to result
                result['model_metadata'] = {
                    'filename': model_file.name,
                    'target_formulation': model_data.get('target_formulation', 'unknown'),
                    'target_transformation': model_data.get('target_transformation', 'unknown'),
                    'model_name': model_data.get('model_name', 'unknown'),
                    'scaler_name': model_data.get('scaler_name', 'unknown'),
                    'training_r2': model_data.get('performance_metrics', {}).get('r2', 0),
                    'sample_size': model_data.get('training_configuration', {}).get('sample_size', 'unknown')
                }
                
                validation_results[model_file.stem] = result
                successful_validations += 1
                
                # Handle NaN values in metrics
                spearman = result['metrics']['spearman_correlation']
                r2 = result['metrics']['r2_score']
                spearman_str = f"{spearman:.4f}" if not pd.isna(spearman) else "NaN"
                r2_str = f"{r2:.4f}" if not pd.isna(r2) else "NaN"
                
                logger.info(f"    ✅ Validation complete: Spearman={spearman_str}, R²={r2_str}")
                
            except Exception as e:
                logger.error(f"    ❌ Validation failed for {model_file.name}: {e}")
                validation_results[model_file.stem] = {
                    'error': str(e),
                    'filename': model_file.name,
                    'error_type': type(e).__name__
                }
                failed_validations += 1
        
        # Create timestamp for unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save comprehensive validation results with timestamp
        results_path = self.output_dir / f"all_models_validation_{grid_size_km}km_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        # Create a summary CSV with all model results
        summary_data = []
        for model_name, result in validation_results.items():
            if 'error' not in result:
                summary_data.append({
                    'model_name': model_name,
                    'target_formulation': result.get('model_metadata', {}).get('target_formulation', 'unknown'),
                    'target_transformation': result.get('model_metadata', {}).get('target_transformation', 'unknown'),
                    'model_algorithm': result.get('model_metadata', {}).get('model_name', 'unknown'),
                    'scaler_name': result.get('model_metadata', {}).get('scaler_name', 'unknown'),
                    'pearson_correlation': result['metrics']['pearson_correlation'],
                    'spearman_correlation': result['metrics']['spearman_correlation'],
                    'r2_score': result['metrics']['r2_score'],
                    'rmse': result['metrics']['rmse'],
                    'mae': result['metrics']['mae'],
                    'n_samples': result['metrics']['n_samples']
                })
            else:
                summary_data.append({
                    'model_name': model_name,
                    'target_formulation': 'error',
                    'target_transformation': 'error',
                    'model_algorithm': 'error',
                    'scaler_name': 'error',
                    'pearson_correlation': 'N/A',
                    'spearman_correlation': 'N/A',
                    'r2_score': 'N/A',
                    'rmse': 'N/A',
                    'mae': 'N/A',
                    'n_samples': 'N/A',
                    'error': result['error']
                })
        
        # Save summary CSV
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.output_dir / f"validation_summary_{grid_size_km}km_{timestamp}.csv"
        summary_df.to_csv(summary_path, index=False)
        
        # Generate summary statistics
        valid_results = [r for r in validation_results.values() if 'error' not in r]
        if valid_results:
            spearman_scores = [r['metrics']['spearman_correlation'] for r in valid_results]
            r2_scores = [r['metrics']['r2_score'] for r in valid_results]
            
            logger.info(f"\n📊 Validation Summary:")
            logger.info(f"   - Total models: {len(model_files)}")
            logger.info(f"   - Successful validations: {successful_validations}")
            logger.info(f"   - Failed validations: {failed_validations}")
            logger.info(f"   - Best Spearman correlation: {max(spearman_scores):.4f}")
            logger.info(f"   - Best R² score: {max(r2_scores):.4f}")
            logger.info(f"   - Average Spearman correlation: {np.mean(spearman_scores):.4f}")
            logger.info(f"   - Average R² score: {np.mean(r2_scores):.4f}")
        
        logger.info(f"💾 Results saved to:")
        logger.info(f"  • Detailed results: {results_path}")
        logger.info(f"  • Summary CSV: {summary_path}")
        logger.info(f"  • Individual model files: {len(validation_results)} files")
        
        return validation_results
    
    def _assign_data_to_grids(self, ved_data: pd.DataFrame, grid_cells: List[Dict]) -> pd.DataFrame:
        """Assign VED data points to grid cells."""
        from shapely.geometry import Point, Polygon
        import geopandas as gpd
        
        # Drop conflicting columns if they exist
        cols_to_drop = ['index_right', 'cell_id']
        ved_data = ved_data.drop(columns=[col for col in cols_to_drop if col in ved_data.columns])

        # Create GeoDataFrame from VED data
        ved_gdf = gpd.GeoDataFrame(
            ved_data,
            geometry=gpd.points_from_xy(ved_data['lon'], ved_data['lat']),
            crs='EPSG:4326'
        )
        
        # Create GeoDataFrame from grid cells
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
            grid_data.append(cell)
        
        grid_gdf = gpd.GeoDataFrame(grid_data, geometry=grid_polygons, crs='EPSG:4326')
        
        # Spatial join
        ved_with_grids = gpd.sjoin(ved_gdf, grid_gdf, how='inner', predicate='within')
        
        return ved_with_grids
    
    def _calculate_actual_demand_scores(self, ved_with_grids: pd.DataFrame) -> pd.DataFrame:
        """Calculate actual demand scores from VED data."""
        # Group by grid and calculate demand metrics
        demand_metrics = ved_with_grids.groupby('grid_id').agg({
            'VehId': 'nunique',  # Number of unique vehicles
            'timestamp': 'count',      # Number of charging events
            'min_lat': 'first',
            'max_lat': 'first', 
            'min_lon': 'first',
            'max_lon': 'first'
        }).reset_index()
        
        demand_metrics.columns = ['grid_id', 'unique_vehicles', 'total_events', 
                                'min_lat', 'max_lat', 'min_lon', 'max_lon']
        
        # Normalize to create demand score (0-1 scale)
        max_vehicles = demand_metrics['unique_vehicles'].max()
        max_events = demand_metrics['total_events'].max()
        
        # Combined score based on both unique vehicles and total events
        demand_metrics['vehicle_score'] = demand_metrics['unique_vehicles'] / max_vehicles
        demand_metrics['event_score'] = demand_metrics['total_events'] / max_events
        demand_metrics['actual_demand'] = (demand_metrics['vehicle_score'] + demand_metrics['event_score']) / 2
        
        return demand_metrics
    
    def _calculate_validation_metrics(self, validation_results: pd.DataFrame) -> Dict:
        """Calculate comprehensive validation metrics with robust error handling."""
        actual = validation_results['actual_demand'].values
        predicted = validation_results['predicted_demand'].values
        
        # Remove any NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        
        if len(actual_clean) == 0:
            return {
                'error': 'No valid data points for validation',
                'pearson_correlation': None,
                'pearson_p_value': None,
                'spearman_correlation': None,
                'spearman_p_value': None,
                'r2_score': None,
                'rmse': None,
                'mae': None,
                'n_samples': 0,
                'actual_mean': None,
                'actual_std': None,
                'predicted_mean': None,
                'predicted_std': None,
                'actual_constant': False,
                'predicted_constant': False
            }
        
        # Check for constant values (which cause correlation issues)
        actual_constant = np.all(actual_clean == actual_clean[0])
        predicted_constant = np.all(predicted_clean == predicted_clean[0])
        
        # Calculate correlation metrics with error handling
        try:
            if actual_constant and predicted_constant:
                pearson_corr, pearson_p = np.nan, np.nan
                spearman_corr, spearman_p = np.nan, np.nan
            elif actual_constant or predicted_constant:
                pearson_corr, pearson_p = np.nan, np.nan
                spearman_corr, spearman_p = np.nan, np.nan
            else:
                pearson_corr, pearson_p = pearsonr(actual_clean, predicted_clean)
                spearman_corr, spearman_p = spearmanr(actual_clean, predicted_clean)
        except Exception:
            pearson_corr, pearson_p = np.nan, np.nan
            spearman_corr, spearman_p = np.nan, np.nan
        
        # Calculate regression metrics with error handling
        try:
            r2 = r2_score(actual_clean, predicted_clean)
        except Exception:
            r2 = np.nan
            
        try:
            rmse = np.sqrt(mean_squared_error(actual_clean, predicted_clean))
        except Exception:
            rmse = np.nan
            
        try:
            mae = mean_absolute_error(actual_clean, predicted_clean)
        except Exception:
            mae = np.nan
        
        return {
            'pearson_correlation': float(pearson_corr) if not pd.isna(pearson_corr) else None,
            'pearson_p_value': float(pearson_p) if not pd.isna(pearson_p) else None,
            'spearman_correlation': float(spearman_corr) if not pd.isna(spearman_corr) else None,
            'spearman_p_value': float(spearman_p) if not pd.isna(spearman_p) else None,
            'r2_score': float(r2) if not pd.isna(r2) else None,
            'rmse': float(rmse) if not pd.isna(rmse) else None,
            'mae': float(mae) if not pd.isna(mae) else None,
            'n_samples': int(len(actual_clean)),
            'actual_mean': float(np.mean(actual_clean)) if len(actual_clean) > 0 else None,
            'actual_std': float(np.std(actual_clean)) if len(actual_clean) > 0 else None,
            'predicted_mean': float(np.mean(predicted_clean)) if len(predicted_clean) > 0 else None,
            'predicted_std': float(np.std(predicted_clean)) if len(predicted_clean) > 0 else None,
            'actual_constant': bool(actual_constant),
            'predicted_constant': bool(predicted_constant)
        }
    
    def _find_best_grid_size(self, sensitivity_results: List[Dict]) -> Optional[Dict]:
        """Find the best performing grid size based on Spearman correlation."""
        valid_results = [r for r in sensitivity_results if 'error' not in r and 'spearman_correlation' in r]
        
        if not valid_results:
            return None
        
        # Handle NaN values in spearman correlation
        valid_numeric_results = [
            r for r in valid_results 
            if not pd.isna(r.get('spearman_correlation', np.nan))
        ]
        
        if not valid_numeric_results:
            # If no valid numeric results, fall back to R² score
            valid_numeric_results = [
                r for r in valid_results 
                if not pd.isna(r.get('r2_score', np.nan))
            ]
            if valid_numeric_results:
                best_result = max(valid_numeric_results, key=lambda x: x['r2_score'])
                return best_result
            return valid_results[0] if valid_results else None
        
        best_result = max(valid_numeric_results, key=lambda x: x['spearman_correlation'])
        return best_result
    
    def _debug_feature_mismatch(self, available_features: List[str], expected_features: List[str]):
        """Debug helper to understand feature mismatches."""
        logger.info("🔍 Feature Analysis:")
        logger.info(f"  Available features: {len(available_features)}")
        logger.info(f"  Expected features: {len(expected_features)}")
        
        missing = set(expected_features) - set(available_features)
        extra = set(available_features) - set(expected_features)
        
        if missing:
            logger.warning(f"  Missing features ({len(missing)}): {sorted(list(missing))[:10]}...")
        if extra:
            logger.info(f"  Extra features ({len(extra)}): {sorted(list(extra))[:10]}...")
        
        common = set(available_features) & set(expected_features)
        logger.info(f"  Common features: {len(common)}/{len(expected_features)}")
    
    def _create_predictor_from_model_data(self, model_data: Dict) -> 'DemandPredictor':
        """
        Create a DemandPredictor instance from individual model data.
        
        Args:
            model_data: Dictionary containing model data from individual model file
            
        Returns:
            DemandPredictor instance
        """
        # Create a temporary predictor instance
        predictor = DemandPredictor.__new__(DemandPredictor)
        
        # Set the model and feature columns directly following MLPrediction structure
        predictor.model = model_data['model_object']
        predictor.feature_columns = model_data['feature_columns']
        predictor.model_metadata = {
            'target_formulation': model_data.get('target_formulation', 'unknown'),
            'target_transformation': model_data.get('target_transformation', 'unknown'),
            'model_name': model_data.get('model_name', 'unknown'),
            'scaler_name': model_data.get('scaler_name', 'unknown'),
            'performance_metrics': model_data.get('performance_metrics', {}),
            'training_configuration': model_data.get('training_configuration', {}),
            'saved_timestamp': model_data.get('saved_timestamp', 'unknown')
        }
        predictor.model_path = "individual_model"  # Placeholder
        
        return predictor


def main():
    parser = argparse.ArgumentParser(
        description="Validate Demand Prediction Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--grid-size",
        type=float,
        default=1.0,
        help="Grid size in kilometers for validation"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model (uses default if not specified)"
    )
    
    parser.add_argument(
        "--validation-data",
        type=str,
        default="data/validation/ved_processed_with_grids.parquet",
        help="Path to validation data"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="validation_results",
        help="Output directory for validation results"
    )
    
    parser.add_argument(
        "--sensitivity-analysis",
        action="store_true",
        help="Run grid size sensitivity analysis"
    )
    
    parser.add_argument(
        "--validate-all-models",
        action="store_true",
        help="Validate all models in the models directory"
    )
    
    parser.add_argument(
        "--models-dir",
        type=str,
        default="results/models",
        help="Directory containing model PKL files"
    )
    
    parser.add_argument(
        "--grid-sizes",
        nargs="+",
        type=float,
        default=[0.5, 0.75, 1.0, 1.5, 2.0],  # Match training scales exactly
        help="Grid sizes to test in sensitivity analysis"
    )
    
    parser.add_argument(
        "--target-filter",
        type=str,
        default=None,
        help="Filter models by target formulation (e.g., 'demand_score_balanced')"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of models to validate (for testing)"
    )
    
    args = parser.parse_args()
    
    # Create validator
    validator = DemandModelValidator(
        validation_data_path=args.validation_data,
        output_dir=args.output_dir,
        models_dir=args.models_dir
    )
    
    try:
        if args.validate_all_models:
            # Validate all models in models directory
            print(f"🔬 Validating ALL models with {args.grid_size}km grid size...")
            if args.target_filter:
                print(f"🎯 Filtering for target: {args.target_filter}")
            
            results = validator.validate_all_models(
                grid_size_km=args.grid_size,
                target_filter=args.target_filter,
                limit=args.limit
            )
            
            print(f"\n✅ All models validation complete!")
            print(f"📊 Validated {len(results)} model combinations")
            print(f"📁 Results saved to: {args.output_dir}")
            
        elif args.sensitivity_analysis:
            # Run sensitivity analysis
            print(f"🔬 Running grid size sensitivity analysis...")
            print(f"📊 Testing grid sizes: {args.grid_sizes}")
            
            results = validator.sensitivity_analysis(
                grid_sizes=args.grid_sizes,
                model_path=args.model_path
            )
            
            print(f"\n✅ Sensitivity analysis complete!")
            print(f"📁 Results saved to: {args.output_dir}")
            
        else:
            # Run single validation
            print(f"🔬 Validating model with {args.grid_size}km grid size...")
            
            result = validator.validate_model(
                grid_size_km=args.grid_size,
                model_path=args.model_path
            )
            
            print(f"\n✅ Validation complete!")
            print(f"📊 Spearman Correlation: {result['metrics']['spearman_correlation']:.4f}")
            print(f"📊 R² Score: {result['metrics']['r2_score']:.4f}")
            print(f"📁 Results saved to: {args.output_dir}")
            
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
