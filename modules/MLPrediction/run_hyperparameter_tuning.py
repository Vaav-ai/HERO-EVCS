#!/usr/bin/env python3
"""
Standalone Hyperparameter Tuning Runner

This script provides a command-line interface to run hyperparameter optimization
separately from the main pipeline. It can be used to optimize models after the
pipeline has identified the best performing algorithms.

Usage:
------
.. code-block:: bash

    # Run hyperparameter tuning on a dataset
    python -m modules.MLPrediction.run_hyperparameter_tuning --data-path data.csv --target-column demand_score_balanced

    # Run with specific algorithms and more trials
    python -m modules.MLPrediction.run_hyperparameter_tuning --data-path data.csv --algorithms random_forest xgboost --n-trials 200

    # Run with GPU acceleration
    python -m modules.MLPrediction.run_hyperparameter_tuning --data-path data.csv --use-gpu --n-trials 500

"""

import argparse
import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
import pickle
from datetime import datetime

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from modules.MLPrediction.hyperparameter_tuning import HyperparameterTuner
from modules.MLPrediction.model_training import ModelTrainer
from modules.MLPrediction.feature_engineering import UrbanFeatureEngineer
from modules.utils.log_configs import setup_logging

def load_training_results(results_path: str) -> dict:
    """
    Load training results from pipeline output.
    
    Args:
        results_path: Path to the pipeline results directory
        
    Returns:
        Dictionary with training results
    """
    results_file = Path(results_path) / "training_results.json"
    
    if not results_file.exists():
        # Try alternative locations
        alt_paths = [
            Path(results_path) / "analysis_artifacts" / "training_results.json",
            Path(results_path) / "global_demand_model.pkl"
        ]
        
        for alt_path in alt_paths:
            if alt_path.exists():
                if alt_path.suffix == '.pkl':
                    # Load from pickle file
                    import pickle
                    with open(alt_path, 'rb') as f:
                        model_package = pickle.load(f)
                    return model_package.get('training_results', {})
                else:
                    # Load from JSON
                    with open(alt_path, 'r') as f:
                        return json.load(f)
        
        raise FileNotFoundError(f"Training results not found in {results_path}")
    
    with open(results_file, 'r') as f:
        return json.load(f)

def create_dummy_training_results(algorithms: list) -> dict:
    """
    Create dummy training results for testing purposes.
    
    Args:
        algorithms: List of algorithms to include
        
    Returns:
        Dictionary with dummy training results
    """
    dummy_results = {
        'target_dummy': {}
    }
    
    for i, algorithm in enumerate(algorithms):
        # Create realistic dummy scores
        base_score = 0.6 + (i * 0.05)  # Varying base scores
        dummy_results['target_dummy'][f'model_{i+1}'] = {
            'algorithm': algorithm,
            'test_r2': base_score + np.random.normal(0, 0.02),
            'cv_mean': base_score + np.random.normal(0, 0.01),
            'pearson_correlation': base_score + np.random.normal(0, 0.01),
            'test_rmse': 1.0 - base_score + np.random.normal(0, 0.01),
            'test_mae': 0.8 - base_score + np.random.normal(0, 0.01)
        }
    
    return dummy_results

def prepare_features_and_target(data: pd.DataFrame, target_column: str, 
                              feature_selection_file: str = None,
                              skip_spatial_features: bool = True) -> tuple:
    """
    Prepare features and target from the dataset using MLPrediction feature engineering.
    
    Args:
        data: Input DataFrame
        target_column: Name of the target column
        feature_selection_file: Path to JSON file with pre-selected features
        skip_spatial_features: Whether to skip spatial feature generation
        
    Returns:
        Tuple of (X, y, feature_columns)
    """
    logger = logging.getLogger(__name__)
    
    # Check if data already has engineered features
    has_engineered_features = any(col.startswith(('commercial_residential_ratio', 'distance_from_center', 
                                                'economic_activity_index', 'grid_position_')) for col in data.columns)
    
    if has_engineered_features:
        logger.info("✅ Data already contains engineered features, using existing features")
        engineered_data = data
    else:
        logger.info("🔧 Applying MLPrediction feature engineering pipeline...")
        
        # Initialize feature engineer
        feature_engineer = UrbanFeatureEngineer(enable_caching=True, use_parallel=True)
        
        # Apply complete feature engineering pipeline
        engineered_data = feature_engineer.apply_complete_feature_engineering(
            data, skip_spatial_features=skip_spatial_features
        )
        
        logger.info(f"Feature engineering complete: {len(data.columns)} → {len(engineered_data.columns)} features")
    
    # Load pre-selected features if provided
    if feature_selection_file and Path(feature_selection_file).exists():
        logger.info(f"📁 Loading pre-selected features from {feature_selection_file}")
        with open(feature_selection_file, 'r') as f:
            feature_data = json.load(f)
        feature_columns = feature_data.get('selected_features', [])
        logger.info(f"Using {len(feature_columns)} pre-selected features")
    else:
        # Use existing feature preparation logic from MLPrediction
        all_demand_score_cols = [col for col in engineered_data.columns if col.startswith('demand_score')]
        leaky_columns = [
            'num_stations', 'num_chargers', 'total_kwh', 'total_hours',
            'kwh_per_charger', 'hours_per_charger', 'kwh_normalized', 
            'hours_normalized', 'avg_occupancy'
        ]
        leaky_columns.extend(all_demand_score_cols)
        
        base_metadata_cols = [
            'grid_type', 'grid_variation', 'grid_scale', 'grid_scale_x', 'grid_scale_y', 'actual_size_km',
            'size_scale', 'offset_x_km', 'offset_y_km', 'latitude', 'longitude'
        ]
        
        grid_metadata_cols = []
        for col in engineered_data.columns:
            for base_col in base_metadata_cols:
                if col.startswith(base_col):
                    grid_metadata_cols.append(col)
        
        exclude_cols = ['grid_id'] + leaky_columns + list(set(grid_metadata_cols))
        feature_columns = [col for col in engineered_data.columns if col not in exclude_cols and not col.startswith('target_')]
    
    # Prepare X and y
    X = engineered_data[feature_columns]
    y = engineered_data[target_column]
    
    logger.info(f"Prepared features: {len(feature_columns)} features, {len(X)} samples")
    logger.info(f"Target column: {target_column}")
    
    return X, y, feature_columns

def progress_callback(trial_num: int, total_trials: int, score: float):
    """Progress callback for hyperparameter optimization."""
    if trial_num % max(1, total_trials // 10) == 0 or trial_num == total_trials:
        logger = logging.getLogger(__name__)
        logger.info(f"   Trial {trial_num}/{total_trials}: R²={score:.4f}")

def run_hyperparameter_tuning_for_target(data: pd.DataFrame, target_column: str, 
                                        algorithms: list, n_trials: int, 
                                        feature_columns: list, args) -> dict:
    """
    Run hyperparameter tuning for a specific target column.
    
    Args:
        data: Input DataFrame
        target_column: Name of the target column
        algorithms: List of algorithms to optimize
        n_trials: Number of trials per algorithm
        feature_columns: List of feature column names
        args: Command line arguments
        
    Returns:
        Dictionary with optimization results
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"\n🔬 OPTIMIZING TARGET: {target_column}")
    logger.info("=" * 60)
    
    # Prepare features and target for this specific target
    X, y, target_feature_columns = prepare_features_and_target(
        data, target_column, args.skip_spatial_features
    )
    
    # Initialize hyperparameter tuner
    tuner = HyperparameterTuner(
        random_state=args.random_state,
        use_gpu=args.use_gpu,
        optimization_strategy=args.strategy
    )
    
    # Run optimization for this target
    results = {}
    for algorithm in algorithms:
        logger.info(f"\n🔬 Optimizing {algorithm} for {target_column}...")
        try:
            result = tuner.optimize_with_optuna(
                X, y, algorithm, 
                n_trials=n_trials, 
                timeout=args.timeout,
                progress_callback=progress_callback
            )
            if result:
                results[algorithm] = result
                logger.info(f"✅ {algorithm} optimization complete for {target_column}: R²={result['best_score']:.4f}")
            else:
                logger.warning(f"❌ {algorithm} optimization failed for {target_column}")
        except Exception as e:
            logger.error(f"❌ {algorithm} optimization failed for {target_column}: {e}")
    
    # Save results for this target
    if results:
        target_output_dir = Path(args.output_dir) / f"target_{target_column}"
        tuner._save_optimization_results(
            results, 
            target_formulation=target_column,
            feature_columns=target_feature_columns,
            output_dir=str(target_output_dir)
        )
        logger.info(f"💾 Results for {target_column} saved to: {target_output_dir}")
    
    return results

def main():
    """Main function for simple hyperparameter tuning."""
    
    parser = argparse.ArgumentParser(
        description="Simple Hyperparameter Tuning for EV Demand Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the training data CSV file."
    )
    
    # Model configuration (either specify manually or use AutoML)
    parser.add_argument(
        "--target-formulation",
        type=str,
        default="demand_score_balanced",
        help="Target formulation (e.g., demand_score_balanced)."
    )
    parser.add_argument(
        "--target-transformation",
        type=str,
        default="original",
        choices=["original", "log", "quantile"],
        help="Target transformation to use."
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default="none",
        choices=["none", "standard", "robust", "quantile"],
        help="Feature scaler to use."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="xgboost",
        choices=["random_forest", "xgboost", "lightgbm", "gradient_boosting", "extra_trees", "neural_network", "ridge", "elastic_net", "lasso", "bayesian_ridge"],
        help="Algorithm to optimize."
    )
    
    # AutoML option
    parser.add_argument(
        "--auto-select",
        type=str,
        default=None,
        help="Path to results JSON file to automatically select best model for tuning."
    )
    parser.add_argument(
        "--auto-metric",
        type=str,
        default="r2_score",
        choices=["r2_score", "rmse", "mae"],
        help="Metric to use for auto-selection ranking."
    )
    
    # Feature selection
    parser.add_argument(
        "--feature-selection-file",
        type=str,
        default=None,
        help="Path to JSON file with pre-selected features."
    )
    
    # Optimization parameters
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of optimization trials."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Maximum time in seconds for optimization."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Randomly sample this many rows for faster testing. If not specified, uses full dataset."
    )
    parser.add_argument(
        "--use-full-dataset",
        action="store_true",
        help="Use the full dataset for hyperparameter tuning (overrides sample-size)."
    )
    
    # Output and other options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hyperparameter_optimization",
        help="Directory to save optimization results."
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable GPU acceleration when available."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="optuna",
        choices=["optuna"],
        help="Optimization strategy to use."
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file_name = f"hyperparameter_tuning_{args.target_formulation}.log"
    setup_logging(log_file_name=log_file_name)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 STARTING SIMPLE HYPERPARAMETER TUNING")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  - Data Path: {args.data_path}")
    logger.info(f"  - Target Formulation: {args.target_formulation}")
    logger.info(f"  - Target Transformation: {args.target_transformation}")
    logger.info(f"  - Scaler: {args.scaler}")
    logger.info(f"  - Algorithm: {args.algorithm}")
    logger.info(f"  - Trials: {args.n_trials}")
    logger.info(f"  - Random State: {args.random_state}")
    logger.info(f"  - GPU Acceleration: {'Enabled' if args.use_gpu else 'Disabled'}")
    logger.info(f"  - Strategy: {args.strategy}")
    if args.auto_select:
        logger.info(f"  - Auto-Select: {args.auto_select}")
    if args.feature_selection_file:
        logger.info(f"  - Feature Selection: {args.feature_selection_file}")
    if args.timeout:
        logger.info(f"  - Timeout: {args.timeout} seconds")
    logger.info("-" * 60)
    
    try:
        # Load data
        logger.info("📊 Loading data...")
        data = pd.read_csv(args.data_path)
        logger.info(f"   Loaded {len(data):,} rows, {len(data.columns)} columns")
        
        # Handle sampling if requested
        if args.use_full_dataset:
            logger.info("🚀 Using FULL DATASET for hyperparameter tuning")
            logger.info(f"   Full dataset: {len(data):,} rows")
        elif args.sample_size and len(data) > args.sample_size:
            logger.info(f"🎯 Sampling {args.sample_size:,} rows for faster testing")
            data = data.sample(n=args.sample_size, random_state=args.random_state)
            logger.info(f"   Sampled data: {len(data):,} rows")
        else:
            logger.info(f"📊 Using full dataset: {len(data):,} rows")
        
        # Determine model configuration
        if args.auto_select:
            logger.info("🤖 Auto-selecting best model from results...")
            tuner = HyperparameterTuner(
                random_state=args.random_state,
                use_gpu=args.use_gpu,
                optimization_strategy=args.strategy
            )
            best_model_config = tuner.auto_select_best_model(
                args.auto_select, 
                target_formulation=args.target_formulation,
                metric=args.auto_metric
            )
            
            if not best_model_config:
                logger.error("❌ Failed to auto-select model")
                sys.exit(1)
            
            target_formulation = best_model_config['target_formulation']
            target_transformation = best_model_config['target_transformation']
            scaler_name = best_model_config['scaler_name']
            algorithm = best_model_config['algorithm']
            
            logger.info(f"✅ Auto-selected model: {algorithm} with {target_transformation} transformation and {scaler_name} scaler")
        else:
            target_formulation = args.target_formulation
            target_transformation = args.target_transformation
            scaler_name = args.scaler
            algorithm = args.algorithm
        
        # Check if target column exists
        if target_formulation not in data.columns:
            logger.error(f"❌ Target column '{target_formulation}' not found in dataset")
            logger.info(f"Available columns: {list(data.columns)}")
            sys.exit(1)
        
        # Prepare features and target
        logger.info("🔧 Preparing features and target...")
        X, y, feature_columns = prepare_features_and_target(
            data, target_formulation, 
            feature_selection_file=args.feature_selection_file,
            skip_spatial_features=True
        )
        
        # Initialize hyperparameter tuner
        logger.info("🔧 Initializing hyperparameter tuner...")
        tuner = HyperparameterTuner(
            random_state=args.random_state,
            use_gpu=args.use_gpu,
            optimization_strategy=args.strategy
        )
        
        # Run optimization
        logger.info("🔬 Starting hyperparameter optimization...")
        start_time = time.time()
        
        result = tuner.tune_specific_model(
            X, y, target_formulation, target_transformation, 
            scaler_name, algorithm, feature_columns,
            n_trials=args.n_trials, timeout=args.timeout
        )
        
        total_time = time.time() - start_time
        
        if result:
            logger.info("\n" + "=" * 80)
            logger.info("🎉 HYPERPARAMETER OPTIMIZATION COMPLETED!")
            logger.info("=" * 80)
            logger.info(f"⏱️  Total time: {total_time:.2f} seconds")
            logger.info(f"🎯 Target: {target_formulation}")
            logger.info(f"🔧 Transformation: {target_transformation}")
            logger.info(f"📏 Scaler: {scaler_name}")
            logger.info(f"🤖 Algorithm: {algorithm}")
            logger.info(f"📊 Features: {len(feature_columns)}")
            logger.info(f"🎯 Best Score (R²): {result['best_score']:.4f}")
            logger.info(f"📈 Final R²: {result['r2_score']:.4f}")
            logger.info(f"📉 RMSE: {result['rmse']:.4f}")
            logger.info(f"📉 MAE: {result['mae']:.4f}")
            logger.info(f"🔄 Trials: {result['n_trials_completed']}")
            logger.info(f"⏱️  Optimization Time: {result['optimization_time']:.2f}s")
            
            # Save results
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save optimized model following MLPrediction format
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{target_formulation}_{target_transformation}_{scaler_name}_{algorithm}_optimized_{timestamp}.pkl"
            model_path = output_path / model_filename
            
            # Create model package following MLPrediction structure
            model_package = {
                'model_object': result['model'],
                'target_formulation': target_formulation,
                'target_transformation': target_transformation,
                'scaler_name': scaler_name,
                'feature_columns': feature_columns,
                'scaler_object': result.get('scaler_object'),
                'performance_metrics': {
                    'r2': result['r2_score'],
                    'rmse': result['rmse'],
                    'mae': result['mae'],
                    'cv_mean': result['best_score'],
                    'cv_std': 0
                },
                'training_configuration': {
                    'random_state': args.random_state,
                    'optimization_applied': True,
                    'hyperparameter_tuning': True,
                    'best_params': result['best_params'],
                    'n_trials': result['n_trials'],
                    'optimization_time': result['optimization_time']
                },
                'saved_timestamp': datetime.now().isoformat()
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_package, f)
            
            # Save optimization summary
            summary_file = output_path / f"optimization_summary_{timestamp}.json"
            summary_data = {
                'target_formulation': target_formulation,
                'target_transformation': target_transformation,
                'scaler_name': scaler_name,
                'algorithm': algorithm,
                'feature_columns': feature_columns,
                'best_params': result['best_params'],
                'best_score': result['best_score'],
                'r2_score': result['r2_score'],
                'rmse': result['rmse'],
                'mae': result['mae'],
                'n_trials': result['n_trials'],
                'n_trials_completed': result['n_trials_completed'],
                'optimization_time': result['optimization_time'],
                'total_time': total_time,
                'random_state': args.random_state,
                'gpu_enabled': args.use_gpu,
                'strategy': args.strategy,
                'model_file': str(model_path),
                'saved_timestamp': datetime.now().isoformat()
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
            
            logger.info(f"\n💾 Results saved to: {output_path}")
            logger.info(f"🤖 Optimized model: {model_path}")
            logger.info(f"📋 Summary: {summary_file}")
            
        else:
            logger.error("❌ Hyperparameter optimization failed")
            sys.exit(1)
        
        logger.info(f"\n✅ Hyperparameter tuning completed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"❌ Error: Input data file not found at '{args.data_path}'. Please check the path.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred during hyperparameter tuning: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
