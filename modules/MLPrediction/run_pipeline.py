#!/usr/bin/env python3
"""
Main Runner for the EV Charging Demand Prediction Pipeline
=========================================================

This script provides a command-line interface to run the complete EV charging
demand prediction pipeline. It handles data loading, feature engineering,
model training, and evaluation, generating a comprehensive report.

Usage:
------
.. code-block:: bash

    # Run the pipeline with default settings
    python -m modules.MLPrediction.run_pipeline

    # Run with a specific data file and output directory
    python -m modules.MLPrediction.run_pipeline --data-path /path/to/data.csv --output-dir /path/to/results

    # See all available options
    python -m modules.MLPrediction.run_pipeline --help

"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import pickle

# Add project root to Python path to allow relative imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from modules.MLPrediction.demand_prediction_pipeline import EVDemandPredictionPipeline
# Simple model saving - no complex registry needed
from modules.utils.log_configs import setup_logging

def _save_models_with_sample_indicator(pipeline, target_column, sample_size, output_dir):
    """
    Save models with consistent naming that includes sample size indicator.
    
    Args:
        pipeline: The trained pipeline
        target_column: Name of the target column
        sample_size: Size of the sample used for training
        output_dir: Output directory for saving models
    """
    logger = logging.getLogger(__name__)
    
    # Create models directory
    models_dir = Path("results/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Get training results from pipeline
    training_results = getattr(pipeline, 'training_results', {})
    
    if not training_results:
        logger.warning("No training results found to save")
        return
    
    models_saved = 0
    
    # Save all trained models with consistent naming
    for target_name, target_results in training_results.items():
        for model_config, model_details in target_results.items():
            if model_details.get('model_object') is not None:
                # Extract model information
                target_transformation = model_details.get('target_transformation', 'original')
                model_name = model_details.get('model_name', model_details.get('algorithm', 'Unknown'))
                scaler_name = model_details.get('scaler', 'unknown')
                
                # Create descriptive filename: target_formulation_target_transformation_scaler_modelname_sampleSize.pkl
                filename = f"{target_column}_{target_transformation}_{scaler_name}_{model_name}_sample{sample_size}.pkl"
                model_path = models_dir / filename
                
                # Create comprehensive model package
                model_package = {
                    'model_object': model_details.get('model_object'),
                    'target_formulation': target_column,
                    'target_transformation': target_transformation,
                    'model_name': model_name,
                    'scaler_name': scaler_name,
                    'sample_size': sample_size,
                    'feature_columns': model_details.get('feature_names', []),
                    'performance_metrics': {
                        'r2': model_details.get('test_r2', model_details.get('cv_r2_mean', 0)),
                        'rmse': model_details.get('test_rmse', model_details.get('cv_rmse_mean', 0)),
                        'mae': model_details.get('test_mae', model_details.get('cv_mae_mean', 0)),
                        'pearson_correlation': model_details.get('pearson_correlation', model_details.get('cv_pearson_mean', 0)),
                        'spearman_correlation': model_details.get('spearman_correlation', 0),
                        'cv_mean': model_details.get('cv_mean', model_details.get('cv_r2_mean', 0)),
                        'cv_std': model_details.get('cv_std', model_details.get('cv_r2_std', 0))
                    },
                    'training_configuration': {
                        'random_state': pipeline.random_state,
                        'sample_size': sample_size,
                        'target_column': target_column,
                        'optimization_applied': model_details.get('optimization_applied', False)
                    },
                    'saved_timestamp': pd.Timestamp.now().isoformat()
                }
                
                # Save model
                try:
                    with open(model_path, 'wb') as f:
                        pickle.dump(model_package, f)
                    models_saved += 1
                    logger.info(f"✅ Saved: {filename} (R²={model_details.get('test_r2', model_details.get('cv_r2_mean', 0)):.4f})")
                except Exception as e:
                    logger.error(f"❌ Failed to save {filename}: {e}")
    
    logger.info(f"📊 Saved {models_saved} models with sample size indicator to {models_dir}")
    
    # Also save a summary file
    summary_file = models_dir / f"model_summary_sample{sample_size}.json"
    summary_data = {
        'target_column': target_column,
        'sample_size': sample_size,
        'models_saved': models_saved,
        'models_directory': str(models_dir),
        'naming_convention': 'target_formulation_target_transformation_scaler_modelname_sampleSize.pkl',
        'saved_timestamp': pd.Timestamp.now().isoformat()
    }
    
    try:
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        logger.info(f"📋 Model summary saved to: {summary_file}")
    except Exception as e:
        logger.warning(f"Failed to save model summary: {e}")

def main():
    """Main function to orchestrate the pipeline execution."""
    
    parser = argparse.ArgumentParser(
        description="Run the EV Charging Demand Prediction Pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="results/shenzhen_urbanev_training_data.csv",
        help="Path to the training data CSV file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ml_pipeline_output",
        help="Directory to save pipeline outputs (reports, models, etc.)."
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="demand_score_balanced",
        help="Name of the target column in the dataset. For multiple variations, use: demand_score_hours_only, demand_score_kwh25_hrs75, demand_score_balanced, demand_score_kwh75_hrs25, demand_score_kwh_only."
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--skip-synthetic-data",
        action="store_true",
        default=False,
        help="Skip synthetic data generation (CTGAN, SMOTE, etc.) for faster execution."
    )
    parser.add_argument(
        "--run-all-variations",
        action="store_true",
        default=False,
        help="Run the pipeline for all 'demand_score_*' variations found in the data."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Randomly sample this many rows for faster testing (useful for large datasets). Use all data if not specified."
    )
    parser.add_argument(
        "--skip-spatial-features",
        action="store_true",
        default=False,
        help="Skip spatial feature generation (useful for large datasets or when coordinates cause issues)."
    )
    parser.add_argument(
        "--disable-gpu",
        action="store_true",
        default=False,
        help="Disable GPU acceleration (use CPU-only computation)."
    )
    parser.add_argument(
        "--skip-feature-selection",
        action="store_true", 
        default=False,
        help="Skip feature selection entirely (much faster for large datasets)."
    )
    parser.add_argument(
        "--disable-feature-caching",
        action="store_true",
        default=False,
        help="Disable feature engineering caching (uses more computation but less disk space)."
    )
    parser.add_argument(
        "--optimize-for-multiple-targets",
        action="store_true",
        default=True,
        help="Optimize pipeline by performing feature selection once for all target variations (faster and scientifically robust)."
    )
    parser.add_argument(
        "--skip-rfecv-validation",
        action="store_true",
        default=False,
        help="Skip RFECV validation for faster execution (saves significant time on large datasets)."
    )
    parser.add_argument(
        "--enable-hyperparameter-tuning",
        action="store_true",
        default=False,
        help="Enable automatic hyperparameter tuning for the best performing model using Optuna."
    )
    parser.add_argument(
        "--hyperparameter-trials",
        type=int,
        default=50,
        help="Number of Optuna trials for hyperparameter optimization (default: 50)."
    )
    parser.add_argument(
        "--limit-target-transformations",
        action="store_true",
        default=False,
        help="Limit target transformations to only the most effective ones (original, log) for faster training."
    )
    parser.add_argument(
        "--pre-selected-features",
        type=str,
        default=None,
        help="Path to JSON file containing pre-selected features for reproducible runs."
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        default=False,
        help="Force retraining even if models already exist (overwrites existing models)."
    )
    parser.add_argument(
        "--run-standalone-hyperparameter-tuning",
        action="store_true",
        default=False,
        help="Run standalone hyperparameter tuning after pipeline completion (separate from integrated tuning)."
    )
    parser.add_argument(
        "--hyperparameter-algorithms",
        nargs="+",
        default=["random_forest", "xgboost", "lightgbm"],
        help="Algorithms to optimize in standalone hyperparameter tuning."
    )
    parser.add_argument(
        "--standalone-hyperparameter-trials",
        type=int,
        default=200,
        help="Number of trials for standalone hyperparameter tuning (default: 200)."
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file_name = f"demand_prediction_{'all_variations' if args.run_all_variations else args.target_column}.log"
    setup_logging(log_file_name=log_file_name)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 STARTING EV DEMAND PREDICTION PIPELINE 🚀")
    logger.info(f"Configuration:")
    logger.info(f"  - Data Path: {args.data_path}")
    logger.info(f"  - Output Directory: {args.output_dir}")
    logger.info(f"  - Target Column: {'ALL' if args.run_all_variations else args.target_column}")
    logger.info(f"  - Synthetic Data: {'Disabled' if args.skip_synthetic_data else 'Enabled'}")
    logger.info(f"  - Sample Size: {'All data' if args.sample_size is None else f'{args.sample_size:,} samples'}")
    logger.info(f"  - Spatial Features: {'Disabled' if args.skip_spatial_features else 'Enabled'}")
    logger.info(f"  - GPU Acceleration: {'Disabled' if args.disable_gpu else 'Auto-detect'}")
    logger.info(f"  - Feature Selection: {'Disabled' if args.skip_feature_selection else 'Auto (fast for large datasets)'}")
    logger.info(f"  - Feature Caching: {'Disabled' if args.disable_feature_caching else 'Enabled'}")
    logger.info(f"  - Multi-Target Optimization: {'Enabled' if args.optimize_for_multiple_targets else 'Disabled'}")
    logger.info(f"  - Hyperparameter Tuning: {'Enabled' if args.enable_hyperparameter_tuning else 'Disabled'}")
    if args.enable_hyperparameter_tuning:
        logger.info(f"  - Optuna Trials: {args.hyperparameter_trials}")
    logger.info(f"  - Target Transformations: {'Limited (log only)' if args.limit_target_transformations else 'All transformations'}")
    logger.info("-" * 50)
    
    try:
        if args.run_all_variations:
            run_for_all_variations(args)
        else:
            run_single_pipeline(args)
            
    except FileNotFoundError as e:
        logger.error(f"Error: Input data file not found at '{args.data_path}'. Please check the path.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during pipeline execution: {e}", exc_info=True)
        sys.exit(1)

def run_single_pipeline(args: argparse.Namespace):
    """Runs the pipeline for a single target column."""
    logger = logging.getLogger(__name__)
    
    pipeline = EVDemandPredictionPipeline(
        random_state=args.random_state,
        enable_synthetic_data=not args.skip_synthetic_data,
        use_gpu=not args.disable_gpu,
        skip_feature_selection=args.skip_feature_selection,
        enable_feature_caching=not args.disable_feature_caching,
        limit_transformations=args.limit_target_transformations,
        pre_selected_features_path=args.pre_selected_features
    )
    
    # Handle sampling if requested
    if args.sample_size is not None:
        logger.info(f"🎯 Sampling {args.sample_size:,} rows from the dataset for faster testing")
        data = pd.read_csv(args.data_path)
        if len(data) > args.sample_size:
            data = data.sample(n=args.sample_size, random_state=args.random_state)
            logger.info(f"✅ Sampled data: {len(data):,} rows")
            
            # Save sampled data temporarily
            temp_data_path = f"temp_sampled_data_{args.sample_size}.csv"
            data.to_csv(temp_data_path, index=False)
            data_path = temp_data_path
        else:
            logger.info(f"Dataset ({len(data):,} rows) is smaller than sample size, using full dataset")
            data_path = args.data_path
    else:
        data_path = args.data_path
    
    results = pipeline.run_complete_pipeline(
        data_path=data_path,
        target_column=args.target_column,
        output_dir=args.output_dir,
        skip_spatial_features=args.skip_spatial_features,
        enable_hyperparameter_tuning=args.enable_hyperparameter_tuning,
        hyperparameter_trials=args.hyperparameter_trials,
        sample_size=args.sample_size
    )
    
    # Note: Models are already saved in the pipeline with proper naming
    
    # Clean up temporary file if created
    if args.sample_size is not None and 'temp_sampled_data' in data_path:
        import os
        os.remove(data_path)
        logger.info("🧹 Cleaned up temporary sampled data file")
    
    # Display summary results
    evaluation = results['evaluation']
    logger.info("\n" + "="*60)
    logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    logger.info(f"📊 Best Configuration: {evaluation['overall_best']['configuration']}")
    logger.info(f"📈 Best Performance Score: {evaluation['overall_best']['score']:.4f}")
    logger.info(f"⭐ Performance Assessment: {evaluation['performance_assessment']}")
    logger.info(f"📋 Report saved to: {results['report_path']}")
    logger.info(f"🔬 Analysis artifacts saved to: {results['analysis_artifacts_path']}")
    
    # Run standalone hyperparameter tuning if requested
    if args.run_standalone_hyperparameter_tuning:
        logger.info("\n" + "="*60)
        logger.info("🔬 RUNNING STANDALONE HYPERPARAMETER TUNING")
        logger.info("="*60)
        
        try:
            from modules.MLPrediction.hyperparameter_tuning import HyperparameterTuner
            
            # Prepare data for hyperparameter tuning
            data = pd.read_csv(data_path)
            engineered_data = pipeline.prepare_features(data, skip_spatial_features=args.skip_spatial_features)
            
            # Prepare features and target
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
            
            X = engineered_data[feature_columns]
            y = engineered_data[args.target_column]
            
            # Initialize standalone hyperparameter tuner
            standalone_tuner = HyperparameterTuner(
                random_state=args.random_state,
                use_gpu=not args.disable_gpu,
                optimization_strategy="optuna"
            )
            
            # Run optimization for specified algorithms
            logger.info(f"Optimizing algorithms: {args.hyperparameter_algorithms}")
            standalone_results = {}
            
            for algorithm in args.hyperparameter_algorithms:
                logger.info(f"\n🔬 Optimizing {algorithm}...")
                try:
                    result = standalone_tuner.optimize_with_optuna(
                        X, y, algorithm, n_trials=args.standalone_hyperparameter_trials
                    )
                    if result:
                        standalone_results[algorithm] = result
                        logger.info(f"✅ {algorithm} optimization complete: R²={result['best_score']:.4f}")
                    else:
                        logger.warning(f"❌ {algorithm} optimization failed")
                except Exception as e:
                    logger.error(f"❌ {algorithm} optimization failed: {e}")
            
            if standalone_results:
                logger.info(f"\n🎉 Standalone hyperparameter tuning completed for {len(standalone_results)} algorithms!")
                
                # Save standalone results
                standalone_output_dir = Path(args.output_dir) / "standalone_hyperparameter_tuning"
                standalone_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save results
                import json
                results_file = standalone_output_dir / "standalone_optimization_results.json"
                with open(results_file, 'w') as f:
                    json.dump(standalone_results, f, indent=2, default=str)
                
                logger.info(f"💾 Standalone results saved to: {standalone_output_dir}")
                
                # Display summary
                logger.info("\n📊 Standalone Optimization Summary:")
                for algorithm, result in standalone_results.items():
                    logger.info(f"  {algorithm}: R²={result['best_score']:.4f}, RMSE={result['rmse']:.4f}")
            else:
                logger.warning("No algorithms were successfully optimized")
                
        except Exception as e:
            logger.error(f"Standalone hyperparameter tuning failed: {e}")
            logger.info("Continuing with pipeline completion...")

def run_for_all_variations(args: argparse.Namespace):
    """Runs the pipeline for all discovered demand_score variations with optimized caching."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading data from {args.data_path} to identify all target variations...")
    try:
        data = pd.read_csv(args.data_path)
        
        # Handle sampling if requested  
        if args.sample_size is not None:
            logger.info(f"🎯 Sampling {args.sample_size:,} rows from the dataset for faster testing")
            if len(data) > args.sample_size:
                data = data.sample(n=args.sample_size, random_state=args.random_state)
                logger.info(f"✅ Sampled data: {len(data):,} rows")
            else:
                logger.info(f"Dataset ({len(data):,} rows) is smaller than sample size, using full dataset")
                
    except FileNotFoundError:
        logger.error(f"Data file not found at {args.data_path}")
        raise

    demand_score_columns = sorted([col for col in data.columns if col.startswith('demand_score')])
    
    if not demand_score_columns:
        logger.error("No 'demand_score' variations found in the dataset. Cannot proceed with --run-all-variations.")
        return

    logger.info(f"Found {len(demand_score_columns)} demand score variations: {demand_score_columns}")
    
    # 🚀 OPTIMIZATION: Do feature engineering ONCE for all variations
    logger.info("\n" + "="*70)
    logger.info("🔧 PERFORMING FEATURE ENGINEERING (ONCE FOR ALL VARIATIONS)")
    logger.info("="*70)
    
    pipeline = EVDemandPredictionPipeline(
        random_state=args.random_state,
        enable_synthetic_data=not args.skip_synthetic_data,
        use_gpu=not args.disable_gpu,
        skip_feature_selection=args.skip_feature_selection,
        enable_feature_caching=not args.disable_feature_caching,
        limit_transformations=args.limit_target_transformations,
        pre_selected_features_path=args.pre_selected_features
    )
    
    # Set pipeline attributes for model saving
    pipeline.sample_size = args.sample_size
    
    # Load and engineer features once
    logger.info("Loading and engineering features...")
    engineered_data = pipeline.prepare_features(data, skip_spatial_features=args.skip_spatial_features)
    logger.info(f"✅ Feature engineering complete: {len(data.columns)} → {len(engineered_data.columns)} features")
    
    # Prepare feature columns (same logic as in pipeline)
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
    
    X = engineered_data[feature_columns]
    logger.info(f"Prepared features: {len(X)} samples with {len(feature_columns)} features")
    
    # 🚀 OPTIMIZATION: Perform feature selection ONCE for all target variations
    logger.info("\n" + "="*70)
    logger.info("🔬 PERFORMING OPTIMIZED FEATURE SELECTION (ONCE FOR ALL TARGETS)")
    logger.info("="*70)
    
    # Use the first target to determine optimal features (scientifically valid approach)
    primary_target = demand_score_columns[0]
    y_primary = engineered_data[primary_target]
    
    # Perform feature selection once using the primary target
    if args.skip_feature_selection:
        logger.info("⏭️  Skipping feature selection (disabled for speed)")
        # Use pre-selected features if available, otherwise use all features
        if args.pre_selected_features and Path(args.pre_selected_features).exists():
            logger.info("✅ Using pre-selected features from file")
            # Load pre-selected features and filter the data
            import json
            with open(args.pre_selected_features, 'r') as f:
                feature_data = json.load(f)
            selected_features = feature_data['selected_features']
            # Filter X to only include selected features
            available_features = [f for f in selected_features if f in X.columns]
            missing_features = [f for f in selected_features if f not in X.columns]
            if missing_features:
                logger.warning(f"⚠️  Missing features in data: {missing_features}")
            selected_features = available_features
            X_selected = X[selected_features]
            logger.info(f"🎯 Using {len(selected_features)} pre-selected features")
        else:
            logger.info("⚠️  No pre-selected features file found, using all features")
            selected_features = feature_columns
            X_selected = X
    else:
        logger.info(f"📊 Large dataset detected - using fast feature selection with primary target: {primary_target}")
        from modules.MLPrediction.model_training import ModelTrainer
        temp_trainer = ModelTrainer(
            random_state=args.random_state,
            skip_feature_selection=False
        )
        # Use smart feature selection based on dataset size (don't just use all features!)
        # For multiple targets, we want a robust set but not ALL features to prevent overfitting
        if len(feature_columns) <= 20:
            optimal_feature_count = len(feature_columns)  # Use all for small sets
        elif len(feature_columns) <= 50:
            optimal_feature_count = max(15, min(30, len(feature_columns) // 2))  # Conservative selection
        else:
            optimal_feature_count = max(25, min(40, len(feature_columns) // 2))  # Large dataset selection
            
        logger.info(f"🎯 Smart feature selection: {len(feature_columns)} → {optimal_feature_count} features")
        selected_features = temp_trainer.fast_feature_selection(X, y_primary, max_features=optimal_feature_count)
        X_selected = X[selected_features]
        
        # 🔬 RFECV VALIDATION - Validate feature selection with different sample sizes
        if not args.skip_rfecv_validation:
            logger.info("\n" + "="*70)
            logger.info("🔬 PERFORMING RFECV VALIDATION FOR SCIENTIFIC RIGOR")
            logger.info("="*70)
            rfecv_results = temp_trainer.validate_feature_selection_with_rfecv(
                X, y_primary, 
                sample_sizes=[10000, 20000, 30000] if len(X) > 30000 else [5000, 10000, 15000],
                n_samples=3
            )
            logger.info("✅ RFECV validation complete - results saved for paper documentation")
        else:
            logger.info("⏭️  Skipping RFECV validation for faster execution")
            rfecv_results = None
    
    logger.info(f"✅ Feature selection complete: {len(feature_columns)} → {len(selected_features)} features")
    logger.info(f"🎯 Selected features will be used for ALL target variations (scientifically robust)")
    
    # Simple model saving directory
    models_dir = Path("results/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    logger.info("🏗️  Using simple directory-based model saving")
    
    # Track total models saved across all variations
    total_models_saved = 0
    
    # Now run training for each target variation using the SAME selected features
    all_results = {}
    for target in demand_score_columns:
        logger.info("\n" + "="*70)
        logger.info(f"🚀 TRAINING MODELS FOR TARGET: {target} 🚀")
        logger.info("="*70)

        # Set the current target for model saving
        pipeline.original_target_column = target

        variation_output_dir = Path(args.output_dir) / target
        variation_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if target not in engineered_data.columns:
                logger.error(f"Target column '{target}' not found in the data.")
                all_results[target] = {"error": f"Target column '{target}' not found"}
                continue
            
            y = engineered_data[target]
            
            # Check for existing models (continuation support)
            existing_models = []
            for pkl_file in models_dir.glob(f"{target}_*.pkl"):
                existing_models.append(pkl_file.stem)
            
            if existing_models and not args.force_retrain:
                logger.info(f"🔄 Found {len(existing_models)} existing models for {target}")
                
                # Load existing training results if available
                # Structure: {target_transformation: {config_name: result}}
                training_results = {}
                loaded_models = 0
                failed_models = []
                
                for model_name in existing_models:
                    model_path = models_dir / f"{model_name}.pkl"
                    try:
                        with open(model_path, 'rb') as f:
                            model_data = pickle.load(f)
                        
                        # Check if this is a GradientBoostingRegressor with version issues
                        if 'gradient_boosting' in model_name and 'model_object' in model_data:
                            try:
                                # Try to access the model to see if it's broken
                                model = model_data['model_object']
                                if hasattr(model, 'predict'):
                                    # Test if the model works by trying a simple operation
                                    import numpy as np
                                    dummy_X = np.array([[0.0] * 28])  # 28 features
                                    _ = model.predict(dummy_X)
                            except Exception as model_error:
                                if 'No module named' in str(model_error) or '_loss' in str(model_error):
                                    logger.warning(f"GradientBoostingRegressor model {model_name} has version compatibility issues, will retrain")
                                    failed_models.append(model_name)
                                    continue
                        
                        # Extract target transformation, scaler and algorithm from model data
                        # The model data contains the correct information directly
                        target_transformation = model_data.get('target_transformation', 'original')
                        scaler_name = model_data.get('scaler_name', 'unknown')
                        algorithm_name = model_data.get('model_name', 'unknown')
                        
                        config_name = f"{scaler_name}_{algorithm_name}"
                        target_key = f"target_{target_transformation}"
                        
                        # Initialize target transformation if not exists
                        if target_key not in training_results:
                            training_results[target_key] = {}
                        
                        training_results[target_key][config_name] = {
                            'model': model_data['model_object'],
                            'target_transformation': target_transformation,
                            'model_name': algorithm_name,
                            'algorithm': algorithm_name,  # String name, not object
                            'scaler': scaler_name,
                            'test_r2': model_data['performance_metrics'].get('r2', 0),
                            'test_rmse': model_data['performance_metrics'].get('rmse', 0),
                            'test_mae': model_data['performance_metrics'].get('mae', 0),
                            'pearson_correlation': model_data['performance_metrics'].get('pearson_correlation', 0),
                            'spearman_correlation': model_data['performance_metrics'].get('spearman_correlation', 0),
                            'cv_mean': model_data['performance_metrics'].get('cv_mean', 0),
                            'cv_std': model_data['performance_metrics'].get('cv_std', 0),
                            'model_object': model_data['model_object'],  # Add model_object for compatibility
                            'scaler_object': None,  # Scaler object not saved in individual models
                            'feature_names': model_data.get('feature_columns', []),
                            'feature_importance': None  # Feature importance not saved in individual models
                        }
                        loaded_models += 1
                    except Exception as e:
                        logger.warning(f"Could not load existing model {model_name}: {e}")
                        failed_models.append(model_name)
                
                logger.info(f"✅ Loaded {loaded_models} existing models for {target}")
                if failed_models:
                    logger.info(f"⚠️  {len(failed_models)} models failed to load (likely version mismatch)")
                
                # Check if we need to continue training missing models
                # Calculate expected number of models based on actual configuration
                target_transformations = ['original', 'log'] if args.limit_target_transformations else ['original', 'log', 'sqrt', 'zscore', 'robust']
                scalers = ['standard', 'robust', 'quantile']
                algorithms = ['ridge', 'elastic_net', 'lasso', 'bayesian_ridge', 'random_forest', 'extra_trees', 'gradient_boosting', 'neural_network', 'xgboost', 'lightgbm', 'catboost']
                expected_models = len(target_transformations) * len(scalers) * len(algorithms)
                
                # Count actual loaded models by checking the training_results structure
                actual_loaded_models = sum(len(target_results) for target_results in training_results.values())
                
                if failed_models or actual_loaded_models < expected_models:
                    logger.info(f"🔄 Continuing training for missing/failed models...")
                    logger.info(f"   Expected: {expected_models} models, Loaded: {actual_loaded_models}, Missing: {expected_models - actual_loaded_models}")
                    # Run training with pre-selected features to complete missing models
                    additional_results = pipeline.train_models_with_preselected_features(X_selected, y, selected_features, cv_folds=5)
                    
                    # Merge additional results with existing ones
                    for target_key, target_results in additional_results.items():
                        if target_key not in training_results:
                            training_results[target_key] = {}
                        training_results[target_key].update(target_results)
                    
                    logger.info(f"✅ Completed training for all target variations")
                else:
                    logger.info("⏭️  All models already exist, skipping training")
                
                pipeline.training_results = training_results
            else:
                # Run training with pre-selected features (much faster!)
                logger.info(f"🚀 Training new models for {target}...")
                training_results = pipeline.train_models_with_preselected_features(X_selected, y, selected_features, cv_folds=5)
                pipeline.training_results = training_results
            
            # Results evaluation
            evaluation = pipeline.evaluate_results(training_results)
            
            # Hyperparameter optimization for best model
            if args.enable_hyperparameter_tuning:
                logger.info(f"🔬 Optimizing hyperparameters for the best performing model (target: {target})...")
                optimized_results = pipeline.model_trainer.optimize_best_models_only(
                    training_results, X_selected, y, top_n=1, n_trials=args.hyperparameter_trials
                )
                # Update training results with optimized model
                training_results = optimized_results
                # Re-evaluate with optimized results
                evaluation = pipeline.evaluate_results(training_results)
                logger.info(f"✅ Hyperparameter optimization complete for {target}!")
            else:
                logger.info("⏭️  Skipping hyperparameter optimization (disabled)")
            
            # Generate report
            report_path = pipeline.generate_comprehensive_report(
                training_results, evaluation, str(variation_output_dir)
            )
            
            # Save analysis artifacts
            artifacts_path = pipeline._save_analysis_artifacts(
                training_results, evaluation, str(variation_output_dir)
            )
            
            # Save processed data with enhanced metadata
            processed_data_path = variation_output_dir / "processed_training_data.csv"
            engineered_data.to_csv(processed_data_path, index=False)
            
            # Save comprehensive feature engineering metadata for scientific reproducibility
            feature_metadata = {
                'total_features_before_engineering': len(data.columns),
                'total_features_after_engineering': len(engineered_data.columns),
                'feature_columns_used': selected_features,  # Use the actually selected features
                'excluded_columns': [col for col in engineered_data.columns if col not in selected_features],
                'engineering_timestamp': pd.Timestamp.now().isoformat(),
                'dataset_shape': engineered_data.shape,
                'target_column': target,
                'optimization_applied': True,
                'feature_selection_method': 'mutual_information_with_cleaning',
                'rfecv_validation_performed': not args.skip_rfecv_validation,
                'feature_analysis_directory': 'results/feature_analysis/',
                'paper_ready': True
            }
            
            metadata_path = variation_output_dir / "feature_engineering_metadata.json"
            with open(metadata_path, 'w') as f:
                import json
                json.dump(feature_metadata, f, indent=2)
            logger.info(f"📋 Feature metadata saved to: {metadata_path}")
            
            # Note: Individual models are saved during training for continuation support
            
            # Also save the traditional single model for backward compatibility
            best_model_details = evaluation.get('overall_best', {}).get('details')
            if best_model_details:
                model_package = {
                    'model': best_model_details,
                    'feature_columns': selected_features,
                    'feature_metadata': feature_metadata,
                    'evaluation_results': evaluation,
                    'training_configuration': {
                        'random_state': args.random_state,
                        'skip_feature_selection': args.skip_feature_selection,
                        'enable_synthetic_data': not args.skip_synthetic_data,
                        'target_column': target,
                        'optimization_mode': 'multi_target_optimized'
                    },
                    'pipeline_version': '2.0_optimized_multi_target',
                    'saved_timestamp': pd.Timestamp.now().isoformat()
                }
                model_save_path = variation_output_dir / "global_demand_model.pkl"
                with open(model_save_path, 'wb') as f:
                    pickle.dump(model_package, f)
                logger.info(f"💾 Traditional model package saved to: {model_save_path}")
            
            results = {
                "report_path": str(report_path),
                "analysis_artifacts_path": str(artifacts_path),
                "evaluation": evaluation,
                "best_config": evaluation['overall_best']['configuration'],
                "best_score": evaluation['overall_best']['score'],
                "assessment": evaluation['performance_assessment']
            }
            
            all_results[target] = results
            
            logger.info(f"✅ PIPELINE FOR {target} COMPLETED SUCCESSFULLY!")
            logger.info(f"   - Best Score: {results['best_score']:.4f}")
            logger.info(f"   - Best Config: {results['best_config']}")
            logger.info(f"   - Outputs saved to: {variation_output_dir}")

        except Exception as e:
            logger.error(f"Pipeline run for target '{target}' failed: {e}", exc_info=True)
            all_results[target] = {"error": str(e)}

    # Final summary
    logger.info("\n" + "="*70)
    logger.info("🏆 ALL PIPELINE RUNS COMPLETED 🏆")
    logger.info("="*70)
    logger.info("Summary of best performance across all demand variations:")
    
    summary_data = []
    for target, result in all_results.items():
        if "error" in result:
            summary_data.append({
                "Target Variation": target,
                "Best Score": "ERROR",
                "Best Configuration": result['error'],
                "Assessment": "N/A"
            })
        else:
            summary_data.append({
                "Target Variation": target,
                "Best Score": f"{result['best_score']:.4f}",
                "Best Configuration": result['best_config'],
                "Assessment": result['assessment']
            })
    
    summary_df = pd.DataFrame(summary_data)
    logger.info("\n" + summary_df.to_string(index=False))
    
    # Note: Individual models are saved during training for continuation support
    
    # Generate research documentation
    # Research documentation generation removed for simplification

if __name__ == "__main__":
    main()
