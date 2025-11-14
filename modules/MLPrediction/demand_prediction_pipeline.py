
"""
EV Demand Prediction Pipeline

This module provides an integrated pipeline for EV charging demand prediction,
encompassing feature engineering, model training, and evaluation.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pickle
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import json

from .feature_engineering import UrbanFeatureEngineer
from .target_transformation import DemandTargetTransformer
from .model_training import ModelTrainer
from .hyperparameter_tuning import HyperparameterTuner
# Simple model saving - no complex registry needed

logger = logging.getLogger(__name__)

class EVDemandPredictionPipeline:
    """
    Orchestrates the complete ML pipeline for EV charging demand prediction.
    
    This class integrates all steps from data loading and feature engineering
    to model training and comprehensive evaluation.
    """
    
    def __init__(self, random_state: int = 42, enable_synthetic_data: bool = True, use_gpu: bool = True, 
                 skip_feature_selection: bool = False, enable_feature_caching: bool = True,
                 skip_synthetic_data: bool = False, pre_selected_features_path: str = None,
                 limit_transformations: bool = False):
        """
        Initialize the pipeline.
        
        Args:
            random_state: Random seed for reproducibility
            enable_synthetic_data: Whether to use synthetic data augmentation
            use_gpu: Whether to enable GPU acceleration when available
            skip_feature_selection: Whether to skip feature selection (faster for large datasets)
            enable_feature_caching: Whether to enable feature engineering caching
            skip_synthetic_data: Whether to skip synthetic data generation
            pre_selected_features_path: Path to JSON file with pre-selected features for reproducibility
            limit_transformations: Whether to limit target transformations to only the most effective ones
        """
        self.random_state = random_state
        self.enable_synthetic_data = enable_synthetic_data and not skip_synthetic_data
        self.use_gpu = use_gpu
        self.skip_feature_selection = skip_feature_selection
        self.enable_feature_caching = enable_feature_caching
        self.pre_selected_features_path = pre_selected_features_path
        self.pre_selected_features = None
        self.limit_transformations = limit_transformations
        
        # Set global random seeds for reproducibility
        np.random.seed(random_state)
        # Ensure consistent random state across all components
        
        # Initialize component modules
        self.feature_engineer = UrbanFeatureEngineer(enable_caching=enable_feature_caching, use_parallel=True)
        self.target_transformer = DemandTargetTransformer(random_state=random_state)
        self.model_trainer = ModelTrainer(random_state=random_state, enable_synthetic_data=self.enable_synthetic_data, 
                                        use_gpu=use_gpu, skip_feature_selection=skip_feature_selection)
        
        # Pipeline results
        self.processed_data = None
        self.training_results = {}
        self.best_models = {}
        self.original_target_column = None
        self.sample_size = None
        
        # Load pre-selected features if provided
        if self.pre_selected_features_path:
            self._load_pre_selected_features()
    
    def _load_pre_selected_features(self):
        """Load pre-selected features from JSON file for reproducible runs."""
        try:
            import json
            with open(self.pre_selected_features_path, 'r') as f:
                feature_data = json.load(f)
            
            self.pre_selected_features = feature_data['selected_features']
            logger.info(f"✅ Loaded {len(self.pre_selected_features)} pre-selected features from {self.pre_selected_features_path}")
            logger.info(f"Pre-selected features: {self.pre_selected_features[:5]}{'...' if len(self.pre_selected_features) > 5 else ''}")
            
        except Exception as e:
            logger.error(f"Failed to load pre-selected features from {self.pre_selected_features_path}: {e}")
            raise
        
    def load_training_data(self, data_path: str) -> pd.DataFrame:
        """
        Load training data from CSV file with enhanced dataset support.
        
        Args:
            data_path: Path to training data CSV
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading training data from: {data_path}")
        
        data_file = Path(data_path)
        if not data_file.exists():
            raise FileNotFoundError(f"Training data not found: {data_path}")
        
        data = pd.read_csv(data_path)
        logger.info(f"Loaded data: {len(data)} samples, {len(data.columns)} features")
        
        # Check if this is an enhanced gridding dataset
        if 'grid_type' in data.columns:
            logger.info("📊 Enhanced gridding dataset detected!")
            grid_composition = data['grid_type'].value_counts()
            logger.info("Dataset composition by grid type:")
            for grid_type, count in grid_composition.items():
                percentage = (count / len(data)) * 100
                logger.info(f"   {grid_type}: {count} samples ({percentage:.1f}%)")
                
            # Log scale diversity if available
            if 'grid_scale' in data.columns:
                unique_scales = sorted(data['grid_scale'].unique())
                logger.info(f"Grid scales present: {unique_scales}")
        
        return data
    
    def prepare_features(self, data: pd.DataFrame, skip_spatial_features: bool = False) -> pd.DataFrame:
        """
        Apply comprehensive feature engineering to the data.
        
        Args:
            data: Raw data with OSM features
            skip_spatial_features: If True, skip spatial feature generation
            
        Returns:
            Data with engineered features
        """
        logger.info("=== FEATURE ENGINEERING PHASE ===")
        
        # Apply complete feature engineering pipeline
        engineered_data = self.feature_engineer.apply_complete_feature_engineering(data, skip_spatial_features=skip_spatial_features)
        
        logger.info(f"Feature engineering complete: {len(data.columns)} → {len(engineered_data.columns)} features")
        return engineered_data
    
    def train_models_with_cv(self, X: pd.DataFrame, y: pd.Series, feature_columns: List[str], 
                           cv_folds: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Train models using K-Fold Cross-Validation for robust evaluation.
        """
        logger.info("=== MODEL TRAINING WITH K-FOLD CROSS-VALIDATION ===")
        logger.info(f"Using {cv_folds}-fold cross-validation for robust model evaluation")
        
        # Initialize K-Fold
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Fit the target transformer on the full training data
        self.target_transformer.fit(y)
        
        # Transform targets
        targets_df = self.target_transformer.transform(y, include_original=True, limit_transformations=self.limit_transformations)
        target_columns = list(targets_df.columns)
        
        logger.info(f"Training with {len(feature_columns)} features and {len(target_columns)} target formulations")
        
        all_results = {}
        best_overall_score = -np.inf
        best_overall_config = None
        
        for target_col in target_columns:
            logger.info(f"\n--- Cross-validating models for {target_col} ---")
            
            y_transformed = targets_df[target_col]
            
            # Extract target transformation name from column name
            target_transformation = target_col.replace('target_', '') if target_col.startswith('target_') else 'original'
            
            # Create individual model saver callback
            individual_model_saver = self._create_individual_model_saver(target_transformation, feature_columns)
            
            # Use the new CV-based training method with individual model saving
            target_results = self.model_trainer.train_target_with_cv(
                X, y_transformed, feature_columns, cv_folds=cv_folds, target_transformation=target_transformation,
                save_individual_model_callback=individual_model_saver
            )
            
            if target_results:
                all_results[target_col] = target_results
                
                best_config, best_result = self.model_trainer.find_best_model(target_results)
                best_score = self.model_trainer._calculate_performance_score(best_result)
                if best_score > best_overall_score:
                    best_overall_score = best_score
                    best_overall_config = f"{target_col}_{best_config}"
                logger.info(f"Best for {target_col}: {best_config} (CV Score: {best_score:.3f})")
        
        logger.info(f"\n🏆 OVERALL BEST: {best_overall_config} (CV Score: {best_overall_score:.3f})")
        return all_results
    
    def train_models_with_preselected_features(self, X: pd.DataFrame, y: pd.Series, 
                                             selected_features: List[str], cv_folds: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Train models using pre-selected features (optimized for multiple target variations).
        
        This method skips feature selection and directly trains models with the provided features,
        making it much faster when training multiple targets with the same feature set.
        
        Args:
            X: Pre-filtered feature DataFrame with selected features only
            y: Target Series
            selected_features: List of selected feature names
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with training results for all configurations
        """
        logger.info("=== OPTIMIZED MODEL TRAINING WITH PRE-SELECTED FEATURES ===")
        logger.info(f"Using {cv_folds}-fold cross-validation with {len(selected_features)} pre-selected features")
        
        # Initialize K-Fold
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Fit the target transformer on the full training data
        self.target_transformer.fit(y)
        
        # Transform targets
        targets_df = self.target_transformer.transform(y, include_original=True, limit_transformations=self.limit_transformations)
        target_columns = list(targets_df.columns)
        
        logger.info(f"Training with {len(selected_features)} features and {len(target_columns)} target formulations")
        
        all_results = {}
        best_overall_score = -np.inf
        best_overall_config = None
        
        for target_col in target_columns:
            logger.info(f"\n--- Cross-validating models for {target_col} ---")
            
            y_transformed = targets_df[target_col]
            
            # Extract target transformation name from column name
            target_transformation = target_col.replace('target_', '') if target_col.startswith('target_') else 'original'
            
            # Create individual model saver callback
            individual_model_saver = self._create_individual_model_saver(target_transformation, selected_features)
            
            # Use the optimized training method that skips feature selection with individual model saving
            target_results = self.model_trainer.train_target_with_preselected_features(
                X, y_transformed, selected_features, cv_folds=cv_folds, target_transformation=target_transformation,
                save_individual_model_callback=individual_model_saver
            )
            
            if target_results:
                all_results[target_col] = target_results
                
                best_config, best_result = self.model_trainer.find_best_model(target_results)
                best_score = self.model_trainer._calculate_performance_score(best_result)
                if best_score > best_overall_score:
                    best_overall_score = best_score
                    best_overall_config = f"{target_col}_{best_config}"
                logger.info(f"Best for {target_col}: {best_config} (CV Score: {best_score:.3f})")
        
        logger.info(f"\n🏆 OVERALL BEST: {best_overall_config} (CV Score: {best_overall_score:.3f})")
        return all_results
    
    def train_missing_models_only(self, X: pd.DataFrame, y: pd.Series, selected_features: List[str], 
                                 missing_configs: set, cv_folds: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Train only the missing model configurations for efficient continuation.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            selected_features: List of selected feature names
            missing_configs: Set of (target_key, config_name) tuples that need training
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with training results for missing configurations only
        """
        logger.info(f"🎯 Training {len(missing_configs)} missing model configurations...")
        
        # Initialize K-Fold
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Fit the target transformer on the full training data
        self.target_transformer.fit(y)
        
        # Transform targets
        targets_df = self.target_transformer.transform(y, include_original=True, limit_transformations=self.limit_transformations)
        
        # Group missing configurations by target transformation
        missing_by_target = {}
        for target_key, config_name in missing_configs:
            if target_key not in missing_by_target:
                missing_by_target[target_key] = []
            missing_by_target[target_key].append(config_name)
        
        all_results = {}
        
        for target_key, configs_to_train in missing_by_target.items():
            logger.info(f"Training {len(configs_to_train)} configurations for {target_key}...")
            
            # Get the target transformation name
            target_transformation = target_key.replace('target_', '') if target_key.startswith('target_') else 'original'
            
            # Get the transformed target
            if target_key in targets_df.columns:
                y_transformed = targets_df[target_key]
            else:
                logger.warning(f"Target {target_key} not found in transformed targets, skipping")
                continue
            
            # Create individual model saver callback
            individual_model_saver = self._create_individual_model_saver(target_transformation, selected_features)
            
            # Train only the missing configurations for this target
            target_results = self.model_trainer.train_specific_configurations(
                X, y_transformed, selected_features, configs_to_train, cv_folds=cv_folds, 
                target_transformation=target_transformation, save_individual_model_callback=individual_model_saver
            )
            
            if target_results:
                all_results[target_key] = target_results
                logger.info(f"✅ Trained {len(target_results)} configurations for {target_key}")
            else:
                logger.warning(f"❌ No configurations were successfully trained for {target_key}")
        
        logger.info(f"🎉 Completed training {len(missing_configs)} missing configurations")
        return all_results
    
    def train_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
                     feature_columns: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Train models for all target formulations using a data leakage-proof workflow.
        This method is kept for backwards compatibility but now uses the CV approach.
        """
        logger.info("=== MODEL TRAINING PHASE ===")
        
        # Combine train and test for CV approach
        X_combined = pd.concat([X_train, X_test], ignore_index=True)
        y_combined = pd.concat([y_train, y_test], ignore_index=True)
        
        # Use the new CV-based training
        return self.train_models_with_cv(X_combined, y_combined, feature_columns)
    
    def evaluate_results(self, training_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Comprehensive evaluation of training results.
        
        Args:
            training_results: Results from model training
            
        Returns:
            Dictionary with evaluation metrics and analysis
        """
        logger.info("=== RESULTS EVALUATION PHASE ===")
        
        evaluation = {
            'target_formulations_tested': len(training_results),
            'total_configurations': sum(len(results) for results in training_results.values()),
            'best_results_by_target': {},
            'overall_best': None,
            'performance_summary': []
        }
        
        best_overall_score = -np.inf
        best_overall_config = None
        best_overall_result = None
        
        # Analyze results for each target formulation
        for target_name, target_results in training_results.items():
            if target_results:
                # Generate performance report for this target
                performance_df = self.model_trainer.generate_performance_report(target_results)
                
                # Get best configuration
                best_config = performance_df.iloc[0]['configuration']
                best_score = performance_df.iloc[0]['performance_score']
                best_result = target_results[best_config]
                
                evaluation['best_results_by_target'][target_name] = {
                    'configuration': best_config,
                    'score': best_score,
                    'test_r2': best_result.get('test_r2', best_result.get('cv_r2_mean', 0)),
                    'pearson_correlation': best_result.get('pearson_correlation', best_result.get('cv_pearson_mean', 0)),
                    'spearman_correlation': best_result.get('spearman_correlation', 0)
                }
                
                # Track overall best
                if best_score > best_overall_score:
                    best_overall_score = best_score
                    best_overall_config = f"{target_name}_{best_config}"
                    best_overall_result = best_result
                
                # Add to performance summary
                evaluation['performance_summary'].append({
                    'target': target_name,
                    'best_config': best_config,
                    'score': best_score,
                    'r2': best_result.get('test_r2', best_result.get('cv_r2_mean', 0)),
                    'correlation': best_result.get('pearson_correlation', best_result.get('cv_pearson_mean', 0))
                })
        
        evaluation['overall_best'] = {
            'configuration': best_overall_config,
            'score': best_overall_score,
            'details': best_overall_result
        }
        
        # Assess performance
        if best_overall_score > 0.8:
            evaluation['performance_assessment'] = "Excellent"
            evaluation['recommendation'] = "Model performance is outstanding and suitable for production deployment."
        elif best_overall_score > 0.6:
            evaluation['performance_assessment'] = "Very Good"
            evaluation['recommendation'] = "Model performance is strong and can be considered for production."
        elif best_overall_score > 0.4:
            evaluation['performance_assessment'] = "Good"
            evaluation['recommendation'] = "Model shows solid predictive power."
        elif best_overall_score > 0.2:
            evaluation['performance_assessment'] = "Moderate"
            evaluation['recommendation'] = "Model has some predictive power but may need further improvements."
        else:
            evaluation['performance_assessment'] = "Needs Improvement"
            evaluation['recommendation'] = "Model performance is low; significant improvements are recommended."
        
        logger.info(f"Performance Assessment: {evaluation['performance_assessment']}")
        
        return evaluation
    
    def generate_comprehensive_report(self, training_results: Dict[str, Dict[str, Any]], 
                                    evaluation: Dict[str, Any],
                                    output_dir: str = "results") -> str:
        """
        Generate comprehensive report of pipeline results.
        
        Args:
            training_results: Results from model training
            evaluation: Evaluation metrics
            output_dir: Directory to save report
            
        Returns:
            Path to generated report file
        """
        logger.info("Generating comprehensive pipeline report...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        report_lines = []
        report_lines.append("# EV Charging Demand Prediction - Pipeline Results")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("## Executive Summary")
        report_lines.append(f"- **Target Formulations**: {evaluation['target_formulations_tested']}")
        report_lines.append(f"- **Total Configurations**: {evaluation['total_configurations']}")
        report_lines.append(f"- **Best Configuration**: {evaluation['overall_best']['configuration']}")
        report_lines.append(f"- **Best Score**: {evaluation['overall_best']['score']:.4f}")
        report_lines.append(f"- **Performance Assessment**: {evaluation['performance_assessment']}")
        report_lines.append(f"- **Recommendation**: {evaluation['recommendation']}")
        report_lines.append("")
        
        # Best Results by Target
        report_lines.append("## Best Results by Target Formulation")
        report_lines.append("| Target | Configuration | Score | R² | Correlation |")
        report_lines.append("|--------|---------------|-------|-----|-------------|")
        
        for target, results in evaluation['best_results_by_target'].items():
            target_clean = target.replace('target_', '')
            report_lines.append(
                f"| {target_clean} | {results['configuration']} | "
                f"{results['score']:.4f} | {results['test_r2']:.4f} | "
                f"{results['pearson_correlation']:.4f} |"
            )
        
        report_lines.append("")
        
        # Key Features of the Pipeline
        report_lines.append("## Key Pipeline Features")
        report_lines.append("1. **Urban Feature Engineering**: Domain-specific features based on urban planning theory.")
        report_lines.append("2. **Multiple Target Formulations**: Tested multiple target transformations for robustness.")
        report_lines.append("3. **Comprehensive Model Training**: Compared multiple algorithms and scaling approaches.")
        report_lines.append("4. **Robust Evaluation**: Used cross-validation and a composite performance score for model selection.")
        report_lines.append("5. **Ensemble Modeling**: A voting regressor combines top models to improve stability and performance.")
        report_lines.append("")
        
        # Technical Details
        if evaluation['overall_best']['details']:
            best_details = evaluation['overall_best']['details']
            report_lines.append("## Best Model Technical Details")
            report_lines.append(f"- **Algorithm**: {best_details.get('algorithm', 'N/A')}")
            report_lines.append(f"- **Scaler**: {best_details.get('scaler', 'N/A')}")
            report_lines.append(f"- **Test R²**: {best_details.get('test_r2', best_details.get('cv_r2_mean', 0)):.4f}")
            report_lines.append(f"- **RMSE**: {best_details.get('test_rmse', best_details.get('cv_rmse_mean', 0)):.4f}")
            report_lines.append(f"- **MAE**: {best_details.get('test_mae', best_details.get('cv_mae_mean', 0)):.4f}")
            report_lines.append(f"- **Pearson Correlation**: {best_details.get('pearson_correlation', best_details.get('cv_pearson_mean', 0)):.4f}")
            report_lines.append(f"- **Spearman Correlation**: {best_details.get('spearman_correlation', 0):.4f}")
            report_lines.append("")
        
        # Next Steps
        report_lines.append("## Recommended Next Steps")
        report_lines.append("1. Deploy the best model configuration for demand prediction.")
        report_lines.append("2. Analyze feature importance for urban planning insights.")
        report_lines.append("3. Validate model performance on independent datasets from other cities.")
        report_lines.append("4. Integrate the demand prediction model into the RL optimization framework.")
        
        # Save report
        report_path = output_path / "pipeline_results_report.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Save detailed data for each target formulation
        for target_name, results in training_results.items():
            target_output_dir = output_path / target_name
            target_output_dir.mkdir(exist_ok=True)
            if results:
                self.model_trainer.save_training_results(results, str(target_output_dir))
        
        logger.info(f"Comprehensive report saved to: {report_path}")
        return str(report_path)
    
    def _save_analysis_artifacts(self, training_results: Dict[str, Dict[str, Any]],
                                 evaluation: Dict[str, Any],
                                 output_dir: str) -> Path:
        """
        Saves comprehensive results as CSV files for academic papers or detailed analysis.
        """
        artifacts_dir = Path(output_dir) / "analysis_artifacts"
        artifacts_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"💾 Saving analysis artifacts as CSVs to {artifacts_dir}...")

        # 1. Save overall performance summary
        if 'performance_summary' in evaluation:
            perf_summary_df = pd.DataFrame(evaluation['performance_summary'])
            perf_summary_df.to_csv(artifacts_dir / "performance_summary.csv", index=False)

        # 2. Save detailed metrics for all models and target transformations
        all_metrics = []
        for target_name, target_results in training_results.items():
            for model_name, model_data in target_results.items():
                metrics = {
                    'target_transformation': target_name,
                    'model_configuration': model_name,
                    'cv_r2_mean': model_data.get('cv_r2_mean'),
                    'cv_r2_std': model_data.get('cv_r2_std'),
                    'cv_rmse_mean': model_data.get('cv_rmse_mean'),
                    'cv_rmse_std': model_data.get('cv_rmse_std'),
                    'score': model_data.get('score'),
                }
                all_metrics.append(metrics)
        
        if all_metrics:
            all_metrics_df = pd.DataFrame(all_metrics)
            all_metrics_df.sort_values(by='score', ascending=False, inplace=True)
            all_metrics_df.to_csv(artifacts_dir / "all_model_metrics.csv", index=False)

        # 3. Save feature importance for the best model
        try:
            best_model_details = evaluation.get('overall_best', {}).get('details', {})
            if best_model_details:
                target_name = best_model_details.get('target_transformation')
                model_name = best_model_details.get('model_name')
                
                if target_name and model_name:
                    model_results = training_results.get(target_name, {}).get(model_name, {})
                    if 'feature_importance' in model_results:
                        df_importance = pd.DataFrame(model_results['feature_importance'])
                        if not df_importance.empty:
                            df_importance.to_csv(artifacts_dir / "best_model_feature_importance.csv", index=False)
        except Exception as e:
            logger.warning(f"Could not save feature importance: {e}")

        # 4. Save training_results.json for hyperparameter tuning continuation
        try:
            # Convert training results to serializable format for hyperparameter tuning
            serializable_training_results = {}
            for target_name, target_results in training_results.items():
                serializable_training_results[target_name] = {}
                for model_name, model_data in target_results.items():
                    if model_data.get('model_object') is not None:
                        # Create serializable version (exclude model objects)
                        serializable_model = {
                            'algorithm': model_data.get('model_name', 'unknown'),
                            'scaler': model_data.get('scaler', 'unknown'),
                            'target_transformation': model_data.get('target_transformation', 'original'),
                            'test_r2': model_data.get('test_r2', model_data.get('cv_r2_mean', 0)),
                            'test_rmse': model_data.get('test_rmse', model_data.get('cv_rmse_mean', 0)),
                            'test_mae': model_data.get('test_mae', model_data.get('cv_mae_mean', 0)),
                            'pearson_correlation': model_data.get('pearson_correlation', model_data.get('cv_pearson_mean', 0)),
                            'spearman_correlation': model_data.get('spearman_correlation', 0),
                            'cv_mean': model_data.get('cv_mean', model_data.get('cv_r2_mean', 0)),
                            'cv_std': model_data.get('cv_std', model_data.get('cv_r2_std', 0)),
                            'score': model_data.get('score', 0),
                            'optimization_applied': model_data.get('optimization_applied', False)
                        }
                        serializable_training_results[target_name][model_name] = serializable_model
            
            # Save training results for hyperparameter tuning
            training_results_file = artifacts_dir / "training_results.json"
            with open(training_results_file, 'w') as f:
                json.dump(serializable_training_results, f, indent=2, default=str)
            
            logger.info(f"💾 Training results saved for hyperparameter tuning: {training_results_file}")
            
        except Exception as e:
            logger.warning(f"Could not save training results for hyperparameter tuning: {e}")

        logger.info("✅ Analysis artifacts saved successfully.")
        return artifacts_dir
    
    def _save_individual_models_during_training(self, training_results: Dict[str, Dict[str, Any]], 
                                              target_column: str, feature_columns: List[str], 
                                              output_dir: str, sample_size: int = None) -> None:
        """
        Save individual models during training for continuation and hyperparameter tuning.
        
        This ensures models are available immediately after training, not just at the end.
        """
        logger.info("💾 Saving individual models during training for continuation...")
        
        # Create models directory
        models_dir = Path("results/models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        models_saved = 0
        
        for target_name, target_results in training_results.items():
            for model_config, model_details in target_results.items():
                if model_details.get('model_object') is not None:
                    # Extract model information
                    target_transformation = model_details.get('target_transformation', 'original')
                    model_name = model_details.get('model_name', 'Unknown')
                    scaler_name = model_details.get('scaler', 'unknown')
                    
                    # Create descriptive filename: target_formulation_target_transformation_scaler_modelname[_sampleSize].pkl
                    sample_suffix = f"_sample{sample_size}" if sample_size is not None else ""
                    filename = f"{target_column}_{target_transformation}_{scaler_name}_{model_name}{sample_suffix}.pkl"
                    model_path = models_dir / filename
                    
                    # Create comprehensive model package
                    model_package = {
                        'model_object': model_details.get('model_object'),
                        'target_formulation': target_column,
                        'target_transformation': target_transformation,
                        'model_name': model_name,
                        'scaler_name': scaler_name,
                        'feature_columns': feature_columns,
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
                            'random_state': self.random_state,
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
        
        logger.info(f"📊 Saved {models_saved} individual models to {models_dir}")
    
    def _create_individual_model_saver(self, target_transformation: str, feature_columns: List[str]) -> callable:
        """
        Create a callback function for saving individual models during training.
        
        Args:
            target_transformation: Name of the target transformation
            feature_columns: List of feature column names
            
        Returns:
            Callback function that saves individual models
        """
        def save_individual_model(config_name: str, model_result: Dict[str, Any], transformation: str):
            """Save a single model immediately after training."""
            try:
                # Create models directory
                models_dir = Path("results/models")
                models_dir.mkdir(parents=True, exist_ok=True)
                
                # Extract model information
                model_name = model_result.get('model_name', 'Unknown')
                scaler_name = model_result.get('scaler', 'unknown')
                
                # Create descriptive filename: target_formulation_target_transformation_scaler_modelname.pkl
                filename = f"{self.original_target_column}_{transformation}_{scaler_name}_{model_name}.pkl"
                model_path = models_dir / filename
                
                # Create comprehensive model package
                model_package = {
                    'model_object': model_result.get('model_object'),
                    'target_formulation': self.original_target_column,
                    'target_transformation': transformation,
                    'model_name': model_name,
                    'scaler_name': scaler_name,
                    'feature_columns': feature_columns,
                    'performance_metrics': {
                        'r2': model_result.get('test_r2', model_result.get('cv_r2_mean', 0)),
                        'rmse': model_result.get('test_rmse', model_result.get('cv_rmse_mean', 0)),
                        'mae': model_result.get('test_mae', model_result.get('cv_mae_mean', 0)),
                        'pearson_correlation': model_result.get('pearson_correlation', model_result.get('cv_pearson_mean', 0)),
                        'spearman_correlation': model_result.get('spearman_correlation', 0),
                        'cv_mean': model_result.get('cv_mean', model_result.get('cv_r2_mean', 0)),
                        'cv_std': model_result.get('cv_std', model_result.get('cv_r2_std', 0))
                    },
                    'training_configuration': {
                        'random_state': self.random_state,
                        'sample_size': self.sample_size,
                        'target_column': self.original_target_column,
                        'optimization_applied': model_result.get('optimization_applied', False)
                    },
                    'saved_timestamp': pd.Timestamp.now().isoformat()
                }
                
                # Save model
                with open(model_path, 'wb') as f:
                    pickle.dump(model_package, f)
                
                logger.info(f"✅ Saved: {filename} (R²={model_result.get('test_r2', model_result.get('cv_r2_mean', 0)):.4f})")
                
            except Exception as e:
                logger.error(f"❌ Failed to save individual model {config_name}: {e}")
        
        return save_individual_model
    
    def run_complete_pipeline(self, data_path: str, target_column: str = 'demand_score',
                            output_dir: str = "results", skip_spatial_features: bool = False,
                            enable_hyperparameter_tuning: bool = True, hyperparameter_trials: int = 50,
                            sample_size: int = None) -> Dict[str, Any]:
        """
        Run the complete integrated pipeline.
        
        Args:
            data_path: Path to training data
            target_column: Name of original target column
            output_dir: Directory for outputs
            skip_spatial_features: If True, skip spatial feature generation
            
        Returns:
            Dictionary with pipeline results
        """
        logger.info("🚀 STARTING EV DEMAND PREDICTION PIPELINE 🚀")
        
        # Store sample size and original target column for model saving
        self.sample_size = sample_size
        self.original_target_column = target_column
        
        try:
            # Step 1: Load data
            data = self.load_training_data(data_path)
            
            # Step 2: Feature engineering
            data_with_features = self.prepare_features(data, skip_spatial_features=skip_spatial_features)
            
            # --- Step 3: Prepare features and target (prevent data leakage) ---
            logger.info("Preparing features and target for cross-validation...")
            
            # Separate features (X) and the original target (y)
            # --- CRITICAL: Exclude ALL target-leaking columns from features ---
            
            # Dynamically find all demand score columns to prevent leakage across variations
            all_demand_score_cols = [col for col in data_with_features.columns if col.startswith('demand_score')]
            
            leaky_columns = [
                'num_stations', 'num_chargers', 'total_kwh', 'total_hours',
                'kwh_per_charger', 'hours_per_charger', 'kwh_normalized', 
                'hours_normalized', 'avg_occupancy'
            ]
            leaky_columns.extend(all_demand_score_cols)
            
            # Base metadata columns that should not be used as features
            base_metadata_cols = [
                'grid_type', 'grid_variation', 'grid_scale', 'grid_scale_x', 'grid_scale_y', 'actual_size_km',
                'size_scale', 'offset_x_km', 'offset_y_km', 'latitude', 'longitude'
            ]
            
            # Find all actual metadata columns by checking for prefixes (handles _x, _y suffixes)
            grid_metadata_cols = []
            for col in data_with_features.columns:
                for base_col in base_metadata_cols:
                    if col.startswith(base_col):
                        grid_metadata_cols.append(col)
            
            exclude_cols = ['grid_id'] + leaky_columns + list(set(grid_metadata_cols))
            all_feature_columns = [col for col in data_with_features.columns if col not in exclude_cols and not col.startswith('target_')]
            
            # Use pre-selected features if available, otherwise use all features
            if self.pre_selected_features:
                # Filter pre-selected features to only include those available in the data
                available_pre_selected = [f for f in self.pre_selected_features if f in all_feature_columns]
                missing_features = [f for f in self.pre_selected_features if f not in all_feature_columns]
                
                if missing_features:
                    logger.warning(f"Some pre-selected features not found in data: {missing_features}")
                
                if available_pre_selected:
                    feature_columns = available_pre_selected
                    logger.info(f"Using {len(feature_columns)} pre-selected features (from {len(self.pre_selected_features)} total)")
                else:
                    logger.warning("No pre-selected features available in data, using all features")
                    feature_columns = all_feature_columns
            else:
                feature_columns = all_feature_columns
            
            if target_column not in data_with_features.columns:
                raise ValueError(f"Target column '{target_column}' not found in the data.")
            
            X = data_with_features[feature_columns]
            y = data_with_features[target_column]
            
            logger.info(f"Training data: {len(X)} samples with {len(feature_columns)} features")

            # Store processed data
            self.processed_data = data_with_features
            
            # Step 4: Model training with K-Fold Cross-Validation (robust evaluation)
            # Check for existing models (continuation support)
            # Look in the main results/models directory, not the target-specific one
            models_dir = Path("results/models")
            models_dir.mkdir(parents=True, exist_ok=True)
            
            existing_models = []
            for pkl_file in models_dir.glob(f"{target_column}_*.pkl"):
                existing_models.append(pkl_file.stem)
            
            if existing_models:
                logger.info(f"🔄 Found {len(existing_models)} existing models for {target_column}")
                
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
                
                logger.info(f"✅ Loaded {loaded_models} existing models for {target_column}")
                if failed_models:
                    logger.info(f"⚠️  {len(failed_models)} models failed to load (likely version mismatch)")
                
                # Check which specific models are missing and train only those
                target_transformations = ['original', 'log'] if self.limit_transformations else ['original', 'log', 'sqrt', 'zscore', 'robust']
                scalers = ['standard', 'robust', 'quantile']
                algorithms = ['ridge', 'elastic_net', 'lasso', 'bayesian_ridge', 'random_forest', 'extra_trees', 'gradient_boosting', 'neural_network', 'xgboost', 'lightgbm', 'catboost']
                
                # Create a set of all expected model configurations
                expected_configs = set()
                for target_transformation in target_transformations:
                    for scaler in scalers:
                        for algorithm in algorithms:
                            config_name = f"{scaler}_{algorithm}"
                            target_key = f"target_{target_transformation}"
                            expected_configs.add((target_key, config_name))
                
                # Create a set of actually loaded model configurations
                loaded_configs = set()
                for target_key, target_results in training_results.items():
                    for config_name in target_results.keys():
                        loaded_configs.add((target_key, config_name))
                
                # Find missing configurations
                missing_configs = expected_configs - loaded_configs
                
                # Log detailed model status
                logger.info(f"📊 Model Status Summary:")
                logger.info(f"   - Expected configurations: {len(expected_configs)}")
                logger.info(f"   - Successfully loaded: {len(loaded_configs)}")
                logger.info(f"   - Failed to load: {len(failed_models)}")
                logger.info(f"   - Missing configurations: {len(missing_configs)}")
                
                if missing_configs or failed_models:
                    logger.info(f"🔄 Training {len(missing_configs)} missing model configurations...")
                    
                    # Train only the missing configurations
                    missing_results = self.train_missing_models_only(
                        X, y, feature_columns, missing_configs, cv_folds=5
                    )
                    
                    # Merge missing results with existing ones
                    for target_key, target_results in missing_results.items():
                        if target_key not in training_results:
                            training_results[target_key] = {}
                        training_results[target_key].update(target_results)
                    
                    # Count final models
                    final_loaded_models = sum(len(target_results) for target_results in training_results.values())
                    logger.info(f"✅ Completed training: {final_loaded_models} total models now available")
                else:
                    logger.info("⏭️  All models already exist, skipping training")
            else:
                # No existing models, train from scratch
                logger.info(f"🚀 Training new models for {target_column}...")
                training_results = self.train_models_with_cv(X, y, feature_columns, cv_folds=5)
            
            self.training_results = training_results
            
            # Step 5: Results evaluation
            evaluation = self.evaluate_results(training_results)
            
            # Step 5.5: Hyperparameter optimization for best model
            if enable_hyperparameter_tuning:
                logger.info("=== HYPERPARAMETER OPTIMIZATION PHASE ===")
                if evaluation['overall_best']['details']:
                    logger.info("🔬 Optimizing hyperparameters for the best performing model...")
                    
                    # Use the new standalone hyperparameter tuner
                    hyperparameter_tuner = HyperparameterTuner(
                        random_state=self.random_state,
                        use_gpu=self.use_gpu,
                        optimization_strategy="optuna"
                    )
                    
                    optimized_results = hyperparameter_tuner.optimize_top_models(
                        training_results, X, y, top_n=1, n_trials=hyperparameter_trials
                    )
                    
                    if optimized_results:
                        # Update training results with optimized models
                        for algorithm, result in optimized_results.items():
                            # Find the corresponding model in training results and update it
                            for target_name, target_results in training_results.items():
                                for model_name, model_data in target_results.items():
                                    if model_data.get('algorithm') == algorithm:
                                        model_data['model_object'] = result['model']
                                        model_data['optimization_applied'] = True
                                        model_data['optuna_params'] = result['best_params']
                                        model_data['optuna_score'] = result['best_score']
                                        break
                        
                        # Re-evaluate with optimized results
                        evaluation = self.evaluate_results(training_results)
                        logger.info("✅ Hyperparameter optimization complete!")
                        logger.info(f"   Optimized {len(optimized_results)} models")
                    else:
                        logger.warning("Hyperparameter optimization failed, using original models")
                else:
                    logger.warning("No best model found for hyperparameter optimization")
            else:
                logger.info("⏭️  Skipping hyperparameter optimization (disabled)")
            
            # Step 6: Generate report
            report_path = self.generate_comprehensive_report(
                training_results, evaluation, output_dir
            )
            
            # Step 6.5: Save comprehensive analysis artifacts
            artifacts_path = self._save_analysis_artifacts(
                training_results, evaluation, output_dir
            )
            
            # Save processed data with enhanced metadata
            output_path = Path(output_dir)
            processed_data_path = output_path / "processed_training_data.csv"
            data_with_features.to_csv(processed_data_path, index=False)
            
            # Save feature engineering metadata for scientific reproducibility
            feature_metadata = {
                'total_features_before_engineering': len(data.columns),
                'total_features_after_engineering': len(data_with_features.columns),
                'feature_columns_used': feature_columns,
                'excluded_columns': [col for col in data_with_features.columns if col not in feature_columns],
                'engineering_timestamp': pd.Timestamp.now().isoformat(),
                'dataset_shape': data_with_features.shape,
                'target_column': target_column
            }
            
            metadata_path = output_path / "feature_engineering_metadata.json"
            with open(metadata_path, 'w') as f:
                import json
                json.dump(feature_metadata, f, indent=2)
            logger.info(f"📋 Feature metadata saved to: {metadata_path}")
            
            # --- Save the best model with comprehensive package ---
            best_model_details = evaluation['overall_best']['details']
            if best_model_details:
                # Enhanced model package for scientific reproducibility
                model_package = {
                    'model': best_model_details,
                    'feature_columns': feature_columns,
                    'feature_metadata': feature_metadata,
                    'evaluation_results': evaluation,
                    'training_configuration': {
                        'random_state': self.random_state,
                        'skip_feature_selection': self.skip_feature_selection,
                        'enable_synthetic_data': self.enable_synthetic_data,
                        'target_column': target_column,
                        'hyperparameter_tuning_enabled': enable_hyperparameter_tuning,
                        'hyperparameter_trials': hyperparameter_trials
                    },
                    'pipeline_version': '2.0_optimized_with_hyperparameter_tuning',
                    'saved_timestamp': pd.Timestamp.now().isoformat()
                }
                
                model_save_path = output_path / "global_demand_model.pkl"
                with open(model_save_path, 'wb') as f:
                    pickle.dump(model_package, f)
                
                logger.info(f"💾 Enhanced model package saved to: {model_save_path}")
            
            # Note: Individual models are saved during training for continuation support
            
            return {
                "report_path": str(report_path),
                "analysis_artifacts_path": str(artifacts_path),
                "evaluation": evaluation,
                "best_config": evaluation['overall_best']['configuration'],
                "best_score": evaluation['overall_best']['score'],
                "assessment": evaluation['performance_assessment']
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
