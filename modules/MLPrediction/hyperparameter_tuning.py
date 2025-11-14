#!/usr/bin/env python3
"""
Standalone Hyperparameter Tuning Module for EV Charging Demand Prediction

This module provides comprehensive hyperparameter optimization capabilities that can be run
separately from the main pipeline. It supports multiple optimization strategies and can be
triggered automatically after the pipeline identifies the best performing models.

Key Features:
- AutoML capabilities with Optuna
- Support for multiple algorithms and optimization strategies
- Standalone execution with minimal dependencies
- Integration with existing pipeline results
- Comprehensive reporting and model persistence
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import time

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Ridge, BayesianRidge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
import xgboost as xgb

# Optuna for advanced hyperparameter optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# GPU-optimized libraries (with fallbacks)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)

class HyperparameterTuner:
    """
    Simple hyperparameter tuning system for specific model configurations.
    
    This class provides focused hyperparameter optimization for a specific model
    configuration (target, transform, scaler, algorithm) with AutoML capabilities
    to automatically select the best model from results.
    """
    
    def __init__(self, random_state: int = 42, use_gpu: bool = True, 
                 optimization_strategy: str = "optuna"):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            random_state: Random seed for reproducibility
            use_gpu: Whether to enable GPU acceleration when available
            optimization_strategy: Strategy to use ('optuna', 'grid_search', 'random_search')
        """
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.optimization_strategy = optimization_strategy
        self.optimization_results = {}
        self.best_models = {}
        
        # Set global random seeds for reproducibility
        np.random.seed(random_state)
        
        # Initialize GPU configuration
        self.gpu_config = self._get_gpu_config()
        
        logger.info(f"🔧 HyperparameterTuner initialized with strategy: {optimization_strategy}")
        logger.info(f"🎯 Random state: {random_state}")
        logger.info(f"🚀 GPU acceleration: {'Enabled' if use_gpu else 'Disabled'}")
    
    def _get_gpu_config(self) -> Dict[str, Any]:
        """Get GPU configuration for supported algorithms."""
        gpu_config = {'use_gpu': False}
        
        if self.use_gpu:
            try:
                # Try to detect GPU availability
                import torch
                if torch.cuda.is_available():
                    gpu_config['use_gpu'] = True
                    gpu_config['xgboost_params'] = {'tree_method': 'gpu_hist', 'gpu_id': 0}
                    gpu_config['lightgbm_params'] = {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0}
                    gpu_config['catboost_params'] = {'task_type': 'GPU', 'devices': '0'}
                    logger.info("🚀 GPU acceleration enabled")
                else:
                    logger.info("💻 GPU not available, using CPU")
            except ImportError:
                logger.info("💻 PyTorch not available, using CPU")
        
        return gpu_config
    
    def get_algorithm_configurations(self) -> Dict[str, Any]:
        """
        Get comprehensive algorithm configurations for hyperparameter optimization.
        
        Returns:
            Dictionary of algorithm names and their base configurations
        """
        algorithms = {
            # Linear models
            'ridge': Ridge(random_state=self.random_state),
            'elastic_net': ElasticNet(random_state=self.random_state, max_iter=3000),
            'lasso': Lasso(random_state=self.random_state, max_iter=3000),
            'bayesian_ridge': BayesianRidge(),
            
            # Tree-based ensemble methods
            'random_forest': RandomForestRegressor(random_state=self.random_state, n_jobs=-1),
            'extra_trees': ExtraTreesRegressor(random_state=self.random_state, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(random_state=self.random_state),
            
            # Neural network
            'neural_network': MLPRegressor(random_state=self.random_state, early_stopping=True),
            
            # XGBoost
            'xgboost': xgb.XGBRegressor(random_state=self.random_state)
        }
        
        # Add GPU-optimized algorithms if available
        if LIGHTGBM_AVAILABLE:
            lgb_params = {'random_state': self.random_state, 'verbosity': -1}
            if self.gpu_config['use_gpu']:
                lgb_params.update(self.gpu_config.get('lightgbm_params', {}))
            algorithms['lightgbm'] = lgb.LGBMRegressor(**lgb_params)
        
        if CATBOOST_AVAILABLE:
            cb_params = {'random_state': self.random_state, 'verbose': False, 'allow_writing_files': False}
            if self.gpu_config['use_gpu']:
                cb_params.update(self.gpu_config.get('catboost_params', {}))
            algorithms['catboost'] = cb.CatBoostRegressor(**cb_params)
        
        return algorithms
    
    def get_scaling_configurations(self) -> Dict[str, Any]:
        """Get feature scaling configurations."""
        return {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'quantile': QuantileTransformer(output_distribution='normal', random_state=self.random_state)
        }
    
    def optimize_with_optuna(self, X: pd.DataFrame, y: pd.Series, 
                            algorithm_name: str, n_trials: int = 100,
                            timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna with advanced search strategies.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            algorithm_name: Name of the algorithm to optimize
            n_trials: Number of optimization trials
            timeout: Maximum time in seconds for optimization
            
        Returns:
            Dictionary with optimization results
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna")
        
        logger.info(f"🔬 Optimizing {algorithm_name} with Optuna ({n_trials} trials)")
        
        def objective(trial):
            # Get hyperparameter search space based on algorithm
            params = self._get_hyperparameter_space(trial, algorithm_name)
            
            # Create model with suggested parameters
            model = self._create_model_from_params(algorithm_name, params)
            
            # Use cross-validation for robust evaluation
            try:
                cv_scores = cross_val_score(
                    model, X, y, 
                    cv=KFold(n_splits=5, shuffle=True, random_state=self.random_state),
                    scoring='r2', n_jobs=1
                )
                return cv_scores.mean()
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float('-inf')
        
        try:
            # Create study with advanced configuration
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=self.random_state),
                pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
            )
            
            # Optimize with progress tracking
            study.optimize(objective, n_trials=n_trials, timeout=timeout)
            
            # Extract results
            best_params = study.best_params
            best_score = study.best_value
            
            # Create optimized model
            optimized_model = self._create_model_from_params(algorithm_name, best_params)
            optimized_model.fit(X, y)
            
            # Calculate additional metrics
            y_pred = optimized_model.predict(X)
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            
            results = {
                'algorithm': algorithm_name,
                'best_params': best_params,
                'best_score': best_score,
                'r2_score': r2,
                'rmse': rmse,
                'mae': mae,
                'n_trials': len(study.trials),
                'n_trials_completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                'n_trials_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                'optimization_time': time.time() - study.start_time if hasattr(study, 'start_time') else 0,
                'model': optimized_model,
                'study': study
            }
            
            logger.info(f"✅ {algorithm_name} optimization complete: R²={best_score:.4f}")
            logger.info(f"   Best params: {best_params}")
            
            return results
            
        except Exception as e:
            logger.error(f"Optuna optimization failed for {algorithm_name}: {e}")
            return {}
    
    def _get_hyperparameter_space(self, trial, algorithm_name: str) -> Dict[str, Any]:
        """Get hyperparameter search space for a given algorithm."""
        
        if algorithm_name == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': self.random_state,
                'n_jobs': -1
            }
        
        elif algorithm_name == 'extra_trees':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': self.random_state,
                'n_jobs': -1
            }
        
        elif algorithm_name == 'gradient_boosting':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': self.random_state
            }
        
        elif algorithm_name == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': self.random_state
            }
            params.update(self.gpu_config.get('xgboost_params', {}))
            return params
        
        elif algorithm_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': self.random_state,
                'verbosity': -1
            }
            if self.gpu_config['use_gpu']:
                params.update(self.gpu_config.get('lightgbm_params', {}))
            return params
        
        elif algorithm_name == 'catboost' and CATBOOST_AVAILABLE:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': self.random_state,
                'verbose': False,
                'allow_writing_files': False
            }
            if self.gpu_config['use_gpu']:
                params.update(self.gpu_config.get('catboost_params', {}))
            return params
        
        elif algorithm_name == 'neural_network':
            return {
                'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', 
                    [(64,), (128,), (64, 32), (128, 64), (128, 64, 32), (256, 128, 64)]),
                'max_iter': trial.suggest_int('max_iter', 500, 2000),
                'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
                'batch_size': trial.suggest_categorical('batch_size', ['auto', 32, 64, 128, 256]),
                'random_state': self.random_state,
                'early_stopping': True,
                'validation_fraction': 0.1
            }
        
        elif algorithm_name == 'ridge':
            return {
                'alpha': trial.suggest_float('alpha', 0.1, 100, log=True),
                'random_state': self.random_state
            }
        
        elif algorithm_name == 'elastic_net':
            return {
                'alpha': trial.suggest_float('alpha', 0.01, 10, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0, 1),
                'random_state': self.random_state,
                'max_iter': 3000
            }
        
        elif algorithm_name == 'lasso':
            return {
                'alpha': trial.suggest_float('alpha', 0.01, 10, log=True),
                'random_state': self.random_state,
                'max_iter': 3000
            }
        
        elif algorithm_name == 'bayesian_ridge':
            return {
                'alpha_1': trial.suggest_float('alpha_1', 1e-8, 1e-3, log=True),
                'alpha_2': trial.suggest_float('alpha_2', 1e-8, 1e-3, log=True),
                'lambda_1': trial.suggest_float('lambda_1', 1e-8, 1e-3, log=True),
                'lambda_2': trial.suggest_float('lambda_2', 1e-8, 1e-3, log=True)
            }
        
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    def _create_model_from_params(self, algorithm_name: str, params: Dict[str, Any]):
        """Create model instance from parameters."""
        
        if algorithm_name == 'random_forest':
            return RandomForestRegressor(**params)
        elif algorithm_name == 'extra_trees':
            return ExtraTreesRegressor(**params)
        elif algorithm_name == 'gradient_boosting':
            return GradientBoostingRegressor(**params)
        elif algorithm_name == 'xgboost':
            return xgb.XGBRegressor(**params)
        elif algorithm_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
            return lgb.LGBMRegressor(**params)
        elif algorithm_name == 'catboost' and CATBOOST_AVAILABLE:
            return cb.CatBoostRegressor(**params)
        elif algorithm_name == 'neural_network':
            return MLPRegressor(**params)
        elif algorithm_name == 'ridge':
            return Ridge(**params)
        elif algorithm_name == 'elastic_net':
            return ElasticNet(**params)
        elif algorithm_name == 'lasso':
            return Lasso(**params)
        elif algorithm_name == 'bayesian_ridge':
            return BayesianRidge(**params)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    def tune_specific_model(self, X: pd.DataFrame, y: pd.Series, 
                           target_formulation: str, target_transformation: str, 
                           scaler_name: str, algorithm: str, 
                           feature_columns: List[str] = None,
                           n_trials: int = 100, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Tune hyperparameters for a specific model configuration.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            target_formulation: Target formulation (e.g., 'demand_score_balanced')
            target_transformation: Target transformation (e.g., 'original', 'log', 'quantile')
            scaler_name: Scaler name (e.g., 'standard', 'robust', 'quantile', 'none')
            algorithm: Algorithm name (e.g., 'xgboost', 'random_forest')
            feature_columns: List of feature column names (if None, uses all X columns)
            n_trials: Number of optimization trials
            timeout: Maximum time in seconds for optimization
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"🔬 TUNING SPECIFIC MODEL CONFIGURATION")
        logger.info(f"   Target: {target_formulation}")
        logger.info(f"   Transformation: {target_transformation}")
        logger.info(f"   Scaler: {scaler_name}")
        logger.info(f"   Algorithm: {algorithm}")
        logger.info(f"   Features: {len(feature_columns) if feature_columns else len(X.columns)}")
        logger.info(f"   Trials: {n_trials}")
        
        # Use specified features or all features
        if feature_columns:
            X_tuning = X[feature_columns]
        else:
            X_tuning = X
            feature_columns = list(X.columns)
        
        # Apply target transformation if needed
        y_tuning = self._apply_target_transformation(y, target_transformation)
        
        # Apply scaling if needed
        X_tuning, scaler = self._apply_scaling(X_tuning, scaler_name)
        
        # Run optimization
        try:
            if self.optimization_strategy == 'optuna':
                result = self.optimize_with_optuna(X_tuning, y_tuning, algorithm, n_trials, timeout)
            else:
                raise ValueError(f"Optimization strategy '{self.optimization_strategy}' not implemented")
            
            if result:
                # Add configuration metadata
                result['target_formulation'] = target_formulation
                result['target_transformation'] = target_transformation
                result['scaler_name'] = scaler_name
                result['feature_columns'] = feature_columns
                result['scaler_object'] = scaler
                
                # Save the optimized model
                self.best_models[f"{target_formulation}_{target_transformation}_{scaler_name}_{algorithm}"] = result['model']
                
                logger.info(f"✅ Model tuning complete: R²={result['best_score']:.4f}")
                return result
            else:
                logger.error(f"❌ Model tuning failed for {algorithm}")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Model tuning failed: {e}")
            return {}
    
    def auto_select_best_model(self, results_file: str, 
                              target_formulation: str = None,
                              metric: str = 'r2_score') -> Dict[str, Any]:
        """
        Automatically select the best model from results file for tuning.
        
        Args:
            results_file: Path to results JSON file
            target_formulation: Specific target formulation to filter (optional)
            metric: Metric to use for ranking ('r2_score', 'rmse', 'mae')
            
        Returns:
            Dictionary with best model configuration
        """
        logger.info(f"🤖 AUTO-SELECTING BEST MODEL FROM {results_file}")
        
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            best_model = None
            best_score = float('-inf') if metric in ['r2_score'] else float('inf')
            
            # Search through all results
            for target_name, target_results in results.items():
                if target_formulation and target_name != target_formulation:
                    continue
                    
                for model_name, model_data in target_results.items():
                    if not model_data or 'algorithm' not in model_data:
                        continue
                    
                    # Get the score for ranking
                    score = model_data.get(metric, 0)
                    
                    # Determine if this is better
                    is_better = False
                    if metric in ['r2_score']:
                        is_better = score > best_score
                    else:  # rmse, mae (lower is better)
                        is_better = score < best_score
                    
                    if is_better:
                        best_score = score
                        best_model = {
                            'target_formulation': target_name,
                            'target_transformation': model_data.get('target_transformation', 'original'),
                            'scaler_name': model_data.get('scaler', 'unknown'),
                            'algorithm': model_data.get('algorithm', 'unknown'),
                            'score': score,
                            'model_data': model_data
                        }
            
            if best_model:
                logger.info(f"✅ Best model selected:")
                logger.info(f"   Target: {best_model['target_formulation']}")
                logger.info(f"   Transformation: {best_model['target_transformation']}")
                logger.info(f"   Scaler: {best_model['scaler_name']}")
                logger.info(f"   Algorithm: {best_model['algorithm']}")
                logger.info(f"   Score: {best_model['score']:.4f}")
                return best_model
            else:
                logger.warning("❌ No suitable model found in results file")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Failed to auto-select model: {e}")
            return {}
    
    def _apply_target_transformation(self, y: pd.Series, transformation: str) -> pd.Series:
        """Apply target transformation using MLPrediction's DemandTargetTransformer."""
        from .target_transformation import DemandTargetTransformer
        
        transformer = DemandTargetTransformer(random_state=self.random_state)
        
        if transformation == 'original':
            return y
        elif transformation == 'log':
            return transformer.apply_log_transformation(y)
        elif transformation == 'quantile':
            return transformer.apply_quantile_transformation(y)
        else:
            logger.warning(f"Unknown transformation '{transformation}', using original")
            return y
    
    def _apply_scaling(self, X: pd.DataFrame, scaler_name: str) -> Tuple[pd.DataFrame, Any]:
        """Apply feature scaling using MLPrediction's scaling approach."""
        if scaler_name == 'none' or scaler_name == 'unknown':
            return X, None
        elif scaler_name == 'standard':
            scaler = StandardScaler()
        elif scaler_name == 'robust':
            scaler = RobustScaler()
        elif scaler_name == 'quantile':
            scaler = QuantileTransformer(output_distribution='normal', random_state=self.random_state)
        else:
            logger.warning(f"Unknown scaler '{scaler_name}', using none")
            return X, None
        
        X_scaled = scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index), scaler

    def optimize_top_models(self, training_results: Dict[str, Dict[str, Any]], 
                           X: pd.DataFrame, y: pd.Series, 
                           top_n: int = 3, n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize hyperparameters for the top N best performing models.
        
        Args:
            training_results: Results from initial training
            X: Feature DataFrame
            y: Target Series
            top_n: Number of top models to optimize
            n_trials: Number of trials per model
            
        Returns:
            Dictionary with optimization results for all models
        """
        logger.info(f"🔬 OPTIMIZING TOP {top_n} MODELS")
        logger.info("=" * 60)
        
        # Identify top performing models
        top_models = self._identify_top_models(training_results, top_n)
        
        if not top_models:
            logger.warning("No models found for optimization")
            return {}
        
        logger.info(f"Top {len(top_models)} models selected for optimization:")
        for i, model_info in enumerate(top_models, 1):
            logger.info(f"  {i}. {model_info['algorithm']} (Score: {model_info['score']:.4f})")
        
        # Optimize each top model
        optimization_results = {}
        
        for i, model_info in enumerate(top_models, 1):
            algorithm = model_info['algorithm']
            logger.info(f"\n🔬 Optimizing {i}/{len(top_models)}: {algorithm}")
            
            try:
                if self.optimization_strategy == 'optuna':
                    result = self.optimize_with_optuna(X, y, algorithm, n_trials=n_trials)
                else:
                    logger.warning(f"Optimization strategy '{self.optimization_strategy}' not implemented")
                    continue
                
                if result:
                    optimization_results[algorithm] = result
                    self.best_models[algorithm] = result['model']
                    logger.info(f"✅ {algorithm} optimization complete: R²={result['best_score']:.4f}")
                else:
                    logger.warning(f"Optimization failed for {algorithm}")
                    
            except Exception as e:
                logger.error(f"Optimization failed for {algorithm}: {e}")
        
        # Save results
        self.optimization_results = optimization_results
        self._save_optimization_results(optimization_results)
        
        logger.info(f"🎉 Optimization complete for {len(optimization_results)} models")
        return optimization_results
    
    def _identify_top_models(self, training_results: Dict[str, Dict[str, Any]], 
                           top_n: int) -> List[Dict[str, Any]]:
        """Identify the top N performing models from training results."""
        
        all_scores = []
        
        for target_name, target_results in training_results.items():
            for model_name, model_data in target_results.items():
                if model_data and 'algorithm' in model_data:
                    # Calculate performance score
                    score = self._calculate_performance_score(model_data)
                    all_scores.append({
                        'target_name': target_name,
                        'model_name': model_name,
                        'algorithm': model_data['algorithm'],
                        'score': score,
                        'model_data': model_data
                    })
        
        # Sort by score and return top N
        all_scores.sort(key=lambda x: x['score'], reverse=True)
        return all_scores[:top_n]
    
    def _calculate_performance_score(self, model_data: Dict[str, Any]) -> float:
        """Calculate combined performance score from model metrics."""
        
        # Weighted combination of metrics
        r2_weight = 0.4
        pearson_weight = 0.3
        cv_weight = 0.3
        
        r2_score = max(model_data.get('test_r2', model_data.get('cv_r2_mean', 0)), 0)
        pearson_score = abs(model_data.get('pearson_correlation', model_data.get('cv_pearson_mean', 0)))
        cv_score = max(model_data.get('cv_mean', model_data.get('cv_r2_mean', 0)), 0)
        
        combined_score = (
            r2_weight * r2_score +
            pearson_weight * pearson_score +
            cv_weight * cv_score
        )
        
        return combined_score
    
    def _save_optimization_results(self, results: Dict[str, Any], 
                                 output_dir: str = "results/hyperparameter_optimization") -> None:
        """Save optimization results to files."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = output_path / f"optimization_results_{timestamp}.json"
        
        # Convert results to serializable format
        serializable_results = {}
        for algorithm, result in results.items():
            serializable_results[algorithm] = {
                'algorithm': result['algorithm'],
                'best_params': result['best_params'],
                'best_score': result['best_score'],
                'r2_score': result['r2_score'],
                'rmse': result['rmse'],
                'mae': result['mae'],
                'n_trials': result['n_trials'],
                'n_trials_completed': result['n_trials_completed'],
                'n_trials_pruned': result['n_trials_pruned'],
                'optimization_time': result['optimization_time']
            }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Save best models
        models_file = output_path / f"best_models_{timestamp}.pkl"
        with open(models_file, 'wb') as f:
            pickle.dump(self.best_models, f)
        
        # Generate summary report
        self._generate_optimization_report(results, output_path, timestamp)
        
        logger.info(f"📊 Optimization results saved to: {output_path}")
        logger.info(f"📁 Detailed results: {results_file}")
        logger.info(f"🤖 Best models: {models_file}")
    
    def _generate_optimization_report(self, results: Dict[str, Any], 
                                    output_path: Path, timestamp: str) -> None:
        """Generate comprehensive optimization report."""
        
        report_lines = []
        report_lines.append("# Hyperparameter Optimization Report")
        report_lines.append("")
        report_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**Strategy**: {self.optimization_strategy}")
        report_lines.append(f"**Random State**: {self.random_state}")
        report_lines.append("")
        
        # Summary
        report_lines.append("## Summary")
        report_lines.append(f"- **Models Optimized**: {len(results)}")
        report_lines.append(f"- **Total Trials**: {sum(r.get('n_trials', 0) for r in results.values())}")
        report_lines.append(f"- **Total Time**: {sum(r.get('optimization_time', 0) for r in results.values()):.2f} seconds")
        report_lines.append("")
        
        # Results by Algorithm
        report_lines.append("## Results by Algorithm")
        report_lines.append("")
        
        for algorithm, result in results.items():
            report_lines.append(f"### {algorithm}")
            report_lines.append(f"- **Best Score (R²)**: {result['best_score']:.4f}")
            report_lines.append(f"- **Final R²**: {result['r2_score']:.4f}")
            report_lines.append(f"- **RMSE**: {result['rmse']:.4f}")
            report_lines.append(f"- **MAE**: {result['mae']:.4f}")
            report_lines.append(f"- **Trials Completed**: {result['n_trials_completed']}")
            report_lines.append(f"- **Trials Pruned**: {result['n_trials_pruned']}")
            report_lines.append(f"- **Optimization Time**: {result['optimization_time']:.2f}s")
            report_lines.append("")
            
            # Best parameters
            report_lines.append("**Best Parameters:**")
            for param, value in result['best_params'].items():
                report_lines.append(f"- {param}: {value}")
            report_lines.append("")
        
        # Save report
        report_file = output_path / f"optimization_report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"📋 Optimization report saved to: {report_file}")
    
    def load_optimized_models(self, models_file: str) -> Dict[str, Any]:
        """Load previously optimized models from file."""
        
        with open(models_file, 'rb') as f:
            self.best_models = pickle.load(f)
        
        logger.info(f"🤖 Loaded {len(self.best_models)} optimized models from {models_file}")
        return self.best_models
    
    def predict_with_best_model(self, X: pd.DataFrame, algorithm: str) -> np.ndarray:
        """Make predictions using the best optimized model for a given algorithm."""
        
        if algorithm not in self.best_models:
            raise ValueError(f"No optimized model found for algorithm: {algorithm}")
        
        model = self.best_models[algorithm]
        return model.predict(X)
    
    def get_optimization_summary(self) -> pd.DataFrame:
        """Get summary of optimization results as DataFrame."""
        
        if not self.optimization_results:
            return pd.DataFrame()
        
        summary_data = []
        for algorithm, result in self.optimization_results.items():
            summary_data.append({
                'algorithm': algorithm,
                'best_score': result['best_score'],
                'r2_score': result['r2_score'],
                'rmse': result['rmse'],
                'mae': result['mae'],
                'n_trials': result['n_trials'],
                'optimization_time': result['optimization_time']
            })
        
        return pd.DataFrame(summary_data).sort_values('best_score', ascending=False)


def main():
    """Main function for standalone hyperparameter tuning."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Standalone Hyperparameter Tuning for EV Demand Prediction")
    parser.add_argument("--data-path", required=True, help="Path to training data CSV file")
    parser.add_argument("--target-column", default="demand_score_balanced", help="Target column name")
    parser.add_argument("--output-dir", default="results/hyperparameter_optimization", help="Output directory")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of optimization trials")
    parser.add_argument("--top-n", type=int, default=3, help="Number of top models to optimize")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU acceleration")
    parser.add_argument("--strategy", default="optuna", choices=["optuna"], help="Optimization strategy")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("🚀 STARTING STANDALONE HYPERPARAMETER TUNING")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Target: {args.target_column}")
    logger.info(f"Trials: {args.n_trials}")
    logger.info(f"Top N: {args.top_n}")
    
    try:
        # Load data
        data = pd.read_csv(args.data_path)
        logger.info(f"Loaded data: {data.shape}")
        
        # Prepare features and target
        feature_columns = [col for col in data.columns 
                          if not col.startswith('demand_score') and col != 'grid_id']
        X = data[feature_columns]
        y = data[args.target_column]
        
        logger.info(f"Features: {len(feature_columns)}, Samples: {len(X)}")
        
        # Initialize tuner
        tuner = HyperparameterTuner(
            random_state=args.random_state,
            use_gpu=args.use_gpu,
            optimization_strategy=args.strategy
        )
        
        # Create dummy training results for demonstration
        # In practice, this would come from the main pipeline
        dummy_results = {
            'target_dummy': {
                'model_1': {'algorithm': 'random_forest', 'test_r2': 0.7, 'cv_mean': 0.65},
                'model_2': {'algorithm': 'xgboost', 'test_r2': 0.75, 'cv_mean': 0.70},
                'model_3': {'algorithm': 'lightgbm', 'test_r2': 0.72, 'cv_mean': 0.68}
            }
        }
        
        # Run optimization
        results = tuner.optimize_top_models(
            dummy_results, X, y, 
            top_n=args.top_n, n_trials=args.n_trials
        )
        
        logger.info("🎉 Hyperparameter tuning completed successfully!")
        
    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {e}")
        raise


if __name__ == "__main__":
    main()
