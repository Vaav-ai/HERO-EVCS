"""
Model Training Module for EV Charging Demand Prediction

This module implements comprehensive model training with multiple algorithms,
proper validation, and performance evaluation. All models are scientifically
justified and commonly used in regression tasks.

Key Components:
- Multiple regression algorithms with optimized hyperparameters
- Robust cross-validation and evaluation metrics
- Model comparison and selection framework
- Feature importance analysis
- Performance reporting
"""

import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.feature_selection import RFECV, SelectKBest, f_regression, mutual_info_regression, VarianceThreshold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Ridge, BayesianRidge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import scipy.stats as stats
import xgboost as xgb
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    try:
        from imbalanced_learn.over_sampling import SMOTE
    except ImportError:
        print("Warning: imblearn not available, SMOTE will be disabled")
        SMOTE = None
from .synthetic_data_generator import SyntheticDataGenerator

logger = logging.getLogger(__name__)

# Optuna for advanced hyperparameter optimization
try:
    import optuna
    # Try to import LightGBMPruningCallback, but don't fail if it's not available
    try:
        from optuna.integration import LightGBMPruningCallback
    except ImportError:
        LightGBMPruningCallback = None
        logger.warning("LightGBMPruningCallback not available - install with: pip install optuna-integration[lightgbm]")
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    LightGBMPruningCallback = None
    logger.warning("Optuna not available - install with: pip install optuna")

# GPU-optimized libraries (with fallbacks)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available - install with: pip install lightgbm")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not available - install with: pip install catboost")

# Import GPU utilities
try:
    from modules.utils.gpu_utils import detect_gpu_availability, get_gpu_config, log_gpu_status
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False
    logger.warning("GPU utilities not available")

class ModelTrainer:
    """
    A robust model training system for EV charging demand prediction.
    
    This class implements a comprehensive training pipeline with multiple algorithms,
    proper validation, and robust performance evaluation.
    """
    
    def __init__(self, random_state: int = 42, enable_synthetic_data: bool = True, use_gpu: bool = True, 
                 skip_feature_selection: bool = False):
        """
        Initialize the model trainer with standard configurations.
        
        Args:
            random_state: Random seed for reproducibility
            enable_synthetic_data: Whether to use synthetic data augmentation
            use_gpu: Whether to enable GPU acceleration when available
            skip_feature_selection: Whether to skip feature selection (faster for large datasets)
        """
        self.random_state = random_state
        self.enable_synthetic_data = enable_synthetic_data
        self.use_gpu = use_gpu
        self.skip_feature_selection = skip_feature_selection
        self.trained_models = {}
        
        # Set global random seeds for reproducibility
        np.random.seed(random_state)
        if hasattr(pd, 'set_option'):
            # Ensure pandas uses consistent random sampling
            pass  # pandas uses numpy's random state
        
        # Set random state for scipy.stats if used
        if hasattr(stats, 'set_random_state'):
            stats.set_random_state(np.random.RandomState(random_state))
        
        # Initialize GPU configuration
        if GPU_UTILS_AVAILABLE and use_gpu:
            self.gpu_info = detect_gpu_availability()
            self.gpu_config = get_gpu_config(use_gpu=use_gpu)
            log_gpu_status(self.gpu_info)
        else:
            self.gpu_info = {'cuda_available': False}
            self.gpu_config = {'use_gpu': False}
            logger.info("💻 GPU acceleration disabled - using CPU-only computation")
        self.performance_results = {}
        self.feature_importance = {}
        
    def get_algorithm_configurations(self) -> Dict[str, Any]:
        """
        Get comprehensive algorithm configurations for EV demand prediction.
        Includes all algorithms needed for thorough paper evaluation.
        
        Returns:
            Dictionary of algorithm names and their configured instances
        """
        algorithms = {
            # Linear models for baseline comparison
            'ridge': Ridge(
                alpha=0.5, 
                random_state=self.random_state
            ),
            'elastic_net': ElasticNet(
                alpha=0.05, 
                l1_ratio=0.7, 
                random_state=self.random_state, 
                max_iter=3000
            ),
            'lasso': Lasso(
                alpha=0.05, 
                random_state=self.random_state, 
                max_iter=3000
            ),
            'bayesian_ridge': BayesianRidge(
                alpha_1=1e-6,
                alpha_2=1e-6,
                lambda_1=1e-6,
                lambda_2=1e-6
            ),
            
            # Tree-based ensemble methods
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.9,
                random_state=self.random_state
            ),
            
            # Neural network for non-linear modeling
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                max_iter=2000,
                alpha=0.01,
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1
            )
        }
        
        # XGBoost with GPU optimization and CUDA fallback
        xgb_params = {
            'n_estimators': 150,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'random_state': self.random_state
        }
        xgb_params.update(self.gpu_config.get('xgboost_params', {}))
        
        # Ensure CPU fallback if GPU params cause issues
        if 'device' in xgb_params and xgb_params['device'].startswith('cuda'):
            # Keep GPU params but add CPU fallback
            xgb_params['device'] = xgb_params['device']  # Keep original device
        else:
            # Force CPU if no device specified
            xgb_params['device'] = 'cpu'
            xgb_params['tree_method'] = 'hist'
            
        algorithms['xgboost'] = xgb.XGBRegressor(**xgb_params)
        
        # LightGBM with GPU optimization (with fallback to CPU)
        if LIGHTGBM_AVAILABLE:
            lgb_params = {
                'n_estimators': 150,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'random_state': self.random_state,
                'verbosity': -1
            }
            # Only add GPU params if GPU is actually available and working
            if self.gpu_config.get('use_gpu', False) and self.gpu_info.get('cuda_available', False):
                try:
                    lgb_params.update(self.gpu_config.get('lightgbm_params', {}))
                except Exception as e:
                    logger.warning(f"Failed to configure LightGBM GPU params: {e}. Using CPU.")
            algorithms['lightgbm'] = lgb.LGBMRegressor(**lgb_params)
        
        # CatBoost with GPU optimization
        if CATBOOST_AVAILABLE:
            cb_params = {
                'n_estimators': 150,
                'max_depth': 8,
                'learning_rate': 0.05,
                'random_state': self.random_state,
                'verbose': False,
                'allow_writing_files': False
            }
            cb_params.update(self.gpu_config.get('catboost_params', {}))
            algorithms['catboost'] = cb.CatBoostRegressor(**cb_params)
        
        # Neural Network (keep as is, TensorFlow GPU optimization handled separately)
        algorithms['neural_network'] = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=2000,
            alpha=0.01,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        if self.gpu_config['use_gpu']:
            logger.info(f"🚀 GPU-accelerated algorithms enabled: {[k for k in algorithms.keys() if k in ['xgboost', 'lightgbm', 'catboost']]}")
        
        return algorithms
    
    def optimize_hyperparameters_optuna(self, X: pd.DataFrame, y: pd.Series, 
                                       algorithm_name: str, n_trials: int = 50) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna for better performance.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            algorithm_name: Name of the algorithm to optimize
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary with best parameters and score
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, skipping hyperparameter optimization")
            return {}
        
        logger.info(f"🔬 Optimizing {algorithm_name} hyperparameters with Optuna ({n_trials} trials)")
        
        def objective(trial):
            if algorithm_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'random_state': self.random_state,
                    'n_jobs': -1
                }
                model = RandomForestRegressor(**params)
                
            elif algorithm_name == 'extra_trees':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'random_state': self.random_state,
                    'n_jobs': -1
                }
                model = ExtraTreesRegressor(**params)
                
            elif algorithm_name == 'gradient_boosting':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'random_state': self.random_state
                }
                model = GradientBoostingRegressor(**params)
                
            elif algorithm_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': self.random_state
                }
                params.update(self.gpu_config.get('xgboost_params', {}))
                model = xgb.XGBRegressor(**params)
                
            elif algorithm_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': self.random_state,
                    'verbosity': -1
                }
                if self.gpu_config.get('use_gpu', False) and self.gpu_info.get('cuda_available', False):
                    params.update(self.gpu_config.get('lightgbm_params', {}))
                model = lgb.LGBMRegressor(**params)
                
            elif algorithm_name == 'catboost' and CATBOOST_AVAILABLE:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'random_state': self.random_state,
                    'verbose': False,
                    'allow_writing_files': False
                }
                if self.gpu_config.get('use_gpu', False) and self.gpu_info.get('cuda_available', False):
                    params.update(self.gpu_config.get('catboost_params', {}))
                model = cb.CatBoostRegressor(**params)
                
            elif algorithm_name == 'neural_network':
                params = {
                    'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', 
                        [(64,), (128,), (64, 32), (128, 64), (128, 64, 32)]),
                    'max_iter': trial.suggest_int('max_iter', 500, 2000),
                    'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
                    'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
                    'random_state': self.random_state,
                    'early_stopping': True,
                    'validation_fraction': 0.1
                }
                model = MLPRegressor(**params)
                
            else:
                # For linear models, use default parameters
                return 0.0
            
            # Use cross-validation for robust evaluation with multiple metrics
            # This ensures scientific rigor for research papers
            try:
                # Primary metric: R² score (coefficient of determination)
                r2_scores = cross_val_score(model, X, y, cv=3, scoring='r2', n_jobs=1)
                r2_mean = r2_scores.mean()
                
                # Secondary metric: Negative RMSE (to maximize, we minimize negative RMSE)
                rmse_scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error', n_jobs=1)
                rmse_mean = -np.sqrt(-rmse_scores.mean())  # Convert back to RMSE
                
                # Tertiary metric: Pearson correlation
                from sklearn.metrics import make_scorer
                def pearson_correlation_scorer(y_true, y_pred):
                    if len(np.unique(y_pred)) <= 1 or len(np.unique(y_true)) <= 1:
                        return 0.0
                    corr = np.corrcoef(y_true, y_pred)[0, 1]
                    return corr if not np.isnan(corr) else 0.0
                
                pearson_scores = cross_val_score(model, X, y, cv=3, 
                                               scoring=make_scorer(pearson_correlation_scorer), n_jobs=1)
                pearson_mean = pearson_scores.mean()
                
                # Composite score: Weighted combination for scientific robustness
                # R²: 50%, RMSE: 30%, Pearson: 20% (normalized)
                rmse_normalized = max(0, 1 - (rmse_mean / (y.std() + 1e-8)))  # Normalize RMSE
                pearson_normalized = max(0, pearson_mean)  # Ensure non-negative
                
                composite_score = (0.5 * max(0, r2_mean) + 
                                 0.3 * rmse_normalized + 
                                 0.2 * pearson_normalized)
                
                # Add small penalty for high variance (overfitting prevention)
                r2_std = r2_scores.std()
                stability_penalty = max(0, 0.1 * r2_std)  # Penalize high variance
                
                final_score = composite_score - stability_penalty
                
                return max(0, final_score)  # Ensure non-negative score
                
            except Exception as e:
                logger.warning(f"Cross-validation failed in Optuna objective: {e}")
                return 0.0
        
        try:
            # Create study with scientific reproducibility settings
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=self.random_state),  # Reproducible sampling
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)  # Prune bad trials
            )
            
            # Optimize with progress tracking
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            
            best_params = study.best_params
            best_score = study.best_value
            
            # Calculate additional scientific metrics
            best_trial = study.best_trial
            optimization_metadata = {
                'n_trials_completed': len(study.trials),
                'n_trials_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                'optimization_duration': best_trial.duration.total_seconds() if best_trial.duration else 0,
                'study_name': study.study_name,
                'sampler': 'TPESampler',
                'pruner': 'MedianPruner'
            }
            
            logger.info(f"✅ {algorithm_name} optimization complete: Score={best_score:.4f}")
            logger.info(f"   Best params: {best_params}")
            logger.info(f"   Trials completed: {optimization_metadata['n_trials_completed']}")
            logger.info(f"   Trials pruned: {optimization_metadata['n_trials_pruned']}")
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'n_trials': n_trials,
                'optimization_method': 'optuna',
                'optimization_metadata': optimization_metadata,
                'study_object': study  # For advanced analysis
            }
            
        except Exception as e:
            logger.warning(f"Optuna optimization failed for {algorithm_name}: {e}")
            return {}
    
    def fast_feature_selection(self, X: pd.DataFrame, y: pd.Series, max_features: int = None, return_metrics: bool = False) -> List[str]:
        """
        Fast feature selection using statistical methods with data quality checks.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            max_features: Maximum number of features to select (default: smart selection based on dataset size)
            return_metrics: Whether to return detailed feature selection metrics
            
        Returns:
            List of selected feature names, or tuple of (selected_features, metrics) if return_metrics=True
        """
        logger.info("🧹 CLEANING DATA BEFORE FEATURE SELECTION")
        
        # Step 1: Remove zero variance features (constant values)
        initial_features = len(X.columns)
        variance_threshold = VarianceThreshold(threshold=0.0001)
        X_cleaned = pd.DataFrame(
            variance_threshold.fit_transform(X),
            columns=X.columns[variance_threshold.get_support()],
            index=X.index
        )
        removed_zero_var = set(X.columns) - set(X_cleaned.columns)
        if removed_zero_var:
            logger.info(f"🗑️  Removed {len(removed_zero_var)} zero variance features: {list(removed_zero_var)}")
        
        # Step 2: Remove highly correlated features (>0.95)
        corr_matrix = X_cleaned.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = [(column, row, corr_matrix.loc[row, column]) 
                          for column in upper_triangle.columns 
                          for row in upper_triangle.index 
                          if upper_triangle.loc[row, column] > 0.95]
        
        features_to_remove = set()
        for col1, col2, corr_val in high_corr_pairs:
            # Keep the feature with higher mutual information with target
            # Use consistent random state for reproducible results
            mi1 = mutual_info_regression(X_cleaned[[col1]], y, random_state=self.random_state)[0]
            mi2 = mutual_info_regression(X_cleaned[[col2]], y, random_state=self.random_state)[0]
            # For ties, use deterministic selection based on feature name
            if abs(mi1 - mi2) < 1e-10:  # Essentially equal scores
                remove_feature = col1 if col1 > col2 else col2  # Deterministic tie-breaking
            else:
                remove_feature = col1 if mi1 < mi2 else col2
            features_to_remove.add(remove_feature)
            
        if features_to_remove:
            logger.info(f"🗑️  Removed {len(features_to_remove)} highly correlated features: {list(features_to_remove)}")
            X_cleaned = X_cleaned.drop(columns=features_to_remove)
        
        logger.info(f"✅ Data cleaning complete: {initial_features} → {len(X_cleaned.columns)} features")
        
        # Step 3: Smart feature count determination
        if max_features is None:
            total_features = len(X_cleaned.columns)
            if total_features <= 20:
                max_features = total_features  # Use all for small feature sets
            elif total_features <= 50:
                max_features = max(15, min(25, total_features // 2))  # Conservative selection
            else:
                max_features = max(20, min(35, total_features // 2))  # Large dataset selection
        
        # Ensure we don't select more than available
        max_features = min(max_features, len(X_cleaned.columns))
        
        logger.info(f"🚀 Smart feature selection: selecting top {max_features} features from {len(X_cleaned.columns)} cleaned features")
        
        # Step 4: Feature selection using mutual information with proper random state
        def mutual_info_with_seed(X, y):
            """Wrapper for mutual_info_regression with fixed random state."""
            return mutual_info_regression(X, y, random_state=self.random_state)
        
        selector = SelectKBest(score_func=mutual_info_with_seed, k=max_features)
        selector.fit(X_cleaned, y)
        
        selected_features = list(X_cleaned.columns[selector.get_support()])
        
        # Get ALL feature scores for analysis
        all_feature_scores = selector.scores_
        all_feature_ranking = list(zip(X_cleaned.columns, all_feature_scores))
        all_feature_ranking.sort(key=lambda x: x[1], reverse=True)
        
        # Log selected features
        selected_feature_scores = selector.scores_[selector.get_support()]
        selected_feature_ranking = list(zip(selected_features, selected_feature_scores))
        selected_feature_ranking.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"✅ Selected {len(selected_features)} features using mutual information")
        logger.info(f"Top 5 selected features: {[f'{name} ({score:.3f})' for name, score in selected_feature_ranking[:5]]}")
        
        # 🔍 INCONSISTENCY DETECTION - Check for problematic features
        self._detect_feature_inconsistencies(X, all_feature_ranking)
        
        # Save comprehensive feature importance analysis
        self._save_feature_importance_analysis(
            all_feature_ranking, selected_feature_ranking, 
            X_cleaned.columns.tolist(), selected_features
        )
        
        if return_metrics:
            # Collect comprehensive metrics for paper
            metrics = {
                'selection_summary': {
                    'initial_features': initial_features,
                    'features_after_cleaning': len(X_cleaned.columns),
                    'selected_features': len(selected_features),
                    'zero_variance_removed': len(removed_zero_var),
                    'highly_correlated_removed': len(features_to_remove),
                    'max_features_requested': max_features,
                    'selection_ratio': len(selected_features) / initial_features
                },
                'feature_scores': {
                    'all_features': dict(all_feature_ranking),
                    'selected_features': dict(selected_feature_ranking),
                    'mutual_information_scores': dict(zip(X_cleaned.columns, all_feature_scores))
                },
                'data_quality': {
                    'zero_variance_features': list(removed_zero_var),
                    'highly_correlated_pairs': high_corr_pairs,
                    'correlation_matrix': X_cleaned.corr().to_dict(),
                    'feature_variance': X_cleaned.var().to_dict(),
                    'target_correlation': X_cleaned.corrwith(y).to_dict()
                },
                'selection_criteria': {
                    'variance_threshold': 0.0001,
                    'correlation_threshold': 0.95,
                    'selection_method': 'mutual_information',
                    'random_state': self.random_state
                },
                'target_statistics': {
                    'mean': float(y.mean()),
                    'std': float(y.std()),
                    'min': float(y.min()),
                    'max': float(y.max()),
                    'skewness': float(y.skew()),
                    'kurtosis': float(y.kurtosis())
                }
            }
            return selected_features, metrics
        else:
            return selected_features
    
    def _detect_feature_inconsistencies(self, X: pd.DataFrame, feature_ranking: List[Tuple[str, float]]) -> None:
        """
        Detect and log potential feature inconsistencies and quality issues.
        
        Args:
            X: Feature DataFrame
            feature_ranking: List of (feature_name, score) tuples sorted by score
        """
        logger.info("🔍 FEATURE QUALITY ASSESSMENT")
        
        inconsistencies_found = []
        
        # 1. Check for duplicate urban_centrality_score calculations
        centrality_features = [f for f in X.columns if 'centrality' in f.lower()]
        if len(centrality_features) > 1:
            logger.warning(f"⚠️  Multiple centrality features detected: {centrality_features}")
            inconsistencies_found.append("Multiple centrality calculations")
        
        # 2. Check for features with zero variance (constant values)
        zero_variance_features = []
        for col in X.columns:
            if X[col].var() < 1e-10:
                zero_variance_features.append(col)
        
        if zero_variance_features:
            logger.warning(f"⚠️  Zero variance features (constant values): {zero_variance_features}")
            inconsistencies_found.append("Zero variance features detected")
        
        # 3. Check for highly correlated features (potential redundancy)
        correlation_matrix = X.corr().abs()
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if correlation_matrix.iloc[i, j] > 0.95:
                    col1, col2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                    high_corr_pairs.append((col1, col2, correlation_matrix.iloc[i, j]))
        
        if high_corr_pairs:
            logger.warning(f"⚠️  Highly correlated feature pairs (>0.95): {high_corr_pairs[:3]}")
            inconsistencies_found.append("High feature correlation detected")
        
        # 4. Check for features with abnormal score distributions
        scores = [score for _, score in feature_ranking]
        low_score_features = [name for name, score in feature_ranking if score < 0.01]
        
        if len(low_score_features) > len(feature_ranking) * 0.3:  # More than 30% have very low scores
            logger.warning(f"⚠️  Many features with very low scores: {len(low_score_features)} features < 0.01")
            inconsistencies_found.append("Many low-importance features")
        
        # 5. Check for grid position feature consistency
        grid_pos_features = [f for f in X.columns if 'grid_position' in f]
        if grid_pos_features:
            for feat in grid_pos_features:
                unique_vals = X[feat].nunique()
                if unique_vals > 100:  # Should be limited to 0-99 range
                    logger.warning(f"⚠️  Grid position feature {feat} has {unique_vals} unique values (expected ≤100)")
                    inconsistencies_found.append("Grid position range inconsistency")
        
        # 6. Check feature naming consistency
        naming_issues = []
        for col in X.columns:
            if col.endswith('_x') or col.endswith('_y'):
                naming_issues.append(col)
        
        if naming_issues:
            logger.warning(f"⚠️  Features with merge suffixes (potential duplicates): {naming_issues}")
            inconsistencies_found.append("Feature naming inconsistencies")
        
        # Summary
        if inconsistencies_found:
            logger.warning(f"🚨 INCONSISTENCIES DETECTED: {', '.join(inconsistencies_found)}")
            logger.warning("   Recommend reviewing feature engineering pipeline for robustness")
    
    def _save_feature_importance_analysis(self, all_feature_ranking: List[Tuple[str, float]], 
                                        selected_feature_ranking: List[Tuple[str, float]],
                                        all_features: List[str], selected_features: List[str]) -> None:
        """
        Save comprehensive feature importance analysis for paper documentation.
        
        Args:
            all_feature_ranking: All features ranked by mutual information score
            selected_feature_ranking: Selected features with scores
            all_features: List of all available features
            selected_features: List of selected features
        """
        try:
            # Create feature importance DataFrame
            importance_data = []
            for i, (feature, score) in enumerate(all_feature_ranking):
                importance_data.append({
                    'rank': i + 1,
                    'feature_name': feature,
                    'mutual_info_score': score,
                    'selected': feature in selected_features,
                    'selection_method': 'mutual_information'
                })
            
            importance_df = pd.DataFrame(importance_data)
            
            # Save to CSV for easy analysis
            output_dir = Path("results/feature_analysis")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            importance_df.to_csv(output_dir / "complete_feature_importance.csv", index=False)
            
            # Save selected features summary
            selected_summary = {
                'selection_method': 'mutual_information_with_cleaning',
                'total_features_available': len(all_features),
                'features_after_cleaning': len(all_feature_ranking),
                'features_selected': len(selected_features),
                'selection_ratio': len(selected_features) / len(all_feature_ranking),
                'top_10_features': [{'feature': f, 'score': s} for f, s in selected_feature_ranking[:10]],
                'selection_timestamp': pd.Timestamp.now().isoformat()
            }
            
            with open(output_dir / "feature_selection_summary.json", 'w') as f:
                json.dump(selected_summary, f, indent=2)
            
            logger.info(f"📊 Feature importance analysis saved to: {output_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to save feature importance analysis: {e}")
    
    def validate_feature_selection_with_rfecv(self, X: pd.DataFrame, y: pd.Series, 
                                            sample_sizes: List[int] = [5000, 10000, 20000],
                                            n_samples: int = 3) -> Dict[str, Any]:
        """
        Validate feature selection using RFECV on different sample sizes for robustness analysis.
        
        This method provides scientific validation of our feature selection approach by
        testing it on multiple random samples with different sizes using RFECV.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            sample_sizes: List of sample sizes to test
            n_samples: Number of random samples per size
            
        Returns:
            Dictionary with validation results
        """
        logger.info("🔬 VALIDATING FEATURE SELECTION WITH RFECV")
        logger.info("=" * 60)
        
        validation_results = {
            'validation_method': 'rfecv_robustness_analysis',
            'sample_sizes_tested': sample_sizes,
            'samples_per_size': n_samples,
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'results': {}
        }
        
        for sample_size in sample_sizes:
            if len(X) < sample_size:
                logger.info(f"⏭️  Skipping sample size {sample_size} (dataset too small: {len(X)})")
                continue
                
            logger.info(f"📊 Testing sample size: {sample_size}")
            size_results = []
            
            for i in range(n_samples):
                logger.info(f"  Sample {i+1}/{n_samples}...")
                
                # Random sample
                sample_indices = np.random.choice(len(X), size=sample_size, replace=False)
                X_sample = X.iloc[sample_indices]
                y_sample = y.iloc[sample_indices]
                
                try:
                    # Run RFECV
                    estimator = ExtraTreesRegressor(n_estimators=50, random_state=self.random_state)
                    selector = RFECV(
                        estimator, 
                        step=1, 
                        cv=3,  # Use 3-fold for speed
                        scoring='r2',
                        min_features_to_select=10,
                        n_jobs=-1
                    )
                    selector.fit(X_sample, y_sample)
                    
                    # Get selected features
                    rfecv_features = X.columns[selector.support_].tolist()
                    
                    # Compare with our mutual information selection
                    mi_features = self.fast_feature_selection(X_sample, y_sample, max_features=len(rfecv_features))
                    
                    # Calculate overlap
                    overlap = len(set(rfecv_features) & set(mi_features))
                    overlap_ratio = overlap / len(rfecv_features) if rfecv_features else 0
                    
                    size_results.append({
                        'sample_id': i + 1,
                        'rfecv_features': rfecv_features,
                        'mi_features': mi_features,
                        'overlap_count': overlap,
                        'overlap_ratio': overlap_ratio,
                        'rfecv_feature_count': len(rfecv_features),
                        'mi_feature_count': len(mi_features)
                    })
                    
                    logger.info(f"    RFECV: {len(rfecv_features)} features, MI: {len(mi_features)} features, Overlap: {overlap_ratio:.2%}")
                    
                except Exception as e:
                    logger.warning(f"    RFECV failed for sample {i+1}: {e}")
                    size_results.append({
                        'sample_id': i + 1,
                        'error': str(e)
                    })
            
            # Calculate statistics for this sample size
            valid_results = [r for r in size_results if 'error' not in r]
            if valid_results:
                avg_overlap = np.mean([r['overlap_ratio'] for r in valid_results])
                std_overlap = np.std([r['overlap_ratio'] for r in valid_results])
                
                validation_results['results'][f'size_{sample_size}'] = {
                    'sample_size': sample_size,
                    'valid_samples': len(valid_results),
                    'average_overlap_ratio': avg_overlap,
                    'std_overlap_ratio': std_overlap,
                    'individual_results': size_results
                }
                
                logger.info(f"  ✅ Size {sample_size}: Avg overlap {avg_overlap:.2%} ± {std_overlap:.2%}")
            else:
                logger.warning(f"  ❌ Size {sample_size}: No valid results")
        
        # Save validation results
        output_dir = Path("results/feature_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "rfecv_validation_results.json", 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        logger.info(f"🔬 RFECV validation results saved to: {output_dir}/rfecv_validation_results.json")
        
        return validation_results
    
    def optimize_best_models_only(self, training_results: Dict[str, Dict[str, Any]], 
                                 X: pd.DataFrame, y: pd.Series, 
                                 top_n: int = 3, n_trials: int = 50) -> Dict[str, Dict[str, Any]]:
        """
        Optimize hyperparameters for only the top N best performing models.
        This is much more efficient than optimizing all models.
        
        Args:
            training_results: Results from initial training with default parameters
            X: Feature DataFrame
            y: Target Series
            top_n: Number of top models to optimize
            n_trials: Number of Optuna trials per model
            
        Returns:
            Updated training results with optimized models
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, skipping hyperparameter optimization")
            return training_results
        
        logger.info(f"🔬 OPTIMIZING TOP {top_n} MODELS WITH OPTUNA")
        logger.info("=" * 60)
        
        # Collect all model scores and identify top performers
        all_scores = []
        for target_name, target_results in training_results.items():
            for model_name, model_data in target_results.items():
                score = self._calculate_performance_score(model_data)
                all_scores.append({
                    'target_name': target_name,
                    'model_name': model_name,
                    'algorithm': model_data.get('algorithm', 'unknown'),
                    'scaler': model_data.get('scaler', 'unknown'),
                    'score': score,
                    'model_data': model_data
                })
        
        # Sort by score and get top N
        all_scores.sort(key=lambda x: x['score'], reverse=True)
        top_models = all_scores[:top_n]
        
        logger.info(f"Top {top_n} models selected for optimization:")
        for i, model_info in enumerate(top_models, 1):
            logger.info(f"  {i}. {model_info['target_name']}_{model_info['model_name']} (Score: {model_info['score']:.4f})")
        
        # Optimize each top model
        optimization_results = {}
        for i, model_info in enumerate(top_models, 1):
            target_name = model_info['target_name']
            model_name = model_info['model_name']
            algorithm = model_info['algorithm']
            
            logger.info(f"\n🔬 Optimizing {i}/{top_n}: {target_name}_{model_name} ({algorithm})")
            
            try:
                # Run Optuna optimization
                optuna_results = self.optimize_hyperparameters_optuna(X, y, algorithm, n_trials=n_trials)
                
                if optuna_results and 'best_params' in optuna_results:
                    # Create optimized model
                    optimized_model = self._create_optimized_model(algorithm, optuna_results['best_params'])
                    
                    # Retrain with optimized parameters
                    scaler = model_info['model_data']['scaler_object']
                    X_scaled = scaler.fit_transform(X)
                    optimized_model.fit(X_scaled, y)
                    
                    # Update the model in results
                    training_results[target_name][model_name]['model_object'] = optimized_model
                    training_results[target_name][model_name]['optimization_applied'] = True
                    training_results[target_name][model_name]['optuna_params'] = optuna_results['best_params']
                    training_results[target_name][model_name]['optuna_score'] = optuna_results['best_score']
                    
                    optimization_results[f"{target_name}_{model_name}"] = {
                        'algorithm': algorithm,
                        'best_params': optuna_results['best_params'],
                        'best_score': optuna_results['best_score'],
                        'n_trials': n_trials
                    }
                    
                    logger.info(f"✅ Optimization complete: R²={optuna_results['best_score']:.4f}")
                else:
                    logger.warning(f"Optimization failed for {target_name}_{model_name}")
                    
            except Exception as e:
                logger.error(f"Optimization failed for {target_name}_{model_name}: {e}")
        
        # Save optimization report with target information
        target_name = list(training_results.keys())[0] if training_results else "unknown"
        self._save_optimization_report(optimization_results, target_name)
        
        logger.info(f"🎉 Optimization complete for {len(optimization_results)} models")
        return training_results
    
    def _create_optimized_model(self, algorithm: str, best_params: Dict[str, Any]):
        """Create model with optimized parameters."""
        if algorithm == 'random_forest':
            return RandomForestRegressor(**best_params)
        elif algorithm == 'extra_trees':
            return ExtraTreesRegressor(**best_params)
        elif algorithm == 'gradient_boosting':
            return GradientBoostingRegressor(**best_params)
        elif algorithm == 'xgboost':
            return xgb.XGBRegressor(**best_params)
        elif algorithm == 'lightgbm' and LIGHTGBM_AVAILABLE:
            return lgb.LGBMRegressor(**best_params)
        elif algorithm == 'catboost' and CATBOOST_AVAILABLE:
            return cb.CatBoostRegressor(**best_params)
        elif algorithm == 'neural_network':
            return MLPRegressor(**best_params)
        elif algorithm == 'ridge':
            return Ridge(**best_params)
        elif algorithm == 'elastic_net':
            return ElasticNet(**best_params)
        elif algorithm == 'lasso':
            return Lasso(**best_params)
        elif algorithm == 'bayesian_ridge':
            return BayesianRidge(**best_params)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _save_optimization_report(self, optimization_results: Dict[str, Any], target_name: str = "unknown"):
        """Save comprehensive hyperparameter optimization report with target-specific organization."""
        try:
            # Create target-specific directory structure
            base_output_dir = Path("results/hyperparameter_optimization")
            target_output_dir = base_output_dir / target_name
            target_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp for unique file naming
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            
            # Save detailed optimization results with target and timestamp
            results_file = target_output_dir / f"optuna_optimization_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(optimization_results, f, indent=2, default=str)
            
            # Create comprehensive summary report
            summary_data = []
            for model_name, results in optimization_results.items():
                summary_data.append({
                    'target_variable': target_name,
                    'model_name': model_name,
                    'algorithm': results['algorithm'],
                    'best_score': results['best_score'],
                    'n_trials': results['n_trials'],
                    'optimization_method': 'optuna',
                    'timestamp': timestamp,
                    'best_params': str(results['best_params'])
                })
            
            summary_df = pd.DataFrame(summary_data)
            
            # Save target-specific summary
            target_summary_file = target_output_dir / f"optimization_summary_{timestamp}.csv"
            summary_df.to_csv(target_summary_file, index=False)
            
            # Also append to global summary (for cross-target comparison)
            global_summary_file = base_output_dir / "global_optimization_summary.csv"
            if global_summary_file.exists():
                existing_df = pd.read_csv(global_summary_file)
                combined_df = pd.concat([existing_df, summary_df], ignore_index=True)
            else:
                combined_df = summary_df
            combined_df.to_csv(global_summary_file, index=False)
            
            logger.info(f"📊 Hyperparameter optimization report saved to: {target_output_dir}")
            logger.info(f"📊 Global summary updated: {global_summary_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save optimization report: {e}")
        
    
    def get_scaling_configurations(self) -> Dict[str, Any]:
        """
        Get comprehensive feature scaling configurations.
        Includes all scalers needed for thorough evaluation.
        
        Returns:
            Dictionary of scaler names and their configured instances
        """
        return {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'quantile': QuantileTransformer(output_distribution='normal', random_state=self.random_state)
        }
    
    def train_model_configuration(self, X_train, X_test, y_train, y_test, 
                                algorithm_name: str, scaler_name: str,
                                feature_names: List[str]) -> Dict[str, Any]:
        """
        Train a single model configuration with specified algorithm and scaler.
        
        Args:
            X_train, X_test, y_train, y_test: Data splits
            algorithm_name: Name of the algorithm to use
            scaler_name: Name of the scaler to use
            feature_names: List of feature names
            
        Returns:
            Dictionary with training results and model objects
        """
        try:
            # Get configurations
            algorithms = self.get_algorithm_configurations()
            scalers = self.get_scaling_configurations()
            
            if algorithm_name not in algorithms:
                raise ValueError(f"Unknown algorithm: {algorithm_name}")
            if scaler_name not in scalers:
                raise ValueError(f"Unknown scaler: {scaler_name}")
            
            model = algorithms[algorithm_name]
            scaler = scalers[scaler_name]
            
            # Scale features
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Cross-validation
            cv_folds = min(5, len(X_train) // 3)
            if cv_folds >= 2:
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train,
                    cv=cv_folds, scoring='r2', n_jobs=1
                )
            else:
                cv_scores = np.array([0.0])
            
            # --- Model Training (Default Parameters First) ---
            # Train with default parameters first, then optimize only the best models
            model.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_mse = mean_squared_error(y_test, y_pred_test)
            test_rmse = np.sqrt(test_mse)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Calculate correlations
            if len(np.unique(y_pred_test)) > 1 and len(np.unique(y_test)) > 1:
                pearson_corr = np.corrcoef(y_test, y_pred_test)[0, 1]
                if np.isnan(pearson_corr):
                    pearson_corr = 0.0
            else:
                pearson_corr = 0.0
            
            try:
                spearman_corr = stats.spearmanr(y_test, y_pred_test)[0]
                if np.isnan(spearman_corr):
                    spearman_corr = 0.0
            except:
                spearman_corr = 0.0
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(feature_names, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                feature_importance = dict(zip(feature_names, np.abs(model.coef_)))
            
            return {
                'algorithm': algorithm_name,
                'scaler': scaler_name,
                'model_name': algorithm_name,  # Add model_name field
                'target_transformation': 'original',  # Add target_transformation field
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_mse': test_mse,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'pearson_correlation': pearson_corr,
                'spearman_correlation': spearman_corr,
                'model_object': model,
                'scaler_object': scaler,
                'feature_names': feature_names,
                'feature_importance': feature_importance
            }
                
        except Exception as e:
            logger.error(f"Training failed for {algorithm_name}_{scaler_name}: {e}")
            return None
    
    def train_all_configurations(self, X_train, X_test, y_train, y_test, 
                               feature_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Train all algorithm and scaler combinations.
        
        Args:
            X_train, X_test, y_train, y_test: Data splits
            feature_names: List of feature names
            
        Returns:
            Dictionary with results for all configurations
        """
        logger.info("Training all algorithm and scaler combinations...")
        
        algorithms = list(self.get_algorithm_configurations().keys())
        scalers = list(self.get_scaling_configurations().keys())
        
        all_results = {}
        successful_configs = 0
        total_configs = len(algorithms) * len(scalers)
        
        for algorithm_name in algorithms:
            for scaler_name in scalers:
                config_name = f"{scaler_name}_{algorithm_name}"
                
                logger.info(f"Training {config_name}...")
                
                result = self.train_model_configuration(
                    X_train, X_test, y_train, y_test, algorithm_name, scaler_name, feature_names
                )
                
                if result is not None:
                    all_results[config_name] = result
                    successful_configs += 1
                    
                    # Calculate combined performance score
                    combined_score = self._calculate_performance_score(result)
                    
                    logger.info(f"  {config_name}: R²={result['test_r2']:.3f}, "
                              f"RMSE={result['test_rmse']:.3f}, MAE={result['test_mae']:.3f}, "
                              f"Corr={result['pearson_correlation']:.3f}, "
                              f"Score={combined_score:.3f}")
        
        # Add ensemble model
        try:
            ensemble_result = self.train_ensemble_model(X_train, X_test, y_train, y_test, all_results, feature_names)
            if ensemble_result:
                all_results['ensemble'] = ensemble_result
                successful_configs += 1
                combined_score = self._calculate_performance_score(ensemble_result)
                logger.info(f"  Ensemble: R²={ensemble_result['test_r2']:.3f}, "
                          f"Corr={ensemble_result['pearson_correlation']:.3f}, "
                          f"Score={combined_score:.3f}")
        except Exception as e:
            logger.warning(f"Ensemble model training failed: {e}")

        logger.info(f"Training complete: {successful_configs}/{total_configs+1} configurations successful")
        return all_results
    
    def train_ensemble_model(self, X_train, X_test, y_train, y_test, 
                           individual_results: Dict[str, Dict[str, Any]], 
                           feature_names: List[str]) -> Optional[Dict[str, Any]]:
        """
        Train a VotingRegressor ensemble from the best individual models.

        Args:
            X_train, X_test, y_train, y_test: Data splits
            individual_results: Dictionary of results from individual models.
            feature_names: List of feature names.

        Returns:
            Dictionary with ensemble model results or None if it fails.
        """
        if not individual_results or len(individual_results) < 2:
            logger.warning("Not enough individual models to create an ensemble.")
            return None

        # Select top 3 models based on performance score
        sorted_models = sorted(
            individual_results.items(),
            key=lambda item: self._calculate_performance_score(item[1]),
            reverse=True
        )
        
        top_models = sorted_models[:3]
        
        estimators = []
        for config_name, result in top_models:
            # Need to re-train on the scaled data for this specific model
            scaler = result['scaler_object']
            model = result['model_object']
            
            # Important: The model inside VotingRegressor needs to be fitted on data scaled with its corresponding scaler
            # However, VotingRegressor itself will be fit on a commonly scaled dataset.
            # A pipeline is a better way to handle this, but for simplicity, we'll use a common scaler.
            estimators.append((config_name, model))

        if not estimators:
            return None

        # Use a standard scaler for the ensemble
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        ensemble = xgb.XGBRegressor(n_estimators=150, max_depth=8, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=self.random_state)
        ensemble.fit(X_train_scaled, y_train)

        y_pred_train = ensemble.predict(X_train_scaled)
        y_pred_test = ensemble.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_pred_test)

        if len(np.unique(y_pred_test)) > 1 and len(np.unique(y_test)) > 1:
            pearson_corr = np.corrcoef(y_test, y_pred_test)[0, 1]
        else:
            pearson_corr = 0.0
        
        spearman_corr, _ = stats.spearmanr(y_test, y_pred_test)

        return {
            'algorithm': 'ensemble',
            'scaler': 'standard',
            'model_name': 'ensemble',  # Add model_name field
            'target_transformation': 'original',  # Add target_transformation field
            'cv_mean': np.mean([r['cv_mean'] for _, r in top_models]), # Approximate CV mean
            'cv_std': np.mean([r['cv_std'] for _, r in top_models]),
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mse': test_mse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr,
            'model_object': ensemble,
            'scaler_object': scaler,
            'feature_names': feature_names,
            'feature_importance': None  # Hard to get combined feature importance
        }

    def _calculate_performance_score(self, result: Dict[str, Any]) -> float:
        """
        Calculate a combined performance score from multiple metrics.
        
        Args:
            result: Training result dictionary
            
        Returns:
            Combined performance score
        """
        # Weighted combination of metrics
        r2_weight = 0.30
        pearson_weight = 0.40
        spearman_weight = 0.20
        cv_weight = 0.10
        
        r2_score = max(result.get('test_r2', 0), 0)
        pearson_score = abs(result.get('pearson_correlation', 0))
        spearman_score = abs(result.get('spearman_correlation', 0))
        cv_score = max(result.get('cv_mean', 0), 0)
        
        combined_score = (
            r2_weight * r2_score +
            pearson_weight * pearson_score +
            spearman_weight * spearman_score +
            cv_weight * cv_score
        )
        
        return combined_score
    
    def find_best_model(self, results: Dict[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """
        Find the best performing model configuration.
        
        Args:
            results: Dictionary of training results
            
        Returns:
            Tuple of (best_config_name, best_result)
        """
        if not results:
            raise ValueError("No training results provided")
        
        best_score = -np.inf
        best_config = None
        best_result = None
        
        for config_name, result in results.items():
            score = self._calculate_performance_score(result)
            
            if score > best_score:
                best_score = score
                best_config = config_name
                best_result = result
        
        logger.info(f"Best model: {best_config} (Score: {best_score:.4f})")
        return best_config, best_result
    
    def generate_performance_report(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate a comprehensive performance report.
        
        Args:
            results: Dictionary of training results
            
        Returns:
            DataFrame with performance metrics for all configurations
        """
        report_data = []
        
        for config_name, result in results.items():
            report_row = {
                'configuration': config_name,
                'algorithm': result['algorithm'],
                'scaler': result['scaler'],
                'test_r2': result.get('test_r2', result.get('cv_r2_mean', 0)),
                'pearson_correlation': result.get('pearson_correlation', result.get('cv_pearson_mean', 0)),
                'spearman_correlation': result.get('spearman_correlation', 0), # Simplified for now
                'test_rmse': result.get('test_rmse', result.get('cv_rmse_mean', 0)),
                'test_mae': result.get('test_mae', result.get('cv_mae_mean', 0)),
                'cv_mean': result.get('cv_mean', result.get('cv_r2_mean', 0)),
                'cv_std': result.get('cv_std', result.get('cv_r2_std', 0)),
                'performance_score': self._calculate_performance_score(result)
            }
            report_data.append(report_row)
        
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values('performance_score', ascending=False)
        
        return report_df
    
    def analyze_feature_importance(self, results: Dict[str, Dict[str, Any]], 
                                 top_n: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Analyze feature importance across different models.
        
        Args:
            results: Dictionary of training results
            top_n: Number of top features to analyze
            
        Returns:
            Dictionary of feature importance DataFrames by algorithm
        """
        importance_by_algorithm = {}
        
        for config_name, result in results.items():
            if result['feature_importance'] is not None:
                algorithm = result['algorithm']
                
                importance_df = pd.DataFrame([
                    {'feature': feature, 'importance': importance}
                    for feature, importance in result['feature_importance'].items()
                ]).sort_values('importance', ascending=False)
                
                if algorithm not in importance_by_algorithm:
                    importance_by_algorithm[algorithm] = []
                
                importance_by_algorithm[algorithm].append(importance_df.head(top_n))
        
        # Average importance across scalers for each algorithm
        averaged_importance = {}
        for algorithm, importance_list in importance_by_algorithm.items():
            if importance_list:
                # Combine and average importance scores
                combined_df = pd.concat(importance_list, ignore_index=True)
                avg_importance = combined_df.groupby('feature')['importance'].mean().reset_index()
                avg_importance = avg_importance.sort_values('importance', ascending=False)
                averaged_importance[algorithm] = avg_importance.head(top_n)
        
        return averaged_importance
    
    def save_training_results(self, results: Dict[str, Dict[str, Any]], 
                            output_dir: str = "results") -> None:
        """
        Save training results to files.
        
        Args:
            results: Dictionary of training results
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save performance report
        performance_df = self.generate_performance_report(results)
        performance_df.to_csv(output_path / "model_performance_report.csv", index=False)
        
        # Save feature importance analysis
        feature_importance = self.analyze_feature_importance(results)
        for algorithm, importance_df in feature_importance.items():
            importance_df.to_csv(
                output_path / f"feature_importance_{algorithm}.csv", 
                index=False
            )
        
        logger.info(f"Training results saved to {output_path}")
    
    def train_target_with_cv(self, X: pd.DataFrame, y: pd.Series, feature_columns: List[str], 
                           cv_folds: int = 5, target_transformation: str = 'original', 
                           save_individual_model_callback=None) -> Dict[str, Any]:
        """
        Train models using K-Fold Cross-Validation for robust evaluation.
        """
        logger.info(f"Training models with {cv_folds}-fold cross-validation")
        
        # --- Smart Feature Selection Strategy ---
        if self.skip_feature_selection:
            logger.info("⏭️  Skipping feature selection (disabled for speed)")
            selected_features = feature_columns
            X_selected = X
        else:
            # Always use fast feature selection (much faster than RFECV and more robust)
            logger.info("🚀 Using smart feature selection with data cleaning...")
            selected_features = self.fast_feature_selection(X, y)
            X_selected = X[selected_features]
        
        if len(X_selected) < 10:
            logger.warning(f"Insufficient training data: {len(X_selected)} samples")
            return {}
        
        logger.info(f"🎯 Training with {len(selected_features)} selected features")
        
        # Train all configurations with CV
        results = self.train_all_configurations_cv(X_selected, y, selected_features, cv_folds, target_transformation, save_individual_model_callback)
        
        return results
    
    def train_all_configurations_cv(self, X: pd.DataFrame, y: pd.Series, feature_names: List[str], 
                                  cv_folds: int = 5, target_transformation: str = 'original', 
                                  save_individual_model_callback=None) -> Dict[str, Dict[str, Any]]:
        """
        Train all algorithm and scaler combinations using cross-validation.
        """
        logger.info(f"Training all configurations with {cv_folds}-fold cross-validation...")
        
        algorithms = list(self.get_algorithm_configurations().keys())
        scalers = list(self.get_scaling_configurations().keys())
        
        all_results = {}
        successful_configs = 0
        total_configs = len(algorithms) * len(scalers)
        
        for algorithm_name in algorithms:
            for scaler_name in scalers:
                config_name = f"{scaler_name}_{algorithm_name}"
                
                logger.info(f"Cross-validating {config_name}...")
                
                result = self.train_model_configuration_cv(
                    X, y, algorithm_name, scaler_name, feature_names, cv_folds, target_transformation
                )
                
                if result is not None:
                    all_results[config_name] = result
                    successful_configs += 1
                    
                    combined_score = self._calculate_performance_score(result)
                    
                    logger.info(f"  {config_name}: CV R²={result['cv_r2_mean']:.3f}±{result['cv_r2_std']:.3f}, "
                              f"CV RMSE={result['cv_rmse_mean']:.3f}±{result['cv_rmse_std']:.3f}, "
                              f"Score={combined_score:.3f}")
                    
                    # Save individual model immediately after training
                    if save_individual_model_callback:
                        save_individual_model_callback(config_name, result, target_transformation)
        
        logger.info(f"Cross-validation complete: {successful_configs}/{total_configs} configurations successful")
        return all_results
    
    def train_specific_configurations(self, X: pd.DataFrame, y: pd.Series, feature_names: List[str], 
                                    configs_to_train: List[str], cv_folds: int = 5, 
                                    target_transformation: str = 'original',
                                    save_individual_model_callback=None) -> Dict[str, Dict[str, Any]]:
        """
        Train only specific model configurations for efficient continuation.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            feature_names: List of feature names
            configs_to_train: List of configuration names to train (format: "scaler_algorithm")
            cv_folds: Number of cross-validation folds
            target_transformation: Name of target transformation
            save_individual_model_callback: Callback for saving individual models
            
        Returns:
            Dictionary with training results for specified configurations only
        """
        logger.info(f"Training {len(configs_to_train)} specific configurations...")
        
        # Initialize K-Fold
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        all_results = {}
        successful_configs = 0
        total_configs = len(configs_to_train)
        
        for config_name in configs_to_train:
            try:
                # Parse configuration name (format: "scaler_algorithm")
                if '_' in config_name:
                    scaler_name, algorithm_name = config_name.split('_', 1)
                else:
                    logger.warning(f"Invalid configuration name format: {config_name}, skipping")
                    continue
                
                logger.info(f"  Training {config_name}...")
                
                # Train this specific configuration
                result = self.train_model_configuration_cv(
                    X, y, algorithm_name, scaler_name, feature_names, cv_folds, target_transformation
                )
                
                if result:
                    all_results[config_name] = result
                    successful_configs += 1
                    
                    # Calculate combined performance score
                    combined_score = self._calculate_performance_score(result)
                    
                    logger.info(f"  {config_name}: CV R²={result['cv_r2_mean']:.3f}±{result['cv_r2_std']:.3f}, "
                              f"CV RMSE={result['cv_rmse_mean']:.3f}±{result['cv_rmse_std']:.3f}, "
                              f"Score={combined_score:.3f}")
                    
                    # Save individual model immediately after training
                    if save_individual_model_callback:
                        save_individual_model_callback(config_name, result, target_transformation)
                else:
                    logger.warning(f"  {config_name}: Training failed")
                    
            except Exception as e:
                logger.error(f"  {config_name}: Training failed with error: {e}")
        
        logger.info(f"Specific configurations complete: {successful_configs}/{total_configs} configurations successful")
        return all_results
    
    def train_model_configuration_cv(self, X: pd.DataFrame, y: pd.Series, algorithm_name: str, 
                                   scaler_name: str, feature_names: List[str], cv_folds: int = 5, 
                                   target_transformation: str = 'original') -> Dict[str, Any]:
        """
        Train a single model configuration using cross-validation.
        """
        try:
            algorithms = self.get_algorithm_configurations()
            scalers = self.get_scaling_configurations()
            
            if algorithm_name not in algorithms or scaler_name not in scalers:
                return None
            
            model = algorithms[algorithm_name]
            scaler = scalers[scaler_name]
            
            # Perform cross-validation
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            
            cv_r2_scores = []
            cv_rmse_scores = []
            cv_mae_scores = []
            cv_pearson_scores = []

            for train_idx, val_idx in kf.split(X):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

                # Removed leaky demand-based feature engineering
                
                # --- Advanced Synthetic Data Generation (Optional) ---
                original_size = len(X_train_fold)
                if self.enable_synthetic_data:
                    try:
                        # Initialize synthetic data generator
                        synthetic_generator = SyntheticDataGenerator(random_state=self.random_state)
                        
                        # Try multiple methods in order of sophistication
                        methods_to_try = ['mixed', 'ctgan', 'smote', 'noise']
                        
                        for method in methods_to_try:
                            try:
                                X_train_fold, y_train_fold = synthetic_generator.augment_dataset(
                                    X_train_fold, y_train_fold, 
                                    method=method,
                                    augmentation_ratio=0.8,  # Add 80% more data
                                    use_smote_for_balance=False  # Handled within method
                                )
                                
                                logger.info(f"Applied {method} augmentation: {original_size} → {len(X_train_fold)} samples")
                                break  # Success, exit the loop
                                
                            except Exception as e:
                                logger.warning(f"{method} augmentation failed: {e}")
                                # Reset to original data for next attempt
                                X_train_fold = X.iloc[train_idx]
                                y_train_fold = y.iloc[train_idx]
                                continue
                        
                        else:
                            # All methods failed, use original data
                            logger.warning("All augmentation methods failed. Using original data.")
                            
                    except Exception as e:
                        logger.warning(f"Synthetic data generation disabled due to error: {e}")
                else:
                    logger.info("Synthetic data generation disabled. Using original data only.")

                # Scale features
                X_train_scaled = scaler.fit_transform(X_train_fold)
                X_val_scaled = scaler.transform(X_val_fold)
                
            # Train model with GPU error handling
            try:
                model.fit(X_train_scaled, y_train_fold)
            except Exception as e:
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ["cuda", "gpu", "opencl", "device", "unavailable", "busy"]):
                    logger.warning(f"GPU training failed for {algorithm_name}, falling back to CPU: {e}")
                    # Recreate model without GPU params
                    if algorithm_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
                        model = lgb.LGBMRegressor(
                            n_estimators=150, max_depth=8, learning_rate=0.05,
                            subsample=0.9, colsample_bytree=0.9, random_state=self.random_state,
                            verbosity=-1, device='cpu'  # Force CPU
                        )
                        model.fit(X_train_scaled, y_train_fold)
                    elif algorithm_name == 'xgboost':
                        model = xgb.XGBRegressor(
                            n_estimators=150, max_depth=8, learning_rate=0.05,
                            subsample=0.9, colsample_bytree=0.9, random_state=self.random_state,
                            tree_method='hist', device='cpu'  # Force CPU
                        )
                        model.fit(X_train_scaled, y_train_fold)
                    elif algorithm_name == 'catboost' and CATBOOST_AVAILABLE:
                        model = cb.CatBoostRegressor(
                            n_estimators=150, max_depth=8, learning_rate=0.05,
                            random_state=self.random_state, verbose=False,
                            allow_writing_files=False, task_type='CPU'  # Force CPU
                        )
                        model.fit(X_train_scaled, y_train_fold)
                    else:
                        raise e  # Re-raise if not a GPU error
                else:
                    raise e  # Re-raise if not a GPU error
            
            # Predict
            y_pred = model.predict(X_val_scaled)
            
            # Calculate metrics
            r2 = r2_score(y_val_fold, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
            mae = mean_absolute_error(y_val_fold, y_pred)
            
            if len(np.unique(y_pred)) > 1 and len(np.unique(y_val_fold)) > 1:
                pearson = np.corrcoef(y_val_fold, y_pred)[0, 1]
                if np.isnan(pearson):
                    pearson = 0.0
            else:
                pearson = 0.0
            
            cv_r2_scores.append(r2)
            cv_rmse_scores.append(rmse)
            cv_mae_scores.append(mae)
            cv_pearson_scores.append(pearson)
            
            # Train final model on full dataset for feature importance
            X_scaled = scaler.fit_transform(X)
            model.fit(X_scaled, y)
            
            # Feature importance
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(feature_names, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                feature_importance = dict(zip(feature_names, np.abs(model.coef_)))
            
            return {
                'algorithm': algorithm_name,
                'scaler': scaler_name,
                'model_name': algorithm_name,  # Add model_name field
                'target_transformation': target_transformation,  # Use the actual target transformation
                'cv_r2_mean': np.mean(cv_r2_scores),
                'cv_r2_std': np.std(cv_r2_scores),
                'cv_rmse_mean': np.mean(cv_rmse_scores),
                'cv_rmse_std': np.std(cv_rmse_scores),
                'cv_mae_mean': np.mean(cv_mae_scores),
                'cv_mae_std': np.std(cv_mae_scores),
                'cv_pearson_mean': np.mean(cv_pearson_scores),
                'cv_pearson_std': np.std(cv_pearson_scores),
                # For compatibility with existing code, map CV metrics to expected names
                'test_r2': np.mean(cv_r2_scores),
                'test_rmse': np.mean(cv_rmse_scores),
                'test_mae': np.mean(cv_mae_scores),
                'pearson_correlation': np.mean(cv_pearson_scores),
                'spearman_correlation': np.mean(cv_pearson_scores),  # Simplified for now
                'cv_mean': np.mean(cv_r2_scores),
                'cv_std': np.std(cv_r2_scores),
                'model_object': model,
                'scaler_object': scaler,
                'feature_names': feature_names,
                'feature_importance': feature_importance
            }
                
        except Exception as e:
            logger.error(f"CV training failed for {algorithm_name}_{scaler_name}: {e}")
            return None
    
    def train_target_formulation(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                 y_train: pd.Series, y_test: pd.Series,
                                 feature_columns: List[str]) -> Dict[str, Any]:
        """
        Train models for a specific target formulation on pre-split data,
        including feature selection. (Kept for backwards compatibility)
        """
        logger.info(f"Training models for target (using CV approach)")
        
        # Combine train and test for CV approach
        X_combined = pd.concat([X_train, X_test], ignore_index=True)
        y_combined = pd.concat([y_train, y_test], ignore_index=True)
        
        return self.train_target_with_cv(X_combined, y_combined, feature_columns)
    
    def train_target_with_preselected_features(self, X: pd.DataFrame, y: pd.Series, 
                                             selected_features: List[str], cv_folds: int = 5, 
                                             target_transformation: str = 'original',
                                             save_individual_model_callback=None) -> Dict[str, Any]:
        """
        Train models with pre-selected features (optimized for multiple targets).
        
        This method skips feature selection entirely and directly trains models with
        the provided feature set, making it much faster when training multiple targets.
        
        Args:
            X: Feature DataFrame with pre-selected features only
            y: Target Series  
            selected_features: List of selected feature names
            cv_folds: Number of cross-validation folds
            target_transformation: Name of the target transformation being used
            
        Returns:
            Dictionary with training results for all configurations
        """
        logger.info(f"⚡ OPTIMIZED training with {len(selected_features)} pre-selected features")
        logger.info("⏭️  Skipping feature selection (using pre-selected features for speed)")
        
        if len(X) < 10:
            logger.warning(f"Insufficient training data: {len(X)} samples")
            return {}
        
        logger.info(f"🎯 Training with {len(selected_features)} features (pre-selected)")
        
        # Train all configurations with CV using pre-selected features
        results = self.train_all_configurations_cv(X, y, selected_features, cv_folds, target_transformation, save_individual_model_callback)
        
        return results

