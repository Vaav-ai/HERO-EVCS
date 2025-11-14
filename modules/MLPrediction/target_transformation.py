"""
Target Transformation for EV Charging Demand Prediction
=======================================================

This module provides tools for creating and evaluating multiple target variable
formulations. Using different transformations helps in building more robust
regression models that are less sensitive to the target's distribution.

Key Components:
---------------
- `DemandTargetTransformer`: A class to apply various mathematical transformations
  (log, sqrt, quantile, etc.) to the demand score and evaluate their quality.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
from sklearn.preprocessing import RobustScaler
from scipy.stats import skew, kurtosis
import scipy.stats as stats

logger = logging.getLogger(__name__)

class DemandTargetTransformer:
    """
    Creates and evaluates multiple target variable formulations for robust modeling.
    This class now follows a fit/transform pattern to prevent data leakage.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize the target transformer with standard configurations."""
        self.random_state = random_state
        self.transformers = {}
        self.target_statistics = {}
        
    def fit(self, y_train: pd.Series):
        """
        Fit essential transformers on the training data only.
        Optimized to keep only the most effective transformations.
        
        Args:
            y_train: The training target variable series.
        """
        logger.info("Fitting essential target transformers on training data...")
        
        # Only keep the most effective and stable transformations
        rs = RobustScaler()
        self.transformers['robust'] = rs.fit(y_train.values.reshape(-1, 1))
        
        # Calculate and store parameters for manual transformers
        self.transformers['zscore'] = {'mean': y_train.mean(), 'std': y_train.std()}
        
        logger.info("Essential target transformers fitted successfully.")
        return self

    def transform(self, y: pd.Series, include_original: bool = True, limit_transformations: bool = False) -> pd.DataFrame:
        """
        Apply essential transformations to a target variable series.
        Optimized to keep only the most effective and stable transformations.
        
        Args:
            y: The target variable series to transform (train or test).
            include_original: Whether to include the original target in the output.
            limit_transformations: If True, only use the most effective transformations (log + sqrt).
            
        Returns:
            DataFrame with transformed target formulations.
        """
        if not self.transformers:
            raise RuntimeError("Transformer has not been fitted yet. Call fit() first.")
            
        result = pd.DataFrame(index=y.index)
        if include_original:
            result['target_original'] = y

        if limit_transformations:
            # Use only the most effective transformations for faster training
            result['target_log'] = np.log1p(y)
            logger.info("Using limited target transformations (original, log) for faster training")
        else:
            # Apply essential transformations for comprehensive analysis
            result['target_log'] = np.log1p(y)
            result['target_sqrt'] = np.sqrt(y.abs())

            # Transformations using fitted parameters
            zscore_params = self.transformers['zscore']
            if zscore_params['std'] > 0:
                result['target_zscore'] = (y - zscore_params['mean']) / zscore_params['std']
            else:
                result['target_zscore'] = 0

            # Transformations using fitted scikit-learn objects
            transformer = self.transformers.get('robust')
            if transformer:
                result['target_robust'] = transformer.transform(y.values.reshape(-1, 1)).flatten()
        
        logger.info(f"Applied {len([col for col in result.columns if col.startswith('target_')])} target transformations")
        return result

    def analyze_target_distribution(self, target: pd.Series) -> Dict:
        """
        Analyze the distribution characteristics of the target variable.
        
        Args:
            target: Target variable series
            
        Returns:
            Dictionary with distribution statistics
        """
        logger.info("Analyzing target variable distribution characteristics...")
        
        stats_dict = {
            'count': len(target),
            'mean': float(target.mean()),
            'std': float(target.std()),
            'min': float(target.min()),
            'max': float(target.max()),
            'skewness': float(skew(target)),
            'kurtosis': float(kurtosis(target)),
            'q25': float(target.quantile(0.25)),
            'q50': float(target.quantile(0.50)),
            'q75': float(target.quantile(0.75)),
            'zeros_count': int((target == 0).sum()),
            'near_zeros_count': int((target < 0.01).sum())
        }
        
        # Assess transformation needs
        needs_transform = abs(stats_dict['skewness']) > 1.0 or stats_dict['kurtosis'] > 3.0
        stats_dict['needs_transformation'] = needs_transform
        
        logger.info(f"Target analysis: mean={stats_dict['mean']:.4f}, "
                   f"skew={stats_dict['skewness']:.3f}, "
                   f"transform_needed={needs_transform}")
        
        self.target_statistics = stats_dict
        return stats_dict
    

    def get_target_formulations(self) -> List[str]:
        """
        Get list of available target formulation names.
        Optimized to keep only the most effective transformations.
        
        Returns:
            List of target column names
        """
        return [
            'target_original',
            'target_log', 
            'target_sqrt',
            'target_zscore',
            'target_robust'
        ]
    
    def inverse_transform_predictions(self, predictions: np.ndarray, 
                                    transformation_type: str) -> np.ndarray:
        """
        Apply inverse transformation to model predictions.
        Optimized to handle only essential transformations.
        
        Args:
            predictions: Model predictions array
            transformation_type: Type of transformation to reverse
            
        Returns:
            Inverse-transformed predictions
        """
        if transformation_type == 'original':
            return predictions
        
        elif transformation_type == 'log':
            return np.expm1(predictions)
        
        elif transformation_type == 'sqrt':
            return predictions ** 2
        
        elif transformation_type == 'zscore':
            # Z-score inverse transformation
            zscore_params = self.transformers.get('zscore', {})
            if zscore_params:
                return predictions * zscore_params['std'] + zscore_params['mean']
            else:
                logger.warning("Z-score parameters not found. Returning predictions as-is.")
                return predictions
        
        elif transformation_type == 'robust':
            # Inverse transformation using fitted scikit-learn objects
            transformer = self.transformers.get('robust')
            if transformer and hasattr(transformer, 'inverse_transform'):
                return transformer.inverse_transform(predictions.reshape(-1, 1)).flatten()
            else:
                logger.warning(f"Inverse transformation not available for {transformation_type}. Returning predictions as-is.")
                return predictions
        
        else:
            logger.warning(f"Unknown transformation type: {transformation_type}. Returning predictions as-is.")
            return predictions
    
    def evaluate_transformation_quality(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate the quality of different target transformations.
        Optimized for efficiency while maintaining research value.
        
        Args:
            data: DataFrame with target formulations
            
        Returns:
            DataFrame with transformation quality metrics
        """
        logger.info("Evaluating transformation quality metrics...")
        
        target_columns = [col for col in data.columns if col.startswith('target_')]
        quality_metrics = []
        
        for target_col in target_columns:
            target_data = data[target_col].dropna()
            
            # Sample data for faster computation if dataset is large
            if len(target_data) > 5000:
                target_data = target_data.sample(5000, random_state=self.random_state)
            
            metrics = {
                'transformation': target_col.replace('target_', ''),
                'skewness': skew(target_data),
                'kurtosis': kurtosis(target_data),
                'normality_pvalue': stats.shapiro(target_data)[1] if len(target_data) <= 5000 else 0.0,
                'variance': target_data.var(),
                'range_ratio': (target_data.max() - target_data.min()) / (target_data.std() + 1e-8)
            }
            
            quality_metrics.append(metrics)
        
        quality_df = pd.DataFrame(quality_metrics)
        
        # Rank transformations by quality (lower skewness and kurtosis is better)
        quality_df['skewness_abs'] = abs(quality_df['skewness'])
        quality_df['kurtosis_abs'] = abs(quality_df['kurtosis'])
        quality_df['quality_score'] = 1 / (1 + quality_df['skewness_abs'] + quality_df['kurtosis_abs'])
        quality_df = quality_df.sort_values('quality_score', ascending=False)
        
        logger.info("Transformation quality evaluation complete")
        return quality_df
    
    def recommend_best_transformations(self, data: pd.DataFrame, top_n: int = 3) -> List[str]:
        """
        Recommend the best target transformations based on statistical properties.
        
        Args:
            data: DataFrame with target formulations
            top_n: Number of top transformations to recommend
            
        Returns:
            List of recommended transformation names
        """
        quality_df = self.evaluate_transformation_quality(data)
        
        # Get top N transformations
        top_transformations = quality_df.head(top_n)['transformation'].tolist()
        
        logger.info(f"Recommended transformations: {top_transformations}")
        
        return top_transformations
    
    def create_target_metadata(self, data: pd.DataFrame) -> Dict:
        """
        Create comprehensive metadata about target transformations.
        Optimized for efficiency while maintaining research value.
        
        Args:
            data: DataFrame with target formulations
            
        Returns:
            Dictionary with transformation metadata
        """
        metadata = {
            'original_statistics': self.target_statistics,
            'transformations_created': self.get_target_formulations(),
            'quality_evaluation': self.evaluate_transformation_quality(data).to_dict('records'),
            'recommendations': self.recommend_best_transformations(data),
            'transformation_parameters': {
                name: params for name, params in self.transformers.items()
            }
        }
        
        return metadata

