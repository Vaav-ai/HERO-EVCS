#!/usr/bin/env python3
"""
Demand Predictor for EV Charging Demand Prediction

This module provides a simple interface for loading and using trained demand prediction models.
"""

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

class DemandPredictor:
    """
    Simple demand predictor that loads and uses trained models.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the demand predictor.
        
        Args:
            model_path: Path to the trained model pickle file. If None, uses default model.
        """
        self.model_path = model_path or "models/global_demand_model.pkl"
        self.model = None
        self.feature_columns = None
        self.model_metadata = None
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model from pickle file."""
        try:
            model_path = Path(self.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            logger.info(f"Loading model from: {self.model_path}")
            with open(model_path, 'rb') as f:
                model_package = pickle.load(f)
            
            # Extract model and metadata
            if isinstance(model_package, dict):
                # Check if it's the new packaged format with 'model_object' key
                if 'model_object' in model_package:
                    # New format: model_package contains model_object directly
                    self.model = model_package.get('model_object')
                    self.feature_columns = model_package.get('feature_columns', [])
                    self.model_metadata = model_package.get('feature_metadata', {})
                elif 'model' in model_package and isinstance(model_package['model'], dict):
                    # Alternative format: model_package['model'] contains the model data
                    model_data = model_package['model']
                    self.model = model_data.get('model_object')
                    self.feature_columns = model_data.get('feature_names', model_package.get('feature_columns', []))
                    self.model_metadata = model_package.get('feature_metadata', {})
                else:
                    # Old format: model_package contains the model directly
                    self.model = model_package.get('model')
                    self.feature_columns = model_package.get('feature_columns', [])
                    self.model_metadata = model_package.get('feature_metadata', {})
            else:
                # Assume it's just the model
                self.model = model_package
                self.feature_columns = []
            
            logger.info(f"Model loaded successfully")
            if self.feature_columns:
                logger.info(f"Model expects {len(self.feature_columns)} features")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _get_expected_features(self) -> List[str]:
        """
        Get the list of features expected by the model.
        
        Returns:
            List of feature column names
        """
        if self.feature_columns:
            return self.feature_columns
        
        # If no feature columns specified, try to get them from the model
        if hasattr(self.model, 'feature_names_in_'):
            return list(self.model.feature_names_in_)
        
        # Fallback: return empty list
        logger.warning("No feature columns specified in model")
        return []
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the loaded model.
        
        Args:
            features: DataFrame with features for prediction
            
        Returns:
            Array of predicted demand scores
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Get expected features
        expected_features = self._get_expected_features()
        
        if expected_features:
            # Check if all expected features are present
            missing_features = set(expected_features) - set(features.columns)
            if missing_features:
                raise ValueError(f"Missing required OSM feature columns: {sorted(missing_features)}")
            
            # Select only the expected features in the correct order
            prediction_features = features[expected_features]
        else:
            # Use all available features
            prediction_features = features
        
        # Make predictions
        try:
            predictions = self.model.predict(prediction_features)
            logger.info(f"Made predictions for {len(predictions)} samples")
            return predictions
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_path': self.model_path,
            'model_type': type(self.model).__name__ if self.model else None,
            'feature_count': len(self.feature_columns) if self.feature_columns else 0,
            'feature_columns': self.feature_columns,
            'metadata': self.model_metadata
        }
        return info
