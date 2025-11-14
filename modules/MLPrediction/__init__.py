"""
EV Charging Demand Prediction Module
====================================

This module provides a complete pipeline for predicting EV charging demand based on
urban infrastructure data. It is designed to be a reusable and extensible
framework for demand modeling.

Key Components:
---------------
- `EVDemandPredictionPipeline`: The main orchestrator class that runs the complete
  pipeline from data loading to model training and evaluation.
- `UrbanFeatureEngineer`: A component for creating domain-specific features based
  on urban planning principles.
- `DemandTargetTransformer`: A component for creating and evaluating multiple
  target variable formulations to ensure robust modeling.
- `ModelTrainer`: A component for training and comparing a wide range
  of regression models, including ensembles.

Usage Example:
--------------
.. code-block:: python

    from modules.MLPrediction import EVDemandPredictionPipeline

    # Initialize the pipeline
    pipeline = EVDemandPredictionPipeline(random_state=42)

    # Run the complete pipeline
    results = pipeline.run_complete_pipeline(
        data_path="path/to/your/training_data.csv",
        target_column="demand_score",
        output_dir="results/"
    )

    # Access results
    print(f"Best model configuration: {results['evaluation']['overall_best']['configuration']}")
    print(f"Best model score: {results['evaluation']['overall_best']['score']:.4f}")

"""

from .demand_prediction_pipeline import EVDemandPredictionPipeline
from .feature_engineering import UrbanFeatureEngineer
from .model_training import ModelTrainer
from .target_transformation import DemandTargetTransformer
from .demand_score_calculator import DemandScoreCalculator
from .portable_osm_extractor import PortableOSMExtractor
# Simple model saving - no complex registry needed
from . import run_feature_selection

__all__ = [
    "EVDemandPredictionPipeline",
    "UrbanFeatureEngineer",
    "ModelTrainer",
    "DemandTargetTransformer",
    "DemandScoreCalculator",
    "PortableOSMExtractor",
    "run_feature_selection"
]
