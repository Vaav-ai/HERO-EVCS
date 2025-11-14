#!/usr/bin/env python3
"""
Standalone Feature Selection Script for EV Charging Demand Prediction

This script performs feature selection once and saves the selected features
for reproducible use across multiple pipeline runs.

Usage:
    python run_feature_selection.py --data-path data.csv --output-dir results/
"""

import argparse
import logging
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from .demand_prediction_pipeline import EVDemandPredictionPipeline
from .model_training import ModelTrainer
from modules.utils.log_configs import setup_logging

def main():
    """Main function to perform feature selection once."""
    parser = argparse.ArgumentParser(
        description="Perform feature selection once for reproducible pipeline runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="results/enhanced_training_data_all_variations.csv",
        help="Path to the training data CSV file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/feature_selection_results",
        help="Directory to save feature selection results."
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample size for feature selection (use full dataset if not specified)."
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Maximum number of features to select (auto-determined if not specified)."
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_file_name="feature_selection.log")
    logger = logging.getLogger(__name__)
    
    logger.info("🔬 STARTING FEATURE SELECTION FOR REPRODUCIBILITY 🔬")
    logger.info(f"Configuration:")
    logger.info(f"  - Data Path: {args.data_path}")
    logger.info(f"  - Output Directory: {args.output_dir}")
    logger.info(f"  - Random State: {args.random_state}")
    logger.info(f"  - Sample Size: {'Full dataset' if args.sample_size is None else f'{args.sample_size:,} samples'}")
    logger.info(f"  - Max Features: {args.max_features or 'Auto-determined'}")
    logger.info("-" * 50)
    
    try:
        # Load data
        logger.info(f"Loading data from {args.data_path}...")
        data = pd.read_csv(args.data_path)
        logger.info(f"Loaded data: {len(data):,} rows, {len(data.columns)} columns")
        
        # Sample data for feature selection if needed
        if args.sample_size is not None and len(data) > args.sample_size:
            logger.info(f"Sampling {args.sample_size:,} rows for feature selection...")
            data = data.sample(n=args.sample_size, random_state=args.random_state)
            logger.info(f"Sampled data: {len(data):,} rows")
        elif args.sample_size is None:
            logger.info(f"Using full dataset for feature selection: {len(data):,} rows")
        
        # Initialize pipeline for feature engineering
        pipeline = EVDemandPredictionPipeline(
            random_state=args.random_state,
            enable_synthetic_data=False,  # No synthetic data for feature selection
            use_gpu=False,  # Disable GPU for consistency
            skip_feature_selection=False,  # We want to do feature selection
            enable_feature_caching=False  # Disable caching for consistency
        )
        
        # Apply feature engineering
        logger.info("Applying feature engineering...")
        engineered_data = pipeline.prepare_features(data, skip_spatial_features=True)
        logger.info(f"Feature engineering complete: {len(data.columns)} → {len(engineered_data.columns)} features")
        
        # Prepare features and target
        logger.info("Preparing features and target for feature selection...")
        
        # Find all demand score columns
        all_demand_score_cols = [col for col in engineered_data.columns if col.startswith('demand_score')]
        
        # Define leaky columns (should be excluded from features)
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
        
        # Find all actual metadata columns by checking for prefixes
        grid_metadata_cols = []
        for col in engineered_data.columns:
            for base_col in base_metadata_cols:
                if col.startswith(base_col):
                    grid_metadata_cols.append(col)
        
        exclude_cols = ['grid_id'] + leaky_columns + list(set(grid_metadata_cols))
        feature_columns = [col for col in engineered_data.columns if col not in exclude_cols and not col.startswith('target_')]
        
        X = engineered_data[feature_columns]
        y = engineered_data['demand_score_balanced']  # Use balanced target for feature selection
        
        logger.info(f"Prepared features: {len(X)} samples with {len(feature_columns)} features")
        
        # Perform feature selection
        logger.info("Performing feature selection...")
        trainer = ModelTrainer(
            random_state=args.random_state,
            enable_synthetic_data=False,
            use_gpu=False,
            skip_feature_selection=False
        )
        
        # Determine optimal number of features
        if args.max_features is None:
            total_features = len(feature_columns)
            if total_features <= 20:
                max_features = total_features
            elif total_features <= 50:
                max_features = max(15, min(30, total_features // 2))
            else:
                max_features = max(25, min(40, total_features // 2))
        else:
            max_features = args.max_features
        
        logger.info(f"Selecting top {max_features} features from {len(feature_columns)} available features")
        
        # Perform feature selection and get detailed metrics
        selected_features, feature_metrics = trainer.fast_feature_selection(X, y, max_features=max_features, return_metrics=True)
        
        logger.info(f"✅ Feature selection complete: {len(feature_columns)} → {len(selected_features)} features")
        logger.info(f"Selected features: {selected_features[:10]}{'...' if len(selected_features) > 10 else ''}")
        
        # Create output directory
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save selected features with comprehensive metrics
        feature_selection_results = {
            'selected_features': selected_features,
            'all_available_features': feature_columns,
            'excluded_features': [col for col in feature_columns if col not in selected_features],
            'feature_selection_metadata': {
                'total_features_available': len(feature_columns),
                'features_selected': len(selected_features),
                'selection_ratio': len(selected_features) / len(feature_columns),
                'max_features_requested': max_features,
                'selection_method': 'mutual_information_with_cleaning',
                'random_state': args.random_state,
                'sample_size_used': len(X),
                'target_column_used': 'demand_score_balanced',
                'selection_timestamp': pd.Timestamp.now().isoformat(),
                'pipeline_version': '2.0_reproducible_feature_selection'
            },
            'comprehensive_metrics': feature_metrics
        }
        
        # Save results as JSON
        results_file = output_path / "selected_features.json"
        with open(results_file, 'w') as f:
            json.dump(feature_selection_results, f, indent=2)
        
        # Save selected features as a simple text file for easy reading
        features_file = output_path / "selected_features.txt"
        with open(features_file, 'w') as f:
            f.write("Selected Features for Reproducible Pipeline\n")
            f.write("=" * 50 + "\n\n")
            for i, feature in enumerate(selected_features, 1):
                f.write(f"{i:2d}. {feature}\n")
        
        # Save feature importance scores if available
        try:
            # Get feature importance from the trainer
            if hasattr(trainer, '_last_feature_importance'):
                importance_data = []
                for feature in feature_columns:
                    importance_data.append({
                        'feature': feature,
                        'importance_score': trainer._last_feature_importance.get(feature, 0.0),
                        'selected': feature in selected_features
                    })
                
                importance_df = pd.DataFrame(importance_data)
                importance_df = importance_df.sort_values('importance_score', ascending=False)
                
                importance_file = output_path / "feature_importance_scores.csv"
                importance_df.to_csv(importance_file, index=False)
                logger.info(f"Feature importance scores saved to: {importance_file}")
        except Exception as e:
            logger.warning(f"Could not save feature importance scores: {e}")
        
        # Generate comprehensive report
        report_lines = []
        report_lines.append("# Comprehensive Feature Selection Results for Reproducible Pipeline")
        report_lines.append("")
        report_lines.append(f"**Selection Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Basic Summary
        report_lines.append("## 📊 Selection Summary")
        report_lines.append(f"- **Total Features Available**: {len(feature_columns)}")
        report_lines.append(f"- **Features Selected**: {len(selected_features)}")
        report_lines.append(f"- **Selection Ratio**: {len(selected_features)/len(feature_columns):.2%}")
        report_lines.append(f"- **Selection Method**: Mutual Information with Data Cleaning")
        report_lines.append(f"- **Random State**: {args.random_state}")
        report_lines.append(f"- **Sample Size Used**: {len(X):,}")
        report_lines.append("")
        
        # Detailed Metrics from feature selection
        if feature_metrics:
            metrics = feature_metrics
            report_lines.append("## 🔍 Detailed Selection Metrics")
            report_lines.append("")
            
            # Selection Summary
            selection_summary = metrics['selection_summary']
            report_lines.append("### Selection Process")
            report_lines.append(f"- **Initial Features**: {selection_summary['initial_features']}")
            report_lines.append(f"- **After Zero Variance Removal**: {selection_summary['features_after_cleaning']}")
            report_lines.append(f"- **Zero Variance Features Removed**: {selection_summary['zero_variance_removed']}")
            report_lines.append(f"- **Highly Correlated Features Removed**: {selection_summary['highly_correlated_removed']}")
            report_lines.append(f"- **Final Selected Features**: {selection_summary['selected_features']}")
            report_lines.append(f"- **Selection Efficiency**: {selection_summary['selection_ratio']:.2%}")
            report_lines.append("")
            
            # Target Statistics
            target_stats = metrics['target_statistics']
            report_lines.append("### Target Variable Statistics")
            report_lines.append(f"- **Mean**: {target_stats['mean']:.4f}")
            report_lines.append(f"- **Standard Deviation**: {target_stats['std']:.4f}")
            report_lines.append(f"- **Min**: {target_stats['min']:.4f}")
            report_lines.append(f"- **Max**: {target_stats['max']:.4f}")
            report_lines.append(f"- **Skewness**: {target_stats['skewness']:.4f}")
            report_lines.append(f"- **Kurtosis**: {target_stats['kurtosis']:.4f}")
            report_lines.append("")
            
            # Selection Criteria
            criteria = metrics['selection_criteria']
            report_lines.append("### Selection Criteria")
            report_lines.append(f"- **Variance Threshold**: {criteria['variance_threshold']}")
            report_lines.append(f"- **Correlation Threshold**: {criteria['correlation_threshold']}")
            report_lines.append(f"- **Selection Method**: {criteria['selection_method']}")
            report_lines.append(f"- **Random State**: {criteria['random_state']}")
            report_lines.append("")
            
            # Top Features with Scores
            selected_scores = metrics['feature_scores']['selected_features']
            sorted_selected = sorted(selected_scores.items(), key=lambda x: x[1], reverse=True)
            report_lines.append("### Selected Features with Mutual Information Scores")
            for i, (feature, score) in enumerate(sorted_selected, 1):
                report_lines.append(f"{i:2d}. **{feature}**: {score:.6f}")
            report_lines.append("")
            
            # Data Quality Issues
            data_quality = metrics['data_quality']
            if data_quality['zero_variance_features']:
                report_lines.append("### Data Quality Issues Detected")
                report_lines.append(f"- **Zero Variance Features Removed**: {data_quality['zero_variance_features']}")
                if data_quality['highly_correlated_pairs']:
                    report_lines.append("- **Highly Correlated Pairs Removed**:")
                    for col1, col2, corr_val in data_quality['highly_correlated_pairs'][:5]:  # Show top 5
                        report_lines.append(f"  - {col1} ↔ {col2} (r={corr_val:.3f})")
                    if len(data_quality['highly_correlated_pairs']) > 5:
                        report_lines.append(f"  - ... and {len(data_quality['highly_correlated_pairs']) - 5} more pairs")
                report_lines.append("")
        
        # Selected Features List
        report_lines.append("## 📋 Selected Features List")
        for i, feature in enumerate(selected_features, 1):
            report_lines.append(f"{i:2d}. {feature}")
        report_lines.append("")
        
        # Usage Instructions
        report_lines.append("## 🚀 Usage Instructions")
        report_lines.append("To use these selected features in your pipeline, run:")
        report_lines.append("```bash")
        report_lines.append("python -m modules.MLPrediction.run_pipeline \\")
        report_lines.append("    --data-path 'your_data.csv' \\")
        report_lines.append("    --output-dir 'results/' \\")
        report_lines.append("    --target-column 'demand_score_balanced' \\")
        report_lines.append("    --skip-spatial-features \\")
        report_lines.append("    --skip-synthetic-data \\")
        report_lines.append("    --skip-rfecv-validation \\")
        report_lines.append("    --random-state 42 \\")
        report_lines.append("    --skip-feature-selection \\")
        report_lines.append("    --pre-selected-features 'results/feature_selection_results/selected_features.json'")
        report_lines.append("```")
        report_lines.append("")
        
        # Paper Documentation
        report_lines.append("## 📄 Paper Documentation")
        report_lines.append("### Feature Selection Methodology")
        report_lines.append("1. **Data Cleaning**: Removed zero-variance features and highly correlated features (>0.95)")
        report_lines.append("2. **Feature Selection**: Used mutual information regression with SelectKBest")
        report_lines.append("3. **Selection Criteria**: Selected top features based on mutual information scores")
        report_lines.append("4. **Reproducibility**: Fixed random state (42) for consistent results")
        report_lines.append("")
        report_lines.append("### Key Statistics for Paper")
        report_lines.append(f"- **Total Features Evaluated**: {len(feature_columns)}")
        report_lines.append(f"- **Features Selected**: {len(selected_features)}")
        report_lines.append(f"- **Selection Method**: Mutual Information Regression")
        report_lines.append(f"- **Data Quality Controls**: Zero variance removal, correlation filtering")
        report_lines.append(f"- **Reproducibility**: Random state = {args.random_state}")
        
        report_file = output_path / "feature_selection_report.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info("=" * 60)
        logger.info("🎉 FEATURE SELECTION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"📊 Selected {len(selected_features)} features from {len(feature_columns)} available")
        logger.info(f"📁 Results saved to: {output_path}")
        logger.info(f"📋 Report: {report_file}")
        logger.info(f"🔧 Features file: {results_file}")
        logger.info(f"📝 Features list: {features_file}")
        logger.info("")
        logger.info("✅ You can now use these features for reproducible pipeline runs!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Feature selection failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
