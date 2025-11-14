#!/usr/bin/env python3
"""
Main Evaluation Runner for EV Charging Station Placement

This script orchestrates the modularized evaluation framework by coordinating
the DataLoader, ExecutionEngine, and MetricsAnalyzer modules.
"""

import os
import sys
import logging
from datetime import datetime
import time
import traceback
from typing import Dict
import warnings
import argparse
import numpy as np
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.abspath('.'))

from evaluation.microplacement_validation.data_loader import DataLoader
from evaluation.microplacement_validation.execution_engine import ExecutionEngine
from evaluation.microplacement_validation.metrics_analyzer import MetricsAnalyzer
from evaluation.microplacement_validation.visualization import ChargingStationVisualizer


class ComprehensiveResearchEvaluation:
    """
    Main evaluation class that orchestrates the modularized evaluation framework.
    
    This class coordinates the DataLoader, ExecutionEngine, and MetricsAnalyzer
    to provide comprehensive evaluation of EV charging station placement methods.
    """
    
    def __init__(self, random_seed=42, grid_id=None, episodes=10, num_chargers=3, 
                 adaptive_mode=True, confidence_threshold=0.95, override_episodes=None):
        """
        Initialize the comprehensive evaluation framework for single grid optimization.
        
        Args:
            random_seed: Random seed for reproducibility
            grid_id: Specific grid ID to use for testing (if None, will auto-select)
            episodes: Number of episodes to run for hybrid methods (base episodes)
            num_chargers: Number of charging stations to place
            adaptive_mode: If True, use adaptive episode limits based on action space size
            confidence_threshold: Confidence threshold for convergence (0.0-1.0)
            override_episodes: Override episodes for testing (if None, uses adaptive calculation)
        """
        self.random_seed = random_seed
        self.grid_id = grid_id
        self.episodes = episodes
        self.num_chargers = num_chargers
        self.adaptive_mode = adaptive_mode
        self.confidence_threshold = confidence_threshold
        self.override_episodes = override_episodes
        self.logger = self._setup_logging()
        
        # Initialize modules
        self.data_loader = DataLoader(random_seed=random_seed, logger=self.logger, grid_id=grid_id)
        self.execution_engine = None  # Will be initialized after data loading
        self.metrics_analyzer = None  # Will be initialized after data loading
        
        self.logger.info(f"🎲 Using random seed: {random_seed}")
        if grid_id:
            self.logger.info(f"🎯 Using specified grid ID: {grid_id}")
        else:
            self.logger.info("🎯 Auto-selecting grid with most data")
        self.logger.info(f"📊 Base episodes: {episodes}")
        self.logger.info(f"🔌 Placing {num_chargers} charging stations")
        self.logger.info(f"📊 Adaptive mode: {adaptive_mode}")
        self.logger.info(f"🎯 Confidence threshold: {confidence_threshold}")
        if override_episodes:
            self.logger.info(f"🔧 Override episodes: {override_episodes}")
        else:
            self.logger.info("🔧 Using adaptive episode calculation")
    
    def _setup_logging(self):
        """Setup comprehensive logging for the evaluation framework."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f"./logs/microplacement_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            ]
        )
        return logging.getLogger(__name__)
    
    def _validate_seed_consistency(self):
        """Validate that seed is properly set across all modules."""
        self.logger.info("🔍 Validating seed consistency...")
        
        # Check numpy random state
        np_state = np.random.get_state()
        self.logger.info(f"  NumPy random state: {np_state[1][0]} (first value)")
        
        # Check Python random state
        import random
        random_state = random.getstate()
        self.logger.info(f"  Python random state: {random_state[1][0]} (first value)")
        
        # Verify seed propagation
        if hasattr(self.data_loader, 'random_seed'):
            self.logger.info(f"  DataLoader seed: {self.data_loader.random_seed}")
        
        self.logger.info("✅ Seed validation completed")
    
    def run_comprehensive_test(self, max_episodes=None):
        """
        Run the comprehensive test for all 6 methods: 3 Hybrid + 3 Baseline.
        
        Args:
            max_episodes: Maximum episodes for hybrid methods (if None, uses self.episodes)
            
        Returns:
            bool: True if all methods completed successfully, False otherwise
        """
        if max_episodes is None:
            max_episodes = self.episodes
        self.logger.info("🚀 Starting Comprehensive Research Evaluation")
        self.logger.info("Testing 6 methods: 3 Hybrid + 3 Baseline with comprehensive research metrics")
        self.logger.info(f"Configuration: {self.num_chargers} chargers, {max_episodes} episodes per hybrid method")
        
        # Create logs directory
        os.makedirs("./logs", exist_ok=True)
        
        try:
            # Step 1: Load test data
            self.logger.info("\n" + "="*60)
            self.logger.info("STEP 1: LOADING TEST DATA")
            self.logger.info("="*60)
            
            if not self.data_loader.load_test_data():
                self.logger.error("❌ Failed to load test data")
                return False
            
            # Validate data
            if not self.data_loader.validate_data():
                self.logger.error("❌ Data validation failed")
                return False
            
            self.logger.info("✅ Test data loaded and validated successfully")
            
            # Step 2: Initialize execution engine and metrics analyzer
            self.logger.info("\n" + "="*60)
            self.logger.info("STEP 2: INITIALIZING MODULES")
            self.logger.info("="*60)
            
            test_data = self.data_loader.get_test_data()
            self.execution_engine = ExecutionEngine(
                test_data=test_data,
                random_seed=self.random_seed,
                logger=self.logger,
                episodes=self.episodes,
                adaptive_mode=self.adaptive_mode,
                confidence_threshold=self.confidence_threshold
            )
            
            self.metrics_analyzer = MetricsAnalyzer(
                test_data=test_data,
                random_seed=self.random_seed,
                logger=self.logger
            )
            
            self.logger.info("✅ Modules initialized successfully")
            
            # Step 3: Run all methods
            self.logger.info("\n" + "="*60)
            self.logger.info("STEP 3: RUNNING ALL 6 METHODS")
            self.logger.info("="*60)
            
            # Run hybrid methods with adaptive episodes
            self.logger.info("🔀 Running Hybrid Methods (with adaptive episodes)...")
            hybrid_results = self.execution_engine.run_hybrid_methods(
                self.num_chargers, 
                max_episodes=self.override_episodes
            )
            
            # Log exploration statistics for hybrid methods
            self.execution_engine._log_exploration_statistics(hybrid_results)
            
            # Run baseline methods
            self.logger.info("🧪 Running Baseline Methods...")
            baseline_results = self.execution_engine.run_baseline_methods(self.num_chargers)
            
            # Combine all results
            all_results = {**hybrid_results, **baseline_results}
            
            # Step 4: Calculate comparison metrics
            self.logger.info("\n" + "="*60)
            self.logger.info("STEP 4: CALCULATING COMPARISON METRICS")
            self.logger.info("="*60)
            
            comparison_metrics = self.metrics_analyzer.calculate_comparison_metrics(all_results)
            
            # Step 5: Calculate comprehensive research metrics
            self.logger.info("\n" + "="*60)
            self.logger.info("STEP 5: CALCULATING COMPREHENSIVE RESEARCH METRICS")
            self.logger.info("="*60)
            
            research_metrics = {}
            for method_name, result in all_results.items():
                research_metrics[method_name] = self.metrics_analyzer.calculate_comprehensive_research_metrics(method_name, result)
            
            # Step 6: Generate reports and save results
            self.logger.info("\n" + "="*60)
            self.logger.info("STEP 6: GENERATING REPORTS AND SAVING RESULTS")
            self.logger.info("="*60)
            
            # Generate comprehensive research report
            self.metrics_analyzer.generate_comprehensive_research_report(all_results, comparison_metrics, research_metrics)
            
            # Save comprehensive results
            self.metrics_analyzer.save_comprehensive_results(all_results, comparison_metrics, research_metrics)
            
            # Step 7: Create visualizations
            self.logger.info("\n" + "="*60)
            self.logger.info("STEP 7: CREATING VISUALIZATIONS")
            self.logger.info("="*60)
            
            self._create_single_grid_visualizations(all_results, comparison_metrics, research_metrics)
            
            # Step 6: Summary
            self.logger.info("\n" + "="*60)
            self.logger.info("EVALUATION SUMMARY")
            self.logger.info("="*60)
            
            successful_methods = sum(1 for r in all_results.values() if 'error' not in r)
            total_methods = len(all_results)
            
            self.logger.info(f"✅ Successfully completed: {successful_methods}/{total_methods} methods")
            
            if successful_methods == total_methods:
                self.logger.info("🎉 ALL METHODS COMPLETED SUCCESSFULLY!")
                self.logger.info("Check test_results/ directory for comprehensive research results and metrics.")
                return True
            else:
                failed_methods = total_methods - successful_methods
                self.logger.warning(f"⚠️ {failed_methods} methods failed - check logs for details")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Evaluation failed with error: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _create_single_grid_visualizations(self, all_results: Dict, comparison_metrics: Dict, research_metrics: Dict):
        """Create visualizations for single grid evaluation."""
        self.logger.info("🎨 Creating single grid visualizations...")
        
        try:
            # Initialize visualizer
            visualizer = ChargingStationVisualizer(logger=self.logger)
            
            # Get test data for grid information
            test_data = self.data_loader.get_test_data()
            grid_id = test_data.get('test_grid_id', 'unknown')
            
            # Create single grid visualization
            self.logger.info(f"📊 Creating visualization for grid {grid_id}...")
            
            # Create a mock station allocation for this single grid
            station_allocations = {grid_id: self.num_chargers}
            
            # Create grid data from test data bounds
            grid_bounds = test_data.get('grid_bounds', {})
            city_grids_data = [{
                'grid_id': grid_id,
                'min_lat': grid_bounds.get('min_lat', 42.28),
                'max_lat': grid_bounds.get('max_lat', 42.29),
                'min_lon': grid_bounds.get('min_lon', -83.74),
                'max_lon': grid_bounds.get('max_lon', -83.73)
            }]
            
            # Create comprehensive visualization
            comprehensive_path = visualizer.visualize_all_stations_on_map(
                station_allocations=station_allocations,
                city_grids_data=city_grids_data,
                results_data={grid_id: {
                    'num_stations': self.num_chargers,
                    'results': all_results,
                    'comparison_metrics': comparison_metrics,
                    'test_data_summary': {
                        'trajectory_count': len(test_data.get('trajectory_df', [])),
                        'grid_bounds': grid_bounds
                    }
                }},
                title=f"Single Grid Evaluation - Grid {grid_id} ({self.num_chargers} stations)"
            )
            
            # Create single grid detailed visualization
            detailed_path = visualizer.visualize_single_grid_results(
                grid_id=grid_id,
                station_count=self.num_chargers,
                results_data={
                    'results': all_results,
                    'comparison_metrics': comparison_metrics,
                    'test_data_summary': {
                        'trajectory_count': len(test_data.get('trajectory_df', [])),
                        'grid_bounds': grid_bounds
                    }
                }
            )
            
            self.logger.info(f"✅ Comprehensive visualization: {comprehensive_path}")
            self.logger.info(f"✅ Detailed grid visualization: {detailed_path}")
            self.logger.info(f"✅ Single grid visualizations completed!")
            
        except Exception as e:
            self.logger.error(f"❌ Single grid visualization creation failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Continue execution even if visualization fails


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="EV Charging Station Placement Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (auto-select grid, 10 episodes, 3 chargers)
  python run_evaluation.py
  
  # Run with specific grid ID and 5 chargers
  python run_evaluation.py --grid-id "cell_45" --chargers 5
  
  # Run with more episodes for better convergence
  python run_evaluation.py --episodes 20 --chargers 4
  
  # Run with custom random seed and more chargers
  python run_evaluation.py --seed 123 --episodes 15 --chargers 5
  
  # Run with fewer episodes for quick testing
  python run_evaluation.py --episodes 5 --chargers 3 --grid-id "cell_45"
  
  # List available grids
  python run_evaluation.py --list-grids
        """
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--grid-id', '-g',
        type=str,
        default=None,
        help='Specific grid ID to use for testing (default: auto-select)'
    )
    
    parser.add_argument(
        '--episodes', '-e',
        type=int,
        default=10,
        help='Number of episodes to run for hybrid methods (default: 10)'
    )
    
    parser.add_argument(
        '--chargers', '-c',
        type=int,
        default=3,
        help='Number of charging stations to place (default: 3)'
    )
    
    parser.add_argument(
        '--list-grids', '-l',
        action='store_true',
        help='List available grids and exit'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--adaptive-mode',
        action='store_true',
        default=True,
        help='Use adaptive episode calculation based on action space size (default: True)'
    )
    
    parser.add_argument(
        '--no-adaptive-mode',
        action='store_true',
        help='Disable adaptive episode calculation (use fixed scaling)'
    )
    
    parser.add_argument(
        '--confidence-threshold', '--conf',
        type=float,
        default=0.95,
        help='Confidence threshold for convergence (0.0-1.0, default: 0.95)'
    )
    
    parser.add_argument(
        '--override-episodes', '-o',
        type=int,
        default=None,
        help='Override episodes for testing (disables adaptive calculation)'
    )
    
    return parser.parse_args()


def main():
    """Main function to run the comprehensive evaluation."""
    args = parse_arguments()
    
    # Handle list grids option
    if args.list_grids:
        print("📋 Listing available grids...")
        try:
            from evaluation.microplacement_validation.data_loader import DataLoader
            data_loader = DataLoader(random_seed=args.seed)
            grids = data_loader.list_available_grids()
            if grids:
                print("\n✅ Grid listing completed successfully!")
            else:
                print("\n❌ No grids found!")
                sys.exit(1)
        except Exception as e:
            print(f"\n❌ Error listing grids: {e}")
            sys.exit(1)
        return
    
    # Handle adaptive mode
    adaptive_mode = args.adaptive_mode and not args.no_adaptive_mode
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print(f"🎲 Using random seed: {args.seed}")
    if args.grid_id:
        print(f"🎯 Using specified grid ID: {args.grid_id}")
    else:
        print("🎯 Auto-selecting grid with most data")
    print(f"📊 Base episodes: {args.episodes}")
    print(f"🔌 Placing {args.chargers} charging stations")
    print(f"📊 Adaptive mode: {adaptive_mode}")
    print(f"🎯 Confidence threshold: {args.confidence_threshold}")
    if args.override_episodes:
        print(f"🔧 Override episodes: {args.override_episodes}")
    else:
        print("🔧 Using adaptive episode calculation")
    
    # Create and run evaluation
    evaluation = ComprehensiveResearchEvaluation(
        random_seed=args.seed,
        grid_id=args.grid_id,
        episodes=args.episodes,
        num_chargers=args.chargers,
        adaptive_mode=adaptive_mode,
        confidence_threshold=args.confidence_threshold,
        override_episodes=args.override_episodes
    )
    success = evaluation.run_comprehensive_test()
    
    if success:
        print("\n✅ ALL METHODS TESTED SUCCESSFULLY!")
        print("Check test_results/ directory for comprehensive research results and metrics.")
    else:
        print("\n❌ SOME METHODS FAILED!")
        print("Check logs for detailed error information.")
        sys.exit(1)


if __name__ == "__main__":
    main()
