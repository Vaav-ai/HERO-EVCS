#!/usr/bin/env python3
"""
Comprehensive All-Grids Evaluation Runner for EV Charging Station Placement

This script orchestrates the evaluation framework to run on all grids with
station allocation from ML demand prediction, adaptive episode calculation,
and comprehensive debugging.
"""

import os
import sys
import logging
from datetime import datetime
import time
import traceback
import warnings
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
import json
from multiprocessing import Pool, cpu_count
from functools import partial
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.abspath('.'))

from evaluation.microplacement_validation.data_loader import DataLoader
from evaluation.microplacement_validation.execution_engine import ExecutionEngine
from evaluation.microplacement_validation.metrics_analyzer import MetricsAnalyzer
from evaluation.microplacement_validation.visualization import ChargingStationVisualizer, visualize_all_stations_placement, visualize_exact_station_locations

# Import seed utilities to ensure consistency
from modules.utils.seed_utils import set_global_seeds, validate_seed_consistency


class AllGridsComprehensiveEvaluation:
    """
    Comprehensive evaluation class that runs evaluation on all grids with station allocation.
    
    This class coordinates the DataLoader, ExecutionEngine, and MetricsAnalyzer
    to provide comprehensive evaluation of EV charging station placement methods
    across all grids with ML-based station allocation.
    """
    
    def __init__(self, random_seed=42, total_station_budget=150, adaptive_mode=True, 
                 confidence_threshold=0.95, override_episodes=None, num_workers=None, resume=True):
        """
        Initialize the comprehensive evaluation framework for all grids.
        
        Args:
            random_seed: Random seed for reproducibility
            total_station_budget: Total number of charging stations to allocate across all grids
            adaptive_mode: If True, use adaptive episode limits based on action space size
            confidence_threshold: Confidence threshold for convergence (0.0-1.0)
            override_episodes: Override episodes for testing (if None, uses adaptive calculation)
            num_workers: Number of parallel workers (None = auto-detect, 1 = sequential)
        """
        self.random_seed = random_seed
        self.total_station_budget = total_station_budget
        self.adaptive_mode = adaptive_mode
        self.confidence_threshold = confidence_threshold
        self.override_episodes = override_episodes
        self.num_workers = num_workers if num_workers is not None else max(1, cpu_count() - 1)
        self.resume = resume
        self.logger = self._setup_logging()
        
        # Set global seeds for consistency with main.py
        set_global_seeds(random_seed)
        validate_seed_consistency(random_seed, self.logger)
        
        # Initialize modules
        self.data_loader = DataLoader(
            random_seed=random_seed, 
            logger=self.logger, 
            total_station_budget=total_station_budget
        )
        self.execution_engine = None  # Will be initialized after data loading
        self.metrics_analyzer = None  # Will be initialized after data loading
        
        self.logger.info(f"🎲 Using random seed: {random_seed}")
        self.logger.info(f"🎯 Total station budget: {total_station_budget}")
        self.logger.info(f"📊 Adaptive mode: {adaptive_mode}")
        self.logger.info(f"🎯 Confidence threshold: {confidence_threshold}")
        self.logger.info(f"⚡ Parallel workers: {self.num_workers} (CPU cores: {cpu_count()})")
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
                logging.FileHandler(f"./logs/all_grids_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            ]
        )
        return logging.getLogger(__name__)
    
    def _calculate_adaptive_episodes(self, num_stations: int, grid_data: Dict, actual_action_space: int = None) -> int:
        """
        Calculate episodes based on ACTUAL action space from heuristic-based sampling.
        
        Highly optimized, time-efficient formula:
        - Target: 4 samples per action for confidence intervals (minimal but valid)
        - Coverage: 20% of actual action space (focused exploration-exploitation)
        - Bounds: [200, 800] for maximum computational efficiency
        
        Args:
            num_stations: Number of charging stations
            grid_data: Grid information (for complexity if needed)
            actual_action_space: ACTUAL number of placement combinations (if known)
        """
        try:
            # If we don't have actual action space yet, estimate conservatively
            if actual_action_space is None:
                # Conservative estimate based on increased heuristic candidates
                # We now generate 15-30x candidates per method
                candidates_per_method = num_stations * 20  # Use middle estimate
                num_methods = 3  # kmeans, random, uniform
                total_candidates = candidates_per_method * num_methods
                
                # Estimate combinations (capped at 2000 from intelligent sampling)
                import math
                estimated_combinations = min(math.comb(total_candidates, num_stations), 2000)
                actual_action_space = estimated_combinations
                
                self.logger.info(f"📊 Estimating action space: {actual_action_space} (will use actual from bandits)")
            else:
                self.logger.info(f"📊 Using ACTUAL action space from bandits: {actual_action_space}")
            
            # Highly optimized, time-efficient formula
            MIN_SAMPLES_PER_ACTION = 4  # Minimal but valid for confidence intervals (n≥4)
            COVERAGE_TARGET = 0.2  # 20% of action space (highly focused exploration-exploitation)
            
            # Actions to sample (20% coverage)
            actions_to_sample = int(actual_action_space * COVERAGE_TARGET)
            
            # Episodes needed: each sampled action gets 4 samples
            target_episodes = actions_to_sample * MIN_SAMPLES_PER_ACTION
            
            # Highly optimized bounds for maximum efficiency
            MIN_EPISODES = 200  # Minimum for any configuration
            MAX_EPISODES = 800  # Maximum computational limit for speed
            
            final_episodes = max(MIN_EPISODES, min(target_episodes, MAX_EPISODES))
            
            # Log everything clearly
            self.logger.info(f"📈 Adaptive episodes calculation:")
            self.logger.info(f"   • Actual action space: {actual_action_space}")
            self.logger.info(f"   • Coverage target: {COVERAGE_TARGET:.0%}")
            self.logger.info(f"   • Actions to sample: {actions_to_sample}")
            self.logger.info(f"   • Samples per action: {MIN_SAMPLES_PER_ACTION}")
            self.logger.info(f"   • Target episodes: {target_episodes}")
            self.logger.info(f"   • Final episodes: {final_episodes} (bounds: {MIN_EPISODES}-{MAX_EPISODES})")
            self.logger.info(f"   • Expected coverage: {min(final_episodes / MIN_SAMPLES_PER_ACTION / actual_action_space, 1.0):.1%}")
            self.logger.info(f"   🚀 Ultra-fast mode: 20% coverage + 4 samples/action = maximum speed!")
            
            return final_episodes
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate adaptive episodes: {e}, using default 400")
            return 400  # Ultra-fast default
    
    def clear_checkpoints(self):
        """Clear all existing checkpoints to force a fresh run."""
        checkpoint_dir = "./test_results/checkpoints"
        if os.path.exists(checkpoint_dir):
            import shutil
            try:
                shutil.rmtree(checkpoint_dir)
                self.logger.info(f"🗑️  Cleared all checkpoints from {checkpoint_dir}")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to clear checkpoints: {e}")
        else:
            self.logger.info("ℹ️  No checkpoints to clear")
    
    
    def run_all_grids_evaluation(self):
        """
        Run comprehensive evaluation on all grids with station allocation.
        
        Returns:
            bool: True if evaluation completed successfully, False otherwise
        """
        self.logger.info("🚀 Starting All-Grids Comprehensive Evaluation")
        self.logger.info("Testing all grids with ML-based station allocation and adaptive episodes")
        
        # Create logs directory
        os.makedirs("./logs", exist_ok=True)
        os.makedirs("./test_results", exist_ok=True)
        
        try:
            # Step 1: Load all grids with station allocation
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 1: LOADING ALL GRIDS WITH STATION ALLOCATION")
            self.logger.info("="*80)
            
            grids_data = self.data_loader.load_all_grids_with_station_allocation()
            if not grids_data:
                self.logger.error("❌ Failed to load grids with station allocation")
                return False
            
            self.logger.info("✅ All grids loaded and station allocation completed")
            
            # Step 2: Process each grid
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 2: RUNNING EVALUATION ON ALL GRIDS")
            self.logger.info("="*80)
            
            station_allocations = grids_data['station_allocations']
            city_grids_data = grids_data['city_grids_data']
            ved_data = grids_data['ved_data']
            
            all_results = {}
            grid_summary = []
            
            # Prepare grid info for parallel processing
            grid_info_list = [
                (grid_id, num_stations, ved_data, city_grids_data, grids_data)
                for grid_id, num_stations in station_allocations.items()
            ]
            
            # Process grids in parallel or sequential
            if self.num_workers > 1 and len(grid_info_list) > 1:
                self.logger.info(f"⚡ Processing {len(grid_info_list)} grids in PARALLEL with {self.num_workers} workers")
                
                with Pool(processes=self.num_workers) as pool:
                    grid_results_list = list(tqdm(
                        pool.imap(self._process_single_grid, grid_info_list),
                        total=len(grid_info_list),
                        desc="Processing Grids (Parallel)"
                    ))
            else:
                self.logger.info(f"📝 Processing {len(grid_info_list)} grids SEQUENTIALLY")
                grid_results_list = []
                for grid_info in tqdm(grid_info_list, desc="Processing Grids (Sequential)"):
                    grid_results_list.append(self._process_single_grid(grid_info))
            
            # Collect results
            for grid_result in grid_results_list:
                if grid_result is None:
                    continue
                    
                grid_id = grid_result['grid_id']
                
                if 'error' in grid_result:
                    self.logger.error(f"❌ Grid {grid_id} failed: {grid_result.get('error', 'Unknown error')}")
                    grid_summary.append(grid_result)
                else:
                    self.logger.info(f"✅ Grid {grid_id} completed: {grid_result['successful_methods']}/{grid_result['total_methods']} methods")
                    
                    # Store results
                    all_results[grid_id] = {
                        'num_stations': grid_result['num_stations'],
                        'results': grid_result['results'],
                        'comparison_metrics': grid_result['comparison_metrics'],
                        'research_metrics': grid_result['research_metrics'],
                        'test_data_summary': grid_result['test_data_summary']
                    }
                    
                    # Add to summary
                    grid_summary.append({
                        'grid_id': grid_id,
                        'num_stations': grid_result['num_stations'],
                        'successful_methods': grid_result['successful_methods'],
                        'total_methods': grid_result['total_methods'],
                        'trajectory_count': grid_result['trajectory_count']
                    })
            
            # Step 3: Generate comprehensive analysis
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 3: GENERATING COMPREHENSIVE ANALYSIS")
            self.logger.info("="*80)
            
            # Create summary DataFrame
            summary_df = pd.DataFrame(grid_summary)
            
            # Initialize best placements tracking
            best_placements_per_grid = {}
            
            # Generate analysis report
            self._generate_analysis_report(all_results, summary_df)
            
            # Step 4: Generate comprehensive research metrics and visualizations for all grids
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 4: GENERATING COMPREHENSIVE RESEARCH ANALYSIS")
            self.logger.info("="*80)
            
            # Calculate comprehensive research metrics for all grids combined
            all_grids_research_metrics = {}
            best_placements_per_grid = {}  # Initialize the dictionary
            
            for grid_id, grid_data in all_results.items():
                # Handle both full results and simplified checkpoints
                if 'reward_summary' in grid_data:
                    # This is from a simplified checkpoint
                    reward_summary = grid_data['reward_summary']
                    best_placements_per_grid[grid_id] = {}
                    for method_name, reward_info in reward_summary.items():
                        if 'error' not in reward_info:
                            best_placements_per_grid[grid_id][method_name] = {
                                'method': method_name,
                                'reward': reward_info.get('reward', 0.0),
                                'best_reward': reward_info.get('best_reward', reward_info.get('reward', 0.0)),
                                'simulation_reward': reward_info.get('simulation_reward', reward_info.get('reward', 0.0)),
                                'simulation_success': reward_info.get('simulation_success', True),
                                'total_episodes': reward_info.get('total_episodes', 0),
                                'convergence_achieved': reward_info.get('convergence_achieved', False),
                                'episodes_to_convergence': reward_info.get('episodes_to_convergence', 0),
                                'algorithm': reward_info.get('algorithm', 'unknown'),
                                'method_type': reward_info.get('method_type', 'unknown'),
                                'num_placements': reward_info.get('num_placements', 0)
                            }
                        else:
                            best_placements_per_grid[grid_id][method_name] = {
                                'method': method_name,
                                'error': reward_info['error']
                            }
                else:
                    # This is from full results
                    grid_results = grid_data['results']
                    best_placements_per_grid[grid_id] = {}
                    
                    for method_name, result in grid_results.items():
                        if 'error' not in result:
                            # Get the best reward (prefer best_reward, fallback to reward)
                            best_reward = result.get('best_reward', result.get('reward', 0.0))
                            reward = result.get('reward', best_reward)
                            
                            best_placements_per_grid[grid_id][method_name] = {
                                'method': method_name,
                                'reward': reward,
                                'best_reward': best_reward,
                                'placements': result.get('placements', []),
                                'best_placement': result.get('best_placement', []),
                                'simulation_reward': result.get('simulation_reward', reward),
                                'simulation_success': result.get('simulation_success', True),
                                'convergence_rate': result.get('convergence_rate', 0.0),
                                'episodes_to_convergence': result.get('episodes_to_convergence', 0),
                                'total_episodes': result.get('total_episodes', 0),
                                'convergence_achieved': result.get('convergence_achieved', False),
                                'algorithm': result.get('algorithm', 'unknown'),
                                'method_type': result.get('method', 'unknown'),
                                'num_placements': len(result.get('placements', []))
                            }
                
                # Calculate comprehensive metrics for all methods (access from best_placements_per_grid)
                grid_methods = best_placements_per_grid[grid_id]
                for method_name, method_info in grid_methods.items():
                    if 'error' not in method_info:
                        # Create a unique key for each method across all grids
                        unique_key = f"{grid_id}_{method_name}"
                        all_grids_research_metrics[unique_key] = self._calculate_comprehensive_research_metrics_for_grid(method_name, method_info, grid_id)
            
            # Calculate aggregated metrics for best placements
            aggregated_best_placements = self._calculate_aggregated_best_placements(best_placements_per_grid)
            
            # Save results including aggregated best placements
            self._save_comprehensive_results(all_results, summary_df, grids_data, aggregated_best_placements)
            
            # Calculate aggregated metrics with deviation analysis
            self.logger.info("📊 Calculating aggregated metrics with deviation analysis...")
            aggregated_metrics_with_deviation = self._calculate_aggregated_metrics_with_deviation(all_results)
            
            # Generate comprehensive visualizations for all grids
            self.logger.info("🎨 Generating comprehensive visualizations for all grids...")
            self._generate_comprehensive_visualizations_all_grids(all_results, all_grids_research_metrics, aggregated_metrics_with_deviation)
            
            # Create individual grid visualizations
            self.logger.info("🎨 Creating individual grid visualizations...")
            self._create_visualizations(station_allocations, city_grids_data, all_results)
            
            # Step 4: Final summary
            self.logger.info("\n" + "="*80)
            self.logger.info("EVALUATION SUMMARY")
            self.logger.info("="*80)
            
            successful_grids = len([g for g in grid_summary if g['successful_methods'] > 0])
            total_grids = len(grid_summary)
            total_stations_allocated = sum(g['num_stations'] for g in grid_summary)
            
            self.logger.info(f"✅ Successfully processed: {successful_grids}/{total_grids} grids")
            self.logger.info(f"🎯 Total stations allocated: {total_stations_allocated}")
            self.logger.info(f"📊 Average stations per grid: {total_stations_allocated/total_grids:.2f}")
            
            if successful_grids == total_grids:
                self.logger.info("🎉 ALL GRIDS PROCESSED SUCCESSFULLY!")
                self.logger.info("Check test_results/ directory for comprehensive results and metrics.")
                return True
            else:
                failed_grids = total_grids - successful_grids
                self.logger.warning(f"⚠️ {failed_grids} grids failed - check logs for details")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Evaluation failed with error: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _process_single_grid(self, grid_info):
        """
        Process a single grid (designed for parallel execution).
        
        Args:
            grid_info: Tuple of (grid_id, num_stations, ved_data, city_grids_data, grid_data)
            
        Returns:
            Dict with grid results or None if failed
        """
        grid_id, num_stations, ved_data, city_grids_data, grid_data = grid_info
        
        # Check for existing checkpoint (if resume is enabled)
        if self.resume:
            checkpoint_file = f"./test_results/checkpoints/grid_{grid_id}_checkpoint.json"
            if os.path.exists(checkpoint_file):
                try:
                    with open(checkpoint_file, 'r') as f:
                        checkpoint_data = json.load(f)

                    # Validate checkpoint configuration
                    cfg = checkpoint_data.get('config', {})
                    if (checkpoint_data.get('num_stations') == num_stations and
                        cfg.get('random_seed') == self.random_seed and
                        cfg.get('adaptive_mode') == self.adaptive_mode and
                        cfg.get('confidence_threshold') == self.confidence_threshold and
                        cfg.get('override_episodes') == self.override_episodes):
                        
                        self.logger.info(f"✅ Resuming from a valid checkpoint: {grid_id}")
                        
                        # Convert checkpoint format to expected format
                        if 'results' in checkpoint_data:
                            # New comprehensive checkpoint format - use full results
                            self.logger.info(f"✅ Using comprehensive checkpoint with full results: {grid_id}")
                            # Ensure all required fields are present
                            checkpoint_data['comparison_metrics'] = checkpoint_data.get('comparison_metrics', {})
                            checkpoint_data['research_metrics'] = checkpoint_data.get('research_metrics', {})
                            checkpoint_data['trajectory_count'] = checkpoint_data.get('trajectory_count', 
                                checkpoint_data.get('test_data_summary', {}).get('trajectory_count', 0))
                        elif 'reward_summary' in checkpoint_data and 'results' not in checkpoint_data:
                            # Legacy simplified checkpoint format - convert reward_summary to results
                            self.logger.info(f"✅ Converting legacy checkpoint format: {grid_id}")
                            checkpoint_data['results'] = checkpoint_data['reward_summary']
                            
                            # Add missing fields with default values
                            checkpoint_data['comparison_metrics'] = checkpoint_data.get('comparison_metrics', {})
                            checkpoint_data['research_metrics'] = checkpoint_data.get('research_metrics', {})
                            
                            # Add trajectory_count from test_data_summary if available
                            if 'trajectory_count' not in checkpoint_data and 'test_data_summary' in checkpoint_data:
                                checkpoint_data['trajectory_count'] = checkpoint_data['test_data_summary'].get('trajectory_count', 0)
                        
                        return checkpoint_data
                    else:
                        self.logger.warning(f"⚠️ Checkpoint for {grid_id} has a configuration mismatch. Rerunning.")
                
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to load or validate checkpoint for {grid_id}: {e}, rerunning...")
        
        try:
            # Prepare test data for this specific grid
            grid_test_data = self._prepare_grid_test_data(grid_id, ved_data, city_grids_data)
            
            if not grid_test_data:
                return {
                    'grid_id': grid_id,
                    'num_stations': num_stations,
                    'successful_methods': 0,
                    'total_methods': 0,
                    'trajectory_count': 0,
                    'error': 'No test data'
                }
            
            # Create a simple logger for this worker (avoids multiprocessing logging issues)
            import logging
            worker_logger = logging.getLogger(f"grid_{grid_id}")
            
            # Initialize execution engine for this grid
            # IMPORTANT: Use deterministic hash for grid_id (Python's hash() is non-deterministic!)
            grid_hash = sum(ord(c) for c in str(grid_id)) % 1000
            execution_engine = ExecutionEngine(
                test_data=grid_test_data,
                random_seed=self.random_seed + grid_hash,  # Deterministic unique seed per grid
                logger=worker_logger,
                episodes=10,
                adaptive_mode=self.adaptive_mode,
                confidence_threshold=self.confidence_threshold,
                grid_id=grid_id  # Pass grid_id for parallel-safe SUMO output files
            )
            
            # Initialize metrics analyzer for this grid
            metrics_analyzer = MetricsAnalyzer(
                test_data=grid_test_data,
                random_seed=self.random_seed,
                logger=worker_logger
            )
            
            # Calculate episodes
            if self.override_episodes is not None:
                max_episodes = self.override_episodes
            else:
                max_episodes = self._calculate_adaptive_episodes(num_stations, grid_data)
            
            # Run hybrid methods
            hybrid_results = execution_engine.run_hybrid_methods(
                num_chargers=num_stations,
                max_episodes=max_episodes
            )
            
            # Log actual action space used (for statistical validation)
            for method_name, result in hybrid_results.items():
                if 'actual_action_space' in result and 'error' not in result:
                    actual_space = result['actual_action_space']
                    coverage = min(max_episodes / 10 / actual_space, 1.0) if actual_space > 0 else 0
                    self.logger.info(f"✅ {method_name}: Actual action space = {actual_space}, "
                                   f"Episodes = {max_episodes}, Coverage = {coverage:.1%}")
            
            # Run baseline methods
            baseline_results = execution_engine.run_baseline_methods(num_chargers=num_stations)
            
            # Combine results
            grid_results = {**hybrid_results, **baseline_results}
            
            # Calculate metrics
            comparison_metrics = metrics_analyzer.calculate_comparison_metrics(grid_results)
            
            # Calculate research metrics
            research_metrics = {}
            for method_name, result in grid_results.items():
                if 'error' not in result:
                    research_metrics[method_name] = metrics_analyzer.calculate_comprehensive_research_metrics(method_name, result)
            
            # Save results (worker saves independently)
            metrics_analyzer.save_basic_results_only(grid_results, comparison_metrics, research_metrics)
            
            # Prepare simplified result for checkpointing (essential data only)
            simplified_results = {}
            for method_name, method_result in grid_results.items():
                if 'error' not in method_result:
                    # Get the best reward (prefer best_reward, fallback to reward)
                    best_reward = method_result.get('best_reward', method_result.get('reward', 0.0))
                    reward = method_result.get('reward', best_reward)
                    
                    simplified_results[method_name] = {
                        'reward': reward,
                        'best_reward': best_reward,
                        'simulation_reward': method_result.get('simulation_reward', reward),
                        'simulation_success': method_result.get('simulation_success', True),
                        'total_episodes': method_result.get('total_episodes', 0),
                        'convergence_achieved': method_result.get('convergence_achieved', False),
                        'episodes_to_convergence': method_result.get('episodes_to_convergence', 0),
                        'num_placements': len(method_result.get('placements', [])),
                        'algorithm': method_result.get('algorithm', 'unknown'),
                        'method_type': method_result.get('method', 'unknown')
                    }
                else:
                    simplified_results[method_name] = {'error': method_result['error']}
            
            # Prepare result for checkpointing and final output
            result = {
                'grid_id': grid_id,
                'num_stations': num_stations,
                'config': {
                    'random_seed': self.random_seed,
                    'adaptive_mode': self.adaptive_mode,
                    'confidence_threshold': self.confidence_threshold,
                    'override_episodes': self.override_episodes
                },
                # Keep full results for main analysis, but use simplified for checkpoint if needed
                'results': grid_results,
                'simplified_results': simplified_results,  # New optimized section
                'comparison_metrics': comparison_metrics,
                'research_metrics': research_metrics,
                'test_data_summary': {
                    'trajectory_count': len(grid_test_data['trajectory_df']),
                    'grid_bounds': grid_test_data['grid_bounds']
                },
                'successful_methods': sum(1 for r in grid_results.values() if 'error' not in r),
                'total_methods': len(grid_results),
                'trajectory_count': len(grid_test_data['trajectory_df'])
            }
            
            # Save simplified checkpoint for resumability
            os.makedirs("./test_results/checkpoints", exist_ok=True)
            checkpoint_file = f"./test_results/checkpoints/grid_{grid_id}_checkpoint.json"
            try:
                # Create comprehensive checkpoint with all necessary data for visualizations
                comprehensive_checkpoint = {
                    'grid_id': grid_id,
                    'num_stations': num_stations,
                    'config': {
                        'random_seed': self.random_seed,
                        'adaptive_mode': self.adaptive_mode,
                        'confidence_threshold': self.confidence_threshold,
                        'override_episodes': self.override_episodes
                    },
                    # Include both simplified and full results for compatibility
                    'reward_summary': simplified_results,
                    'results': grid_results,  # Full results with placement data
                    'comparison_metrics': comparison_metrics,
                    'research_metrics': research_metrics,
                    'test_data_summary': {
                        'trajectory_count': len(grid_test_data['trajectory_df']),
                        'grid_bounds': grid_test_data['grid_bounds']
                    },
                    'successful_methods': sum(1 for r in grid_results.values() if 'error' not in r),
                    'total_methods': len(grid_results),
                    'trajectory_count': len(grid_test_data['trajectory_df']),
                    'timestamp': datetime.now().isoformat()
                }
                
                with open(checkpoint_file, 'w') as f:
                    json.dump(comprehensive_checkpoint, f, indent=2, default=str)
                self.logger.info(f"💾 Saved comprehensive checkpoint: {grid_id}")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to save checkpoint for {grid_id}: {e}")
            
            return result
            
        except Exception as e:
            # Enhanced error handling for parallel execution robustness
            error_msg = f"Grid {grid_id} processing failed: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            
            # Log traceback for debugging but don't let it crash the entire job
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            
            return {
                'grid_id': grid_id,
                'num_stations': num_stations,
                'successful_methods': 0,
                'total_methods': 6,  # Expected total methods
                'trajectory_count': 0,
                'error': error_msg,
                'processing_failed': True
            }
    
    def _prepare_grid_test_data(self, grid_id, ved_data, city_grids_data):
        """Prepare test data for a specific grid."""
        try:
            # Filter VED data for this grid
            grid_ved_data = ved_data[ved_data['grid_id'] == grid_id].copy()
            
            if len(grid_ved_data) == 0:
                self.logger.warning(f"No VED data found for grid {grid_id}")
                return None
            
            # Create trajectory DataFrame
            trajectory_df = pd.DataFrame()
            for vehicle_id, vehicle_data in grid_ved_data.groupby('VehId'):
                if len(vehicle_data) >= 2:  # Need at least 2 points for trajectory
                    traj_copy = vehicle_data[['lat', 'lon', 'timestamp']].copy()
                    traj_copy['VehId'] = vehicle_id
                    trajectory_df = pd.concat([trajectory_df, traj_copy], ignore_index=True)
            
            if len(trajectory_df) == 0:
                self.logger.warning(f"No valid trajectories found for grid {grid_id}")
                return None
            
            # Find grid bounds
            grid_cell = next((cell for cell in city_grids_data if cell['grid_id'] == grid_id), None)
            if not grid_cell:
                self.logger.error(f"Grid cell not found for grid {grid_id}")
                return None
            
            grid_bounds = {
                'min_lat': grid_cell['min_lat'],
                'max_lat': grid_cell['max_lat'],
                'min_lon': grid_cell['min_lon'],
                'max_lon': grid_cell['max_lon']
            }
            
            # Network file path
            network_file = os.path.abspath("generated_files/city_network/ann_arbor.osm.net.xml")
            
            return {
                'test_grid_id': grid_id,
                'trajectory_df': trajectory_df,
                'grid_bounds': grid_bounds,
                'network_file': network_file
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing test data for grid {grid_id}: {e}")
            return None
    
    def _save_comprehensive_results(self, all_results, summary_df, grids_data, aggregated_best_placements=None, aggregated_metrics_with_deviation=None):
        """Save comprehensive results to files."""
        self.logger.info("💾 Saving comprehensive results...")
        
        # Save summary DataFrame
        summary_path = "./test_results/all_grids_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        self.logger.info(f"✅ Summary saved to: {summary_path}")
        
        # Save station allocation
        allocation_path = "./test_results/station_allocation.csv"
        grids_data['allocation_df'].to_csv(allocation_path, index=False)
        self.logger.info(f"✅ Station allocation saved to: {allocation_path}")
        
        # Save detailed results (including placement data)
        results_summary = {}
        for grid_id, grid_data in all_results.items():
            results_summary[grid_id] = {
                'num_stations': grid_data['num_stations'],
                'successful_methods': sum(1 for r in grid_data['results'].values() if 'error' not in r),
                'total_methods': len(grid_data['results']),
                'trajectory_count': grid_data['test_data_summary']['trajectory_count'],
                'comparison_metrics': grid_data['comparison_metrics'],
                'results': grid_data['results']  # Include full results with placement data
            }
        
        import json
        results_path = "./test_results/all_grids_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=4, default=str)
        self.logger.info(f"✅ Detailed results saved to: {results_path}")
        
        # Save aggregated best placements
        if aggregated_best_placements:
            aggregated_path = "./test_results/aggregated_best_placements.json"
            with open(aggregated_path, 'w') as f:
                json.dump(aggregated_best_placements, f, indent=4, default=str)
            self.logger.info(f"✅ Aggregated best placements saved to: {aggregated_path}")
        
        # Save aggregated metrics with deviation analysis
        if aggregated_metrics_with_deviation:
            aggregated_metrics_path = "./test_results/aggregated_metrics_with_deviation.json"
            with open(aggregated_metrics_path, 'w') as f:
                json.dump(aggregated_metrics_with_deviation, f, indent=4, default=str)
            self.logger.info(f"✅ Aggregated metrics with deviation saved to: {aggregated_metrics_path}")
            
            # Also save as CSV for easy analysis
            if 'method_performance' in aggregated_metrics_with_deviation:
                method_performance_data = []
                for method_name, metrics in aggregated_metrics_with_deviation['method_performance'].items():
                    row = {'method': method_name}
                    row.update(metrics)
                    method_performance_data.append(row)
                
                if method_performance_data:
                    import pandas as pd
                    method_performance_df = pd.DataFrame(method_performance_data)
                    method_performance_path = "./test_results/method_performance_with_deviation.csv"
                    method_performance_df.to_csv(method_performance_path, index=False)
                    self.logger.info(f"✅ Method performance with deviation saved to: {method_performance_path}")
    
    def _generate_analysis_report(self, all_results, summary_df):
        """Generate comprehensive analysis report."""
        self.logger.info("📊 Generating analysis report...")
        
        # Calculate overall statistics
        total_grids = len(summary_df)
        successful_grids = len(summary_df[summary_df['successful_methods'] > 0])
        total_stations = summary_df['num_stations'].sum()
        avg_stations = summary_df['num_stations'].mean()
        
        # Method success rates
        method_stats = {}
        for grid_id, grid_data in all_results.items():
            for method_name, result in grid_data['results'].items():
                if method_name not in method_stats:
                    method_stats[method_name] = {'success': 0, 'total': 0}
                method_stats[method_name]['total'] += 1
                if 'error' not in result:
                    method_stats[method_name]['success'] += 1
        
        # Generate report
        report_lines = [
            "# All-Grids EV Charging Station Placement Evaluation Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Overall Statistics",
            f"- Total grids processed: {total_grids}",
            f"- Successful grids: {successful_grids}",
            f"- Success rate: {successful_grids/total_grids*100:.1f}%",
            f"- Total stations allocated: {total_stations}",
            f"- Average stations per grid: {avg_stations:.2f}",
            "",
            "## Method Performance",
            ""
        ]
        
        for method_name, stats in method_stats.items():
            success_rate = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
            report_lines.append(f"- {method_name}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
        
        report_lines.extend([
            "",
            "## Grid-by-Grid Results",
            ""
        ])
        
        for _, row in summary_df.iterrows():
            status = "✅" if row['successful_methods'] > 0 else "❌"
            report_lines.append(f"- {row['grid_id']}: {row['num_stations']} stations, {row['successful_methods']}/{row['total_methods']} methods {status}")
        
        # Save report
        report_path = "./test_results/all_grids_evaluation_report.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"✅ Analysis report saved to: {report_path}")
    
    def _create_visualizations(self, station_allocations: Dict[str, int], city_grids_data: List[Dict], all_results: Dict):
        """Create comprehensive visualizations of station placement."""
        self.logger.info("🎨 Creating comprehensive visualizations...")
        
        try:
            # Initialize visualizer
            visualizer = ChargingStationVisualizer(logger=self.logger)
            
            # 1. Create comprehensive station placement visualization
            self.logger.info("📊 Creating comprehensive station placement map...")
            comprehensive_path = visualizer.visualize_all_stations_on_map(
                station_allocations=station_allocations,
                city_grids_data=city_grids_data,
                results_data=all_results,
                title=f"EV Charging Station Placement - {self.total_station_budget} Stations Total"
            )
            
            # 1.5. Create exact station location visualization
            self.logger.info("📍 Creating exact station location visualization...")
            exact_path = visualizer.visualize_exact_station_locations(
                all_results=all_results,
                city_grids_data=city_grids_data,
                title=f"Exact EV Charging Station Locations - {self.total_station_budget} Stations Total"
            )
            
            # 2. Create interactive map (if folium is available)
            self.logger.info("🌐 Creating interactive map...")
            interactive_path = visualizer.create_interactive_map(
                station_allocations=station_allocations,
                city_grids_data=city_grids_data,
                results_data=all_results
            )
            
            # 3. Create individual grid visualizations for top grids
            self.logger.info("📈 Creating individual grid visualizations...")
            top_grids = sorted(station_allocations.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for grid_id, station_count in top_grids:
                if grid_id in all_results:
                    grid_viz_path = visualizer.visualize_single_grid_results(
                        grid_id=grid_id,
                        station_count=station_count,
                        results_data=all_results[grid_id]
                    )
                    self.logger.info(f"   ✅ Grid {grid_id} visualization: {grid_viz_path}")
            
            # 4. Save visualization summary
            viz_summary = {
                'comprehensive_map': comprehensive_path,
                'exact_locations_map': exact_path,
                'interactive_map': interactive_path,
                'top_grids_visualized': len(top_grids),
                'total_stations': sum(station_allocations.values()),
                'total_grids': len(station_allocations),
                'timestamp': datetime.now().isoformat()
            }
            
            viz_summary_path = "./test_results/visualization_summary.json"
            with open(viz_summary_path, 'w') as f:
                json.dump(viz_summary, f, indent=4, default=str)
            
            self.logger.info(f"✅ Visualization summary saved to: {viz_summary_path}")
            self.logger.info(f"✅ All visualizations completed successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Visualization creation failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Continue execution even if visualization fails
    
    def _calculate_aggregated_best_placements(self, best_placements_per_grid: Dict) -> Dict:
        """Calculate aggregated metrics for all methods across all grids."""
        try:
            if not best_placements_per_grid:
                return {}
            
            aggregated = {
                'total_grids': len(best_placements_per_grid),
                'method_distribution': {},
                'reward_statistics': {},
                'placement_statistics': {},
                'quality_metrics': {},
                'method_performance': {}
            }
            
            # Collect all data by method
            method_data = {}
            all_rewards = []
            all_placements = []
            
            for grid_id, grid_methods in best_placements_per_grid.items():
                for method_name, method_result in grid_methods.items():
                    if method_name not in method_data:
                        method_data[method_name] = {
                            'rewards': [],
                            'placements': [],
                            'simulation_rewards': [],
                            'convergence_rates': [],
                            'episodes_to_convergence': []
                        }
                    
                    method_data[method_name]['rewards'].append(method_result['reward'])
                    method_data[method_name]['simulation_rewards'].append(method_result['simulation_reward'])
                    method_data[method_name]['convergence_rates'].append(method_result.get('convergence_rate', 0.0))
                    method_data[method_name]['episodes_to_convergence'].append(method_result.get('episodes_to_convergence', 0))
                    
                    placements = method_result.get('placements', []) or method_result.get('best_placement', [])
                    method_data[method_name]['placements'].extend(placements)
                    all_placements.extend(placements)
                    all_rewards.append(method_result['reward'])
            
            # Calculate method distribution and performance
            total_grids = len(best_placements_per_grid)
            for method_name, data in method_data.items():
                count = len(data['rewards'])
                aggregated['method_distribution'][method_name] = {
                    'count': count,
                    'percentage': (count / total_grids) * 100,
                    'avg_reward': np.mean(data['rewards']),
                    'avg_simulation_reward': np.mean(data['simulation_rewards']),
                    'avg_convergence_rate': np.mean(data['convergence_rates']),
                    'avg_episodes_to_convergence': np.mean(data['episodes_to_convergence'])
                }
                
                # Calculate method performance with std deviation
                aggregated['method_performance'][method_name] = {
                    'reward_mean': np.mean(data['rewards']),
                    'reward_std': np.std(data['rewards']),
                    'simulation_reward_mean': np.mean(data['simulation_rewards']),
                    'simulation_reward_std': np.std(data['simulation_rewards']),
                    'convergence_rate_mean': np.mean(data['convergence_rates']),
                    'convergence_rate_std': np.std(data['convergence_rates']),
                    'episodes_to_convergence_mean': np.mean(data['episodes_to_convergence']),
                    'episodes_to_convergence_std': np.std(data['episodes_to_convergence']),
                    'success_rate': count / total_grids,
                    'total_placements': len(data['placements'])
                }
            
            # Calculate overall reward statistics
            if all_rewards:
                aggregated['reward_statistics'] = {
                    'mean': np.mean(all_rewards),
                    'std': np.std(all_rewards),
                    'min': np.min(all_rewards),
                    'max': np.max(all_rewards),
                    'median': np.median(all_rewards)
                }
            
            # Calculate placement statistics
            if all_placements:
                aggregated['placement_statistics'] = {
                    'total_placements': len(all_placements),
                    'unique_edges': len(set(p.get('edge_id', p.get('edge', '')) for p in all_placements if isinstance(p, dict))),
                    'avg_placements_per_grid': len(all_placements) / total_grids
                }
                
                # Calculate spatial diversity
                if len(all_placements) > 1:
                    lats = [p.get('lat', 0) for p in all_placements if isinstance(p, dict) and 'lat' in p]
                    lons = [p.get('lon', 0) for p in all_placements if isinstance(p, dict) and 'lon' in p]
                    
                    if lats and lons:
                        lat_range = max(lats) - min(lats) if len(set(lats)) > 1 else 0
                        lon_range = max(lons) - min(lons) if len(set(lons)) > 1 else 0
                        aggregated['placement_statistics']['spatial_diversity'] = lat_range + lon_range
            
            # Calculate quality metrics
            aggregated['quality_metrics'] = {
                'success_rate': len(best_placements_per_grid) / total_grids if total_grids > 0 else 0,
                'reward_consistency': 1.0 - (np.std(all_rewards) / np.mean(all_rewards)) if all_rewards and np.mean(all_rewards) > 0 else 0,
                'method_diversity': len(method_data) / total_grids if total_grids > 0 else 0
            }
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Error calculating aggregated best placements: {e}")
            return {'error': str(e)}
    
    def _calculate_comprehensive_research_metrics_for_grid(self, method_name: str, result: Dict, grid_id: str) -> Dict:
        """Calculate comprehensive research metrics for a single grid result."""
        try:
            # Use fallback bounds for Ann Arbor
            default_grid_bounds = {
                'min_lat': 42.2170, 'max_lat': 42.3479,
                'min_lon': -83.8084, 'max_lon': -83.6747
            }
            
            # Initialize a temporary metrics analyzer for this calculation
            temp_metrics_analyzer = MetricsAnalyzer(
                test_data={'grid_bounds': default_grid_bounds},
                random_seed=self.random_seed,
                logger=self.logger
            )
            
            # Calculate comprehensive research metrics
            research_metrics = temp_metrics_analyzer.calculate_comprehensive_research_metrics(method_name, result)
            
            # Add grid-specific information
            research_metrics['grid_id'] = grid_id
            research_metrics['grid_specific'] = True
            
            # Ensure all required metrics are present
            if 'exploration_metrics' not in research_metrics:
                research_metrics['exploration_metrics'] = {
                    'exploration_efficiency': 0,
                    'action_diversity': 0,
                    'exploration_vs_exploitation': 0,
                    'placement_diversity': 0,
                    'spatial_diversity': 0,
                    'edge_diversity': 0,
                    'overall_diversity': 0,
                    'learning_curve_quality': 0,
                    'convergence_rate': 0,
                    'sample_efficiency': 0
                }
            
            # Ensure placement quality metrics are present
            if 'placement_metrics' not in research_metrics:
                research_metrics['placement_metrics'] = {
                    'num_placements': 0,
                    'edge_coverage': 0.0,
                    'unique_edges': 0,
                    'spatial_diversity': 0.0,
                    'grid_compliance': 0.0,
                    'network_integration': 0.0
                }
            
            # Ensure simulation metrics are present
            if 'simulation_metrics' not in research_metrics:
                research_metrics['simulation_metrics'] = {
                    'simulation_reward': 0.0,
                    'simulation_success': False,
                    'simulation_error': None,
                    'charging_efficiency': 0.0,
                    'network_utilization': 0.0,
                    'battery_management': 0.0,
                    'traffic_impact': 0.0
                }
            
            return research_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating research metrics for {method_name} in grid {grid_id}: {e}")
            return {
                'method_name': method_name,
                'grid_id': grid_id,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'exploration_metrics': {},
                'placement_metrics': {},
                'simulation_metrics': {}
            }
    
    def _calculate_aggregated_metrics_with_deviation(self, all_results: Dict) -> Dict:
        """Calculate aggregated metrics across all grids with deviation analysis."""
        try:
            # Initialize a temporary metrics analyzer for this calculation
            temp_metrics_analyzer = MetricsAnalyzer(
                test_data={'grid_bounds': {}},  # Minimal test data
                random_seed=self.random_seed,
                logger=self.logger
            )
            
            # Calculate aggregated metrics with deviation
            aggregated_metrics = temp_metrics_analyzer.calculate_aggregated_metrics_with_deviation(all_results)
            
            return aggregated_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating aggregated metrics with deviation: {e}")
            return {'error': str(e)}
    
    def _generate_comprehensive_visualizations_all_grids(self, all_results: Dict, all_grids_research_metrics: Dict, aggregated_metrics_with_deviation: Dict = None):
        """Generate comprehensive visualizations for all grids combined."""
        try:
            # Initialize a temporary metrics analyzer for visualization generation
            # Use the first grid's bounds as default (all grids should have similar bounds)
            default_grid_bounds = None
            for grid_id, grid_data in all_results.items():
                if 'grid_bounds' in grid_data:
                    default_grid_bounds = grid_data['grid_bounds']
                    break
            
            if default_grid_bounds is None:
                # Fallback bounds for Ann Arbor
                default_grid_bounds = {
                    'min_lat': 42.2170, 'max_lat': 42.3479,
                    'min_lon': -83.8084, 'max_lon': -83.6747
                }
            
            temp_metrics_analyzer = MetricsAnalyzer(
                test_data={'grid_bounds': default_grid_bounds},
                random_seed=self.random_seed,
                logger=self.logger
            )
            
            # Create comparison metrics for all grids
            all_grids_comparison_metrics = {}
            for grid_id, grid_data in all_results.items():
                grid_results = grid_data['results']
                grid_comparison_metrics = temp_metrics_analyzer.calculate_comparison_metrics(grid_results)
                all_grids_comparison_metrics[grid_id] = grid_comparison_metrics
            
            # Generate comprehensive visualizations
            temp_metrics_analyzer._generate_comprehensive_visualizations(
                all_results, 
                all_grids_comparison_metrics, 
                all_grids_research_metrics, 
                datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            
            self.logger.info("✅ Comprehensive visualizations generated for all grids")
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive visualization generation failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Continue execution even if visualization fails


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="All-Grids EV Charging Station Placement Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (100 stations, adaptive episodes, resume enabled)
  python run_all_grids_evaluation.py
  
  # Run from scratch (ignore checkpoints)
  python run_all_grids_evaluation.py --no-resume
  
  # Clear all checkpoints and run fresh
  python run_all_grids_evaluation.py --clear-checkpoints
  
  # Run with custom station budget and fixed episodes
  python run_all_grids_evaluation.py --station-budget 50 --override-episodes 20
  
  # Run with higher confidence threshold
  python run_all_grids_evaluation.py --confidence-threshold 0.99
  
  # Run with custom seed and disable adaptive mode
  python run_all_grids_evaluation.py --seed 123 --no-adaptive-mode
  
  # Quick test with fewer stations and episodes
  python run_all_grids_evaluation.py --station-budget 20 --override-episodes 5
        """
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--station-budget', '-b',
        type=int,
        default=150,
        help='Total station budget to allocate across all grids (default: 150)'
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
        '--confidence-threshold', '-c',
        type=float,
        default=0.95,
        help='Confidence threshold for convergence (0.0-1.0, default: 0.95)'
    )
    
    parser.add_argument(
        '--override-episodes', '-e',
        type=int,
        default=None,
        help='Override episodes for testing (disables adaptive calculation)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help='Number of parallel workers (None=auto-detect, 1=sequential). Default: auto'
    )
    
    parser.add_argument(
        '--clear-checkpoints',
        action='store_true',
        help='Clear all existing checkpoints before starting (forces full re-run)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='Resume from checkpoints if available (default: True)'
    )
    
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Ignore existing checkpoints and re-run everything'
    )
    
    return parser.parse_args()


def main():
    """Main function to run the all-grids evaluation."""
    args = parse_arguments()
    
    # Handle adaptive mode - disable if override_episodes is provided
    adaptive_mode = args.adaptive_mode and not args.no_adaptive_mode and args.override_episodes is None
    
    # Handle resume mode
    resume = args.resume and not args.no_resume
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print(f"🎲 Using random seed: {args.seed}")
    print(f"🎯 Total station budget: {args.station_budget}")
    print(f"📊 Adaptive mode: {adaptive_mode}")
    print(f"🎯 Confidence threshold: {args.confidence_threshold}")
    if args.override_episodes:
        print(f"🔧 Override episodes: {args.override_episodes}")
    else:
        print("🔧 Using adaptive episode calculation")
    
    if resume:
        print("💾 Resume mode: Enabled (will use checkpoints if available)")
    else:
        print("🔄 Resume mode: Disabled (will re-run everything)")
    
    # Create and run evaluation
    evaluation = AllGridsComprehensiveEvaluation(
        random_seed=args.seed,
        total_station_budget=args.station_budget,
        adaptive_mode=adaptive_mode,
        confidence_threshold=args.confidence_threshold,
        override_episodes=args.override_episodes,
        num_workers=args.workers,
        resume=resume
    )
    
    # Clear checkpoints if requested
    if args.clear_checkpoints:
        print("🗑️  Clearing all existing checkpoints...")
        evaluation.clear_checkpoints()
    
    
    success = evaluation.run_all_grids_evaluation()
    
    if success:
        print("\n✅ ALL GRIDS EVALUATED SUCCESSFULLY!")
        print("Check test_results/ directory for comprehensive results and metrics.")
    else:
        print("\n❌ SOME GRIDS FAILED!")
        print("Check logs for detailed error information.")
        sys.exit(1)


if __name__ == "__main__":
    main()

