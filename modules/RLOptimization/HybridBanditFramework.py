"""
Hybrid Bandit Framework: Combining Heuristics with Multi-Armed Bandits

This module implements a hybrid approach where heuristics generate candidate
placements, and bandit algorithms explore and exploit these candidates.
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict
import random
import sys
from datetime import datetime

from .BanditOptimizationFramework import BanditOptimizationFramework
from .HeuristicBaselines import HeuristicBaselines
from .metrics import calculate_comprehensive_metrics, compare_methods_performance, save_comparison_results

class HybridBanditFramework:
    """
    Hybrid framework combining heuristics with bandit algorithms.
    
    The approach:
    1. Use heuristics (K-Means, Centrality) to generate candidate placements
    2. Use bandit algorithms (UCB, Epsilon-Greedy, Thompson Sampling) to explore/exploit
    3. Combine the best of both worlds: smart initialization + adaptive learning
    """
    
    def _setup_logging(self):
        """Setup simple logging for the hybrid framework."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def __init__(self, 
                 network_file: str,
                 simulation_config: Any,
                 ev_config: Any,
                 base_seed: int = 42,
                 grid_id: str = None):
        """
        Initialize hybrid framework.
        
        Args:
            network_file: Path to SUMO network file
            simulation_config: Simulation configuration
            ev_config: EV configuration
            base_seed: Random seed for reproducibility
            grid_id: Grid ID for unique SUMO output files (parallel execution safety)
        """
        # Setup comprehensive logging first
        self._setup_logging()
        
        self.network_file = network_file
        self.simulation_config = simulation_config
        self.ev_config = ev_config
        self.base_seed = base_seed
        self.grid_id = grid_id or "default"  # Store grid_id for parallel execution
        
        # Set random seeds for reproducibility
        np.random.seed(base_seed)
        random.seed(base_seed)
        
        self.logger.info(f"Initializing Hybrid Bandit Framework")
        self.logger.info(f"  Network file: {network_file}")
        self.logger.info(f"  Base seed: {base_seed}")
        self.logger.info(f"  Simulation config: {type(simulation_config).__name__}")
        self.logger.info(f"  EV config: {type(ev_config).__name__}")
        
        # Initialize components
        self.heuristic_baselines = HeuristicBaselines(network_file)
        
        # Initialize bandit framework for evaluation (optional)
        self.bandit_framework = None
        try:
            # Only initialize if we have a proper network file
            if network_file.endswith('.net.xml'):
                # Note: num_stations will be set dynamically in run_hybrid_optimization
                self.bandit_framework = BanditOptimizationFramework(
                    osm_file_path=network_file, 
                    num_stations=3,  # Default, will be overridden
                    sim_config=simulation_config, 
                    ev_config=ev_config,
                    base_seed=base_seed,
                    enable_enhancements=True
                )
        except Exception as e:
            self.logger.warning(f"Could not initialize bandit framework: {e}")
            self.bandit_framework = None
        
        # Candidate storage
        self.candidate_placements = {}
        self.placement_scores = {}
        
    def generate_heuristic_candidates(self, 
                                    ved_trajectories: pd.DataFrame,
                                    num_chargers: int,
                                    heuristic_methods: List[str] = None,
                                    grid_bounds: Dict = None) -> Dict[str, List[Dict]]:
        """
        Generate candidate placements using multiple heuristics.
        
        Args:
            ved_trajectories: VED trajectory data
            num_chargers: Number of charging stations to place
            heuristic_methods: List of heuristic methods to use
            grid_bounds: Grid boundaries to constrain placement
            
        Returns:
            Dictionary mapping method names to candidate placements
        """
        if heuristic_methods is None:
            heuristic_methods = ['kmeans', 'random', 'uniform', 'uniform_random']  # More diverse methods
        
        candidates = {}
        
        # Log grid bounds information
        if grid_bounds:
            self.logger.info(f"Using grid bounds: lat=[{grid_bounds['min_lat']:.4f}, {grid_bounds['max_lat']:.4f}], "
                           f"lon=[{grid_bounds['min_lon']:.4f}, {grid_bounds['max_lon']:.4f}]")
        else:
            self.logger.info("No grid bounds provided - using full network")
        
        # Generate more candidates per method for better exploration
        # Use 15-30x more candidates than needed for robust statistics
        candidate_multiplier = max(15, min(30, 100 // num_chargers))  # Increased for better diversity
        candidates_per_method = num_chargers * candidate_multiplier
        
        self.logger.info(f"Generating {candidates_per_method} candidates per method for exploration")
        
        # CRITICAL: Set deterministic seed before generating candidates for reproducibility
        # This ensures the same candidates are generated for the same (base_seed, grid_id) combination
        np.random.seed(self.base_seed)
        random.seed(self.base_seed)
        self.logger.info(f"Set seed {self.base_seed} for deterministic candidate generation")
        
        for method in heuristic_methods:
            try:
                # Re-seed before each method to ensure deterministic candidate generation
                # Use different seeds for different methods to ensure diversity
                # REPRODUCIBILITY FIX: Use deterministic hash instead of Python's hash()
                method_hash = sum(ord(c) for c in str(method)) % 1000
                method_seed = self.base_seed + method_hash
                np.random.seed(method_seed)
                random.seed(method_seed)
                
                if method == 'kmeans':
                    candidates[method] = self.heuristic_baselines.demand_driven_clustering_baseline(
                        ved_trajectories, candidates_per_method, grid_bounds=grid_bounds
                    )
                elif method == 'random':
                    candidates[method] = self.heuristic_baselines.random_placement_baseline(
                        ved_trajectories, candidates_per_method, grid_bounds=grid_bounds
                    )
                elif method == 'uniform':
                    candidates[method] = self.heuristic_baselines.uniform_spacing_baseline(
                        ved_trajectories, candidates_per_method, grid_bounds=grid_bounds
                    )
                elif method == 'uniform_random':
                    candidates[method] = self.heuristic_baselines.uniform_random_baseline(
                        candidates_per_method
                    )
                else:
                    self.logger.warning(f"Unknown heuristic method: {method}")
                    continue
                
                self.logger.info(f"Generated {len(candidates[method])} candidates using {method}")
                
                # Log placement details for debugging
                if candidates[method]:
                    for i, placement in enumerate(candidates[method][:3]):  # Show first 3
                        self.logger.debug(f"  {method} placement {i}: lat={placement.get('lat', 'N/A'):.4f}, "
                                        f"lon={placement.get('lon', 'N/A'):.4f}, "
                                        f"edge_id={placement.get('edge_id', 'N/A')}")
                
            except Exception as e:
                self.logger.error(f"Failed to generate candidates using {method}: {e}")
                candidates[method] = []
        
        return candidates
    
    def create_hybrid_bandit_algorithms(self, 
                                      candidates: Dict[str, List[Dict]],
                                      num_chargers: int,
                                      bandit_types: List[str] = None) -> Dict[str, Any]:
        """
        Create bandit algorithms that work with heuristic candidates.
        
        Instead of treating individual candidates as arms, we create combinations
        of candidates as arms to properly optimize placement configurations.
        
        Args:
            candidates: Dictionary of heuristic candidates
            num_chargers: Number of charging stations to place
            bandit_types: List of bandit algorithm types
            
        Returns:
            Dictionary of bandit algorithm instances
        """
        if bandit_types is None:
            bandit_types = ['ucb', 'epsilon_greedy', 'thompson_sampling']
        
        # Generate placement combinations from candidates
        placement_combinations = self._generate_placement_combinations(candidates, num_chargers)
        
        self.logger.info(f"Generated {len(placement_combinations)} placement combinations from candidates")
        
        # Create bandit algorithms with proper number of arms
        bandit_algorithms = {}
        num_arms = len(placement_combinations)
        
        for bandit_type in bandit_types:
            try:
                if bandit_type == 'ucb':
                    from .models import UCB
                    bandit_algorithms[bandit_type] = UCB(
                        exploration_parameter=3.0,  # Even higher exploration
                        min_pulls_per_action=1,  # Allow immediate UCB
                        max_consecutive_same=1,  # Force exploration after 1 same action
                        seed=self.base_seed  # Use base seed for reproducibility
                    )
                elif bandit_type == 'epsilon_greedy':
                    from .models import EpsilonGreedy
                    bandit_algorithms[bandit_type] = EpsilonGreedy(
                        epsilon=0.6,  # Even higher initial exploration
                        epsilon_decay=0.95,  # Much slower decay
                        min_epsilon=0.2,  # Higher minimum exploration
                        max_consecutive_same=1,  # Force exploration after 1 same action
                        seed=self.base_seed  # Use base seed for reproducibility
                    )
                elif bandit_type == 'thompson_sampling':
                    from .models import ThompsonSampling
                    bandit_algorithms[bandit_type] = ThompsonSampling(
                        success_threshold=0.0,
                        max_consecutive_same=2,  # Force exploration after 2 same actions
                        seed=self.base_seed  # Use base seed for reproducibility
                    )
                else:
                    self.logger.warning(f"Unknown bandit type: {bandit_type}")
                    continue
                
                self.logger.info(f"Created {bandit_type} bandit algorithm with {num_arms} arms")
                
            except Exception as e:
                self.logger.error(f"Failed to create {bandit_type} bandit: {e}")
                continue
        
        # Store placement combinations for bandit algorithms
        self.placement_combinations = placement_combinations
        
        return bandit_algorithms
    
    def _generate_placement_combinations(self, candidates: Dict[str, List[Dict]], num_chargers: int) -> List[List[Dict]]:
        """
        Generate placement combinations from heuristic candidates with efficient sampling.
        
        Uses adaptive strategies based on candidate pool size:
        1. Small pools (≤20): Exhaustive combinations
        2. Medium pools (21-100): Stratified sampling by method
        3. Large pools (>100): Intelligent sampling with diversity
        
        Args:
            candidates: Dictionary of heuristic candidates
            num_chargers: Number of charging stations to place
            
        Returns:
            List of placement combinations (each combination is a list of placements)
        """
        import itertools
        import random
        import math
        
        # CRITICAL: Set deterministic seed for combination generation
        # This ensures the same combinations are selected for the same base_seed
        np.random.seed(self.base_seed)
        random.seed(self.base_seed)
        
        # Flatten all candidates with source tracking
        all_candidates = []
        method_counts = {}
        for method, placements in candidates.items():
            method_counts[method] = len(placements)
            for placement in placements:
                placement_copy = placement.copy()
                placement_copy['source_method'] = method
                all_candidates.append(placement_copy)
        
        total_candidates = len(all_candidates)
        self.logger.info(f"Generating combinations for {num_chargers} chargers from {total_candidates} candidates")
        self.logger.info(f"Candidate distribution: {method_counts}")
        
        if total_candidates < num_chargers:
            self.logger.warning(f"Not enough candidates ({total_candidates}) for {num_chargers} chargers")
            return []
        
        # Calculate theoretical combination count
        theoretical_combinations = math.comb(total_candidates, num_chargers)
        self.logger.info(f"Theoretical combinations: {theoretical_combinations:,}")
        
        # Always use intelligent sampling for better diversity
        self.logger.info("Using intelligent sampling for better diversity")
        max_combinations = min(2000, theoretical_combinations)
        combinations = self._intelligent_candidate_sampling(all_candidates, num_chargers, max_combinations)
        
        # Convert to list of lists and validate
        placement_combinations = []
        for combo in combinations:
            combo_list = list(combo)
            # Ensure no duplicate edge_ids in combination (one station per edge)
            edge_ids = [p.get('edge_id', '') for p in combo_list]
            # For single charger, we want unique edge_ids (no duplicates)
            # For multiple chargers, we want exactly num_chargers unique edge_ids
            if len(set(edge_ids)) == len(combo_list):  # No duplicate edge_ids
                placement_combinations.append(combo_list)
        
        self.logger.info(f"Generated {len(placement_combinations)} valid placement combinations")
        
        # Debug: Log first few combinations to verify they're different
        if placement_combinations:
            self.logger.info("Debug: First 3 placement combinations:")
            for i, combo in enumerate(placement_combinations[:3]):
                self.logger.info(f"  Combination {i}: {len(combo)} placements")
                for j, placement in enumerate(combo):
                    self.logger.info(f"    Placement {j}: edge_id={placement.get('edge_id', 'N/A')}, "
                                   f"lat={placement.get('lat', 'N/A'):.4f}, lon={placement.get('lon', 'N/A'):.4f}")
        
        return placement_combinations
    
    def _stratified_candidate_sampling(self, candidates: Dict[str, List[Dict]], num_chargers: int, max_combinations: int) -> List[Tuple]:
        """
        Stratified sampling ensuring representation from each heuristic method.
        
        Args:
            candidates: Dictionary of heuristic candidates by method
            num_chargers: Number of charging stations to place
            max_combinations: Maximum number of combinations to generate
            
        Returns:
            List of sampled combinations
        """
        import random
        import itertools
        
        # Use episode-specific seed for diversity while maintaining reproducibility
        random.seed(self.base_seed + len(candidates))  # Add variation based on candidate count
        
        # Calculate proportional representation
        total_candidates = sum(len(placements) for placements in candidates.values())
        method_weights = {method: len(placements) / total_candidates for method, placements in candidates.items()}
        
        # Generate combinations ensuring method diversity
        combinations = []
        target_per_method = max_combinations // len(candidates)
        
        for method, placements in candidates.items():
            if not placements:
                continue
                
            # Generate combinations for this method (with other methods)
            method_combinations = []
            for _ in range(min(target_per_method, 500)):  # Limit per method
                # Select 1-2 from this method, rest from others
                from_this_method = min(2, max(1, num_chargers // 2))
                from_others = num_chargers - from_this_method
                
                if from_this_method <= len(placements) and from_others <= (total_candidates - len(placements)):
                    # Sample from this method
                    selected_from_method = random.sample(placements, from_this_method)
                    
                    # Sample from other methods
                    other_candidates = [p for m, plist in candidates.items() 
                                      for p in plist if m != method]
                    if len(other_candidates) >= from_others:
                        selected_from_others = random.sample(other_candidates, from_others)
                        combo = tuple(selected_from_method + selected_from_others)
                        method_combinations.append(combo)
            
            combinations.extend(method_combinations)
        
        # Add some pure random combinations for exploration
        all_candidates = [p for placements in candidates.values() for p in placements]
        random_combinations = []
        for _ in range(min(200, max_combinations - len(combinations))):
            if len(all_candidates) >= num_chargers:
                combo = tuple(random.sample(all_candidates, num_chargers))
                random_combinations.append(combo)
        
        combinations.extend(random_combinations)
        
        # Convert to hashable format and remove duplicates
        hashable_combinations = []
        for combo in combinations:
            if isinstance(combo[0], dict):
                # Convert dictionary tuples to hashable tuples
                hashable_combo = tuple(sorted((loc.get('edge_id', ''), loc.get('lat', 0), loc.get('lon', 0)) for loc in combo))
            else:
                # Already hashable
                hashable_combo = tuple(sorted(combo))
            hashable_combinations.append(hashable_combo)
        
        # Remove duplicates using hashable format
        unique_combinations = list(set(hashable_combinations))
        
        # Convert back to original format
        final_combinations = []
        for combo in unique_combinations:
            if isinstance(combo[0], tuple) and len(combo[0]) == 3:
                # Convert back from hashable format
                original_combo = []
                for edge_id, lat, lon in combo:
                    # Find the original dictionary
                    for candidate in all_candidates:
                        if (candidate.get('edge_id', '') == edge_id and 
                            abs(candidate.get('lat', 0) - lat) < 1e-6 and 
                            abs(candidate.get('lon', 0) - lon) < 1e-6):
                            original_combo.append(candidate)
                            break
                if len(original_combo) == len(combo):
                    final_combinations.append(tuple(original_combo))
            else:
                final_combinations.append(combo)
        
        return final_combinations[:max_combinations]
    
    def _intelligent_candidate_sampling(self, all_candidates: List[Dict], num_chargers: int, max_combinations: int) -> List[Tuple]:
        """
        Intelligent sampling using multiple strategies for large candidate pools.
        
        Args:
            all_candidates: List of all candidates
            num_chargers: Number of charging stations to place
            max_combinations: Maximum number of combinations to generate
            
        Returns:
            List of sampled combinations
        """
        import random
        
        # CRITICAL FIX: Use consistent seed, not len(all_candidates) which can vary!
        # Using len(all_candidates) causes different seeds when candidate count varies slightly
        random.seed(self.base_seed)
        np.random.seed(self.base_seed)
        combinations = []
        
        # Strategy 1: Method diversity sampling (20% of combinations)
        diversity_count = int(0.2 * max_combinations)
        if diversity_count > 0:
            diversity_combinations = self._generate_method_diverse_combinations(all_candidates, num_chargers, diversity_count)
            combinations.extend(diversity_combinations)
        
        # Strategy 2: Geographic diversity sampling (20% of combinations)
        geo_diversity_count = int(0.2 * max_combinations)
        if geo_diversity_count > 0:
            geo_combinations = self._generate_geographic_diverse_combinations(all_candidates, num_chargers, geo_diversity_count)
            combinations.extend(geo_combinations)
        
        # Strategy 3: Random sampling (60% of combinations) - Increased for more diversity
        random_count = max_combinations - len(combinations)
        if random_count > 0:
            random_combinations = self._generate_random_candidate_combinations(all_candidates, num_chargers, random_count)
            combinations.extend(random_combinations)
        
        # Convert to hashable format and remove duplicates
        hashable_combinations = []
        for combo in combinations:
            if isinstance(combo[0], dict):
                # Convert dictionary tuples to hashable tuples using edge_id and position
                hashable_combo = tuple(sorted((loc.get('edge_id', ''), loc.get('lat', 0), loc.get('lon', 0)) for loc in combo))
            else:
                # Already hashable
                hashable_combo = tuple(sorted(combo))
            hashable_combinations.append(hashable_combo)
        
        # CRITICAL FIX: Remove duplicates using dict.fromkeys() to preserve deterministic order
        # Using set() causes non-deterministic ordering across runs!
        unique_combinations = list(dict.fromkeys(hashable_combinations))
        
        # Convert back to original format
        final_combinations = []
        for combo in unique_combinations:
            if isinstance(combo[0], tuple) and len(combo[0]) == 3:
                # Convert back from hashable format
                original_combo = []
                for edge_id, lat, lon in combo:
                    # Find the original dictionary
                    for candidate in all_candidates:
                        if (candidate.get('edge_id', '') == edge_id and 
                            abs(candidate.get('lat', 0) - lat) < 1e-6 and 
                            abs(candidate.get('lon', 0) - lon) < 1e-6):
                            original_combo.append(candidate)
                            break
                if len(original_combo) == len(combo):
                    final_combinations.append(tuple(original_combo))
            else:
                final_combinations.append(combo)
        
        # CRITICAL FIX: Re-seed before shuffle with consistent seed (not max_combinations which can vary!)
        random.seed(self.base_seed + 999)  # Use fixed offset for shuffle
        np.random.seed(self.base_seed + 999)
        
        # Shuffle to ensure diverse combinations are seen early
        random.shuffle(final_combinations)
        
        return final_combinations[:max_combinations]
    
    def _generate_method_diverse_combinations(self, all_candidates: List[Dict], num_chargers: int, count: int) -> List[Tuple]:
        """Generate combinations ensuring diversity across heuristic methods."""
        import random
        
        # CRITICAL: Set deterministic seed for this sampling strategy
        random.seed(self.base_seed + 1)
        np.random.seed(self.base_seed + 1)
        
        # Group candidates by method
        method_groups = {}
        for candidate in all_candidates:
            method = candidate.get('source_method', 'unknown')
            method_groups.setdefault(method, []).append(candidate)
        
        combinations = []
        for _ in range(count):
            selected = []
            remaining_methods = list(method_groups.keys())
            
            # Try to select from different methods
            for i in range(num_chargers):
                if not remaining_methods:
                    # Fallback to random selection
                    if len(all_candidates) >= num_chargers - len(selected):
                        additional = random.sample(
                            [c for c in all_candidates if c not in selected],
                            num_chargers - len(selected)
                        )
                        selected.extend(additional)
                    break
                
                # Select from a random method
                method = random.choice(remaining_methods)
                method_candidates = method_groups[method]
                
                if method_candidates:
                    candidate = random.choice(method_candidates)
                    selected.append(candidate)
                    remaining_methods.remove(method)
            
            if len(selected) == num_chargers:
                combinations.append(tuple(selected))
        
        return combinations
    
    def _generate_geographic_diverse_combinations(self, all_candidates: List[Dict], num_chargers: int, count: int) -> List[Tuple]:
        """Generate combinations maximizing geographic diversity."""
        import random
        
        # CRITICAL: Set deterministic seed for this sampling strategy
        random.seed(self.base_seed + 2)
        np.random.seed(self.base_seed + 2)
        
        # Sort candidates by geographic spread
        if all_candidates and 'lat' in all_candidates[0] and 'lon' in all_candidates[0]:
            center_lat = sum(c.get('lat', 0) for c in all_candidates) / len(all_candidates)
            center_lon = sum(c.get('lon', 0) for c in all_candidates) / len(all_candidates)
            
            def distance_from_center(candidate):
                lat, lon = candidate.get('lat', 0), candidate.get('lon', 0)
                return ((lat - center_lat) ** 2 + (lon - center_lon) ** 2) ** 0.5
            
            sorted_candidates = sorted(all_candidates, key=distance_from_center)
        else:
            sorted_candidates = all_candidates
        
        combinations = []
        for _ in range(count):
            selected = []
            remaining = sorted_candidates.copy()
            
            for i in range(num_chargers):
                if not remaining:
                    break
                
                if i == 0:
                    # First candidate: random from all
                    selected.append(remaining.pop(random.randint(0, len(remaining) - 1)))
                else:
                    # Subsequent candidates: maximize distance from already selected
                    best_candidate = None
                    best_distance = -1
                    
                    for candidate in remaining:
                        if 'lat' in candidate and 'lon' in candidate:
                            min_distance = min(
                                ((candidate.get('lat', 0) - sel.get('lat', 0)) ** 2 + 
                                 (candidate.get('lon', 0) - sel.get('lon', 0)) ** 2) ** 0.5
                                for sel in selected if 'lat' in sel and 'lon' in sel
                            )
                            if min_distance > best_distance:
                                best_distance = min_distance
                                best_candidate = candidate
                        else:
                            # Fallback to random selection
                            best_candidate = candidate
                            break
                    
                    if best_candidate:
                        selected.append(best_candidate)
                        remaining.remove(best_candidate)
            
            if len(selected) == num_chargers:
                combinations.append(tuple(selected))
        
        return combinations
    
    def _generate_random_candidate_combinations(self, all_candidates: List[Dict], num_chargers: int, count: int) -> List[Tuple]:
        """Generate random combinations for exploration."""
        import random
        
        # CRITICAL: Set deterministic seed for this sampling strategy
        random.seed(self.base_seed + 3)
        np.random.seed(self.base_seed + 3)
        
        combinations = []
        for _ in range(count):
            if len(all_candidates) >= num_chargers:
                combo = tuple(random.sample(all_candidates, num_chargers))
                combinations.append(combo)
        
        return combinations
    
    
    def _log_performance_summary(self, comparison_results: Dict[str, Any]) -> None:
        """
        Log a comprehensive performance summary.
        
        ONLY logs if there are meaningful results to display.
        Prevents logging pointless 0.0000 values before methods have run.
        """
        # Check if there are actual results to report
        methods_compared = comparison_results.get('methods_compared', {})
        has_meaningful_results = False
        
        if isinstance(methods_compared, dict):
            for method_name, result in methods_compared.items():
                if isinstance(result, dict) and result.get('best_reward', 0.0) > 0.0:
                    has_meaningful_results = True
                    break
        elif isinstance(methods_compared, list):
            for result in methods_compared:
                if isinstance(result, dict) and result.get('best_reward', 0.0) > 0.0:
                    has_meaningful_results = True
                    break
        
        # Only log if there are meaningful results
        if not has_meaningful_results:
            self.logger.debug("Skipping performance summary - no meaningful results yet")
            return
        
        self.logger.info("=" * 60)
        self.logger.info("📊 HYBRID OPTIMIZATION PERFORMANCE SUMMARY")
        self.logger.info("=" * 60)
        
        # Performance ranking
        if 'performance_ranking' in comparison_results:
            ranking = comparison_results['performance_ranking']
            if 'by_reward' in ranking:
                self.logger.info(f"🏆 Performance Ranking (by reward): {ranking['by_reward']}")
            if 'by_convergence_speed' in ranking:
                self.logger.info(f"⚡ Convergence Speed Ranking: {ranking['by_convergence_speed']}")
        
        # Efficiency analysis
        if 'efficiency_analysis' in comparison_results:
            efficiency = comparison_results['efficiency_analysis']
            if 'fastest_method' in efficiency:
                self.logger.info(f"🚀 Fastest Method: {efficiency['fastest_method']}")
            if 'speedup_factors' in efficiency:
                self.logger.info("⚡ Speedup Factors:")
                for method, factor in efficiency['speedup_factors'].items():
                    if factor != float('inf'):
                        self.logger.info(f"   {method}: {factor:.2f}x")
        
        # Summary statistics
        if 'summary_statistics' in comparison_results:
            stats = comparison_results['summary_statistics']
            self.logger.info("📈 Summary Statistics:")
            # CRITICAL FIX: Handle missing or zero values properly
            best_reward = stats.get('best_overall_reward', 0.0)
            if best_reward == 0.0 or best_reward == 'N/A':
                # Try to get actual reward from the results
                if isinstance(methods_compared, dict):
                    for method_name, result in methods_compared.items():
                        if isinstance(result, dict) and 'best_reward' in result:
                            best_reward = max(best_reward, result['best_reward'])
                elif isinstance(methods_compared, list):
                    # Handle case where methods_compared is a list
                    for result in methods_compared:
                        if isinstance(result, dict) and 'best_reward' in result:
                            best_reward = max(best_reward, result['best_reward'])
            self.logger.info(f"   Best Overall Reward: {best_reward:.4f}")
            
            reward_range = stats.get('reward_range', 0.0)
            if reward_range == 0.0 or reward_range == 'N/A':
                # Calculate actual range from results
                rewards = []
                if isinstance(methods_compared, dict):
                    for method_name, result in methods_compared.items():
                        if isinstance(result, dict) and 'best_reward' in result:
                            rewards.append(result['best_reward'])
                elif isinstance(methods_compared, list):
                    # Handle case where methods_compared is a list
                    for result in methods_compared:
                        if isinstance(result, dict) and 'best_reward' in result:
                            rewards.append(result['best_reward'])
                if rewards:
                    reward_range = max(rewards) - min(rewards)
            self.logger.info(f"   Reward Range: {reward_range:.4f}")
            
            convergence_rate = stats.get('convergence_rate', 0.0)
            if convergence_rate == 0.0 or convergence_rate == 'N/A':
                # Calculate actual convergence rate
                converged_count = 0
                total_count = 0
                if isinstance(methods_compared, dict):
                    for method_name, result in methods_compared.items():
                        if isinstance(result, dict):
                            total_count += 1
                            if result.get('convergence_achieved', False):
                                converged_count += 1
                elif isinstance(methods_compared, list):
                    # Handle case where methods_compared is a list
                    for result in methods_compared:
                        if isinstance(result, dict):
                            total_count += 1
                            if result.get('convergence_achieved', False):
                                converged_count += 1
                if total_count > 0:
                    convergence_rate = converged_count / total_count
            self.logger.info(f"   Convergence Rate: {convergence_rate:.2%}")
            
            avg_episodes = stats.get('avg_episodes_to_convergence', None)
            if avg_episodes is not None and avg_episodes != 'N/A':
                self.logger.info(f"   Avg Episodes to Convergence: {avg_episodes:.1f}")
            else:
                self.logger.info(f"   Avg Episodes to Convergence: N/A")
        
        self.logger.info("=" * 60)
    
    def save_optimization_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Save comprehensive optimization results to file.
        
        Args:
            results: Optimization results dictionary
            output_path: Path to save results
        """
        import json
        import os
        from datetime import datetime
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Add metadata
        results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'framework_version': '1.0.0',
            'num_chargers': getattr(self, 'num_chargers', 'unknown'),
            'grid_id': getattr(self, 'grid_id', 'unknown')
        }
        
        # Save main results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        self.logger.info(f"💾 Optimization results saved to: {output_path}")
        
        # Save comparison analysis separately if available
        if 'comparison_analysis' in results:
            comparison_path = output_path.replace('.json', '_comparison.json')
            save_comparison_results(results['comparison_analysis'], comparison_path)
        
        # Save comprehensive metrics separately if available
        if 'comprehensive_metrics' in results:
            metrics_path = output_path.replace('.json', '_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(results['comprehensive_metrics'], f, indent=4, default=str)
            self.logger.info(f"📊 Comprehensive metrics saved to: {metrics_path}")
    
    def run_hybrid_optimization(self,
                              ved_trajectories: pd.DataFrame,
                              num_chargers: int,
                              max_episodes: int = 100,
                              heuristic_methods: List[str] = None,
                              bandit_types: List[str] = None,
                              grid_bounds: Dict = None) -> Dict[str, Any]:
        """
        Run hybrid optimization combining heuristics and bandits.
        
        Args:
            ved_trajectories: VED trajectory data
            num_chargers: Number of charging stations to place
            max_episodes: Maximum number of episodes
            heuristic_methods: Heuristic methods to use
            bandit_types: Bandit algorithm types to use
            grid_bounds: Grid boundaries for constraint
            
        Returns:
            Dictionary containing optimization results
        """
        self.logger.info(f"🚀 Starting Hybrid Optimization")
        self.logger.info(f"   • Chargers: {num_chargers}")
        self.logger.info(f"   • Episodes: {max_episodes}")
        
        # Step 1: Generate heuristic candidates
        self.logger.info("📊 Generating heuristic candidates...")
        try:
            candidates = self.generate_heuristic_candidates(
                ved_trajectories, num_chargers, heuristic_methods, grid_bounds
            )
        except Exception as e:
            self.logger.error(f"Error generating heuristic candidates: {e}")
            return {'error': f'Failed to generate candidates: {str(e)}'}
        
        if not any(candidates.values()):
            self.logger.error("No heuristic candidates generated")
            return {'error': 'No candidates generated'}
        
        # Step 2: Update bandit framework with correct number of stations
        if self.bandit_framework:
            self.logger.info(f"Before update: bandit_framework.num_stations = {self.bandit_framework.num_stations}")
            self.bandit_framework.num_stations = num_chargers
            self.logger.info(f"After update: bandit_framework.num_stations = {self.bandit_framework.num_stations}")
            self.logger.info(f"Updated bandit framework to use {num_chargers} stations")
        else:
            self.logger.warning("bandit_framework is None, cannot update num_stations")
        
        # Step 3: Create bandit algorithms with placement combinations
        self.logger.info("🎰 Creating bandit algorithms...")
        bandit_algorithms = self.create_hybrid_bandit_algorithms(
            candidates, num_chargers, bandit_types
        )
        
        if not bandit_algorithms:
            self.logger.error("No bandit algorithms created")
            return {'error': 'No bandit algorithms created'}
        
        # Step 3: Run optimization for each bandit algorithm
        results = {}
        comprehensive_metrics = {}
        
        for algo_name, bandit_algo in bandit_algorithms.items():
            self.logger.info(f"🔍 Running {algo_name} optimization...")
            
            try:
                algo_result = self._run_single_bandit_optimization(
                    bandit_algo, ved_trajectories, max_episodes, algo_name
                )
                results[algo_name] = algo_result
                
                # Calculate comprehensive metrics for this algorithm
                if 'error' not in algo_result:
                    method_name = f"hybrid_{algo_name.lower()}"
                    grid_id = getattr(self, 'grid_id', 'unknown')
                    
                    comprehensive_metric = calculate_comprehensive_metrics(
                        algo_result, method_name, num_chargers, grid_id
                    )
                    comprehensive_metrics[method_name] = comprehensive_metric
                
            except Exception as e:
                self.logger.error(f"Failed to run {algo_name}: {e}")
                results[algo_name] = {'error': str(e)}
        
        # Step 4: Compare methods and generate comprehensive analysis
        if comprehensive_metrics:
            self.logger.info("📊 Generating comprehensive method comparison...")
            comparison_results = compare_methods_performance(comprehensive_metrics)
            
            # Add comparison results to main results
            results['comparison_analysis'] = comparison_results
            results['comprehensive_metrics'] = comprehensive_metrics
            
            # Log performance summary
            self._log_performance_summary(comparison_results)
        
        return results
    
    def _run_single_bandit_optimization(self, 
                                      bandit_algo: Any,
                                      ved_trajectories: pd.DataFrame,
                                      max_episodes: int,
                                      algo_name: str) -> Dict[str, Any]:
        """
        Run optimization with a single bandit algorithm.
        
        Args:
            bandit_algo: Bandit algorithm instance
            ved_trajectories: VED trajectory data
            max_episodes: Maximum episodes
            algo_name: Algorithm name for logging
            
        Returns:
            Optimization results
        """
        episode_results = []
        episode_confidence = []  # Track confidence data per episode
        episode_reward_components = []  # Track reward components per episode
        best_reward = float('-inf')
        best_placement = None
        
        for episode in range(max_episodes):
            try:
                self.logger.info(f"   Episode {episode + 1}/{max_episodes}")
                
                # Select action (placement combination)
                # Pass available actions as indices
                available_actions = list(range(len(self.placement_combinations)))
                action_idx = bandit_algo.select_action(available_actions)
                
                if action_idx is None:
                    self.logger.warning(f"No action selected in episode {episode}")
                    continue
                    
                if action_idx >= len(self.placement_combinations):
                    self.logger.warning(f"Invalid action index {action_idx}")
                    continue
                
                placement_combo = self.placement_combinations[action_idx]
                
                # Debug: Log placement details
                self.logger.info(f"     Action {action_idx}: {len(placement_combo)} placements")
                for i, placement in enumerate(placement_combo):
                    self.logger.info(f"       Placement {i}: edge_id={placement.get('edge_id', 'N/A')}, "
                                   f"lat={placement.get('lat', 'N/A'):.4f}, lon={placement.get('lon', 'N/A'):.4f}")
                
                # Evaluate placement combination
                reward, reward_components = self._evaluate_placement_combo(placement_combo, ved_trajectories, episode)
                self.logger.info(f"     Reward: {reward:.4f}")
                
                # Update bandit algorithm
                bandit_algo.update(action_idx, reward)
                
                # Track confidence data for this episode
                confidence_data = self._extract_confidence_data(bandit_algo, action_idx)
                episode_confidence.append(confidence_data)
                self.logger.info(f"Episode {episode + 1}: Confidence data extracted: {confidence_data}")
                
                # Track reward components for this episode
                episode_reward_components.append(reward_components)
                
                # Track best result
                if reward > best_reward:
                    best_reward = reward
                    best_placement = placement_combo
                
                episode_results.append({
                    'episode': episode,
                    'action_idx': action_idx,
                    'reward': reward,
                    'placement': placement_combo
                })
                
                if episode % 1 == 0:  # Log every episode for debugging
                    self.logger.info(f"   Episode {episode + 1}: reward={reward:.4f}, best={best_reward:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Episode {episode} failed: {e}")
                continue
        
        # Proper convergence check: need at least 10 episodes and stable performance
        convergence_achieved = False
        if len(episode_results) >= 10:
            # Check if last 5 episodes show stable performance (reward variance < 0.01)
            recent_rewards = [ep['reward'] for ep in episode_results[-5:]]
            if len(recent_rewards) >= 5:
                reward_variance = np.var(recent_rewards)
                convergence_achieved = reward_variance < 0.01
        
        return {
            'algorithm': algo_name,
            'best_placement': best_placement,
            'best_reward': best_reward,
            'episode_results': episode_results,
            'total_episodes': len(episode_results),
            'convergence_achieved': convergence_achieved,
            'episode_rewards': [ep['reward'] for ep in episode_results],  # For CSV export
            'episode_confidence': episode_confidence,  # Include confidence data
            'episode_reward_components': episode_reward_components,  # Include reward components
            'actual_action_space': len(self.placement_combinations)  # Actual number of combinations
        }
    
    def _extract_confidence_data(self, bandit_algo: Any, selected_action: Any) -> Dict:
        """Extract confidence-related data from bandit algorithm statistics."""
        try:
            self.logger.info(f"Extracting confidence data for action {selected_action}")
            
            confidence_data = {
                'confidence_lower': 0.0,
                'confidence_upper': 0.0,
                'confidence_width': 0.0,
                'best_action_confidence': 0.0,
                'action_diversity': 0,
                'exploration_bonus': 0.0
            }
            
            # Get algorithm statistics
            algorithm_stats = bandit_algo.get_action_statistics()
            self.logger.info(f"Algorithm stats: {algorithm_stats}")
            
            # Get confidence interval for selected action
            if selected_action in algorithm_stats:
                action_stats = algorithm_stats[selected_action]
                ci_lower, ci_upper = action_stats.get('confidence_interval', (0.0, 0.0))
                confidence_data.update({
                    'confidence_lower': ci_lower,
                    'confidence_upper': ci_upper,
                    'confidence_width': ci_upper - ci_lower
                })
                self.logger.info(f"Selected action confidence: {ci_lower} - {ci_upper}")
            
            # Get best action confidence
            best_action = bandit_algo.get_best_action()
            if best_action and best_action in algorithm_stats:
                best_stats = algorithm_stats[best_action]
                ci_lower, ci_upper = best_stats.get('confidence_interval', (0.0, 0.0))
                confidence_data['best_action_confidence'] = ci_upper - ci_lower
                self.logger.info(f"Best action confidence: {ci_upper - ci_lower}")
            
            # Get exploration bonus for UCB algorithm
            if hasattr(bandit_algo, 'get_exploration_bonus'):
                try:
                    exploration_bonus = bandit_algo.get_exploration_bonus(selected_action)
                    confidence_data['exploration_bonus'] = exploration_bonus
                    self.logger.info(f"Exploration bonus: {exploration_bonus}")
                except Exception as e:
                    self.logger.info(f"Failed to get exploration bonus: {e}")
            
            # Set action diversity
            confidence_data['action_diversity'] = len(algorithm_stats)
            
            # CRITICAL FIX: If confidence intervals are still zero, provide meaningful fallback values
            if confidence_data['confidence_width'] == 0.0 and confidence_data['best_action_confidence'] == 0.0:
                # Use exploration bonus as a proxy for confidence uncertainty
                if confidence_data['exploration_bonus'] > 0:
                    # Higher exploration bonus means lower confidence
                    avg_reward = algorithm_stats.get(selected_action, {}).get('average_reward', 0.0)
                    if avg_reward != 0:
                        # Create a confidence interval based on exploration bonus
                        margin = min(abs(avg_reward) * 0.3, confidence_data['exploration_bonus'] * 0.1)
                        confidence_data['confidence_lower'] = avg_reward - margin
                        confidence_data['confidence_upper'] = avg_reward + margin
                        confidence_data['confidence_width'] = 2 * margin
                        confidence_data['best_action_confidence'] = 2 * margin
                        self.logger.info(f"Generated fallback confidence interval: {confidence_data['confidence_lower']} - {confidence_data['confidence_upper']}")
            
            self.logger.info(f"Final confidence data: {confidence_data}")
            return confidence_data
            
        except Exception as e:
            self.logger.warning(f"Failed to extract confidence data: {e}")
            return {
                'confidence_lower': 0.0,
                'confidence_upper': 0.0,
                'confidence_width': 0.0,
                'best_action_confidence': 0.0,
                'action_diversity': 0,
                'exploration_bonus': 0.0
            }
    
    def _evaluate_placement_combo(self, placement_combo: List[Dict], ved_trajectories: pd.DataFrame, episode: int) -> float:
        """
        Evaluate a placement combination using proper SUMO simulation.
        
        Args:
            placement_combo: List of placement dictionaries
            ved_trajectories: VED trajectory data
            episode: Episode number for route generation
            
        Returns:
            Reward score for the placement combination
        """
        if not placement_combo or len(ved_trajectories) == 0:
            return 0.0
        
        try:
            # Use the bandit framework's simulation capabilities for proper evaluation
            if self.bandit_framework:
                # Convert DataFrame to dict format expected by RouteGenerator
                if isinstance(ved_trajectories, pd.DataFrame) and not ved_trajectories.empty:
                    trajectories_dict = {}
                    for vehicle_id, group in ved_trajectories.groupby('VehId'):
                        group_clean = group[['lat', 'lon', 'timestamp']].copy().dropna()
                        if len(group_clean) >= 2:
                            trajectories_dict[str(vehicle_id)] = group_clean.reset_index(drop=True)
                    
                    if trajectories_dict:
                        # Convert placement_combo to the format expected by BanditOptimizationFramework
                        # placement_combo is a list of dicts, but _evaluate_placement_multiple_episodes
                        # expects a list of placement dicts with proper structure
                        converted_placements = []
                        for i, placement in enumerate(placement_combo):
                            if isinstance(placement, dict):
                                # Ensure the placement has the required structure for ChargingStationManager
                                edge_id = placement.get('edge_id', f'edge_{i}')
                                
                                # Validate edge exists in SUMO network and is suitable for charging stations
                                try:
                                    edge = self.bandit_framework.station_manager.net.getEdge(edge_id)
                                    if edge:
                                        # Check if edge is suitable for charging stations (not sink/source)
                                        if edge_id.endswith('-sink') or edge_id.endswith('-source'):
                                            self.logger.warning(f"Edge {edge_id} is a sink/source edge, skipping placement")
                                            continue
                                        
                                        # Get the first lane and validate it
                                        lanes = edge.getLanes()
                                        if not lanes:
                                            self.logger.warning(f"Edge {edge_id} has no lanes, skipping placement")
                                            continue
                                        
                                        lane = lanes[0]
                                        lane_id = lane.getID()
                                        edge_length = edge.getLength()
                                        
                                        # Ensure position is within edge bounds
                                        position = placement.get('position', edge_length / 2)
                                        position = max(5.0, min(position, edge_length - 5.0))  # Keep 5m from edges
                                        
                                        converted_placement = {
                                            'station_id': f'station_{i}',
                                            'edge_id': edge_id,
                                            'lane_id': lane_id,
                                            'lane': lane_id,  # Required by ChargingStationManager
                                            'position': position,
                                            'end_position': min(position + 10.0, edge_length - 1.0),  # 10m station length, keep 1m from end
                                            'lat': placement.get('lat', 0.0),
                                            'lon': placement.get('lon', 0.0)
                                        }
                                    else:
                                        self.logger.warning(f"Edge {edge_id} not found in SUMO network, skipping placement")
                                        continue
                                except Exception as e:
                                    self.logger.warning(f"Error validating edge {edge_id}: {e}, skipping placement")
                                    continue
                                converted_placements.append(converted_placement)
                        
                        if not converted_placements:
                            self.logger.warning(f"No valid placements found for episode {episode}")
                            reward = 0.0
                        else:
                            # Use the bandit framework's evaluation method with proper episode
                            # Generate routes with the correct episode number
                            route_file, sim_duration = self.bandit_framework.route_generator.generate_routes_from_data(
                                trajectories_dict, episode_id=episode
                            )
                            
                            if route_file:
                                self.logger.info(f"     Generated route file: {route_file}")
                                try:
                                    # Execute simulation with the placement using the correct method
                                    self.logger.info(f"     Creating charging stations for episode {episode}")
                                    
                                    # Create charging stations first
                                    charging_stations, additional_file = self.bandit_framework.station_manager.create_additional_file(
                                        converted_placements, file_path=None, episode_id=episode
                                    )
                                    
                                    self.logger.info(f"     Created {len(charging_stations)} charging stations")
                                    
                                    # Convert to EV routes
                                    formatted_route_file = f'{os.path.splitext(route_file)[0]}_formatted.rou.xml'
                                    self.logger.info(f"     Converting routes to EV format: {formatted_route_file}")
                                    self.bandit_framework.route_generator.convert_to_ev_routes(route_file, formatted_route_file, charging_stations)
                                    
                                    if os.path.exists(formatted_route_file):
                                        self.logger.info(f"     Running SUMO simulation for episode {episode}")
                                        # Run simulation with unique output files per grid (parallel execution safety)
                                        # Pass episode_id and grid_id for deterministic domain randomization
                                        self.bandit_framework.sim_manager.run_simulation(
                                            additional_file, formatted_route_file, 
                                            max_duration=300, 
                                            output_prefix=self.grid_id,
                                            episode_id=episode,
                                            grid_id=self.grid_id,
                                            base_seed=self.base_seed
                                        )
                                        
                                        # CRITICAL FIX: Use the actual output prefix from SimulationManager
                                        # which includes worker_id, timestamp, and episode_id for parallel safety
                                        actual_prefix = self.bandit_framework.sim_manager.last_output_prefix
                                        self.logger.debug(f"Using actual prefix for file reads: {actual_prefix}")
                                        
                                        # Parse results using comprehensive SimulationAnalyzer with unique prefixed filenames
                                        output_dir = getattr(self.bandit_framework.sim_manager, 'last_output_dir', 
                                                            os.path.abspath("logs/simulation_outputs"))
                                        # CRITICAL FIX: Include output_dir in file paths
                                        battery_data = self.bandit_framework.analyzer.parse_battery_data(os.path.join(output_dir, f"{actual_prefix}_Battery.out.xml"))
                                        charging_data = self.bandit_framework.analyzer.parse_charging_events(os.path.join(output_dir, f"{actual_prefix}_chargingevents.xml"))
                                        summary_data = self.bandit_framework.analyzer.parse_summary(os.path.join(output_dir, f"{actual_prefix}_summary.xml"))
                                        
                                        # Collect network statistics with unique output prefix and directory
                                        network_state = self.bandit_framework.analyzer.collect_network_statistics(
                                            output_dir=output_dir, output_prefix=actual_prefix)
                                        
                                        # Calculate comprehensive reward using SimulationAnalyzer
                                        reward, reward_components = self.bandit_framework.analyzer.calculate_reward(
                                            battery_data, charging_data, summary_data, network_state, return_components=True
                                        )
                                        
                                        # Clean up temporary simulation files to save disk space
                                        self.bandit_framework.sim_manager.cleanup_output_files()
                                        
                                        self.logger.info(f"     Comprehensive reward: {reward:.4f}")
                                        return reward, reward_components
                                    else:
                                        self.logger.warning(f"Formatted route file not created: {formatted_route_file}")
                                        reward = 0.0
                                        reward_components = {
                                            'charging_score': 0.0, 'network_score': 0.0, 'battery_score': 0.0, 'traffic_score': 0.0,
                                            'weighted_charging': 0.0, 'weighted_network': 0.0, 'weighted_battery': 0.0, 'weighted_traffic': 0.0
                                        }
                                except Exception as e:
                                    self.logger.warning(f"SUMO evaluation failed: {e}, using simplified evaluation")
                                    reward = 0.0
                                    reward_components = {
                                        'charging_score': 0.0, 'network_score': 0.0, 'battery_score': 0.0, 'traffic_score': 0.0,
                                        'weighted_charging': 0.0, 'weighted_network': 0.0, 'weighted_battery': 0.0, 'weighted_traffic': 0.0
                                    }
                            else:
                                self.logger.warning(f"No route file generated for episode {episode}, using simplified evaluation")
                                reward = 0.0
                                reward_components = {
                                    'charging_score': 0.0, 'network_score': 0.0, 'battery_score': 0.0, 'traffic_score': 0.0,
                                    'weighted_charging': 0.0, 'weighted_network': 0.0, 'weighted_battery': 0.0, 'weighted_traffic': 0.0
                                }
                        
                        return reward, reward_components
            else:
                # Fallback if bandit framework not available
                self.logger.warning("Bandit framework not available, returning 0.0")
                return 0.0, {
                    'charging_score': 0.0, 'network_score': 0.0, 'battery_score': 0.0, 'traffic_score': 0.0,
                    'weighted_charging': 0.0, 'weighted_network': 0.0, 'weighted_battery': 0.0, 'weighted_traffic': 0.0
                }
                
        except Exception as e:
            self.logger.warning(f"SUMO evaluation failed: {e}, returning 0.0")
            return 0.0, {
                'charging_score': 0.0, 'network_score': 0.0, 'battery_score': 0.0, 'traffic_score': 0.0,
                'weighted_charging': 0.0, 'weighted_network': 0.0, 'weighted_battery': 0.0, 'weighted_traffic': 0.0
            }
    
    
    def _run_bandit_episodes(self, 
                           bandit_algorithm: Any,
                           max_episodes: int,
                           ved_trajectories: pd.DataFrame,
                           num_chargers: int) -> List[Dict]:
        """
        Run bandit episodes and collect results.
        
        Args:
            bandit_algorithm: Bandit algorithm instance
            max_episodes: Maximum number of episodes
            ved_trajectories: VED trajectory data
            num_chargers: Number of charging stations
            
        Returns:
            List of episode results
        """
        episode_results = []
        num_candidates = len(self.candidate_pool)
        
        # Initialize available actions for bandit algorithm
        available_actions = list(range(num_candidates))
        
        for episode in range(max_episodes):
            try:
                # Select action (candidate index)
                selected_arm = bandit_algorithm.select_action(available_actions)
                
                # Handle case where no action is selected
                if selected_arm is None:
                    selected_arm = 0  # Fallback to first candidate
                
                # Ensure selected_arm is within bounds
                if selected_arm >= num_candidates:
                    selected_arm = selected_arm % num_candidates
                
                selected_placement = self.candidate_pool[selected_arm]
                
                # Evaluate placement (simplified - in real implementation, run simulation)
                reward = self._evaluate_placement(selected_placement, ved_trajectories)
                
                # Update bandit algorithm
                bandit_algorithm.update(selected_arm, reward)
                
                # Store episode result
                episode_results.append({
                    'episode': episode,
                    'selected_arm': selected_arm,
                    'placement': selected_placement,
                    'reward': reward,
                    'metadata': self.candidate_metadata[selected_arm]
                })
                
                if episode % 20 == 0:
                    self.logger.info(f"   Episode {episode}: Arm {selected_arm}, Reward {reward:.4f}")
                    
            except Exception as e:
                self.logger.error(f"Error in episode {episode}: {e}")
                continue
        
        return episode_results
    
    def _evaluate_placement(self, 
                          placement: Dict, 
                          ved_trajectories: pd.DataFrame) -> float:
        """
        Evaluate a single placement using proper SUMO simulation.
        
        Args:
            placement: Placement configuration with lat/lon and edge info
            ved_trajectories: VED trajectory data
            
        Returns:
            Evaluation score (reward)
        """
        try:
            # Use the bandit framework's simulation capabilities for proper evaluation
            if hasattr(self, 'bandit_framework') and self.bandit_framework:
                # Convert single placement to the format expected by BanditOptimizationFramework
                if isinstance(placement, dict):
                    converted_placement = {
                        'station_id': 'station_0',
                        'edge_id': placement.get('edge_id', 'edge_0'),
                        'lane_id': placement.get('lane_id', 'edge_0_0'),
                        'position': placement.get('position', 0.0),
                        'lat': placement.get('lat', 0.0),
                        'lon': placement.get('lon', 0.0)
                    }
                    station_placements = [converted_placement]
                else:
                    station_placements = [placement]
                
                # Convert DataFrame to dict format expected by RouteGenerator
                if isinstance(ved_trajectories, pd.DataFrame) and not ved_trajectories.empty:
                    trajectories_dict = {}
                    for vehicle_id, group in ved_trajectories.groupby('VehId'):
                        group_clean = group[['lat', 'lon', 'timestamp']].copy().dropna()
                        if len(group_clean) >= 2:
                            trajectories_dict[str(vehicle_id)] = group_clean.reset_index(drop=True)
                    
                    if trajectories_dict:
                        rewards = self.bandit_framework._evaluate_placement_multiple_episodes(
                            station_placements, 
                            trajectories_dict,
                            num_episodes=1  # Single episode for efficiency
                        )
                    else:
                        rewards = []
                elif isinstance(ved_trajectories, dict) and len(ved_trajectories) > 0:
                    # Already in dict format
                    rewards = self.bandit_framework._evaluate_placement_multiple_episodes(
                        station_placements, 
                        ved_trajectories,
                        num_episodes=1  # Single episode for efficiency
                    )
                else:
                    rewards = []
                
                if rewards:
                    return rewards[0]
                else:
                    return self._simplified_evaluation(placement, ved_trajectories)
            else:
                # Fallback to simplified evaluation if bandit framework not available
                return self._simplified_evaluation(placement, ved_trajectories)
                
        except Exception as e:
            self.logger.warning(f"Error in placement evaluation: {e}, using simplified evaluation")
            return self._simplified_evaluation(placement, ved_trajectories)
    
    def _simplified_evaluation(self, placement: Dict, ved_trajectories: pd.DataFrame) -> float:
        """Simplified evaluation as fallback."""
        if len(ved_trajectories) == 0:
            return 0.0
        
        # Ensure placement has lat/lon coordinates
        if 'lat' not in placement or 'lon' not in placement:
            self.logger.warning("Placement missing lat/lon coordinates")
            return 0.0
        
        # Calculate distance to nearest demand points
        station_lat = placement['lat']
        station_lon = placement['lon']
        
        # Calculate distances using proper geographic distance
        distances = np.sqrt(
            (ved_trajectories['lat'] - station_lat) ** 2 +
            (ved_trajectories['lon'] - station_lon) ** 2
        )
        
        # Reward based on proximity to demand (closer = better)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        
        # Normalize to 0-1 scale (closer = higher reward)
        if max_distance > min_distance:
            reward = 1.0 - (min_distance - min_distance) / (max_distance - min_distance)
        else:
            reward = 1.0
        
        # Add bonus for having proper SUMO network integration
        if 'edge_id' in placement and not placement['edge_id'].startswith('fallback_'):
            reward += 0.1  # Small bonus for proper network integration
        
        return min(reward, 1.0)  # Cap at 1.0
    
    def _get_best_placement(self, episode_results: List[Dict]) -> Dict:
        """
        Get the best placement from episode results.
        
        Args:
            episode_results: List of episode results
            
        Returns:
            Best placement configuration
        """
        if not episode_results:
            return {}
        
        # Find episode with highest reward
        best_episode = max(episode_results, key=lambda x: x['reward'])
        return best_episode['placement']
    
    def _analyze_candidate_usage(self, episode_results: List[Dict]) -> List[Dict]:
        """
        Analyze which candidates were used most frequently.
        
        Args:
            episode_results: List of episode results
            
        Returns:
            List of candidate usage statistics
        """
        usage_count = defaultdict(int)
        
        for result in episode_results:
            metadata = result['metadata']
            method = metadata['method']
            usage_count[method] += 1
        
        return [
            {'method': method, 'count': count}
            for method, count in usage_count.items()
        ]
    
    def compare_methods(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare all methods and generate summary statistics.
        
        Args:
            results: Results from hybrid optimization
            
        Returns:
            Comparison summary
        """
        comparison = {
            'method_count': len(results),
            'methods': list(results.keys()),
            'summary': {}
        }
        
        for method_name, result in results.items():
            if 'error' in result:
                comparison['summary'][method_name] = {'status': 'failed', 'error': result['error']}
                continue
            
            if 'best_placement' in result:
                # Bandit method
                placement = result['best_placement']
                comparison['summary'][method_name] = {
                    'status': 'success',
                    'type': 'bandit',
                    'placement_count': 1,
                    'coordinates': [(placement['lat'], placement['lon'])] if placement else []
                }
            elif 'placement' in result:
                # Pure heuristic method
                placements = result['placement']
                comparison['summary'][method_name] = {
                    'status': 'success',
                    'type': 'heuristic',
                    'placement_count': len(placements),
                    'coordinates': [(p['lat'], p['lon']) for p in placements]
                }
        
        return comparison
