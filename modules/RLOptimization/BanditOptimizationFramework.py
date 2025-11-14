"""
Bandit-Based Optimization Framework for EV Charging Station Placement

This module implements a mathematically rigorous bandit-based optimization framework
for EV charging station placement. The framework treats each possible placement
configuration as an "arm" in a multi-armed bandit problem and uses bandit algorithms
to efficiently explore the space of possible placements while converging to optimal solutions.

Mathematical Foundation:
- Each placement configuration is treated as an "arm" with unknown reward distribution
- Bandit algorithms balance exploration vs exploitation to minimize regret
- Confidence thresholds ensure statistical significance of results
- Multiple bandit algorithms (UCB, Epsilon-Greedy, Thompson Sampling) for comparison

Key Features:
1. Confidence-based stopping criteria
2. Multiple bandit algorithm support
3. Comprehensive statistical analysis
4. Integration with SUMO simulation
5. Heuristic baseline comparison
"""

import os
import pandas as pd
import numpy as np
import math
import random
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict
import logging
import json
from datetime import datetime
import sys

# Core framework imports
from modules.config.SimulationConfig import SimulationConfig
from modules.config.EVConfig import EVConfig
from modules.RLOptimization.SumoNetwork import SUMONetwork
from modules.RLOptimization.RouteGenerator import RouteGenerator
from modules.RLOptimization.SimulationManager import SimulationManager
from modules.RLOptimization.SimulationAnalyzer import SimulationAnalyzer
from modules.RLOptimization.ChargingStationManager import ChargingStationManager
from modules.RLOptimization.metrics import fleet_battery_metrics, charging_metrics

# Bandit algorithms
from modules.RLOptimization.models import BaseBandit, UCB, EpsilonGreedy, ThompsonSampling

# Heuristic baselines
from modules.RLOptimization.HeuristicBaselines import HeuristicBaselines

# Optional enhanced components
try:
    from modules.RLOptimization.DomainRandomization import DomainRandomization, apply_domain_randomization
    from modules.RLOptimization.ValidationFramework import ValidationFramework
except ImportError as e:
    print(f"Optional enhanced components not available: {e}")
    DomainRandomization = None
    ValidationFramework = None


class BanditOptimizationFramework:
    """
    Enhanced optimization framework using bandit algorithms for EV charging station placement.
    
    This framework implements a rigorous bandit-based approach to optimize charging station
    placement by treating each possible placement configuration as an "arm" in a multi-armed
    bandit problem. The framework supports multiple bandit algorithms and includes confidence
    thresholds to ensure statistical significance of results.
    """
    
    def _setup_logging(self):
        """Setup simple logging for the framework."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def __init__(self,
                 osm_file_path: str,
                 num_stations: int,
                 sim_config: SimulationConfig = SimulationConfig(),
                 ev_config: EVConfig = EVConfig(),
                 reward_weights: Dict[str, float] = None,
                 placement_grid_id: str = None,
                 grid_cells_data: Optional[List[Dict]] = None,
                 enable_enhancements: bool = True,
                 randomization_config: Optional[Dict] = None,
                 base_seed: int = 42,
                 grid_id: str = None):
        """
        Initialize the bandit-based optimization framework.
        
        Parameters
        ----------
        osm_file_path : str
            Path to OSM or SUMO network file (.osm.xml or .net.xml)
        num_stations : int
            Number of charging stations to place
        sim_config : SimulationConfig
            SUMO simulation configuration
        ev_config : EVConfig
            EV configuration
        reward_weights : Dict[str, float], optional
            Weights for different reward components
        placement_grid_id : str, optional
            Grid ID for placement
        grid_cells_data : List[Dict], optional
            Grid cell data for placement constraints
        enable_enhancements : bool
            Whether to enable enhanced features
        randomization_config : Dict, optional
            Domain randomization configuration
        """
        # Setup comprehensive logging first
        self._setup_logging()
        
        # Set random seeds for reproducibility
        np.random.seed(base_seed)
        random.seed(base_seed)
        
        # Validate inputs
        if not os.path.exists(osm_file_path):
            self.logger.error(f"Network file not found: {osm_file_path}")
            raise FileNotFoundError(f"Network file not found: {osm_file_path}")
        
        self.logger.info(f"Initializing Bandit Optimization Framework")
        self.logger.info(f"  Grid ID: {placement_grid_id}")
        self.logger.info(f"  Number of stations: {num_stations}")
        self.logger.info(f"  Network file: {osm_file_path}")
        self.logger.info(f"  Base seed: {base_seed}")
        self.logger.info(f"  Enhancements enabled: {enable_enhancements}")
        
        # Store base seed and grid_id for reproducibility
        self.base_seed = base_seed
        self.grid_id = grid_id or placement_grid_id or "default"
        
        # Initialize network and core components
        try:
            self.network = SUMONetwork(osm_file_path)
            self.logger.info(f"Network initialized with {len(self.network.get_edges())} edges")
        except Exception as e:
            self.logger.error(f"Error initializing network: {e}")
            raise
            
        # Core components
        # REPRODUCIBILITY FIX: Pass base_seed and grid_id to RouteGenerator
        self.route_generator = RouteGenerator(self.network, ev_config, base_seed=base_seed, grid_id=grid_id)
        self.sim_manager = SimulationManager(self.network, sim_config)
        self.num_stations = num_stations
        self.station_manager = ChargingStationManager(
            self.network.net_file_path,
            placement_grid_id=placement_grid_id,
            grid_cells_data=grid_cells_data
        )
        self.analyzer = SimulationAnalyzer(reward_weights)
        
        # Grid information
        self.placement_grid_id = placement_grid_id
        self.grid_cells_data = grid_cells_data
        
        # Enhanced components
        self.enable_enhancements = enable_enhancements
        if enable_enhancements:
            self._initialize_enhanced_components(randomization_config)
        
        # Optimization state
        self.optimization_history = []
        self.best_placement = None
        self.best_reward = float('-inf')
        self.convergence_metrics = {}
        
    def _initialize_enhanced_components(self, randomization_config: Optional[Dict] = None):
        """Initialize enhanced components for advanced optimization."""
        try:
            # Domain randomization
            if DomainRandomization is not None:
                self.domain_randomizer = DomainRandomization(randomization_config)
                self.logger.info("✅ Domain randomization enabled")
            
            # Heuristic baselines
            self.heuristic_baselines = HeuristicBaselines(self.network.net_file_path)
            self.logger.info("✅ Heuristic baselines available")
            
            # Validation framework
            if ValidationFramework is not None:
                self.validation_framework = ValidationFramework()
                self.logger.info("✅ Validation framework enabled")
                
        except Exception as e:
            self.logger.warning(f"Some enhanced components unavailable: {e}")
    
    def run_bandit_optimization(self,
                               algorithm: BaseBandit,
                               max_episodes: int = 1000,
                               confidence_threshold: float = 0.95,
                               min_episodes: int = 50,
                               data_source: Any = 'random',
                               early_stopping_patience: int = 100,
                               ved_trajectories: Optional[pd.DataFrame] = None) -> Tuple[Optional[List[Dict]], Dict]:
        """
        Run bandit-based optimization with confidence thresholds.
        
        This is the core optimization method that uses bandit algorithms to find
        optimal charging station placements while ensuring statistical significance.
        
        Parameters
        ----------
        algorithm : BaseBandit
            Bandit algorithm to use (UCB, EpsilonGreedy, or ThompsonSampling)
        max_episodes : int
            Maximum number of episodes to run
        confidence_threshold : float
            Required confidence level for convergence (0.0 to 1.0)
        min_episodes : int
            Minimum number of episodes before checking convergence
        data_source : Any
            Data source for route generation
        early_stopping_patience : int
            Episodes to wait without improvement before early stopping
        ved_trajectories : pd.DataFrame, optional
            VED trajectory data for enhanced route generation
            
        Returns
        -------
        Tuple[Optional[List[Dict]], Dict]
            (optimal_placement, optimization_metrics)
        """
        self.logger.info(f"🚀 Starting bandit optimization with {algorithm.__class__.__name__}")
        self.logger.info(f"Confidence threshold: {confidence_threshold}")
        self.logger.info(f"Max episodes: {max_episodes}, Min episodes: {min_episodes}")
        
        # Initialize optimization state
        self.optimization_history = []
        self.best_placement = None
        self.best_reward = float('-inf')
        self.convergence_metrics = {}
        
        # Statistics tracking
        episode_rewards = []
        action_rewards = defaultdict(list)
        episode_confidence = []  # Track confidence values episode by episode
        convergence_episode = None
        
        # Generate initial placement configurations
        placement_configs = self._generate_placement_configurations()
        self.logger.info(f"Generated {len(placement_configs)} placement configurations")
        
        # Main optimization loop
        for episode in range(max_episodes):
            try:
                # Set episode-specific seed for reproducible action selection
                if hasattr(algorithm, 'set_episode_seed'):
                    algorithm.set_episode_seed(episode)
                
                # Select action (placement configuration) using bandit algorithm
                selected_config = algorithm.select_action(placement_configs)
                
                if selected_config is None:
                    self.logger.warning(f"No action selected in episode {episode}")
                    continue
                
                # Generate routes for this episode
                route_file, simulation_duration = self._generate_routes_for_episode(
                    data_source, episode, ved_trajectories
                )
                
                if route_file is None:
                    self.logger.warning(f"Route generation failed in episode {episode}")
                    continue
                
                # Execute simulation with selected placement
                reward, placement_data = self._execute_simulation(
                    selected_config, route_file, episode
                )
                
                if reward is None:
                    self.logger.warning(f"Simulation failed in episode {episode}")
                    continue
                
                # Update bandit algorithm
                algorithm.update(selected_config, reward)
                
                # Track statistics
                episode_rewards.append(reward)
                action_rewards[selected_config].append(reward)
                
                # Track confidence values for this episode
                algorithm_stats = algorithm.get_action_statistics()
                confidence_data = self._extract_confidence_data(algorithm_stats, selected_config, algorithm)
                episode_confidence.append(confidence_data)
                self.logger.info(f"Episode {episode + 1}: Confidence data extracted: {confidence_data}")
                
                # Update best placement
                if reward > self.best_reward:
                    self.best_reward = reward
                    self.best_placement = placement_data
                    self.logger.info(f"✅ New best reward: {reward:.4f} (Episode {episode + 1})")
                
                # Record optimization step
                self.optimization_history.append({
                    'episode': episode + 1,
                    'action': selected_config,
                    'reward': reward,
                    'best_reward': self.best_reward,
                    'algorithm_stats': algorithm_stats
                })
                
                # Check convergence criteria
                if episode >= min_episodes:
                    if self._check_convergence(algorithm, confidence_threshold, episode):
                        convergence_episode = episode + 1
                        self.logger.info(f"🎯 Convergence achieved at episode {convergence_episode}")
                        break
                
                # Early stopping check
                if self._check_early_stopping(episode_rewards, early_stopping_patience):
                    self.logger.info(f"⏹️ Early stopping at episode {episode + 1}")
                    break
                    
            except Exception as e:
                self.logger.error(f"Error in episode {episode + 1}: {e}")
                continue
        
        # Calculate final metrics
        metrics = self._calculate_optimization_metrics(
            algorithm, episode_rewards, action_rewards, convergence_episode, episode_confidence
        )
        
        self.logger.info(f"Optimization completed: {len(episode_rewards)} successful episodes")
        self.logger.info(f"Best reward: {self.best_reward:.4f}")
        
        return self.best_placement, metrics
    
    def _generate_placement_configurations(self) -> List[Any]:
        """
        Generate all possible placement configurations for the bandit algorithm.
        
        This method creates a comprehensive set of placement configurations by:
        1. Generating all valid placement locations within the grid
        2. Creating combinations of these locations for the required number of stations
        3. Ensuring proper exploration of the action space
        
        Returns
        -------
        List[Any]
            List of placement configuration identifiers
        """
        self.logger.info(f"Generating placement configurations for {self.num_stations} stations")
        
        # Generate all valid placement locations
        valid_locations = self._generate_valid_placement_locations()
        
        if not valid_locations:
            self.logger.error("No valid placement locations found")
            return []
        
        self.logger.info(f"Found {len(valid_locations)} valid placement locations")
        
        # Generate combinations of locations
        configs = self._generate_placement_combinations(valid_locations, self.num_stations)
        
        self.logger.info(f"Generated {len(configs)} placement configurations")
        return configs
    
    def _generate_valid_placement_locations(self) -> List[Dict]:
        """Generate all valid placement locations within the grid constraints."""
        valid_locations = []
        
        try:
            # Use the station manager to get valid locations
            # This respects grid constraints and network topology
            locations = self.station_manager.generate_placement(self.num_stations * 10)  # Get more locations
            
            # Add additional locations by sampling from the network
            if self.network and hasattr(self.network, 'get_edges'):
                edges = self.network.get_edges()
                for edge in edges[:50]:  # Sample first 50 edges
                    # Handle both dictionary and SUMO Edge object formats
                    if isinstance(edge, dict):
                        edge_id = edge.get('id', '')
                        edge_length = edge.get('length', 100)
                        lat = edge.get('lat', 0)
                        lon = edge.get('lon', 0)
                    else:
                        # SUMO Edge object
                        edge_id = edge.getID()
                        edge_length = edge.getLength()
                        # Get coordinates from the edge
                        from_lane = edge.getLanes()[0] if edge.getLanes() else None
                        if from_lane:
                            lat, lon = self.network.to_lat_lon(from_lane.getShape()[0][0], from_lane.getShape()[0][1])
                        else:
                            lat, lon = 0, 0
                    
                    if not edge_id.startswith(':'):  # Skip junction edges
                        # Add multiple positions along the edge
                        for pos in [edge_length * 0.2, edge_length * 0.5, edge_length * 0.8]:
                            if pos < edge_length:
                                valid_locations.append({
                                    'edge_id': edge_id,
                                    'lane_id': f"{edge_id}_0",
                                    'position': pos,
                                    'lat': lat,
                                    'lon': lon
                                })
            
            # Add the locations from station manager
            valid_locations.extend(locations)
            
        except Exception as e:
            self.logger.error(f"Error generating valid locations: {e}")
            return []
        
        # Remove duplicates based on (edge_id, position)
        unique_locations = []
        seen = set()
        for loc in valid_locations:
            key = (loc.get('edge_id'), loc.get('position'))
            if key not in seen:
                seen.add(key)
                unique_locations.append(loc)
        
        return unique_locations
    
    def _generate_placement_combinations(self, locations: List[Dict], num_stations: int) -> List[Tuple]:
        """
        Generate combinations of placement locations with efficient sampling for large spaces.
        
        Uses adaptive sampling strategies to balance exploration with computational efficiency:
        1. For small spaces (≤20 locations): Generate all combinations
        2. For medium spaces (21-100 locations): Use stratified sampling
        3. For large spaces (>100 locations): Use intelligent sampling with diversity
        """
        import itertools
        import random
        import math
        
        total_locations = len(locations)
        self.logger.info(f"Generating combinations for {num_stations} stations from {total_locations} locations")
        
        # Calculate theoretical combination count
        if total_locations >= num_stations:
            theoretical_combinations = math.comb(total_locations, num_stations)
        else:
            self.logger.warning(f"Not enough locations ({total_locations}) for {num_stations} stations")
            return []
        
        self.logger.info(f"Theoretical combinations: {theoretical_combinations:,}")
        
        # Adaptive sampling strategy based on combination space size
        if theoretical_combinations <= 1000:
            # Small space: Generate all combinations
            self.logger.info("Using exhaustive generation for small combination space")
            selected_locations = locations
            combinations = list(itertools.combinations(selected_locations, num_stations))
            
        elif theoretical_combinations <= 10000:
            # Medium space: Stratified sampling
            self.logger.info("Using stratified sampling for medium combination space")
            # Sample locations to reduce space while maintaining diversity
            max_locations = min(30, total_locations)  # Reduce to manageable size
            selected_locations = self._stratified_location_sampling(locations, max_locations)
            combinations = list(itertools.combinations(selected_locations, num_stations))
            
        else:
            # Large space: Intelligent sampling
            self.logger.info("Using intelligent sampling for large combination space")
            max_combinations = 2000  # Reasonable limit for computational efficiency
            combinations = self._intelligent_combination_sampling(locations, num_stations, max_combinations)
        
        # Convert to configuration keys
        configs = []
        for combo in combinations:
            # Create a unique key for this combination
            # Handle both dictionary and tuple formats
            if isinstance(combo[0], dict):
                config_key = tuple(sorted((loc.get('edge_id', ''), loc.get('position', 0)) for loc in combo))
            else:
                # combo is already a tuple of (edge_id, position) pairs
                config_key = tuple(sorted(combo))
            configs.append(config_key)
        
        # Final sampling if still too many
        if len(configs) > 2000:
            random.seed(self.base_seed)  # For reproducibility
            configs = random.sample(configs, 2000)
            self.logger.info(f"Final sampling: reduced to {len(configs)} combinations")
        
        self.logger.info(f"Generated {len(configs)} placement configurations")
        return configs
    
    def _stratified_location_sampling(self, locations: List[Dict], max_locations: int) -> List[Dict]:
        """
        Stratified sampling of locations to maintain geographic diversity.
        
        Args:
            locations: List of all available locations
            max_locations: Maximum number of locations to select
            
        Returns:
            List of sampled locations maintaining diversity
        """
        if len(locations) <= max_locations:
            return locations
        
        # Group locations by geographic regions (simplified: by lat/lon ranges)
        regions = {}
        for loc in locations:
            lat = loc.get('lat', 0)
            lon = loc.get('lon', 0)
            # Create region key based on lat/lon grid
            region_key = (round(lat, 1), round(lon, 1))
            regions.setdefault(region_key, []).append(loc)
        
        # Sample from each region proportionally
        sampled_locations = []
        locations_per_region = max(1, max_locations // len(regions))
        
        for region_locations in regions.values():
            if len(region_locations) <= locations_per_region:
                sampled_locations.extend(region_locations)
            else:
                import random
                random.seed(self.base_seed)  # For reproducibility
                sampled_locations.extend(random.sample(region_locations, locations_per_region))
        
        # If we still need more locations, add randomly
        if len(sampled_locations) < max_locations:
            remaining_locations = [loc for loc in locations if loc not in sampled_locations]
            needed = max_locations - len(sampled_locations)
            if remaining_locations and needed > 0:
                import random
                random.seed(self.base_seed)
                sampled_locations.extend(random.sample(remaining_locations, min(needed, len(remaining_locations))))
        
        return sampled_locations[:max_locations]
    
    def _intelligent_combination_sampling(self, locations: List[Dict], num_stations: int, max_combinations: int) -> List[Tuple]:
        """
        Intelligent sampling of combinations for large spaces.
        
        Uses multiple strategies:
        1. Diversity-based sampling to ensure geographic spread
        2. Random sampling for exploration
        3. Heuristic-guided sampling based on network topology
        
        Args:
            locations: List of all available locations
            num_stations: Number of stations to place
            max_combinations: Maximum number of combinations to generate
            
        Returns:
            List of sampled combinations
        """
        import random
        import itertools
        
        random.seed(self.base_seed)  # For reproducibility
        combinations = []
        
        # Strategy 1: Diversity-based sampling (40% of combinations)
        diversity_count = int(0.4 * max_combinations)
        if diversity_count > 0:
            diversity_combinations = self._generate_diverse_combinations(locations, num_stations, diversity_count)
            combinations.extend(diversity_combinations)
        
        # Strategy 2: Random sampling (40% of combinations)
        random_count = int(0.4 * max_combinations)
        if random_count > 0:
            random_combinations = self._generate_random_combinations(locations, num_stations, random_count)
            combinations.extend(random_combinations)
        
        # Strategy 3: Heuristic-guided sampling (20% of combinations)
        heuristic_count = max_combinations - len(combinations)
        if heuristic_count > 0:
            heuristic_combinations = self._generate_heuristic_combinations(locations, num_stations, heuristic_count)
            combinations.extend(heuristic_combinations)
        
        # Convert to hashable format and remove duplicates
        hashable_combinations = []
        skipped_duplicates = 0
        for combo in combinations:
            if isinstance(combo[0], dict):
                # Check for duplicate edge_ids within the same combination
                edge_ids = [loc.get('edge_id', '') for loc in combo]
                if len(edge_ids) != len(set(edge_ids)):
                    # Skip combinations with duplicate edge_ids
                    skipped_duplicates += 1
                    continue
                # Convert dictionary tuples to hashable tuples
                hashable_combo = tuple(sorted((loc.get('edge_id', ''), loc.get('position', 0)) for loc in combo))
            else:
                # Already hashable
                hashable_combo = tuple(sorted(combo))
            hashable_combinations.append(hashable_combo)
        
        if skipped_duplicates > 0:
            self.logger.warning(f"Skipped {skipped_duplicates} combinations with duplicate edge_ids")
        
        unique_combinations = list(set(hashable_combinations))
        
        # If we still need more, fill with random combinations
        if len(unique_combinations) < max_combinations:
            needed = max_combinations - len(unique_combinations)
            additional_combinations = self._generate_random_combinations(locations, num_stations, needed)
            unique_combinations.extend(additional_combinations)
        
        return unique_combinations[:max_combinations]
    
    def _generate_diverse_combinations(self, locations: List[Dict], num_stations: int, count: int) -> List[Tuple]:
        """Generate combinations that maximize geographic diversity."""
        import random
        
        # Sort locations by distance from center to ensure spread
        center_lat = sum(loc.get('lat', 0) for loc in locations) / len(locations)
        center_lon = sum(loc.get('lon', 0) for loc in locations) / len(locations)
        
        def distance_from_center(loc):
            lat, lon = loc.get('lat', 0), loc.get('lon', 0)
            return ((lat - center_lat) ** 2 + (lon - center_lon) ** 2) ** 0.5
        
        sorted_locations = sorted(locations, key=distance_from_center)
        
        combinations = []
        for _ in range(count):
            # Select stations with maximum spread
            selected = []
            remaining = sorted_locations.copy()
            
            for i in range(num_stations):
                if not remaining:
                    break
                
                if i == 0:
                    # First station: random from all
                    selected.append(remaining.pop(random.randint(0, len(remaining) - 1)))
                else:
                    # Subsequent stations: maximize distance from already selected
                    best_location = None
                    best_distance = -1
                    
                    for loc in remaining:
                        min_distance = min(
                            ((loc.get('lat', 0) - sel.get('lat', 0)) ** 2 + 
                             (loc.get('lon', 0) - sel.get('lon', 0)) ** 2) ** 0.5
                            for sel in selected
                        )
                        if min_distance > best_distance:
                            best_distance = min_distance
                            best_location = loc
                    
                    if best_location:
                        selected.append(best_location)
                        remaining.remove(best_location)
            
            if len(selected) == num_stations:
                combinations.append(tuple(selected))
        
        return combinations
    
    def _generate_random_combinations(self, locations: List[Dict], num_stations: int, count: int) -> List[Tuple]:
        """Generate random combinations for exploration."""
        import random
        
        combinations = []
        for _ in range(count):
            if len(locations) >= num_stations:
                combo = random.sample(locations, num_stations)
                combinations.append(tuple(combo))
        
        return combinations
    
    def _generate_heuristic_combinations(self, locations: List[Dict], num_stations: int, count: int) -> List[Tuple]:
        """Generate combinations using network topology heuristics."""
        import random
        
        # Group locations by edge type or network features
        edge_groups = {}
        for loc in locations:
            edge_id = loc.get('edge_id', '') if isinstance(loc, dict) else getattr(loc, 'edge_id', '')
            # Group by edge characteristics (simplified)
            group_key = edge_id.split('_')[0] if '_' in edge_id else 'default'
            edge_groups.setdefault(group_key, []).append(loc)
        
        combinations = []
        for _ in range(count):
            selected = []
            remaining_groups = list(edge_groups.keys())
            
            # Try to select from different edge groups
            for i in range(num_stations):
                if not remaining_groups:
                    # Fallback to random selection
                    if len(locations) >= num_stations - len(selected):
                        additional = random.sample(
                            [loc for loc in locations if loc not in selected],
                            num_stations - len(selected)
                        )
                        selected.extend(additional)
                    break
                
                # Select from a random group
                group_key = random.choice(remaining_groups)
                group_locations = edge_groups[group_key]
                
                if group_locations:
                    loc = random.choice(group_locations)
                    selected.append(loc)
                    # Remove the selected location from the group to prevent duplicates
                    group_locations.remove(loc)
                    # Remove group if empty
                    if not group_locations:
                        remaining_groups.remove(group_key)
            
            if len(selected) == num_stations:
                combinations.append(tuple(selected))
        
        return combinations
    
    def _generate_routes_for_episode(self, 
                                   data_source: Any, 
                                   episode: int, 
                                   ved_trajectories: Optional[pd.DataFrame] = None) -> Tuple[Optional[str], int]:
        """Generate routes for a specific episode with improved error handling and reproducibility."""
        try:
            # Set episode-specific seed for reproducibility
            episode_seed = self.base_seed + episode
            import random
            random.seed(episode_seed)
            np.random.seed(episode_seed)
            
            if data_source == 'random':
                return self.route_generator.generate_random_trips(episode)
            elif ved_trajectories is not None and ((isinstance(ved_trajectories, pd.DataFrame) and not ved_trajectories.empty) or (isinstance(ved_trajectories, dict) and len(ved_trajectories) > 0)):
                # Apply domain randomization if enabled
                if self.enable_enhancements and hasattr(self, 'domain_randomizer'):
                    try:
                        # Convert DataFrame to records format
                        train_records = ved_trajectories.to_dict('records')
                        randomized_routes, _ = apply_domain_randomization(
                            train_records, self.network.net_file_path, episode
                        )
                        randomized_df = pd.DataFrame(randomized_routes)
                        
                        # Convert to expected format
                        randomized_dict = {}
                        for vehicle_id, group in randomized_df.groupby('VehId'):
                            group_clean = group[['lat', 'lon', 'timestamp']].copy().dropna()
                            if len(group_clean) >= 2:
                                randomized_dict[str(vehicle_id)] = group_clean.reset_index(drop=True)
                        
                        if randomized_dict:
                            return self.route_generator.generate_routes_from_data(randomized_dict, episode_id=episode)
                    except Exception as e:
                        self.logger.warning(f"Domain randomization failed: {e}")
                        # Continue with fallback
                
                # Convert DataFrame to dict format for route generation
                trajectories_dict = {}
                try:
                    for vehicle_id, group in ved_trajectories.groupby('VehId'):
                        # Ensure required columns exist
                        required_cols = ['lat', 'lon', 'timestamp']
                        if all(col in group.columns for col in required_cols):
                            group_clean = group[required_cols].copy().dropna()
                            if len(group_clean) >= 2:
                                # Sort by timestamp to ensure proper order
                                group_clean = group_clean.sort_values('timestamp').reset_index(drop=True)
                                trajectories_dict[str(vehicle_id)] = group_clean
                        else:
                            self.logger.warning(f"Missing required columns for vehicle {vehicle_id}")
                    
                    if trajectories_dict:
                        return self.route_generator.generate_routes_from_data(trajectories_dict, episode_id=episode)
                    else:
                        self.logger.warning("No valid trajectories found, falling back to random routes")
                        return self.route_generator.generate_random_trips(episode)
                        
                except Exception as e:
                    self.logger.error(f"Error processing trajectories: {e}")
                    return self.route_generator.generate_random_trips(episode)
            else:
                # No valid data, use random routes
                return self.route_generator.generate_random_trips(episode)
                
        except Exception as e:
            self.logger.error(f"Route generation failed: {e}")
            # Final fallback to random routes
            try:
                return self.route_generator.generate_random_trips(episode)
            except Exception as fallback_error:
                self.logger.error(f"Even fallback route generation failed: {fallback_error}")
                return None, 0
    
    def _generate_comprehensive_routes(self, 
                                     episode: int, 
                                     ved_trajectories: Optional[pd.DataFrame] = None,
                                     use_real_data: bool = True,
                                     use_synthetic_routes: bool = True) -> Tuple[Optional[str], int]:
        """
        Generate routes using comprehensive strategy with real data and synthetic routes.
        
        Args:
            episode: Episode identifier
            ved_trajectories: Real-world trajectory data
            use_real_data: Whether to use real trajectory data
            use_synthetic_routes: Whether to add synthetic routes
            
        Returns:
            Tuple of (route_file_path, simulation_duration)
        """
        try:
            # Convert DataFrame to dict format if needed
            trajectories_dict = None
            if ved_trajectories is not None and ((isinstance(ved_trajectories, pd.DataFrame) and not ved_trajectories.empty) or (isinstance(ved_trajectories, dict) and len(ved_trajectories) > 0)):
                trajectories_dict = {}
                for vehicle_id, group in ved_trajectories.groupby('VehId'):
                    required_cols = ['lat', 'lon', 'timestamp']
                    if all(col in group.columns for col in required_cols):
                        group_clean = group[required_cols].copy().dropna()
                        if len(group_clean) >= 2:
                            group_clean = group_clean.sort_values('timestamp').reset_index(drop=True)
                            trajectories_dict[str(vehicle_id)] = group_clean
            
            # Use comprehensive route generation
            return self.route_generator.generate_episode_routes(
                episode_id=episode,
                trajectories=trajectories_dict,
                use_real_data=use_real_data,
                use_synthetic_routes=use_synthetic_routes
            )
            
        except Exception as e:
            self.logger.error(f"Comprehensive route generation failed: {e}")
            # Fallback to simple random routes
            return self.route_generator._generate_simple_random_routes(episode)
    
    def _execute_simulation(self, 
                          config: Any, 
                          route_file: str, 
                          episode: int) -> Tuple[Optional[float], Optional[List[Dict]]]:
        """Execute simulation with given placement configuration."""
        try:
            # Convert config back to placement
            placement = self._config_to_placement(config)
            if not placement:
                return None, None
            
            # Use direct execution method for better error handling
            return self._execute_simulation_direct(placement, route_file, episode)
            
        except Exception as e:
            self.logger.error(f"Simulation execution failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None, None
    
    def _execute_simulation_direct(self, 
                                 placement: List[Dict], 
                                 route_file: str, 
                                 episode: int) -> Tuple[Optional[float], Optional[List[Dict]]]:
        """Execute simulation directly with placement list (no config conversion needed)."""
        try:
            if not placement:
                return None, None
            
            # Convert placement data to expected format for ChargingStationManager
            converted_placement = []
            for p_info in placement:
                # Get actual lane ID from the network instead of generating fake ones
                edge_id = p_info.get('edge_id', 'unknown')
                lane_id = self._get_actual_lane_id(edge_id)
                
                converted_placement.append({
                    'lane': lane_id,
                    'position': p_info.get('position', 0.0),
                    'end_position': p_info.get('end_position', p_info.get('position', 0.0) + 15.0),  # Default 15m length
                    'lat': p_info.get('lat', 0.0),
                    'lon': p_info.get('lon', 0.0),
                    'edge_id': edge_id
                })
            
            # Create charging stations
            charging_stations, additional_file = self.station_manager.create_additional_file(
                converted_placement, file_path=None, episode_id=episode
            )
            
            if not charging_stations:
                self.logger.error("No valid charging stations created")
                return None, None
            
            # Convert to EV routes
            formatted_route_file = f'{os.path.splitext(route_file)[0]}_formatted.rou.xml'
            self.route_generator.convert_to_ev_routes(route_file, formatted_route_file, charging_stations)
            
            if not os.path.exists(formatted_route_file):
                self.logger.error(f"Formatted route file not created: {formatted_route_file}")
                return None, None
            
            # Run simulation with reasonable timeout (5 minutes max)
            # PARALLEL EXECUTION FIX: Pass grid_id as output_prefix for file isolation
            # Pass episode_id and grid_id for deterministic domain randomization
            self.sim_manager.run_simulation(
                additional_file, formatted_route_file, 
                max_duration=300,
                output_prefix=self.grid_id,  # Use grid_id for file naming
                episode_id=episode,
                grid_id=self.placement_grid_id,  # Use placement_grid_id for domain randomization
                base_seed=self.base_seed
            )
            
            # CRITICAL FIX: Use the actual output prefix from SimulationManager
            # which includes worker_id, timestamp, and episode_id for parallel safety
            actual_prefix = self.sim_manager.last_output_prefix
            self.logger.debug(f"Using actual prefix for file reads: {actual_prefix}")
            
            # Parse results with unique prefixed filenames
            output_dir = getattr(self.sim_manager, 'last_output_dir', 
                                os.path.abspath("logs/simulation_outputs"))
            # CRITICAL FIX: Include output_dir in file paths
            battery_data = self.analyzer.parse_battery_data(os.path.join(output_dir, f"{actual_prefix}_Battery.out.xml"))
            charging_data = self.analyzer.parse_charging_events(os.path.join(output_dir, f"{actual_prefix}_chargingevents.xml"))
            summary_data = self.analyzer.parse_summary(os.path.join(output_dir, f"{actual_prefix}_summary.xml"))
            network_state = self.analyzer.collect_network_statistics(output_dir=output_dir, 
                                                                    output_prefix=actual_prefix)
            
            # Calculate reward
            reward = self.analyzer.calculate_reward(battery_data, charging_data, summary_data, network_state)
            
            # Clean up temporary simulation files to save disk space
            self.sim_manager.cleanup_output_files()
            
            # Create placement data
            placement_data = []
            for i, p_info in enumerate(placement):
                station_id = f'CS_{i}'
                station_utilizations = self.analyzer.get_station_utilization(charging_data, 300)  # 5 min default
                placement_data.append({
                    'station_id': station_id,
                    'edge_id': p_info.get('edge_id', f'edge_{i}'),
                    'lane_id': p_info.get('lane_id', p_info.get('lane', f'edge_{i}_0')),  # Handle both 'lane' and 'lane_id'
                    'position': p_info.get('position', 0.0),
                    'lat': p_info.get('lat', 0.0),
                    'lon': p_info.get('lon', 0.0),
                    'utilization': station_utilizations.get(station_id, 0.0)
                })
            
            return reward, placement_data
            
        except Exception as e:
            self.logger.error(f"Direct simulation execution failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None, None
    
    def _get_actual_lane_id(self, edge_id: str) -> str:
        """Get actual lane ID from the SUMO network for a given edge ID."""
        try:
            if not self.network or not hasattr(self.network, 'get_edges'):
                # Fallback to simple lane ID generation
                return f"{edge_id}_0"
            
            # Get edges from the network
            edges = self.network.get_edges()
            
            # Find the edge with the given ID
            for edge in edges:
                # Handle both dictionary and SUMO Edge object formats
                if isinstance(edge, dict) and edge.get('id') == edge_id:
                    # Dictionary format
                    lanes = edge.get('lanes', [])
                    if lanes:
                        return lanes[0]
                    else:
                        return f"{edge_id}_0"
                elif hasattr(edge, 'getID') and edge.getID() == edge_id:
                    # SUMO Edge object format
                    try:
                        lanes = edge.getLanes()
                        if lanes and len(lanes) > 0:
                            return lanes[0].getID()  # Return first lane ID
                        else:
                            return f"{edge_id}_0"
                    except Exception:
                        return f"{edge_id}_0"
            
            # If edge not found, return fallback
            self.logger.warning(f"Edge {edge_id} not found in network, using fallback lane ID")
            return f"{edge_id}_0"
            
        except Exception as e:
            self.logger.warning(f"Error getting lane ID for edge {edge_id}: {e}")
            return f"{edge_id}_0"
    
    def _config_to_placement(self, config: Any) -> Optional[List[Dict]]:
        """Convert configuration identifier back to placement list."""
        try:
            # config is a tuple of (edge_id, position) pairs
            if not isinstance(config, tuple):
                self.logger.error(f"Invalid config format: {type(config)}")
                return None
            
            placement = []
            for i, (edge_id, position) in enumerate(config):
                # Create placement entry
                placement_entry = {
                    'station_id': f'station_{i}',
                    'edge_id': edge_id,
                    'lane_id': f"{edge_id}_0",
                    'position': position,
                    'lat': 0.0,  # Will be filled by station manager
                    'lon': 0.0   # Will be filled by station manager
                }
                placement.append(placement_entry)
            
            # Get coordinates from network
            if self.network and hasattr(self.network, 'get_edges'):
                for entry in placement:
                    try:
                        # Find the edge and get coordinates
                        edge_id = entry['edge_id']
                        position = entry['position']
                        
                        # This is a simplified approach - in practice you'd use SUMO's API
                        # For now, we'll use the station manager to get coordinates
                        temp_placement = self.station_manager.generate_placement(1)
                        if temp_placement:
                            entry['lat'] = temp_placement[0].get('lat', 0.0)
                            entry['lon'] = temp_placement[0].get('lon', 0.0)
                    except Exception as e:
                        self.logger.debug(f"Error getting coordinates for {entry['edge_id']}: {e}")
                        continue
            
            return placement
            
        except Exception as e:
            self.logger.error(f"Config to placement conversion failed: {e}")
            return None
    
    def _check_convergence(self, 
                          algorithm: BaseBandit, 
                          confidence_threshold: float, 
                          episode: int) -> bool:
        """Check if optimization has converged based on confidence criteria."""
        try:
            # Check if algorithm-specific confidence threshold is met
            if hasattr(algorithm, 'get_confidence_threshold_met'):
                return algorithm.get_confidence_threshold_met(confidence_threshold)
            
            # Fallback: check if we have enough samples and stable performance
            stats = algorithm.get_action_statistics()
            if not stats:
                return False
            
            # Check if best action has been sampled enough times
            best_action = algorithm.get_best_action()
            if best_action is None:
                return False
            
            best_stats = stats.get(best_action, {})
            if best_stats.get('count', 0) < 50:  # Public policy rigor: 50 minimum samples
                return False
            
            # Check if confidence interval is narrow enough
            ci_lower, ci_upper = best_stats.get('confidence_interval', (0, 0))
            ci_width = ci_upper - ci_lower
            avg_reward = best_stats.get('average_reward', 0)
            
            if avg_reward == 0:
                return False
            
            relative_width = ci_width / abs(avg_reward)
            return relative_width < 0.05  # Public policy rigor: 5% relative width threshold
            
        except Exception as e:
            self.logger.error(f"Convergence check failed: {e}")
            return False
    
    def _check_early_stopping(self, 
                            episode_rewards: List[float], 
                            patience: int) -> bool:
        """Check if early stopping criteria are met."""
        if len(episode_rewards) < patience:
            return False
        
        # Check if recent rewards have been consistently low
        recent_rewards = episode_rewards[-patience:]
        if len(recent_rewards) < patience:
            return False
        
        # Check if there's been no improvement in recent episodes
        recent_avg = np.mean(recent_rewards)
        overall_avg = np.mean(episode_rewards)
        
        return recent_avg < 0.8 * overall_avg  # 20% degradation threshold
    
    def _extract_confidence_data(self, algorithm_stats: Dict, selected_action: Any, algorithm: BaseBandit) -> Dict:
        """Extract confidence-related data from algorithm statistics."""
        try:
            self.logger.info(f"Extracting confidence data for action {selected_action}")
            self.logger.info(f"Algorithm stats: {algorithm_stats}")
            
            confidence_data = {
                'confidence_lower': 0.0,
                'confidence_upper': 0.0,
                'confidence_width': 0.0,
                'best_action_confidence': 0.0,
                'action_diversity': len(algorithm_stats),
                'exploration_bonus': 0.0
            }
            
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
            best_action = algorithm.get_best_action()
            if best_action and best_action in algorithm_stats:
                best_stats = algorithm_stats[best_action]
                ci_lower, ci_upper = best_stats.get('confidence_interval', (0.0, 0.0))
                confidence_data['best_action_confidence'] = ci_upper - ci_lower
                self.logger.info(f"Best action confidence: {ci_upper - ci_lower}")
            
            # Get exploration bonus for UCB algorithm
            if hasattr(algorithm, 'get_exploration_bonus'):
                try:
                    exploration_bonus = algorithm.get_exploration_bonus(selected_action)
                    confidence_data['exploration_bonus'] = exploration_bonus
                    self.logger.info(f"Exploration bonus: {exploration_bonus}")
                except Exception as e:
                    self.logger.info(f"Failed to get exploration bonus: {e}")
            
            # CRITICAL FIX: If confidence intervals are still zero, provide meaningful fallback values
            if confidence_data['confidence_width'] == 0.0 and confidence_data['best_action_confidence'] == 0.0:
                # Use exploration bonus as a proxy for confidence uncertainty
                if confidence_data['exploration_bonus'] > 0:
                    # Higher exploration bonus means lower confidence
                    # Convert to percentage-based confidence
                    exploration_factor = min(confidence_data['exploration_bonus'] * 10, 50.0)  # Scale to percentage
                    confidence_data['confidence_lower'] = max(0.0, 50.0 - exploration_factor)
                    confidence_data['confidence_upper'] = min(100.0, 50.0 + exploration_factor)
                    confidence_data['confidence_width'] = 2 * exploration_factor
                    confidence_data['best_action_confidence'] = 2 * exploration_factor
                    self.logger.info(f"Generated fallback confidence interval: {confidence_data['confidence_lower']} - {confidence_data['confidence_upper']}")
                else:
                    # Default to maximum uncertainty
                    confidence_data['confidence_lower'] = 0.0
                    confidence_data['confidence_upper'] = 100.0
                    confidence_data['confidence_width'] = 100.0
                    confidence_data['best_action_confidence'] = 100.0
            
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
    
    def _calculate_optimization_metrics(self, 
                                      algorithm: BaseBandit,
                                      episode_rewards: List[float],
                                      action_rewards: Dict[Any, List[float]],
                                      convergence_episode: Optional[int],
                                      episode_confidence: List[Dict] = None) -> Dict:
        """Calculate comprehensive optimization metrics."""
        metrics = {
            'total_episodes': len(episode_rewards),
            'convergence_episode': convergence_episode,
            'best_reward': self.best_reward,
            'final_reward': episode_rewards[-1] if episode_rewards else 0,
            'average_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'reward_std': np.std(episode_rewards) if episode_rewards else 0,
            'algorithm_stats': algorithm.get_action_statistics(),
            'convergence_achieved': convergence_episode is not None,
            'episode_rewards': episode_rewards,  # Include episode rewards for regret calculation
            'episode_confidence': episode_confidence or []  # Include confidence data
        }
        
        # Calculate regret (if we have a theoretical optimum)
        if episode_rewards:
            max_possible_reward = max(episode_rewards)  # Simplified
            cumulative_reward = sum(episode_rewards)
            cumulative_regret = max_possible_reward * len(episode_rewards) - cumulative_reward
            metrics['cumulative_regret'] = cumulative_regret
            metrics['average_regret'] = cumulative_regret / len(episode_rewards)
        
        # Action diversity metrics
        unique_actions = len(action_rewards)
        metrics['action_diversity'] = unique_actions
        metrics['exploration_efficiency'] = unique_actions / len(episode_rewards) if episode_rewards else 0
        
        return metrics
    
    def compare_with_heuristics(self, 
                              ved_trajectories: pd.DataFrame,
                              test_episodes: int = 10) -> Dict[str, Dict]:
        """
        Compare bandit optimization results with heuristic baselines.
        
        Parameters
        ----------
        ved_trajectories : pd.DataFrame
            VED trajectory data for evaluation
        test_episodes : int
            Number of test episodes for evaluation
            
        Returns
        -------
        Dict[str, Dict]
            Comparison results for each method
        """
        if not self.enable_enhancements or not hasattr(self, 'heuristic_baselines'):
            self.logger.warning("Heuristic comparison not available")
            return {}
        
        self.logger.info("🔍 Comparing with heuristic baselines...")
        
        # Generate heuristic baselines
        heuristic_placements = self.heuristic_baselines.compare_baselines(
            ved_trajectories, self.num_stations, self.grid_cells_data
        )
        
        # Evaluate each method
        results = {}
        
        # Evaluate bandit result
        if self.best_placement:
            bandit_rewards = self._evaluate_placement_multiple_episodes(
                self.best_placement, ved_trajectories, test_episodes
            )
            results['bandit_optimization'] = {
                'placement': self.best_placement,
                'rewards': bandit_rewards,
                'average_reward': np.mean(bandit_rewards),
                'std_reward': np.std(bandit_rewards)
            }
        
        # Evaluate heuristic baselines
        for method_name, placement in heuristic_placements.items():
            if placement:
                heuristic_rewards = self._evaluate_placement_multiple_episodes(
                    placement, ved_trajectories, test_episodes
                )
                results[method_name] = {
                    'placement': placement,
                    'rewards': heuristic_rewards,
                    'average_reward': np.mean(heuristic_rewards),
                    'std_reward': np.std(heuristic_rewards)
                }
        
        # Statistical comparison
        if len(results) > 1:
            self._perform_statistical_comparison(results)
        
        return results
    
    def _evaluate_placement_multiple_episodes(self, 
                                            placement: List[Dict],
                                            ved_trajectories: Any,
                                            num_episodes: int) -> List[float]:
        """Evaluate a placement over multiple episodes."""
        rewards = []
        
        for episode in range(num_episodes):
            try:
                # Generate routes - handle both DataFrame and dict formats
                if isinstance(ved_trajectories, pd.DataFrame) and len(ved_trajectories) > 0:
                    # Convert DataFrame to dict format
                    trajectories_dict = {}
                    for vehicle_id, group in ved_trajectories.groupby('VehId'):
                        group_clean = group[['lat', 'lon', 'timestamp']].copy().dropna()
                        if len(group_clean) >= 2:
                            trajectories_dict[str(vehicle_id)] = group_clean.reset_index(drop=True)
                    route_file, sim_duration = self.route_generator.generate_routes_from_data(
                        trajectories_dict, episode_id=episode
                    )
                elif isinstance(ved_trajectories, dict) and len(ved_trajectories) > 0:
                    # Already in dict format
                    route_file, sim_duration = self.route_generator.generate_routes_from_data(
                        ved_trajectories, episode_id=episode
                    )
                else:
                    self.logger.warning(f"No valid trajectory data for episode {episode}")
                    continue
                
                if route_file is None:
                    continue
                
                # Execute simulation directly with placement (not config)
                reward, _ = self._execute_simulation_direct(placement, route_file, episode)
                if reward is not None:
                    rewards.append(reward)
                    
            except Exception as e:
                self.logger.warning(f"Evaluation episode {episode} failed: {e}")
                continue
        
        return rewards
    
    def _perform_statistical_comparison(self, results: Dict[str, Dict]):
        """Perform statistical comparison between methods."""
        try:
            from scipy import stats
            
            method_names = list(results.keys())
            if len(method_names) < 2:
                return
            
            # Pairwise t-tests
            comparisons = []
            for i in range(len(method_names)):
                for j in range(i + 1, len(method_names)):
                    method1, method2 = method_names[i], method_names[j]
                    rewards1 = results[method1]['rewards']
                    rewards2 = results[method2]['rewards']
                    
                    if len(rewards1) > 1 and len(rewards2) > 1:
                        t_stat, p_value = stats.ttest_ind(rewards1, rewards2)
                        comparisons.append({
                            'method1': method1,
                            'method2': method2,
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        })
            
            self.logger.info("📊 Statistical comparison results:")
            for comp in comparisons:
                significance = "✓" if comp['significant'] else "✗"
                self.logger.info(f"  {comp['method1']} vs {comp['method2']}: "
                               f"t={comp['t_statistic']:.3f}, p={comp['p_value']:.3f} {significance}")
                
        except ImportError:
            self.logger.warning("SciPy not available for statistical comparison")
        except Exception as e:
            self.logger.error(f"Statistical comparison failed: {e}")
    
    def save_optimization_results(self, 
                                filepath: str,
                                algorithm_name: str,
                                metrics: Dict) -> None:
        """Save optimization results to file."""
        results = {
            'algorithm': algorithm_name,
            'timestamp': datetime.now().isoformat(),
            'grid_id': self.placement_grid_id,
            'num_stations': self.num_stations,
            'best_placement': self.best_placement,
            'best_reward': self.best_reward,
            'metrics': metrics,
            'optimization_history': self.optimization_history[-100:]  # Last 100 episodes
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        self.logger.info(f"Results saved to: {filepath}")
