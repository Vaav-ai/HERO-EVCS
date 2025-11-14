"""
Domain Randomization for Robust RL Training

Implements stochastic variations in driver behavior and traffic demand
to improve generalization of the learned charging station placement policy.

REPRODUCIBILITY: All randomization is deterministic based on (base_seed, grid_id, episode_id)
to ensure reproducible results while maintaining variation across episodes and grids.
"""

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
import logging
import random
import os
from pathlib import Path


def get_deterministic_seed(base_seed: int, grid_id: str = None, episode_id: int = None) -> int:
    """
    Generate a deterministic seed from base_seed, grid_id, and episode_id.
    
    This ensures:
    - Same (base_seed, grid_id, episode_id) → Same randomization (reproducible)
    - Different episodes → Different randomization (variation)
    - Different grids → Different randomization (variation)
    
    Args:
        base_seed: Base random seed from configuration
        grid_id: Grid identifier (optional)
        episode_id: Episode number (optional)
        
    Returns:
        Deterministic seed value
    """
    seed = base_seed
    
    # Add grid_id contribution (deterministic hash)
    if grid_id is not None:
        # Use a DETERMINISTIC hash (Python's hash() is non-deterministic across sessions!)
        # Simple deterministic hash: sum of character codes
        grid_hash = sum(ord(c) for c in str(grid_id)) % 100000
        seed += grid_hash
    
    # Add episode_id contribution
    if episode_id is not None:
        seed += episode_id * 10000
    
    # Ensure seed is within valid range
    return seed % (2**31 - 1)


class DomainRandomization:
    """
    Implements domain randomization strategies for robust RL training
    in EV charging station placement.
    """
    
    def __init__(self, randomization_config: Optional[Dict] = None, base_seed: int = 42):
        """
        Initialize domain randomization with configuration.
        
        Args:
            randomization_config: Configuration for randomization parameters
            base_seed: Base random seed for reproducibility
        """
        self.logger = logging.getLogger(__name__)
        self.config = randomization_config or self.get_default_config()
        self.base_seed = base_seed
        
    @staticmethod
    def get_default_config() -> Dict:
        """Get default randomization configuration based on methodological review."""
        return {
            # Driver behavior randomization
            'driver_behavior': {
                'enabled': True,
                'accel_range': (0.8, 4.0),           # m/s² acceleration
                'decel_range': (1.0, 6.0),           # m/s² deceleration  
                'tau_range': (0.8, 2.0),             # desired time gap
                'min_gap_range': (1.5, 3.0),         # minimum gap
                'lc_assertive_range': (0.2, 2.0),    # lane change assertiveness
                'lc_speed_gain_range': (0.5, 2.0),   # lane change speed gain
            },
            
            # Traffic demand randomization
            'traffic_demand': {
                'enabled': True,
                'time_shift_range': (-600, 600),      # ±10 minutes in seconds
                'demand_scale_range': (0.9, 1.1),     # 90% to 110% of original
                'route_perturbation': 0.1,            # 10% edge weight randomization
            },
            
            # Environmental randomization
            'environment': {
                'enabled': True,
                'weather_effects': True,              # Simulate weather impact
                'incident_probability': 0.05,         # 5% chance of random incidents
                'construction_probability': 0.02,     # 2% chance of construction
            }
        }
    
    def create_randomized_vehicle_types(self, output_file: str, 
                                      num_types: int = 10, 
                                      episode_id: int = None, 
                                      grid_id: str = None) -> str:
        """
        Create SUMO vehicle type distribution with randomized parameters.
        
        Args:
            output_file: Path to save the vehicle type distribution
            num_types: Number of different vehicle types to create
            episode_id: Episode number for deterministic seeding
            grid_id: Grid identifier for deterministic seeding
            
        Returns:
            Path to the created vehicle type file
        """
        if not self.config['driver_behavior']['enabled']:
            return None
        
        # Set deterministic seed for reproducibility
        seed = get_deterministic_seed(self.base_seed, grid_id, episode_id)
        np.random.seed(seed)
        random.seed(seed)
        
        self.logger.debug(f"Creating vehicle types with seed {seed} (base={self.base_seed}, grid={grid_id}, ep={episode_id})")
        
        # Create XML structure for vehicle types
        root = ET.Element("vTypeDistribution", id="mixed_traffic")
        
        for i in range(num_types):
            # Sample parameters from configured ranges (now deterministic)
            accel = np.random.uniform(*self.config['driver_behavior']['accel_range'])
            decel = np.random.uniform(*self.config['driver_behavior']['decel_range'])
            tau = np.random.uniform(*self.config['driver_behavior']['tau_range'])
            min_gap = np.random.uniform(*self.config['driver_behavior']['min_gap_range'])
            lc_assertive = np.random.uniform(*self.config['driver_behavior']['lc_assertive_range'])
            lc_speed_gain = np.random.uniform(*self.config['driver_behavior']['lc_speed_gain_range'])
            
            # Create vehicle type element
            vtype = ET.SubElement(root, "vType")
            vtype.set("id", f"randomType_{i}")
            vtype.set("probability", str(1.0 / num_types))
            
            # Car-following model parameters
            vtype.set("accel", f"{accel:.2f}")
            vtype.set("decel", f"{decel:.2f}")
            vtype.set("tau", f"{tau:.2f}")
            vtype.set("minGap", f"{min_gap:.2f}")
            
            # Lane-changing model parameters
            vtype.set("lcAssertive", f"{lc_assertive:.2f}")
            vtype.set("lcSpeedGain", f"{lc_speed_gain:.2f}")
            
            # EV-specific parameters
            vtype.set("emissionClass", "Energy/unknown")
            vtype.set("maximumBatteryCapacity", "35000")  # 35 kWh typical EV
            vtype.set("actualBatteryCapacity", "35000")
            
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
        
        self.logger.info(f"Created randomized vehicle types: {output_file}")
        return output_file
    
    def randomize_traffic_demand(self, original_routes: List[Dict], 
                                 episode_id: int = None, 
                                 grid_id: str = None) -> List[Dict]:
        """
        Apply randomization to traffic demand patterns.
        
        Args:
            original_routes: List of original route dictionaries
            episode_id: Episode number for deterministic seeding
            grid_id: Grid identifier for deterministic seeding
            
        Returns:
            List of randomized route dictionaries
        """
        if not self.config['traffic_demand']['enabled']:
            return original_routes
        
        # Set deterministic seed for reproducibility
        seed = get_deterministic_seed(self.base_seed, grid_id, episode_id)
        np.random.seed(seed)
        random.seed(seed)
        
        self.logger.debug(f"Randomizing traffic with seed {seed} (base={self.base_seed}, grid={grid_id}, ep={episode_id})")
        
        randomized_routes = []
        
        # Apply demand scaling (now deterministic)
        scale_factor = np.random.uniform(*self.config['traffic_demand']['demand_scale_range'])
        num_routes = max(1, int(len(original_routes) * scale_factor))
        
        # Sample routes (with replacement if scaling up)
        if scale_factor >= 1.0:
            # Scale up: duplicate some routes
            base_routes = original_routes[:]
            additional_routes = random.choices(original_routes, 
                                             k=num_routes - len(original_routes))
            selected_routes = base_routes + additional_routes
        else:
            # Scale down: randomly sample subset
            selected_routes = random.sample(original_routes, num_routes)
        
        # Apply time shifting
        time_shift_range = self.config['traffic_demand']['time_shift_range']
        
        for route in selected_routes:
            randomized_route = route.copy()
            
            # Apply random time shift to timestamp field (for VED data)
            time_shift = np.random.uniform(*time_shift_range)
            if 'timestamp' in randomized_route:
                original_timestamp = float(randomized_route['timestamp'])
                randomized_route['timestamp'] = max(0, original_timestamp + time_shift)
            elif 'depart' in randomized_route:
                original_depart = float(randomized_route['depart'])
                randomized_route['depart'] = str(max(0, original_depart + time_shift))
            
            randomized_routes.append(randomized_route)
        
        self.logger.info(f"Randomized traffic demand: {len(original_routes)} -> {len(randomized_routes)} routes (seed={seed})")
        return randomized_routes
    
    def create_randomized_edge_weights(self, network_file: str, 
                                     output_file: str,
                                     episode_id: int = None,
                                     grid_id: str = None) -> str:
        """
        Create randomized edge weights for route perturbation.
        
        Args:
            network_file: Path to SUMO network file
            output_file: Path to save randomized weights
            episode_id: Episode number for deterministic seeding
            grid_id: Grid identifier for deterministic seeding
            
        Returns:
            Path to the created weights file
        """
        if not self.config['traffic_demand']['enabled']:
            return None
        
        # Set deterministic seed for reproducibility
        seed = get_deterministic_seed(self.base_seed, grid_id, episode_id)
        np.random.seed(seed)
        random.seed(seed)
        
        self.logger.debug(f"Creating edge weights with seed {seed} (base={self.base_seed}, grid={grid_id}, ep={episode_id})")
        
        try:
            import sumolib
            self.logger.info(f"Reading network file: {network_file}")
            net = sumolib.net.readNet(network_file)
            self.logger.info(f"Successfully loaded network with {len(net.getEdges())} edges")
            
            # Create weights XML
            root = ET.Element("weights")
            
            perturbation_factor = self.config['traffic_demand']['route_perturbation']
            
            for edge in net.getEdges():
                if not edge.getID().startswith(':'):  # Skip junction internal edges
                    # Random perturbation around 1.0 (now deterministic)
                    weight = np.random.uniform(1.0 - perturbation_factor, 
                                             1.0 + perturbation_factor)
                    
                    edge_elem = ET.SubElement(root, "edge")
                    edge_elem.set("id", edge.getID())
                    edge_elem.set("weight", f"{weight:.4f}")
            
            # Write to file
            tree = ET.ElementTree(root)
            tree.write(output_file, encoding='utf-8', xml_declaration=True)
            
            self.logger.info(f"Created randomized edge weights: {output_file}")
            return output_file
            
        except ImportError:
            self.logger.warning("sumolib not available for edge weight randomization")
            return None
        except Exception as e:
            self.logger.error(f"Error creating randomized edge weights: {e}")
            self.logger.error(f"Network file: {network_file}")
            self.logger.error(f"File exists: {os.path.exists(network_file) if 'os' in globals() else 'Unknown'}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def simulate_environmental_conditions(self, config_file: str) -> str:
        """
        Add environmental randomization to SUMO configuration.
        
        Args:
            config_file: Path to SUMO configuration file
            
        Returns:
            Modified configuration with environmental effects
        """
        if not self.config['environment']['enabled']:
            return config_file
            
        try:
            tree = ET.parse(config_file)
            root = tree.getroot()
            
            # Add random incidents
            if random.random() < self.config['environment']['incident_probability']:
                self._add_random_incident(root)
            
            # Add construction zones  
            if random.random() < self.config['environment']['construction_probability']:
                self._add_construction_zone(root)
            
            # Weather effects (simplified as speed reduction)
            if self.config['environment']['weather_effects']:
                if random.random() < 0.3:  # 30% chance of adverse weather
                    self._add_weather_effects(root)
            
            # Save modified config
            modified_config = config_file.replace('.sumocfg', '_randomized.sumocfg')
            tree.write(modified_config, encoding='utf-8', xml_declaration=True)
            
            return modified_config
            
        except Exception as e:
            self.logger.error(f"Error adding environmental conditions: {e}")
            return config_file
    
    def _add_random_incident(self, config_root: ET.Element):
        """Add a random traffic incident to the simulation."""
        # Simplified implementation - would need network analysis in practice
        additional_files = config_root.find('.//additional-files')
        if additional_files is not None:
            # Add incident as variable speed sign
            incident_file = "random_incident.add.xml"
            additional_files.set('value', 
                               additional_files.get('value', '') + f',{incident_file}')
    
    def _add_construction_zone(self, config_root: ET.Element):
        """Add a construction zone affecting traffic flow."""
        # Simplified implementation
        self.logger.debug("Adding construction zone simulation")
    
    def _add_weather_effects(self, config_root: ET.Element):
        """Add weather effects (reduced speeds, visibility)."""
        # Simplified implementation - reduce global speed factor
        self.logger.debug("Adding weather effects simulation")
    
    def create_randomized_episode_config(self, base_config: Dict, 
                                       episode_id: int,
                                       grid_id: str = None) -> Dict:
        """
        Create a complete randomized configuration for an episode.
        
        Args:
            base_config: Base simulation configuration
            episode_id: Unique episode identifier
            grid_id: Grid identifier for deterministic seeding
            
        Returns:
            Randomized configuration for the episode
        """
        # Set deterministic seed for reproducibility
        seed = get_deterministic_seed(self.base_seed, grid_id, episode_id)
        np.random.seed(seed)
        random.seed(seed)
        
        self.logger.info(f"Creating episode config with seed {seed} (base={self.base_seed}, grid={grid_id}, ep={episode_id})")
        
        randomized_config = base_config.copy()
        
        # Create episode-specific output directory
        grid_prefix = f"grid_{grid_id}_" if grid_id else ""
        episode_dir = f"{grid_prefix}episode_{str(episode_id)}_randomized"
        os.makedirs(episode_dir, exist_ok=True)
        
        # Create randomized vehicle types
        if self.config['driver_behavior']['enabled']:
            vtypes_file = os.path.join(episode_dir, "vehicle_types.xml")
            randomized_config['vehicle_types_file'] = self.create_randomized_vehicle_types(
                vtypes_file, episode_id=episode_id, grid_id=grid_id
            )
        
        # Create randomized edge weights
        if self.config['traffic_demand']['enabled'] and 'network_file' in base_config:
            weights_file = os.path.join(episode_dir, "edge_weights.xml")
            randomized_config['edge_weights_file'] = self.create_randomized_edge_weights(
                base_config['network_file'], weights_file, episode_id=episode_id, grid_id=grid_id
            )
        
        # Store randomization metadata
        randomized_config['randomization_seed'] = seed
        randomized_config['episode_dir'] = episode_dir
        randomized_config['episode_id'] = episode_id
        randomized_config['grid_id'] = grid_id
        
        return randomized_config
    
    def get_randomization_summary(self) -> Dict:
        """
        Get summary of current randomization settings.
        
        Returns:
            Dictionary summarizing randomization configuration
        """
        summary = {
            'driver_behavior_enabled': self.config['driver_behavior']['enabled'],
            'traffic_demand_enabled': self.config['traffic_demand']['enabled'],
            'environment_enabled': self.config['environment']['enabled'],
        }
        
        if self.config['driver_behavior']['enabled']:
            summary['driver_parameters'] = {
                'accel_range': self.config['driver_behavior']['accel_range'],
                'decel_range': self.config['driver_behavior']['decel_range'],
                'tau_range': self.config['driver_behavior']['tau_range'],
            }
        
        if self.config['traffic_demand']['enabled']:
            summary['demand_parameters'] = {
                'time_shift_range': self.config['traffic_demand']['time_shift_range'],
                'scale_range': self.config['traffic_demand']['demand_scale_range'],
                'route_perturbation': self.config['traffic_demand']['route_perturbation'],
            }
        
        return summary


# Utility functions for integration with RL training

def apply_domain_randomization(base_routes: List[Dict], 
                             network_file: str,
                             episode_id: int,
                             config: Optional[Dict] = None,
                             base_seed: int = 42,
                             grid_id: str = None) -> Tuple[List[Dict], Dict]:
    """
    Apply complete domain randomization for an RL training episode.
    
    Args:
        base_routes: Original route data
        network_file: SUMO network file path
        episode_id: Episode identifier for reproducible randomization
        config: Randomization configuration
        base_seed: Base random seed for reproducibility
        grid_id: Grid identifier for deterministic seeding
        
    Returns:
        Tuple of (randomized_routes, randomization_metadata)
    """
    randomizer = DomainRandomization(config, base_seed=base_seed)
    
    # Create episode configuration
    base_config = {
        'network_file': network_file,
        'routes': base_routes
    }
    
    episode_config = randomizer.create_randomized_episode_config(
        base_config, episode_id, grid_id=grid_id
    )
    
    # Apply traffic demand randomization
    randomized_routes = randomizer.randomize_traffic_demand(
        base_routes, episode_id=episode_id, grid_id=grid_id
    )
    
    # Calculate deterministic seed for metadata
    deterministic_seed = get_deterministic_seed(base_seed, grid_id, episode_id)
    
    metadata = {
        'episode_id': episode_id,
        'grid_id': grid_id,
        'base_seed': base_seed,
        'deterministic_seed': deterministic_seed,
        'original_routes': len(base_routes),
        'randomized_routes': len(randomized_routes),
        'config': episode_config,
        'summary': randomizer.get_randomization_summary()
    }
    
    return randomized_routes, metadata
