"""
Heuristic Baselines for Comparative Performance Analysis

Implements two methodologically distinct baseline approaches:
1. Demand-driven K-Means clustering approach
2. Network topology-driven centrality approach
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict
import xml.etree.ElementTree as ET
from scipy.spatial.distance import cdist
import sys
from datetime import datetime
try:
    import sumolib
except ImportError:
    print("⚠️ SUMO not available - baseline functionality may be limited")
    sumolib = None


class HeuristicBaselines:
    """
    Implements strong heuristic baselines for benchmarking RL performance.
    """
    
    def _setup_logging(self):
        """Setup simple logging for heuristic baselines."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def __init__(self, network_file: str, base_seed: int = 42):
        """
        Initialize heuristic baselines with SUMO network.
        
        Args:
            network_file: Path to SUMO network file (.net.xml)
            base_seed: Random seed for reproducibility
        """
        # Setup comprehensive logging first
        self._setup_logging()
        
        self.network_file = network_file
        self.base_seed = base_seed
        self.network = None
        self.networkx_graph = None
        
        # Set random seed for reproducibility
        np.random.seed(base_seed)
        import random
        random.seed(base_seed)
        
        self.logger.info(f"Initializing Heuristic Baselines")
        self.logger.info(f"  Network file: {network_file}")
        self.logger.info(f"  Base seed: {base_seed}")
        
        try:
            if sumolib is None:
                self.logger.warning("SUMO not available - using fallback mode")
                self.network = None
                self.networkx_graph = None
            else:
                self.network = sumolib.net.readNet(network_file)
                self.networkx_graph = self._create_networkx_graph()
                self.logger.info(f"Loaded network with {len(self.network.getEdges())} edges")
        except Exception as e:
            self.logger.error(f"Error loading network: {e}")
            self.network = None
            self.networkx_graph = None
    
    def demand_driven_clustering_baseline(self, 
                                        ved_trajectories: pd.DataFrame,
                                        num_chargers: int,
                                        dwell_threshold_hours: float = 2.0,
                                        low_soc_threshold: float = 0.2,
                                        grid_bounds: Dict = None) -> List[Dict]:
        """
        Baseline 1: Demand-driven K-Means clustering approach.
        
        Places chargers at locations with highest historical charging demand
        using sophisticated demand point identification and proper SUMO edge validation.
        
        IMPORTANT: This method uses the SAME trajectory data for both placement optimization
        AND evaluation. This is FAIR because:
        1. RL methods also optimize on the same data they're evaluated on
        2. All methods have equal access to historical demand information
        3. Real-world scenario: you optimize based on available data and test on same network
        
        The comparison is fair - all methods optimize and evaluate on the same trajectories.
        
        Args:
            ved_trajectories: VED trajectory data
            num_chargers: Number of charging stations to place
            dwell_threshold_hours: Minimum dwell time to consider as charging opportunity
            low_soc_threshold: SoC threshold for simulated charging need
            grid_bounds: Grid boundaries to constrain placement
            
        Returns:
            List of charging station placement dictionaries
        """
        self.logger.info("Running Demand-Driven K-Means Clustering Baseline")
        
        # Step 1: Get valid placement locations from ChargingStationManager
        valid_locations = self._get_valid_placement_locations(num_chargers * 3, grid_bounds)
        
        if not valid_locations:
            self.logger.warning("No valid placement locations found, using fallback")
            return self._fallback_uniform_placement(num_chargers, grid_bounds)
        
        # Step 2: Identify comprehensive demand points
        demand_points = self._extract_demand_points(ved_trajectories, 
                                                  dwell_threshold_hours,
                                                  low_soc_threshold)
        
        # Filter demand points by grid boundaries if provided
        if grid_bounds:
            demand_points = self._filter_by_grid_bounds(demand_points, grid_bounds)
            self.logger.info(f"Filtered to {len(demand_points)} demand points within grid bounds")
        
        if len(demand_points) < num_chargers:
            self.logger.warning(f"Only {len(demand_points)} demand points found, "
                              f"fewer than {num_chargers} requested chargers")
            return self._fallback_uniform_placement(num_chargers, grid_bounds)
        
        # Step 3: Apply K-Means clustering to demand points
        coordinates = np.array([[p['lat'], p['lon']] for p in demand_points])
        
        try:
            kmeans = KMeans(n_clusters=num_chargers, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(coordinates)
            centroids = kmeans.cluster_centers_
            
            # Step 4: Map centroids to nearest valid placement locations
            placements = []
            for i, centroid in enumerate(centroids):
                centroid_lat, centroid_lon = centroid
                
                # Find nearest valid location
                nearest_location = self._find_nearest_valid_location(
                    centroid_lat, centroid_lon, valid_locations
                )
                
                if nearest_location:
                    placement = {
                        'station_id': f'kmeans_station_{i}',
                        'lat': nearest_location['lat'],
                        'lon': nearest_location['lon'],
                        'edge_id': nearest_location.get('edge_id', ''),
                        'lane': nearest_location['lane'],
                        'position': nearest_location['position'],
                        'method': 'k_means_clustering',
                        'cluster_size': np.sum(cluster_labels == i),
                        'demand_density': len([p for j, p in enumerate(demand_points) 
                                             if cluster_labels[j] == i]),
                        'grid_constrained': grid_bounds is not None,
                        'original_centroid_lat': centroid_lat,
                        'original_centroid_lon': centroid_lon
                    }
                    placements.append(placement)
                else:
                    # Use random valid location as fallback
                    if valid_locations:
                        fallback_location = valid_locations[i % len(valid_locations)]
                        placement = {
                            'station_id': f'kmeans_station_{i}',
                            'lat': fallback_location['lat'],
                            'lon': fallback_location['lon'],
                            'edge_id': fallback_location.get('edge_id', ''),
                            'lane': fallback_location['lane'],
                            'position': fallback_location['position'],
                            'method': 'k_means_clustering_fallback',
                            'cluster_size': np.sum(cluster_labels == i),
                            'demand_density': len([p for j, p in enumerate(demand_points) 
                                                 if cluster_labels[j] == i]),
                            'grid_constrained': grid_bounds is not None,
                            'original_centroid_lat': centroid_lat,
                            'original_centroid_lon': centroid_lon
                        }
                        placements.append(placement)
                    
            self.logger.info(f"K-Means baseline created {len(placements)} placements")
            return placements
            
        except Exception as e:
            self.logger.error(f"Error in K-Means clustering: {e}")
            return self._fallback_uniform_placement(num_chargers, grid_bounds)
    
    def network_centrality_baseline(self, 
                                  ved_trajectories: pd.DataFrame,
                                  num_chargers: int,
                                  grid_bounds: Dict = None) -> List[Dict]:
        """
        Baseline 2: Network topology-driven centrality approach.
        
        Places chargers on edges with highest flow-weighted betweenness centrality,
        identifying critical traffic corridors.
        
        Args:
            ved_trajectories: VED trajectory data for traffic flow estimation
            num_chargers: Number of charging stations to place
            
        Returns:
            List of charging station placement dictionaries
        """
        self.logger.info("Running Network Centrality Baseline")
        
        # Check if we have network and trajectories
        if self.networkx_graph is None or self.network is None:
            self.logger.warning("Network not available, using fallback placement")
            return self._fallback_uniform_placement(num_chargers)
        
        if len(ved_trajectories) == 0:
            self.logger.warning("No trajectories available, using fallback placement")
            return self._fallback_uniform_placement(num_chargers)
        
        # Step 1: Calculate flow-weighted edge betweenness centrality
        edge_centralities = self._calculate_flow_weighted_centrality(ved_trajectories)
        
        if not edge_centralities:
            self.logger.warning("No centrality scores calculated, using fallback placement")
            return self._fallback_uniform_placement(num_chargers)
        
        # Step 2: Select top-k edges
        sorted_edges = sorted(edge_centralities.items(), 
                            key=lambda x: x[1], reverse=True)
        top_edges = sorted_edges[:num_chargers]
        
        # Step 3: Place chargers on high-centrality edges (filtered by grid bounds)
        placements = []
        for i, (edge_id, centrality_score) in enumerate(top_edges):
            try:
                edge = self.network.getEdge(edge_id)
                
                # Place at geometric midpoint of edge
                shape = edge.getShape()
                mid_point_idx = len(shape) // 2
                lon, lat = self.network.convertXY2LonLat(shape[mid_point_idx][0], 
                                                       shape[mid_point_idx][1])
                
                # Check if placement is within grid bounds
                if grid_bounds:
                    if not (grid_bounds['min_lat'] <= lat <= grid_bounds['max_lat'] and
                           grid_bounds['min_lon'] <= lon <= grid_bounds['max_lon']):
                        self.logger.warning(f"Centrality placement {i}: Outside grid bounds, skipping")
                        continue
                
                placement = {
                    'station_id': f'centrality_station_{i}',
                    'lat': lat,
                    'lon': lon,
                    'edge_id': edge_id,
                    'lane': f"{edge_id}_0",  # Use first lane
                    'position': edge.getLength() / 2,  # Midpoint position
                    'method': 'flow_weighted_centrality',
                    'centrality_score': centrality_score,
                    'edge_length': edge.getLength()
                }
                placements.append(placement)
                
            except Exception as e:
                self.logger.warning(f"Error placing charger on edge {edge_id}: {e}")
                continue
        
        # If we couldn't place enough chargers, fill with fallback
        if len(placements) < num_chargers:
            self.logger.warning(f"Only placed {len(placements)} chargers, filling with fallback")
            fallback_placements = self._fallback_uniform_placement(num_chargers - len(placements))
            for i, fallback in enumerate(fallback_placements):
                fallback['station_id'] = f'centrality_fallback_{i}'
                fallback['method'] = 'centrality_fallback'
            placements.extend(fallback_placements)
        
        self.logger.info(f"Centrality baseline created {len(placements)} placements")
        return placements
    
    def _extract_demand_points(self, 
                             ved_trajectories: pd.DataFrame,
                             dwell_threshold_hours: float,
                             low_soc_threshold: float) -> List[Dict]:
        """
        Extract comprehensive demand points from VED trajectories.
        
        Returns:
            List of demand point dictionaries with location and metadata
        """
        demand_points = []
        
        # Trip destinations
        trip_destinations = ved_trajectories.groupby('VehId').tail(1)
        for _, row in trip_destinations.iterrows():
            demand_points.append({
                'lat': row['lat'],
                'lon': row['lon'],
                'type': 'trip_destination',
                'vehicle_id': row['VehId'],
                'timestamp': row['timestamp']
            })
        
        # Long-dwell locations
        dwell_threshold_seconds = dwell_threshold_hours * 3600
        
        for vehicle_id, group in ved_trajectories.groupby('VehId'):
            group = group.sort_values('timestamp').reset_index(drop=True)
            
            # Find stationary periods
            for i in range(len(group) - 1):
                current = group.iloc[i]
                next_point = group.iloc[i + 1]
                
                # Check if vehicle is stationary (within 50m) for extended period
                distance = self._calculate_distance(current['lat'], current['lon'],
                                                  next_point['lat'], next_point['lon'])
                time_diff = next_point['timestamp'] - current['timestamp']
                
                if distance < 0.05 and time_diff > dwell_threshold_seconds:  # 50m, threshold hours
                    demand_points.append({
                        'lat': current['lat'],
                        'lon': current['lon'],
                        'type': 'long_dwell',
                        'vehicle_id': vehicle_id,
                        'dwell_duration': time_diff,
                        'timestamp': current['timestamp']
                    })
        
        # Simulated low-SoC points (simplified energy model)
        for vehicle_id, group in ved_trajectories.groupby('VehId'):
            group = group.sort_values('timestamp').reset_index(drop=True)
            
            # Simple energy consumption model
            initial_soc = 1.0  # Start with full battery
            energy_per_km = 0.2  # 200 Wh/km typical consumption
            battery_capacity = 35.0  # 35 kWh typical capacity
            
            current_energy = battery_capacity
            
            for i in range(1, len(group)):
                prev_point = group.iloc[i-1]
                current_point = group.iloc[i]
                
                # Calculate distance and energy consumption
                distance = self._calculate_distance(prev_point['lat'], prev_point['lon'],
                                                  current_point['lat'], current_point['lon'])
                energy_consumed = distance * energy_per_km
                current_energy -= energy_consumed
                
                current_soc = current_energy / battery_capacity
                
                # Add demand point if SoC drops below threshold
                if current_soc < low_soc_threshold:
                    demand_points.append({
                        'lat': current_point['lat'],
                        'lon': current_point['lon'],
                        'type': 'low_soc',
                        'vehicle_id': vehicle_id,
                        'soc': current_soc,
                        'timestamp': current_point['timestamp']
                    })
                    
                    # "Recharge" for continued simulation
                    current_energy = battery_capacity * 0.8  # 80% after charging
        
        self.logger.info(f"Extracted {len(demand_points)} demand points: "
                        f"{len([p for p in demand_points if p['type'] == 'trip_destination'])} destinations, "
                        f"{len([p for p in demand_points if p['type'] == 'long_dwell'])} dwell locations, "
                        f"{len([p for p in demand_points if p['type'] == 'low_soc'])} low-SoC points")
        
        return demand_points
    
    def _calculate_flow_weighted_centrality(self, 
                                          ved_trajectories: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate flow-weighted edge betweenness centrality.
        
        Returns:
            Dictionary mapping edge_id to centrality score
        """
        edge_centralities = defaultdict(float)
        
        # Group trajectories by trip (vehicle)
        trip_od_pairs = []
        for vehicle_id, group in ved_trajectories.groupby('VehId'):
            group = group.sort_values('timestamp')
            if len(group) >= 2:
                origin = (group.iloc[0]['lat'], group.iloc[0]['lon'])
                destination = (group.iloc[-1]['lat'], group.iloc[-1]['lon'])
                trip_od_pairs.append((origin, destination))
        
        # Map coordinates to network nodes
        for origin_coords, dest_coords in trip_od_pairs:
            try:
                # Find nearest nodes in network
                origin_edge = self._find_nearest_edge(origin_coords[0], origin_coords[1])
                dest_edge = self._find_nearest_edge(dest_coords[0], dest_coords[1])
                
                if origin_edge and dest_edge:
                    origin_node = origin_edge['from_node']
                    dest_node = dest_edge['to_node']
                    
                    # Calculate shortest path
                    try:
                        path = nx.shortest_path(self.networkx_graph, 
                                              origin_node, dest_node, 
                                              weight='weight')
                        
                        # Increment centrality for edges on path
                        for i in range(len(path) - 1):
                            if self.networkx_graph.has_edge(path[i], path[i+1]):
                                edge_data = self.networkx_graph[path[i]][path[i+1]]
                                edge_id = edge_data.get('id', f"{path[i]}_{path[i+1]}")
                                edge_centralities[edge_id] += 1.0
                                
                    except nx.NetworkXNoPath:
                        continue
                        
            except Exception as e:
                self.logger.debug(f"Error processing trip: {e}")
                continue
        
        return dict(edge_centralities)
    
    def _create_networkx_graph(self) -> nx.DiGraph:
        """Convert SUMO network to NetworkX graph for centrality analysis."""
        G = nx.DiGraph()
        
        try:
            for edge in self.network.getEdges():
                if edge.getID().startswith(':'):  # Skip junction internal edges
                    continue
                    
                from_node = edge.getFromNode().getID()
                to_node = edge.getToNode().getID()
                
                G.add_edge(from_node, to_node, 
                          weight=edge.getLength(),
                          id=edge.getID())
            
            self.logger.info(f"Created NetworkX graph with {G.number_of_nodes()} nodes, "
                           f"{G.number_of_edges()} edges")
            return G
            
        except Exception as e:
            self.logger.error(f"Error creating NetworkX graph: {e}")
            return nx.DiGraph()
    
    def _find_nearest_edge(self, lat: float, lon: float) -> Optional[Dict]:
        """Find the nearest SUMO edge to given coordinates."""
        try:
            if self.network is None or sumolib is None:
                # Return None to trigger fallback mode
                return None
                
            x, y = self.network.convertLonLat2XY(lon, lat)
            edges = self.network.getNeighboringEdges(x, y, r=1000)  # 1km search radius
            
            if edges:
                # Get closest edge
                closest_edge, distance = min(edges, key=lambda e: e[1])
                
                return {
                    'edge_id': closest_edge.getID(),
                    'lane_id': f"{closest_edge.getID()}_0",
                    'position': closest_edge.getLength() / 2,  # Midpoint
                    'from_node': closest_edge.getFromNode().getID(),
                    'to_node': closest_edge.getToNode().getID(),
                    'distance_to_point': distance
                }
            
        except Exception as e:
            self.logger.debug(f"Error finding nearest edge for ({lat}, {lon}): {e}")
        
        return None
    
    def _get_valid_placement_locations(self, num_locations: int, grid_bounds: Dict = None) -> List[Dict]:
        """Get valid placement locations using ChargingStationManager for proper SUMO edge validation."""
        try:
            # Create a temporary ChargingStationManager for this grid
            from .ChargingStationManager import ChargingStationManager
            
            # Determine grid_id from grid_bounds if available
            grid_id = None
            if grid_bounds:
                # REPRODUCIBILITY FIX: Use deterministic hash instead of Python's hash()
                bounds_str = f"{grid_bounds['min_lat']}_{grid_bounds['max_lat']}_{grid_bounds['min_lon']}_{grid_bounds['max_lon']}"
                deterministic_hash = sum(ord(c) for c in bounds_str) % 100000
                grid_id = f"grid_{deterministic_hash}"
            
            # Create grid cells data for ChargingStationManager
            grid_cells_data = []
            if grid_bounds:
                grid_cells_data = [{
                    'grid_id': grid_id,
                    'min_lat': grid_bounds['min_lat'],
                    'max_lat': grid_bounds['max_lat'],
                    'min_lon': grid_bounds['min_lon'],
                    'max_lon': grid_bounds['max_lon']
                }]
            
            # Initialize ChargingStationManager
            station_manager = ChargingStationManager(
                net_file_path=self.network_file,
                placement_grid_id=grid_id,
                grid_cells_data=grid_cells_data
            )
            
            # Generate valid locations
            valid_locations = station_manager.generate_placement(num_locations)
            
            # Convert to our format
            converted_locations = []
            for i, location in enumerate(valid_locations):
                # Extract edge_id from lane_id if possible
                edge_id = location.get('lane', '').split('_')[0] if '_' in location.get('lane', '') else ''
                
                converted_location = {
                    'lat': location['lat'],
                    'lon': location['lon'],
                    'lane': location['lane'],
                    'position': location['position'],
                    'edge_id': edge_id,
                    'end_position': location.get('end_position', location['position'] + 15)
                }
                converted_locations.append(converted_location)
            
            self.logger.info(f"Retrieved {len(converted_locations)} valid placement locations from ChargingStationManager")
            return converted_locations
            
        except Exception as e:
            self.logger.error(f"Error getting valid placement locations: {e}")
            return []
    
    def _find_nearest_valid_location(self, lat: float, lon: float, valid_locations: List[Dict]) -> Optional[Dict]:
        """Find the nearest valid location to given coordinates."""
        if not valid_locations:
            return None
        
        min_distance = float('inf')
        nearest_location = None
        
        for location in valid_locations:
            loc_lat = location.get('lat', 0)
            loc_lon = location.get('lon', 0)
            
            # Calculate Euclidean distance
            distance = np.sqrt((lat - loc_lat)**2 + (lon - loc_lon)**2)
            
            if distance < min_distance:
                min_distance = distance
                nearest_location = location
        
        return nearest_location
    
    def _find_nearest_edge_within_bounds(self, lat: float, lon: float, grid_bounds: Dict = None) -> Optional[Dict]:
        """Find the nearest SUMO edge to given coordinates, respecting grid bounds."""
        try:
            if self.network is None or sumolib is None:
                return None
                
            x, y = self.network.convertLonLat2XY(lon, lat)
            edges = self.network.getNeighboringEdges(x, y, r=1000)  # 1km search radius
            
            if not edges:
                return None
            
            # Filter edges by grid bounds if provided
            valid_edges = []
            for edge, distance in edges:
                if grid_bounds:
                    # Check if edge is within grid bounds
                    edge_shape = edge.getShape()
                    if edge_shape:
                        # Check multiple points along the edge
                        for point in edge_shape[::max(1, len(edge_shape)//3)]:  # Sample 3 points
                            lon, lat = self.network.convertXY2LonLat(point[0], point[1])
                            if (grid_bounds['min_lat'] <= lat <= grid_bounds['max_lat'] and
                                grid_bounds['min_lon'] <= lon <= grid_bounds['max_lon']):
                                valid_edges.append((edge, distance))
                                break
                    else:
                        # If no shape, check edge center
                        lon, lat = self.network.convertXY2LonLat(
                            edge.getFromNode().getCoord()[0], edge.getFromNode().getCoord()[1]
                        )
                        if (grid_bounds['min_lat'] <= lat <= grid_bounds['max_lat'] and
                            grid_bounds['min_lon'] <= lon <= grid_bounds['max_lon']):
                            valid_edges.append((edge, distance))
                else:
                    valid_edges.append((edge, distance))
            
            if not valid_edges:
                self.logger.warning(f"No valid edges found within grid bounds for ({lat}, {lon})")
                return None
            
            # Find closest valid edge
            closest_edge, min_distance = min(valid_edges, key=lambda x: x[1])
            
            # Calculate position along edge
            edge_shape = closest_edge.getShape()
            if edge_shape:
                # Find closest point on edge shape
                min_dist_to_shape = float('inf')
                best_position = 0.0
                best_lat, best_lon = lat, lon
                
                for i in range(len(edge_shape) - 1):
                    p1 = edge_shape[i]
                    p2 = edge_shape[i + 1]
                    
                    # Project point onto line segment
                    lon1, lat1 = self.network.convertXY2LonLat(p1[0], p1[1])
                    lon2, lat2 = self.network.convertXY2LonLat(p2[0], p2[1])
                    
                    # Calculate projection
                    t = max(0, min(1, ((lon - lon1) * (lon2 - lon1) + 
                                       (lat - lat1) * (lat2 - lat1)) / 
                                  ((lon2 - lon1)**2 + (lat2 - lat1)**2 + 1e-10)))
                    
                    proj_lon = lon1 + t * (lon2 - lon1)
                    proj_lat = lat1 + t * (lat2 - lat1)
                    
                    dist = np.sqrt((lon - proj_lon)**2 + (lat - proj_lat)**2)
                    if dist < min_dist_to_shape:
                        min_dist_to_shape = dist
                        best_position = i + t
                        best_lat, best_lon = proj_lat, proj_lon
                
                # Convert position to meters along edge
                edge_length = closest_edge.getLength()
                position_meters = best_position * edge_length / (len(edge_shape) - 1)
            else:
                # Fallback if no shape
                best_lat, best_lon = lat, lon
                position_meters = 0.0
            
            return {
                'edge_id': closest_edge.getID(),
                'lane_id': f"{closest_edge.getID()}_0",
                'position': position_meters,
                'lat': best_lat,
                'lon': best_lon,
                'distance_to_original': min_distance
            }
            
        except Exception as e:
            self.logger.debug(f"Error finding nearest edge within bounds for ({lat}, {lon}): {e}")
        
        return None
    
    def _calculate_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates in kilometers."""
        # Haversine formula
        R = 6371  # Earth's radius in km
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_lat/2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def _fallback_uniform_placement(self, num_chargers: int, grid_bounds: Dict = None) -> List[Dict]:
        """Fallback to uniform placement if other methods fail."""
        self.logger.warning("Using fallback uniform placement")
        
        placements = []
        try:
            # Get all non-junction edges
            valid_edges = [e for e in self.network.getEdges() 
                          if not e.getID().startswith(':')]
            
            # Filter edges by grid bounds if provided
            if grid_bounds:
                filtered_edges = []
                for edge in valid_edges:
                    edge_shape = edge.getShape()
                    if edge_shape:
                        # Check if any point of the edge is within bounds
                        for point in edge_shape[::max(1, len(edge_shape)//3)]:
                            lon, lat = self.network.convertXY2LonLat(point[0], point[1])
                            if (grid_bounds['min_lat'] <= lat <= grid_bounds['max_lat'] and
                                grid_bounds['min_lon'] <= lon <= grid_bounds['max_lon']):
                                filtered_edges.append(edge)
                                break
                valid_edges = filtered_edges
            
            if not valid_edges:
                self.logger.warning("No valid edges found for fallback placement")
                return []
            
            # Select evenly spaced edges
            step = max(1, len(valid_edges) // num_chargers)
            selected_edges = valid_edges[::step][:num_chargers]
            
            for i, edge in enumerate(selected_edges):
                shape = edge.getShape()
                mid_point = shape[len(shape) // 2]
                lon, lat = self.network.convertXY2LonLat(mid_point[0], mid_point[1])
                
                placement = {
                    'station_id': f'fallback_station_{i}',
                    'lat': lat,
                    'lon': lon,
                    'edge_id': edge.getID(),
                    'lane': f"{edge.getID()}_0",
                    'position': edge.getLength() / 2,
                    'method': 'uniform_fallback',
                    'grid_constrained': grid_bounds is not None
                }
                placements.append(placement)
                
        except Exception as e:
            self.logger.error(f"Error in fallback placement: {e}")
        
        return placements
    
    def uniform_random_baseline(self, num_chargers: int) -> List[Dict]:
        """
        Baseline 3: Uniform random placement for comparison.
        
        Places chargers randomly across the network to establish a lower bound.
        
        Args:
            num_chargers: Number of charging stations to place
            
        Returns:
            List of charging station placement dictionaries
        """
        self.logger.info("Running Uniform Random Baseline")
        
        placements = []
        try:
            # Get all valid edges (not junctions, sinks, or sources)
            valid_edges = [e for e in self.network.getEdges() 
                          if not e.getID().startswith(':') and 
                          not e.getID().endswith('-sink') and 
                          not e.getID().endswith('-source') and
                          e.getLength() > 10.0]  # Ensure edge is long enough for charging station
            
            if len(valid_edges) < num_chargers:
                self.logger.warning(f"Not enough edges for {num_chargers} chargers")
                return self._fallback_uniform_placement(num_chargers)
            
            # Randomly select edges
            import random
            selected_edges = random.sample(valid_edges, num_chargers)
            
            for i, edge in enumerate(selected_edges):
                # Random position along edge
                position = random.uniform(0, edge.getLength())
                
                # Get coordinates
                shape = edge.getShape()
                if len(shape) > 1:
                    # Interpolate position along edge
                    total_length = 0
                    for j in range(len(shape) - 1):
                        segment_length = np.linalg.norm(np.array(shape[j+1]) - np.array(shape[j]))
                        if total_length + segment_length >= position:
                            # Interpolate within this segment
                            t = (position - total_length) / segment_length
                            point = np.array(shape[j]) + t * (np.array(shape[j+1]) - np.array(shape[j]))
                            lon, lat = self.network.convertXY2LonLat(point[0], point[1])
                            break
                        total_length += segment_length
                    else:
                        # Use last point if position exceeds edge length
                        lat, lon = self.network.convertXY2LonLat(shape[-1][0], shape[-1][1])
                else:
                    lat, lon = self.network.convertXY2LonLat(shape[0][0], shape[0][1])
                
                placement = {
                    'station_id': f'uniform_random_{i}',
                    'lat': lat,
                    'lon': lon,
                    'edge_id': edge.getID(),
                    'lane': f"{edge.getID()}_0",
                    'position': position,
                    'method': 'uniform_random',
                    'edge_length': edge.getLength()
                }
                placements.append(placement)
                
        except Exception as e:
            self.logger.error(f"Uniform random baseline failed: {e}")
            return self._fallback_uniform_placement(num_chargers)
        
        self.logger.info(f"Uniform random baseline created {len(placements)} placements")
        return placements
    
    def grid_center_baseline(self, num_chargers: int, grid_cells_data: List[Dict]) -> List[Dict]:
        """
        Baseline 4: Grid center placement for comparison.
        
        Places chargers at the center of grid cells for systematic coverage.
        
        Args:
            num_chargers: Number of charging stations to place
            grid_cells_data: Grid cell data for placement
            
        Returns:
            List of charging station placement dictionaries
        """
        self.logger.info("Running Grid Center Baseline")
        
        placements = []
        try:
            if not grid_cells_data or len(grid_cells_data) < num_chargers:
                self.logger.warning("Not enough grid cells for placement")
                return self._fallback_uniform_placement(num_chargers)
            
            # Select grid cells (can be random or systematic)
            import random
            selected_cells = random.sample(grid_cells_data, min(num_chargers, len(grid_cells_data)))
            
            for i, cell in enumerate(selected_cells):
                # Calculate center of grid cell
                center_lat = (cell['min_lat'] + cell['max_lat']) / 2
                center_lon = (cell['min_lon'] + cell['max_lon']) / 2
                
                # Find nearest edge to grid center
                nearest_edge = self._find_nearest_edge(center_lat, center_lon)
                
                if nearest_edge:
                    placement = {
                        'station_id': f'grid_center_{i}',
                        'lat': center_lat,
                        'lon': center_lon,
                        'edge_id': nearest_edge['edge_id'],
                        'lane': nearest_edge['lane_id'],
                        'position': nearest_edge['position'],
                        'method': 'grid_center',
                        'grid_id': cell.get('grid_id', f'grid_{i}'),
                        'distance_to_center': nearest_edge['distance_to_point']
                    }
                else:
                    # Fallback to grid center coordinates
                    placement = {
                        'station_id': f'grid_center_{i}',
                        'lat': center_lat,
                        'lon': center_lon,
                        'edge_id': f'fallback_edge_{i}',
                        'lane': f'fallback_lane_{i}',
                        'position': 0.0,
                        'method': 'grid_center_fallback',
                        'grid_id': cell.get('grid_id', f'grid_{i}')
                    }
                
                placements.append(placement)
                
        except Exception as e:
            self.logger.error(f"Grid center baseline failed: {e}")
            return self._fallback_uniform_placement(num_chargers)
        
        self.logger.info(f"Grid center baseline created {len(placements)} placements")
        return placements

    def compare_baselines(self, 
                         ved_trajectories: pd.DataFrame,
                         num_chargers: int,
                         grid_cells_data: Optional[List[Dict]] = None) -> Dict[str, List[Dict]]:
        """
        Generate all baseline placements for comparison.
        
        Returns:
            Dictionary with baseline names as keys and placements as values
        """
        baselines = {}
        
        # K-Means clustering baseline
        try:
            baselines['k_means_clustering'] = self.demand_driven_clustering_baseline(
                ved_trajectories, num_chargers)
        except Exception as e:
            self.logger.error(f"K-Means baseline failed: {e}")
            baselines['k_means_clustering'] = []
        
        # Network centrality baseline
        try:
            baselines['flow_weighted_centrality'] = self.network_centrality_baseline(
                ved_trajectories, num_chargers)
        except Exception as e:
            self.logger.error(f"Centrality baseline failed: {e}")
            baselines['flow_weighted_centrality'] = []
        
        # Uniform random baseline
        try:
            baselines['uniform_random'] = self.uniform_random_baseline(num_chargers)
        except Exception as e:
            self.logger.error(f"Uniform random baseline failed: {e}")
            baselines['uniform_random'] = []
        
        # Grid center baseline
        if grid_cells_data:
            try:
                baselines['grid_center'] = self.grid_center_baseline(num_chargers, grid_cells_data)
            except Exception as e:
                self.logger.error(f"Grid center baseline failed: {e}")
                baselines['grid_center'] = []
        
        return baselines
    
    def random_placement_baseline(self, 
                                ved_trajectories: pd.DataFrame,
                                num_chargers: int,
                                grid_bounds: Dict = None) -> List[Dict]:
        """
        Baseline: Random placement for comparison using valid SUMO edges.
        
        Args:
            ved_trajectories: VED trajectory data for coordinate bounds
            num_chargers: Number of charging stations to place
            grid_bounds: Grid boundaries to constrain placement
            
        Returns:
            List of charging station placement dictionaries
        """
        self.logger.info("Running Random Placement Baseline")
        
        # Get valid placement locations from ChargingStationManager
        valid_locations = self._get_valid_placement_locations(num_chargers * 2, grid_bounds)
        
        if not valid_locations:
            self.logger.warning("No valid placement locations found, using fallback")
            return self._fallback_uniform_placement(num_chargers, grid_bounds)
        
        try:
            # Randomly select from valid locations
            # Use base seed for reproducibility
            import random
            random.seed(self.base_seed)
            selected_locations = random.sample(valid_locations, min(num_chargers, len(valid_locations)))
            
            placements = []
            for i, location in enumerate(selected_locations):
                placement = {
                    'station_id': f'random_station_{i}',
                    'lat': location['lat'],
                    'lon': location['lon'],
                    'edge_id': location.get('edge_id', ''),
                    'lane': location['lane'],
                    'position': location['position'],
                    'method': 'random_placement',
                    'grid_constrained': grid_bounds is not None
                }
                placements.append(placement)
                
        except Exception as e:
            self.logger.error(f"Error in random placement: {e}")
            return self._fallback_uniform_placement(num_chargers, grid_bounds)
        
        self.logger.info(f"Random baseline created {len(placements)} placements")
        return placements
    
    def uniform_spacing_baseline(self, 
                               ved_trajectories: pd.DataFrame,
                               num_chargers: int,
                               grid_bounds: Dict = None) -> List[Dict]:
        """
        Baseline: Uniformly spaced grid placement using valid SUMO edges.
        
        Args:
            ved_trajectories: VED trajectory data for coordinate bounds
            num_chargers: Number of charging stations to place
            grid_bounds: Grid boundaries to constrain placement
            
        Returns:
            List of charging station placement dictionaries
        """
        self.logger.info("Running Uniform Spacing Baseline")
        
        # Get valid placement locations from ChargingStationManager
        valid_locations = self._get_valid_placement_locations(num_chargers * 2, grid_bounds)
        
        if not valid_locations:
            self.logger.warning("No valid placement locations found, using fallback")
            return self._fallback_uniform_placement(num_chargers, grid_bounds)
        
        try:
            # Create uniform grid of target coordinates
            if grid_bounds:
                min_lat = grid_bounds['min_lat']
                max_lat = grid_bounds['max_lat']
                min_lon = grid_bounds['min_lon']
                max_lon = grid_bounds['max_lon']
            elif len(ved_trajectories) > 0:
                min_lat = ved_trajectories['lat'].min()
                max_lat = ved_trajectories['lat'].max()
                min_lon = ved_trajectories['lon'].min()
                max_lon = ved_trajectories['lon'].max()
            else:
                # Default Ann Arbor bounds
                min_lat, max_lat = 42.25, 42.32
                min_lon, max_lon = -83.8, -83.7
            
            # Create uniform grid
            grid_size = int(np.ceil(np.sqrt(num_chargers)))
            lat_step = (max_lat - min_lat) / (grid_size + 1)
            lon_step = (max_lon - min_lon) / (grid_size + 1)
            
            target_coordinates = []
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(target_coordinates) >= num_chargers:
                        break
                    lat = min_lat + (i + 1) * lat_step
                    lon = min_lon + (j + 1) * lon_step
                    target_coordinates.append((lat, lon))
            
            # Map each target coordinate to nearest valid location
            placements = []
            for i, (target_lat, target_lon) in enumerate(target_coordinates):
                nearest_location = self._find_nearest_valid_location(
                    target_lat, target_lon, valid_locations
                )
                
                if nearest_location:
                    placement = {
                        'station_id': f'uniform_station_{i}',
                        'lat': nearest_location['lat'],
                        'lon': nearest_location['lon'],
                        'edge_id': nearest_location.get('edge_id', ''),
                        'lane': nearest_location['lane'],
                        'position': nearest_location['position'],
                        'method': 'uniform_spacing',
                        'grid_constrained': grid_bounds is not None,
                        'target_lat': target_lat,
                        'target_lon': target_lon
                    }
                else:
                    # Use random valid location as fallback
                    if valid_locations:
                        fallback_location = valid_locations[i % len(valid_locations)]
                        placement = {
                            'station_id': f'uniform_station_{i}',
                            'lat': fallback_location['lat'],
                            'lon': fallback_location['lon'],
                            'edge_id': fallback_location.get('edge_id', ''),
                            'lane': fallback_location['lane'],
                            'position': fallback_location['position'],
                            'method': 'uniform_spacing_fallback',
                            'grid_constrained': grid_bounds is not None,
                            'target_lat': target_lat,
                            'target_lon': target_lon
                        }
                    else:
                        continue
                
                placements.append(placement)
                
        except Exception as e:
            self.logger.error(f"Error in uniform spacing placement: {e}")
            return self._fallback_uniform_placement(num_chargers, grid_bounds)
        
        self.logger.info(f"Uniform spacing baseline created {len(placements)} placements")
        return placements
    
    def _filter_by_grid_bounds(self, demand_points: List[Dict], grid_bounds: Dict) -> List[Dict]:
        """
        Filter demand points to only include those within grid boundaries.
        
        Args:
            demand_points: List of demand point dictionaries
            grid_bounds: Dictionary with 'min_lat', 'max_lat', 'min_lon', 'max_lon' keys
            
        Returns:
            Filtered list of demand points
        """
        if not grid_bounds:
            return demand_points
        
        filtered_points = []
        for point in demand_points:
            lat = point.get('lat')
            lon = point.get('lon')
            
            if (lat is not None and lon is not None and
                grid_bounds['min_lat'] <= lat <= grid_bounds['max_lat'] and
                grid_bounds['min_lon'] <= lon <= grid_bounds['max_lon']):
                filtered_points.append(point)
        
        self.logger.info(f"Filtered {len(demand_points)} demand points to {len(filtered_points)} within grid bounds")
        return filtered_points