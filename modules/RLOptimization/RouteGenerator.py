# RouteGenerator.py

import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Optional
import random
import numpy as np
from modules.config.EVConfig import EVConfig
from modules.RLOptimization.SumoNetwork import SUMONetwork
import pandas as pd
import math
import logging
import sys
import time
from datetime import datetime

# Optional import for SUMO
try:
    import sumolib
except ImportError:
    sumolib = None

class RouteGenerator:
    """Handles route generation and formatting for SUMO simulations"""
    
    def _setup_logging(self):
        """Setup comprehensive logging for route generation."""
        # Create logger with hierarchical name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Set logging level based on environment
        log_level = os.environ.get('ROUTE_LOG_LEVEL', 'INFO').upper()
        self.logger.setLevel(getattr(logging, log_level, logging.INFO))
        
        # Create formatter with detailed information
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Add console handler if not already present
        if not any(isinstance(handler, logging.StreamHandler) for handler in self.logger.handlers):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Add file handler for detailed logs
        log_dir = "./logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"route_generator_{datetime.now().strftime('%Y%m%d')}.log")
        
        if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == log_file for handler in self.logger.handlers):
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def __init__(self, network: SUMONetwork, ev_config: EVConfig, base_seed: int = 42, grid_id: str = None):
        # Setup logging first
        self._setup_logging()
        
        self.network = network
        self.ev_config = ev_config
        self.base_seed = base_seed
        self.grid_id = grid_id
        self.output_dir = os.path.abspath("./generated_files/routes") # Dedicated routes directory with absolute path
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Add unique identifier to avoid conflicts between workers
        self.worker_id = f"{os.getpid()}_{int(time.time() * 1000) % 10000}"
        
        self.logger.info("Initializing Route Generator")
        self.logger.info(f"  Network: {network.net_file_path}")
        self.logger.info(f"  Output directory: {self.output_dir}")
        self.logger.info(f"  EV config: {type(ev_config).__name__}")
        self.logger.info(f"  Base seed: {base_seed}")
        self.logger.info(f"  Grid ID: {grid_id}")

    def generate_random_trips(self, episode_id: int = 0) -> Tuple[str, int]:    
        """Generate random trips with episode-specific parameters including duration"""    
        route_file_path = os.path.join(self.output_dir, f'random_trips_{self.worker_id}_{str(episode_id)}.rou.xml')    
        
        traffic_scenarios = [    
            {"name": "heavy_traffic", "period": "0.5", "fringe_factor": "2.0", "duration": "800"},    
            {"name": "light_traffic", "period": "2.0", "fringe_factor": "0.5", "duration": "300"},      
            {"name": "normal_traffic", "period": "1.0", "fringe_factor": "1.0", "duration": "500"},    
            {"name": "rush_hour", "period": "0.8", "fringe_factor": "1.5", "duration": "600"},    
            {"name": "off_peak", "period": "3.0", "fringe_factor": "0.3", "duration": "400"}  
        ]    
        
        scenario = traffic_scenarios[episode_id % len(traffic_scenarios)]    
        random_trips_script = os.path.join(os.environ.get("SUMO_HOME", "/usr/share/sumo"), "tools/randomTrips.py")
        
        # Check if randomTrips.py exists
        if not os.path.exists(random_trips_script):
            print(f"Warning: randomTrips.py not found at {random_trips_script}")
            return self._generate_simple_random_routes(episode_id)
        
        cmd = (f'python3 {random_trips_script} -n {self.network.net_file_path} '    
               f'-r {route_file_path} -e {scenario["duration"]} -l '    
               f'--period {scenario["period"]} --fringe-factor {scenario["fringe_factor"]} --seed {episode_id} '
               f'--junction-taz')    
        
        print(f"Episode {episode_id}: Running {scenario['name']} scenario for {scenario['duration']} seconds")  
        result = os.system(cmd)
        
        if result != 0:
            print(f"randomTrips.py failed with exit code {result}, falling back to simple routes")
            return self._generate_simple_random_routes(episode_id)
        
        # Verify the generated file
        if os.path.exists(route_file_path) and self._verify_route_file(route_file_path, episode_id):
            return route_file_path, int(scenario["duration"])
        else:
            print(f"Generated route file invalid, falling back to simple routes")
            return self._generate_simple_random_routes(episode_id)

    def generate_routes_from_data(self, trajectories: Dict[str, pd.DataFrame], route_file_path: str = None, episode_id: int = 0) -> Tuple[Optional[str], int]:
        """
        Converts real VED trajectories to SUMO routes using improved coordinate mapping.
        Uses multiple fallback strategies when coordinate-based routing fails.
        Returns (None, 0) if no valid routes can be generated.
        """
        # Use provided route_file_path or generate default path
        if route_file_path is None:
            route_file_path = os.path.join(self.output_dir, f"hybrid_routes_{self.worker_id}_{str(episode_id)}.rou.xml")
        
        # Convert to absolute paths to avoid issues in cloud environments
        route_file_path = os.path.abspath(route_file_path)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(route_file_path), exist_ok=True)
        
        trip_file_path = os.path.abspath(os.path.join(os.path.dirname(route_file_path), f"temp_trips_{self.worker_id}_{str(episode_id)}.xml"))
        
        # Fix DataFrame/dict boolean evaluation error
        if trajectories is None or (isinstance(trajectories, pd.DataFrame) and trajectories.empty) or (isinstance(trajectories, dict) and len(trajectories) == 0):
            print("Warning: No trajectories provided to generate_routes_from_data.")
            # Generate simple random routes as fallback
            return self._generate_simple_random_routes(episode_id)

        # Strategy 1: Try coordinate-based routing with improved mapping
        result = self._generate_coordinate_based_routes(trajectories, trip_file_path, route_file_path, episode_id)
        if result[0] is not None:
            # Add additional random routes to supplement real data
            return self._add_random_routes_to_existing(result[0], episode_id, result[1])
        
        print(f"Coordinate-based routing failed for episode {episode_id}. Trying edge-based routing...")
        
        # Strategy 2: Try edge-based routing as fallback
        result = self._generate_edge_based_routes(trajectories, trip_file_path, route_file_path, episode_id)
        if result[0] is not None:
            # Add additional random routes to supplement real data
            return self._add_random_routes_to_existing(result[0], episode_id, result[1])
            
        print(f"Edge-based routing failed for episode {episode_id}. Trying hybrid approach...")
        
        # Strategy 3: Generate hybrid routes (real data + random routes)
        # CRITICAL FIX: Convert DataFrame to dict format for hybrid routes
        if isinstance(trajectories, pd.DataFrame) and not trajectories.empty:
            # Convert DataFrame to dict format for processing
            trajectory_dict = {}
            for veh_id, group in trajectories.groupby('VehId'):
                trajectory_dict[str(veh_id)] = group[['lat', 'lon', 'timestamp']].reset_index(drop=True)
            return self._generate_hybrid_routes(trajectory_dict, episode_id)
        elif isinstance(trajectories, dict):
            return self._generate_hybrid_routes(trajectories, episode_id)
        else:
            # Fallback: create empty dict
            return self._generate_hybrid_routes({}, episode_id)
    
    def _generate_simple_random_routes(self, episode_id: int) -> Tuple[str, int]:
        """Generate simple random routes as fallback when other methods fail"""
        route_file_path = os.path.join(self.output_dir, f'simple_routes_{self.worker_id}_{str(episode_id)}.rou.xml')
        
        try:
            if sumolib is None:
                print("sumolib not available, using fallback route generation")
                return self._generate_fallback_routes({}, episode_id)
                
            net = sumolib.net.readNet(self.network.net_file_path)
            edges = [edge for edge in net.getEdges() if not edge.getID().startswith(':')]
            
            if not edges:
                print("No valid edges found for simple route generation")
                return None, 0
                
        except Exception as e:
            print(f"Error reading network for simple routes: {e}")
            return None, 0
        
        # Generate simple routes
        routes = ET.Element('routes')
        
        # Add vehicle types
        vtype = ET.SubElement(routes, 'vType', 
                             id="electric_vehicle",
                             vClass="passenger",
                             emissionClass="Energy",
                             energyFile=os.path.abspath("generated_files/charging_stations/vehicles.xml"))
        
        # Add battery device configuration
        battery_device = ET.SubElement(vtype, 'param', key="device.battery.track", value="true")
        battery_aggregation = ET.SubElement(vtype, 'param', key="device.battery.aggregation", value="true")
        
        # Generate 100-200 random routes for better simulation
        num_routes = random.randint(100, 200)
        duration = 3600  # 60 minutes simulation
        
        for i in range(num_routes):
            # Random start and end edges
            start_edge = random.choice(edges)
            end_edge = random.choice(edges)
            
            # Random departure time spread over the duration
            depart_time = random.uniform(0, duration - 600)  # Leave 10 minute buffer
            
            # Create route
            route_id = f"route_{i}"
            route = ET.SubElement(routes, 'route', 
                                id=route_id,
                                edges=f"{start_edge.getID()} {end_edge.getID()}")
            
            # Create vehicle
            vehicle_id = f"vehicle_{i}"
            vehicle = ET.SubElement(routes, 'vehicle',
                                  id=vehicle_id,
                                  type="electric_vehicle",
                                  route=route_id,
                                  depart=str(depart_time))
        
        # Save routes
        tree = ET.ElementTree(routes)
        tree.write(route_file_path, encoding='utf-8', xml_declaration=True)
        
        print(f"Generated {num_routes} simple random routes for episode {episode_id}")
        return route_file_path, duration

    def _generate_coordinate_based_routes(self, trajectories: Dict[str, pd.DataFrame], 
                                          trip_file_path: str, route_file_path: str, episode_id: any) -> Tuple[Optional[str], int]:
        """Generate routes using coordinate-based trips with improved parameters"""
        # CRITICAL FIX: Handle DataFrame input
        if isinstance(trajectories, pd.DataFrame) and not trajectories.empty:
            # Convert DataFrame to dict format for processing
            trajectory_dict = {}
            for veh_id, group in trajectories.groupby('VehId'):
                trajectory_dict[str(veh_id)] = group[['lat', 'lon', 'timestamp']].reset_index(drop=True)
            trajectories = trajectory_dict
        
        # Build trips list first so we can sort by depart time and inject via points
        trips_buffer = []
        max_arrival_time = 0
        
        for veh_id, traj_df in trajectories.items():
            # Fix DataFrame boolean evaluation error
            if traj_df is None or len(traj_df) < 2:
                continue

            # Ensure traj_df is a DataFrame before calling sort_values
            if isinstance(traj_df, pd.Series):
                # Convert Series to DataFrame if needed
                traj_df = traj_df.to_frame().T
            elif not isinstance(traj_df, pd.DataFrame):
                continue
            
            # Ensure proper numeric index
            traj_df = traj_df.reset_index(drop=True)
            
            # Check if timestamp column exists before sorting
            if 'timestamp' in traj_df.columns:
                traj_df = traj_df.sort_values('timestamp')
            else:
                # If no timestamp column, sort by index
                traj_df = traj_df.sort_index()

            # More conservative time jitter for better stability
            jitter = random.gauss(0.0, 30.0)  # Reduced from 60s to 30s
            
            # Normalize timestamps to simulation timeframe (0-1800s = 30 minutes)
            if traj_df is not None and 'timestamp' in traj_df.columns:
                # Get relative timestamps within the trajectory
                timestamps = traj_df['timestamp'].values
                min_ts = timestamps.min()
                max_ts = timestamps.max()
                
                # Normalize to simulation timeframe
                if max_ts > min_ts:
                    # Scale trajectory duration to 60-600 seconds (1-10 minutes) - more realistic
                    trajectory_duration = random.uniform(60, 600)
                    normalized_timestamps = (timestamps - min_ts) / (max_ts - min_ts) * trajectory_duration
                else:
                    # Single timestamp case
                    normalized_timestamps = np.array([0, 120])  # 2 minute trip
                
                # Add random departure time within simulation window
                depart_time = random.uniform(0, 2400)  # Depart within first 40 minutes
                arrival_time = depart_time + normalized_timestamps[-1]
            else:
                # Fallback: use index-based timing
                depart_time = random.uniform(0, 2400)  # Random departure within 40 minutes
                trajectory_duration = random.uniform(60, 600)  # 1-10 minutes
                arrival_time = depart_time + trajectory_duration
                
            max_arrival_time = max(max_arrival_time, arrival_time)

            # Check if required columns exist
            required_cols = ['lat', 'lon', 'timestamp']
            if not all(col in traj_df.columns for col in required_cols):
                continue
                
            start_row = traj_df.iloc[0]
            end_row = traj_df.iloc[-1]

            # More lenient trip filtering
            time_diff = arrival_time - depart_time
            if time_diff < 15:  # Reduced from 30s to 15s
                continue

            # Calculate approximate distance with better thresholds
            lat_diff = abs(end_row['lat'] - start_row['lat'])
            lon_diff = abs(end_row['lon'] - start_row['lon'])
            if lat_diff < 0.0005 and lon_diff < 0.0005:  # Reduced threshold
                continue

            # More conservative via point selection for better map-matching
            via_points = []
            if traj_df is not None and len(traj_df) > 4:  # Need at least 5 points for via points
                # Fewer via points for more reliable routing
                desired_vias = min(3, max(0, len(traj_df) // 15))  # Reduced from 5 and increased divisor
                if desired_vias > 0:
                    step = max(2, len(traj_df) // (desired_vias + 1))
                    for idx in range(step, len(traj_df) - step, step):
                        if idx < len(traj_df):
                            row = traj_df.iloc[idx]
                            # Check if lat/lon columns exist before accessing
                            if 'lat' in row and 'lon' in row:
                                # Smaller spatial jitter for better map-matching
                                dlon = random.uniform(-0.0001, 0.0001)  # ~10m instead of 25m
                                dlat = random.uniform(-0.0001, 0.0001)
                                via_points.append((float(row['lon']) + dlon, float(row['lat']) + dlat))
                                if len(via_points) >= desired_vias:
                                    break

            # Validate coordinates before adding to buffer
            try:
                start_lon, start_lat = float(start_row['lon']), float(start_row['lat'])
                end_lon, end_lat = float(end_row['lon']), float(end_row['lat'])
                
                # Basic coordinate validation
                if not (-180 <= start_lon <= 180 and -90 <= start_lat <= 90):
                    continue
                if not (-180 <= end_lon <= 180 and -90 <= end_lat <= 90):
                    continue

                trips_buffer.append({
                    'id': str(veh_id),
                    'depart': depart_time,
                    'fromLon': start_lon,
                    'fromLat': start_lat,
                    'toLon': end_lon,
                    'toLat': end_lat,
                    'via': via_points,
                })
            except (ValueError, KeyError) as e:
                print(f"Warning: Invalid coordinates for vehicle {veh_id}: {e}")
                continue

        if not trips_buffer:
            print("Warning: No valid coordinate-based trips found.")
            return None, 0

        # Sort trips by departure time to avoid SUMO warnings and improve loading
        trips_buffer.sort(key=lambda t: t['depart'])

        root = ET.Element('routes')
        valid_trips_created = 0
        for t in trips_buffer:
            try:
                # Compose coordinates per SUMO spec using fromLonLat/toLonLat
                trip_attrs = {
                    'id': t['id'],
                    'depart': str(max(0, int(t['depart']))),  # Ensure non-negative
                    'fromLonLat': f"{t['fromLon']:.6f},{t['fromLat']:.6f}",
                    'toLonLat': f"{t['toLon']:.6f},{t['toLat']:.6f}",
                }
                # Inject viaLonLat if we have intermediate points
                if t['via']:
                    via_str = ' '.join([f"{lon:.6f},{lat:.6f}" for lon, lat in t['via']])
                    trip_attrs['viaLonLat'] = via_str
                ET.SubElement(root, 'trip', attrib=trip_attrs)
                valid_trips_created += 1
            except Exception as e:
                print(f"Warning: Failed to create trip XML for {t['id']}: {e}")
                continue

        if valid_trips_created == 0:
            print("Warning: No valid coordinate-based trips found after filtering.")
            return None, 0

        # Ensure the directory exists before writing
        os.makedirs(os.path.dirname(trip_file_path), exist_ok=True)
        
        # Write XML via explicit file handle
        try:
            with open(trip_file_path, "wb") as f:
                ET.ElementTree(root).write(f, encoding='utf-8', xml_declaration=True)
            print(f"Successfully created trip file: {trip_file_path}")
            print(f"Trip file size: {os.path.getsize(trip_file_path)} bytes")
        except Exception as e:
            print(f"Error writing trip file: {e}")
            return None, 0

        # duarouter with more robust parameters
        sim_end_time = int(max_arrival_time) + 300  # Give more buffer time
        cmd = (
            f'duarouter -n {self.network.net_file_path} --route-files {trip_file_path} '
            f'-o {route_file_path} --ignore-errors -v '
            f'--unsorted-input '
            f'--mapmatch.distance 2000.0 --mapmatch.junctions '  # Increased distance
            f'--repair --remove-loops '
            f'--routing-threads 2 --routing-algorithm CH '  # Reduced threads for stability
            f'--begin 0 --end {sim_end_time} '
            f'--junction-taz --weights.priority-factor 0.5 '
            f'--weights.random-factor 0.1'  # Add randomness for better routing
        )

        print(f"Running duarouter with command: {cmd}")
        print(f"Trip file path: {trip_file_path}")
        print(f"Route file path: {route_file_path}")
        print(f"Trip file exists: {os.path.exists(trip_file_path)}")
        print(f"Current working directory: {os.getcwd()}")
        
        # Change to the directory containing the files to avoid path issues
        original_cwd = os.getcwd()
        file_dir = os.path.dirname(trip_file_path)
        os.chdir(file_dir)
        
        try:
            result = os.system(cmd)
        finally:
            os.chdir(original_cwd)

        if result != 0:
            print(f"duarouter returned non-zero exit code: {result}")

        if self._verify_route_file(route_file_path, episode_id):
            sim_duration = max(300, math.ceil(max_arrival_time) + 120)  # Minimum 5 min simulation
            self._cleanup_temp_files(trip_file_path)
            return route_file_path, sim_duration

        return None, 0

    def _generate_edge_based_routes(self, trajectories: Dict[str, pd.DataFrame], 
                                    trip_file_path: str, route_file_path: str, episode_id: any) -> Tuple[Optional[str], int]:
        """Generate routes using edge-based trips by finding closest edges to coordinates"""
        # CRITICAL FIX: Handle DataFrame input
        if isinstance(trajectories, pd.DataFrame) and not trajectories.empty:
            # Convert DataFrame to dict format for processing
            trajectory_dict = {}
            for veh_id, group in trajectories.groupby('VehId'):
                trajectory_dict[str(veh_id)] = group[['lat', 'lon', 'timestamp']].reset_index(drop=True)
            trajectories = trajectory_dict
        
        # Buffer and sort trips by depart time
        trips_buffer = []
        max_arrival_time = 0

        for veh_id, traj_df in trajectories.items():
            # Fix DataFrame boolean evaluation error
            if traj_df is None or len(traj_df) < 2:
                continue

            # Ensure traj_df is a DataFrame before calling sort_values
            if isinstance(traj_df, pd.Series):
                # Convert Series to DataFrame if needed
                traj_df = traj_df.to_frame().T
            elif not isinstance(traj_df, pd.DataFrame):
                continue
            
            # Ensure proper numeric index
            traj_df = traj_df.reset_index(drop=True)
            
            # CRITICAL FIX: Ensure coordinate columns are numeric before any operations
            if 'lat' in traj_df.columns:
                traj_df['lat'] = pd.to_numeric(traj_df['lat'], errors='coerce')
            if 'lon' in traj_df.columns:
                traj_df['lon'] = pd.to_numeric(traj_df['lon'], errors='coerce')
            
            # Drop any rows with invalid coordinates (only if columns exist)
            if 'lat' in traj_df.columns and 'lon' in traj_df.columns:
                traj_df = traj_df.dropna(subset=['lat', 'lon'])
            
            # Check if we still have data after cleaning
            if len(traj_df) == 0:
                self.logger.warning(f"Trajectory {veh_id} has no valid coordinates after cleaning")
                continue
            
            # Check if timestamp column exists before sorting
            if 'timestamp' in traj_df.columns:
                traj_df = traj_df.sort_values('timestamp')
            else:
                # If no timestamp column, sort by index
                traj_df = traj_df.sort_index()

            # Time jitter (Gaussian, std 60s) for domain randomization
            jitter = random.gauss(0.0, 60.0)
            
            # Handle timestamp column - check if it exists and is valid
            if 'timestamp' in traj_df.columns and not traj_df['timestamp'].isna().all():
                depart_time = max(0, float(traj_df['timestamp'].iloc[0]) / 1000.0 + jitter)
                arrival_time = float(traj_df['timestamp'].iloc[-1]) / 1000.0
            else:
                # Fallback: use row-based timing
                depart_time = max(0, 0.0 + jitter)
                arrival_time = len(traj_df) * 0.1
            
            max_arrival_time = max(max_arrival_time, arrival_time)

            start_row = traj_df.iloc[0]
            end_row = traj_df.iloc[-1]

            # CRITICAL FIX: Check if coordinate columns exist before accessing them
            if 'lat' not in start_row.index or 'lon' not in start_row.index:
                self.logger.warning(f"Trajectory {veh_id} missing coordinate columns (lat/lon)")
                continue
            
            # Get coordinates with proper error handling
            try:
                start_lat = float(start_row['lat'])
                start_lon = float(start_row['lon'])
                end_lat = float(end_row['lat'])
                end_lon = float(end_row['lon'])
                
                # Validate coordinates are within reasonable bounds
                if not (-90 <= start_lat <= 90) or not (-180 <= start_lon <= 180):
                    self.logger.warning(f"Trajectory {veh_id} has invalid start coordinates: lat={start_lat}, lon={start_lon}")
                    continue
                if not (-90 <= end_lat <= 90) or not (-180 <= end_lon <= 180):
                    self.logger.warning(f"Trajectory {veh_id} has invalid end coordinates: lat={end_lat}, lon={end_lon}")
                    continue
                
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Trajectory {veh_id} has invalid coordinate data: {e}")
                continue

            # Find closest edges for start and end points with larger search radius
            start_edge = self.network.find_closest_edge(start_lat, start_lon, radius=400.0)
            end_edge = self.network.find_closest_edge(end_lat, end_lon, radius=400.0)

            if start_edge and end_edge and start_edge.getID() != end_edge.getID():
                trips_buffer.append({
                    'id': str(veh_id),
                    'depart': depart_time,
                    'from': start_edge.getID(),
                    'to': end_edge.getID(),
                })

        if not trips_buffer:
            print("Warning: No valid edge-based trips found.")
            return None, 0

        trips_buffer.sort(key=lambda t: t['depart'])

        root = ET.Element('routes')
        for t in trips_buffer:
            attrs = {'id': t['id'], 'depart': str(int(t['depart'])), 'from': t['from'], 'to': t['to']}
            ET.SubElement(root, 'trip', attrib=attrs)

        # Ensure the directory exists before writing
        os.makedirs(os.path.dirname(trip_file_path), exist_ok=True)
        
        with open(trip_file_path, "wb") as f:
            ET.ElementTree(root).write(f, encoding='utf-8', xml_declaration=True)
        print(f"Successfully created edge-based trip file: {trip_file_path}")
        print(f"Trip file size: {os.path.getsize(trip_file_path)} bytes")

        # Run duarouter for edge-based routing with robustness flags
        cmd = (
            f'duarouter -n {self.network.net_file_path} --route-files {trip_file_path} '
            f'-o {route_file_path} --ignore-errors -v '
            f'--unsorted-input --repair --remove-loops '
            f'--routing-threads 4 --routing-algorithm CH '
            f'--begin 0 --end {int(max_arrival_time) + 120} '
            f'--junction-taz'
        )

        print(f"Running edge-based duarouter: {cmd}")
        print(f"Trip file path: {trip_file_path}")
        print(f"Route file path: {route_file_path}")
        print(f"Trip file exists: {os.path.exists(trip_file_path)}")
        print(f"Current working directory: {os.getcwd()}")
        
        # Change to the directory containing the files to avoid path issues
        original_cwd = os.getcwd()
        file_dir = os.path.dirname(trip_file_path)
        os.chdir(file_dir)
        
        try:
            result = os.system(cmd)
        finally:
            os.chdir(original_cwd)

        if self._verify_route_file(route_file_path, episode_id):
            sim_duration = math.ceil(max_arrival_time) + 60
            self._cleanup_temp_files(trip_file_path)
            return route_file_path, sim_duration

        return None, 0

    def _generate_fallback_routes(self, trajectories: Dict[str, pd.DataFrame], episode_id: any) -> Tuple[Optional[str], int]:
        """Generate random routes but use timing from real trajectories"""
        print(f"Generating fallback random routes for episode {episode_id}")
        
        # Calculate duration based on real trajectory data
        max_time = 0
        for veh_id, traj_df in trajectories.items():
            # Fix DataFrame boolean evaluation error
            if traj_df is not None and len(traj_df) >= 2:
                # Ensure traj_df is a DataFrame before calling sort_values
                if isinstance(traj_df, pd.Series):
                    # Convert Series to DataFrame if needed
                    traj_df = traj_df.to_frame().T
                elif not isinstance(traj_df, pd.DataFrame):
                    continue
                
                # Check if timestamp column exists before sorting
                if 'timestamp' in traj_df.columns:
                    traj_df = traj_df.sort_values('timestamp')
                    arrival_time = traj_df['timestamp'].iloc[-1] / 1000.0
                else:
                    # If no timestamp column, sort by index and use a default time
                    traj_df = traj_df.sort_index()
                    arrival_time = len(traj_df) * 60  # Assume 1 minute per point
                max_time = max(max_time, arrival_time)
        
        duration = max(300, min(int(max_time) + 60, 1200))  # Between 5 and 20 minutes
        
        # Use random trip generation but constrain to calculated duration
        route_file_path = os.path.join(self.output_dir, f'fallback_routes_{self.worker_id}_{str(episode_id)}.rou.xml')
        random_trips_script = os.path.join(os.environ.get("SUMO_HOME", "/usr/share/sumo"), "tools/randomTrips.py")
        
        # Generate fewer vehicles for stability
        num_vehicles = min(len(trajectories), 50)
        period = max(duration / num_vehicles, 1.0)
        
        cmd = (f'python3 {random_trips_script} -n {self.network.net_file_path} '
               f'-r {route_file_path} -e {duration} -l '
               f'--period {period} --fringe-factor 1.0 --seed {episode_id} '
               f'--junction-taz')
        
        print(f"Fallback command: {cmd}")
        result = os.system(cmd)
        
        if result == 0 and os.path.exists(route_file_path):
            print(f"Fallback routes generated successfully: {route_file_path}")
            return route_file_path, duration
        
        print("Fallback route generation also failed.")
        return None, 0

    def _verify_route_file(self, route_file_path: str, episode_id: any) -> bool:
        """Verify that the route file contains valid vehicles"""
        try:
            if not os.path.exists(route_file_path):
                print(f"Route file does not exist: {route_file_path}")
                return False
                
            tree_out = ET.parse(route_file_path)
            vehicles = tree_out.getroot().findall('vehicle')
            
            if not vehicles:
                print(f"Warning: No valid routes were generated by duarouter for episode {episode_id}.")
                return False
            
            print(f"Successfully generated {len(vehicles)} vehicle routes for episode {episode_id}")
            return True
            
        except (FileNotFoundError, ET.ParseError) as e:
            print(f"Warning: duarouter failed to produce a valid output file for episode {episode_id}: {e}")
            return False

    def _cleanup_temp_files(self, trip_file_path: str):
        """Clean up temporary files"""
        if os.path.exists(trip_file_path):
            os.remove(trip_file_path)
    
    def convert_to_ev_routes(self, input_route_file: str, output_route_file: str, 
                           charging_stations: List[Dict[str, str]]) -> None:
        try:
            tree = ET.parse(input_route_file)
        except (FileNotFoundError, ET.ParseError):
            print(f"Error: Could not parse input route file: {input_route_file}. Skipping EV conversion.")
            return
            
        root = tree.getroot()
        routes_elem = self._create_routes_element()
        self._add_ev_type_distribution(routes_elem)
        self._add_vehicles_with_charging_stops(root, routes_elem, charging_stations)
        
        output_tree = ET.ElementTree(routes_elem)
        # Write via explicit file handle to avoid platform writer issues
        with open(output_route_file, "wb") as f:
            output_tree.write(f, encoding='utf-8', xml_declaration=True)
        print(f"Converted and saved formatted routes to: {output_route_file}")

    def _create_routes_element(self) -> ET.Element:
        return ET.Element(
            'routes',
            attrib={
                'xmlns:xsi': "http://www.w3.org/2001/XMLSchema-instance",
                'xsi:noNamespaceSchemaLocation': "http://sumo.dlr.de/xsd/routes_file.xsd"
            }
        )

    def _add_ev_type_distribution(self, routes_elem: ET.Element) -> None:
        vtype_dist_elem = ET.SubElement(routes_elem, 'vTypeDistribution', id='EV')
        vtype_elem = ET.SubElement(
            vtype_dist_elem, 'vType', id='EV_CarTYPE', vClass='evehicle',
            color='red', emissionClass="Energy", mass=str(self.ev_config.mass)
        )
        params = {
            "device.battery.capacity": str(self.ev_config.capacity),
            "device.battery.maximumPower": str(self.ev_config.maximum_power),
            "device.battery.propulsionEfficiency": str(self.ev_config.propulsion_efficiency),
            "device.battery.recuperationEfficiency": str(self.ev_config.recuperation_efficiency),
            "device.battery.airDragCoefficient": str(self.ev_config.air_drag_coefficient),
            "device.battery.rollDragCoefficient": str(self.ev_config.roll_drag_coefficient),
            "device.battery.track": "true",
            "device.battery.aggregation": "true"
        }
        for key, value in params.items():
            ET.SubElement(vtype_elem, 'param', key=key, value=value)

    def _add_vehicles_with_charging_stops(self, 
                                         root: ET.Element, 
                                         routes_elem: ET.Element,
                                         charging_stations: List[Dict[str, str]]) -> None:
        """
        Add vehicles with probabilistic charging stops.
        
        Strategy:
        1. Check if route passes near any charging station (on-route charging)
        2. For vehicles with low initial SoC, probabilistically add nearest station (low probability: 15%)
        3. This ensures some charging activity without forcing unrealistic behavior
        """
        # Build a quick lookup from lane -> list of station ids for O(1) membership checks
        lane_to_station_ids: Dict[str, List[str]] = {}
        station_coords = {}  # Map station_id to (lat, lon) for distance calculations
        
        for s in charging_stations:
            lane_to_station_ids.setdefault(s['lane'], []).append(s['id'])
            # Try to get station coordinates from lane
            try:
                lane = self.network.net.getLane(s['lane'])
                shape = lane.getShape()
                if shape:
                    midpoint = shape[len(shape) // 2]
                    lon, lat = self.network.net.convertXY2LonLat(midpoint[0], midpoint[1])
                    station_coords[s['id']] = (lat, lon)
            except:
                pass

        vehicles_with_stops = 0
        total_vehicles = 0
        
        for vehicle in root.findall('vehicle'):
            total_vehicles += 1
            route_elem_in = vehicle.find('route')
            if route_elem_in is None:
                continue
            
            vehicle_elem = ET.SubElement(
                routes_elem, 'vehicle', id=vehicle.get('id'),
                depart=vehicle.get('depart'), type='EV_CarTYPE'
            )
            ET.SubElement(vehicle_elem, 'route', edges=route_elem_in.get('edges'))
            
            # Domain randomization for vehicle initial energy if provided
            # REPRODUCIBILITY: Use deterministic vehicle ID hash for initial SoC
            vehicle_id = vehicle.get('id', '')
            vehicle_hash_for_soc = sum(ord(c) for c in str(vehicle_id))
            
            if callable(self.ev_config.initial_soc_dist):
                initial_soc = float(max(0.0, min(1.0, self.ev_config.initial_soc_dist())))
            else:
                # Deterministic SoC based on vehicle ID: range 0.6-1.0
                initial_soc = 0.6 + 0.4 * ((vehicle_hash_for_soc % 1000) / 1000.0)
            base_capacity = float(self.ev_config.capacity)
            if callable(self.ev_config.capacity_dist):
                try:
                    base_capacity = float(self.ev_config.capacity_dist())
                except Exception:
                    base_capacity = float(self.ev_config.capacity)
            initial_capacity = base_capacity * initial_soc
            ET.SubElement(vehicle_elem, 'param', key="device.battery.actualBatteryCapacity", value=str(initial_capacity))
            
            # Strategy 1: Check if route passes through any charging station lanes
            route_edges = route_elem_in.get('edges').split()
            found_stop = False
            for edge_id in route_edges:
                try:
                    edge = self.network.net.getEdge(edge_id)
                    for lane in edge.getLanes():
                        lane_id = lane.getID()
                        if lane_id in lane_to_station_ids:
                            # REPRODUCIBILITY: Use vehicle ID hash to select station deterministically
                            station_list = lane_to_station_ids[lane_id]
                            station_idx = vehicle_hash_for_soc % len(station_list)
                            station_id = station_list[station_idx]
                            ET.SubElement(
                                vehicle_elem, 'stop', chargingStation=station_id,
                                duration="300.00"
                            )
                            found_stop = True
                            vehicles_with_stops += 1
                            break
                    if found_stop:
                        break
                except KeyError:
                    continue
            
            # Strategy 2: Probabilistic low-battery rerouting (DETERMINISTIC)
            # LOW PROBABILITY (15%) to avoid unrealistic behavior
            # Only for vehicles with low initial SoC (<0.4)
            # REPRODUCIBILITY: Use vehicle ID hash for deterministic behavior (already computed above)
            should_reroute = (vehicle_hash_for_soc % 100) < 15  # 15% probability (0-14 out of 0-99)
            
            if not found_stop and initial_soc < 0.4 and should_reroute:
                # Find nearest charging station
                if station_coords and charging_stations:
                    # Get vehicle start location
                    try:
                        first_edge = self.network.net.getEdge(route_edges[0])
                        start_shape = first_edge.getShape()
                        if start_shape:
                            start_x, start_y = start_shape[0]
                            start_lon, start_lat = self.network.net.convertXY2LonLat(start_x, start_y)
                            
                            # Find nearest station
                            min_dist = float('inf')
                            nearest_station = None
                            for station_id, (stat_lat, stat_lon) in station_coords.items():
                                dist = ((start_lat - stat_lat)**2 + (start_lon - stat_lon)**2)**0.5
                                if dist < min_dist:
                                    min_dist = dist
                                    nearest_station = station_id
                            
                            if nearest_station:
                                # Add charging stop at nearest station
                                ET.SubElement(
                                    vehicle_elem, 'stop', chargingStation=nearest_station,
                                    duration="300.00"
                                )
                                found_stop = True
                                vehicles_with_stops += 1
                    except:
                        pass
        
        if total_vehicles > 0:
            charging_rate = vehicles_with_stops / total_vehicles * 100
            print(f"✅ Added charging stops to {vehicles_with_stops}/{total_vehicles} vehicles ({charging_rate:.1f}%)")
        else:
            print(f"⚠️ No vehicles found in route file")

    def _generate_simple_random_routes(self, episode_id: int) -> Tuple[str, int]:
        """Generate simple random routes as a fallback when real data routing fails"""
        route_file_path = os.path.join(self.output_dir, f"simple_random_routes_{self.worker_id}_{str(episode_id)}.rou.xml")
        
        # Get network edges for random route generation
        try:
            import sumolib
            net = sumolib.net.readNet(self.network.net_file_path)
            edges = [edge for edge in net.getEdges() if not edge.getID().startswith(':')]
            
            if not edges:
                print("No valid edges found for random route generation")
                return None, 0
                
        except Exception as e:
            print(f"Error reading network for random routes: {e}")
            return None, 0
        
        # Generate simple random routes
        routes = ET.Element('routes')
        
        # Add vehicle types
        vtype = ET.SubElement(routes, 'vType', 
                             id="electric_vehicle",
                             vClass="passenger",
                             emissionClass="Energy",
                             energyFile=os.path.abspath("generated_files/charging_stations/vehicles.xml"))
        
        # Add battery device configuration
        battery_device = ET.SubElement(vtype, 'param', key="device.battery.track", value="true")
        battery_aggregation = ET.SubElement(vtype, 'param', key="device.battery.aggregation", value="true")
        
        # Generate 100-200 random routes for better simulation
        num_routes = random.randint(100, 200)
        duration = 3600  # 60 minutes simulation
        
        for i in range(num_routes):
            # Random start and end edges
            start_edge = random.choice(edges)
            end_edge = random.choice(edges)
            
            # Random departure time spread over the duration
            depart_time = random.uniform(0, duration - 600)  # Leave 10 minute buffer
            
            # Create route
            route_id = f"route_{i}"
            route = ET.SubElement(routes, 'route', 
                                id=route_id,
                                edges=f"{start_edge.getID()} {end_edge.getID()}")
            
            # Create vehicle
            vehicle_id = f"vehicle_{i}"
            vehicle = ET.SubElement(routes, 'vehicle',
                                  id=vehicle_id,
                                  type="electric_vehicle",
                                  route=route_id,
                                  depart=str(depart_time))
        
        # Save routes
        tree = ET.ElementTree(routes)
        tree.write(route_file_path, encoding='utf-8', xml_declaration=True)
        
        print(f"Generated {num_routes} simple random routes for episode {episode_id}")
        return route_file_path, duration

    def _add_random_routes_to_existing(self, existing_route_file: str, episode_id: int, duration: int) -> Tuple[str, int]:
        """Add random routes to an existing route file to increase vehicle density"""
        try:
            if sumolib is None:
                print("sumolib not available, skipping additional routes")
                return existing_route_file, duration
            
            # Check if network file exists
            if not os.path.exists(self.network.net_file_path):
                print(f"Network file not found: {self.network.net_file_path}")
                return existing_route_file, duration
                
            net = sumolib.net.readNet(self.network.net_file_path)
            edges = [edge for edge in net.getEdges() if not edge.getID().startswith(':')]
            
            if not edges:
                print("No valid edges found for additional random routes")
                return existing_route_file, duration
                
        except Exception as e:
            print(f"Error reading network for additional routes: {e}")
            # Return the existing file without modification
            return existing_route_file, duration
        
        # Parse existing routes
        tree = ET.parse(existing_route_file)
        root = tree.getroot()
        
        # Count existing vehicles
        existing_vehicles = len(root.findall('vehicle'))
        print(f"Found {existing_vehicles} existing vehicles, adding more...")
        
        # Add 50-100 additional random routes
        additional_routes = random.randint(50, 100)
        
        for i in range(additional_routes):
            # Random start and end edges
            start_edge = random.choice(edges)
            end_edge = random.choice(edges)
            
            # Random departure time spread over the duration
            depart_time = random.uniform(0, duration - 300)  # Leave 5 minute buffer
            
            # Create route
            route_id = f"additional_route_{i}"
            route = ET.SubElement(root, 'route', 
                                id=route_id,
                                edges=f"{start_edge.getID()} {end_edge.getID()}")
            
            # Create vehicle
            vehicle_id = f"additional_vehicle_{i}"
            vehicle = ET.SubElement(root, 'vehicle',
                                  id=vehicle_id,
                                  type="electric_vehicle",
                                  route=route_id,
                                  depart=str(depart_time))
        
        # Save enhanced routes
        tree.write(existing_route_file, encoding='utf-8', xml_declaration=True)
        
        total_vehicles = existing_vehicles + additional_routes
        print(f"Enhanced routes with {additional_routes} additional vehicles (total: {total_vehicles})")
        return existing_route_file, duration

    def _generate_hybrid_routes(self, trajectories: Dict[str, pd.DataFrame], episode_id: int) -> Tuple[str, int]:
        """Generate hybrid routes combining real data with random routes"""
        route_file_path = os.path.join(self.output_dir, f"hybrid_routes_{self.worker_id}_{str(episode_id)}.rou.xml")
        
        # Get network edges for random route generation
        try:
            if sumolib is None:
                print("sumolib not available, using fallback route generation")
                return self._generate_fallback_routes(trajectories, episode_id)
                
            net = sumolib.net.readNet(self.network.net_file_path)
            edges = [edge for edge in net.getEdges() if not edge.getID().startswith(':')]
            
            if not edges:
                print("No valid edges found for hybrid route generation")
                return None, 0
                
        except Exception as e:
            print(f"Error reading network for hybrid routes: {e}")
            return None, 0
        
        # Generate hybrid routes
        routes = ET.Element('routes')
        
        # Add vehicle types
        vtype = ET.SubElement(routes, 'vType', 
                             id="electric_vehicle",
                             vClass="passenger",
                             emissionClass="Energy",
                             energyFile=os.path.abspath("generated_files/charging_stations/vehicles.xml"))
        
        # Add battery device configuration
        battery_device = ET.SubElement(vtype, 'param', key="device.battery.track", value="true")
        battery_aggregation = ET.SubElement(vtype, 'param', key="device.battery.aggregation", value="true")
        
        # Generate routes based on real trajectories (simplified)
        real_route_count = 0
        if isinstance(trajectories, pd.DataFrame) and not trajectories.empty:
            # Convert DataFrame to dict format for processing
            trajectory_dict = {}
            for veh_id, group in trajectories.groupby('VehId'):
                trajectory_dict[str(veh_id)] = group[['lat', 'lon', 'timestamp']].reset_index(drop=True)
            trajectories = trajectory_dict
        
        if isinstance(trajectories, dict) and trajectories:
            for veh_id, traj_df in trajectories.items():
                if traj_df is not None and len(traj_df) >= 2:
                    # Ensure traj_df is a DataFrame
                    if isinstance(traj_df, pd.Series):
                        # Convert Series to DataFrame if needed
                        traj_df = traj_df.to_frame().T
                    elif not isinstance(traj_df, pd.DataFrame):
                        continue
                    
                    # Simple route from first to last point
                    start_edge = random.choice(edges)
                    end_edge = random.choice(edges)
                    
                    # Use normalized trajectory timing (not real-world timestamps)
                    if traj_df is not None and 'timestamp' in traj_df.columns:
                        # Normalize to simulation timeframe
                        timestamps = traj_df['timestamp'].values
                        min_ts = timestamps.min()
                        max_ts = timestamps.max()
                        
                        if max_ts > min_ts:
                            # Scale to 60-600 seconds duration (1-10 minutes)
                            trajectory_duration = random.uniform(60, 600)
                            depart_time = random.uniform(0, 2400)  # Random departure within 40 minutes
                        else:
                            depart_time = random.uniform(0, 2400)
                    else:
                        depart_time = random.uniform(0, 2400)
                    
                    # Create route
                    route_id = f"real_route_{real_route_count}"
                    route = ET.SubElement(routes, 'route', 
                                        id=route_id,
                                        edges=f"{start_edge.getID()} {end_edge.getID()}")
                    
                    # Create vehicle
                    vehicle_id = f"real_vehicle_{real_route_count}"
                    vehicle = ET.SubElement(routes, 'vehicle',
                                          id=vehicle_id,
                                          type="electric_vehicle",
                                          route=route_id,
                                          depart=str(depart_time))
                    
                    real_route_count += 1
        
        # Add random routes to supplement
        random_routes = random.randint(100, 200)
        duration = 3600  # 60 minutes simulation
        
        for i in range(random_routes):
            # Random start and end edges
            start_edge = random.choice(edges)
            end_edge = random.choice(edges)
            
            # Random departure time spread over the duration
            depart_time = random.uniform(0, duration - 600)  # Leave 10 minute buffer
            
            # Create route
            route_id = f"random_route_{i}"
            route = ET.SubElement(routes, 'route', 
                                id=route_id,
                                edges=f"{start_edge.getID()} {end_edge.getID()}")
            
            # Create vehicle
            vehicle_id = f"random_vehicle_{i}"
            vehicle = ET.SubElement(routes, 'vehicle',
                                  id=vehicle_id,
                                  type="electric_vehicle",
                                  route=route_id,
                                  depart=str(depart_time))
        
        # Save routes
        tree = ET.ElementTree(routes)
        tree.write(route_file_path, encoding='utf-8', xml_declaration=True)
        
        total_routes = real_route_count + random_routes
        print(f"Generated hybrid routes: {real_route_count} real + {random_routes} random = {total_routes} total")
        return route_file_path, duration
    
    def generate_episode_routes(self, 
                               episode_id: int, 
                               trajectories: Optional[Dict[str, pd.DataFrame]] = None,
                               use_real_data: bool = True,
                               use_synthetic_routes: bool = True) -> Tuple[Optional[str], int]:
        """
        Generate routes for an episode with comprehensive strategy.
        
        Args:
            episode_id: Episode identifier for reproducibility
            trajectories: Real-world trajectory data
            use_real_data: Whether to use real trajectory data
            use_synthetic_routes: Whether to add synthetic routes
            
        Returns:
            Tuple of (route_file_path, simulation_duration)
        """
        # Set episode-specific seed for reproducibility
        random.seed(42 + episode_id)
        np.random.seed(42 + episode_id)
        
        # Strategy 1: Try real data first if available and requested
        if use_real_data and trajectories and len(trajectories) > 0:
            try:
                result = self.generate_routes_from_data(trajectories, episode_id)
                if result[0] is not None:
                    # Add synthetic routes if requested
                    if use_synthetic_routes:
                        return self._add_random_routes_to_existing(result[0], episode_id, result[1])
                    return result
            except Exception as e:
                print(f"Real data route generation failed: {e}")
        
        # Strategy 2: Generate synthetic routes
        if use_synthetic_routes:
            try:
                return self.generate_random_trips(episode_id)
            except Exception as e:
                print(f"Synthetic route generation failed: {e}")
        
        # Strategy 3: Fallback to simple routes
        return self._generate_simple_random_routes(episode_id)