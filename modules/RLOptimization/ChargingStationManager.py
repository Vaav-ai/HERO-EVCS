# ChargingStationManager.py

import os
import xml.etree.ElementTree as ET
import random
from typing import List, Tuple, Dict, Optional
import sumolib
import math
from shapely.geometry import Point, box # <-- ADDED

# (Keep the _lane_xy helper function as is)
def _lane_xy(lane, pos):
    """
    Return a (x,y) point that lies `pos` metres from the lane start.
    Works with every current version of sumolib.
    """
    shape = lane.getShape()
    if not shape:
        return None, None
    travelled = 0.0
    for (x0, y0), (x1, y1) in zip(shape, shape[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        if travelled + seg >= pos:
            ratio = (pos - travelled) / seg
            return (x0 + ratio * (x1 - x0), y0 + ratio * (y1 - y0))
        travelled += seg
    return shape[-1]

class ChargingStationManager:
    """Manages charging station placement and configuration"""
    
    def __init__(self, net_file_path: str, placement_grid_id: str = None, grid_cells_data: Optional[List[Dict]] = None):
        self.net_file_path = net_file_path
        self.net = sumolib.net.readNet(net_file_path)
        self.output_dir = os.path.abspath("./generated_files/charging_stations")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # PARALLEL EXECUTION FIX: Add unique worker ID for file naming
        import time
        self.worker_id = f"{os.getpid()}_{int(time.time() * 1000) % 100000}"
        
        self.placement_bbox = None
        if placement_grid_id and grid_cells_data:
            self._set_placement_boundary(placement_grid_id, grid_cells_data)

    def _set_placement_boundary(self, grid_id: str, grid_cells: List[Dict]):
        """Finds the target grid and stores its geographic bounding box."""
        try:
            # Look for cell by both grid_id and cell_id for backward compatibility
            target_cell = next((cell for cell in grid_cells if 
                              cell.get('grid_id') == grid_id or cell.get('cell_id') == grid_id), None)
            if target_cell:
                self.placement_bbox = box(
                    target_cell['min_lon'], 
                    target_cell['min_lat'], 
                    target_cell['max_lon'], 
                    target_cell['max_lat']
                )
                print(f"✅ Placement constrained to grid {grid_id}: ({target_cell['min_lat']:.4f}, {target_cell['min_lon']:.4f}) to ({target_cell['max_lat']:.4f}, {target_cell['max_lon']:.4f})")
            else:
                print(f"⚠️ Warning: Could not find grid {grid_id} to constrain placement.")
                print(f"Available grids: {[cell.get('grid_id', cell.get('cell_id')) for cell in grid_cells[:5]]}")
        except Exception as e:
            print(f"❌ Error setting placement boundary: {e}")
            self.placement_bbox = None

    def generate_placement(self, num_stations: int) -> List[Dict]:
        """
        Generate random placement for charging stations, constrained to a grid if specified.
        """
        valid_locations = []
        total_edges = 0
        checked_edges = 0
        valid_lanes = 0
        
        print(f"🔍 Searching for valid station locations (target: {num_stations} stations)...")
        
        # Use more generous boundary expansion for grid constraints
        expanded_bbox = None
        if self.placement_bbox:
            bounds = self.placement_bbox.bounds
            print(f"   Grid boundary: ({bounds[1]:.4f}, {bounds[0]:.4f}) to ({bounds[3]:.4f}, {bounds[2]:.4f})")
            # Much more generous buffer - ~500m instead of ~100m
            expanded_bbox = self.placement_bbox.buffer(0.005)  # ~500m buffer
        else:
            print("   No grid boundary constraint")
        
        # First pass: try to find locations within expanded boundary
        for edge in self.net.getEdges():
            total_edges += 1
            if edge.getID().startswith(':'):
                continue
                
            checked_edges += 1
            edge_length = edge.getLength()
            
            # Skip very short edges
            if edge_length < 20:
                continue
            
            for lane in edge.getLanes():
                if not lane.allows("passenger"):
                    continue

                valid_lanes += 1
                lane_id = lane.getID()
                
                # Check if any part of the lane is within our area of interest
                lane_valid = True
                if expanded_bbox:
                    lane_valid = False
                    # Check multiple points along the lane
                    check_positions = [edge_length * 0.1, edge_length * 0.3, edge_length * 0.5, edge_length * 0.7, edge_length * 0.9]
                    for check_pos in check_positions:
                        if check_pos >= edge_length:
                            continue
                            
                        x, y = _lane_xy(lane, check_pos)
                        if x is None: 
                            continue
                            
                        try:
                            lon, lat = self.net.convertXY2LonLat(x, y)
                            if expanded_bbox.contains(Point(lon, lat)):
                                lane_valid = True
                                break
                        except Exception:
                            continue
                
                if not lane_valid:
                    continue
                
                # Add multiple placement spots along the valid lane
                step_size = max(25, int(edge_length / 8))  # More placement options
                for pos in range(15, int(edge_length) - 10, step_size):
                    try:
                        x_pos, y_pos = _lane_xy(lane, pos)
                        if x_pos is None: 
                            continue
                            
                        lon_pos, lat_pos = self.net.convertXY2LonLat(x_pos, y_pos)
                        
                        # Final boundary check for this specific position
                        if expanded_bbox and not expanded_bbox.contains(Point(lon_pos, lat_pos)):
                            continue

                        valid_locations.append({
                            'lane': lane_id, 
                            'position': pos,
                            'end_position': min(pos + 15, edge_length),
                            'lat': lat_pos, 
                            'lon': lon_pos
                        })
                    except Exception as e:
                        # Skip problematic positions but continue with the lane
                        continue
                
                # Limit locations per lane to avoid clustering
                if len(valid_locations) > num_stations * 10:  # Collect plenty of options
                    break
            
            if len(valid_locations) > num_stations * 10:
                break

        print(f"📊 Network analysis: {total_edges} total edges, {checked_edges} checked, {valid_lanes} valid lanes")
        print(f"📍 Found {len(valid_locations)} valid station locations")

        # If still insufficient locations, try progressively more permissive approaches
        if len(valid_locations) < num_stations:
            print(f"⚠️ Only {len(valid_locations)} locations found, need {num_stations}. Trying fallback approaches...")
            
            # Try with even larger buffer
            if self.placement_bbox and len(valid_locations) == 0:
                print("🔄 Trying with very large boundary buffer...")
                fallback_locations = self._generate_fallback_placement_with_large_buffer(num_stations)
                if fallback_locations:
                    valid_locations.extend(fallback_locations)
            
            # If still not enough, try without boundary constraints
            if len(valid_locations) < num_stations:
                print("🚨 Trying fallback without boundary constraints...")
                fallback_locations = self._generate_fallback_placement(num_stations)
                if fallback_locations:
                    # Take what we need
                    needed = num_stations - len(valid_locations)
                    valid_locations.extend(fallback_locations[:needed])

        if not valid_locations:
            print(f"❌ Warning: No valid locations found even with fallbacks.")
            return []

        # Select the requested number of stations
        if len(valid_locations) >= num_stations:
            selected = random.sample(valid_locations, num_stations)
            print(f"✅ Selected {len(selected)} station locations from {len(valid_locations)} candidates")
        else:
            selected = valid_locations
            print(f"⚠️ Using all {len(selected)} available locations (requested {num_stations})")
            
        return selected

    def _generate_fallback_placement_with_large_buffer(self, num_stations: int) -> List[Dict]:
        """Generate placement with a very large boundary buffer as intermediate fallback."""
        if not self.placement_bbox:
            return []
            
        print(f"🔄 Trying with very large boundary buffer (~2km)...")
        # Very large buffer - ~2km
        very_large_bbox = self.placement_bbox.buffer(0.02)
        
        valid_locations = []
        edges_checked = 0
        max_edges_to_check = 2000  # Check more edges with large buffer
        
        for edge in self.net.getEdges():
            if edges_checked >= max_edges_to_check:
                break
                
            if edge.getID().startswith(':'):
                continue
                
            edges_checked += 1
            edge_length = edge.getLength()
            
            if edge_length < 15:
                continue
            
            # Check first valid lane
            for lane in edge.getLanes():
                if lane.allows("passenger"):
                    # Check midpoint
                    pos = edge_length / 2
                    try:
                        x, y = _lane_xy(lane, pos)
                        if x is not None:
                            lon, lat = self.net.convertXY2LonLat(x, y)
                            
                            if very_large_bbox.contains(Point(lon, lat)):
                                valid_locations.append({
                                    'lane': lane.getID(), 
                                    'position': pos,
                                    'end_position': min(pos + 15, edge_length),
                                    'lat': lat, 
                                    'lon': lon
                                })
                                break  # One location per edge
                    except Exception:
                        continue
            
            if len(valid_locations) >= num_stations * 2:  # Get some extras
                break
        
        print(f"🔧 Large buffer found {len(valid_locations)} locations from {edges_checked} edges")
        return valid_locations

    def _generate_fallback_placement(self, num_stations: int) -> List[Dict]:
        """Generate fallback placement without boundary constraints."""
        print(f"🚨 Generating fallback placement without boundary constraints...")
        valid_locations = []
        
        edges_checked = 0
        max_edges_to_check = 1000  # Limit to avoid infinite loops
        
        for edge in self.net.getEdges():
            if edges_checked >= max_edges_to_check:
                break
                
            if edge.getID().startswith(':'):
                continue
                
            edges_checked += 1
            edge_length = edge.getLength()
            
            # Take first valid lane
            for lane in edge.getLanes():
                if lane.allows("passenger"):
                    # Just take midpoint of lane
                    pos = edge_length / 2
                    x, y = _lane_xy(lane, pos)
                    if x is not None:
                        lon, lat = self.net.convertXY2LonLat(x, y)
                        
                        valid_locations.append({
                            'lane': lane.getID(), 
                            'position': pos,
                            'end_position': min(pos + 10, edge_length),
                            'lat': lat, 
                            'lon': lon
                        })
                        break  # One location per edge
            
            if len(valid_locations) >= num_stations * 3:  # Get some extras
                break
        
        print(f"🔧 Fallback found {len(valid_locations)} locations from {edges_checked} edges")
        
        if len(valid_locations) >= num_stations:
            selected = random.sample(valid_locations, num_stations)
            print(f"✅ Fallback selected {len(selected)} station locations")
            return selected
        else:
            print(f"⚠️ Fallback using all {len(valid_locations)} available locations")
            return valid_locations

    def create_additional_file(self, placements: List[Dict], file_path: str = None, episode_id: any = 0) -> Tuple[List[Dict[str, str]], str]:
        # Debug: Check input parameters
        print(f"DEBUG: create_additional_file called with file_path={file_path}, episode_id={episode_id}, type(episode_id)={type(episode_id)}")
        
        if file_path is None:
            # PARALLEL EXECUTION FIX: Include worker_id in filename for isolation
            # Ensure episode_id is converted to string to avoid path issues
            episode_id_str = str(episode_id) if episode_id is not None else "0"
            file_path = f"{self.output_dir}/charging_stations_{self.worker_id}_{episode_id_str}.add.xml"
        
        # Debug: Check file_path before os.path.dirname
        print(f"DEBUG: file_path before dirname: {file_path}, type: {type(file_path)}")
        
        # Ensure the directory exists
        try:
            dirname = os.path.dirname(file_path)
            print(f"DEBUG: dirname result: {dirname}, type: {type(dirname)}")
            # Only create directory if dirname is not empty (not current directory)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
        except Exception as e:
            print(f"DEBUG: Error in os.path.dirname: {e}")
            print(f"DEBUG: file_path value: {file_path}")
            print(f"DEBUG: file_path type: {type(file_path)}")
            raise e
        root = ET.Element('additional')
        charging_stations = []
        
        for i, placement in enumerate(placements):
            station_id = f'CS_{i}'
            
            # Get lane ID with proper validation
            lane_id = placement.get('lane', placement.get('lane_id', ''))
            
            # Validate lane exists in network
            try:
                if lane_id and self.net.getLane(lane_id):
                    # Lane exists, use it
                    pass
                else:
                    # Lane doesn't exist, try to find a valid lane for the edge
                    edge_id = placement.get('edge_id', '')
                    if edge_id:
                        try:
                            edge = self.net.getEdge(edge_id)
                            if edge and edge.getLanes():
                                # Use the first valid lane
                                lane_id = edge.getLanes()[0].getID()
                                print(f"⚠️ Using fallback lane {lane_id} for edge {edge_id}")
                            else:
                                print(f"❌ No valid lanes found for edge {edge_id}, skipping station {station_id}")
                                continue
                        except Exception as e:
                            print(f"❌ Error finding lane for edge {edge_id}: {e}, skipping station {station_id}")
                            continue
                    else:
                        print(f"❌ No lane_id or edge_id provided for station {station_id}, skipping")
                        continue
            except Exception as e:
                print(f"❌ Error validating lane {lane_id}: {e}, skipping station {station_id}")
                continue
            
            # Validate position values
            start_pos = float(placement.get('position', 0))
            end_pos = float(placement.get('end_position', start_pos + 15))
            
            # Ensure positions are within lane bounds
            try:
                lane = self.net.getLane(lane_id)
                lane_length = lane.getLength()
                
                # CRITICAL FIX: Ensure lane is long enough for charging station
                if lane_length < 5:
                    print(f"⚠️ Lane {lane_id} too short ({lane_length:.2f}m), skipping station {station_id}")
                    continue
                
                # Ensure positions are valid
                start_pos = max(0, min(start_pos, lane_length - 5))  # Leave at least 5m at end
                end_pos = max(start_pos + 5, min(end_pos, lane_length))  # At least 5m long station
                
                # Final validation
                if start_pos >= end_pos or end_pos > lane_length:
                    print(f"⚠️ Invalid positions for lane {lane_id}: start={start_pos}, end={end_pos}, length={lane_length}")
                    # Use safe defaults
                    start_pos = 0
                    end_pos = min(15, lane_length)
                
            except Exception as e:
                print(f"⚠️ Could not validate positions for lane {lane_id}: {e}")
                # Use safe defaults
                start_pos = 0
                end_pos = 15
            
            # Create charging station element
            try:
                ET.SubElement(
                    root, 'chargingStation', 
                    id=station_id, 
                    lane=lane_id,
                    startPos=str(start_pos), 
                    endPos=str(end_pos),
                    power="50000", 
                    efficiency="0.95", 
                    chargeDelay="1"
                )
                charging_stations.append({'id': station_id, 'lane': lane_id})
                print(f"✅ Created charging station {station_id} on lane {lane_id}")
            except Exception as e:
                print(f"❌ Error creating charging station {station_id}: {e}")
                continue
        
        if not charging_stations:
            print("❌ No valid charging stations created")
            return [], file_path
        
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding='utf-8', xml_declaration=True)
        print(f"✅ Created {len(charging_stations)} charging stations in {file_path}")
        
        # PARALLEL EXECUTION FIX: Store file path for cleanup
        if not hasattr(self, 'temp_files'):
            self.temp_files = []
        self.temp_files.append(file_path)
        
        return charging_stations, file_path
    
    def cleanup_temp_files(self):
        """
        PARALLEL EXECUTION FIX: Clean up temporary charging station files.
        Should be called after files have been used in simulation.
        """
        import time as time_module
        if not hasattr(self, 'temp_files'):
            return
        
        for file_path in self.temp_files:
            max_retries = 3
            for retry in range(max_retries):
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"🧹 Cleaned up temp file: {file_path}")
                        break
                except Exception as e:
                    if retry < max_retries - 1:
                        time_module.sleep(0.3)
                        continue
                    else:
                        print(f"⚠️ Could not remove {file_path}: {e}")
        
        self.temp_files = []
        
    def get_station_coordinates(self, additional_file_path: str) -> List[Dict]:
        stations = []
        try:
            tree = ET.parse(additional_file_path)
            root = tree.getroot()
            for station_elem in root.findall('chargingStation'):
                lane_id = station_elem.get('lane')
                pos = float(station_elem.get('startPos'))
                x, y = _lane_xy(self.net.getLane(lane_id), pos)
                if x is None: continue
                lon, lat = self.net.convertXY2LonLat(x, y)
                stations.append({
                    'station_id': station_elem.get('id'), 'lane_id': lane_id,
                    'lat': lat, 'lon': lon,
                    'power': float(station_elem.get('power', '50000')) / 1000
                })
        except Exception as e:
            print(f"Error extracting station coordinates: {e}")
        return stations
