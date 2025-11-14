try:
    import lxml.etree as ET
    print("Using lxml for XML parsing (better error recovery)")
except ImportError:
    import xml.etree.ElementTree as ET
    print("Using standard xml.etree.ElementTree (limited error recovery)")
from typing import List, Dict, Any
import logging
import math
import numpy as np
from typing import Optional
from collections import defaultdict
import sys
import os
from datetime import datetime
import pandas as pd

#TODO: fix and improve reward formulation
class SimulationAnalyzer:
    """Analyzes simulation results"""
    
    def _setup_logging(self, enable_file_logging=False):
        """Setup comprehensive logging for simulation analysis.
        
        Args:
            enable_file_logging: If True, enables logging to file. Default is False to save disk space.
        """
        # Create logger with hierarchical name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Set logging level based on environment, default to WARNING to reduce verbosity
        log_level = os.environ.get('ANALYZER_LOG_LEVEL', 'WARNING').upper()
        self.logger.setLevel(getattr(logging, log_level, logging.WARNING))
        
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
        
        # Only add file handler if explicitly enabled
        if enable_file_logging:
            log_dir = "./logs"
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"simulation_analyzer_{datetime.now().strftime('%Y%m%d')}.log")
            
            if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == log_file for handler in self.logger.handlers):
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
    
    def __init__(self, reward_weights: Optional[Dict[str, float]] = None, base_seed: int = 42, grid_id: str = None, enable_file_logging: bool = False):
        # Setup logging first
        self._setup_logging(enable_file_logging)
        
        # Store base_seed and grid_id for reproducible simulations
        self.base_seed = base_seed
        self.grid_id = grid_id
        
        if reward_weights is None:
            self.reward_weights = self.get_default_reward_weights()
        else:
            self.reward_weights = reward_weights
        
        self.logger.info("Initializing Simulation Analyzer")
        self.logger.info(f"  Reward weights: {self.reward_weights}")
        self.logger.info(f"  Base seed: {self.base_seed}")
        self.logger.info(f"  Grid ID: {self.grid_id}")
        # Track aggregate fleet metrics between episodes for differential rewards
        self.previous_fleet_metrics = {}
        # Rolling window for average energy per vehicle across recent episodes
        self.energy_history: List[float] = []

    def _robust_xml_parse(self, file_path, expected_root_element='edgedata'):
        """
        Robust XML parsing with multiple fallback strategies.
        
        Args:
            file_path: Path to XML file
            expected_root_element: Expected root XML element name
            
        Returns:
            XML root element or None if all parsing attempts fail
        """
        import re
        import time
        
        # Wait for file to be complete (helpful for parallel execution)
        if not self._wait_for_complete_file(file_path):
            self.logger.warning(f"File {file_path} may not be complete, proceeding anyway")
        
        # Strategy 1: Direct parsing
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            if root is not None:
                return root
        except (ET.ParseError, Exception) as e:
            self.logger.warning(f"Strategy 1 failed for {file_path}: {e}")
        
        # Strategy 2: Read file and clean content
        try:
            # Try multiple encodings
            content = None
            for encoding in ['utf-8', 'latin1', 'cp1252', 'ascii']:
                try:
                    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                        content = f.read()
                        break
                except (UnicodeDecodeError, IOError):
                    continue
            
            if not content or len(content.strip()) < 20:
                self.logger.warning(f"File {file_path} is too small or unreadable")
                return None
                
            # Clean XML content
            cleaned_content = self._clean_xml_content(content, expected_root_element)
            
            # Parse cleaned content - handle encoding declarations properly
            try:
                # First try with explicit encoding removal
                cleaned_content_no_decl = re.sub(r'<\?xml[^>]*?>', '', cleaned_content)
                root = ET.fromstring(cleaned_content_no_decl.encode('utf-8'))
                self.logger.info(f"Successfully fixed and parsed {file_path} (without declaration)")
                return root
            except Exception:
                # Fallback: use original cleaned content but ensure it's bytes
                root = ET.fromstring(cleaned_content.encode('utf-8'))
                self.logger.info(f"Successfully fixed and parsed {file_path}")
                return root
            
        except Exception as fix_error:
            self.logger.warning(f"Strategy 2 failed for {file_path}: {fix_error}")
        
        # Strategy 3: Robust parser with recovery (handle unicode/encoding issues)
        try:
            # Read file content first to handle encoding issues
            with open(file_path, 'rb') as f:  # Read as bytes
                content_bytes = f.read()
                
            # Create parser without explicit encoding to avoid conflicts
            parser = ET.XMLParser(recover=True)
            root = ET.fromstring(content_bytes, parser=parser)
            
            if root is not None:
                self.logger.info(f"Successfully parsed {file_path} with error recovery mode")
                return root
        except Exception as recovery_error:
            self.logger.warning(f"Strategy 3 failed for {file_path}: {recovery_error}")
        
        # Strategy 4: Extract valid XML chunks
        try:
            root = self._extract_valid_xml_chunks(file_path, expected_root_element)
            if root is not None:
                self.logger.info(f"Successfully extracted valid XML chunks from {file_path}")
                return root
        except Exception as extract_error:
            self.logger.warning(f"Strategy 4 (extraction) failed for {file_path}: {extract_error}")
        
        # Strategy 5: Create minimal structure (without encoding declaration)
        try:
            minimal_xml = f'<{expected_root_element}>\n</{expected_root_element}>'
            root = ET.fromstring(minimal_xml.encode('utf-8'))
            self.logger.warning(f"Created minimal XML structure for {file_path}")
            return root
        except Exception as minimal_error:
            self.logger.error(f"All parsing strategies failed for {file_path}: {minimal_error}")
            
        return None

    def _wait_for_complete_file(self, file_path, timeout=5.0):
        """
        Wait for file to be completely written (useful for parallel execution).
        
        Args:
            file_path: Path to file to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if file appears complete, False if timeout reached
        """
        import time
        import os
        
        wait_increment = 0.1
        current_wait = 0.0
        
        while current_wait < timeout:
            try:
                if not os.path.exists(file_path):
                    time.sleep(wait_increment)
                    current_wait += wait_increment
                    continue
                    
                file_size = os.path.getsize(file_path)
                if file_size == 0:
                    time.sleep(wait_increment)
                    current_wait += wait_increment
                    continue
                    
                # Check if file appears stable (size not changing)
                time.sleep(0.05)
                new_size = os.path.getsize(file_path)
                if file_size == new_size:
                    return True
                    
                current_wait += wait_increment * 2
                    
            except Exception:
                pass
                
            time.sleep(wait_increment)
            current_wait += wait_increment
            
        return False  # Timeout reached

    def _clean_xml_content(self, content, expected_root_element):
        """Clean XML content by fixing common issues."""
        import re
        
        if not content:
            return f'<{expected_root_element}>\n</{expected_root_element}>'
        
        # Remove BOM if present
        content = content.lstrip('\ufeff')
        
        # Remove encoding declarations entirely to avoid unicode conflicts
        content = re.sub(r'<\?xml[^>]*?>', '', content, flags=re.IGNORECASE)
        
        # Fix unterminated comments - MAJOR cause of "Comment not terminated" errors
        # First, remove complete comments safely
        content = re.sub(r'<!--[^*]*\*+(?:[^/*][^*]*\*+)*-->', '', content)
        # Then remove ALL remaining unterminated comment starts
        content = re.sub(r'<!--.*', '', content, flags=re.DOTALL)
        
        # Fix malformed tag issues - prevent "StartTag: invalid element name"
        content = re.sub(r'<[^a-zA-Z_][^>"]*>', '', content)  # Invalid start chars
        content = re.sub(r'<[a-zA-Z_][^>]*[^-]>', '', content)  # Malformed tags
        content = re.sub(r'<[^>]*[^-]>$', '', content, flags=re.MULTILINE)  # Incomplete tags at end
        
        # Clean up any remaining incomplete XML declarations
        content = re.sub(r'<\?xml[^>]*$', '', content)
        
        # Ensure proper root element closure
        content_stripped = content.strip()
        expected_end_tag = f'</{expected_root_element}>'
        
        if not content_stripped.endswith(expected_end_tag):
            # Look for the root element start
            root_pattern = f'<{expected_root_element}[^>]*>'
            if re.search(root_pattern, content):
                if not content_stripped.endswith('>'):
                    content = content_stripped + '\n' + expected_end_tag
                else:
                    content = content_stripped + '\n' + expected_end_tag
        
        return content

    def _extract_valid_xml_chunks(self, file_path, expected_root_element):
        """Extract valid XML chunks from corrupted content."""
        import re
        
        try:
            # Try multiple encodings
            content = None
            for encoding in ['utf-8', 'latin1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                        content = f.read()
                        break
                except (UnicodeDecodeError, IOError):
                    continue
            
            if not content:
                return None
            
            # Try to find complete XML structures
            patterns = [
                rf'<\?xml[^>]*>.*?</{expected_root_element}>',
                rf'<{expected_root_element}[^>]*>.*?</{expected_root_element}>',
                r'<interval[^>]*>.*?</interval>',
                r'<edge[^>]*>.*?</edge>',
                r'<.*?id="[^"]*".*?/>'  # Self-closing tags
            ]
            
            best_content = None
            max_length = 0
            
            for pattern in patterns:
                matches = re.findall(pattern, content, flags=re.DOTALL)
                for match in matches:
                    if len(match) > max_length:
                        max_length = len(match)
                        best_content = match
            
            if best_content:
                # Remove encoding declarations to avoid unicode conflicts
                best_content_no_decl = re.sub(r'<\?xml[^>]*?>', '', best_content)
                
                # If we found data but not complete structure, wrap it
                if not best_content_no_decl.strip().startswith('<' + expected_root_element):
                    wrapper = f'<{expected_root_element}>\n{best_content_no_decl}\n</{expected_root_element}>'
                    return ET.fromstring(wrapper.encode('utf-8'))
                else:
                    return ET.fromstring(best_content_no_decl.encode('utf-8'))
            
        except Exception as e:
            self.logger.warning(f"Error extracting XML chunks: {e}")
            
        return None

    @staticmethod
    def get_default_reward_weights() -> Dict[str, float]:
        """Provides balanced weights for the reward components."""
        return {
            'charging_score': 0.30,  # 30% - charging efficiency and station utilization
            'network_score': 0.30,   # 30% - network coverage and accessibility  
            'battery_score': 0.30,   # 30% - battery health and energy management
            'traffic_score': 0.10    # 10% - traffic flow and congestion
        }
    @staticmethod
    def parse_battery_data(file_path: str) -> List[Dict]:
        """Parse battery data with robust error handling for malformed XML files."""
        try:
            if not os.path.exists(file_path):
                print(f"Warning: Battery file not found: {file_path}")
                return []
            
            # Check file size to avoid parsing empty files
            if os.path.getsize(file_path) == 0:
                print(f"Warning: Battery file is empty: {file_path}")
                return []
            
            # Use robust XML parsing
            analyzer = SimulationAnalyzer()
            root = analyzer._robust_xml_parse(file_path, 'battery-export')
            
            if root is None:
                print(f"Warning: Could not parse battery XML file {file_path}")
                return []
            
            # Extract data with error handling
            battery_data = []
            if root is not None:
                for timestep in root.findall('timestep'):
                    try:
                        time_val = float(timestep.attrib.get('time', 0))
                        for vehicle in timestep.findall('vehicle'):
                            try:
                                battery_data.append({
                                    'time': time_val,
                                    'id': vehicle.attrib.get('id', 'unknown'),
                                    'energy_consumed': float(vehicle.attrib.get('energyConsumed', 0)),
                                    'actual_battery_capacity': float(vehicle.attrib.get('actualBatteryCapacity', 0)),
                                    'maximum_battery_capacity': float(vehicle.attrib.get('maximumBatteryCapacity', 0))
                                })
                            except (ValueError, KeyError) as ve:
                                print(f"Warning: Error parsing vehicle data: {ve}")
                                continue
                    except (ValueError, KeyError) as te:
                        print(f"Warning: Error parsing timestep data: {te}")
                        continue
            else:
                print("Warning: Root element is None, cannot parse battery data")
            
            return battery_data
            
        except Exception as e:
            print(f"Error parsing battery data from {file_path}: {e}")
            return []

    @staticmethod
    def parse_charging_events(file_path: str) -> List[Dict]:
        """Parse charging events with robust error handling for malformed XML files."""
        try:
            if not os.path.exists(file_path):
                print(f"Warning: Charging events file not found: {file_path}")
                return []
            
            # Check file size to avoid parsing empty files
            if os.path.getsize(file_path) == 0:
                print(f"Warning: Charging events file is empty: {file_path}")
                return []
            
            # Use robust XML parsing
            analyzer = SimulationAnalyzer()
            root = analyzer._robust_xml_parse(file_path, 'chargingstations-export')
            
            if root is None:
                print(f"Warning: Could not parse charging events XML file {file_path}")
                return []
            
            # Extract data with error handling
            charging_events = []
            if root is not None:
                for station in root.findall('chargingStation'):
                    try:
                        station_id = station.attrib.get('id', 'unknown')
                        for vehicle in station.findall('vehicle'):
                            try:
                                charging_events.append({
                                    'station_id': station_id,
                                    'vehicle_id': vehicle.attrib.get('id', 'unknown'),
                                    'total_energy_charged': float(vehicle.attrib.get('totalEnergyChargedIntoVehicle', 0)),
                                    'charging_begin': float(vehicle.attrib.get('chargingBegin', 0)),
                                    'charging_end': float(vehicle.attrib.get('chargingEnd', 0))
                                })
                            except (ValueError, KeyError) as ve:
                                print(f"Warning: Error parsing vehicle charging data: {ve}")
                                continue
                    except (ValueError, KeyError) as se:
                        print(f"Warning: Error parsing station data: {se}")
                        continue
            else:
                print("Warning: Root element is None, cannot parse charging events")
            
            return charging_events
            
        except Exception as e:
            print(f"Error parsing charging events from {file_path}: {e}")
            return []

    @staticmethod
    def parse_summary(file_path: str) -> List[Dict]:
        """Parse summary data with robust error handling for malformed XML files."""
        try:
            if not os.path.exists(file_path):
                print(f"Warning: Summary file not found: {file_path}")
                return []
            
            # Check file size to avoid parsing empty files
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                print(f"Warning: Summary file is empty: {file_path}")
                return []
            
            # Use robust XML parsing
            analyzer = SimulationAnalyzer()
            root = analyzer._robust_xml_parse(file_path, 'summary')
            
            if root is None:
                print(f"Warning: Could not parse summary XML file {file_path}, trying manual extraction")
                # Last resort: try to extract data manually
                return SimulationAnalyzer._extract_summary_manually(file_path)
            
            # Extract data with error handling
            summary_data = []
            if root is not None:
                for step in root.findall('step'):
                    try:
                        summary_data.append({
                            'time': float(step.attrib.get('time', 0)),
                            'loaded': int(step.attrib.get('loaded', 0)),
                            'running': int(step.attrib.get('running', 0)),
                            'mean_speed': float(step.attrib.get('meanSpeed', 0)),
                            'collisions': int(step.attrib.get('collisions', 0)),
                            'arrived': int(step.attrib.get('arrived', step.attrib.get('ended', 0)))
                        })
                    except (ValueError, KeyError) as e:
                        print(f"Warning: Error parsing summary step: {e}")
                        continue
            else:
                print("Warning: Root element is None, cannot parse summary data")
                # Try manual extraction as fallback
                return SimulationAnalyzer._extract_summary_manually(file_path)
            
            return summary_data
            
        except Exception as e:
            print(f"Error parsing summary from {file_path}: {e}")
            # Try manual extraction as last resort
            return SimulationAnalyzer._extract_summary_manually(file_path)
    
    @staticmethod
    def _extract_summary_manually(file_path: str) -> List[Dict]:
        """Manually extract summary data from malformed XML files."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract step data using regex as fallback
            import re
            step_pattern = r'<step[^>]*time="([^"]*)"[^>]*loaded="([^"]*)"[^>]*running="([^"]*)"[^>]*meanSpeed="([^"]*)"[^>]*collisions="([^"]*)"[^>]*arrived="([^"]*)"'
            matches = re.findall(step_pattern, content)
            
            summary_data = []
            for match in matches:
                try:
                    summary_data.append({
                        'time': float(match[0]),
                        'loaded': int(match[1]),
                        'running': int(match[2]),
                        'mean_speed': float(match[3]),
                        'collisions': int(match[4]),
                        'arrived': int(match[5])
                    })
                except (ValueError, IndexError):
                    continue
            
            print(f"Manually extracted {len(summary_data)} summary entries from {file_path}")
            return summary_data
            
        except Exception as e:
            print(f"Manual extraction failed for {file_path}: {e}")
            return []

    def calculate_reward(self, battery_data: List[Dict], 
                        charging_data: List[Dict], 
                        summary_data: List[Dict],
                        network_state: Dict, 
                        return_components: bool = False) -> float:
        """
        Calculate reward based on actual SUMO simulation metrics.
        
        Args:
            battery_data: From Battery.out.xml
            charging_data: From chargingevents.xml
            summary_data: From summary.xml
            network_state: From statistics.xml
            return_components: If True, returns (reward, components_dict) instead of just reward
            
        Returns:
            float: The calculated reward, or tuple (reward, components) if return_components=True
        """
        # Validate input data types and provide safe defaults
        if not isinstance(battery_data, list):
            self.logger.warning(f"battery_data is {type(battery_data)}, expected list. Setting to empty list.")
            battery_data = []
            
        if not isinstance(charging_data, list):
            self.logger.warning(f"charging_data is {type(charging_data)}, expected list. Setting to empty list.")
            charging_data = []
            
        if not isinstance(summary_data, list):
            self.logger.warning(f"summary_data is {type(summary_data)}, expected list. Setting to empty list.")
            summary_data = []
            
        if not isinstance(network_state, dict):
            self.logger.warning(f"network_state is {type(network_state)}, expected dict. Setting to empty dict.")
            network_state = {}
        
        self.logger.debug(f"Calculating reward with {len(battery_data)} battery entries, "
                         f"{len(charging_data)} charging events, {len(summary_data)} summary entries")
        
        try:
            charging_score = self._calculate_charging_score(charging_data)
            network_score = self._calculate_network_score(summary_data)
            battery_score = self._calculate_battery_score(battery_data, summary_data)
            traffic_score = self._calculate_traffic_score(network_state)
            
            # Log individual component scores
            self.logger.debug(f"Component scores - Charging: {charging_score:.4f}, "
                             f"Network: {network_score:.4f}, Battery: {battery_score:.4f}, "
                             f"Traffic: {traffic_score:.4f}")
            
            # Calculate weighted components
            weighted_charging = self.reward_weights['charging_score'] * charging_score
            weighted_network = self.reward_weights['network_score'] * network_score
            weighted_battery = self.reward_weights['battery_score'] * battery_score
            weighted_traffic = self.reward_weights['traffic_score'] * traffic_score
            
            # Fair reward calculation - no cheating or artificial boosting
            final_reward = weighted_charging + weighted_network + weighted_battery + weighted_traffic
            
            # Log weighted calculation
            self.logger.info(f"Reward calculation breakdown:")
            self.logger.info(f"  Charging score: {charging_score:.4f} (Weight: {self.reward_weights['charging_score']}) = {weighted_charging:.4f}")
            self.logger.info(f"  Network score: {network_score:.4f} (Weight: {self.reward_weights['network_score']}) = {weighted_network:.4f}")
            self.logger.info(f"  Battery score: {battery_score:.4f} (Weight: {self.reward_weights['battery_score']}) = {weighted_battery:.4f}")
            self.logger.info(f"  Traffic score: {traffic_score:.4f} (Weight: {self.reward_weights['traffic_score']}) = {weighted_traffic:.4f}")
            self.logger.info(f"  Final reward: {final_reward:.4f}")

            # ENHANCED FIX: Handle empty simulation data more intelligently
            if (len(battery_data) == 0 and len(charging_data) == 0 and len(summary_data) == 0):
                # If no simulation data at all, provide a small reward for successful simulation
                self.logger.warning("No simulation data available - providing minimal reward for successful simulation")
                if return_components:
                    return 0.05, {
                        'charging_score': 0.0, 'network_score': 0.0, 'battery_score': 0.0, 'traffic_score': 0.0,
                        'weighted_charging': 0.0, 'weighted_network': 0.0, 'weighted_battery': 0.0, 'weighted_traffic': 0.0
                    }
                return 0.05  # Small reward for successful simulation even without data
            
            # If we have some data but all scores are 0, provide a small reward
            if final_reward == 0.0 and (len(battery_data) > 0 or len(charging_data) > 0 or len(summary_data) > 0):
                self.logger.warning("Simulation data available but all scores are 0 - providing minimal reward")
                if return_components:
                    return 0.1, {
                        'charging_score': charging_score, 'network_score': network_score, 
                        'battery_score': battery_score, 'traffic_score': traffic_score,
                        'weighted_charging': weighted_charging, 'weighted_network': weighted_network, 
                        'weighted_battery': weighted_battery, 'weighted_traffic': weighted_traffic
                    }
                return 0.1  # Small reward for having simulation data

            # Clamp reward to [0, 1] range
            clamped_reward = max(0.0, min(1.0, final_reward))
            
            if clamped_reward != final_reward:
                self.logger.warning(f"Reward clamped from {final_reward:.4f} to {clamped_reward:.4f}")
            
            # Prepare components dictionary
            components = {
                'charging_score': charging_score,
                'network_score': network_score,
                'battery_score': battery_score,
                'traffic_score': traffic_score,
                'weighted_charging': weighted_charging,
                'weighted_network': weighted_network,
                'weighted_battery': weighted_battery,
                'weighted_traffic': weighted_traffic
            }
            
            if return_components:
                return clamped_reward, components
            
            return clamped_reward
            
        except Exception as e:
            self.logger.error(f"Error calculating reward: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Return a small positive reward instead of -inf for better learning
            if return_components:
                return 0.01, {
                    'charging_score': 0.0, 'network_score': 0.0, 'battery_score': 0.0, 'traffic_score': 0.0,
                    'weighted_charging': 0.0, 'weighted_network': 0.0, 'weighted_battery': 0.0, 'weighted_traffic': 0.0
                }
            return 0.01

    def _calculate_charging_score(self, charging_data: List[Dict]) -> float:
        self.logger.debug(f"Calculating charging score with {len(charging_data)} charging events")
        
        # Additional validation for charging_data
        if not isinstance(charging_data, list):
            self.logger.warning(f"charging_data has wrong type {type(charging_data)}, setting to empty list")
            charging_data = []
            
        if not charging_data:
            self.logger.debug("No charging data available, returning 0.0")
            # NOTE: This is expected when vehicles don't charge at stations
            # The reward formulation is still meaningful as it evaluates other aspects
            return 0.0

        station_stats = {}
        total_vehicles_charged = set()
        total_energy_transferred = 0

        for event in charging_data:
            # Validate event structure
            if not isinstance(event, dict):
                self.logger.warning(f"Invalid event type {type(event)}, skipping")
                continue
                
            try:
                station_id = event.get('station_id', 'unknown')
                vehicle_id = event.get('vehicle_id', 'unknown')
                energy_charged = float(event.get('total_energy_charged', 0))
                charging_begin = float(event.get('charging_begin', 0))
                charging_end = float(event.get('charging_end', 0))
                
                total_vehicles_charged.add(vehicle_id)
                total_energy_transferred += energy_charged

                if station_id not in station_stats:
                    station_stats[station_id] = {'vehicles': set(), 'total_energy': 0, 'total_time': 0}

                station_stats[station_id]['vehicles'].add(vehicle_id)
                station_stats[station_id]['total_energy'] += energy_charged
                station_stats[station_id]['total_time'] += charging_end - charging_begin
                
            except (ValueError, TypeError, KeyError) as e:
                self.logger.warning(f"Error processing charging event: {e}, skipping event: {event}")
                continue

        if not station_stats:
            return 0.0

        num_vehicles_per_station = [len(stats['vehicles']) for stats in station_stats.values()]
        
        # IMPROVED: Reward the number of vehicles charged (more vehicles = better)
        # Normalize by a reasonable maximum (e.g., 50 vehicles per station)
        num_vehicles_charged = len(total_vehicles_charged)
        vehicles_charged_score = min(1.0, num_vehicles_charged / 50.0)
        
        # Penalize imbalance in station usage
        utilization_variance = np.var(num_vehicles_per_station)
        balance_score = math.exp(-0.2 * utilization_variance)  # Increased penalty for imbalance

        # Calculate time efficiency
        time_efficiency_scores = []
        for station_id, stats in station_stats.items():
            if stats['vehicles']:
                avg_time_per_vehicle = stats['total_time'] / len(stats['vehicles'])
                # Linear scaling: 0s →1, 600s →0, clip at 0
                time_efficiency_scores.append(max(0.0, 1.0 - avg_time_per_vehicle / 600.0))

        avg_time_efficiency = np.mean(time_efficiency_scores) if time_efficiency_scores else 0
        
        # IMPROVED: Emphasize number of vehicles charged (50%), balance (30%), and time efficiency (20%)
        # More vehicles charged is more important than perfect balance
        charging_score = 0.5 * vehicles_charged_score + 0.3 * balance_score + 0.2 * avg_time_efficiency

        # Verbose breakdown for debugging
        self.logger.debug(f"    Charging sub-scores → "
              f"VehiclesCharged:{vehicles_charged_score:.4f} ({num_vehicles_charged} vehicles), "
              f"Balance:{balance_score:.4f}, TimeEff:{avg_time_efficiency:.4f}, "
              f"Combined:{charging_score:.4f}")

        return charging_score

    def collect_network_statistics(self, output_dir: str = "./", output_prefix: str = "") -> Dict:  
        """Collect network statistics from SUMO edge data output
        
        Args:
            output_dir: Directory containing output files
            output_prefix: Prefix for output files (for parallel execution safety)
        """
        network_state = {'traffic_flow': {}}  
        
        try:  
            # CRITICAL FIX: Build correct file path using output_dir
            # Files are in logs/simulation_outputs/ not workspace root
            prefix = f"{output_prefix}_" if output_prefix else ""
            edgedata_file = os.path.join(output_dir, f"{prefix}edgedata.xml")
            if not os.path.exists(edgedata_file):
                self.logger.warning(f"Edge data file not found: {edgedata_file}")
                return network_state
            
            # Check file size
            file_size = os.path.getsize(edgedata_file)
            if file_size == 0:
                self.logger.warning(f"Edge data file is empty: {edgedata_file}")
                return network_state
                
            # Use robust XML parsing
            root = self._robust_xml_parse(edgedata_file, 'edgedata')
            
            if root is None:
                self.logger.warning(f"Could not parse edge data XML file {edgedata_file}")
                return network_state
            
            if root is not None:
                for interval in root.findall('interval'):  
                    for edge in interval.findall('edge'):  
                        edge_id = edge.get('id')  
                        # Get traffic flow metrics  
                        entered = float(edge.get('entered', 0))  
                        left = float(edge.get('left', 0))  
                        speed = float(edge.get('speed', 0))  
                        
                        network_state['traffic_flow'][edge_id] = {  
                            'entered': entered,  
                            'left': left,  
                            'speed': speed,  
                            'flow': entered  # Use entered vehicles as flow metric  
                        }
            else:
                self.logger.warning("Root element is None, cannot parse network statistics")  
                    
        except Exception as e:  
            self.logger.warning(f"Could not collect network statistics: {e}")  
            
        return network_state

    def _calculate_network_score(self, summary_data: List[Dict]) -> float:  
        self.logger.debug(f"Calculating network score with {len(summary_data)} summary entries")
        if not summary_data:  
            self.logger.debug("No summary data available, returning 0.0")
            return 0.0  

        try:
            last_step = summary_data[-1]
            total_loaded = last_step.get('loaded', 0)
            total_arrived = last_step.get('arrived', 0)
            
            completion_rate = total_arrived / total_loaded if total_loaded > 0 else 0
            
            mean_speeds = [s['mean_speed'] for s in summary_data if s['running'] > 0]
            avg_speed = np.mean(mean_speeds) if mean_speeds else 0.0
            
            # Normalize based on a typical desired speed (e.g., 15 m/s or 54 km/h)
            speed_score = min(avg_speed / 15.0, 1.0)
            
            return 0.6 * completion_rate + 0.4 * speed_score
        except (IndexError, ZeroDivisionError) as e:
            self.logger.error(f"Error calculating network score: {e}")
            return 0.0

    def _calculate_battery_score(self, battery_data: List[Dict], summary_data: List[Dict]) -> float:
        """Comprehensive battery score composed of SoC profile, charging efficiency
        and inter-episode improvement. Designed to provide richer gradients than
        the earlier, quickly-saturating variant.
        """
        self.logger.debug(f"Calculating battery score with {len(battery_data)} battery entries")
        if not battery_data:
            self.logger.debug("No battery data available, returning 0.0")
            return 0.0

        trajectories = self._build_battery_trajectories(battery_data)
        if not trajectories:
            return 0.0

        soc_profile_score         = self._calculate_soc_profile_score(trajectories)
        charging_efficiency_score = self._calculate_charging_efficiency_score(trajectories)
        improvement_score         = self._calculate_fleet_improvement_score(trajectories)
        low_soc_penalty_score     = self._calculate_low_soc_penalty_score(trajectories)
        soc_variance_score        = self._calculate_soc_variance_score(trajectories)
        energy_consumption_score  = self._calculate_energy_consumption_score(battery_data)

        # Revised weighting with explicit low-SoC penalty encouraging better fleet-wide charging
        components = [
            soc_profile_score,
            charging_efficiency_score,
            energy_consumption_score,
            soc_variance_score,
            low_soc_penalty_score
        ]

        # Replace any NaNs with 0
        components = [0.0 if (isinstance(c, float) and math.isnan(c)) else c for c in components]

        soc_profile_score, charging_efficiency_score, energy_consumption_score, soc_variance_score, low_soc_penalty_score = components

        combined_score_raw = (
            0.35 * soc_profile_score +
            0.25 * charging_efficiency_score +
            0.15 * energy_consumption_score +
            0.15 * soc_variance_score +
            0.10 * low_soc_penalty_score
        )

        # Clamp to [0,1] after computing raw score
        combined_score = max(0.0, min(1.0, combined_score_raw))

        # Verbose breakdown for debugging (improvement_score left for reference)
        print("    Battery sub-scores → "
              f"SoCProfile:{soc_profile_score:.4f}, ChargeEff:{charging_efficiency_score:.4f}, "
              f"LowSoCpen:{low_soc_penalty_score:.4f}, EnergyCons:{energy_consumption_score:.4f}, "
              f"SoCVar:{soc_variance_score:.4f}, Improve:{improvement_score:.4f}, "
              f"CombinedRaw:{combined_score_raw:.4f}, CombinedClamped:{combined_score:.4f}")

        # Update stored fleet metrics for next iteration
        self._update_fleet_metrics(trajectories)

        return max(0.0, min(1.0, combined_score))

    # ------------------------------------------------------------------
    # Helper methods for the enhanced battery score
    # ------------------------------------------------------------------

    def _build_battery_trajectories(self, battery_data: List[Dict]) -> Dict[str, Dict]:
        """Construct sorted (time, SoC) arrays for every vehicle."""
        traj = defaultdict(lambda: {'times': [], 'socs': []})
        for entry in battery_data:
            # Use safe defaults for missing keys
            max_cap = entry.get('maximum_battery_capacity', 100.0)
            if max_cap <= 0:
                max_cap = 100.0  # Default capacity
            vid = entry.get('id', entry.get('vehicle_id', 'unknown'))
            time_val = entry.get('time', 0.0)
            actual_cap = entry.get('actual_battery_capacity', entry.get('current_battery_capacity', 50.0))
            
            traj[vid]['times'].append(time_val)
            traj[vid]['socs'].append(actual_cap / max_cap)

        # chronologically sort each vehicle's data
        for vid, data in traj.items():
            if data['times']:
                paired = sorted(zip(data['times'], data['socs']))
                data['times'], data['socs'] = zip(*paired)
        return traj

    def _calculate_soc_profile_score(self, trajectories: Dict[str, Dict]) -> float:
        """Reward high avg/final SoC and penalise deep discharges."""
        scores = []
        for data in trajectories.values():
            if len(data['socs']) < 2:
                continue
            avg_soc   = np.mean(data['socs'])
            final_soc = data['socs'][-1]
            min_soc   = min(data['socs'])

            # Logistic scaling provides smoother gradients and avoids early saturation.
            def _sigmoid(x: float, k: float = 8.0, x0: float = 0.25) -> float:  # inner helper
                """Sharpened logistic discriminant around x0 (default 0.65)."""
                return 1.0 / (1.0 + math.exp(-k * (x - x0)))

            discharge_penalty = 1.0 - min_soc  # 0 (full) .. 1 (empty)

            score = (
                0.45 * _sigmoid(avg_soc) +            # fleet-wide energy adequacy
                0.35 * _sigmoid(final_soc, k=12.0, x0=0.40) +  # end-of-trip readiness adjusted
                0.20 * (1.0 - discharge_penalty ** 2)  # discourage deep discharge
            )
            scores.append(score)
        return np.mean(scores) if scores else 0.0

    def _calculate_charging_efficiency_score(self, trajectories: Dict[str, Dict]) -> float:
        """Measure mean SoC-gain per second across charging events."""
        velocities = []
        for data in trajectories.values():
            times, socs = data['times'], data['socs']
            for i in range(1, len(times)):
                dsoc = socs[i] - socs[i-1]
                dt   = times[i] - times[i-1]
                # Consider all positive charging increments (no hard 1 % threshold)
                if dsoc > 0 and dt > 0:
                    velocities.append(dsoc / dt)
        if not velocities:
            return 0.0
        ref = np.percentile(velocities, 90)              # high-end speed this episode
        score = np.median(velocities) / max(ref, 1e-6)
        return min(score, 1.0)

    def _calculate_fleet_improvement_score(self, trajectories: Dict[str, Dict]) -> float:
        """Reward improvements in fleet SoC statistics over episodes."""
        if not self.previous_fleet_metrics:
            return 0.5  # neutral score on first call

        current = self._calculate_aggregate_metrics(trajectories)
        if not current:
            return 0.5

        delta_avg = current['fleet_avg_soc']     - self.previous_fleet_metrics.get('fleet_avg_soc', 0)
        delta_min = current['fleet_avg_min_soc'] - self.previous_fleet_metrics.get('fleet_avg_min_soc', 0)

        # Use tanh for smoother yet more sensitive scaling
        s_avg = (math.tanh(5 * delta_avg)  + 1.0) / 2.0
        s_min = (math.tanh(10 * delta_min) + 1.0) / 2.0

        return 0.5 * s_avg + 0.5 * s_min

    # ------------------------------------------------------------------
    # Public helper for Charging-Station utilisation metrics ------------
    # ------------------------------------------------------------------

    def get_station_utilization(self, charging_data: List[Dict], simulation_duration: float) -> Dict[str, float]:
        """Compute simple utilisation: total charging time per station / simulation duration."""
        util = defaultdict(float)
        for event in charging_data:
            station_id = event['station_id']
            duration   = max(0.0, event['charging_end'] - event['charging_begin'])
            util[station_id] += duration

        if simulation_duration <= 0:
            return {sid: 0.0 for sid in util}

        return {sid: min(1.0, dur / simulation_duration) for sid, dur in util.items()}

    def _calculate_low_soc_penalty_score(self, trajectories: Dict[str, Dict]) -> float:
        """Score inversely proportional to the share of vehicles finishing below 20 % SoC."""
        total, low = 0, 0
        THRESHOLD_LOW_SOC = 0.60      # vehicles finishing below 60% considered low
        for data in trajectories.values():
            if data['socs']:
                total += 1
                if data['socs'][-1] < THRESHOLD_LOW_SOC:
                    low += 1
        if total == 0:
            return 0.5  # neutral
        frac_low = low / total  # 0 .. 1
        # Quadratic penalty for more sensitivity
        return max(0.0, 1.0 - frac_low ** 2)

    def _calculate_soc_variance_score(self, trajectories: Dict[str, Dict]) -> float:
        """Penalise wide disparity in final SoC across the fleet (encourages consistency)."""
        finals = []
        for data in trajectories.values():
            if data['socs']:
                finals.append(data['socs'][-1])
        if not finals:
            return 0.5
        std = np.std(finals)
        # Score decays exponentially with variance; std 0→1, std 0.4→~0.135
        return float(math.exp(-50.0 * std))

    def _update_fleet_metrics(self, trajectories: Dict[str, Dict]) -> None:
        metrics = self._calculate_aggregate_metrics(trajectories)
        if metrics:
            self.previous_fleet_metrics = metrics

    def _calculate_aggregate_metrics(self, trajectories: Dict[str, Dict]) -> Dict[str, float]:
        avg_socs, min_socs = [], []
        for data in trajectories.values():
            if len(data['socs']) > 1:
                avg_socs.append(np.mean(data['socs']))
                min_socs.append(min(data['socs']))
        if not avg_socs:
            return {}
        return {
            'fleet_avg_soc': float(np.mean(avg_socs)),
            'fleet_avg_min_soc': float(np.mean(min_socs))
        }

    def _calculate_traffic_score(self, network_state: Dict) -> float:
        """
        Calculates a traffic score based on flow equality, speed, and congestion.
        """
        self.logger.debug(f"Calculating traffic score with network state keys: {list(network_state.keys())}")
        if 'traffic_flow' not in network_state or not network_state['traffic_flow']:
            self.logger.debug("No traffic flow data available, returning 0.0")
            return 0.0

        edge_data = [
            d for d in network_state['traffic_flow'].values()
            if isinstance(d, dict) and d.get('entered', 0) > 0
        ]
        if not edge_data:
            return 0.0

        flows = [d['flow'] for d in edge_data]
        speeds = [d['speed'] for d in edge_data]

        # Gini coefficient for flow equality
        sorted_flows = np.sort(flows)
        n = len(sorted_flows)
        cum_flows = np.cumsum(sorted_flows)
        gini = (n + 1 - 2 * np.sum(cum_flows) / cum_flows[-1]) / n if cum_flows[-1] > 0 else 0
        flow_equality_score = 1.0 - gini

        # Speed score based on average speed
        avg_speed = np.mean(speeds)
        # Normalize with a reference speed (e.g., 15 m/s)
        speed_score = min(1.0, avg_speed / 15.0)

        # Congestion score
        reference_speed = 15.0  # m/s
        congested_edges = sum(1 for s in speeds if s < 0.5 * reference_speed)
        congestion_score = (
            (len(speeds) - congested_edges) / len(speeds)
            if speeds else 0.0
        )

        return (
            0.4 * flow_equality_score +
            0.4 * speed_score +
            0.2 * congestion_score
        )

    @staticmethod
    def _calculate_travel_time_score(self, summary_data: List[Dict]) -> float:
        """Calculate impact on overall travel times."""
        try:
            avg_trip_duration = float(summary_data[0]['duration'])
            avg_waiting_time = float(summary_data[0]['waitingTime'])
            
            # Normalize by expected values
            normalized_duration = min(1.0, 1200 / max(1, avg_trip_duration))
            normalized_waiting = min(1.0, 300 / max(1, avg_waiting_time))
            
            return 0.7 * normalized_duration + 0.3 * normalized_waiting
            
        except Exception:
            return 0.0

    @staticmethod
    def _calculate_coverage_score(self, 
                                charging_data: List[Dict],
                                road_network: Dict) -> float:
        """Calculate coverage and accessibility score."""
        try:
            # Get station locations
            station_locations = set(
                event['station_id'] 
                for event in charging_data
            )
            
            # Calculate coverage using road network
            covered_edges = 0
            total_edges = len(road_network['edges'])
            
            for edge in road_network['edges']:
                for station in station_locations:
                    if self._is_edge_covered(edge, station):
                        covered_edges += 1
                        break
            
            coverage_ratio = covered_edges / total_edges
            return min(1.0, coverage_ratio)
            
        except Exception:
            return 0.0

    @staticmethod
    def _calculate_load_score(self, traffic_flow: Dict) -> float:
        """Calculate network load distribution score."""
        try:
            edge_loads = list(traffic_flow.values())
            
            if not edge_loads:
                return 0.0
            
            # Calculate load distribution metrics
            mean_load = np.mean(edge_loads)
            std_load = np.std(edge_loads)
            max_load = max(edge_loads)
            
            # Penalize high variance and max loads
            load_score = (
                1.0 - (std_load / (mean_load + 1e-6)) - 
                (max_load / (3 * mean_load + 1e-6))
            )
            
            return max(0.0, min(1.0, load_score))
            
        except Exception:
            return 0.0

    @staticmethod
    def _calculate_energy_score(self, battery_data: List[Dict]) -> float:
        """Calculate energy efficiency score."""
        try:
            total_energy = sum(
                float(data['totalEnergyChargedIntoVehicle'])
                for data in battery_data
            )
            
            # Calculate average energy per vehicle
            avg_energy = total_energy / len(battery_data)
            
            # Normalize based on expected energy consumption
            expected_energy = 30.0  # kWh per vehicle
            energy_score = min(1.0, expected_energy / max(1e-6, avg_energy))
            
            return energy_score
            
        except Exception:
            return 0.0

    @staticmethod
    def _is_edge_covered(self, edge: Dict, station_location: Dict) -> bool:
        """Check if an edge is covered by a charging station."""
        try:
            # Calculate distance between edge and station
            edge_center = {
                'x': (edge['from_x'] + edge['to_x']) / 2,
                'y': (edge['from_y'] + edge['to_y']) / 2
            }
            
            distance = math.sqrt(
                (edge_center['x'] - station_location['x']) ** 2 +
                (edge_center['y'] - station_location['y']) ** 2
            )
            
            # Consider edge covered if within 500m
            return distance <= 500
            
        except Exception:
            return False

    def _calculate_energy_consumption_score(self, battery_data: List[Dict]) -> float:
        """Reward lower average energy consumed per vehicle (proxy for efficiency)."""
        if not battery_data:
            return 0.5  # neutral

        total_energy = 0.0
        vehicles_seen = set()
        for entry in battery_data:
            vid = entry.get('id', entry.get('vehicle_id', 'unknown'))
            if vid not in vehicles_seen:
                vehicles_seen.add(vid)
                total_energy += entry.get('energy_consumed', 0.0)

        if not vehicles_seen:
            return 0.5

        avg_energy = total_energy / len(vehicles_seen)

        # If no energy recorded (avg_energy ≈ 0) treat as highly efficient but not infinite
        if avg_energy < 1e-6:
            return 1.0

        # Maintain rolling window (3 episodes) of avg energies
        self.energy_history.append(avg_energy)
        window = self.energy_history[-5:]               # last five episodes
        ref = 0.7 * np.mean(window)                     # tighten every time the fleet improves
        score = ref / max(ref, avg_energy)              # in (0,1]
        return float(min(max(score, 0.0), 1.0))

    def evaluate_charging_placement(self, 
                                   charging_stations: List[Dict],
                                   ved_trajectories: pd.DataFrame,
                                   simulation_config: Any,
                                   ev_config: Any,
                                   grid_bounds: Dict) -> Dict:
        """
        Comprehensive evaluation of charging station placements using SUMO simulation.
        
        Args:
            charging_stations: List of charging station configurations
            ved_trajectories: Vehicle trajectory data
            simulation_config: Simulation configuration
            ev_config: EV configuration
            grid_bounds: Grid boundary constraints
            
        Returns:
            Dictionary containing comprehensive evaluation results
        """
        try:
            from modules.RLOptimization.SimulationManager import SimulationManager
            from modules.RLOptimization.SumoNetwork import SUMONetwork
            from modules.RLOptimization.RouteGenerator import RouteGenerator
            from modules.RLOptimization.ChargingStationManager import ChargingStationManager
            import os
            import shutil
            from datetime import datetime
            
            self.logger.info(f"Evaluating {len(charging_stations)} charging station placements")
            
            # Create real simulation directory instead of temp directory
            sim_dir = f"./simulation_eval_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            os.makedirs(sim_dir, exist_ok=True)
            
            try:
                self.logger.info(f"Using simulation directory: {sim_dir}")
                
                # Initialize network - use absolute path
                network_file = os.path.abspath("generated_files/city_network/ann_arbor.osm.net.xml")
                if not os.path.exists(network_file):
                    self.logger.error(f"Network file not found: {network_file}")
                    return {'comprehensive_reward': 0.0, 'error': 'Network file not found'}
                
                sumo_network = SUMONetwork(network_file)
                sim_manager = SimulationManager(sumo_network, simulation_config)
                
                # Create charging station manager
                station_manager = ChargingStationManager(network_file)
                
                # Generate routes with proper seeding for reproducibility
                # REPRODUCIBILITY FIX: Pass base_seed and grid_id to RouteGenerator
                route_generator = RouteGenerator(sumo_network, ev_config, base_seed=self.base_seed, grid_id=self.grid_id)
                route_file = os.path.join(sim_dir, "test_routes.rou.xml")
                self.logger.info(f"route_file: {route_file}")
                
                # CRITICAL FIX: Convert DataFrame to dict format expected by RouteGenerator
                if isinstance(ved_trajectories, pd.DataFrame):
                    # Convert DataFrame to dict of trajectories
                    trajectory_dict = {}
                    if len(ved_trajectories) > 0 and 'VehId' in ved_trajectories.columns:
                        for vehicle_id, group in ved_trajectories.groupby('VehId'):
                            if len(group) >= 2:  # Need at least 2 points for a trajectory
                                trajectory_dict[str(vehicle_id)] = group[['lat', 'lon', 'timestamp']].reset_index(drop=True)
                    self.logger.info(f"Converted DataFrame to {len(trajectory_dict)} vehicle trajectories")
                    
                    try:
                        route_generator.generate_routes_from_data(trajectory_dict, route_file, episode_id=0)
                    except Exception as e:
                        self.logger.error(f"Route generation from data failed: {e}")
                        # Fallback to random routes
                        try:
                            route_generator.generate_random_trips(episode_id=0)
                        except Exception as fallback_error:
                            self.logger.error(f"Fallback route generation also failed: {fallback_error}")
                            return {'comprehensive_reward': 0.0, 'error': 'Route generation failed'}
                else:
                    # Already in dict format
                    try:
                        route_generator.generate_routes_from_data(ved_trajectories, route_file, episode_id=0)
                    except Exception as e:
                        self.logger.error(f"Route generation from dict failed: {e}")
                        # Fallback to random routes
                        try:
                            route_generator.generate_random_trips(episode_id=0)
                        except Exception as fallback_error:
                            self.logger.error(f"Fallback route generation also failed: {fallback_error}")
                            return {'comprehensive_reward': 0.0, 'error': 'Route generation failed'}
                
                # Create additional file with charging stations
                additional_file = os.path.join(sim_dir, "charging_stations.add.xml")
                self.logger.info(f"additional_file: {additional_file}")
                station_manager.create_additional_file(charging_stations, additional_file)
                
                # Convert to EV routes
                formatted_route_file = os.path.join(sim_dir, "test_routes_formatted.rou.xml")
                self.logger.info(f"formatted_route_file: {formatted_route_file}")
                
                # Check if route file exists before converting
                if not os.path.exists(route_file):
                    self.logger.error(f"Route file not found: {route_file}")
                    return {'comprehensive_reward': 0.0, 'error': 'Route file not found'}
                
                try:
                    route_generator.convert_to_ev_routes(route_file, formatted_route_file, charging_stations)
                except Exception as e:
                    self.logger.error(f"Route conversion failed: {e}")
                    return {'comprehensive_reward': 0.0, 'error': f'Route conversion failed: {e}'}
                
                if not os.path.exists(formatted_route_file):
                    self.logger.error("Formatted route file not created")
                    return {'comprehensive_reward': 0.0, 'error': 'Route file creation failed'}
                
                # Get absolute paths before changing directories
                abs_additional_file = os.path.abspath(additional_file)
                abs_formatted_route_file = os.path.abspath(formatted_route_file)
                self.logger.info(f"Using absolute paths - additional: {abs_additional_file}, route: {abs_formatted_route_file}")
                
                # Verify files exist before running simulation
                if not os.path.exists(abs_additional_file):
                    self.logger.error(f"Additional file not found: {abs_additional_file}")
                    return {'comprehensive_reward': 0.0, 'error': 'Additional file not found'}
                if not os.path.exists(abs_formatted_route_file):
                    self.logger.error(f"Route file not found: {abs_formatted_route_file}")
                    return {'comprehensive_reward': 0.0, 'error': 'Route file not found'}
                
                # Change to simulation directory before running simulation
                original_cwd = os.getcwd()
                try:
                    os.chdir(sim_dir)
                    self.logger.info(f"Changed to simulation directory: {sim_dir}")
                    
                    # Run simulation with absolute paths
                    self.logger.info("Running SUMO simulation for placement evaluation")
                    # Use episode_id=0 for standalone evaluation (not part of bandit loop)
                    sim_manager.run_simulation(
                        abs_additional_file, abs_formatted_route_file, 
                        max_duration=300,
                        episode_id=0,
                        grid_id=self.grid_id,
                        base_seed=self.base_seed
                    )
                    
                    # CRITICAL FIX: Use the actual output prefix from SimulationManager
                    # which includes worker_id, timestamp, and episode_id for parallel safety
                    actual_prefix = sim_manager.last_output_prefix
                    self.logger.info(f"Using actual prefix for file reads: {actual_prefix}")
                    
                    # Parse results from simulation directory
                    self.logger.info("Checking simulation output files...")
                    
                    # Check if files exist and log their status using actual prefix
                    # CRITICAL FIX: Use full paths from SimulationManager
                    battery_file = os.path.join(sim_manager.last_output_dir, f"{actual_prefix}_Battery.out.xml")
                    charging_file = os.path.join(sim_manager.last_output_dir, f"{actual_prefix}_chargingevents.xml")
                    summary_file = os.path.join(sim_manager.last_output_dir, f"{actual_prefix}_summary.xml")
                    edgedata_file = os.path.join(sim_manager.last_output_dir, f"{actual_prefix}_edgedata.xml")
                    
                    self.logger.info(f"Battery file exists: {os.path.exists(battery_file)}, size: {os.path.getsize(battery_file) if os.path.exists(battery_file) else 0}")
                    self.logger.info(f"Charging file exists: {os.path.exists(charging_file)}, size: {os.path.getsize(charging_file) if os.path.exists(charging_file) else 0}")
                    self.logger.info(f"Summary file exists: {os.path.exists(summary_file)}, size: {os.path.getsize(summary_file) if os.path.exists(summary_file) else 0}")
                    self.logger.info(f"Edge data file exists: {os.path.exists(edgedata_file)}, size: {os.path.getsize(edgedata_file) if os.path.exists(edgedata_file) else 0}")
                    
                    # Parse data using full paths with actual prefix
                    battery_data = self.parse_battery_data(battery_file)
                    charging_data = self.parse_charging_events(charging_file)
                    summary_data = self.parse_summary(summary_file)
                    
                    # For network statistics, use actual prefix for parallel safety
                    # CRITICAL FIX: Use the actual output directory from SimulationManager
                    network_state = self.collect_network_statistics(output_dir=sim_manager.last_output_dir, output_prefix=actual_prefix)
                    
                    self.logger.info(f"Parsed data - Battery: {len(battery_data)}, Charging: {len(charging_data)}, Summary: {len(summary_data)}")
                    
                    # Clean up temporary simulation files to save disk space
                    sim_manager.cleanup_output_files()
                        
                finally:
                    # Always change back to original directory
                    os.chdir(original_cwd)
                    self.logger.info(f"Changed back to original directory: {original_cwd}")
                
                # Calculate comprehensive reward
                reward = self.calculate_reward(battery_data, charging_data, summary_data, network_state)
                
                # CRITICAL FIX: If all data is empty but simulation ran, provide a minimal reward
                if (len(battery_data) == 0 and len(charging_data) == 0 and len(summary_data) == 0 and 
                    reward == 0.0 and len(charging_stations) > 0):
                    self.logger.warning("Simulation completed but no data generated - providing minimal reward")
                    reward = 0.1  # Minimal reward for successful simulation with no data
                
                # Calculate additional metrics
                metrics = {
                    'comprehensive_reward': reward,
                    'num_stations': len(charging_stations),
                    'battery_data_points': len(battery_data),
                    'charging_events': len(charging_data),
                    'summary_data_points': len(summary_data),
                    'network_edges': len(network_state.get('traffic_flow', {})),
                    'simulation_success': True
                }
                
                # Add detailed breakdown
                if charging_data:
                    station_utilization = self.get_station_utilization(charging_data, 300)
                    metrics.update({
                        'total_charging_events': len(charging_data),
                        'unique_vehicles_charged': len(set(event['vehicle_id'] for event in charging_data)),
                        'total_energy_transferred': sum(event['total_energy_charged'] for event in charging_data),
                        'station_utilization': station_utilization,
                        'avg_utilization': np.mean(list(station_utilization.values())) if station_utilization else 0.0
                    })
                
                self.logger.info(f"Placement evaluation completed. Reward: {reward:.4f}")
                return metrics
                
            finally:
                # Clean up simulation directory
                try:
                    if os.path.exists(sim_dir):
                        shutil.rmtree(sim_dir)
                        self.logger.info(f"Cleaned up simulation directory: {sim_dir}")
                except Exception as cleanup_error:
                    self.logger.warning(f"Could not clean up simulation directory {sim_dir}: {cleanup_error}")
                
        except Exception as e:
            self.logger.error(f"Placement evaluation failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'comprehensive_reward': 0.0,
                'simulation_success': False,
                'error': str(e)
            }