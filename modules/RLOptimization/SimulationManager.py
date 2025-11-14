from modules.RLOptimization.SumoNetwork import SUMONetwork
from modules.config import SimulationConfig
import traci
import random
import signal
import time
import logging
import sys
import os
from datetime import datetime

class SimulationManager:
    """Manages SUMO simulation execution and data collection"""
    
    def _setup_logging(self):
        """Setup comprehensive logging for simulation management."""
        # Create logger with hierarchical name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Set logging level based on environment
        log_level = os.environ.get('SIMULATION_LOG_LEVEL', 'INFO').upper()
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
        log_file = os.path.join(log_dir, f"simulation_manager_{datetime.now().strftime('%Y%m%d')}.log")
        
        if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == log_file for handler in self.logger.handlers):
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def __init__(self, network: SUMONetwork, config: SimulationConfig):
        # Setup logging first
        self._setup_logging()
        
        self.network = network
        self.config = config
        
        # PARALLEL EXECUTION FIX: Allocate unique port for TraCI connection
        # Use PID + timestamp to ensure uniqueness across parallel workers
        import socket
        self.worker_id = f"{os.getpid()}_{int(time.time() * 1000) % 100000}"
        self.traci_port = self._allocate_unique_port()
        
        self.logger.info("Initializing Simulation Manager")
        self.logger.info(f"  Worker ID: {self.worker_id}")
        self.logger.info(f"  TraCI Port: {self.traci_port}")
        self.logger.info(f"  Network: {network.net_file_path}")
        self.logger.info(f"  Config: {type(config).__name__}")
        self.logger.info(f"  GUI enabled: {config.use_gui}")
        self.logger.info(f"  Domain randomization: {getattr(config, 'enable_domain_randomization', False)}")
        
        # PARALLEL EXECUTION FIX: Clean up any stale processes from previous runs
        self.cleanup_stale_processes()
        
    def _allocate_unique_port(self, base_port=8813, max_attempts=100):
        """Allocate a unique port for TraCI connection to avoid conflicts in parallel execution."""
        import socket
        for attempt in range(max_attempts):
            port = base_port + (os.getpid() % 1000) + attempt
            # Check if port is available
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('localhost', port))
                    # Port is available
                    return port
                except OSError:
                    # Port in use, try next
                    continue
        # Fallback: use random port
        self.logger.warning(f"Could not find available port after {max_attempts} attempts, using random port")
        return base_port + random.randint(1000, 9999)
        
    def run_simulation(self, additional_file: str, route_file: str, max_duration: int = 0, output_prefix: str = "", episode_id: int = None, grid_id: str = None, base_seed: int = 42) -> None:  
        """Run SUMO simulation with given configuration and timeout
        
        Args:
            additional_file: Path to SUMO additional file (charging stations)
            route_file: Path to SUMO route file
            max_duration: Maximum simulation duration (0 = no limit)
            output_prefix: Prefix for output files to enable parallel execution (e.g., "cell_45_")
            episode_id: Episode number for deterministic domain randomization
            grid_id: Grid identifier for deterministic domain randomization
            base_seed: Base random seed for reproducibility
        """
        sumo_executable = "sumo-gui" if self.config.use_gui else "sumo"  
        
        # Add timeout controls to SUMO command - reduced duration for efficiency
        # Use absolute paths for all files to avoid issues when changing directories
        network_file = os.path.abspath(self.network.net_file_path)
        additional_file = os.path.abspath(additional_file)
        route_file = os.path.abspath(route_file)
        
        # ENHANCED WORKER ISOLATION: Create truly unique output filenames for parallel execution
        # Include process ID, timestamp, and episode to prevent any collisions
        worker_id = os.getpid()
        timestamp = int(time.time() * 1000) % 100000  # 5-digit timestamp
        
        # Create output directory for simulation files to avoid cluttering workspace root
        output_dir = os.path.abspath("logs/simulation_outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create comprehensive prefix with all identifiers - place in output directory
        if output_prefix:
            prefix = f"{output_prefix}_{worker_id}_{timestamp}_{episode_id or 0}_"
        elif grid_id:
            prefix = f"{grid_id}_{worker_id}_{timestamp}_{episode_id or 0}_"
        else:
            prefix = f"worker_{worker_id}_{timestamp}_{episode_id or 0}_"
        
        # Create full paths with output directory
        charging_output = os.path.join(output_dir, f"{prefix}chargingevents.xml")
        battery_output = os.path.join(output_dir, f"{prefix}Battery.out.xml")
        summary_output = os.path.join(output_dir, f"{prefix}summary.xml")
        edgedata_output = os.path.join(output_dir, f"{prefix}edgedata.xml")
        
        # Store the actual prefix used for later retrieval
        self.last_output_prefix = prefix.rstrip('_')
        self.last_output_dir = output_dir  # Store directory for analyzer
        
        # Store output file paths for cleanup
        self.last_output_files = [charging_output, battery_output, summary_output, edgedata_output]
        
        # PARALLEL EXECUTION FIX: Don't pass --remote-port in command line
        # TraCI will handle port via traci.start(port=...) parameter
        sumo_cmd = [  
            sumo_executable,  
            "-n", network_file,  
            "-a", additional_file,  
            "-r", route_file,  
            "--step-length", str(self.config.step_length),  
            "--chargingstations-output", charging_output,  
            "--battery-output", battery_output,  
            "--battery-output.precision", str(self.config.battery_precision),  
            "--device.battery.probability", str(self.config.battery_probability),  
            "--summary-output", summary_output,  
            "--edgedata-output", edgedata_output,
            # NOTE: Port is passed via traci.start(port=...), not here
            # No hard time limit - let simulation run until completion
            "--quit-on-end",  # Exit SUMO when simulation ends
            "--ignore-route-errors",  # Ignore route errors for robustness
            "--max-depart-delay", "300",  # Allow vehicles to depart up to 5 minutes late
            "--waiting-time-memory", "100",  # Remember waiting time for better routing
            "--no-warnings",  # Suppress route warnings for cleaner output
            "--error-log", os.path.join(output_dir, f"{prefix}errors.log")  # Log errors to file
        ]
        
        # Safe domain randomization via supported SUMO flags with deterministic seeding
        if getattr(self.config, 'enable_domain_randomization', False):
            try:
                # Import deterministic seed function
                from modules.RLOptimization.DomainRandomization import get_deterministic_seed
                
                # Set deterministic seed for reproducibility
                seed = get_deterministic_seed(base_seed, grid_id, episode_id)
                random.seed(seed)
                
                self.logger.info(f"Domain randomization with seed {seed} (base={base_seed}, grid={grid_id}, ep={episode_id})")
                
                # Now random.uniform calls are deterministic
                speeddev = random.uniform(0.05, 0.20)
                lc_duration = random.uniform(0.0, 2.0)
                depart_offset = random.uniform(0.0, 120.0)
                sumo_cmd += [
                    "--default.speeddev", f"{speeddev:.3f}",
                    "--lanechange.duration", f"{lc_duration:.3f}",
                    "--random-depart-offset", f"{depart_offset:.1f}"
                ]
                self.logger.debug(f"  speeddev={speeddev:.3f}, lc_duration={lc_duration:.3f}, depart_offset={depart_offset:.1f}")
            except Exception as e:
                self.logger.warning(f"Failed to apply domain randomization: {e}")
                pass
        
        print(f"🏃 Starting SUMO simulation (no time limit)...")
        start_time = time.time()
        step_count = 0
        vehicles_started = 0
        vehicles_completed = 0
        
        try:
            # Ensure all command arguments are strings
            sumo_cmd_str = [str(arg) for arg in sumo_cmd]
            print(f"🔧 SUMO command: {' '.join(sumo_cmd_str)}")
            print(f"🔧 Command types: {[type(arg) for arg in sumo_cmd_str]}")
            
            # PARALLEL EXECUTION FIX: Start SUMO with unique port and proper error handling
            try:
                # Use unique port for TraCI connection to avoid conflicts
                traci.start(sumo_cmd_str, port=self.traci_port)
                self.logger.info(f"✅ SUMO started on port {self.traci_port}")
            except Exception as start_error:
                print(f"❌ Failed to start SUMO: {start_error}")
                print("🔧 Trying with simplified command...")
                # Try with a simplified command (port passed via traci.start, not in cmd)
                simple_cmd = [sumo_executable, "-n", network_file, "-r", route_file, 
                             "--quit-on-end"]
                try:
                    traci.start(simple_cmd, port=self.traci_port)
                    print(f"✅ SUMO started with simplified command on port {self.traci_port}")
                except Exception as simple_error:
                    print(f"❌ SUMO failed even with simplified command: {simple_error}")
                    return
            
            # Check if there are any vehicles to simulate
            initial_vehicles = traci.simulation.getMinExpectedNumber()
            if initial_vehicles == 0:
                print("⚠️ No vehicles found in simulation - checking route file...")
                # Try to get vehicle count from route file
                try:
                    import xml.etree.ElementTree as ET
                    tree = ET.parse(route_file)
                    root = tree.getroot()
                    vehicle_count = len(root.findall('vehicle'))
                    print(f"   Route file contains {vehicle_count} vehicles")
                    if vehicle_count == 0:
                        print("❌ No vehicles in route file - simulation will be empty")
                        return
                except Exception as e:
                    print(f"   Could not parse route file: {e}")
            
            # Run simulation until all vehicles arrive or reasonable wall time timeout
            max_steps = 20000  # Increased maximum simulation steps to allow more complete runs
            while traci.simulation.getMinExpectedNumber() > 0 and step_count < max_steps:
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Only timeout if simulation takes too long (15 minutes max wall time)
                if elapsed > 900:  # 15 minutes max wall time
                    print(f"⏰ Simulation wall time timeout after {elapsed:.1f}s")
                    break
                
                try:
                    traci.simulationStep()
                    step_count += 1
                    
                    # Track vehicle statistics
                    vehicles_started = traci.simulation.getMinExpectedNumber()
                    vehicles_completed = traci.simulation.getArrivedNumber()
                    
                    # Progress update every 100 steps
                    if step_count % 100 == 0:
                        sim_time = traci.simulation.getTime()
                        print(f"   Step {step_count}, SimTime: {sim_time:.1f}s, Elapsed: {elapsed:.1f}s, "
                              f"Vehicles: {vehicles_started} active, {vehicles_completed} completed")
                            
                except Exception as step_error:
                    print(f"⚠️ Simulation step error: {step_error}")
                    # Continue simulation if possible - be more tolerant of errors
                    if step_count > 100:  # Only break if we've run for a while and have many errors
                        print("🛑 Breaking simulation due to repeated errors")
                        break
                    continue
                    
        except KeyboardInterrupt:
            print("🛑 Simulation interrupted by user")
            
        except Exception as e:
            print(f"❌ Simulation error: {e}")
            
        finally:
            # PARALLEL EXECUTION FIX: Proper cleanup with connection-specific termination
            try:
                if traci.isLoaded():
                    # Close this specific TraCI connection
                    traci.close(wait=True)
                    # CRITICAL FIX: Increased wait time for parallel execution
                    # When multiple workers are writing simultaneously, need more buffer time
                    time.sleep(1.5)  # Give SUMO 1.5s to flush buffers in parallel scenarios
                    self.logger.info(f"✅ TraCI connection on port {self.traci_port} closed")
                print(f"✅ Simulation completed after {step_count} steps")
            except Exception as e:
                print(f"⚠️ Error closing SUMO: {e}")
                # PARALLEL EXECUTION: Kill only this specific SUMO process by port
                import subprocess
                try:
                    # Find and kill only the SUMO process using this port
                    subprocess.run(["pkill", "-f", f"remote-port {self.traci_port}"], 
                                 capture_output=True, timeout=5)
                    self.logger.info(f"Killed SUMO process on port {self.traci_port}")
                except Exception as cleanup_error:
                    self.logger.warning(f"Could not kill SUMO process: {cleanup_error}")
            
            # PARALLEL EXECUTION FIX: Verify files are written and accessible
            max_retries = 3
            for retry in range(max_retries):
                all_files_ready = True
                for file_path in self.last_output_files:
                    if not os.path.exists(file_path):
                        all_files_ready = False
                        break
                    # Check if file is readable and not empty
                    try:
                        with open(file_path, 'r') as f:
                            f.read(1)
                    except Exception:
                        all_files_ready = False
                        break
                
                if all_files_ready:
                    self.logger.debug(f"All output files ready after {retry} retries")
                    break
                else:
                    if retry < max_retries - 1:
                        self.logger.warning(f"Output files not ready, retry {retry + 1}/{max_retries}")
                        time.sleep(0.5)
                    else:
                        self.logger.error("Some output files not ready after max retries")
    
    def cleanup_output_files(self):
        """
        Clean up simulation output files to prevent disk space issues in parallel execution.
        Should be called after files have been read and processed.
        
        PARALLEL EXECUTION FIX: More robust cleanup with retries and proper error handling.
        """
        if not hasattr(self, 'last_output_files'):
            return
        
        # PARALLEL EXECUTION FIX: Retry cleanup with delays to handle file locks
        max_retries = 3
        for file_path in self.last_output_files:
            for retry in range(max_retries):
                try:
                    if os.path.exists(file_path):
                        # Check if file is open by another process
                        try:
                            # Try to open file exclusively to check if it's accessible
                            with open(file_path, 'r') as f:
                                pass
                            # File is accessible, can delete
                            os.remove(file_path)
                            self.logger.debug(f"Cleaned up: {file_path}")
                            break
                        except PermissionError:
                            if retry < max_retries - 1:
                                # File might still be locked, wait and retry
                                time.sleep(0.5)
                                continue
                            else:
                                self.logger.warning(f"Could not remove {file_path}: file locked")
                    else:
                        # File already deleted or never created
                        break
                except Exception as e:
                    if retry < max_retries - 1:
                        time.sleep(0.3)
                        continue
                    else:
                        # Don't fail if cleanup fails - just log it
                        self.logger.warning(f"⚠️ Could not remove temporary file {file_path}: {e}")
                        break
    
    def cleanup_stale_processes(self):
        """
        PARALLEL EXECUTION FIX: Clean up any stale SUMO processes from this worker.
        Should be called at initialization to ensure clean state.
        """
        import subprocess
        try:
            # Find and kill any stale SUMO processes using this worker's port
            result = subprocess.run(
                ["pgrep", "-f", f"remote-port {self.traci_port}"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.stdout.strip():
                # Found stale process
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    try:
                        os.kill(int(pid), 9)  # Force kill
                        self.logger.info(f"Killed stale SUMO process {pid} on port {self.traci_port}")
                    except Exception as e:
                        self.logger.debug(f"Could not kill process {pid}: {e}")
        except Exception as e:
            self.logger.debug(f"Error checking for stale processes: {e}")