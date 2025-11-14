from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime
import json
import logging
import sys
import os

def _setup_metrics_logging():
    """Setup comprehensive logging for metrics calculations."""
    # Create logger with hierarchical name
    logger = logging.getLogger(f"{__name__}.metrics")
    
    # Set logging level based on environment
    log_level = os.environ.get('METRICS_LOG_LEVEL', 'INFO').upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Create formatter with detailed information
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add console handler if not already present
    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler for detailed logs
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"metrics_{datetime.now().strftime('%Y%m%d')}.log")
    
    if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == log_file for handler in logger.handlers):
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Initialize metrics logger
metrics_logger = _setup_metrics_logging()

def fleet_battery_metrics(battery_data) -> Dict[str, float]:
    """Return mean/final SoC, variance and low-SoC share.
    
    Args:
        battery_data: List of battery data dictionaries from parse_battery_data
    """
    metrics_logger.debug(f"Calculating fleet battery metrics for {len(battery_data) if battery_data else 0} data points")
    
    if not battery_data:
        metrics_logger.warning("No battery data provided - returning zero metrics")
        return {
            'mean_final_soc': 0.0,
            'soc_variance': 0.0,
            'pct_low_soc': 0.0,
            'total_energy_consumed': 0.0,
            'avg_trip_duration': 0.0
        }
    
    # Group by vehicle ID to get final SOC for each vehicle
    vehicle_data = {}
    for entry in battery_data:
        vehicle_id = entry['id']
        if vehicle_id not in vehicle_data:
            vehicle_data[vehicle_id] = []
        vehicle_data[vehicle_id].append(entry)
    
    # Calculate final SOC for each vehicle
    final_socs = []
    total_energy_consumed = 0.0
    trip_durations = []
    
    for vehicle_id, entries in vehicle_data.items():
        if entries:
            # Sort by time to get chronological order
            entries.sort(key=lambda x: x['time'])
            
            # Calculate final SOC
            final_entry = entries[-1]
            max_capacity = final_entry['maximum_battery_capacity']
            actual_capacity = final_entry['actual_battery_capacity']
            
            if max_capacity > 0:
                final_soc = actual_capacity / max_capacity
                final_socs.append(final_soc)
            
            # Calculate energy consumed
            energy_consumed = final_entry['energy_consumed']
            total_energy_consumed += energy_consumed
            
            # Calculate trip duration
            if len(entries) > 1:
                trip_duration = entries[-1]['time'] - entries[0]['time']
                trip_durations.append(trip_duration)
    
    if not final_socs:
        return {
            'mean_final_soc': 0.0,
            'soc_variance': 0.0,
            'pct_low_soc': 0.0,
            'total_energy_consumed': total_energy_consumed,
            'avg_trip_duration': 0.0
        }
    
    final_socs = np.array(final_socs)
    avg_trip_duration = np.mean(trip_durations) if trip_durations else 0.0
    
    return {
        'mean_final_soc': float(np.mean(final_socs)),
        'soc_variance': float(np.var(final_socs)),
        'pct_low_soc': float((final_socs < 0.3).mean()),
        'total_energy_consumed': total_energy_consumed,
        'avg_trip_duration': avg_trip_duration
    }

def charging_metrics(charging_data, sim_duration) -> Dict[str, float]:
    """Queue times / utilisation across stations.
    
    Args:
        charging_data: List of charging event dictionaries from parse_charging_events
        sim_duration: Simulation duration in seconds
    """
    if not charging_data:
        return {
            'utilisation_gini': 0.0,
            'avg_queue_time': 0.0,
            'total_charging_events': 0,
            'avg_charging_duration': 0.0
        }
    
    per_station = {}
    charging_durations = []
    
    for ev in charging_data:
        # Handle different possible field names for station ID
        station_id = ev.get('station_id') or ev.get('stationId') or ev.get('station')
        if station_id is None:
            continue
            
        # Handle different possible field names for charging times
        charging_begin = ev.get('charging_begin') or ev.get('chargingBegin') or ev.get('begin')
        charging_end = ev.get('charging_end') or ev.get('chargingEnd') or ev.get('end')
        
        if charging_begin is not None and charging_end is not None:
            duration = charging_end - charging_begin
            per_station.setdefault(station_id, []).append(duration)
            charging_durations.append(duration)
    
    if not per_station:
        return {
            'utilisation_gini': 0.0,
            'avg_queue_time': 0.0,
            'total_charging_events': len(charging_data),
            'avg_charging_duration': 0.0
        }

    # Calculate utilization (total charging time / simulation duration)
    util = {s: sum(times)/sim_duration if sim_duration > 0 else 0.0 for s, times in per_station.items()}
    queue_times = [np.mean(v) for v in per_station.values() if v]
    avg_charging_duration = np.mean(charging_durations) if charging_durations else 0.0
    
    return {
        'utilisation_gini': _gini(list(util.values())),
        'avg_queue_time': float(np.mean(queue_times) if queue_times else 0.0),
        'total_charging_events': len(charging_data),
        'avg_charging_duration': avg_charging_duration
    }

def _gini(arr):
    arr = np.array(arr)
    if arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n   = arr.size
    cum = np.cumsum(arr)
    return (n + 1 - 2 * (cum / cum[-1]).sum()) / n

def calculate_comprehensive_metrics(simulation_results: Dict, method_name: str, 
                                  num_stations: int, grid_id: str) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for method comparison.
    
    Args:
        simulation_results: Results from simulation
        method_name: Name of the method (e.g., 'hybrid_ucb', 'kmeans_baseline')
        num_stations: Number of charging stations
        grid_id: Grid identifier
        
    Returns:
        Dictionary with comprehensive metrics
    """
    metrics = {
        'method_name': method_name,
        'grid_id': grid_id,
        'num_stations': num_stations,
        'timestamp': datetime.now().isoformat(),
        'simulation_metrics': {},
        'optimization_metrics': {},
        'efficiency_metrics': {},
        'statistical_metrics': {}
    }
    
    # Simulation performance metrics
    if 'fleet_metrics' in simulation_results:
        fleet_data = simulation_results['fleet_metrics']
        metrics['simulation_metrics'].update({
            'mean_final_soc': fleet_data.get('mean_final_soc', 0.0),
            'soc_variance': fleet_data.get('soc_variance', 0.0),
            'pct_low_soc': fleet_data.get('pct_low_soc', 0.0),
            'total_energy_consumed': fleet_data.get('total_energy_consumed', 0.0),
            'avg_trip_duration': fleet_data.get('avg_trip_duration', 0.0)
        })
    
    if 'charging_metrics' in simulation_results:
        charging_data = simulation_results['charging_metrics']
        metrics['simulation_metrics'].update({
            'utilisation_gini': charging_data.get('utilisation_gini', 0.0),
            'avg_queue_time': charging_data.get('avg_queue_time', 0.0),
            'total_charging_events': charging_data.get('total_charging_events', 0),
            'avg_charging_duration': charging_data.get('avg_charging_duration', 0.0)
        })
    
    # Optimization efficiency metrics
    if 'optimization_metrics' in simulation_results:
        opt_data = simulation_results['optimization_metrics']
        metrics['optimization_metrics'].update({
            'total_episodes': opt_data.get('total_episodes', 0),
            'convergence_episode': opt_data.get('convergence_episode', None),
            'convergence_achieved': opt_data.get('convergence_achieved', False),
            'best_reward': opt_data.get('best_reward', 0.0),
            'final_reward': opt_data.get('final_reward', 0.0),
            'average_reward': opt_data.get('average_reward', 0.0),
            'reward_std': opt_data.get('reward_std', 0.0),
            'cumulative_regret': opt_data.get('cumulative_regret', 0.0),
            'action_diversity': opt_data.get('action_diversity', 0.0),
            'exploration_efficiency': opt_data.get('exploration_efficiency', 0.0)
        })
    
    # Efficiency metrics for computational performance
    metrics['efficiency_metrics'] = {
        'total_simulation_time': simulation_results.get('total_simulation_time', 0.0),
        'avg_episode_time': simulation_results.get('avg_episode_time', 0.0),
        'memory_usage_mb': simulation_results.get('memory_usage_mb', 0.0),
        'cpu_utilization': simulation_results.get('cpu_utilization', 0.0)
    }
    
    # Statistical significance metrics
    if 'episode_rewards' in simulation_results:
        rewards = simulation_results['episode_rewards']
        if rewards:
            metrics['statistical_metrics'] = {
                'reward_mean': float(np.mean(rewards)),
                'reward_std': float(np.std(rewards)),
                'reward_median': float(np.median(rewards)),
                'reward_q25': float(np.percentile(rewards, 25)),
                'reward_q75': float(np.percentile(rewards, 75)),
                'reward_min': float(np.min(rewards)),
                'reward_max': float(np.max(rewards)),
                'reward_skewness': float(_calculate_skewness(rewards)),
                'reward_kurtosis': float(_calculate_kurtosis(rewards))
            }
    
    return metrics

def compare_methods_performance(results: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Compare performance across different methods.
    
    Args:
        results: Dictionary mapping method names to their metrics
        
    Returns:
        Comprehensive comparison results
    """
    comparison = {
        'comparison_timestamp': datetime.now().isoformat(),
        'methods_compared': results,  # CRITICAL FIX: Store actual results, not just keys
        'performance_ranking': {},
        'statistical_tests': {},
        'efficiency_analysis': {},
        'summary_statistics': {}
    }
    
    # Extract key metrics for comparison
    method_metrics = {}
    for method, metrics in results.items():
        # CRITICAL FIX: Handle different result structures from hybrid framework
        if 'optimization_metrics' in metrics:
            # Standard structure with optimization_metrics
            method_metrics[method] = {
                'best_reward': metrics.get('optimization_metrics', {}).get('best_reward', 0.0),
                'convergence_episode': metrics.get('optimization_metrics', {}).get('convergence_episode', None),
                'convergence_achieved': metrics.get('optimization_metrics', {}).get('convergence_achieved', False),
                'total_episodes': metrics.get('optimization_metrics', {}).get('total_episodes', 0),
                'avg_simulation_time': metrics.get('efficiency_metrics', {}).get('avg_episode_time', 0.0),
                'mean_final_soc': metrics.get('simulation_metrics', {}).get('mean_final_soc', 0.0),
                'utilisation_gini': metrics.get('simulation_metrics', {}).get('utilisation_gini', 0.0)
            }
        else:
            # Hybrid framework structure - extract from different locations
            method_metrics[method] = {
                'best_reward': metrics.get('best_reward', 0.0),
                'convergence_episode': metrics.get('convergence_episode', None),
                'convergence_achieved': metrics.get('convergence_achieved', False),
                'total_episodes': metrics.get('total_episodes', 0),
                'avg_simulation_time': 0.0,  # Not available in hybrid structure
                'mean_final_soc': 0.0,  # Not available in hybrid structure
                'utilisation_gini': 0.0  # Not available in hybrid structure
            }
    
    # Performance ranking by best reward
    sorted_by_reward = sorted(method_metrics.items(), 
                            key=lambda x: x[1]['best_reward'], reverse=True)
    comparison['performance_ranking']['by_reward'] = [method for method, _ in sorted_by_reward]
    
    # Performance ranking by convergence speed
    converged_methods = [(method, data) for method, data in method_metrics.items() 
                        if data['convergence_achieved'] and data['convergence_episode'] is not None]
    if converged_methods:
        sorted_by_convergence = sorted(converged_methods, 
                                     key=lambda x: x[1]['convergence_episode'])
        comparison['performance_ranking']['by_convergence_speed'] = [method for method, _ in sorted_by_convergence]
    
    # Efficiency analysis
    efficiency_data = {method: data['avg_simulation_time'] for method, data in method_metrics.items()}
    if efficiency_data:
        fastest_method = min(efficiency_data.items(), key=lambda x: x[1])
        comparison['efficiency_analysis']['fastest_method'] = fastest_method[0]
        comparison['efficiency_analysis']['speedup_factors'] = {
            method: fastest_method[1] / time if time > 0 else float('inf')
            for method, time in efficiency_data.items()
        }
    
    # Summary statistics
    rewards = [data['best_reward'] for data in method_metrics.values()]
    if rewards:
        comparison['summary_statistics'] = {
            'best_overall_reward': max(rewards),
            'worst_overall_reward': min(rewards),
            'reward_range': max(rewards) - min(rewards),
            'convergence_rate': sum(1 for data in method_metrics.values() 
                                  if data['convergence_achieved']) / len(method_metrics),
            'avg_episodes_to_convergence': np.mean([data['convergence_episode'] 
                                                   for data in method_metrics.values() 
                                                   if data['convergence_episode'] is not None])
        }
    
    return comparison

def _calculate_skewness(data):
    """Calculate skewness of data."""
    if len(data) < 3:
        return 0.0
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    return np.mean(((data - mean) / std) ** 3)

def _calculate_kurtosis(data):
    """Calculate kurtosis of data."""
    if len(data) < 4:
        return 0.0
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    return np.mean(((data - mean) / std) ** 4) - 3

def save_comparison_results(comparison_results: Dict, output_path: str) -> None:
    """
    Save comparison results to file.
    
    Args:
        comparison_results: Results from compare_methods_performance
        output_path: Path to save results
    """
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_numpy_types(comparison_results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=4, default=str)
    
    print(f"Comparison results saved to: {output_path}")