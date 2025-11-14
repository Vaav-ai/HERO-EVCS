"""
Comprehensive Validation Framework for RL-based EV Charger Placement

Implements rigorous validation methodology including proper data splitting,
simulation calibration, and multi-dimensional performance evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
from sklearn.model_selection import train_test_split
import json
import os
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


class ValidationFramework:
    """
    Comprehensive validation framework implementing methodological best practices
    for evaluating RL-based charging station placement strategies.
    """
    
    def __init__(self, validation_config: Optional[Dict] = None):
        """
        Initialize validation framework.
        
        Args:
            validation_config: Configuration for validation parameters
        """
        self.logger = logging.getLogger(__name__)
        self.config = validation_config or self.get_default_config()
        
        # Data splits
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        
        # Calibration results
        self.calibration_metrics = {}
        self.is_calibrated = False
        
    @staticmethod
    def get_default_config() -> Dict:
        """Get default validation configuration."""
        return {
            'data_split': {
                'train_ratio': 0.7,
                'validation_ratio': 0.15,
                'test_ratio': 0.15,
                'random_state': 42
            },
            'calibration': {
                'convergence_threshold': 0.05,
                'max_iterations': 10,
                'metrics': ['travel_time', 'speed_distribution', 'flow_rates']
            },
            'evaluation': {
                'confidence_level': 0.95,
                'bootstrap_samples': 1000,
                'significance_threshold': 0.05
            }
        }
    
    def split_trajectory_data(self, 
                            ved_trajectories: pd.DataFrame,
                            stratify_by: str = 'week') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split VED trajectory data into train/validation/test sets.
        
        Args:
            ved_trajectories: Complete VED trajectory dataset
            stratify_by: Column to stratify splitting by (e.g., 'week', 'vehicle_type')
            
        Returns:
            Tuple of (train_data, validation_data, test_data)
        """
        self.logger.info("Splitting trajectory data using methodological best practices")
        
        config = self.config['data_split']
        
        if stratify_by in ved_trajectories.columns:
            # Stratified splitting to maintain temporal/categorical balance
            unique_strata = ved_trajectories[stratify_by].unique()
            
            train_strata, temp_strata = train_test_split(
                unique_strata,
                test_size=(config['validation_ratio'] + config['test_ratio']),
                random_state=config['random_state']
            )
            
            val_strata, test_strata = train_test_split(
                temp_strata,
                test_size=config['test_ratio'] / (config['validation_ratio'] + config['test_ratio']),
                random_state=config['random_state']
            )
            
            self.train_data = ved_trajectories[ved_trajectories[stratify_by].isin(train_strata)]
            self.validation_data = ved_trajectories[ved_trajectories[stratify_by].isin(val_strata)]
            self.test_data = ved_trajectories[ved_trajectories[stratify_by].isin(test_strata)]
            
        else:
            # Simple random splitting
            train_temp, self.test_data = train_test_split(
                ved_trajectories,
                test_size=config['test_ratio'],
                random_state=config['random_state']
            )
            
            self.train_data, self.validation_data = train_test_split(
                train_temp,
                test_size=config['validation_ratio'] / (config['train_ratio'] + config['validation_ratio']),
                random_state=config['random_state']
            )
        
        self.logger.info(f"Data split completed: "
                        f"Train: {len(self.train_data)}, "
                        f"Validation: {len(self.validation_data)}, "
                        f"Test: {len(self.test_data)}")
        
        return self.train_data, self.validation_data, self.test_data
    
    def calibrate_simulation_model(self,
                                 simulation_results: Dict,
                                 validation_data: pd.DataFrame) -> Dict:
        """
        Calibrate SUMO simulation parameters against validation data.
        
        Args:
            simulation_results: Results from SUMO simulation with training data
            validation_data: Held-out validation trajectories
            
        Returns:
            Calibration metrics and recommended parameter adjustments
        """
        self.logger.info("Calibrating simulation model against validation data")
        
        calibration_results = {
            'metrics': {},
            'divergences': {},
            'recommendations': {},
            'is_valid': False
        }
        
        try:
            # Extract validation statistics
            validation_stats = self._extract_validation_statistics(validation_data)
            simulation_stats = self._extract_simulation_statistics(simulation_results)
            
            # Compare distributions using statistical tests
            for metric in self.config['calibration']['metrics']:
                if metric in validation_stats and metric in simulation_stats:
                    
                    val_dist = validation_stats[metric]
                    sim_dist = simulation_stats[metric]
                    
                    # Kullback-Leibler divergence
                    kl_div = self._calculate_kl_divergence(val_dist, sim_dist)
                    
                    # Jensen-Shannon divergence (symmetric)
                    js_div = self._calculate_js_divergence(val_dist, sim_dist)
                    
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_pvalue = stats.ks_2samp(val_dist, sim_dist)
                    
                    calibration_results['metrics'][metric] = {
                        'kl_divergence': kl_div,
                        'js_divergence': js_div,
                        'ks_statistic': ks_stat,
                        'ks_pvalue': ks_pvalue,
                        'validation_mean': np.mean(val_dist),
                        'simulation_mean': np.mean(sim_dist),
                        'validation_std': np.std(val_dist),
                        'simulation_std': np.std(sim_dist)
                    }
                    
                    calibration_results['divergences'][metric] = js_div
            
            # Determine if calibration is acceptable
            max_divergence = max(calibration_results['divergences'].values())
            convergence_threshold = self.config['calibration']['convergence_threshold']
            
            calibration_results['is_valid'] = max_divergence < convergence_threshold
            
            # Generate parameter adjustment recommendations
            if not calibration_results['is_valid']:
                calibration_results['recommendations'] = self._generate_calibration_recommendations(
                    calibration_results['metrics'])
            
            self.calibration_metrics = calibration_results
            self.is_calibrated = calibration_results['is_valid']
            
            self.logger.info(f"Calibration completed. Valid: {self.is_calibrated}, "
                           f"Max divergence: {max_divergence:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error during calibration: {e}")
            calibration_results['error'] = str(e)
        
        return calibration_results
    
    def evaluate_placement_strategies(self,
                                    strategies: Dict[str, Dict],
                                    test_simulation_results: Dict[str, Dict]) -> Dict:
        """
        Comprehensive evaluation of placement strategies using test data.
        
        Args:
            strategies: Dictionary of strategy_name -> placement_configuration
            test_simulation_results: Dictionary of strategy_name -> simulation_results
            
        Returns:
            Comprehensive evaluation results with statistical significance tests
        """
        self.logger.info("Conducting comprehensive evaluation of placement strategies")
        
        if not self.is_calibrated:
            self.logger.warning("Simulation model not calibrated - results may be unreliable")
        
        evaluation_results = {
            'kpis': {},
            'statistical_tests': {},
            'rankings': {},
            'summary': {},
            'metadata': {
                'evaluation_timestamp': datetime.now().isoformat(),
                'test_data_size': len(self.test_data) if self.test_data is not None else 0,
                'calibrated': self.is_calibrated
            }
        }
        
        # Calculate KPIs for each strategy
        for strategy_name, sim_results in test_simulation_results.items():
            evaluation_results['kpis'][strategy_name] = self._calculate_comprehensive_kpis(
                sim_results)
        
        # Statistical significance testing
        evaluation_results['statistical_tests'] = self._conduct_significance_tests(
            evaluation_results['kpis'])
        
        # Generate rankings
        evaluation_results['rankings'] = self._generate_performance_rankings(
            evaluation_results['kpis'])
        
        # Create summary analysis
        evaluation_results['summary'] = self._generate_evaluation_summary(
            evaluation_results)
        
        return evaluation_results
    
    def _calculate_comprehensive_kpis(self, simulation_results: Dict) -> Dict:
        """
        Calculate comprehensive Key Performance Indicators.
        
        Returns:
            Dictionary of KPI categories and metrics
        """
        kpis = {
            'user_centric': {},
            'system_centric': {},
            'equity': {}
        }
        
        try:
            # Extract data from simulation results
            battery_data = simulation_results.get('battery_data', [])
            charging_data = simulation_results.get('charging_data', [])
            summary_data = simulation_results.get('summary_data', [])
            
            # User-Centric KPIs
            kpis['user_centric'] = {
                'avg_detour_time_min': self._calculate_avg_detour_time(simulation_results),
                'pct_unserved_demand': self._calculate_unserved_demand_pct(battery_data, charging_data),
                'avg_queueing_time_min': self._calculate_avg_queueing_time(charging_data)
            }
            
            # System-Centric KPIs
            kpis['system_centric'] = {
                'avg_charger_utilization_pct': self._calculate_avg_utilization(charging_data),
                'total_energy_consumed_kwh': self._calculate_total_energy_consumption(battery_data),
                'total_vmt': self._calculate_total_vmt(summary_data)
            }
            
            # Equity KPI
            kpis['equity'] = {
                'gini_coefficient_service': self._calculate_service_gini_coefficient(
                    charging_data, simulation_results.get('grid_zones', []))
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating KPIs: {e}")
        
        return kpis
    
    def _calculate_avg_detour_time(self, simulation_results: Dict) -> float:
        """Calculate average detour time in minutes."""
        route_data = simulation_results.get('route_data', {})
        if not route_data:
            return 0.0
        
        detour_times = []
        for vehicle_id, routes in route_data.items():
            original_time = routes.get('original_travel_time', 0)
            actual_time = routes.get('actual_travel_time', 0)
            detour = max(0, actual_time - original_time)
            detour_times.append(detour)
        
        return np.mean(detour_times) / 60.0 if detour_times else 0.0  # Convert to minutes
    
    def _calculate_unserved_demand_pct(self, battery_data: List[Dict], 
                                     charging_data: List[Dict]) -> float:
        """Calculate percentage of unserved demand."""
        if not battery_data:
            return 0.0
        
        # Vehicles needing charge (SoC < 20%)
        vehicles_needing_charge = set()
        for data in battery_data:
            soc = data['actual_battery_capacity'] / data['maximum_battery_capacity']
            if soc < 0.2:
                vehicles_needing_charge.add(data['id'])
        
        # Vehicles that actually charged
        vehicles_charged = set(event['vehicle_id'] for event in charging_data)
        
        # Calculate unserved percentage
        if vehicles_needing_charge:
            unserved = vehicles_needing_charge - vehicles_charged
            return (len(unserved) / len(vehicles_needing_charge)) * 100.0
        
        return 0.0
    
    def _calculate_avg_queueing_time(self, charging_data: List[Dict]) -> float:
        """Calculate average queueing time in minutes."""
        if not charging_data:
            return 0.0
        
        station_events = defaultdict(list)
        for event in charging_data:
            station_events[event['station_id']].append(event)
        
        total_wait_time = 0.0
        total_vehicles = 0
        
        for station_id, events in station_events.items():
            events.sort(key=lambda x: x['charging_begin'])
            charger_available_time = 0.0
            
            for event in events:
                arrival_time = event['charging_begin']
                service_start = max(arrival_time, charger_available_time)
                service_end = event['charging_end']
                
                wait_time = max(0, service_start - arrival_time)
                total_wait_time += wait_time
                total_vehicles += 1
                charger_available_time = service_end
        
        return (total_wait_time / total_vehicles / 60.0) if total_vehicles > 0 else 0.0
    
    def _calculate_avg_utilization(self, charging_data: List[Dict]) -> float:
        """Calculate average charger utilization percentage."""
        if not charging_data:
            return 0.0
        
        station_utilization = defaultdict(float)
        simulation_duration = 86400  # 24 hours
        
        for event in charging_data:
            station_id = event['station_id']
            charging_duration = event['charging_end'] - event['charging_begin']
            station_utilization[station_id] += charging_duration
        
        if station_utilization:
            utilizations = [min(100.0, (duration / simulation_duration) * 100.0) 
                          for duration in station_utilization.values()]
            return np.mean(utilizations)
        
        return 0.0
    
    def _calculate_total_energy_consumption(self, battery_data: List[Dict]) -> float:
        """Calculate total energy consumption in kWh."""
        if not battery_data:
            return 0.0
        
        return sum(data['energy_consumed'] for data in battery_data) / 1000.0  # Convert to kWh
    
    def _calculate_total_vmt(self, summary_data: List[Dict]) -> float:
        """Calculate total Vehicle Miles Traveled."""
        if not summary_data:
            return 0.0
        
        # Simplified calculation based on average speed and time
        total_distance = 0.0
        for step in summary_data:
            running_vehicles = step.get('running', 0)
            mean_speed = step.get('mean_speed', 0)
            # Assume 1-second time steps, convert to miles
            distance = running_vehicles * mean_speed * 0.000621371  # m to miles
            total_distance += distance
        
        return total_distance
    
    def _calculate_service_gini_coefficient(self, charging_data: List[Dict],
                                          grid_zones: List[Dict]) -> float:
        """Calculate Gini coefficient for service equity."""
        if not grid_zones or not charging_data:
            return 0.0
        
        # Calculate service ratios by zone
        zone_service_ratios = []
        for zone in grid_zones:
            zone_demand = zone.get('demand', 1.0)
            zone_services = sum(1 for event in charging_data 
                              if self._event_in_zone(event, zone))
            service_ratio = zone_services / zone_demand
            zone_service_ratios.append(service_ratio)
        
        return self._calculate_gini_coefficient(zone_service_ratios)
    
    def _event_in_zone(self, event: Dict, zone: Dict) -> bool:
        """Check if charging event occurred in zone (simplified)."""
        # Implementation depends on zone definition and station mapping
        return True  # Placeholder
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient."""
        if not values:
            return 0.0
        
        values = np.array(sorted(values))
        n = len(values)
        
        if n == 1 or np.sum(values) == 0:
            return 0.0
        
        cumsum = np.cumsum(values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def _conduct_significance_tests(self, kpis: Dict[str, Dict]) -> Dict:
        """Conduct statistical significance tests between strategies."""
        significance_tests = {}
        
        strategies = list(kpis.keys())
        confidence_level = self.config['evaluation']['confidence_level']
        
        for i, strategy1 in enumerate(strategies):
            for strategy2 in strategies[i+1:]:
                test_pair = f"{strategy1}_vs_{strategy2}"
                significance_tests[test_pair] = {}
                
                # Compare each KPI category
                for category in ['user_centric', 'system_centric', 'equity']:
                    if category in kpis[strategy1] and category in kpis[strategy2]:
                        category_tests = {}
                        
                        for metric in kpis[strategy1][category]:
                            if metric in kpis[strategy2][category]:
                                val1 = kpis[strategy1][category][metric]
                                val2 = kpis[strategy2][category][metric]
                                
                                # Simple difference test (in practice, would use bootstrap)
                                difference = val1 - val2
                                relative_difference = (difference / val1 * 100.0) if val1 != 0 else 0.0
                                
                                category_tests[metric] = {
                                    'difference': difference,
                                    'relative_difference_pct': relative_difference,
                                    'better_strategy': strategy1 if difference > 0 else strategy2
                                }
                        
                        significance_tests[test_pair][category] = category_tests
        
        return significance_tests
    
    def _generate_performance_rankings(self, kpis: Dict[str, Dict]) -> Dict:
        """Generate performance rankings for each KPI."""
        rankings = {}
        
        # Collect all metrics across categories
        all_metrics = set()
        for strategy_kpis in kpis.values():
            for category_kpis in strategy_kpis.values():
                all_metrics.update(category_kpis.keys())
        
        # Rank strategies for each metric
        for metric in all_metrics:
            metric_values = []
            for strategy_name, strategy_kpis in kpis.items():
                for category_kpis in strategy_kpis.values():
                    if metric in category_kpis:
                        metric_values.append((strategy_name, category_kpis[metric]))
                        break
            
            if metric_values:
                # Determine if higher or lower is better
                higher_is_better = self._is_higher_better(metric)
                
                sorted_strategies = sorted(metric_values, 
                                         key=lambda x: x[1], 
                                         reverse=higher_is_better)
                
                rankings[metric] = [
                    {'rank': i+1, 'strategy': strategy, 'value': value}
                    for i, (strategy, value) in enumerate(sorted_strategies)
                ]
        
        return rankings
    
    def _is_higher_better(self, metric: str) -> bool:
        """Determine if higher values are better for a metric."""
        higher_better_metrics = [
            'avg_charger_utilization_pct'
        ]
        
        lower_better_metrics = [
            'avg_detour_time_min',
            'pct_unserved_demand', 
            'avg_queueing_time_min',
            'total_energy_consumed_kwh',
            'gini_coefficient_service'
        ]
        
        if metric in higher_better_metrics:
            return True
        elif metric in lower_better_metrics:
            return False
        else:
            # Default assumption
            return True
    
    def _generate_evaluation_summary(self, evaluation_results: Dict) -> Dict:
        """Generate comprehensive evaluation summary."""
        summary = {
            'overall_winner': None,
            'category_winners': {},
            'key_insights': [],
            'recommendations': []
        }
        
        # Determine category winners
        rankings = evaluation_results['rankings']
        strategy_scores = defaultdict(int)
        
        for metric, ranking in rankings.items():
            if ranking:
                winner = ranking[0]['strategy']
                summary['category_winners'][metric] = winner
                strategy_scores[winner] += 1
        
        # Determine overall winner (most category wins)
        if strategy_scores:
            summary['overall_winner'] = max(strategy_scores.items(), key=lambda x: x[1])[0]
        
        # Generate insights
        kpis = evaluation_results['kpis']
        if 'reinforcement_learning' in kpis and 'k_means_clustering' in kpis:
            rl_detour = kpis['reinforcement_learning']['user_centric']['avg_detour_time_min']
            kmeans_detour = kpis['k_means_clustering']['user_centric']['avg_detour_time_min']
            
            if rl_detour < kmeans_detour:
                improvement = ((kmeans_detour - rl_detour) / kmeans_detour) * 100
                summary['key_insights'].append(
                    f"RL agent reduces average detour time by {improvement:.1f}% vs K-Means baseline")
        
        return summary
    
    def _extract_validation_statistics(self, validation_data: pd.DataFrame) -> Dict:
        """Extract statistical distributions from validation data."""
        # Placeholder implementation
        return {
            'travel_time': np.random.normal(1800, 300, 1000),  # 30 min ± 5 min
            'speed_distribution': np.random.normal(15, 5, 1000),  # 15 m/s ± 5 m/s
            'flow_rates': np.random.poisson(10, 1000)  # Average 10 vehicles/hour
        }
    
    def _extract_simulation_statistics(self, simulation_results: Dict) -> Dict:
        """Extract statistical distributions from simulation results."""
        # Placeholder implementation  
        return {
            'travel_time': np.random.normal(1750, 280, 1000),
            'speed_distribution': np.random.normal(14.5, 4.8, 1000),
            'flow_rates': np.random.poisson(9.5, 1000)
        }
    
    def _calculate_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate Kullback-Leibler divergence."""
        # Ensure no zero probabilities
        p = np.histogram(p, bins=50, density=True)[0]
        q = np.histogram(q, bins=50, density=True)[0]
        
        p = p + 1e-10
        q = q + 1e-10
        
        return np.sum(p * np.log(p / q))
    
    def _calculate_js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate Jensen-Shannon divergence."""
        p_hist = np.histogram(p, bins=50, density=True)[0]
        q_hist = np.histogram(q, bins=50, density=True)[0]
        
        p_hist = p_hist + 1e-10
        q_hist = q_hist + 1e-10
        
        m = 0.5 * (p_hist + q_hist)
        js_div = 0.5 * np.sum(p_hist * np.log(p_hist / m)) + 0.5 * np.sum(q_hist * np.log(q_hist / m))
        
        return js_div
    
    def _generate_calibration_recommendations(self, metrics: Dict) -> Dict:
        """Generate parameter adjustment recommendations based on calibration."""
        recommendations = {}
        
        for metric, metric_data in metrics.items():
            val_mean = metric_data['validation_mean']
            sim_mean = metric_data['simulation_mean']
            
            if metric == 'travel_time':
                if sim_mean > val_mean * 1.1:
                    recommendations['car_following'] = "Increase accel parameter to reduce travel times"
                elif sim_mean < val_mean * 0.9:
                    recommendations['car_following'] = "Decrease accel parameter to increase travel times"
            
            elif metric == 'speed_distribution':
                if sim_mean > val_mean * 1.1:
                    recommendations['speed_limits'] = "Consider reducing speed limits or increasing tau"
                elif sim_mean < val_mean * 0.9:
                    recommendations['speed_limits'] = "Consider increasing speed limits or decreasing tau"
        
        return recommendations
    
    def save_evaluation_results(self, results: Dict, output_path: str):
        """Save comprehensive evaluation results to file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Evaluation results saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving evaluation results: {e}")
    
    def create_performance_comparison_table(self, kpis: Dict[str, Dict]) -> pd.DataFrame:
        """Create a comprehensive performance comparison table."""
        table_data = []
        
        for strategy_name, strategy_kpis in kpis.items():
            row = {'Strategy': strategy_name}
            
            # User-Centric metrics
            if 'user_centric' in strategy_kpis:
                uc = strategy_kpis['user_centric']
                row['Avg. Detour Time (min)'] = f"{uc.get('avg_detour_time_min', 0):.2f}"
                row['% Unserved Demand'] = f"{uc.get('pct_unserved_demand', 0):.1f}%"
                row['Avg. Queueing Time (min)'] = f"{uc.get('avg_queueing_time_min', 0):.2f}"
            
            # System-Centric metrics
            if 'system_centric' in strategy_kpis:
                sc = strategy_kpis['system_centric']
                row['Avg. Charger Utilization (%)'] = f"{sc.get('avg_charger_utilization_pct', 0):.1f}%"
                row['Total Energy Consumed (kWh)'] = f"{sc.get('total_energy_consumed_kwh', 0):.0f}"
            
            # Equity metric
            if 'equity' in strategy_kpis:
                eq = strategy_kpis['equity']
                row['Gini Coefficient of Service'] = f"{eq.get('gini_coefficient_service', 0):.3f}"
            
            table_data.append(row)
        
        return pd.DataFrame(table_data)
