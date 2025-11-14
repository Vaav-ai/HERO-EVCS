"""
Base Bandit Algorithm Interface

Defines the abstract base class for all bandit algorithms used in EV charging station placement optimization.
This ensures a consistent interface across different bandit implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict
import numpy as np


class BaseBandit(ABC):
    """
    Abstract base class for bandit algorithms.
    
    All bandit algorithms must implement select_action() and update() methods.
    The bandit algorithms are used to optimize charging station placement by treating
    each possible placement configuration as an "arm" and the simulation reward as the payoff.
    """
    
    def __init__(self, seed: int = None):
        """
        Initialize the bandit algorithm.
        
        Args:
            seed: Random seed for reproducibility (optional)
        """
        self.total_pulls = 0
        self.action_rewards = {}  # Track total rewards for each action
        self.action_counts = {}   # Track number of times each action was selected
        self.action_reward_history = {}  # Track individual reward samples for variance calculation
        self.seed = seed
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            import random
            random.seed(seed)
    
    def set_episode_seed(self, episode_id: int):
        """
        Set a deterministic seed for a specific episode.
        This ensures reproducible action selection for the same episode.
        
        Args:
            episode_id: The episode number
        """
        if self.seed is not None:
            episode_seed = self.seed + episode_id * 10000
            np.random.seed(episode_seed)
            import random
            random.seed(episode_seed)
        
    @abstractmethod
    def select_action(self, available_actions: Optional[List[Any]] = None) -> Any:
        """
        Select an action (charging station placement) based on the bandit algorithm.
        
        Parameters
        ----------
        available_actions : List[Any], optional
            List of available actions (placement configurations). If None, the algorithm
            should use its internal state to determine available actions.
            
        Returns
        -------
        Any
            The selected action (placement configuration)
        """
        pass
    
    @abstractmethod
    def update(self, action: Any, reward: float) -> None:
        """
        Update the bandit algorithm with the reward received from the selected action.
        
        Parameters
        ----------
        action : Any
            The action that was selected and executed
        reward : float
            The reward received from executing the action
        """
        # Track reward history for accurate variance calculation
        if action not in self.action_reward_history:
            self.action_reward_history[action] = []
        self.action_reward_history[action].append(reward)
        
        # Basic tracking (subclasses can override)
        if action not in self.action_rewards:
            self.action_rewards[action] = 0.0
        if action not in self.action_counts:
            self.action_counts[action] = 0
            
        self.action_rewards[action] += reward
        self.action_counts[action] += 1
        self.total_pulls += 1
    
    def get_action_statistics(self) -> Dict[Any, Dict[str, float]]:
        """
        Get statistics for all actions that have been tried.
        
        Returns
        -------
        Dict[Any, Dict[str, float]]
            Dictionary mapping actions to their statistics (count, average reward, etc.)
        """
        stats = {}
        for action in self.action_counts:
            if self.action_counts[action] > 0:
                # Use action_rewards if available, otherwise use value_estimates
                if hasattr(self, 'action_rewards') and action in self.action_rewards:
                    total_reward = self.action_rewards[action]
                    avg_reward = total_reward / self.action_counts[action]
                elif hasattr(self, 'value_estimates') and action in self.value_estimates:
                    avg_reward = self.value_estimates[action]
                    total_reward = avg_reward * self.action_counts[action]
                else:
                    total_reward = 0.0
                    avg_reward = 0.0
                    
                stats[action] = {
                    'count': self.action_counts[action],
                    'total_reward': total_reward,
                    'average_reward': avg_reward,
                    'confidence_interval': self._calculate_confidence_interval(action)
                }
        return stats
    
    def _calculate_confidence_interval(self, action: Any, confidence_level: float = 0.95) -> tuple:
        """
        Calculate normalized confidence interval for an action's reward.
        
        This method computes a confidence interval using proper statistical methods
        and normalizes it to a [0, 1] scale where:
        - Width close to 0 = high confidence (narrow interval)
        - Width close to 1 = low confidence (wide interval)
        
        Parameters
        ----------
        action : Any
            The action to calculate confidence interval for
        confidence_level : float
            Confidence level (default 0.95 for 95% confidence)
            
        Returns
        -------
        tuple
            (lower_bound, upper_bound) of the normalized confidence interval on [0, 1] scale
        """
        if action not in self.action_counts:
            return (0.0, 1.0)  # Maximum uncertainty for untried actions
        
        n = self.action_counts[action]
        
        # For untried or single-sample actions, use conservative uncertainty estimates
        if n < 1:
            return (0.0, 1.0)  # Maximum uncertainty
        
        # Get average reward
        if hasattr(self, 'action_rewards') and action in self.action_rewards:
            avg_reward = self.action_rewards[action] / n
        elif hasattr(self, 'value_estimates') and action in self.value_estimates:
            avg_reward = self.value_estimates[action]
        else:
            return (0.0, 1.0)
        
        # SPECIAL CASE: Single sample (n=1) - use Bayesian prior-based estimate
        if n == 1:
            # With only one sample, we have maximum uncertainty about the true mean
            # Use a conservative interval based on the reward's position in [0,1]
            # Assume uniform prior, so uncertainty is proportional to distance from extremes
            margin = 0.5 - abs(avg_reward - 0.5)  # Higher uncertainty near 0.5
            margin = max(0.3, margin)  # At least 30% uncertainty with single sample
            center = 0.5
            lower_bound = max(0.0, center - margin / 2)
            upper_bound = min(1.0, center + margin / 2)
            return (lower_bound, upper_bound)
        
        # IMPROVED: Use t-distribution for small samples (n < 30)
        if n < 30:
            # t-scores for different confidence levels at df=n-1
            # Adjust for very small samples (use df=max(n-1, 1))
            df = max(n - 1, 1)
            if df == 1:
                t_scores = {0.90: 6.314, 0.95: 12.706, 0.99: 63.657}  # df=1
            elif df <= 5:
                t_scores = {0.90: 2.015, 0.95: 2.571, 0.99: 4.032}  # df=5 (conservative)
            else:
                t_scores = {0.90: 1.833, 0.95: 2.262, 0.99: 3.250}  # df=10
            
            t_score = t_scores.get(confidence_level, 2.571)
            
            # Get sample standard deviation
            std = self._get_sample_std(action)
            
            # Handle case where std is 0 (all samples identical)
            if std < 1e-6:
                # Very small std means high confidence
                margin = 0.1  # Small uncertainty for identical samples
            else:
                # Standard error with finite population correction
                se = std / np.sqrt(n)
                
                # Margin of error (adjusted for small samples)
                margin_absolute = t_score * se * (1 + 1/n)
                
                # Normalize to relative margin
                if abs(avg_reward) > 1e-6:
                    margin = min(1.0, margin_absolute / abs(avg_reward))
                else:
                    # For near-zero rewards, use absolute margin capped at 1.0
                    margin = min(1.0, margin_absolute * 2)  # Scale up for visibility
                
        else:
            # Use normal approximation for larger samples
            z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
            z_score = z_scores.get(confidence_level, 1.96)
            
            # Calculate proper confidence interval
            sample_variance = self._calculate_sample_variance(action)
            std_error = np.sqrt(sample_variance / n)
            
            # Normalize margin to 0-1 scale using relative uncertainty
            if abs(avg_reward) > 1e-6:
                # Relative margin as fraction of reward (coefficient of variation)
                margin = min(1.0, z_score * std_error / abs(avg_reward))
            else:
                # For near-zero rewards, use absolute uncertainty
                margin = min(1.0, z_score * std_error)
        
        # Map to normalized [0, 1] confidence interval
        # Narrow margin = high confidence (center around 0.5)
        center = 0.5
        lower_bound = max(0.0, center - margin / 2)
        upper_bound = min(1.0, center + margin / 2)
        
        return (lower_bound, upper_bound)
    
    def _get_sample_std(self, action: Any) -> float:
        """
        Calculate actual sample standard deviation from reward history.
        
        Parameters
        ----------
        action : Any
            The action to calculate standard deviation for
            
        Returns
        -------
        float
            Sample standard deviation
        """
        # Use actual reward history if available
        if action in self.action_reward_history and len(self.action_reward_history[action]) >= 2:
            rewards = self.action_reward_history[action]
            return float(np.std(rewards, ddof=1))  # Sample std with Bessel's correction
        
        # Fallback: estimate from average reward
        if action in self.action_counts and self.action_counts[action] > 0:
            if hasattr(self, 'action_rewards') and action in self.action_rewards:
                avg_reward = self.action_rewards[action] / self.action_counts[action]
            elif hasattr(self, 'value_estimates') and action in self.value_estimates:
                avg_reward = self.value_estimates[action]
            else:
                return 0.5  # Default uncertainty
            
            # Conservative estimate: std ≈ |mean| * 0.5
            return abs(avg_reward) * 0.5 if avg_reward != 0 else 0.5
        
        return 0.5  # Default uncertainty for untried actions
    
    def _calculate_sample_variance(self, action: Any) -> float:
        """
        Calculate sample variance for an action.
        
        Parameters
        ----------
        action : Any
            The action to calculate variance for
            
        Returns
        -------
        float
            Sample variance
        """
        std = self._get_sample_std(action)
        return std ** 2
    
    def get_best_action(self) -> Optional[Any]:
        """
        Get the action with the highest average reward.
        
        Returns
        -------
        Any or None
            The best action found so far, or None if no actions have been tried
        """
        if not self.action_counts:
            return None
        
        best_action = None
        best_avg_reward = float('-inf')
        
        for action in self.action_counts:
            if self.action_counts[action] > 0:
                # Use action_rewards if available, otherwise use value_estimates
                if hasattr(self, 'action_rewards') and action in self.action_rewards:
                    avg_reward = self.action_rewards[action] / self.action_counts[action]
                elif hasattr(self, 'value_estimates') and action in self.value_estimates:
                    avg_reward = self.value_estimates[action]
                else:
                    continue
                    
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_action = action
        
        return best_action
    
    def get_confidence_threshold_met(self, confidence_threshold: float = 0.95) -> bool:
        """
        Check if the confidence threshold has been met for the best action.
        
        Uses normalized confidence interval width to determine convergence:
        - Narrow interval (width < 0.1) indicates high confidence
        - Wide interval indicates more exploration needed
        
        Parameters
        ----------
        confidence_threshold : float
            Required confidence level (default 0.95 for 95% confidence)
            
        Returns
        -------
        bool
            True if confidence threshold is met, False otherwise
        """
        best_action = self.get_best_action()
        if best_action is None:
            return False
        
        # Unified minimum samples requirement
        MIN_SAMPLES = 10
        if self.action_counts[best_action] < MIN_SAMPLES:
            return False
        
        # Calculate normalized confidence interval
        ci_lower, ci_upper = self._calculate_confidence_interval(best_action, confidence_threshold)
        ci_width = ci_upper - ci_lower
        
        # Unified threshold: CI width < 0.1 on normalized [0,1] scale
        # This corresponds to ~10% relative uncertainty
        CONVERGENCE_THRESHOLD = 0.1
        return ci_width < CONVERGENCE_THRESHOLD
