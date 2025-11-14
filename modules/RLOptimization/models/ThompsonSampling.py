"""
Thompson Sampling Bandit Algorithm

Implements Thompson Sampling for multi-armed bandit optimization using Beta distribution.
Thompson Sampling is a Bayesian approach that maintains a probability distribution
over the expected rewards of each action and samples from these distributions.

Mathematical Foundation:
- For each action i, maintain Beta(α_i, β_i) distribution
- α_i = number of successes + 1
- β_i = number of failures + 1
- Sample θ_i ~ Beta(α_i, β_i) for each action
- Select action with highest sampled reward: argmax_i θ_i

Thompson Sampling provides excellent empirical performance and has strong theoretical guarantees.
"""

from .BaseBandit import BaseBandit
from collections import defaultdict
import numpy as np
from scipy.stats import beta
import random
from typing import Any, List, Optional, Dict


class ThompsonSampling(BaseBandit):
    """
    Thompson Sampling bandit algorithm implementation using Beta distribution.
    
    Thompson Sampling is a Bayesian approach that maintains probability distributions
    over the expected rewards of each action. It samples from these distributions
    to select actions, naturally balancing exploration and exploitation.
    """
    
    def __init__(self, success_threshold: float = 0.0, max_consecutive_same: Optional[int] = None, seed: int = None):
        """
        Initialize Thompson Sampling bandit algorithm using Beta distribution.
        
        Parameters
        ----------
        success_threshold : float, optional
            Threshold for considering a reward as a "success". Rewards above this
            threshold are treated as successes, below as failures. Default is 0.0.
        max_consecutive_same : int, optional
            Maximum consecutive pulls of the same action. If None, no limit.
            Default is None for theoretical correctness.
        seed : int, optional
            Random seed for reproducibility.
        """
        super().__init__(seed=seed)
        self.success_threshold = success_threshold
        self.max_consecutive_same = max_consecutive_same
        self.alphas = defaultdict(lambda: 1.0)  # Success count + 1 (Beta prior)
        self.betas = defaultdict(lambda: 1.0)   # Failure count + 1 (Beta prior)
        self.last_selected_action = None
        self.consecutive_same_action = 0
        
    def select_action(self, available_actions: Optional[List[Any]] = None) -> Any:
        """
        Select action using Thompson Sampling with Beta distribution.
        
        Parameters
        ----------
        available_actions : List[Any], optional
            List of available actions. If None, uses all actions that have been seen.
            
        Returns
        -------
        Any
            Selected action based on Thompson Sampling
        """
        if available_actions is None:
            available_actions = list(self.alphas.keys())
            
        if not available_actions:
            return None
        
        # Check for consecutive same action - force exploration (if enabled)
        if (self.max_consecutive_same is not None and
            self.last_selected_action is not None and 
            self.last_selected_action in available_actions and
            self.consecutive_same_action >= self.max_consecutive_same):
            # Force exploration by selecting a different action
            other_actions = [a for a in available_actions if a != self.last_selected_action]
            if other_actions:
                # Sample from Beta distribution for other actions only
                samples = {
                    action: beta.rvs(self.alphas[action], self.betas[action])
                    for action in other_actions
                }
                selected_action = max(samples.items(), key=lambda x: x[1])[0]
                self.last_selected_action = selected_action
                self.consecutive_same_action = 1
                return selected_action
            
        # Sample from Beta distribution for each action
        samples = {
            action: beta.rvs(self.alphas[action], self.betas[action])
            for action in available_actions
        }
        
        # Select action with highest sampled reward
        selected_action = max(samples.items(), key=lambda x: x[1])[0]
        
        # Track consecutive same action
        if selected_action == self.last_selected_action:
            self.consecutive_same_action += 1
        else:
            self.consecutive_same_action = 1
            self.last_selected_action = selected_action
        
        return selected_action
        
    def update(self, action: Any, reward: float) -> None:
        """
        Update Beta distribution parameters based on reward.
        
        Parameters
        ----------
        action : Any
            The action that was selected
        reward : float
            The reward received for taking the action
        """
        # CRITICAL FIX: Initialize tracking dictionaries if not present
        if action not in self.action_counts:
            self.action_counts[action] = 0
        if action not in self.action_rewards:
            self.action_rewards[action] = 0.0
        
        # Update parent class tracking FIRST
        super().update(action, reward)
        
        # Then update Beta distribution parameters
        # Convert reward to success/failure based on threshold
        if reward > self.success_threshold:
            # Treat as success - increment alpha
            self.alphas[action] += 1.0
        else:
            # Treat as failure - increment beta
            self.betas[action] += 1.0
    
    def get_beta_parameters(self, action: Any) -> tuple:
        """
        Get Beta distribution parameters for a specific action.
        
        Parameters
        ----------
        action : Any
            The action to get parameters for
            
        Returns
        -------
        tuple
            (alpha, beta) parameters for the Beta distribution
        """
        return (self.alphas[action], self.betas[action])
    
    def get_expected_reward(self, action: Any) -> float:
        """
        Get the expected reward (mean of Beta distribution) for a specific action.
        
        Parameters
        ----------
        action : Any
            The action to calculate expected reward for
            
        Returns
        -------
        float
            Expected reward (mean of Beta distribution)
        """
        alpha = self.alphas[action]
        beta = self.betas[action]
        return alpha / (alpha + beta)
    
    def get_confidence_interval(self, action: Any, confidence_level: float = 0.95) -> tuple:
        """
        Get normalized confidence interval for the expected reward of an action.
        
        This uses the Beta distribution posterior to compute confidence intervals,
        which is then normalized to the same [0, 1] scale as other bandit algorithms.
        
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
        a = self.alphas[action]
        b = self.betas[action]
        
        # Handle edge cases
        if a <= 0 or b <= 0:
            return (0.0, 1.0)  # Maximum uncertainty
        
        # Calculate percentiles for confidence interval
        lower_percentile = (1 - confidence_level) / 2
        upper_percentile = 1 - lower_percentile
        
        try:
            # Get beta distribution bounds (naturally in [0,1])
            beta_lower = beta.ppf(lower_percentile, a, b)
            beta_upper = beta.ppf(upper_percentile, a, b)
            
            # Beta distribution width as measure of uncertainty
            confidence_width = beta_upper - beta_lower
            
            # Map to normalized [0, 1] confidence interval
            # Narrow interval = high confidence, wide interval = low confidence
            center = 0.5
            lower_bound = max(0.0, center - confidence_width / 2)
            upper_bound = min(1.0, center + confidence_width / 2)
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            # Fallback for numerical issues
            return (0.0, 1.0)
    
    def get_exploration_bonus(self, action: Any) -> float:
        """
        Get the exploration bonus for a specific action.
        
        For Thompson Sampling, the exploration bonus is related to the uncertainty
        in the posterior distribution (variance of the beta distribution).
        
        Parameters
        ----------
        action : Any
            The action to calculate exploration bonus for
            
        Returns
        -------
        float
            The exploration bonus (uncertainty measure)
        """
        if action not in self.alphas:
            return 1.0  # Maximum uncertainty for untried actions
        
        a = self.alphas[action]
        b = self.betas[action]
        
        # Calculate variance of beta distribution as exploration bonus
        # Higher variance = more uncertainty = higher exploration bonus
        variance = (a * b) / ((a + b) ** 2 * (a + b + 1))
        return variance
    
    def get_action_statistics(self) -> Dict[Any, Dict[str, float]]:
        """
        Get statistics for all actions that have been tried.
        
        Returns
        -------
        Dict[Any, Dict[str, float]]
            Dictionary mapping actions to their statistics (count, average reward, etc.)
        """
        stats = {}
        for action in self.alphas:
            if action in self.action_counts and self.action_counts[action] > 0:
                # Use base class statistics if available
                total_reward = self.action_rewards.get(action, 0.0)
                avg_reward = total_reward / self.action_counts[action]
                
                stats[action] = {
                    'count': self.action_counts[action],
                    'total_reward': total_reward,
                    'average_reward': avg_reward,
                    'confidence_interval': self.get_confidence_interval(action),
                    'expected_reward': self.get_expected_reward(action),
                    'alpha': self.alphas[action],
                    'beta': self.betas[action]
                }
        return stats
    
    def get_confidence_threshold_met(self, confidence_threshold: float = 0.95) -> bool:
        """
        Check if confidence threshold is met for Thompson Sampling algorithm.
        
        Uses the unified confidence checking approach, leveraging the Beta distribution
        posterior to compute confidence intervals.
        
        Parameters
        ----------
        confidence_threshold : float, optional
            Required confidence level (0.95 = 95% confidence). Default is 0.95.
            
        Returns
        -------
        bool
            True if confidence threshold is met, False otherwise
        """
        if not self.alphas:
            return False
        
        best_action = self.get_best_action()
        if best_action is None:
            return False
        
        # Unified minimum samples requirement
        MIN_SAMPLES = 10
        if self.action_counts.get(best_action, 0) < MIN_SAMPLES:
            return False
        
        # Use Thompson Sampling's Beta-based confidence interval
        ci_lower, ci_upper = self.get_confidence_interval(best_action, confidence_threshold)
        ci_width = ci_upper - ci_lower
        
        # Unified convergence threshold
        CONVERGENCE_THRESHOLD = 0.1
        return ci_width < CONVERGENCE_THRESHOLD