"""
Epsilon-Greedy Bandit Algorithm

Implements the epsilon-greedy algorithm for multi-armed bandit optimization.
With probability epsilon, the algorithm explores (selects a random action),
otherwise it exploits (selects the action with highest average reward).

Mathematical Foundation:
- With probability ε: select random action (exploration)
- With probability (1-ε): select argmax_i μ_i (exploitation)
- where μ_i is the average reward of action i

The algorithm is simple but effective, especially when combined with epsilon decay
to gradually shift from exploration to exploitation.
"""

from .BaseBandit import BaseBandit
from collections import defaultdict
import numpy as np
import random
from typing import Any, List, Optional


class EpsilonGreedy(BaseBandit):
    """
    Epsilon-Greedy bandit algorithm implementation.
    
    The epsilon-greedy algorithm is one of the simplest and most widely used
    bandit algorithms. It balances exploration and exploitation by randomly
    selecting actions with probability epsilon, and greedily selecting the
    best-known action otherwise.
    """
    
    def __init__(self, epsilon: float = 1.0, epsilon_decay: float = 0.99, min_epsilon: float = 0.1,
                 max_consecutive_same: Optional[int] = None, seed: int = None):
        """
        Initialize EpsilonGreedy bandit algorithm.

        Parameters
        ----------
        epsilon : float, optional
            Probability of exploration. Defaults to 1.0 (pure exploration initially).
        epsilon_decay : float, optional
            Rate of decay for epsilon. Defaults to 0.99.
        min_epsilon : float, optional
            Minimum value for epsilon. Defaults to 0.1.
        max_consecutive_same : int, optional
            Maximum consecutive pulls of the same action. If None, no limit (pure epsilon-greedy).
            Default is None for theoretical correctness.
        seed : int, optional
            Random seed for reproducibility.

        """
        super().__init__(seed=seed)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.max_consecutive_same = max_consecutive_same
        self.value_estimates = defaultdict(float)
        self.action_counts = defaultdict(int)
        self.best_action = None
        self.best_reward = float('-inf')
        self.last_selected_action = None
        self.consecutive_same_action = 0
    
    def select_action(self, available_actions: Optional[List[Any]] = None) -> Any:
        """
        Select action based on epsilon-greedy policy with diversity mechanisms.

        Parameters
        ----------
        available_actions : List[Any], optional
            List of available actions. Defaults to None.

        Returns
        -------
        Any
            Selected action.

        """
        if available_actions is None:
            available_actions = list(self.action_counts.keys())
        
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
                selected_action = random.choice(other_actions)
                self.last_selected_action = selected_action
                self.consecutive_same_action = 1
                return selected_action
        
        # Normal epsilon-greedy selection
        if np.random.random() < self.epsilon:
            selected_action = random.choice(available_actions)
        else:
            selected_action = self.best_action if self.best_action is not None else random.choice(available_actions)
        
        # Track consecutive same action
        if selected_action == self.last_selected_action:
            self.consecutive_same_action += 1
        else:
            self.consecutive_same_action = 1
            self.last_selected_action = selected_action
        
        return selected_action
    
    def update(self, action: Any, reward: float) -> None:
        """
        Update the value estimates and exploration probability based on the given action and reward.

        Parameters
        ----------
        action : Any
            The action that was selected.
        reward : float
            The reward received for taking the action.

        """
        # Parent class update handles counts and reward history
        super().update(action, reward)
        
        # Update value estimate using incremental average
        # Note: action_counts is already incremented in parent
        n = self.action_counts[action]
        self.value_estimates[action] += (reward - self.value_estimates[action]) / n
        
        # Update best action based on current value estimates (not single reward)
        self.best_action = max(self.value_estimates, key=self.value_estimates.get)
        self.best_reward = self.value_estimates[self.best_action]
        
        # Decay epsilon to gradually shift from exploration to exploitation
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def get_exploration_rate(self) -> float:
        """
        Get the current exploration rate (epsilon).
        
        Returns
        -------
        float
            Current exploration rate
        """
        return self.epsilon
    
    def get_exploitation_value(self, action: Any) -> float:
        """
        Get the exploitation value (average reward) for a specific action.
        
        Parameters
        ----------
        action : Any
            The action to calculate exploitation value for
            
        Returns
        -------
        float
            The average reward for the action
        """
        if action not in self.value_estimates:
            return 0.0
        return self.value_estimates[action]
    
    def get_exploration_bonus(self, action: Any) -> float:
        """
        Get the exploration bonus for a specific action.
        
        For epsilon-greedy, the exploration bonus is the current epsilon value,
        representing the probability of exploration.
        
        Parameters
        ----------
        action : Any
            The action to calculate exploration bonus for
            
        Returns
        -------
        float
            The exploration bonus (current epsilon value)
        """
        return self.epsilon
    
    def get_confidence_threshold_met(self, confidence_threshold: float = 0.95) -> bool:
        """
        Check if confidence threshold is met for EpsilonGreedy algorithm.
        
        For epsilon-greedy, we consider the threshold met when:
        1. All actions have been tried at least a minimum number of times
        2. The epsilon value has decayed sufficiently (indicating convergence to exploitation)
        3. The best action is clearly identified
        
        Parameters
        ----------
        confidence_threshold : float, optional
            Required confidence level (0.95 = 95% confidence). Default is 0.95.
            
        Returns
        -------
        bool
            True if confidence threshold is met, False otherwise
        """
        if not self.action_counts:
            return False
        
        # Check if all actions have been tried enough times
        min_tries = 10  # Minimum samples for statistical significance
        if any(self.action_counts[action] < min_tries for action in self.action_counts):
            return False
        
        # Check if epsilon has decayed sufficiently (indicating convergence)
        # Lower epsilon means more exploitation, higher confidence
        epsilon_threshold = 0.2  # Consider converged when epsilon is low (20% exploration or less)
        if self.epsilon > epsilon_threshold:
            return False
        
        # Check if best action is clearly identified
        if self.best_action is None:
            return False
        
        # Check if best action has significantly higher value than others
        best_value = self.value_estimates[self.best_action]
        other_values = [self.value_estimates[action] for action in self.value_estimates 
                       if action != self.best_action]
        
        if not other_values:
            return True
        
        # Check if best action is clearly better than others
        max_other_value = max(other_values)
        value_gap = best_value - max_other_value
        
        # Consider converged if the gap is significant relative to the best value
        if best_value == 0:
            return False
        
        # For 0.95 confidence threshold, require at least 5% gap (1 - 0.95 = 0.05)
        relative_gap = value_gap / abs(best_value)
        return relative_gap > (1 - confidence_threshold)  # e.g., 0.05 for 95% confidence