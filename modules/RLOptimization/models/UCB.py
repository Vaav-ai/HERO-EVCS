"""
Upper Confidence Bound (UCB) Bandit Algorithm

Implements the UCB1 algorithm for multi-armed bandit optimization.
UCB balances exploration and exploitation by selecting actions with high upper confidence bounds.

Mathematical Foundation:
- UCB1 selects action: argmax_i [μ_i + c * sqrt(2*ln(t) / n_i)]
- where μ_i is the average reward of action i
- n_i is the number of times action i has been selected
- t is the total number of pulls
- c is the exploration parameter (typically 1.0)

The algorithm provides theoretical guarantees on regret bounds and is particularly
effective for EV charging station placement where we need to balance exploration
of new placement configurations with exploitation of known good ones.
"""

from .BaseBandit import BaseBandit
from collections import defaultdict
import numpy as np
import math
from typing import Any, List, Optional


class UCB(BaseBandit):
    """
    Upper Confidence Bound (UCB1) bandit algorithm implementation.
    
    UCB1 is a well-established bandit algorithm that provides strong theoretical
    guarantees on regret bounds. It's particularly suitable for EV charging station
    placement optimization where we need to balance exploration and exploitation.
    """
    
    def __init__(self, exploration_parameter: float = 1.0, min_pulls_per_action: int = 1, 
                 max_consecutive_same: Optional[int] = None, confidence_threshold: float = 0.05, 
                 min_confidence_samples: int = 20, seed: int = None):
        """
        Initialize UCB1 bandit algorithm.
        
        Parameters
        ----------
        exploration_parameter : float, optional
            Exploration parameter 'c' in UCB formula. Higher values encourage more exploration.
            Default is 1.0, which provides good balance between exploration and exploitation.
        min_pulls_per_action : int, optional
            Minimum number of times each action must be pulled before UCB formula applies.
            Default is 1 (pure UCB1), but can be increased for more conservative exploration.
        max_consecutive_same : int, optional
            Maximum consecutive pulls of the same action. If None, no limit (pure UCB1).
            Default is None for theoretical correctness.
        confidence_threshold : float, optional
            Relative confidence threshold for convergence check. Default is 0.05.
        min_confidence_samples : int, optional
            Minimum samples required for confidence check. Default is 20.
        seed : int, optional
            Random seed for reproducibility.
        """
        super().__init__(seed=seed)
        self.exploration_parameter = exploration_parameter
        self.min_pulls_per_action = min_pulls_per_action
        self.max_consecutive_same = max_consecutive_same
        self.confidence_threshold = confidence_threshold
        self.min_confidence_samples = min_confidence_samples
        self.action_rewards = defaultdict(float)
        self.action_counts = defaultdict(int)
        self.total_pulls = 0
        self.last_selected_action = None
        self.consecutive_same_action = 0
        self.initialization_phase = True
        self.initialization_order = []
        
    def select_action(self, available_actions: Optional[List[Any]] = None) -> Any:
        """
        Select action using UCB1 algorithm with diversity mechanisms.
        
        The UCB1 algorithm selects the action with the highest upper confidence bound:
        action = argmax_i [μ_i + c * sqrt(2*ln(t) / n_i)]
        
        Parameters
        ----------
        available_actions : List[Any], optional
            List of available actions (placement configurations). If None, uses all actions
            that have been seen before.
            
        Returns
        -------
        Any
            Selected action based on UCB1 criterion
        """
        if available_actions is None:
            # Use all actions that have been tried at least once
            available_actions = list(self.action_counts.keys())
        
        if not available_actions:
            return None
        
        # Initialize action order for round-robin if first time
        if self.initialization_phase and not self.initialization_order:
            self.initialization_order = available_actions.copy()
        
        # Round-robin initialization phase
        if self.initialization_phase:
            # Find actions that haven't been tried enough times
            under_tried_actions = [a for a in available_actions 
                                 if self.action_counts[a] < self.min_pulls_per_action]
            
            if under_tried_actions:
                # Round-robin through under-tried actions
                for action in self.initialization_order:
                    if action in under_tried_actions:
                        self.last_selected_action = action
                        self.consecutive_same_action = 1
                        return action
                # If no action in order is under-tried, pick the least tried
                least_tried = min(under_tried_actions, key=lambda x: self.action_counts[x])
                self.last_selected_action = least_tried
                self.consecutive_same_action = 1
                return least_tried
            else:
                # All actions have been tried enough, exit initialization
                self.initialization_phase = False
        
        # Check for consecutive same action - force exploration (if enabled)
        if (self.max_consecutive_same is not None and
            self.last_selected_action is not None and 
            self.last_selected_action in available_actions and
            self.consecutive_same_action >= self.max_consecutive_same):
            # Force exploration by selecting a different action
            other_actions = [a for a in available_actions if a != self.last_selected_action]
            if other_actions:
                # Select least tried action among others
                least_tried = min(other_actions, key=lambda x: self.action_counts[x])
                self.last_selected_action = least_tried
                self.consecutive_same_action = 1
                return least_tried
        
        # Apply UCB1 formula
        ucb_values = {}
        for action in available_actions:
            if self.action_counts[action] > 0:
                # Calculate average reward
                avg_reward = self.action_rewards[action] / self.action_counts[action]
                
                # Calculate confidence bound with numerical stability
                confidence_bound = self.exploration_parameter * math.sqrt(
                    2 * math.log(max(1, self.total_pulls)) / self.action_counts[action]
                )
                
                # UCB value = average reward + confidence bound
                ucb_values[action] = avg_reward + confidence_bound
            else:
                # If action hasn't been tried, give it maximum UCB value
                ucb_values[action] = float('inf')
        
        # Select action with highest UCB value
        selected_action = max(ucb_values.items(), key=lambda x: x[1])[0]
        
        # Track consecutive same action
        if selected_action == self.last_selected_action:
            self.consecutive_same_action += 1
        else:
            self.consecutive_same_action = 1
            self.last_selected_action = selected_action
        
        return selected_action
    
    def update(self, action: Any, reward: float) -> None:
        """
        Update UCB algorithm with reward from selected action.
        
        Parameters
        ----------
        action : Any
            The action that was selected and executed
        reward : float
            The reward received from executing the action
        """
        # Parent class update handles all the tracking (counts, rewards, history)
        super().update(action, reward)
    
    def get_ucb_values(self, available_actions: Optional[List[Any]] = None) -> dict:
        """
        Get UCB values for all available actions.
        
        Parameters
        ----------
        available_actions : List[Any], optional
            List of actions to calculate UCB values for
            
        Returns
        -------
        dict
            Dictionary mapping actions to their UCB values
        """
        if available_actions is None:
            available_actions = list(self.action_counts.keys())
        
        ucb_values = {}
        for action in available_actions:
            if self.action_counts[action] >= self.min_pulls_per_action:
                avg_reward = self.action_rewards[action] / self.action_counts[action]
                confidence_bound = self.exploration_parameter * math.sqrt(
                    2 * math.log(self.total_pulls) / self.action_counts[action]
                )
                ucb_values[action] = avg_reward + confidence_bound
            else:
                ucb_values[action] = float('inf')
        
        return ucb_values
    
    def get_exploration_bonus(self, action: Any) -> float:
        """
        Get the exploration bonus (confidence bound) for a specific action.
        
        Parameters
        ----------
        action : Any
            The action to calculate exploration bonus for
            
        Returns
        -------
        float
            The exploration bonus (confidence bound) for the action
        """
        if self.action_counts[action] < self.min_pulls_per_action:
            return float('inf')
        
        return self.exploration_parameter * math.sqrt(
            2 * math.log(self.total_pulls) / self.action_counts[action]
        )
    
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
        if self.action_counts[action] == 0:
            return 0.0
        
        return self.action_rewards[action] / self.action_counts[action]
    
    def get_regret_bound(self) -> float:
        """
        Calculate theoretical regret bound for UCB1.
        
        The regret bound for UCB1 is O(sqrt(K * T * log(T))) where K is the number of arms
        and T is the total number of pulls.
        
        Returns
        -------
        float
            Theoretical regret bound
        """
        K = len(self.action_counts)  # Number of arms
        T = self.total_pulls  # Total pulls
        
        if T == 0:
            return 0.0
        
        return 8 * math.sqrt(K * T * math.log(T))
    
    def get_confidence_threshold_met(self, confidence_threshold: float = 0.95) -> bool:
        """
        Check if confidence threshold is met for UCB algorithm.
        
        Uses the unified confidence checking approach from BaseBandit.
        For UCB, we additionally check that the exploration bonus (confidence bound)
        is sufficiently small relative to the reward.
        
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
        
        best_action = self.get_best_action()
        if best_action is None:
            return False
        
        # Unified minimum samples requirement
        MIN_SAMPLES = 10
        if self.action_counts[best_action] < MIN_SAMPLES:
            return False
        
        # Use base class confidence interval check (unified approach)
        ci_lower, ci_upper = self._calculate_confidence_interval(best_action, confidence_threshold)
        ci_width = ci_upper - ci_lower
        
        # Unified convergence threshold
        CONVERGENCE_THRESHOLD = 0.1
        return ci_width < CONVERGENCE_THRESHOLD
