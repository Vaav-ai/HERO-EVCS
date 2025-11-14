#!/usr/bin/env python3
"""
Seed Utilities for EV Placement Pipeline

This module provides consistent seed setting functions across all modules
to ensure reproducibility without circular import issues.
"""

import os
import random
import numpy as np
import logging


def set_global_seeds(seed: int = 42):
    """
    Set all random seeds for complete reproducibility.
    
    Args:
        seed: Random seed for reproducibility
    """
    # Python random module
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # Set environment variable for SUMO
    os.environ['SUMO_RANDOM_SEED'] = str(seed)
    
    # Set additional environment variables for SUMO reproducibility
    os.environ['SUMO_DISABLE_TRACI_FQ'] = '1'
    os.environ['SUMO_DISABLE_AUTO_DEPART'] = '1'
    
    # Set random state for pandas (uses numpy's random state)
    # pandas uses numpy's random state, so no additional setup needed
    
    # Set random state for scipy.stats if used
    try:
        import scipy.stats as stats
        if hasattr(stats, 'set_random_state'):
            stats.set_random_state(np.random.RandomState(seed))
    except ImportError:
        pass
    
    # Set random state for sklearn if used
    try:
        from sklearn.utils import check_random_state
        check_random_state(seed)
    except ImportError:
        pass
    
    # Set random state for tensorflow if used
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    # Set random state for torch if used
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    print(f"🌱 Global seeds set to {seed} for reproducibility")


def validate_seed_consistency(seed: int, logger=None):
    """
    Validate that seed is properly set across all modules.
    
    Args:
        seed: Expected seed value
        logger: Logger instance for logging
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("🔍 Validating seed consistency...")
    
    # Check numpy random state
    np_state = np.random.get_state()
    logger.info(f"  NumPy random state: {np_state[1][0]} (first value)")
    
    # Check Python random state
    random_state = random.getstate()
    logger.info(f"  Python random state: {random_state[1][0]} (first value)")
    
    # Check environment variables
    sumo_seed = os.environ.get('SUMO_RANDOM_SEED', 'Not set')
    logger.info(f"  SUMO random seed: {sumo_seed}")
    
    logger.info("✅ Seed validation completed")

