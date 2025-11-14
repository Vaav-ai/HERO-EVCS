"""
GPU Utilities for EV Demand Prediction Pipeline
===============================================

This module provides utilities for GPU detection, configuration, and 
GPU-accelerated machine learning operations.
"""

import logging
import platform
import subprocess
import sys
from typing import Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)

def detect_gpu_availability() -> Dict[str, Any]:
    """
    Detect available GPU resources and capabilities.
    
    Returns:
        Dictionary containing GPU availability information
    """
    gpu_info = {
        'cuda_available': False,
        'gpu_count': 0,
        'gpu_names': [],
        'rapids_available': False,
        'xgboost_gpu': False,
        'lightgbm_gpu': False,
        'catboost_gpu': False,
        'tensorflow_gpu': False
    }
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['cuda_available'] = True
            gpu_info['gpu_count'] = torch.cuda.device_count()
            gpu_info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(gpu_info['gpu_count'])]
    except ImportError:
        # Try nvidia-ml-py as fallback
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_info['cuda_available'] = True
            gpu_info['gpu_count'] = pynvml.nvmlDeviceGetCount()
            gpu_info['gpu_names'] = [
                pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(i)).decode('utf-8')
                for i in range(gpu_info['gpu_count'])
            ]
        except (ImportError, Exception):
            # Try nvidia-smi as final fallback
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    gpu_info['cuda_available'] = True
                    gpu_names = [name.strip() for name in result.stdout.strip().split('\n') if name.strip()]
                    gpu_info['gpu_count'] = len(gpu_names)
                    gpu_info['gpu_names'] = gpu_names
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                pass
    
    # Check RAPIDS availability
    try:
        import cudf
        import cuml
        gpu_info['rapids_available'] = True
    except ImportError:
        pass
    
    # Check XGBoost GPU support
    try:
        import xgboost as xgb
        # Test if GPU training is available
        if gpu_info['cuda_available']:
            try:
                # Try to create a GPU-enabled booster to test
                import pandas as pd
                import numpy as np
                test_data = pd.DataFrame({'x': np.random.rand(100), 'y': np.random.rand(100)})
                dtrain = xgb.DMatrix(test_data[['x']], label=test_data['y'])
                params = {'tree_method': 'hist', 'device': 'cuda:0'}
                xgb.train(params, dtrain, num_boost_round=1, verbose_eval=False)
                gpu_info['xgboost_gpu'] = True
            except Exception:
                pass
    except ImportError:
        pass
    
    # Check LightGBM GPU support
    try:
        import lightgbm as lgb
        if gpu_info['cuda_available']:
            gpu_info['lightgbm_gpu'] = True
    except ImportError:
        pass
    
    # Check CatBoost GPU support
    try:
        import catboost
        if gpu_info['cuda_available']:
            gpu_info['catboost_gpu'] = True
    except ImportError:
        pass
    
    # Check TensorFlow GPU support
    try:
        import tensorflow as tf
        if len(tf.config.list_physical_devices('GPU')) > 0:
            gpu_info['tensorflow_gpu'] = True
    except ImportError:
        pass
    
    return gpu_info

def get_gpu_config(use_gpu: bool = True, gpu_id: int = 0) -> Dict[str, Any]:
    """
    Get GPU configuration parameters for different ML libraries.
    
    Args:
        use_gpu: Whether to enable GPU usage
        gpu_id: GPU device ID to use
        
    Returns:
        Dictionary containing GPU configuration for each library
    """
    gpu_info = detect_gpu_availability()
    
    if not use_gpu or not gpu_info['cuda_available']:
        return {
            'use_gpu': False,
            'xgboost_params': {},
            'lightgbm_params': {},
            'catboost_params': {},
            'rapids_enabled': False
        }
    
    config = {
        'use_gpu': True,
        'gpu_id': gpu_id,
        'rapids_enabled': gpu_info['rapids_available']
    }
    
    # XGBoost GPU parameters - with fallback to CPU if GPU issues
    if gpu_info['xgboost_gpu']:
        try:
            # Test GPU availability before enabling GPU mode
            import pandas as pd
            import numpy as np
            test_data = pd.DataFrame({'x': np.random.rand(10), 'y': np.random.rand(10)})
            dtrain = xgb.DMatrix(test_data[['x']], label=test_data['y'])
            params = {'tree_method': 'hist', 'device': f'cuda:{gpu_id}'}
            xgb.train(params, dtrain, num_boost_round=1, verbose_eval=False)
            # If we get here, GPU is available
            config['xgboost_params'] = {
                'tree_method': 'hist',
                'device': f'cuda:{gpu_id}',
                'predictor': 'gpu_predictor'
            }
        except Exception as e:
            logger.warning(f"XGBoost GPU test failed: {e}. Falling back to CPU mode.")
            config['xgboost_params'] = {
                'tree_method': 'hist',
                'device': 'cpu'
            }
    else:
        config['xgboost_params'] = {
            'tree_method': 'hist',
            'device': 'cpu'
        }
    
    # LightGBM GPU parameters
    if gpu_info['lightgbm_gpu']:
        config['lightgbm_params'] = {
            'device': 'gpu',
            'gpu_device_id': gpu_id,
            'gpu_use_dp': False  # Use single precision for speed
        }
    else:
        config['lightgbm_params'] = {}
    
    # CatBoost GPU parameters
    if gpu_info['catboost_gpu']:
        config['catboost_params'] = {
            'task_type': 'GPU',
            'devices': f'{gpu_id}',
            'gpu_ram_part': 0.8  # Use 80% of GPU memory
        }
    else:
        config['catboost_params'] = {}
    
    return config

def log_gpu_status(gpu_info: Dict[str, Any] = None):
    """Log GPU availability and configuration status."""
    if gpu_info is None:
        gpu_info = detect_gpu_availability()
    
    if gpu_info['cuda_available']:
        logger.info("🚀 GPU ACCELERATION AVAILABLE!")
        logger.info(f"   - GPU Count: {gpu_info['gpu_count']}")
        for i, name in enumerate(gpu_info['gpu_names']):
            logger.info(f"   - GPU {i}: {name}")
        
        # Log available accelerated libraries
        accelerated_libs = []
        if gpu_info['xgboost_gpu']:
            accelerated_libs.append("XGBoost")
        if gpu_info['lightgbm_gpu']:
            accelerated_libs.append("LightGBM") 
        if gpu_info['catboost_gpu']:
            accelerated_libs.append("CatBoost")
        if gpu_info['rapids_available']:
            accelerated_libs.append("RAPIDS")
        if gpu_info['tensorflow_gpu']:
            accelerated_libs.append("TensorFlow")
            
        if accelerated_libs:
            logger.info(f"   - GPU-Accelerated Libraries: {', '.join(accelerated_libs)}")
        else:
            logger.warning("   - No GPU-accelerated ML libraries detected")
    else:
        logger.info("💻 Using CPU-only computation (no GPU detected)")

def setup_gpu_memory_growth():
    """Setup GPU memory growth for TensorFlow to avoid memory issues."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("✅ GPU memory growth enabled for TensorFlow")
    except (ImportError, Exception) as e:
        logger.debug(f"Could not setup GPU memory growth: {e}")

def get_optimal_gpu_batch_size(dataset_size: int, feature_count: int, gpu_memory_gb: float = None) -> int:
    """
    Estimate optimal batch size for GPU training based on dataset and GPU memory.
    
    Args:
        dataset_size: Number of samples in dataset
        feature_count: Number of features
        gpu_memory_gb: Available GPU memory in GB (auto-detected if None)
        
    Returns:
        Recommended batch size
    """
    if gpu_memory_gb is None:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            else:
                gpu_memory_gb = 8.0  # Assume 8GB default
        except ImportError:
            gpu_memory_gb = 8.0
    
    # Rough estimation: each sample with features takes ~feature_count * 4 bytes (float32)
    # Add overhead for gradients, intermediate calculations, etc.
    bytes_per_sample = feature_count * 4 * 4  # 4x overhead
    available_memory = gpu_memory_gb * 1024**3 * 0.7  # Use 70% of GPU memory
    
    max_batch_size = int(available_memory / bytes_per_sample)
    
    # Clamp to reasonable bounds
    batch_size = min(max_batch_size, dataset_size, 50000)  # Max 50k samples per batch
    batch_size = max(batch_size, 1000)  # Min 1k samples per batch
    
    logger.info(f"🎯 Estimated optimal GPU batch size: {batch_size:,} samples")
    return batch_size
