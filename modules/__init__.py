"""
EVPlacement Modules

This package contains the core modules for the streamlined EV charging station placement pipeline:

1. MLPrediction: Demand prediction using portable OSM features
   - Predicts charging demand across city grids
   - Uses only universally available OpenStreetMap data
   - Enables global deployment and portability

2. RLOptimization: Reinforcement learning for tactical placement optimization
   - Multi-armed bandit algorithms for location optimization
   - Optimization framework for station placement within grids
   - Realistic simulation environment

3. utils: Utility modules for common operations
   - City gridding and spatial operations
   - Data processing and transformation
   - Visualization helpers

4. config: Configuration management
   - Simulation parameters
   - EV characteristics and constraints
   - System-wide settings

The pipeline follows a two-stage approach:
- Strategic Allocation: ML-based demand prediction allocates stations across city grids
- Tactical Placement: RL optimization places stations at optimal locations within grids
"""
