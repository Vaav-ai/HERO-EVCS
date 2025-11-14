from dataclasses import dataclass

@dataclass
class SimulationConfig:
    """Configuration settings for SUMO simulation
    - Domain randomization options allow sampling driver behavior per run.
    """
    step_length: float = 1.0
    use_gui: bool = False
    battery_precision: int = 4
    battery_probability: float = 1.0
    
    # Domain randomization toggles and ranges
    enable_domain_randomization: bool = False
    domain_randomization_enabled: bool = False  # alias flag for convenience
    speeddev_min: float = 0.10
    speeddev_max: float = 0.30
    mismatch_min: float = 0.10
    mismatch_max: float = 0.40
    lanechange_speedgain_min: float = 1.0
    lanechange_speedgain_max: float = 3.0