from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class EVConfig:
    """Configuration for electric vehicles
    - Optionally provide callables for domain randomization per-vehicle.
    """
    capacity: str = "64000"
    maximum_power: str = "2500000"
    propulsion_efficiency: str = "0.99"
    recuperation_efficiency: str = "0.98"
    air_drag_coefficient: str = "0.28"
    roll_drag_coefficient: str = "0.005"
    mass: str = "1200"

    # Optional domain randomization hooks
    initial_soc_dist: Optional[Callable[[], float]] = None  # returns a value in [0,1]
    capacity_dist: Optional[Callable[[], float]] = None     # returns capacity in Wh