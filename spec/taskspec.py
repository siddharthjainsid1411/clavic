# spec/taskspec.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional


@dataclass
class Clause:
    operator: str          # "eventually", "always", "until", "always_during", "eventually_during"
    predicate: str         # name of predicate (or tuple of two names for "until")
    weight: float          # logic weight
    modality: str          # "REQUIRE" or "PREFER"
    parameters: Dict[str, Any] = field(default_factory=dict)
    time_window: Optional[Tuple[float, float]] = None   # [t_start, t_end] for *_during ops


@dataclass
class TaskSpec:
    horizon_sec: float
    clauses: List[Clause]
    auxiliary_weights: Dict[str, float] = field(default_factory=dict)