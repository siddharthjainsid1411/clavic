# spec/taskspec.py

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Clause:
    operator: str          # "eventually", "always", "until"
    predicate: str         # name of predicate
    weight: float          # logic weight
    modality: str          # "REQUIRE" or "PREFER"
    parameters: Dict[str, Any] = field(default_factory=dict)
    deadline_sec: float = None   # optional: evaluate only over [0, deadline_sec]
                                 # None means use full trajectory (normal behaviour)


@dataclass
class TaskSpec:
    horizon_sec: float
    clauses: List[Clause]
    auxiliary_weights: Dict[str, float] = field(default_factory=dict)