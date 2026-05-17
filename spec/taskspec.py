# spec/taskspec.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional


@dataclass
class Clause:
    operator: str          # "eventually", "always", "until", "always_during", "eventually_during"
    predicate: str         # name of predicate (or tuple of two names for "until")
    weight: float          # logic weight
    modality: str          # "HARD" or "SOFT" after parser normalization
    #   HARD    — runtime/structural safety guarantee:
    #               Layer 1: DMP repulsive forcing inside ODE
    #               Layer 2: projector / CBF / CGMS hard mechanism
    #               Layer 3: diagnostic slack penalty in optimizer
    #             The JSON parser extracts obstacle specs automatically; runtime
    #             predicates such as VelocityLimit are wired by the policy.
    #   SOFT    — weighted wTLTL robustness cost; can trade off against
    #             other objectives. Legacy REQUIRE/PREFER JSON labels are
    #             normalized to SOFT by json_parser.
    parameters: Dict[str, Any] = field(default_factory=dict)
    time_window: Optional[Tuple[float, float]] = None   # [t_start, t_end] for *_during ops
    # For HARD clauses only: obstacle spec extracted by json_parser
    hard_obstacle: Optional[Dict[str, Any]] = None


@dataclass
class TaskSpec:
    horizon_sec: float
    clauses: List[Clause]
    auxiliary_weights: Dict[str, float] = field(default_factory=dict)
    # Populated by json_parser from all HARD clauses — fed to set_obstacles()
    hard_obstacle_specs: List[Dict[str, Any]] = field(default_factory=list)
