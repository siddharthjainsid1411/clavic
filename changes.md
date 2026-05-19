# Changes Log

## 2026-05-19 00:37 IST

- Added runtime obstacle HOCBF filter (relative-degree-2) with epsilon-inflated safe radius and per-step diagnostics, applied after DMP repulsion and before velocity CBF. Affects [core/cbf_filter.py](core/cbf_filter.py), [core/cgms/dmp_with_gain.py](core/cgms/dmp_with_gain.py), and [core/multi_phase_policy.py](core/multi_phase_policy.py).
- Added projection-activation logging and obstacle HOCBF plots/diagnostics for Exp 2, including projection mask recording. Affects [core/obstacle_projection.py](core/obstacle_projection.py), [core/multi_phase_policy.py](core/multi_phase_policy.py), and [main_exp2.py](main_exp2.py).

## 2026-05-18 00:28 IST

- Added time-windowed VelocityLimit CBF activation and diagnostics (window_active, vmax_active), including time-aware filtering in the velocity CBF path. Affects [core/cbf_filter.py](core/cbf_filter.py), [core/cgms/dmp_with_gain.py](core/cgms/dmp_with_gain.py), and [core/multi_phase_policy.py](core/multi_phase_policy.py).
- Added phase-to-phase velocity and angular velocity continuity by seeding v_init and omega_init at boundaries. Affects [core/cgms/dmp_with_gain.py](core/cgms/dmp_with_gain.py), [core/cgms/orientation_dmp.py](core/cgms/orientation_dmp.py), and [core/multi_phase_policy.py](core/multi_phase_policy.py).
- Expanded Exp 2 diagnostics and plots: geodesic angle to q_ref, angular velocity, velocity CBF windows, acceleration plots, and terminal debug intervals for CBF/HOCBF activation. Affects [main_exp2.py](main_exp2.py).
- Documented full hard-constraint math flow (DMP + CGMS + CBF/HOCBF + projection) with end-to-end equations and pipeline summary. See [CBF_MATH_FLOW.md](CBF_MATH_FLOW.md).
- Unified hard-constraint semantics and geometry handling (HARD/SOFT normalization, deterministic geometry, shape_points cover) and aligned compiler behavior for HARD slack diagnostics. Affects [spec/json_parser.py](spec/json_parser.py), [llm_interface/validator.py](llm_interface/validator.py), [llm_interface/predicate_catalogue.py](llm_interface/predicate_catalogue.py), [spec/taskspec.py](spec/taskspec.py), and [spec/compiler.py](spec/compiler.py).
- Added runtime safety diagnostics to traces and wired hard constraint enforcement in rollout. Affects [core/certified_policy.py](core/certified_policy.py), [core/cbf_filter.py](core/cbf_filter.py), [core/cgms/dmp_with_gain.py](core/cgms/dmp_with_gain.py), [core/cgms/orientation_dmp.py](core/cgms/orientation_dmp.py), and [core/multi_phase_policy.py](core/multi_phase_policy.py).
