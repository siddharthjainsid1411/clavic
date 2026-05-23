# Changes Log

## 2026-05-20 16:03 IST

- Updated Exp 1 task weights and obstacle repulsion strength for the human scene (AtGoal weight, comfort weight, hard_strength) while keeping projector disabled for HOCBF-only tests. Affects [spec/exp1_task.json](spec/exp1_task.json).

## 2026-05-21 15:13 IST

- Enabled runtime CBF wiring in Exp 3b by invoking setup_hard_obstacles_from_taskspec (velocity/orientation/ang-vel CBFs now applied) while keeping obstacle avoidance set to NONE. Affects [main_exp3b.py](main_exp3b.py).

## 2026-05-21 16:44 IST

- Added repulsive-force activation diagnostics to runtime safety traces and Exp 3a terminal output; extended trace safety with repulsive-force arrays. Affects [core/cgms/dmp_with_gain.py](core/cgms/dmp_with_gain.py), [core/multi_phase_policy.py](core/multi_phase_policy.py), and [main_exp3a.py](main_exp3a.py).

## 2026-05-22 23:57 IST

- Added explicit avoidance_geometry field for obstacle predicates (decoupled from modality), updated LLM prompt/validation, and wired parser to honor the field. Affects [spec/json_parser.py](spec/json_parser.py), [llm_interface/predicate_catalogue.py](llm_interface/predicate_catalogue.py), [llm_interface/validator.py](llm_interface/validator.py), and [llm_interface/prompt_builder.py](llm_interface/prompt_builder.py).
- Updated Exp 3 specs to use avoidance_geometry and switched Exp 3b to HARD obstacle avoidance with object-dependent geometry; updated Exp 3b script text and behavior to match. Affects [spec/exp3a_task.json](spec/exp3a_task.json), [spec/exp3b_task.json](spec/exp3b_task.json), [spec/exp3c_task.json](spec/exp3c_task.json), and [main_exp3b.py](main_exp3b.py).

## 2026-05-19 00:37 IST

- Added runtime obstacle HOCBF filter (relative-degree-2) with epsilon-inflated safe radius and per-step diagnostics, applied after DMP repulsion and before velocity CBF. Affects [core/cbf_filter.py](core/cbf_filter.py), [core/cgms/dmp_with_gain.py](core/cgms/dmp_with_gain.py), and [core/multi_phase_policy.py](core/multi_phase_policy.py).
- Added projection-activation logging and obstacle HOCBF plots/diagnostics for Exp 2, including projection mask recording. Affects [core/obstacle_projection.py](core/obstacle_projection.py), [core/multi_phase_policy.py](core/multi_phase_policy.py), and [main_exp2.py](main_exp2.py).
- Extended obstacle HOCBF diagnostics and plots to Exp 1 and Exp 1b. Affects [main_exp1.py](main_exp1.py) and [main_exp1b.py](main_exp1b.py).
- Added per-obstacle projector disable flag and enabled HOCBF-only testing in Exp 1/1b specs; added CBF activation window prints in Exp 1/1b diagnostics. Affects [spec/json_parser.py](spec/json_parser.py), [core/multi_phase_policy.py](core/multi_phase_policy.py), [spec/exp1_task.json](spec/exp1_task.json), [spec/exp1b_task.json](spec/exp1b_task.json), [main_exp1.py](main_exp1.py), and [main_exp1b.py](main_exp1b.py).

## 2026-05-18 00:28 IST

- Added time-windowed VelocityLimit CBF activation and diagnostics (window_active, vmax_active), including time-aware filtering in the velocity CBF path. Affects [core/cbf_filter.py](core/cbf_filter.py), [core/cgms/dmp_with_gain.py](core/cgms/dmp_with_gain.py), and [core/multi_phase_policy.py](core/multi_phase_policy.py).
- Added phase-to-phase velocity and angular velocity continuity by seeding v_init and omega_init at boundaries. Affects [core/cgms/dmp_with_gain.py](core/cgms/dmp_with_gain.py), [core/cgms/orientation_dmp.py](core/cgms/orientation_dmp.py), and [core/multi_phase_policy.py](core/multi_phase_policy.py).
- Expanded Exp 2 diagnostics and plots: geodesic angle to q_ref, angular velocity, velocity CBF windows, acceleration plots, and terminal debug intervals for CBF/HOCBF activation. Affects [main_exp2.py](main_exp2.py).
- Documented full hard-constraint math flow (DMP + CGMS + CBF/HOCBF + projection) with end-to-end equations and pipeline summary. See [CBF_MATH_FLOW.md](CBF_MATH_FLOW.md).
- Unified hard-constraint semantics and geometry handling (HARD/SOFT normalization, deterministic geometry, shape_points cover) and aligned compiler behavior for HARD slack diagnostics. Affects [spec/json_parser.py](spec/json_parser.py), [llm_interface/validator.py](llm_interface/validator.py), [llm_interface/predicate_catalogue.py](llm_interface/predicate_catalogue.py), [spec/taskspec.py](spec/taskspec.py), and [spec/compiler.py](spec/compiler.py).
- Added runtime safety diagnostics to traces and wired hard constraint enforcement in rollout. Affects [core/certified_policy.py](core/certified_policy.py), [core/cbf_filter.py](core/cbf_filter.py), [core/cgms/dmp_with_gain.py](core/cgms/dmp_with_gain.py), [core/cgms/orientation_dmp.py](core/cgms/orientation_dmp.py), and [core/multi_phase_policy.py](core/multi_phase_policy.py).
