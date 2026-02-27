# Checkpoint & Warm-Start Implementation Guide

## File Organization

### Clean Implementation
- **`main.py`** — Core algorithm as described in your IROS draft
  - Standard PI² optimization loop
  - No checkpoint logic
  - No experimentation overhead
  - Clean, reproducible baseline

### Experimentation
- **`experiment_checkpoint_warmstart.py`** — Feature-rich testing script
  - Checkpoint warm-start with tau-matching
  - Optimal trajectory analysis
  - Multi-tau comparison infrastructure
  - Detailed logging and insights

---

## What the Saved Checkpoint Actually Contains

### The Key Question: "Is it optimal?"

**Answer: It depends on what you mean by "optimal".**

### What was saved:
```
best_theta (parameters optimized for tau = 10s)
optimal_tau_candidate = 4.61s (earliest time goal was satisfied)
best_cost = 0.7587 (cost achieved at tau=10s)
```

### What this means:
✅ **Best parameters found when TRAINING at tau = 10s**
- The optimizer ran 480 rollouts (40 updates × 12 samples) all at exactly tau=10s
- It found parameters `best_theta` that minimize the objective cost at that duration
- These parameters satisfy all constraints when executed for 10 seconds

✅ **The trajectory reaches the goal in 4.61 seconds**
- But it doesn't "end" there — the DMP continues to maintain the goal for the remaining 5.39s
- This is because `eventually` operator (`◇`) only requires satisfaction at ANY point, not throughout

❌ **NOT optimal across all possible tau values**
- Never tested at tau=2s, 3s, 5s, 7s, etc.
- Those tau values would have different optimal parameters
- Only optimal specifically for tau=10s

❌ **NOT the global optimum over all parameter space**
- Just the best found by PI² in 40 iterations
- More iterations or different random seeds could find better parameters

---

## To Find Optimal Across Different Tau Values

### Method 1: Sequential Experimentation
```python
for tau in [2, 5, 7, 10, 15]:
    # 1. Modify spec/example_task.json: "horizon_sec": tau
    # 2. Run: python experiment_checkpoint_warmstart.py
    # 3. Rename checkpoint: mv optimal_checkpoint.npz ckpt_tau{tau}.npz
    # 4. Record the best_cost value
    # 5. Compare costs across all tau values
```

The tau with the **lowest best_cost** indicates the most feasible duration given the constraints.

### Method 2: Comparative Analysis
```python
# Load checkpoints for different tau values
ckpt_2 = np.load("ckpt_tau2.npz")
ckpt_5 = np.load("ckpt_tau5.npz")
ckpt_10 = np.load("ckpt_tau10.npz")

print(f"tau=2s  → cost={float(ckpt_2['best_cost']):.4f}")
print(f"tau=5s  → cost={float(ckpt_5['best_cost']):.4f}")
print(f"tau=10s → cost={float(ckpt_10['best_cost']):.4f}")

# Lower cost = better feasibility under constraints
```

The pattern tells you:
- **Cost decreases with increasing tau** → constraints are easier to satisfy with more time
- **Sharp drop at certain tau** → that tau is a "sweet spot" where constraints can all be met
- **Floor cost** → soft constraints that are fundamentally hard to satisfy (e.g., human distance)

---

## Checkpoint Warm-Start in Practice

### First Run (from scratch):
```bash
python experiment_checkpoint_warmstart.py
```
Output:
```
✗ Checkpoint mismatch: ... — starting fresh
Starting Optimization...
Update 01 | Min: 2.1345 | Mean: 2.2134 | BestSoFar: 2.1345
Update 02 | Min: 1.8765 | Mean: 1.9123 | BestSoFar: 1.8765
...
Update 40 | Min: 0.7587 | Mean: 0.7710 | BestSoFar: 0.7587
✓ Saved checkpoint to optimal_checkpoint.npz
```

### Second Run (warm-started):
```bash
python experiment_checkpoint_warmstart.py
```
Output:
```
✓ Warm-start: loaded checkpoint (tau=10s, cost=0.7587)
Starting Optimization...
Update 01 | Min: 0.7542 | Mean: 0.7612 | BestSoFar: 0.7542  ← starts near previous best
Update 02 | Min: 0.7481 | Mean: 0.7521 | BestSoFar: 0.7481
...
Update 40 | Min: 0.7231 | Mean: 0.7312 | BestSoFar: 0.7231  ← further improvement
✓ Saved checkpoint to optimal_checkpoint.npz
```

**Key difference:** Second run starts with `sigma=1.0` (tight exploration) instead of `sigma=5.0` (broad), so it fine-tunes the already-good solution instead of exploring widely.

---

## Checkpoint File Format

```python
optimal_checkpoint.npz
├── best_theta          # (n_params,) — parameter vector to reload
├── optimal_tau         # scalar — earliest goal satisfaction time
├── horizon_sec         # scalar — tau this was optimized for (for matching)
├── best_cost           # scalar — cost value achieved
├── position            # (T, 3) — full trajectory positions
├── velocity            # (T, 3) — full trajectory velocities
├── time                # (T,) — time points
├── K_trace             # (T, 3, 3) — stiffness evolution
└── D_trace             # (T, 3, 3) — damping evolution
```

---

## Using the Checkpoint in Your Experiments

### Example: Warm-start from checkpoint for fine-tuning
```python
from experiment_checkpoint_warmstart import load_checkpoint, optimize_with_checkpoint_support

# Load previous checkpoint
best_theta, saved_tau, saved_cost, can_use = load_checkpoint(
    "optimal_checkpoint.npz",
    expected_tau=10.0,
    theta_dim=1234  # your theta dimension
)

if can_use:
    # Further optimize from this starting point
    result = optimize_with_checkpoint_support(
        taskspec,
        certified_policy,
        predicate_registry,
        n_updates=20,  # fewer updates since already converged
        use_checkpoint=True
    )
```

### Example: Analyze saved trajectory
```python
from experiment_checkpoint_warmstart import analyze_optimal_trajectory_across_tau

analyze_optimal_trajectory_across_tau(goal_pose, human_position)
# Prints detailed insights about what the checkpoint represents
```

---

## Summary

| Aspect | Meaning |
|---|---|
| **saved best_theta** | Optimal for tau=10s specifically |
| **optimal_tau_candidate** | Earliest time to reach goal with that theta |
| **best_cost** | Quality metric at tau=10s |
| **Is it "optimal"?** | Locally optimal at tau=10s; unknown at other tau |
| **To optimize across tau** | Run separate experiments for each tau; compare costs |
| **Checkpoint warm-start** | Enables fine-tuning without restart |

---

## Next Steps for Experiments

1. **Baseline runs** — Run `main.py` multiple times to establish variance
2. **Warm-start comparison** — Run `experiment_checkpoint_warmstart.py` to see convergence speedup
3. **Multi-tau analysis** — Optimize for tau=[2, 5, 10, 15]s; save separate checkpoints
4. **Report results** — Plot cost vs tau; identify sweet spots for your task
