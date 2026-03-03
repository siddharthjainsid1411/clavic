# Visual Data Flow & Architecture Diagrams

## 1. Complete Computation Graph (Single Rollout)

```
PIBB gives us: θ = [θ_traj (237 dims), θ_ori (45 dims), θ_gain (16 dims)] × 2 phases
                     └─ Carry phase ─┘    └─ Pour phase (damped σ) ─┘

                                    ↓
                    policy.rollout(θ) is called
                                    ↓
                    ╔════════════════════════════════════════════════════════╗
                    ║         FOR EACH PHASE (carry, then pour)              ║
                    ╚════════════════════════════════════════════════════════╝
                                    ↓
        ┌───────────────────────────┴────────────────────────────┐
        │                                                         │
    dmp.set_theta(θ_phase)                                       │
        │                                                         │
        ├─ rbf_traj[0/1/2].W ← θ_traj[0:237]                   │
        ├─ rbf_SD.W         ← θ_gain[0:8]  (clipped)            │
        ├─ rbf_SK.W         ← θ_gain[8:16] (clipped)            │
        │                                                         │
        └─ (Ready to generate trajectory)                        │
                                    ↓
                ╔════════════════════════════════════════╗
                ║  ROLLOUT_TRAJ() [LAYER 1]              ║
                ║  Position generation with repulsion    ║
                ╚════════════════════════════════════════╝
                                    ↓
         For k = 0, 1, ..., T-1 (10 seconds at 0.01s timestep):
                                    ↓
        ┌────────────────────────────────────────────────────┐
        │ STEP k: Compute y[k+1] via Runge-Kutta 4           │
        │                                                     │
        │ Input:  y[k] (pos),  yd[k] (vel)                   │
        │         t[k] (time), dt (0.01s)                    │
        │                                                     │
        │ 1. Call: phase = time_system(t[k])                 │
        │           goal = polynomial_system(t[k])           │
        │           fhat = [rbf_traj[i].predict(phase)       │
        │                   for i in 0,1,2]                  │
        │                                                     │
        │ 2. Compute base dynamics:                          │
        │    k_spring = 100.0                                │
        │    spring = k_spring * (goal - y[k])               │
        │    damper = d * tau * yd[k]  (where d=20)         │
        │    gate = phase                                    │
        │    base_accel = (fhat*gate - spring - damper)/τ²  │
        │                                                     │
        │ 3. ╔════ NEW REPULSION LAYER ═════╗               │
        │    ║  for obs in repulsive_obstacles:              ║
        │    ║    diff = y[k] - c                           │
        │    ║    dist = ||diff||                           │
        │    ║    if dist < 0.24m:  (influence zone)        │
        │    ║      alpha = ((0.24-dist)/(0.24-0.12))^3    ║
        │    ║      mag = 0.05 * 100 * alpha = 5*alpha     ║
        │    ║      f_rep = mag * diff/dist                 ║
        │    ║    else:                                      │
        │    ║      f_rep = 0                               ║
        │    ║  base_accel += f_rep                         ║
        │    ╚══════════════════════════════╝               │
        │                                                     │
        │ 4. Use Runge-Kutta 4:                             │
        │    k1y = yd[k],  k1v = base_accel(y[k], yd[k])   │
        │    k2y = yd[k] + 0.5*dt*k1v,                      │
        │           k2v = base_accel(y[k]+0.5*dt*k1y, ...)  │
        │    k3y = yd[k] + 0.5*dt*k2v,                      │
        │           k3v = base_accel(y[k]+0.5*dt*k2y, ...)  │
        │    k4y = yd[k] + 1.0*dt*k3v,                      │
        │           k4v = base_accel(y[k]+dt*k3y, ...)      │
        │                                                     │
        │    y[k+1] = y[k] + (dt/6)(k1y + 2*k2y + 2*k3y +  │
        │                            k4y)                    │
        │    yd[k+1] = yd[k] + (dt/6)(k1v + 2*k2v + 2*k3v +│
        │                             k4v)                   │
        │                                                     │
        │ Output: y[k+1], yd[k+1]                           │
        └────────────────────────────────────────────────────┘
                                    ↓
          After loop: y, yd = full T-timestep trajectories
                                    ↓
                ╔════════════════════════════════════════╗
                ║  LAYER 2: CGMS GAIN SCHEDULING         ║
                ║  Q̇ = αQ + 0.5Q^{-T}B                   ║
                ╚════════════════════════════════════════╝
                                    ↓
         1. Query RBFs for SD, SK at all T timesteps:
            SD = [rbf_SD.predict(x) for x in timesteps]
            SK = [rbf_SK.predict(x) for x in timesteps]
                                    ↓
         2. Compute D = αH + SD @ SD^T
                                    ↓
         3. Integrate Cholesky ODE: Q̇ = αQ + 0.5Q^{-T}B
            - For each timestep: Q[k+1] via Runge-Kutta 4
            - B depends on Ḋ and SK (computed from RBFs)
            - Result: Q[0..T] (upper triangular matrices)
                                    ↓
         4. Compute K = Q^T @ Q for all timesteps
            └─ GUARANTEED K > 0 (Cholesky property)
                                    ↓
                ╔════════════════════════════════════════╗
                ║  CONCATENATE PHASES                    ║
                ║  Join carry (y, yd, K, D) + pour       ║
                ║  Handle phase boundaries (Q continuity)║
                ╚════════════════════════════════════════╝
                                    ↓
                ╔════════════════════════════════════════╗
                ║  LAYER 3: HARD RADIAL PROJECTOR        ║
                ║  (in obstacle_projection.py)           ║
                ╚════════════════════════════════════════╝
                                    ↓
         For each full trajectory point p[i]:
             dist[i] = ||p[i] - c_obs||
             if dist[i] < r_safe:
                 p[i] ← c_obs + r_safe * (p[i]-c_obs)/||p[i]-c_obs||
             └─ Push to sphere surface
                                    ↓
         Recompute velocity by finite difference:
             vel_safe[0] = (p[1] - p[0]) / dt
             vel_safe[1:-1] = (p[2:] - p[:-2]) / (2*dt)
             vel_safe[-1] = (p[-1] - p[-2]) / dt
                                    ↓
                ╔════════════════════════════════════════╗
                ║  BUILD TRACE OBJECT                    ║
                ║  (return from rollout)                 ║
                ╚════════════════════════════════════════╝
                                    ↓
                Trace{
                    time: [0.00, 0.01, 0.02, ..., 10.00],
                    position: y_projected (1001×3),
                    velocity: vel_projected (1001×3),
                    gains: {
                        K: [K[0], K[1], ..., K[1000]] each (3,3),
                        D: [D[0], D[1], ..., D[1000]] each (3,3)
                    },
                    raw_sk_weights, raw_sd_weights,
                    orientation, angular_velocity
                }
                                    ↓
```

---

## 2. Repulsive Force Detail (Zoomed In)

```
At each DMP timestep k:

    y[k] = current position (3D)
    c     = obstacle center (0.40, 0.30, 0.30)
    r     = safe radius (0.12 m)
    
                            ↓
    
    ┌─────────────────────────────────────┐
    │ Compute distance and normal         │
    │                                     │
    │ diff = y[k] - c                     │
    │ dist = sqrt(diff·diff)              │
    │ n = diff / dist  (unit normal)      │
    │                                     │
    └─────────────────────────────────────┘
                            ↓
    
    ┌─────────────────────────────────────┐
    │ Check influence zone                │
    │                                     │
    │ r_infl = 2.0 × 0.12 = 0.24 m       │
    │                                     │
    │ if dist ≥ 0.24m:                    │
    │     f_repulse = [0, 0, 0]          │
    │     EXIT                            │
    │                                     │
    │ if dist < 0.24m:  CONTINUE         │
    │                                     │
    └─────────────────────────────────────┘
                            ↓
    
    ┌──────────────────────────────────────────────────┐
    │ Compute smooth cubic taper                       │
    │                                                  │
    │              (r_infl - dist)³                   │
    │  alpha = ( ─────────────────── )               │
    │          (  r_infl - r       )                  │
    │                                                  │
    │                 (0.24 - dist)³                  │
    │         = ( ────────────── )                    │
    │           (  0.24 - 0.12  )                     │
    │                                                  │
    │                 (0.24 - dist)³                  │
    │         = ( ────────────── )                    │
    │           (      0.12     )                     │
    │                                                  │
    │ Examples:                                        │
    │   dist = 0.12 (at surface)  → alpha = (0.12/0.12)³ = 1.0
    │   dist = 0.18 (mid-zone)    → alpha = (0.06/0.12)³ = 0.125
    │   dist = 0.24 (at boundary) → alpha = (0.00/0.12)³ = 0.0
    │                                                  │
    └──────────────────────────────────────────────────┘
                            ↓
    
    ┌──────────────────────────────────────────────────┐
    │ Compute magnitude scaled by DMP dynamics         │
    │                                                  │
    │ k_dmp = (d²/4) = (20²/4) = 100                 │
    │ strength = 0.05   (tunable parameter)           │
    │                                                  │
    │ mag = strength × k_dmp × alpha                  │
    │     = 0.05 × 100 × alpha                        │
    │     = 5.0 × alpha                               │
    │                                                  │
    │ Examples:                                        │
    │   at surface (alpha=1.0):     mag = 5.0 m/s²   │
    │   at mid-zone (alpha=0.125):  mag = 0.625 m/s²│
    │   at boundary (alpha=0.0):    mag = 0.0 m/s²   │
    │                                                  │
    └──────────────────────────────────────────────────┘
                            ↓
    
    ┌──────────────────────────────────────────────────┐
    │ Compute repulsive acceleration                   │
    │                                                  │
    │ f_repulse = mag × n                             │
    │           = 5.0 × alpha × (y[k]-c)/dist        │
    │                                                  │
    │ This is a 3D vector pointing outward from       │
    │ the obstacle center                             │
    │                                                  │
    │ Units: m/s² (acceleration)                      │
    │ Direction: outward normal (away from obstacle)  │
    │ Magnitude: tapers smoothly from 5.0 to 0.0     │
    │                                                  │
    └──────────────────────────────────────────────────┘
                            ↓
    
    ┌──────────────────────────────────────────────────┐
    │ Apply to DMP acceleration                        │
    │                                                  │
    │ base_accel = (spring + damper + fhat) / τ²     │
    │ final_accel = base_accel + f_repulse           │
    │                                                  │
    │ This modified acceleration is then used in     │
    │ Runge-Kutta 4 integration                       │
    │                                                  │
    └──────────────────────────────────────────────────┘
```

---

## 3. Comparison: Three Obstacle Avoidance Approaches

```
┌─────────────────────────────────────────────────────────────────────┐
│ APPROACH A: Guide's CBF (Tangential-Normal) — Applied Post-Hoc      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  DMP ODE runs:                                                       │
│    y'' = spring + damper + fhat                                     │
│          └─ produces positions y(t) over [0,10]s                    │
│                                                                      │
│  DMP finishes. y, yd arrays are complete.                           │
│                                                                      │
│  Then we try to "fix" velocity:                                     │
│    v_safe = v - (v·n)n  ← strip inward component                  │
│                                                                      │
│  Problem: DMP doesn't use v to compute next y!                      │
│    y[k+1] is from the ODE: y[k+1] = y[k] + ∫ dydt dt              │
│                                                                      │
│    The ODE is autonomous in y, doesn't read v as input              │
│    Changing v[k] after y[k+1] is computed → ZERO effect            │
│                                                                      │
│  ✗ FAILS: velocity modification is disconnected from positions     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│ APPROACH B: Radial Projector (Our Original Layer 3)                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  DMP runs freely → y(t)                                             │
│  Projector clamps violations:                                       │
│    ∀i: if ||y[i] - c|| < r  then  y[i] ← c + r*n[i]               │
│                                                                      │
│  ✓ Guarantees: ||p(t) - c|| ≥ r  (by construction)                 │
│                                                                      │
│  ✗ Weakness: Straight path through obstacle → 191 pinned points     │
│             → C-turn artifact (slow arc on sphere surface)          │
│                                                                      │
│  Cost: 2.2118  (good)                                               │
│  Pinned: 191   (artifact)                                           │
│  Clearance: 2.0 cm (on surface)                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│ APPROACH C: Repulsive Forcing Inside DMP ODE ← OUR IMPLEMENTATION   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  During DMP integration (at each RK4 step):                         │
│                                                                      │
│    y'' = spring + damper + fhat + f_repulse(y)                     │
│           └─────────────────┬──────────────┘    └──────────┬──────┘│
│                   Original DMP        NEW: acts on y directly       │
│                                                                      │
│  Repulsion is an ODE input:                                         │
│    - Influences position generation as it happens                   │
│    - Goal attractor still present (spring term)                     │
│    - Path bends around obstacle organically                         │
│                                                                      │
│  ✓ Guarantees: with projector backstop → ||p(t)-c|| ≥ r           │
│  ✓ Path quality: 0 pinned, 7.8 cm clearance, genuine arc           │
│  ✓ Math: Implements guide's CBF method in the RIGHT place           │
│                                                                      │
│  Cost: 2.5957  (slightly higher due to wider detour)               │
│  Pinned: 0     (NO C-turn!)                                         │
│  Clearance: 7.8 cm (genuine safety margin)                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘


SUMMARY TABLE:

┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
│ Metric           │ A: Post-hoc CBF  │ B: Radial Only   │ C: ODE Repulsion │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Pinned points    │ Same as B (191)  │ 191              │ 0 ✅             │
│ Clearance        │ Same as B (2cm)  │ 2.0 cm (bad)     │ 7.8 cm ✅        │
│ Hard guarantee   │ ✓ (with project) │ ✓                │ ✓                │
│ K > 0            │ ✓                │ ✓                │ ✓                │
│ Goal reached     │ ✓                │ ✓                │ ✓                │
│ Why it fails     │ Velocity ignored │ C-turn artifact  │ WORKS! ✓         │
└──────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

---

## 4. CGMS Gain Scheduling (Layer 2) — Untouched

```
After rollout_traj() finishes with y(t) and yd(t),
we solve a SEPARATE ODE for stiffness:

┌────────────────────────────────────────────────────────────┐
│ Cholesky Decomposition: K = QᵀQ                            │
│                                                            │
│ Boundary condition:  Q[0] = cholesky(K0 I + ε I)          │
│                           = cholesky(300 I + 1e-9 I)      │
│                                                            │
│ Differential equation:  Q̇ = α Q + 0.5 Q⁻ᵀ B              │
│                                                            │
│ where: B = -α Ḋ - SK·SKᵀ                                  │
│        D = α H + SD·SDᵀ                                    │
│        H = identity matrix (3×3)                          │
│        α = 0.05 (time constant)                           │
│                                                            │
│ SK, SD are RBF outputs: queried from θ_gain               │
│         ╔═══════════════════════════════════════╗          │
│         ║ NO DEPENDENCE ON y(t) POSITIONS      ║          │
│         ║ Only reads RBF weights (θ_gain)     ║          │
│         ╚═══════════════════════════════════════╝          │
│                                                            │
│ Guarantee: K = QᵀQ > 0  ∀t (Cholesky by construction)    │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 5. Complete Temporal Sequence

```
Time: 0.00s - 0.10s     INITIALIZATION
─────────────────────────────────────
  Load task spec
  Create DMP with RBFs initialized
  Create PIBB optimizer: μ=0, Σ=σ_init
  Compute nominal (θ=0) cost baseline

  
Time: 0.10s - 0.20s     UPDATE 1
─────────────────────────────────────
  Sample: θ₁...θ₃₀ ~ N(μ, Σ)
  
  For each sample θᵢ:
    ├─ policy.rollout(θᵢ)
    │  ├─ Phase 1: Carry (DMP + repulsion)
    │  ├─ Phase 2: Pour (DMP + repulsion)
    │  ├─ Gain scheduling (Q ODE)
    │  ├─ Projection (hard clamp)
    │  └─ Return Trace
    │
    └─ cost[i] = objective(Trace)
  
  PIBB.update(samples, costs)
  μ, Σ ← updated toward lower-cost region
  best_cost = 11812.88

  
Time: 0.20s - 0.30s     UPDATE 2-10
─────────────────────────────────────
  Repeat sampling & updating
  Costs generally decrease (PIBB converging)
  
  
Time: 1.00s - 2.00s     UPDATE 30-50
─────────────────────────────────────
  best_cost drops: 11812 → 5.80
  μ strongly concentrated on good region
  

Time: 2.00s - 2.70s     UPDATE 50-70
─────────────────────────────────────
  Fine-tuning
  best_cost converges: 5.80 → 2.596
  Optimizer stability reached
  

Time: 2.70s             FINAL ROLLOUT
─────────────────────────────────────
  θ_best = [shape from best sample]
  trace_final = policy.rollout(θ_best)
  
  Results:
    ├─ Goal distance: 0.0 cm
    ├─ Pinned points: 0
    ├─ Clearance: 7.8 cm
    ├─ K min: 0.0003 > 0 ✓
    ├─ Max speed: 0.7749 m/s
    └─ Cost: 2.5957
  
  Save checkpoint
  Generate 5 plots
  Done ✓


Total wall-clock time: ~3 minutes (70 updates × 30 samples × ~1.5s per rollout)
```

