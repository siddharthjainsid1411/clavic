# Certified Hard Constraints Math Flow (DMP + CGMS + CBF)

Target audience: beginner in robotics domain (UG math level)
Goal: show the exact math at each step and how each output feeds the next step.

---

## Table of contents

0. Problem setup (what we want to guarantee)
1. Dynamic Movement Primitive (DMP) math
2. Obstacle avoidance math (repulsion + projection)
3. Certified gains (CGMS) math
4. Hard runtime constraints (CBF and HOCBF)
5. Temporal logic robustness math
6. Objective and optimizer math (PIBB)
7. Full pipeline flow with inputs/outputs and equations
8. What changed in current hard constraints
9. Issue found and proposed math fix

---

## 0) Problem setup

We need a trajectory that is smooth, safe, and task-compliant:

- Geometric safety: keep distance from obstacles.
- Kinematic safety: keep speed and angular speed within limits.
- Orientation safety: stay inside a cone around a reference orientation.
- Stability: stiffness matrix must be positive definite.
- Task logic: reach goals, hold positions, etc.

We enforce this with three safety layers plus temporal-logic cost:

1) Repulsive forcing inside the DMP ODE (soft steering).
2) Hard projection outside obstacles (hard geometry).
3) Runtime CBF and HOCBF filters for velocity and orientation constraints.

---

## 1) DMP math (position)

State:

$$
x \in \mathbb{R}^3, \quad v = \dot x, \quad a = \ddot x
$$

Phase variable:

$$
\dot s = -\frac{1}{\tau}, \quad s \in (0,1], \quad \gamma(s) = s
$$

DMP equation used in rollout:

$$
	au^2 \ddot x = f(s)\,\gamma(s) - k(x - g(t)) - d\,\tau\,\dot x + a_{\text{rep}}(x)
$$

with $k = d^2/4$.

RBF forcing (per axis):

$$
f(s) = \sum_{i=1}^{N} w_i\,\phi_i(s), \quad
\phi_i(s) = \exp\left(-\frac{(s-c_i)^2}{2\sigma_i^2}\right)
$$

Initialization from a minimum-jerk demo (in code):

$$
f_{\text{target}}(t) = \tau^2 \ddot x_{\text{mj}}(t) + \frac{k(x_{\text{mj}}-g)+d\tau\dot x_{\text{mj}}}{m},
\quad f_{\text{target}} \leftarrow \frac{f_{\text{target}}}{s}
$$

This gives the initial RBF weights $w_i$.

---

## 2) Obstacle avoidance math

### 2.1 Repulsive forcing (inside the DMP ODE)

For a spherical obstacle with center $c$, radius $r$, influence radius $r_{\text{infl}}$:

$$
d = \|x-c\|
$$

$$
\alpha(d)=\left(\frac{r_{\text{infl}}-d}{r_{\text{infl}}-r}\right)^3
$$

$$
a_{\text{rep}} = k_{\text{dmp}}\,\text{strength}\,\alpha(d)\,\frac{x-c}{d},
\quad \text{only if } d < r_{\text{infl}}
$$

For an infinite cylinder around $z$, use $d = \|x_{xy}-c_{xy}\|$ and apply the
repulsion in the $xy$ plane.

### 2.2 Hard projection (by construction)

After rollout, for any point inside the obstacle:

- Sphere:

$$
	ext{if } d < r, \quad x' = c + r\,\frac{x-c}{d}
$$

- Infinite cylinder:

$$
	ext{if } d_{xy} < r, \quad x'_{xy} = c_{xy} + r\,\frac{x_{xy}-c_{xy}}{d_{xy}}
$$

Velocity is recomputed by finite difference on $x'(t)$.

---

## 3) Certified gains (CGMS) math

The stiffness is parameterized as:

$$
K(t) = Q(t)^T Q(t) \succ 0
$$

Gain schedules are produced by RBFs:

$$
D(t) = \alpha H + S_D(t) S_D(t)^T
$$

Define:

$$
B(t) = -\alpha \dot D(t) - S_K(t) S_K(t)^T
$$

Cholesky ODE:

$$
\dot Q = \alpha Q + \tfrac{1}{2} Q^{-T} B(t)
$$

So $K(t)$ is always positive definite by construction.

---

## 4) Hard runtime constraints (CBF and HOCBF)

### 4.1 Velocity CBF

Dynamics:

$$
\dot x = v, \quad \dot v = a
$$

Barrier:

$$
h(v) = v_{\max}^2 - \|v\|^2
$$

CBF condition:

$$
\dot h + \alpha h \ge 0 \quad\Rightarrow\quad 2 v^T a \le \alpha h
$$

If violated, project the nominal acceleration:

$$
a_{\text{safe}} = a_{\text{nom}} - \frac{(2 v^T a_{\text{nom}} - \alpha h)}{\|2v\|^2}\,(2v)
$$

### 4.2 Orientation DMP

Orientation dynamics in log-space:

$$
	au^2 \dot\omega = -k\,e(q,q_g) - \tau d\,\omega + \gamma(s) f_{\text{ori}}(s)
$$

Quaternion kinematics:

$$
\dot q = \tfrac{1}{2}[0,\omega] \otimes q
$$

### 4.3 Orientation HOCBF

Cone barrier:

$$
h(q) = (q^T q_{\text{ref}})^2 - \cos^2\left(\tfrac{\theta_{\max}}{2}\right)
$$

HOCBF condition:

$$
\psi_1 = \dot h + \alpha_1 h, \quad \dot\psi_1 + \alpha_2 \psi_1 \ge 0
$$

This gives an affine constraint on $\beta = \dot\omega$, and we project
$\beta_{\text{nom}}$ onto the HOCBF half-space.

### 4.4 Angular velocity CBF

Barrier:

$$
h(\omega) = \omega_{\max}^2 - \|\omega\|^2
$$

Constraint:

$$
2\,\omega^T \beta \le \alpha h
$$

Again, $\beta_{\text{nom}}$ is projected onto this half-space, and a post-step
projection keeps $\|\omega\| \le \omega_{\max}$.

---

## 5) Temporal logic robustness math

Each predicate returns a robustness signal $\rho(t)$.

Examples:

$$
\rho_{\text{avoid}}(t) = \|x(t)-c\| - r, \quad
\rho_{\text{speed}}(t) = v_{\max} - \|v(t)\|
$$

Operators:

$$
\rho_{\Box} = \min_t \rho(t), \quad
\rho_{\Diamond} = \max_t \rho(t)
$$

Until:

$$
\rho_{\phi\,\mathcal{U}\,\psi} = \max_t \min\Big(\rho_\psi(t),\; \min_{\tau < t} \rho_\phi(\tau)\Big)
$$

Time windows apply min or max only inside $[t_s, t_e]$.

---

## 6) Objective and optimizer math

Total cost:

$$
J = \sum_{\text{HARD}} \lambda_h\,\max(0,-\rho)^2 + \sum_{\text{SOFT}} w\,\max(0,-\rho) + J_{\text{reg}}
$$

Regularizers in code:

- Stiffness raw-weight penalty:

$$
J_{\text{sk}} = \lambda_{\text{sk}}\,\text{mean}\,(\max(0, |w| - \text{SK\_CLIP})^2)
$$

- K ceiling penalty:

$$
J_{\text{K}} = \lambda_K\,\text{mean}\,\left(\frac{\max(0, \text{tr}(K)-K_{\max})}{K_{\max}}\right)^2
$$

- Damping minimum:

$$
J_{\text{D}} = \lambda_D\,\text{mean}\,\left(\frac{\max(0, D_{\min}-\text{tr}(D))}{D_{\min}}\right)^2
$$

PIBB update (sampling, weighting, mean update):

$$
	heta_i = \mu + \sigma \odot z_i, \quad z_i \sim \mathcal{N}(0, I)
$$

$$
w_i = \frac{\exp(-\beta (J_i - J_{\min})/(J_{\max}-J_{\min}))}{\sum_j \exp(-\beta (J_j - J_{\min})/(J_{\max}-J_{\min}))}
$$

$$
\mu \leftarrow \sum_i w_i \theta_i, \quad
\sigma^2 \leftarrow \text{decay}\,\sigma^2 + (1-\text{decay})\sum_i w_i (\theta_i-\mu)^2
$$

---

## 7) Full pipeline flow (inputs, math, outputs)

### Step 1: Task JSON -> TaskSpec

Input:
- JSON with phases, clauses, bindings.

Math:

$$
	ext{parameters}[p] = \text{bindings}[\text{Predicate}.p]
$$

Modality normalization:

$$
	ext{REQUIRE/PREFER} \to \text{SOFT}, \quad \text{HARD remains HARD}
$$

Output:
- TaskSpec with phases, clauses, and hard_obstacle_specs.

### Step 2: Build policy (DMPs)

Input:
- TaskSpec phases.

Math:

$$
\{\text{DMP}_i\}_{i=1}^N, \quad \{\text{OriDMP}_i\}_{i=1}^N
$$

Output:
- Policy object with all phase dynamics.

### Step 3: Wire hard safety

Input:
- TaskSpec hard obstacles and clauses.

Math:

$$
	ext{HARD obstacle} \Rightarrow a_{\text{rep}}(x) + \Pi_{\text{obs}}(x)
$$

$$
	ext{VelocityLimit} \Rightarrow \text{CBF on } (x,v)
$$

$$
	ext{OrientationLimit} \Rightarrow \text{HOCBF on } (q,\omega)
$$

$$
	ext{AngularVelocityLimit} \Rightarrow \text{CBF on } \omega
$$

Output:
- Safety-configured policy.

### Step 4: Rollout

Input:
- Policy parameters $\theta$.

Math:
- DMP equation (Section 1).
- CGMS equation (Section 3).
- CBF/HOCBF equations (Section 4).
- Projection equation (Section 2).

Output:
- Trace with $x(t), v(t), K(t), D(t)$ and optional $q(t), \omega(t)$ plus safety diagnostics.

### Step 5: Cost

Input:
- Trace + TaskSpec clauses.

Math:

$$
J = \sum_{\text{HARD}} \lambda_h\,\max(0,-\rho)^2 + \sum_{\text{SOFT}} w\,\max(0,-\rho) + J_{\text{reg}}
$$

Output:
- Scalar cost for this rollout.

### Step 6: PIBB update

Input:
- Costs for sampled rollouts.

Math:
- PIBB update (Section 6).

Output:
- Updated parameter distribution for the next iteration.

---

## 8) What changed in current hard constraints

- HARD obstacles now map to:
   - DMP repulsion inside the ODE
   - Hard projection after rollout
   - Slack diagnostic cost in the compiler

- VelocityLimit (non-soft) now uses a velocity CBF on acceleration.
- OrientationLimit (non-soft) now uses a quaternion cone HOCBF.
- AngularVelocityLimit (non-soft) now uses an angular velocity CBF plus a
   post-step omega projection.

This unifies hard constraints under runtime filters rather than only cost.

---

## 9) Issue found and proposed math fix

Issue:
VelocityLimit supports time windows in the TaskSpec, but the current CBF
enforces a single global $v_{\max}$ for the whole rollout.

Proposed fix:

$$
v_{\max}(t) = \min\{v_k : t \in [t_{k,s}, t_{k,e}]\}
$$

$$
h(v,t) = v_{\max}(t)^2 - \|v\|^2
$$

Apply the CBF condition only when the window is active:

$$
\dot h + \alpha h \ge 0 \quad \text{for } t \in [t_{k,s}, t_{k,e}]
$$

This matches how orientation and angular velocity CBFs already use time windows.
