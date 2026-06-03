# ReachApprox

Sampling-based reachable-set and uncertainty-set approximation experiments for
autonomous and closed-loop dynamical systems.  The active experiment suite now
contains the quadratic-dynamics examples and the MuJoCo robot-arm benchmark.

The repository is organized around one question: given an initial uncertainty
set \(S_0\), propagate many sampled initial states through a fixed flow map and
measure how well a finite sample approximation captures the terminal set
\[
    S_T = \{\phi(T, x_0): x_0 \in S_0\}.
\]

The experiments compare uniform sampling, adversarial sampling, convex-hull
estimators, Christoffel-style support estimators, and Hausdorff-type numerical
metrics. In the robot-arm experiments, the main metric is currently the
directed Hausdorff distance to the sampled convex hull
\[
    \max_{z \in X_T^{\rm ref}} \operatorname{dist}(z,\operatorname{conv}(X_T^N)),
\]
which measures the terminal reachable-set approximation induced by the sampled
convex-hull estimator.

## Repository Layout

- `README.md`
  - The only Markdown documentation file in the repository.

- `illustration/`
  - Two-dimensional quadratic-dynamics illustrations for
    \(\dot x=x^2,\dot y=0\).
  - Generates sample-flow schematics, Hausdorff-versus-time plots, and
    Christoffel support visualizations.

- `exp/quaddynadv/`
  - Uniform and adversarial sampling experiments for the quadratic dynamics
    \(\dot x=x^2,\dot y=0\), plus a linear comparison flow.
  - Includes convex-hull and Christoffel-estimator sweeps over time, sample
    budgets, initial-set geometry, and adversarial update counts \(n_{\rm adv}\).
  - Reusable flow, geometry, sampling, projection, support-estimator, and
    Hausdorff helper functions live in `exp/quaddynadv/fun/`.
  - Generated figures for this experiment live in `exp/quaddynadv/results/`.

- `exp/robotarm/`
  - MuJoCo vertical planar serial \(n\)-link robot-arm uncertainty propagation
    benchmark for \(n \in \{2,3,4\}\).
  - The state is \(x=[q^\top,v^\top]^\top \in \mathbb{R}^{2n}\).
  - MuJoCo simulates rigid-body dynamics
    \(M(q)\dot v + C(q,v)v + g(q)=\tau\).
  - The current closed-loop controller is a weak adaptive inverse-dynamics
    tracking controller for a slowly varying reference trajectory,
    \[
        q_{d,i}(t)=q_{c,i}+A_i\sin(\omega t+\phi_i).
    \]
    The low-frequency reference is chosen to keep the system realistic without
    immediately collapsing the propagated uncertainty cloud.
  - MuJoCo model construction, controller code, rollout helpers, and shared
    dimension-scaling utilities live in `exp/robotarm/fun/`.
  - The retained result files are the uniform/adversarial combined `n=2,3,4`
    dimension-scaling CSVs and PNGs in `exp/robotarm/results/`.

There is no active `kuramoto` directory and no top-level `utils/` package. The
old shared quadratic-dynamics utilities were moved into `exp/quaddynadv/fun/`.

## Main Dependencies

Core experiments use:

- `numpy`
- `scipy`
- `matplotlib`
- `shapely`

Robot-arm experiments additionally use:

- `mujoco`
- `imageio`

MuJoCo rendering and the passive viewer are optional. Dynamics and endpoint
sampling do not require a GPU. If rendering is unavailable because of OpenGL or
framebuffer configuration, the robot-arm scripts are designed to continue
without blocking the numerical experiment.

## Common Commands

Run commands from `/home/jixia/exp` unless a script explicitly says otherwise.

Quadratic-dynamics illustration:

```bash
.venv/bin/python -u reachapprox/illustration/hausdorff_experiment.py
```

Uniform/adversarial quadratic-dynamics experiment:

```bash
.venv/bin/python -u reachapprox/exp/quaddynadv/quaddyn_adv_experiment.py
```

Christoffel and convex-hull \(n_{\rm adv}\) sweeps:

```bash
.venv/bin/python -u reachapprox/exp/quaddynadv/quaddyn_christoffel_mixture_experiment.py
```

Robot-arm uniform dimension scaling:

```bash
.venv/bin/python -u reachapprox/exp/robotarm/robotarm_dim_scaling.py --n_values 2,3,4
```

Robot-arm adversarial dimension scaling with one adversarial update:

```bash
.venv/bin/python -u reachapprox/exp/robotarm/robotarm_adversarial_dim_scaling.py --n_adv 1 --eta 0.20
```

Robot-arm Hausdorff-versus-time sweep:

```bash
.venv/bin/python -u reachapprox/exp/robotarm/robotarm_time_sweep.py --n_values 2,3,4
```

Generate terminal robot-arm endpoint samples and diagnostics from
`reachapprox/exp/robotarm`:

```bash
python generate_endpoint_samples.py --n 2 --N 1000 --T 1.0
python generate_endpoint_samples.py --n 3 --N 1000 --T 1.0
python generate_endpoint_samples.py --n 4 --N 1000 --T 1.0
```

Launch the optional MuJoCo viewer from `reachapprox/exp/robotarm`:

```bash
python animate_n_link_arm.py --n 3 --T 1.0
```

## Robot-Arm Formulation

For the robot-arm benchmark, the initial uncertainty set is a box around the
nominal zero state:

\[
    q_0 \in [-\rho_q,\rho_q]^n,\qquad
    v_0 \in [-\rho_v,\rho_v]^n,
\]

with default \(\rho_q=\rho_v=0.1\). Each trajectory uses the same fixed
reference trajectory and controller parameters, and the adaptive controller
state is reset for every sampled rollout.

The controller uses MuJoCo inverse dynamics for the feedforward term:

\[
\begin{aligned}
    e &= q-q_d(t),\\
    \dot e &= v-\dot q_d(t),\\
    s &= \dot e + \Lambda e,\\
    \tau_{\rm ff} &= M(q)\ddot q_d(t)+C(q,v)v+g(q),\\
    \dot{\hat d} &= \Gamma s-\sigma \hat d,\\
    \tau &= \tau_{\rm ff}-K_p e-K_d \dot e-\hat d.
\end{aligned}
\]

Torques are clipped for numerical stability. The goal is not controller design
or online safety enforcement; the controller is fixed before the experiment,
and the task is finite-horizon uncertainty propagation.

The dimension-scaling experiments evaluate the convex-hull reachable-set
estimator:

\[
    \widehat S_T^N = \operatorname{conv}(X_T^N).
\]

The reported error is

\[
    \max_{z \in X_T^{\rm ref}}
    \operatorname{dist}(z,\widehat S_T^N).
\]

The implementation uses a farthest-point coreset for large sampled clouds and a
batched Frank-Wolfe projection routine to approximate distances to the convex
hull. The CSV field `metric_implementation` is
`directed_hausdorff_to_convex_hull`.

## Outputs

Most scripts write figures, CSV files, logs, or `.npz` endpoint datasets under
one of:

- `reachapprox/illustration/results/`
- `reachapprox/exp/quaddynadv/results/`
- `reachapprox/exp/robotarm/results/`
- `CoRL_2026/fig/`

The current robot-arm results directory is intentionally kept small. It should
contain only:

- `robotarm_dim_scaling_n234_combined.csv`
- `robotarm_dim_scaling_n234_combined.png`
- `robotarm_adversarial_dim_scaling_n234_combined.csv`
- `robotarm_adversarial_dim_scaling_n234_combined.png`

Large sweeps can be expensive. For smoke tests, reduce `--N_ref`, `--n_seeds`,
`--coverage_subset`, or the sample-budget list before running the full
experiment.

## Notes

- The scripts generally fix random seeds for reproducibility while still
  allowing seed and budget overrides through command-line arguments.
- Some generated figures and result files are intentionally kept in the
  repository because they are used directly in paper figures and tables.
- The active experiment suite is currently the quadratic-dynamics family and
  the MuJoCo robot-arm benchmark.
