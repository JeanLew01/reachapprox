# ReachApprox

This is the repo for the submission named **"On the Limits of Sampling-Based Reachability: Geometry, Dynamics, and Sample Complexity"** for
CoRL 2026.

## Project Structure

```bash
├── __init__.py
├── README.md
├── illustration/                                # illustrations for reachable-set approximation under an autonomous non-Lipschitz system
│   ├── __init__.py
│   ├── hausdorff_experiment.py                 # Hausdorff-distance experiments and illustration figures under convex-hull / support estimators
│   ├── quadratic_dynamics.py                   # non-Lipschitz dynamics: \dot y = 0, \dot x = x^2
│   └── results/                                # generated illustration figures
│       ├── hausdorff_vs_samples.png
│       ├── sample_flow_schematic.png
│       └── star_samples_flow.png
└── exp/                                        # experiment implementations and evaluation tools
    ├── __init__.py
    ├── quaddynadv/                             # quadratic-dynamics uniform/adversarial sampling experiments
    │   ├── __init__.py
    │   ├── quaddyn_adv_experiment.py           # convex-hull reachable-set experiments for uniform vs adversarial sampling
    │   ├── quaddyn_christoffel_mixture_experiment.py
    │   │                                       # Christoffel and convex-hull sweeps over adversarial update counts n_adv
    │   ├── fun/
    │   │   ├── __init__.py
    │   │   ├── quadratic_flow.py               # analytic flow and flow Jacobian for \dot y = 0, \dot x = x^2
    │   │   ├── quaddyn_geometry.py             # equal-area disk / triangle / opened-triangle initial sets
    │   │   ├── quaddyn_sampling.py             # uniform sampling, adversarial updates, and projection routines
    │   │   └── support_estimators.py           # convex hull, Christoffel support, and Hausdorff utilities
    │   └── results/                           
    └── robotarm/                               # MuJoCo n-link robot-arm uncertainty-propagation
        ├── animate_n_link_arm.py               # optional MuJoCo passive-viewer animation script
        ├── compute_endpoint_coverage.py        # command-line wrapper for convex-hull endpoint coverage computation
        ├── generate_endpoint_samples.py        # terminal endpoint sample generation and diagnostic plots
        ├── plot_fit.py                         # slope-fitting utility for robot-arm dimension-scaling results
        ├── robotarm_dim_scaling.py             # uniform-sampling dimension-scaling experiment
        ├── robotarm_adversarial_dim_scaling.py # adversarial-sampling dimension-scaling experiment
        ├── robotarm_time_sweep.py              # Hausdorff-distance time-sweep experiment
        ├── simulate_d_link_arm.py              # simulator for n-link 
        ├── test_n_link_arm.py                  # test
        ├── fun/                               
        │   ├── __init__.py
        │   ├── mujoco_n_link_arm.py            # MuJoCo XML generation, dynamics, tracking controller, rollout, rendering
        │   ├── dim_scaling.py                  # sampling, rollout, convex-hull Hausdorff metric, shared constants
        │   └── coverage.py                     # directed distance to conv(X_T^N) via coreset + Frank-Wolfe projection
        └── results/                            
            ├── robotarm_dim_scaling_n234_combined.csv
            ├── robotarm_dim_scaling_n234_combined.png
            ├── robotarm_adversarial_dim_scaling_n234_combined.csv
            └── robotarm_adversarial_dim_scaling_n234_combined.png
```

## Main Experiments

### Quadratic Dynamics

The quadratic-dynamics experiments use the autonomous non-Lipschitz system

$$
    \dot x = x^2,\qquad \dot y = 0.
$$

The scripts compare uniform sampling, adversarial sampling, convex-hull
estimators, and Christoffel-type support estimators over several initial-set
geometries.

Typical commands from `../`:

for example:
```bash
.venv/bin/python -u reachapprox/illustration/hausdorff_experiment.py
.venv/bin/python -u reachapprox/exp/quaddynadv/quaddyn_adv_experiment.py
.venv/bin/python -u reachapprox/exp/quaddynadv/quaddyn_christoffel_mixture_experiment.py
```

### Robot Arm

The robot-arm benchmark uses MuJoCo vertical planar serial $n$-link arms with
$n \in \{2,3,4\}$. The state is

$$
    x = [q^\top, v^\top]^\top \in \mathbb{R}^{2n}.
$$

MuJoCo simulates rigid-body dynamics

$$
    M(q)\dot v + C(q,v)v + g(q) = \tau.
$$

The controller is a fixed weak adaptive inverse-dynamics tracking controller for
a slowly varying reference trajectory. This benchmark is for closed-loop
uncertainty propagation, not controller design or online safety enforcement.

The initial uncertainty set is the box

$$
    q_0 \in [-\rho_q,\rho_q]^n,\qquad
    v_0 \in [-\rho_v,\rho_v]^n,
$$

with default $\rho_q=\rho_v=0.1$.

The current robot-arm Hausdorff metric is the directed distance from a reference
terminal cloud to the sampled convex-hull estimator:

$$
    \max_{z \in X_T^{\rm ref}}
    \mathrm{dist}\left(z,\mathrm{conv}(X_T^N)\right).
$$

In CSV files, this appears as

```text
metric_implementation = directed_hausdorff_to_convex_hull
```

Typical commands from `../`:

for example
```bash
.venv/bin/python -u reachapprox/exp/robotarm/robotarm_dim_scaling.py --n_values 2,3,4
.venv/bin/python -u reachapprox/exp/robotarm/robotarm_adversarial_dim_scaling.py --n_adv 1 --eta 0.20
.venv/bin/python -u reachapprox/exp/robotarm/robotarm_time_sweep.py --n_values 2,3,4
```

Viewer and endpoint-generation scripts should be run from
`reachapprox/exp/robotarm`:

for example
```bash
python test_n_link_arm.py
python generate_endpoint_samples.py --n 2 --N 3000 --T 1.0
python animate_n_link_arm.py --n 3 --T 1.0
```

## Requirements

Core experiments use:

```text
numpy==1.26.4
scipy==1.17.1
matplotlib==3.10.8
shapely==2.1.2
```

Robot-arm experiments additionally use:

```text
mujoco==3.8.1
imageio==2.37.3
```

MuJoCo rendering and passive viewer scripts are optional. Dynamics and endpoint
sampling do not require a GPU.
