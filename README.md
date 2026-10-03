# On the Limits of Sampling-Based Reachability: Geometry, Dynamics, and Sample Complexity

<div align="center">

**Jixian Liu · Ihab Tabbara · Hussein Sibai · Enrique Mallada**

*10th Conference on Robot Learning (CoRL 2026), Austin, TX*

[![Python Version](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[<img src="https://img.shields.io/badge/Simulator-MuJoCo%203.8-orange.svg"/>](https://mujoco.org/)
[<img src="https://img.shields.io/badge/CoRL-2026-red.svg"/>](https://www.corl.org/)

</div>

This repository contains the code for every experiment in the paper.  Each experiment has one script,
and the raw per-trial results behind it ship in `results/`.

Sampling-based reachability propagates finitely many initial states through the dynamics and builds a set
estimate $\widehat S_N$ from the endpoints.  Existing finite-sample guarantees bound the *probability mass*
that the estimate misses.  Such a bound still allows the estimate to miss a thin region of the reachable
set that is spatially far from the rest.  We study the
geometric question instead: **how many endpoint samples are needed so that $d_H(S_T,\widehat S_N)\le r$**,
where $S_T = \mathcal R_T(S_0)$ is the reachable set of $\dot x = F(x)$ from the initial set $S_0$?


## Theoretical results

Three regularity conditions make a probability-mass guarantee upgradable to Hausdorff accuracy:

* **Geometry:** the complement of $S_0$ has positive reach $r_0$, which rules out cusps and thin parts.
* **Dynamics:** $F$ is $L$-Lipschitz, so the flow distorts distances by at most $e^{LT}$.
* **Sampling:** the initial density is bounded below by $\rho/|S_0|$.

Let $R$ be the volume-equivalent radius of $S_0$ and $n$ the state dimension.  Under these conditions:

| | Sample complexity | Result |
|---|---|---|
| **Upper bound** (any estimator containing the samples) | $N \ge \dfrac{2^{2n}e^{nLT}R^n}{\rho\, r^n}\Big[\log\dfrac{2^{3n}e^{nLT}R^n}{r^n} + \log\dfrac1\delta\Big]$ | Theorem 1 |
| **Minimax lower bound** (every estimator fails on some instance) | $N < \dfrac{e^{nLT}R^n}{2^{n+1} r^n}\log\dfrac{1}{2\delta}$ | Theorem 2 |

The two bounds match up to logarithmic factors, so the burden $\tilde\Theta\big((e^{LT}R/r)^n\big)$ is
intrinsic: it is exponential in the dimension and degrades exponentially with the horizon.  This is a
property of the problem, not of any particular estimator.


## Geometry and dynamics both matter

On the globally Lipschitz system $\dot x = x,\ \dot y = 0$, three initial sets of equal area $\pi$,
centered at $(2, 0)$, are sampled uniformly, propagated by $\varphi_T(x, y) = (e^T x, y)$, and estimated
by the convex hull of the endpoints:

* **circle**, $r_0 = 1$;
* **opened triangle**, $r_0 \approx 0.309$;
* **triangle**, $r_0 = 0$, which violates the positive-reach assumption.

The error is the symmetric Hausdorff distance to the true reachable set, averaged over 50 trials (shading:
95% confidence intervals).  The black dash-dotted and dashed curves are the upper bound (Theorem 1) and
the minimax lower bound (Theorem 2), with $\delta = 0.05$ and $r_0 = 0.309$.

![Lipschitz bound validation](results/lipschitz_bound_validation/lipschitz_bound_validation.png)

* **Geometry.** The triangle, which lacks positive reach, has the largest error.  The circle has the
  smallest.
* **Dynamics.** The error grows with the horizon as the flow expands the $x$-direction.  Over time the
  empirical curves run parallel to the lower-bound scale.
* **Sample size.** The decay in $N$ lies between the upper and the minimax lower bound.


## Experiments

Each experiment is one script in `experiments/`.  Its per-trial results are saved in
`results/<experiment>/`.

* **Quadratic flow illustration** (`quadratic_flow_illustration`)
  * *System:* the non-globally-Lipschitz flow $\dot x = x^2,\ \dot y = 0$,
    $\varphi_T(x, y) = (x/(1 - Tx), y)$.
  * *Initial sets:* a unit disk and a five-pointed star of the same circumradius, centered at $(2, 0)$,
    sampled uniformly.
  * *Estimator and error:* the convex hull on a $170 \times 170$ grid, and the symmetric Hausdorff
    distance to 35,000 propagated points; 50 trials.
  * *Sweeps:* $N \in \{10, \dots, 10^4\}$ at $T = 0.22$, and $T \in [0.01, 0.33]$ at $N = 1000$.
* **Lipschitz bound validation** (`lipschitz_bound_validation`)
  * *Setting:* the experiment shown above.
  * *Sweeps:* $N \in \{3\times10^2, \dots, 3\times10^5\}$ at $T = 1$, and $T \in [0.01, 2]$ at $N = 1000$.
* **Density lower bound** (`density_lower_bound`)
  * *System:* the unit disk centered at $(1, 0)$, $\dot x = 2x,\ \dot y = 0$, $T = 0.5$.
  * *Sampling:* densities $p_\beta \propto (1 - r)^\beta$ with $\beta \in \{0, 2, 4\}$.  They share the same
    support and vanish at the boundary for $\beta > 0$.
  * *Estimators:* convex hull, union of balls of radius $h = 0.05$, and Christoffel sublevel set
    (degree 6).
  * *Trials:* $N \in \{10, \dots, 10^6\}$, 50 seeds, 5–95% bands.
* **Robot-arm dimension scaling** (`robotarm_dimension_scaling`, with `robotarm_slope_fit`)
  * *System:* vertical planar $n$-link arms in MuJoCo, $n \in \{2, 3, 4\}$, state $x = [q, v] \in \mathbb R^{2n}$.
    Links have length 0.5, capsule radius 0.035, density 1000, damping 0.2 and armature 0.01; the time
    step is $2\times10^{-3}$ s.
  * *Controller:* a non-adaptive inverse-dynamics tracking controller,
    $\tau = M(q)\ddot q_d + C(q,v)v + g(q) - K_p e - K_d\dot e$, with $K_p = 0.01$, $K_d = 0.005$ and
    $|\tau| \le 100$.  The reference is $q_{d,i}(t) = q_{c,i} + 0.08\sin(0.5t + \phi_i)$.
  * *Initial set and horizon:* $[-0.1, 0.1]^{2n}$, propagated to $T = 1$.
  * *Sampling:* uniform, or Algorithm 1 with $n_{\rm adv} = 1$ and $\eta = 0.2$.
  * *Error:* directed Hausdorff distance from 2,000 of 20,000 reference endpoints to the samples;
    $N \in \{1, \dots, 3000\}$, 10 seeds.
  * *Slope fit:* `robotarm_slope_fit` fits the log–log slopes with $|{\rm slope}(d)| \approx 1/(a d^b + c)$.
* **Robot-arm time sweep** (`robotarm_time_sweep`)
  * *Setting:* the same arms with uniform sampling and $N = 1000$.
  * *Sweep:* $T \in [0.01, 2]$ (11 values), with a 5,000-point reference cloud and 5 seeds.
* **Adversarial sampling intensity** (`adversarial_intensity`)
  * *System:* $\dot x = x^2,\ \dot y = 0$ with the circle, opened triangle and triangle.
  * *Sampling:* $N \in \{10, 100, 1000\}$ endpoints from Algorithm 1 with $n_{\rm adv} \in \{0, \dots, 4\}$
    ($n_{\rm adv} = 0$ is uniform sampling).
  * *Estimators and error:* convex hull and Christoffel sublevel set; Hausdorff distance to a
    12,000-point reference over $T \in [0.01, 0.29]$; 50 trials.


## Installation

```shell
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

MuJoCo is needed only to *run* the robot-arm experiments.  Re-plotting from the shipped
CSVs and all planar experiments need only NumPy, SciPy, Matplotlib and Shapely.


## Reproduce

Run every command from the repository root.  Each script writes per-trial results to `results/<experiment>/trials.csv`
and then plots from them.  `--plot-only` skips the experiment and regenerates the figures and tables from
the shipped CSVs, in seconds.

```shell
python -m experiments.quadratic_flow_illustration
python -m experiments.lipschitz_bound_validation
python -m experiments.density_lower_bound
python -m experiments.adversarial_intensity
python -m experiments.robotarm_dimension_scaling   # MuJoCo
python -m experiments.robotarm_slope_fit           # uses the output of robotarm_dimension_scaling
python -m experiments.robotarm_time_sweep          # MuJoCo
```

Regenerate all figures and tables from the shipped results:

```shell
for s in quadratic_flow_illustration lipschitz_bound_validation robotarm_dimension_scaling \
         density_lower_bound robotarm_time_sweep adversarial_intensity; do
    python -m experiments.$s --plot-only
done
python -m experiments.robotarm_slope_fit
```

Useful options:

```shell
# subsets of the sweeps
python -m experiments.robotarm_dimension_scaling --methods uniform --n_values 2 --budgets 1,10,100
python -m experiments.density_lower_bound --budgets 10,100,1000 --seeds 10
python -m experiments.adversarial_intensity --estimators christoffel

# robot arm: distance to the convex hull of the samples instead of to the samples
python -m experiments.robotarm_dimension_scaling --metric convex_hull

# tests
python -m pytest
```

## Code structure

```text
reachapprox/                        # shared library
├─ flows.py                         # analytic flows: x' = x^2 (+ Jacobian), x' = a x
├─ geometry.py                      # disk, star, (opened) triangle; uniform sampling, boundary points, projection
├─ estimators.py                    # convex hull, Christoffel sublevel set, evaluation grids
├─ metrics.py                       # Hausdorff distances between clouds, boundaries and estimates
├─ adversarial.py                   # Algorithm 1 for the planar quadratic system
├─ utils.py                         # seeding, 95% CIs, CSV I/O, plotting helpers
└─ robotarm/
   ├─ arm.py                        # MuJoCo n-link arm, inverse-dynamics tracking controller, parallel rollouts
   ├─ sampling.py                   # uniform box sampling and Algorithm 1 on the box
   └─ metrics.py                    # directed Hausdorff to the sample cloud / to its convex hull
experiments/                        # one script per experiment; python -m experiments.<name>
├─ quadratic_flow_illustration.py   # x' = x^2, disk vs. star
├─ lipschitz_bound_validation.py    # x' = x, empirical error vs. Theorems 1-2
├─ robotarm_dimension_scaling.py    # robot arm, uniform vs. adversarial
├─ density_lower_bound.py           # vanishing boundary density
├─ robotarm_slope_fit.py            # slope vs. state dimension
├─ robotarm_time_sweep.py           # robot arm, error vs. horizon
└─ adversarial_intensity.py         # number of adversarial updates
results/<experiment>/               # per-trial CSVs, tables, and figures, one folder per script
tests/                              # flows, geometry, estimators, samplers, metrics
```


## Citation

```bibtex
@inproceedings{liu2026limits,
  title     = {On the Limits of Sampling-Based Reachability: Geometry, Dynamics, and Sample Complexity},
  author    = {Liu, Jixian and Tabbara, Ihab and Sibai, Hussein and Mallada, Enrique},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```
