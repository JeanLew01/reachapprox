# MuJoCo n-link robot arm benchmark

This directory contains a MuJoCo model for a vertical planar serial n-link robot arm under gravity. The main tracking experiments use n = 2 and n = 3. Each state is

```text
x = [q, v] in R^(2n)
```

where `q` are the joint angles and `v` are the joint velocities.

MuJoCo simulates rigid-body dynamics of the form

```text
M(q) vdot + c(q, v) = tau.
```

MuJoCo inverse dynamics is used to build a slowly moving trajectory-tracking controller. The desired trajectory is

```text
q_d,i(t) = q_c,i + A_i sin(omega t + phi_i)
```

with a deliberately low frequency `omega = 0.5`, so the arm moves realistically without immediately collapsing the propagated uncertainty set.

The controller is

```text
e = q - q_d(t)
edot = qdot - qdot_d(t)
s = edot + Lambda e

tau_ff = M(q) qddot_d(t) + C(q, qdot) qdot + g(q)
d_hat_dot = Gamma s - sigma d_hat
tau = tau_ff - Kp e - Kd edot - d_hat
```

Torques are clipped for numerical stability. The adaptive state `d_hat` is reset to zero for every sampled trajectory.

Once the reference and controller are fixed, the system is time-varying over the fixed horizon:

```text
x_dot = F_n(t, x),  x = [q, v].
```

The experiment is closed-loop robot-arm uncertainty propagation. Given an initial uncertainty set `S_0^arm`, the fixed-horizon terminal set is

```text
S_T^arm = { phi_n(T, x0) : x0 in S_0^arm }.
```

The goal is Dimension-Dependent Approximation of Robot-Arm Uncertainty Propagation: generate endpoint samples, build a convex-hull estimator of the terminal propagated uncertainty set, and report its Hausdorff Distance against a large reference endpoint cloud. This is not controller design and not online safety enforcement.

Typical commands from this directory:

```bash
python test_n_link_arm.py
python generate_endpoint_samples.py --n 2 --N 1000 --T 1.0
python generate_endpoint_samples.py --n 3 --N 1000 --T 1.0
python robotarm_dim_scaling.py --N_ref 5000 --coverage_subset 500 --budgets 1,3,10,30,100 --n_seeds 5
python animate_n_link_arm.py --n 2 --T 1.0
```
