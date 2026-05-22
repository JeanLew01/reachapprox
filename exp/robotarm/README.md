# MuJoCo n-link robot arm benchmark

This directory contains a MuJoCo model for a vertical planar serial n-link robot arm under gravity, initially supporting n = 2 and n = 3. Each state is

```text
x = [q, v] in R^(2n)
```

where `q` are the joint angles and `v` are the joint velocities.

MuJoCo simulates rigid-body dynamics of the form

```text
M(q) vdot + c(q, v) = tau.
```

The benchmark fixes a closed-loop controller,

```text
tau = Kp(q_goal - q) - Kd v
```

or optionally

```text
tau = qfrc_bias + Kp(q_goal - q) - Kd v.
```

Once the controller is fixed, the system is autonomous:

```text
x_dot = F_n(x),  x = [q, v].
```

The experiment is closed-loop robot-arm uncertainty propagation. Given an initial uncertainty set `S_0^arm`, the target set is

```text
S_T^arm = { phi_n(T, x0) : x0 in S_0^arm }.
```

The goal is to generate endpoint samples and study finite-sample approximation of this terminal propagated uncertainty set, for example through geometric coverage or Hausdorff-style errors. This is not controller design and not online safety enforcement.

Typical commands from this directory:

```bash
python test_n_link_arm.py
python generate_endpoint_samples.py --n 2 --N 1000 --T 2.0 --controller_mode pd
python generate_endpoint_samples.py --n 3 --N 1000 --T 2.0 --controller_mode pd
python animate_n_link_arm.py --n 2 --T 2.0 --controller_mode pd
```
