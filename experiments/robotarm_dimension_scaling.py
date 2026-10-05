"""Figure 3, Table 2, and Table 4 (Section 5.2, Appendix C.3): robot-arm dimension scaling.

Closed-loop vertical n-link arms (n = 2, 3, 4; state dimension 2n) are
propagated in MuJoCo to T = 1 from the box [-0.1, 0.1]^{2n}.  For each budget
N, endpoints are drawn by uniform sampling or by Algorithm 1 (adversarial
sampling, n_adv = 1, eta = 0.05).  The reachable set is estimated by the convex
hull of the endpoints, and the error is the directed Hausdorff distance from a
2,000-point subset of a 20,000-point reference cloud to that hull.

Outputs (results/robotarm_dimension_scaling/):
  trials.csv                                       per-seed errors
  uniform_sampling.png, adversarial_sampling.png   Figure 3
  loglog_slopes.csv                                Table 2: log-log slopes of mean error vs N
  adversarial_improvement.csv                      Table 4: adversarial improvement over uniform

    python -m experiments.robotarm_dimension_scaling                 # several CPU-hours
    python -m experiments.robotarm_dimension_scaling --plot-only     # figures/tables from CSV
    python -m experiments.robotarm_dimension_scaling --methods adversarial   # keep saved uniform rows
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from reachapprox.robotarm.metrics import convex_hull_directed_hausdorff, point_cloud_directed_hausdorff
from reachapprox.utils import RESULTS_DIR, mean_ci95, read_csv, use_serif_fonts, write_csv


LINK_COUNTS = (2, 3, 4)
SAMPLE_BUDGETS = (1, 3, 10, 30, 100, 300, 1000, 3000)
N_SEEDS = 30
N_BOOTSTRAP = 2000
N_REF = 20_000
REF_SUBSET = 2_000
T_HORIZON = 1.0
REFERENCE_SEED = 202606
EXPERIMENT_SEED = 77531
ADVERSARIAL_SEED_OFFSET = 500_000

OUT_DIR = RESULTS_DIR / "robotarm_dimension_scaling"
TRIALS_CSV = OUT_DIR / "trials.csv"
SLOPES_CSV = OUT_DIR / "loglog_slopes.csv"
IMPROVEMENT_CSV = OUT_DIR / "adversarial_improvement.csv"
TITLES = {"uniform": "Robot-Arm Uniform Sampling", "adversarial": "Robot-Arm Adversarial Sampling"}


def directed_error(metric: str, ref: np.ndarray, samples: np.ndarray, seed: int) -> float:
    if metric == "point_cloud":
        return point_cloud_directed_hausdorff(ref, samples)
    return convex_hull_directed_hausdorff(ref, samples, np.random.default_rng(seed + 9_000_000))


def run_experiment(methods, link_counts, budgets, n_seeds: int, metric: str) -> list[dict]:
    # Imported here so that --plot-only works without MuJoCo.
    from reachapprox.robotarm.arm import propagate
    from reachapprox.robotarm.sampling import adversarial_endpoints, sample_initial_box

    rows = []
    for n in link_counts:
        X_ref = sample_initial_box(np.random.default_rng(REFERENCE_SEED + n), N_REF, n)
        Y_ref = propagate(X_ref, n, T_HORIZON)
        print(f"\nn={n} (state dim {2 * n}): reference cloud of {N_REF} endpoints ready")

        if "uniform" in methods:
            for budget in budgets:
                errors = []
                for seed_index in range(n_seeds):
                    seed = EXPERIMENT_SEED + 10_000 * n + 100 * seed_index + budget
                    rng = np.random.default_rng(seed)
                    ref = Y_ref[rng.choice(N_REF, size=REF_SUBSET, replace=False)]
                    Y = propagate(sample_initial_box(rng, budget, n), n, T_HORIZON)
                    errors.append(directed_error(metric, ref, Y, seed))
                    rows.append({"method": "uniform", "n": n, "state_dim": 2 * n, "N": budget, "seed": seed,
                                 "error": errors[-1], "metric": metric})
                print(f"  uniform     N={budget:<5} mean error={np.mean(errors):.4f}")

        if "adversarial" in methods:
            for seed_index in range(n_seeds):
                # One random stream per seed, shared by all budgets (and one reference subset).
                seed = EXPERIMENT_SEED + ADVERSARIAL_SEED_OFFSET + 10_000 * n + 100 * seed_index
                rng = np.random.default_rng(seed)
                ref = Y_ref[rng.choice(N_REF, size=REF_SUBSET, replace=False)]
                for budget in budgets:
                    Y = adversarial_endpoints(rng, budget, n, T_HORIZON)
                    rows.append({"method": "adversarial", "n": n, "state_dim": 2 * n, "N": budget,
                                 "seed": seed + budget, "error": directed_error(metric, ref, Y, seed + budget),
                                 "metric": metric})
                print(f"  adversarial seed {seed_index + 1}/{n_seeds} done")
    return rows


def mean_errors(rows: list[dict]) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    """(method, n) -> (budgets, per-budget error arrays of shape (n_budgets, n_seeds))."""
    grouped: dict[tuple[str, int, int], list[float]] = {}
    for row in rows:
        grouped.setdefault((row["method"], int(row["n"]), int(row["N"])), []).append(float(row["error"]))
    out = {}
    for method, n in sorted({(m, n) for m, n, _ in grouped}):
        budgets = np.array(sorted(N for m, k, N in grouped if (m, k) == (method, n)))
        out[(method, n)] = (budgets, np.array([grouped[(method, n, N)] for N in budgets]))
    return out


def plot_method(curves, method: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    link_counts = sorted(n for m, n in curves if m == method)
    colors = plt.cm.viridis(np.linspace(0.12, 0.82, len(link_counts)))
    for n, color, marker in zip(link_counts, colors, ("o", "s", "D")):
        budgets, values = curves[(method, n)]
        mean, lo, hi = mean_ci95(values, axis=1)
        ax.plot(budgets, mean, color=color, marker=marker, lw=2.0, ms=6, label=f"n={n}, dim={2 * n}")
        ax.fill_between(budgets, lo, hi, color=color, alpha=0.20)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of samples N", fontsize=18)
    ax.set_ylabel("Hausdorff distance", fontsize=18)
    ax.set_title(TITLES[method], fontsize=20)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(frameon=False, fontsize=12)
    path = OUT_DIR / f"{method}_sampling.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def loglog_slopes(curves) -> list[dict]:
    """Table 2: slope of log(mean error) vs log N over all budgets, with a 95% seed-bootstrap CI."""
    rng = np.random.default_rng(0)
    rows = []
    for (method, n), (budgets, values) in curves.items():
        log_budgets = np.log(budgets)
        slope = float(np.polyfit(log_budgets, np.log(values.mean(axis=1)), 1)[0])
        # Resample seeds independently per budget and refit the slope of the mean curve.
        picks = rng.integers(0, values.shape[1], size=(N_BOOTSTRAP, *values.shape))
        boot_means = np.take_along_axis(values[None, :, :], picks, axis=2).mean(axis=2)
        boot_slopes = np.polyfit(log_budgets, np.log(boot_means).T, 1)[0]
        low, high = np.quantile(boot_slopes, [0.025, 0.975])
        rows.append({"method": method, "n": n, "state_dim": 2 * n, "slope": slope,
                     "ci95_low": float(low), "ci95_high": float(high)})
    return rows


def improvement_table(curves) -> list[dict]:
    """Table 4: absolute and relative improvement of adversarial over uniform mean error."""
    rows = []
    for n in sorted(n for m, n in curves if m == "uniform"):
        if ("adversarial", n) not in curves:
            continue
        budgets, uniform = curves[("uniform", n)]
        _, adversarial = curves[("adversarial", n)]
        for N, u, a in zip(budgets, uniform.mean(axis=1), adversarial.mean(axis=1)):
            rows.append({"n": n, "state_dim": 2 * n, "N": int(N), "uniform_mean": u, "adversarial_mean": a,
                         "absolute_improvement": u - a, "relative_improvement_percent": 100.0 * (u - a) / u})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot-only", action="store_true", help="figures and tables from the saved trials CSV")
    parser.add_argument("--methods", default="uniform,adversarial")
    parser.add_argument("--n_values", default=",".join(map(str, LINK_COUNTS)))
    parser.add_argument("--budgets", default=",".join(map(str, SAMPLE_BUDGETS)))
    parser.add_argument("--n_seeds", type=int, default=N_SEEDS)
    parser.add_argument("--metric", choices=("convex_hull", "point_cloud"), default="convex_hull",
                        help="directed Hausdorff to the convex hull of the samples, or to the samples themselves")
    args = parser.parse_args()
    use_serif_fonts("DejaVu Serif")

    if args.plot_only:
        rows = read_csv(TRIALS_CSV)
    else:
        methods = args.methods.split(",")
        rows = run_experiment(
            methods,
            tuple(int(n) for n in args.n_values.split(",")),
            tuple(int(N) for N in args.budgets.split(",")),
            args.n_seeds,
            args.metric,
        )
        # Rerunning a subset of methods keeps the saved rows of the other methods.
        if TRIALS_CSV.exists():
            rows = [row for row in read_csv(TRIALS_CSV) if row["method"] not in methods] + rows
        write_csv(rows, TRIALS_CSV)
        print(f"saved {TRIALS_CSV}")

    curves = mean_errors(rows)
    for method in sorted({m for m, _ in curves}):
        plot_method(curves, method)

    slopes = loglog_slopes(curves)
    write_csv(slopes, SLOPES_CSV)
    print("\nTable 2: log-log slopes")
    for row in slopes:
        print(f"  {row['method']:<12} dim={row['state_dim']}: {row['slope']:.4f} "
              f"(95% CI {row['ci95_low']:.4f} to {row['ci95_high']:.4f})")

    improvement = improvement_table(curves)
    if improvement:
        write_csv(improvement, IMPROVEMENT_CSV)
        print("\nTable 4: adversarial improvement over uniform")
        for row in improvement:
            print(f"  n={row['n']} N={row['N']:<5} abs={row['absolute_improvement']:+.4f} "
                  f"rel={row['relative_improvement_percent']:+.2f}%")
    print(f"\nsaved {SLOPES_CSV}" + (f"\nsaved {IMPROVEMENT_CSV}" if improvement else ""))


if __name__ == "__main__":
    main()
