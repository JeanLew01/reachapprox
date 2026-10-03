"""Figure 8 (Appendix C.3): robot-arm Hausdorff error versus horizon T under uniform sampling.

For n = 2, 3, 4 links, N = 1000 uniform samples from [-0.1, 0.1]^{2n} are
propagated to T in linspace(0.01, 2, 11).  The error is the directed Hausdorff
distance from a 1,000-point subset of a 5,000-point reference cloud to the
sampled endpoints, averaged over 5 seeds.

    python -m experiments.fig8_robotarm_time_sweep
    python -m experiments.fig8_robotarm_time_sweep --plot-only
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from reachapprox.utils import RESULTS_DIR, mean_ci95, read_csv, use_serif_fonts, write_csv


LINK_COUNTS = (2, 3, 4)
TIME_GRID = tuple(float(t) for t in np.linspace(0.01, 2.0, 11))
BUDGET = 1000
N_SEEDS = 5
N_REF = 5000
REF_SUBSET = 1000
REFERENCE_SEED = 202606 + 30_000
EXPERIMENT_SEED = 77531 + 700_000

OUT_DIR = RESULTS_DIR / "fig8"
TRIALS_CSV = OUT_DIR / "fig8_trials.csv"
FIGURE_PATH = OUT_DIR / "fig8_robotarm_time_sweep.png"


def run_experiment(link_counts) -> list[dict]:
    # Imported here so that --plot-only works without MuJoCo.
    from reachapprox.robotarm.arm import propagate
    from reachapprox.robotarm.metrics import point_cloud_directed_hausdorff
    from reachapprox.robotarm.sampling import sample_initial_box

    rows = []
    for n in link_counts:
        X_ref = sample_initial_box(np.random.default_rng(REFERENCE_SEED + n), N_REF, n)
        for T in TIME_GRID:
            Y_ref = propagate(X_ref, n, T)
            errors = []
            for seed_index in range(N_SEEDS):
                seed = EXPERIMENT_SEED + 10_000 * n + 100 * seed_index + int(round(1000 * T))
                rng = np.random.default_rng(seed)
                ref = Y_ref[rng.choice(N_REF, size=REF_SUBSET, replace=False)]
                Y = propagate(sample_initial_box(rng, BUDGET, n), n, T)
                errors.append(point_cloud_directed_hausdorff(ref, Y))
                rows.append({"n": n, "state_dim": 2 * n, "N": BUDGET, "T": T, "seed": seed, "error": errors[-1]})
            print(f"n={n} T={T:.3f}: mean error={np.mean(errors):.4f}")
    return rows


def plot_figure(rows: list[dict]) -> None:
    grouped: dict[tuple[int, float], list[float]] = {}
    for row in rows:
        grouped.setdefault((int(row["n"]), round(float(row["T"]), 9)), []).append(float(row["error"]))
    link_counts = sorted({n for n, _ in grouped})

    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.12, 0.82, len(link_counts)))
    for n, color, marker in zip(link_counts, colors, ("o", "s", "D")):
        times = sorted(T for k, T in grouped if k == n)
        mean, lo, hi = mean_ci95(np.array([grouped[(n, T)] for T in times]), axis=1)
        ax.plot(times, mean, color=color, marker=marker, lw=2.0, ms=6, label=f"n={n}, dim={2 * n}")
        ax.fill_between(times, lo, hi, color=color, alpha=0.20)
    ax.set_xlabel("time T", fontsize=18)
    ax.set_ylabel("Hausdorff distance", fontsize=18)
    ax.set_title("Robot-Arm Time Sweep", fontsize=20)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(frameon=False, fontsize=12)
    fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot-only", action="store_true", help="re-plot from the saved trials CSV")
    parser.add_argument("--n_values", default=",".join(map(str, LINK_COUNTS)))
    args = parser.parse_args()
    use_serif_fonts("DejaVu Serif")

    if args.plot_only:
        rows = read_csv(TRIALS_CSV)
    else:
        rows = run_experiment(tuple(int(n) for n in args.n_values.split(",")))
        write_csv(rows, TRIALS_CSV)
        print(f"saved {TRIALS_CSV}")
    plot_figure(rows)
    print(f"saved {FIGURE_PATH}")


if __name__ == "__main__":
    main()
