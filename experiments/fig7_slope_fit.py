"""Figure 7 and Table 3 (Appendix C.3): fit of the dimension-dependent log-log slopes.

The magnitude of the Table 2 slopes is fitted as m(d) = 1 / (a d^b + c) over
state dimensions d = 4, 6, 8, separately for uniform and adversarial sampling.
As in the paper, the slopes are rounded to four decimals (as reported in
Table 2) before fitting.  Requires results/fig3/table2_slopes.csv, produced by
`python -m experiments.fig3_robotarm_dim_scaling [--plot-only]`.

    python -m experiments.fig7_slope_fit
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

from reachapprox.utils import RESULTS_DIR, read_csv, write_csv


SLOPES_CSV = RESULTS_DIR / "fig3" / "table2_slopes.csv"
OUT_DIR = RESULTS_DIR / "fig7"
PARAMS_CSV = OUT_DIR / "table3_fit_parameters.csv"
FIGURE_PATH = OUT_DIR / "fig7_slope_fit.png"
INITIAL_GUESS = {"uniform": (0.5, 1.0, 1.0), "adversarial": (0.8, 1.0, 0.5)}
STYLE = {"uniform": ("o", "Uniform"), "adversarial": ("s", "Adversarial")}


def slope_magnitude(d, a, b, c):
    return 1.0 / (a * d**b + c)


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Serif", "mathtext.fontset": "dejavuserif",
                         "axes.unicode_minus": False})
    slopes: dict[str, dict[int, float]] = {}
    for row in read_csv(SLOPES_CSV):
        slopes.setdefault(row["method"], {})[int(row["state_dim"])] = round(float(row["slope"]), 4)

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    d_grid = np.linspace(4, 8, 200)
    params_rows = []
    print("Table 3: m(d) = 1 / (a d^b + c)")
    for method in ("uniform", "adversarial"):
        dims = np.array(sorted(slopes[method]), dtype=float)
        values = np.array([slopes[method][int(d)] for d in dims])
        with warnings.catch_warnings():
            # Three parameters through three points: an exact fit with undefined covariance.
            warnings.simplefilter("ignore", OptimizeWarning)
            (a, b, c), _ = curve_fit(slope_magnitude, dims, np.abs(values), p0=INITIAL_GUESS[method],
                                     maxfev=10_000)
        params_rows.append({"method": method, "a": a, "b": b, "c": c})
        print(f"  {method:<12} a={a:.4f}  b={b:.4f}  c={c:.4f}")

        marker, label = STYLE[method]
        ax.scatter(dims, values, marker=marker, label=f"{label} data")
        ax.plot(d_grid, -slope_magnitude(d_grid, a, b, c), label=f"{label} fit")

    ax.set_xlabel("State dimension")
    ax.set_ylabel("Empirical log-log slope")
    ax.set_title("Dimension-dependent slope fitting")
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(name) for name in ("Uniform data", "Uniform fit", "Adversarial data", "Adversarial fit")]
    ax.legend([handles[i] for i in order], [labels[i] for i in order])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=300)
    plt.close(fig)
    write_csv(params_rows, PARAMS_CSV)
    print(f"saved {FIGURE_PATH}\nsaved {PARAMS_CSV}")


if __name__ == "__main__":
    main()
