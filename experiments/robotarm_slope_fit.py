"""Figure 7 and Table 3 (Appendix C.3): fit of the dimension-dependent log-log slopes.

The inverse-rate exponent 1/|slope| of the Table 2 slopes is fitted by least
squares as a linear function of the state dimension, 1/|slope(d)| = a d + c,
over d = 4, 6, 8, separately for uniform and adversarial sampling; the fitted
slope curve is slope(d) = -1 / (a d + c).  Requires
results/robotarm_dimension_scaling/loglog_slopes.csv, produced by
`python -m experiments.robotarm_dimension_scaling [--plot-only]`.

    python -m experiments.robotarm_slope_fit
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from reachapprox.utils import RESULTS_DIR, read_csv, write_csv


SLOPES_CSV = RESULTS_DIR / "robotarm_dimension_scaling" / "loglog_slopes.csv"
OUT_DIR = RESULTS_DIR / "robotarm_slope_fit"
PARAMS_CSV = OUT_DIR / "fit_parameters.csv"
FIGURE_PATH = OUT_DIR / "slope_fit.png"
STYLE = {"uniform": ("o", "Uniform"), "adversarial": ("s", "Adversarial")}


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Serif", "mathtext.fontset": "dejavuserif",
                         "axes.unicode_minus": False})
    slopes: dict[str, dict[int, float]] = {}
    for row in read_csv(SLOPES_CSV):
        slopes.setdefault(row["method"], {})[int(row["state_dim"])] = float(row["slope"])

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    d_grid = np.linspace(4, 8, 200)
    params_rows = []
    print("Table 3: 1/|slope(d)| = a d + c")
    for method in ("uniform", "adversarial"):
        dims = np.array(sorted(slopes[method]), dtype=float)
        values = np.array([slopes[method][int(d)] for d in dims])
        a, c = np.polyfit(dims, 1.0 / np.abs(values), 1)
        max_residual = float(np.max(np.abs(values + 1.0 / (a * dims + c))))
        params_rows.append({"method": method, "a": a, "c": c, "max_slope_residual": max_residual})
        print(f"  {method:<12} a={a:.4f}  c={c:.4f}  (max slope residual {max_residual:.4f})")

        marker, label = STYLE[method]
        ax.scatter(dims, values, marker=marker, label=f"{label} data")
        ax.plot(d_grid, -1.0 / (a * d_grid + c), label=f"{label} fit")

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
