"""Robot-arm dimension scaling in one figure, labeled by the state dimension.

Re-plots the saved results of `robotarm_dimension_scaling` and
`robotarm_slope_fit`; nothing is simulated.  Left and middle: mean directed
Hausdorff error of the convex hull against the sample size under uniform and
adversarial sampling (shading: 95% confidence intervals; dashed: fitted power
laws).  Right: the inverse rate 1/|slope| against the state dimension with its
linear fit a d + c and the 95% bootstrap intervals of the slopes.

    python -m experiments.robotarm_scaling_summary
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from reachapprox.utils import RESULTS_DIR, mean_ci95, read_csv, use_serif_fonts


TRIALS_CSV = RESULTS_DIR / "robotarm_dimension_scaling" / "trials.csv"
SLOPES_CSV = RESULTS_DIR / "robotarm_dimension_scaling" / "loglog_slopes.csv"
FIT_CSV = RESULTS_DIR / "robotarm_slope_fit" / "fit_parameters.csv"
OUT_DIR = RESULTS_DIR / "robotarm_scaling_summary"
FIGURE_PATH = OUT_DIR / "robotarm_scaling_summary.png"

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"
DIMENSION_STYLES = {4: ("#86b6ef", "o"), 6: ("#2a78d6", "s"), 8: ("#0d366b", "D")}
METHODS = {"uniform": "Uniform sampling", "adversarial": "Adversarial sampling"}
FIT_STYLES = {"uniform": {"marker": "o", "markerfacecolor": INK, "linestyle": "-"},
              "adversarial": {"marker": "s", "markerfacecolor": "white", "linestyle": "--"}}


def load_curves() -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    """(method, state dimension) -> (budgets, errors of shape (n_budgets, n_seeds))."""
    grouped: dict[tuple[str, int, int], list[float]] = {}
    for row in read_csv(TRIALS_CSV):
        grouped.setdefault((row["method"], int(row["state_dim"]), int(row["N"])), []).append(float(row["error"]))
    curves = {}
    for method, dim in sorted({(m, d) for m, d, _ in grouped}):
        budgets = np.array(sorted(N for m, d, N in grouped if (m, d) == (method, dim)))
        curves[(method, dim)] = (budgets, np.array([grouped[(method, dim, N)] for N in budgets]))
    return curves


def plot_figure() -> None:
    curves = load_curves()
    slopes = {(row["method"], int(row["state_dim"])): tuple(float(row[k]) for k in ("slope", "ci95_low", "ci95_high"))
              for row in read_csv(SLOPES_CSV)}
    fits = {row["method"]: (float(row["a"]), float(row["c"])) for row in read_csv(FIT_CSV)}

    fig, axes = plt.subplots(1, 3, figsize=(23.5, 7.4), gridspec_kw={"width_ratios": (1.15, 1.15, 1.0), "wspace": 0.24})
    for ax, (method, title) in zip(axes[:2], METHODS.items()):
        for dim, (color, marker) in DIMENSION_STYLES.items():
            budgets, values = curves[(method, dim)]
            mean, low, high = mean_ci95(values, axis=1)
            slope, intercept = np.polyfit(np.log(budgets), np.log(mean), 1)
            ax.plot(budgets, np.exp(intercept) * budgets**slope, color=color, linestyle="--", linewidth=1.6)
            ax.plot(budgets, mean, color=color, marker=marker, markersize=10, markeredgecolor="white",
                    markeredgewidth=1.2, linewidth=2.2, label=rf"$n={dim}$: slope ${slope:.2f}$")
            ax.fill_between(budgets, low, high, color=color, alpha=0.16, linewidth=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets])
        ax.minorticks_off()
        ax.set_xlabel(r"sample size $N$")
        ax.set_ylabel("Hausdorff distance", fontsize=26)
        ax.set_title(title)
        ax.legend(loc="lower left", frameon=False, fontsize=19, handlelength=2.2, labelspacing=0.4,
                  title="state dimension", title_fontsize=19)
    limits = [limit for ax in axes[:2] for limit in ax.get_ylim()]
    for ax in axes[:2]:
        ax.set_ylim(min(limits), max(limits))
        ax.tick_params(axis="y", which="minor", labelsize=14)

    ax = axes[2]
    dims = np.array(sorted(DIMENSION_STYLES), dtype=float)
    dense = np.linspace(dims[0] - 0.4, dims[-1] + 0.4, 100)
    ax.plot(dense, dense, color=MUTED, linewidth=1.6, linestyle=":")
    ax.text(dims[1] - 0.4, dims[1] + 0.3, r"worst case: $n$", fontsize=19, color=MUTED, ha="right", va="center")
    for method, label in METHODS.items():
        slope, low, high = (np.array([slopes[(method, int(d))][k] for d in dims]) for k in range(3))
        inverse = 1.0 / np.abs(slope)
        a, c = fits[method]
        style = FIT_STYLES[method]
        ax.plot(dense, a * dense + c, color=INK, linestyle=style["linestyle"], linewidth=2.0)
        ax.errorbar(dims, inverse, yerr=(inverse - 1.0 / np.abs(low), 1.0 / np.abs(high) - inverse), color=INK,
                    marker=style["marker"], markerfacecolor=style["markerfacecolor"], markersize=11,
                    markeredgewidth=1.8, linestyle="none", capsize=5,
                    label=rf"{label.split()[0].lower()}: ${a:.2f}\,n+{c:.2f}$")
    ax.set_xticks(dims)
    ax.set_xlabel(r"state dimension $n$")
    ax.set_ylabel(r"inverse rate $1/|\mathrm{slope}|$", fontsize=26)
    ax.set_title("Inverse rate vs. dimension")
    ax.legend(loc="upper left", frameon=False, fontsize=19, handlelength=1.2, labelspacing=0.4)

    for ax in axes:
        ax.xaxis.label.set_size(26)
        ax.title.set_size(26)
        ax.tick_params(axis="both", which="major", labelsize=19)
        ax.grid(True, which="major", color=GRID, linewidth=1.0)
        ax.set_axisbelow(True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main() -> None:
    use_serif_fonts("STIXGeneral")
    plot_figure()
    print(f"saved {FIGURE_PATH}")


if __name__ == "__main__":
    main()
