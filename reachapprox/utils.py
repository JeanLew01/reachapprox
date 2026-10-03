"""Seeding, statistics, CSV, and plotting helpers shared by the experiment scripts."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


def stable_seed_offset(set_name: str, method: str, t: float, budget: int) -> int:
    """Deterministic, platform-independent seed offset for one experimental condition."""
    text = f"{set_name}:{method}:{t:.8f}:{budget}"
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text))


def mean_ci95(values: np.ndarray, axis: int = -1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and normal-approximation 95% confidence interval of the mean."""
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=axis)
    if values.shape[axis] == 1:
        return mean, mean, mean
    half_width = 1.96 * values.std(axis=axis, ddof=1) / np.sqrt(values.shape[axis])
    return mean, np.maximum(mean - half_width, np.finfo(float).tiny), mean + half_width


def parallel_map(fn: Callable, tasks: list, workers: int | None = None) -> Iterator:
    """Ordered map of a top-level function over independent tasks in worker processes.

    Every task seeds its own random generator, so results do not depend on `workers`.
    """
    workers = min(workers or os.cpu_count() or 1, len(tasks))
    if workers <= 1:
        yield from map(fn, tasks)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(fn, tasks)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the experiment without --plot-only first.")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def use_serif_fonts(font: str = "DejaVu Serif") -> None:
    plt.rcParams.update({"font.family": "serif", "font.serif": [font], "mathtext.fontset": "stix"})


def power_of_ten_labels(values: tuple[int, ...]) -> list[str]:
    """Tick labels such as 3x10^2 for sample sizes."""
    labels = []
    for n in values:
        exponent = int(np.floor(np.log10(n)))
        labels.append(rf"${n // 10 ** exponent}\times10^{{{exponent}}}$")
    return labels
