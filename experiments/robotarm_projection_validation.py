"""Check the normalized NNLS hull projection with a supporting-plane dual bound.

The original experiment archives errors but not endpoint clouds. Reconstruct
seed record 0 at uniform budgets 3, 30, 300, 3000 and adversarial budget 30 in
state dimensions 4, 6, 8, preserving its RNG stream. From each original 2,000
point reference subset, propagate 16 initial entries and the 16 remaining
initial states of greatest norm. This limited check does not rerun all trials
or validate error from finite reference sampling or MuJoCo integration.

    python -m experiments.robotarm_projection_validation --workers 8
    python -m experiments.robotarm_projection_validation --cached

The cached endpoint arrays allow the solver check to run without MuJoCo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform

import numpy as np
import scipy

from experiments.robotarm_dimension_scaling import (
    ADVERSARIAL_SEED_OFFSET, EXPERIMENT_SEED, N_REF, REFERENCE_SEED,
    REF_SUBSET, SAMPLE_BUDGETS, T_HORIZON,
)
from reachapprox.robotarm.metrics import (
    NNLS_EQUALITY_WEIGHT, NNLS_ZERO_TOL, convex_hull_projection_certificate,
)
from reachapprox.utils import RESULTS_DIR, write_csv


OUT_DIR = RESULTS_DIR / "robotarm_projection_validation"
CACHE_PATH = OUT_DIR / "endpoints.npz"
ABSOLUTE_TOLERANCE = 1e-5
DIRECTED_RELATIVE_TOLERANCE = 1e-3
UNIFORM_BUDGETS = (3, 30, 300, 3000)


def reconstruct_cases(workers: int) -> tuple[dict[str, np.ndarray], str]:
    import mujoco

    from reachapprox.robotarm.arm import propagate
    from reachapprox.robotarm.sampling import (
        N_ADV, adversarial_endpoints, sample_initial_box,
    )

    arrays = {}
    for n in (2, 3, 4):
        master = sample_initial_box(np.random.default_rng(REFERENCE_SEED + n), N_REF, n)
        for method, budgets in (("uniform", UNIFORM_BUDGETS), ("adversarial", (30,))):
            for budget in budgets:
                seed = EXPERIMENT_SEED + 10_000 * n
                seed += budget if method == "uniform" else ADVERSARIAL_SEED_OFFSET
                rng = np.random.default_rng(seed)
                reference_indices = rng.choice(N_REF, size=REF_SUBSET, replace=False)
                # Keep the first 16 plus the 16 farthest remaining initial states.
                subset_initials = master[reference_indices]
                extreme = 16 + np.argsort(np.linalg.norm(subset_initials[16:], axis=1))[-16:]
                selected = np.concatenate([np.arange(16), extreme])
                references = propagate(subset_initials[selected], n, T_HORIZON, workers=workers)
                if method == "uniform":
                    initials = sample_initial_box(rng, budget, n)
                    samples = propagate(initials, n, T_HORIZON, workers=workers)
                    saved_seed = seed
                else:
                    # Earlier budgets consume only the initial sampling draws.
                    for earlier in SAMPLE_BUDGETS:
                        if earlier >= budget:
                            break
                        sample_initial_box(rng, int(np.ceil(earlier / (N_ADV + 1))), n)
                    samples = adversarial_endpoints(rng, budget, n, T_HORIZON)
                    saved_seed = seed + budget
                key = f"{method}_d{2 * n}_N{budget}"
                arrays[f"{key}_vertices"] = samples
                arrays[f"{key}_references"] = references
                arrays[f"{key}_reference_indices"] = reference_indices[selected]
                arrays[f"{key}_seed"] = np.array(saved_seed)
                print(f"reconstructed {key}, seed={saved_seed}", flush=True)
    return arrays, mujoco.__version__


def validate(arrays: dict[str, np.ndarray]) -> tuple[list[dict], list[dict]]:
    rows, point_rows = [], []
    for key in sorted(k[:-9] for k in arrays if k.endswith("_vertices")):
        vertices = arrays[f"{key}_vertices"]
        references = arrays[f"{key}_references"]
        distances, lower, upper = convex_hull_projection_certificate(references, vertices)
        error_bounds = np.maximum(np.abs(distances - lower), np.abs(upper - distances))
        directed = float(distances.max())
        directed_error_bound = max(abs(directed - float(lower.max())), abs(float(upper.max()) - directed))
        relative = directed_error_bound / max(directed, np.finfo(float).eps)
        passed = bool(np.max(error_bounds) <= ABSOLUTE_TOLERANCE and relative <= DIRECTED_RELATIVE_TOLERANCE)
        rows.append({
            "case": key, "seed": int(arrays[f"{key}_seed"]),
            "state_dim": vertices.shape[1], "N": len(vertices), "n_queries": len(references),
            "max_reported_distance": directed,
            "max_point_distance_error_bound": float(error_bounds.max()),
            "directed_distance_error_bound": directed_error_bound,
            "directed_relative_error_bound": relative, "passed": passed,
        })
        for index, (distance, low, high, error) in enumerate(zip(distances, lower, upper, error_bounds)):
            point_rows.append({
                "case": key, "reference_index": int(arrays[f"{key}_reference_indices"][index]),
                "reported_distance": float(distance), "dual_lower_bound": float(low),
                "primal_upper_bound": float(high), "absolute_error_bound": float(error),
            })
        print(f"{key}: max point bound={error_bounds.max():.3e}, directed relative bound={relative:.3e}", flush=True)
    return rows, point_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cached", action="store_true", help="use archived endpoint clouds; do not simulate")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.cached:
        with np.load(CACHE_PATH, allow_pickle=False) as archive:
            arrays = dict(archive)
        mujoco_version = str(arrays.pop("mujoco_version"))
    else:
        arrays, mujoco_version = reconstruct_cases(args.workers)
        np.savez_compressed(CACHE_PATH, **arrays, mujoco_version=np.array(mujoco_version))
    rows, point_rows = validate(arrays)
    write_csv(rows, OUT_DIR / "summary.csv")
    write_csv(point_rows, OUT_DIR / "point_bounds.csv")
    summary = {
        "method": "Feasible normalized NNLS candidate and supporting-plane dual distance bound",
        "arithmetic": "float64 with a 64 * eps * dimension * coordinate-scale cushion; not interval arithmetic",
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "scipy_version": scipy.__version__, "mujoco_version": mujoco_version,
        "nnls_equality_weight": NNLS_EQUALITY_WEIGHT, "zero_threshold": NNLS_ZERO_TOL,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "directed_relative_tolerance": DIRECTED_RELATIVE_TOLERANCE,
        "endpoint_cache_sha256": hashlib.sha256(CACHE_PATH.read_bytes()).hexdigest(),
        "n_cases": len(rows), "n_queries": len(point_rows),
        "max_point_distance_error_bound": max(r["max_point_distance_error_bound"] for r in rows),
        "max_directed_relative_error_bound": max(r["directed_relative_error_bound"] for r in rows),
        "passed": all(r["passed"] for r in rows),
        "coverage": "Seed index 0, dimensions 4/6/8; uniform N=3/30/300/3000; adversarial N=30; 32 queries per case",
        "limitations": "Reconstructed subsets, not all archived trials; no bound on finite-reference or integration error",
    }
    (OUT_DIR / "validation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["passed"]:
        raise SystemExit("Projection validation exceeds its predeclared tolerance; inspect the saved bounds.")


if __name__ == "__main__":
    main()
