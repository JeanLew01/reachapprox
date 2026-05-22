"""Compute directed point-cloud coverage error for robot-arm endpoints."""

from __future__ import annotations

import argparse

import numpy as np
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True, help="Reference .npz file with XT.")
    parser.add_argument("--sample", required=True, help="Smaller .npz file with XT.")
    parser.add_argument("--out", default=None, help="Optional path for a one-line text result.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ref_data = np.load(args.ref, allow_pickle=False)
    sample_data = np.load(args.sample, allow_pickle=False)
    XT_ref = np.asarray(ref_data["XT"], dtype=float)
    XT_sample = np.asarray(sample_data["XT"], dtype=float)
    if XT_ref.ndim != 2 or XT_sample.ndim != 2:
        raise ValueError("XT arrays must be two-dimensional.")
    if XT_ref.shape[1] != XT_sample.shape[1]:
        raise ValueError(
            f"State dimensions differ: ref has {XT_ref.shape[1]}, sample has {XT_sample.shape[1]}."
        )
    n = int(ref_data["n"]) if "n" in ref_data else XT_ref.shape[1] // 2
    if n not in (2, 3):
        raise ValueError(f"Expected n=2 or n=3; got n={n}.")

    tree = cKDTree(XT_sample)
    distances, _ = tree.query(XT_ref, k=1)
    error = float(np.max(distances))
    result = f"directed coverage error e_N = {error:.12g}"
    print(result)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(result + "\n")


if __name__ == "__main__":
    main()
