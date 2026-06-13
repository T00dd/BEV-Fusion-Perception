import argparse
import csv
from pathlib import Path
import numpy as np

import scipy.io

def write_cones_csv(path: Path, cones: np.ndarray) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y"])
        for x, y in cones:
            writer.writerow([float(x), float(y)])

def is_open_loop(cones_left: np.ndarray, cones_right: np.ndarray, tolerance: float=3.0) -> bool:
    def ratio(p: np.ndarray) -> float:
        step = np.linalg.norm(np.diff(p, axis=0), axis=1)
        return float(np.linalg.norm(p[-1] - p[0])/ np.median(step))
    
    return ratio(cones_left) > tolerance and ratio(cones_right) > tolerance

def initial_direction(cones_left: np.ndarray, cones_right: np.ndarray) -> np.ndarray:
    left_leaf = cones_left[0]
    nearest_right = cones_right[np.argmin(np.linalg.norm(cones_right - left_leaf, axis=1))]
    edge_vector = left_leaf - nearest_right
    edge_vector = edge_vector / np.linalg.norm(edge_vector)
    perp = np.array([edge_vector[1], -edge_vector[0]])

    centroid = 0.5 * (cones_left.mean(axis=0) + cones_right.mean(axis=0))
    start_mp = 0.5 * (left_leaf + nearest_right)

    if np.dot(perp, centroid - start_mp) < 0:
        perp = -perp

    return perp

def shift_to_leaf(cones_left: np.ndarray, cones_right: np.ndarray):
    shift = cones_left[0]
    nearest_right = cones_right[np.argmin(np.linalg.norm(cones_right - shift, axis=1))]
    new_origin = 0.5*(shift + nearest_right)
    return cones_left - new_origin, cones_right - new_origin, new_origin

def process_cones_for_reconstructor(cones_left: np.ndarray, cones_right: np.ndarray):
    if not is_open_loop(cones_left, cones_right):
        print("loop is closed, no shift applied")
        return cones_left, cones_right, np.zeros(2)
    
    print("open loop detected, applying shift to leaf")
    cones_left, cones_right, shift = shift_to_leaf(cones_left, cones_right)

    # check the initial direction matches the one of the chain
    algo_dir = initial_direction(cones_left, cones_right)
    chain_dir = cones_left[1] - cones_left[0]
    chain_dir = chain_dir / np.linalg.norm(chain_dir)

    if np.dot(algo_dir, chain_dir) < 0:
        print("initial direction is opposite to chain direction, flipping")
        cones_left = cones_left[::-1]
        cones_right = cones_right[::-1]
        cones_left, cones_right, shift2 = shift_to_leaf(cones_left, cones_right)
        shift = shift + shift2
    else:
        print("initial direction matches chain direction, no flip needed")

    return cones_left, cones_right, shift

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    mat = scipy.io.loadmat(str(args.mat))

    left = np.asarray(mat["cones_left"], dtype=float)
    right = np.asarray(mat["cones_right"], dtype=float)

    left, right, _ = process_cones_for_reconstructor(left, right)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    write_cones_csv(args.out_dir / "cones_left.csv", left)
    write_cones_csv(args.out_dir / "cones_right.csv", right)
    print(f"csv successfully written")

if __name__ == "__main__":
    main()