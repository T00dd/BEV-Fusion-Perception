import subprocess
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repo_paths import SCRIPTS_DIR, DATA_DIR, RECONSTRUCTOR_BIN

sys.path.insert(0, str(SCRIPTS_DIR))
from cones_to_csv import process_cones_for_reconstructor, write_cones_csv

WAYPOINT_Z = 0.0

def compute_centerline_carla(
        cones: dict,
        start_x: float,
        start_y: float,
        carla_scale: float = 2.0,
        data_dir: Path = DATA_DIR,
        reconstructor_bin: Path = RECONSTRUCTOR_BIN,
        z: float = WAYPOINT_Z
) -> np.ndarray:
    
    left = np.asarray(cones["cones_left"], dtype=float)
    right = np.asarray(cones["cones_right"], dtype=float)

    left, right, shift = process_cones_for_reconstructor(left, right)

    data_dir.mkdir(parents=True, exist_ok=True)
    left_csv = data_dir / "cones_left.csv"
    right_csv = data_dir / "cones_right.csv"
    center_csv = data_dir / "centerline_output.csv"
    write_cones_csv(left_csv, left)
    write_cones_csv(right_csv, right)

    subprocess.run(
        [str(reconstructor_bin), str(left_csv), str(right_csv), str(center_csv)],
        check=True
    )

    centerline = np.loadtxt(center_csv, delimiter=",", skiprows=1)
    centerline = centerline + shift

    carla_x = start_x + carla_scale * centerline[:, 0]
    carla_y = start_y + carla_scale * centerline[:, 1]
    carla_z = np.full(len(centerline), z)
    waypoints = np.column_stack([carla_x, carla_y, carla_z])
    return waypoints

def draw_debug(world, waypoints: np.ndarray, life_time: float = 60.0) -> None:
    import carla
    red = carla.Color(255, 0, 0)
    for i in range(len(waypoints) - 1):
        p0 = carla.Location(x=float(waypoints[i, 0]), y=float(waypoints[i, 1]), z=float(waypoints[i, 2]) + 0.3)
        p1 = carla.Location(x=float(waypoints[i + 1, 0]), y=float(waypoints[i + 1, 1]), z=float(waypoints[i + 1, 2]) + 0.3)
        world.debug.draw_line(p0, p1, 0.1, red, float(life_time))