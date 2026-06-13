import io
import random
import sys
import time

import carla
import yaml

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repo_paths import (
    MAIN_CONFIG, TRACK_GENERATOR_DIR, TRACK_GENERATOR_CONFIG,
)

sys.path.insert(0, str(TRACK_GENERATOR_DIR))
from track_generator import Mode, SimType, TrackGenerator

DEFAULT_START_X = 75.8
DEFAULT_START_Y = 132.5
DEFAULT_CARLA_SCALE = 2.0
DEFAULT_BP_LEFT = "static.prop.blue_cone"
DEFAULT_BP_RIGHT = "static.prop.yellow_cone"


def load_spawner_config():
    cfg = {
        "enabled": True,
        "scale": DEFAULT_CARLA_SCALE,
        "start_x": DEFAULT_START_X,
        "start_y": DEFAULT_START_Y,
        "fit_to_map": True,
        "cone_blueprint_left": DEFAULT_BP_LEFT,
        "cone_blueprint_right": DEFAULT_BP_RIGHT,
    }
    try:
        with open(MAIN_CONFIG, "r") as f:
            data = yaml.safe_load(f) or {}
        ts = data.get("track_spawner", {}) or {}
        cfg.update({k: ts[k] for k in cfg.keys() if k in ts})
    except FileNotFoundError:
        print(f"[track_spawner] {MAIN_CONFIG} not found, using defaults.")
    return cfg


def clean_previous_track(world, client=None):
    all_actors = world.get_actors()
    cone_actors = [a for a in all_actors if "cone" in a.type_id.lower()]
    if not cone_actors:
        return

    if client is not None:
        client.apply_batch_sync(
            [carla.command.DestroyActor(a.id) for a in cone_actors], True
        )
    else:
        for a in cone_actors:
            try:
                a.destroy()
            except RuntimeError:
                pass

    print(f"Destroyed {len(cone_actors)} existing cone actors from previous track.")


def _resolve_blueprint(blueprints, bp_id):
    matches = blueprints.filter(bp_id)
    if len(matches) == 0:
        raise RuntimeError(
            f"Cone blueprint '{bp_id}' not found in this CARLA build. "
            f"Check the name in config.yaml (track_spawner.cone_blueprint_*)."
        )
    return matches[0]


def get_map_bounds(world, margin=5.0):
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        return None
    xs = [sp.location.x for sp in spawn_points]
    ys = [sp.location.y for sp in spawn_points]
    return (min(xs) + margin, max(xs) - margin,
            min(ys) + margin, max(ys) - margin)


def fit_scale_to_map(cones, start_x, start_y, scale, world):
    bounds = get_map_bounds(world)
    if bounds is None:
        print("[track_spawner] Could not determine map bounds; skipping fit check.")
        return scale

    min_x, max_x, min_y, max_y = bounds
    all_cones = list(cones["cones_left"]) + list(cones["cones_right"])
    if not all_cones:
        return scale

    max_off_x = max(abs(c[0]) for c in all_cones)
    max_off_y = max(abs(c[1]) for c in all_cones)

    room_x = min(max_x - start_x, start_x - min_x)
    room_y = min(max_y - start_y, start_y - min_y)

    if room_x <= 0 or room_y <= 0:
        raise RuntimeError(
            f"Start position ({start_x}, {start_y}) is outside the map bounds "
            f"x[{min_x:.1f},{max_x:.1f}] y[{min_y:.1f},{max_y:.1f}]. "
            f"Adjust track_spawner.start_x / start_y in config.yaml."
        )

    max_scale_x = room_x / max_off_x if max_off_x > 0 else scale
    max_scale_y = room_y / max_off_y if max_off_y > 0 else scale
    max_allowed = min(max_scale_x, max_scale_y)

    if scale > max_allowed:
        print(
            f"[track_spawner] Track too large for map: requested scale {scale:.2f} "
            f"exceeds max {max_allowed:.2f}. Shrinking to fit."
        )
        return max_allowed
    return scale


def spawn_track(cones, start_x, start_y, scale, bp_left_id, bp_right_id):
    cones_left = cones["cones_left"]
    cones_right = cones["cones_right"]

    client = carla.Client("localhost", 2000)
    world = client.get_world()

    clean_previous_track(world, client)

    blueprints = world.get_blueprint_library()
    model_cones_left = _resolve_blueprint(blueprints, bp_left_id)
    model_cones_right = _resolve_blueprint(blueprints, bp_right_id)

    spawned = []

    for cone_left in cones_left:
        location = carla.Location(
            start_x + (cone_left[0] * scale), start_y + (cone_left[1] * scale), 0.5
        )
        c = world.spawn_actor(model_cones_left, carla.Transform(location))
        c.set_simulate_physics(True)
        spawned.append(c)

    for cone_right in cones_right:
        location = carla.Location(
            start_x + (cone_right[0] * scale), start_y + (cone_right[1] * scale), 0.5
        )
        c = world.spawn_actor(model_cones_right, carla.Transform(location))
        c.set_simulate_physics(True)
        spawned.append(c)

    return spawned


def generate_and_spawn_track():
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8")

    # Load track_spawner placement/scale config (main project config)
    sp_cfg = load_spawner_config()
    start_x = sp_cfg["start_x"]
    start_y = sp_cfg["start_y"]
    scale = sp_cfg["scale"]
    spawning_enabled = sp_cfg["enabled"]
    bp_left_id = sp_cfg["cone_blueprint_left"]
    bp_right_id = sp_cfg["cone_blueprint_right"]
    fit_to_map = sp_cfg["fit_to_map"]

    # Load track-generator configuration (separate file, untouched)
    with open(TRACK_GENERATOR_CONFIG, "r") as file:
        config = yaml.safe_load(file)

    mode_choice = config["mode"]["parameters"].lower()
    voronoi = config["mode"]["voronoi"].upper()
    open_loop = config["mode"]["open_loop"]
    random_open_loop = config["mode"]["random_open_loop"]
    behind_ratio = config["mode"]["behind_ratio"]
    ahead_ratio = config["mode"]["ahead_ratio"]
    missing_cone_ratio = config["mode"]["missing_cone_ratio"]

    sim_type_str = config["simulation"]["sim_type"].upper()
    if sim_type_str == "FSDS":
        sim_type = SimType.FSDS
    elif sim_type_str == "FSSIM":
        sim_type = SimType.FSSIM
    else:
        sim_type = SimType.GPX

    out_cfg = config["output"]
    plot_track = out_cfg["plot_track"]
    visualise_voronoi = out_cfg["visualise_voronoi"]
    create_output_file = out_cfg["create_output_file"]
    output_location = out_cfg["output_location"]

    off = config["offsets"]
    z_offset = off["z_offset"]
    lat_offset = off["lat_offset"]
    lon_offset = off["lon_offset"]

    def randomize_params():
        n_points = random.randint(40, 100)
        n_regions = random.randint(10, n_points)
        min_bound = random.uniform(0.0, 10.0)
        max_bound = random.uniform(100.0, 200.0)
        mode = random.choice([Mode.EXPAND, Mode.EXTEND, Mode.RANDOM])
        return n_points, n_regions, min_bound, max_bound, mode

    if mode_choice == "custom":
        params = config["track_params"]
        n_points = params["n_points"]
        n_regions = params["n_regions"]
        min_bound = params["min_bound"]
        max_bound = params["max_bound"]
        mode = Mode.RANDOM
    else:
        n_points, n_regions, min_bound, max_bound, mode = randomize_params()

    max_attempts = 10
    attempt = 0
    success = False
    cones = None
    cone_actors = []

    while attempt < max_attempts and not success:
        try:
            print(f"\nAttempt {attempt + 1}: Generating track with parameters:")
            print(
                f"   n_points={n_points}, n_regions={n_regions}, bounds=({min_bound}, {max_bound}), mode={mode.name}"
            )

            track_gen = TrackGenerator(
                n_points=n_points,
                n_regions=n_regions,
                min_bound=min_bound,
                max_bound=max_bound,
                mode=mode,
                open_loop=open_loop,
                missing_cone_ratio=missing_cone_ratio,
                random_open_loop=random_open_loop,
                behind_ratio=behind_ratio,
                ahead_ratio=ahead_ratio,
                plot_track=plot_track,
                visualise_voronoi=visualise_voronoi,
                create_output_file=create_output_file,
                output_location=output_location,
                z_offset=z_offset,
                lat_offset=lat_offset,
                lon_offset=lon_offset,
                sim_type=sim_type,
            )

            cones = track_gen.create_track()
            success = True
            print(" Track successfully created!")

            if spawning_enabled:
                if fit_to_map:
                    client = carla.Client("localhost", 2000)
                    world = client.get_world()
                    scale = fit_scale_to_map(cones, start_x, start_y, scale, world)

                cone_actors = spawn_track(
                    cones, start_x, start_y, scale, bp_left_id, bp_right_id
                )
                print("Track spawned!")
            else:
                print("Track spawning disabled in config (track_spawner.enabled = false).")

        except Exception as e:
            print(f"\nUnable to create track with the parameters above.")
            print(f"   Reason: {e}")
            print("   Randomizing new parameters and retrying...\n")
            n_points, n_regions, min_bound, max_bound, mode = randomize_params()
            attempt += 1
            time.sleep(0.5)

    if not success:
        print("\nFailed to generate a valid track after multiple attempts.")
        raise RuntimeError("Track generation failed")

    return cones, start_x, start_y, scale, cone_actors


if __name__ == "__main__":
    generate_and_spawn_track()