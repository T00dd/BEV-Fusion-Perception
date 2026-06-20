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

# Anchor the START at the LOW-Y EXTREME of the arena (not the centre), so the
# open-loop track extends forward across the long axis and fills the rectangle.
DEFAULT_START_X = -12.9
DEFAULT_START_Y = -23.1
DEFAULT_GROUND_Z = 237.0          # real ground height of the elevated arena
DEFAULT_BP_LEFT = "static.prop.bluecone"
DEFAULT_BP_RIGHT = "static.prop.yellowcone"

# Arena polygon: 4 ground vertices (x, y) in walking order around the perimeter.
DEFAULT_ARENA_CORNERS = [
    (-65.3, 111.2),   # top-left
    (-30.8, 119.1),   # top-right
    (5.0,   -24.1),   # bottom-right
    (-30.8, -32.0),   # bottom-left
]

# Reject tracks whose y-extent (the long axis of the rectangle) is below this
# fraction of the arena's y-span -> enforces "use the full length".
DEFAULT_MIN_Y_FRACTION = 0.5


def load_spawner_config():
    cfg = {
        "enabled": True,
        "ground_z": DEFAULT_GROUND_Z,
        "cone_blueprint_left": DEFAULT_BP_LEFT,
        "cone_blueprint_right": DEFAULT_BP_RIGHT,
        "arena_corners": DEFAULT_ARENA_CORNERS,
        # --- track generation parameters (now read from YAML) ---
        "track_width": 3.5,          # lane width in metres (3-4)
        "missing_cone_ratio": 0.0,   # fraction of cones randomly dropped
        "edge_margin": 2.5,          # min distance of cones from polygon edge
        "length_fill": 0.9,          # fraction of the long axis the track spans
        "amp_fill_min": 0.6,         # min width usage of the short axis
        "amp_fill_max": 0.9,         # max width usage of the short axis
        "lobes_min": 3,              # min number of S-bends
        "lobes_max": 6,              # max number of S-bends
    }
    try:
        with open(MAIN_CONFIG, "r") as f:
            data = yaml.safe_load(f) or {}
        ts = data.get("track_spawner", {}) or {}
        cfg.update({k: ts[k] for k in cfg.keys() if k in ts})
    except FileNotFoundError:
        print(f"[track_spawner] {MAIN_CONFIG} not found, using defaults.")
    return cfg


EXISTING_CONE_BLUEPRINTS = (
    "static.prop.bluecone", "static.prop.yellowcone",
    "static.prop.orangecone", "static.prop.redcone",
)
CONE_MESH_KEYS = ("BlueCone", "OrangeCone", "YellowCone", "RedCone")


def clean_previous_track(world, client=None):
    cone_actors = []
    for a in world.get_actors():
        tid = a.type_id.lower()
        if tid in EXISTING_CONE_BLUEPRINTS or "cone" in tid:
            cone_actors.append(a)
            continue
        if tid == "static.prop.mesh":
            mp = a.attributes.get("mesh_path", "")
            if any(k in mp for k in CONE_MESH_KEYS):
                cone_actors.append(a)

    if not cone_actors:
        print("[track_spawner] No pre-existing cones to remove.")
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
    print(f"[track_spawner] Removed {len(cone_actors)} pre-existing cone(s).")


def _resolve_blueprint(blueprints, bp_id):
    matches = blueprints.filter(bp_id)
    if len(matches) == 0:
        raise RuntimeError(
            f"Cone blueprint '{bp_id}' not found in this CARLA build. "
            f"Check the name in config.yaml (track_spawner.cone_blueprint_*)."
        )
    return matches[0]


def get_arena_polygon(sp_cfg):
    corners = sp_cfg.get("arena_corners")
    if not corners or len(corners) < 3:
        return None
    from shapely.geometry import Polygon
    return Polygon([(float(x), float(y)) for (x, y) in corners])


def check_track_in_arena(cones, start_x, start_y, arena_poly,
                         margin=2.0, min_y_fraction=0.5):
    """
    PASS/FAIL test for a generated track. NO scaling: cones are placed at
    world = (start_x + cx, start_y + cy), preserving the generator's real-metre
    proportions (3 m lane width, 5 m spacing).

    Raises RuntimeError (caught by the retry loop) if the track is unacceptable:
      1. any cone falls outside the arena polygon (eroded by `margin`), or
      2. the track's y-extent is below `min_y_fraction` of the arena's y-span
         (i.e. it doesn't run far enough along the long axis).
    Returns silently if the track is good.
    """
    if arena_poly is None:
        return

    from shapely.geometry import Point

    safe = arena_poly.buffer(-margin) if margin > 0 else arena_poly
    if safe.is_empty:
        raise RuntimeError("[track_spawner] Arena too small for the margin.")

    if not safe.contains(Point(start_x, start_y)):
        raise RuntimeError(
            f"[track_spawner] start ({start_x:.1f},{start_y:.1f}) outside arena."
        )

    all_cones = list(cones["cones_left"]) + list(cones["cones_right"])
    if not all_cones:
        raise RuntimeError("[track_spawner] Empty track.")

    world_pts = [(start_x + c[0], start_y + c[1]) for c in all_cones]

    # Check 1: containment
    for (px, py) in world_pts:
        if not safe.contains(Point(px, py)):
            raise RuntimeError(
                "[track_spawner] Track exits the arena. Retrying."
            )

    # Check 2: runs far enough along the long (y) axis
    ys = [p[1] for p in world_pts]
    track_y_span = max(ys) - min(ys)
    ay = [c[1] for c in arena_poly.exterior.coords]
    arena_y_span = max(ay) - min(ay)
    if track_y_span < min_y_fraction * arena_y_span:
        raise RuntimeError(
            f"[track_spawner] Track too short along the length "
            f"({track_y_span:.0f} m < {min_y_fraction:.0%} of "
            f"{arena_y_span:.0f} m). Retrying for a longer track."
        )


def draw_arena(world, corners, z=237.0, life_time=60.0):
    if not corners or len(corners) != 4:
        return
    debug = world.debug
    locs = [carla.Location(float(x), float(y), float(z)) for (x, y) in corners]
    red = carla.Color(255, 0, 0)
    for i in range(4):
        debug.draw_line(locs[i], locs[(i + 1) % 4],
                        thickness=0.15, color=red, life_time=life_time)


def spawn_track(cones, bp_left_id, bp_right_id, ground_z):
    """
    Spawn cones. The procedural generator already outputs WORLD coordinates
    inside the polygon, so we place them directly -- no start_x/start_y offset.
    """
    cones_left = cones["cones_left"]
    cones_right = cones["cones_right"]

    client = carla.Client("localhost", 2000)
    world = client.get_world()

    clean_previous_track(world, client)

    blueprints = world.get_blueprint_library()
    model_cones_left = _resolve_blueprint(blueprints, bp_left_id)
    model_cones_right = _resolve_blueprint(blueprints, bp_right_id)

    spawned = []

    def _spawn_side(cone_list, model):
        for cone in cone_list:
            location = carla.Location(
                float(cone[0]),             # already world coords
                float(cone[1]),
                ground_z + 0.1,
            )
            c = world.try_spawn_actor(model, carla.Transform(location))
            if c is None:
                continue
            c.set_simulate_physics(True)
            spawned.append(c)

    _spawn_side(cones_left, model_cones_left)
    _spawn_side(cones_right, model_cones_right)

    print(f"[track_spawner] Spawned {len(spawned)} cones inside the polygon.")
    return spawned


def generate_and_spawn_track(seed=None, missing_cone_ratio=None,
                             track_width=None):
    """
    Procedural serpentine GUARANTEED inside the arena polygon. Parameters are
    read from the YAML (track_spawner section); `seed`, `missing_cone_ratio`
    and `track_width` may be passed to OVERRIDE the YAML per scene (e.g. to
    randomise width across scenes). If left None, the YAML values are used.
    """
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8")

    sp_cfg = load_spawner_config()
    ground_z = sp_cfg["ground_z"]
    spawning_enabled = sp_cfg["enabled"]
    bp_left_id = sp_cfg["cone_blueprint_left"]
    bp_right_id = sp_cfg["cone_blueprint_right"]
    arena_corners = sp_cfg.get("arena_corners")

    # YAML values, with optional per-call override.
    if missing_cone_ratio is None:
        missing_cone_ratio = sp_cfg["missing_cone_ratio"]
    if track_width is None:
        track_width = sp_cfg["track_width"]

    from procedural_track_gen import generate_serpentine_in_polygon

    cones = generate_serpentine_in_polygon(
        arena_corners,
        track_width=track_width,
        cone_spacing=4.0,
        edge_margin=sp_cfg["edge_margin"],
        seed=seed,
        lobes_range=(sp_cfg["lobes_min"], sp_cfg["lobes_max"]),
        amp_fill_range=(sp_cfg["amp_fill_min"], sp_cfg["amp_fill_max"]),
        length_fill=sp_cfg["length_fill"],
        missing_cone_ratio=missing_cone_ratio,
    )
    print(f"[track_spawner] Serpentine in polygon: "
          f"{len(cones['cones_left'])}L/{len(cones['cones_right'])}R cones, "
          f"start_side={cones.get('start_side')}, lobes={cones.get('n_lobes')}, "
          f"width={cones.get('track_width')} m, missing={missing_cone_ratio}")

    cone_actors = []
    if spawning_enabled:
        client = carla.Client("localhost", 2000)
        world = client.get_world()
        draw_arena(world, arena_corners)
        cone_actors = spawn_track(cones, bp_left_id, bp_right_id, ground_z)
        print("Track spawned!")
    else:
        print("Track spawning disabled (track_spawner.enabled = false).")

    cl = cones["cones_left"]
    ref_x = float(cl[0][0]) if len(cl) else 0.0
    ref_y = float(cl[0][1]) if len(cl) else 0.0
    return cones, ref_x, ref_y, 1.0, cone_actors


if __name__ == "__main__":
    generate_and_spawn_track()