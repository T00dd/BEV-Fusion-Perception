"""
gt_extraction.py
================
Builds per-frame ground-truth cone annotations.

Pipeline per frame:
  1. Read every cone actor's WORLD position from CARLA.
  2. Classify color from the blueprint id (blue / yellow / orange_small / orange_big).
  3. Transform each cone into the right-handed LiDAR frame (via coordinate_frames).
  4. Crop to the configured grid extent (cones outside the BEV window are dropped,
     since the network can't be supervised on what it can't see).
  5. Attach a STABLE instance id per cone (the CARLA actor id), so the same cone
     keeps the same id across all frames of a scene -> enables tracking later.

The output is a plain list of dicts, ready to dump as JSON.
"""

import numpy as np
from coordinate_frames import world_cone_to_lidar_rh


# Map a CARLA cone blueprint id substring -> dataset class label
CONE_CLASS_BY_KEYWORD = [
    ("blue", "blue"),
    ("yellow", "yellow"),
    ("orange", "orange_small"),   # refined below if 'big'/'large' present
]


def classify_cone(type_id):
    # returns None if not a cone, else one of the class labels above
    t = type_id.lower()
    if "cone" not in t:
        return None
    if "blue" in t:
        return "blue"
    if "yellow" in t:
        return "yellow"
    if "orange" in t:
        # distinguish big/large vs small if specified in the blueprint name
        if "big" in t or "large" in t:
            return "orange_big"
        return "orange_small"
    return "unknown"


def list_cone_actors(world):
    # returns a list of (actor_id, class_label, carla.Location) for all cone actors in the world
    out = []
    for a in world.get_actors():
        label = classify_cone(a.type_id)
        if label is None:
            continue
        loc = a.get_location()
        out.append((a.id, label, loc))
    return out


def _in_extent(p_rh, extent):
    x, y, z = p_rh
    return (extent["x_min"] <= x <= extent["x_max"] and
            extent["y_min"] <= y <= extent["y_max"] and
            extent["z_min"] <= z <= extent["z_max"])


def extract_frame_annotations(world, lidar_world_transform, extent):
    
    # returns a list of dicts, one per cone in the frame, with fields:
    #   instance_id: int (CARLA actor id, stable across frames)
    #   class: str (one of the class labels above)
    #   position: [x, y, z] in right-handed LiDAR frame

    cones = list_cone_actors(world)
    annotations = []
    for actor_id, label, loc in cones:
        world_xyz = np.array([loc.x, loc.y, loc.z], dtype=np.float64)
        p_rh = world_cone_to_lidar_rh(world_xyz, lidar_world_transform)
        p_rh = np.asarray(p_rh).reshape(3)
        if not _in_extent(p_rh, extent):
            continue
        annotations.append({
            "instance_id": int(actor_id),
            "class": label,
            "position": [float(p_rh[0]), float(p_rh[1]), float(p_rh[2])],
        })
    return annotations