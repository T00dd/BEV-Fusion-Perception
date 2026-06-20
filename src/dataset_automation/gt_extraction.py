import numpy as np
from coordinate_frames import world_cone_to_lidar_rh

def classify_cone(type_id):
    # returns None if not a cone, else one of the class labels
    t = type_id.lower()
    if "cone" not in t:
        return None
    if "blue" in t:
        return "blue"
    if "yellow" in t:
        return "yellow"
    if "orange" in t:
        if "big" in t or "large" in t:
            return "orange_big"
        return "orange_small"
    return "unknown"


def list_cone_actors(world):
    # returns a list of (actor_id, class_label, carla.Location) for all cones
    out = []
    for a in world.get_actors():
        label = classify_cone(a.type_id)
        if label is None:
            continue
        loc = a.get_location()
        out.append((a.id, label, loc))
    return out

DEFAULT_CONE_DIMS = {
    "blue":         [0.23, 0.23, 0.32],
    "yellow":       [0.23, 0.23, 0.32],
    "orange_small": [0.23, 0.23, 0.32],
    "orange_big":   [0.28, 0.28, 0.50],
    "unknown":      [0.23, 0.23, 0.32],
}


def _dims_for(label, dims_table):
    return dims_table.get(label, dims_table.get("unknown", [0.23, 0.23, 0.32]))


def _in_extent(xyz, extent):
    x, y, z = xyz
    return (extent["x_min"] <= x <= extent["x_max"] and
            extent["y_min"] <= y <= extent["y_max"] and
            extent["z_min"] <= z <= extent["z_max"])


def _count_points_in_box(points_xyz, center, dims, heading=0.0):

    if points_xyz is None or len(points_xyz) == 0:
        return 0
    p = np.asarray(points_xyz, dtype=np.float64)
    cx, cy, cz = center
    dx, dy, dz = dims
    # translate to box center
    q = p[:, :3] - np.array([cx, cy, cz])
    if abs(heading) > 1e-6:
        c, s = np.cos(-heading), np.sin(-heading)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        q = q @ rot.T
    inside = (
        (np.abs(q[:, 0]) <= dx / 2.0) &
        (np.abs(q[:, 1]) <= dy / 2.0) &
        (np.abs(q[:, 2]) <= dz / 2.0)
    )
    return int(np.count_nonzero(inside))

def extract_frame_annotations(world, lidar_world_transform, extent,
                              cone_cfg=None, lidar_points_rh=None):

    cone_cfg = cone_cfg or {}
    dims_table = cone_cfg.get("dimensions", DEFAULT_CONE_DIMS)
    z_is_base = cone_cfg.get("z_is_base", True)

    cones = list_cone_actors(world)
    annotations = []
    for actor_id, label, loc in cones:
        world_xyz = np.array([loc.x, loc.y, loc.z], dtype=np.float64)
        p_rh = world_cone_to_lidar_rh(world_xyz, lidar_world_transform)
        p_rh = np.asarray(p_rh).reshape(3)

        dims = _dims_for(label, dims_table)
        # Box center: lift by half height if the stored position is the base.
        cz = p_rh[2] + (dims[2] / 2.0 if z_is_base else 0.0)
        center = [float(p_rh[0]), float(p_rh[1]), float(cz)]
        heading = 0.0  # cones are rotationally symmetric

        # Crop using the BOX CENTER against the grid extent.
        if not _in_extent(center, extent):
            continue

        npts = _count_points_in_box(lidar_points_rh, center, dims, heading)
        dist = float(np.linalg.norm(center))

        annotations.append({
            "instance_id": int(actor_id),
            "class": label,
            "position": [float(p_rh[0]), float(p_rh[1]), float(p_rh[2])],
            "box": [center[0], center[1], center[2],
                    float(dims[0]), float(dims[1]), float(dims[2]),
                    float(heading)],
            "num_lidar_points": npts,
            "distance": dist,
        })
    return annotations