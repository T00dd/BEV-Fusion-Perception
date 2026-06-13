import numpy as np


# tag stored in calib.yaml so consumers know what frame the data is in
FRAME_CONVENTION = {
    "handedness": "right-handed",
    "axes": {"x": "forward", "y": "left", "z": "up"},
    "origin": "lidar_sensor",
    "note": "Converted from CARLA left-handed (y-right) by negating y.",
}


def carla_transform_to_matrix(transform):
    # return the map from actor frame to world frame
    return np.array(transform.get_matrix(), dtype=np.float64)


def carla_transform_to_inverse_matrix(transform):
    # return the map from world frame to actor frame
    return np.array(transform.get_inverse_matrix(), dtype=np.float64)


def world_to_sensor_matrix(sensor_transform):
    # return the map from world frame to sensor frame
    return carla_transform_to_inverse_matrix(sensor_transform)


def apply_matrix(matrix_4x4, points_xyz):
    
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]
    n = pts.shape[0]
    homo = np.concatenate([pts, np.ones((n, 1))], axis=1)  # (N,4)
    out = (matrix_4x4 @ homo.T).T  # (N,4)
    return out[:, :3]


def carla_to_right_handed(points_xyz):
    # convert CARLA left-handed coords to right-handed
    pts = np.array(points_xyz, dtype=np.float64, copy=True)
    if pts.ndim == 1:
        pts[1] = -pts[1]
    else:
        pts[:, 1] = -pts[:, 1]
    return pts


def world_cone_to_lidar_rh(cone_world_xyz, lidar_world_transform):
    # essentially a chain of transforms to convert from world to right-handed LiDAR frame
    M = world_to_sensor_matrix(lidar_world_transform)
    in_lidar = apply_matrix(M, cone_world_xyz)
    return carla_to_right_handed(in_lidar)


def lidar_points_to_rh(points_xyz):
    # convert raw CARLA LiDAR points (already in sensor frame) to right-handed
    # CARLA LiDAR returns points in its own sensor frame, left-handed. We only
    # need the handedness flip here, no transform.
    
    return carla_to_right_handed(points_xyz)