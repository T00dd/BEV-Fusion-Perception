import os
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import cv2 as cv
import carla
import yaml

from sensor_capture import SyncSensorRig
from gt_extraction import extract_frame_annotations
from coordinate_frames import FRAME_CONVENTION

from track_spawner import generate_and_spawn_track
from centerline_pipeline import compute_centerline_carla
from pursuit_controller import PurePursuitController


COMPLETE_MARKER = "_COMPLETE"


def apply_condition(world, condition):
    # Apply weather and other environmental conditions to the CARLA world
    w = world.get_weather()
    for k, v in condition.get("weather", {}).items():
        if hasattr(w, k):
            setattr(w, k, float(v))
    world.set_weather(w)


def sensors_cfg_for_condition(cfg, condition):
    # Return a sensor config dict for this scene, with LiDAR noise parameters folded in from the condition.
    import copy
    scfg = copy.deepcopy(cfg["sensors"])
    ln = condition.get("lidar_noise", None)
    if ln is not None:
        if isinstance(ln, dict):
            scfg["lidar"]["noise_stddev"] = float(ln.get("noise_stddev", 0.0))
            if "dropoff_general_rate" in ln:
                scfg["lidar"]["dropoff_general_rate"] = float(ln["dropoff_general_rate"])
        else:
            # scalar shorthand: just the std-dev in metres
            scfg["lidar"]["noise_stddev"] = float(ln)
    return scfg


def scene_is_complete(scene_dir):
    # Check if the scene has been fully recorded by looking for the _COMPLETE marker file
    marker = Path(scene_dir) / COMPLETE_MARKER
    if not marker.exists():
        return False
    try:
        info = json.loads(marker.read_text())
        return info.get("frames_written", 0) > 0
    except Exception:
        return False


def _write_calib(scene_dir, rig, condition):
    calib = rig.calib_dict()
    calib["frame_convention"] = FRAME_CONVENTION
    with open(Path(scene_dir) / "calib.yaml", "w") as f:
        yaml.safe_dump(calib, f, sort_keys=False)
    with open(Path(scene_dir) / "condition.yaml", "w") as f:
        yaml.safe_dump(condition, f, sort_keys=False)


def record_scene(client, world, scene_id, scene_dir, condition, cfg, logger):
    
    scene_dir = Path(scene_dir)
    (scene_dir / "lidar").mkdir(parents=True, exist_ok=True)
    (scene_dir / "images").mkdir(parents=True, exist_ok=True)
    (scene_dir / "labels").mkdir(parents=True, exist_ok=True)

    logger.info(f"[{scene_id}] applying condition '{condition['name']}'")
    apply_condition(world, condition)

    #  Track + cones (cleaning previous cones incorporated)
    logger.info(f"[{scene_id}] generating + spawning track")
    cones, start_x, start_y, scale, cone_actors = generate_and_spawn_track()

    # Centerline -> waypoints
    repo_root = Path(__file__).resolve().parents[2]
    waypoints = compute_centerline_carla(
        cones, start_x, start_y,
        carla_scale=scale,
        data_dir=repo_root / "data",
        reconstructor_bin=repo_root / "build/track_to_centerline",
    )

    # Spawn ego at first waypoint, facing the path.
    bp = world.get_blueprint_library().find(cfg["ego"]["blueprint"])
    first = waypoints[0]
    ahead = waypoints[min(5, len(waypoints) - 1)]
    yaw = math.degrees(math.atan2(ahead[1] - first[1], ahead[0] - first[0]))
    tf = carla.Transform(
        carla.Location(x=float(first[0]), y=float(first[1]), z=float(first[2]) + 1.0),
        carla.Rotation(yaw=yaw))
    vehicle = world.spawn_actor(bp, tf)

    controller = PurePursuitController(
        waypoints,
        target_speed=cfg["ego"]["target_speed"],
        lookahead=cfg["ego"]["lookahead"])

    # Sensor config with this scene's LiDAR noise folded in (orthogonal to weather).
    scene_sensors_cfg = sensors_cfg_for_condition(cfg, condition)
    rig = SyncSensorRig(world, vehicle, scene_sensors_cfg)

    # Calibration is per-scene, written up front.
    _write_calib(scene_dir, rig, condition)

    ego_pose_rows = []
    frames_written = 0
    max_frames = cfg["capture"]["max_frames_per_scene"]
    warmup = cfg["capture"].get("warmup_ticks", 10)
    every = cfg["capture"].get("capture_every_n_ticks", 1)

    try:
        # Let physics/sensors settle before recording.
        for _ in range(warmup):
            vehicle.apply_control(controller.step(vehicle))
            world.tick()

        tick = 0
        while frames_written < max_frames:
            vehicle.apply_control(controller.step(vehicle))
            world.tick()
            tick += 1

            if controller.reached_end:
                logger.info(f"[{scene_id}] car reached end of track")
                break
            if tick % every != 0:
                continue

            frame_id = world.get_snapshot().frame
            data = rig.grab(frame_id)

            idx = frames_written
            stem = f"frame_{idx:06d}"

            # LiDAR .bin (KITTI/nuScenes/OpenPCDet-compatible)
            data["lidar"]["points_rh"].tofile(scene_dir / "lidar" / f"{stem}.bin")

            # Camera images
            for cam_name, cam in data["cameras"].items():
                cv.imwrite(str(scene_dir / "images" / f"{stem}_cam_{cam_name}.png"),
                           cam["image"])

            # Annotations in right-handed lidar frame. Pass the point cloud so
            # each cone box gets a num_lidar_points count, and the cone cfg so
            # box dimensions / z-convention come from one config place.
            ann = extract_frame_annotations(
                world,
                data["lidar_world_transform"],
                cfg["grid"]["extent"],
                cone_cfg=cfg.get("cones", {}),
                lidar_points_rh=data["lidar"]["points_rh"],
            )
            with open(scene_dir / "labels" / f"{stem}.json", "w") as f:
                json.dump({"frame": idx, "cones": ann}, f)

            # Ego world pose for optional temporal use later
            etf = vehicle.get_transform()
            ego_pose_rows.append([
                idx, etf.location.x, etf.location.y, etf.location.z,
                etf.rotation.roll, etf.rotation.pitch, etf.rotation.yaw])

            frames_written += 1
            if frames_written % 50 == 0:
                logger.info(f"[{scene_id}] {frames_written}/{max_frames} frames")

        # ego_poses.csv
        with open(scene_dir / "ego_poses.csv", "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["frame", "x", "y", "z", "roll", "pitch", "yaw"])
            wtr.writerows(ego_pose_rows)

    finally:
        rig.destroy()
        try:
            vehicle.destroy()
        except Exception:
            pass
        if cone_actors:
            client.apply_batch_sync(
                [carla.command.DestroyActor(c.id) for c in cone_actors], True)

    if frames_written == 0:
        raise RuntimeError("no frames written")

    # Completion marker to enable resume/rebuild of splits without re-recording
    marker = {
        "scene_id": scene_id,
        "condition": condition["name"],
        "frames_written": frames_written,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (scene_dir / COMPLETE_MARKER).write_text(json.dumps(marker, indent=2))
    logger.info(f"[{scene_id}] DONE: {frames_written} frames")
    return frames_written