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
from gt_extraction import extract_frame_annotations, list_cone_actors
from coordinate_frames import FRAME_CONVENTION
from projection_2d import project_cones_for_camera

from track_spawner import generate_and_spawn_track
from centerline_pipeline import compute_centerline_carla
from pursuit_controller import PurePursuitController


COMPLETE_MARKER = "_COMPLETE"

GROUND_Z_BY_ZONE = {1: 237.0, 2: 237.0}


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


def _write_calib(scene_dir, rig, condition, scene_meta=None):
    calib = rig.calib_dict()
    calib["frame_convention"] = FRAME_CONVENTION
    with open(Path(scene_dir) / "calib.yaml", "w") as f:
        yaml.safe_dump(calib, f, sort_keys=False)
    # condition.yaml gets the weather/noise AND the per-scene zone/geometry, so
    # every scene on disk is self-describing (traceability for ablations).
    out = dict(condition)
    if scene_meta:
        out["scene_info"] = {
            "zone": scene_meta.get("zone"),
            "track_seed": scene_meta.get("track_seed"),
            "lobes_min": scene_meta.get("lobes_min"),
            "lobes_max": scene_meta.get("lobes_max"),
        }
    with open(Path(scene_dir) / "condition.yaml", "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)


def _dump_labels_readable(scene_path, frame_idx, cones):
    
    lines = ["{", f'  "frame": {frame_idx},', '  "cones": [']
    for i, c in enumerate(cones):
        comma = "," if i < len(cones) - 1 else ""
        lines.append("    " + json.dumps(c, separators=(", ", ": ")) + comma)
    lines.append("  ]")
    lines.append("}")
    Path(scene_path).write_text("\n".join(lines))


def record_scene(client, world, scene_id, scene_dir, condition, cfg, logger,
                 scene_meta=None):
    scene_meta = scene_meta or {}

    scene_dir = Path(scene_dir)
    (scene_dir / "lidar").mkdir(parents=True, exist_ok=True)
    (scene_dir / "images").mkdir(parents=True, exist_ok=True)
    (scene_dir / "labels").mkdir(parents=True, exist_ok=True)
    #new outputs (added for the camera branch): GT depth + projected 2D cones
    (scene_dir / "depth").mkdir(parents=True, exist_ok=True)
    (scene_dir / "labels_2d").mkdir(parents=True, exist_ok=True)

    logger.info(f"[{scene_id}] applying condition '{condition['name']}'")
    apply_condition(world, condition)

    zone = scene_meta.get("zone", cfg.get("track_spawner", {}).get("zone", 1))
    ground_z = GROUND_Z_BY_ZONE.get(zone, 237.0)

    #  Track + cones (cleaning previous cones incorporated)
    logger.info(f"[{scene_id}] generating + spawning track (zone={zone})")
    lobes_range = None
    if "lobes_min" in scene_meta and "lobes_max" in scene_meta:
        lobes_range = (scene_meta["lobes_min"], scene_meta["lobes_max"])
    cones, start_x, start_y, scale, cone_actors = generate_and_spawn_track(
        seed=scene_meta.get("track_seed"),
        zone=zone,
        lobes_range=lobes_range,
        draw_debug_arena=False,   # debug outline would pollute camera frames
    )

    repo_root = Path(__file__).resolve().parents[2]
    waypoints = compute_centerline_carla(
        cones,
        0.0,                       # start_x neutralised
        0.0,                       # start_y neutralised
        carla_scale=1.0,           # no scaling
        data_dir=repo_root / "data",
        reconstructor_bin=repo_root / "build/track_to_centerline",
        z=ground_z,                # centerline at the zone's ground height
    )

    # Spawn ego at first waypoint, facing the path, at ground height.
    bp = world.get_blueprint_library().find(cfg["ego"]["blueprint"])
    first = waypoints[0]
    ahead = waypoints[min(5, len(waypoints) - 1)]
    yaw = math.degrees(math.atan2(ahead[1] - first[1], ahead[0] - first[0]))
    tf = carla.Transform(
        carla.Location(x=float(first[0]), y=float(first[1]), z=ground_z + 0.5),
        carla.Rotation(yaw=yaw))
    vehicle = world.spawn_actor(bp, tf)
    logger.info(f"[{scene_id}] ego spawn z={tf.location.z:.1f}, "
                f"first wp=({first[0]:.1f},{first[1]:.1f},{first[2]:.1f}), "
                f"ground_z={ground_z:.1f}")

    controller = PurePursuitController(
        waypoints,
        target_speed=cfg["ego"]["target_speed"],
        lookahead=cfg["ego"]["lookahead"])

    # Sensor config with this scene's LiDAR noise folded in (orthogonal to weather).
    scene_sensors_cfg = sensors_cfg_for_condition(cfg, condition)
    rig = SyncSensorRig(world, vehicle, scene_sensors_cfg)

    # Calibration is per-scene, written up front.
    _write_calib(scene_dir, rig, condition, scene_meta)

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
            _dump_labels_readable(
                scene_dir / "labels" / f"{stem}.json", idx, ann)

            #gt depth (carla) aligned with the left RGB camera
            if "depth" in data:
                np.save(scene_dir / "depth" / f"{stem}.npy",
                        data["depth"]["depth_m"])

            #gt 2d: project cones into each camera image.
            # IMPORTANT: the 2D GT must reflect what the CAMERA sees, NOT what
            # the LiDAR grid keeps. A cone can be visible in the image while
            # lying outside the LiDAR BEV extent (e.g. beyond x_max, or wide to
            # the side). So we project ALL world cones and let project_cones_for_
            # camera keep only those that land in front of and inside the image.
            cones_world = [
                {"instance_id": int(aid), "class": label,
                "world_xyz": [loc.x, loc.y, loc.z]}
                for (aid, label, loc) in list_cone_actors(world)
            ]
            for cam in cfg["sensors"]["cameras"]:
                cam_name = cam["name"]
                cam_actor = rig.sensors[cam_name]
                cones_2d = project_cones_for_camera(
                    cones_world,
                    cam_actor.get_transform(),
                    np.array(rig.cam_intrinsics[cam_name], dtype=np.float64),
                    cam["width"],
                    cam["height"],
                )
                with open(scene_dir / "labels_2d"
                          / f"{stem}_cam_{cam_name}.json", "w") as f:
                    json.dump({"frame": idx, "camera": cam_name,
                               "cones_in_image": cones_2d}, f, indent=2)

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
        "zone": zone,
        "track_seed": scene_meta.get("track_seed"),
        "lobes_min": scene_meta.get("lobes_min"),
        "lobes_max": scene_meta.get("lobes_max"),
        "frames_written": frames_written,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (scene_dir / COMPLETE_MARKER).write_text(json.dumps(marker, indent=2))
    logger.info(f"[{scene_id}] DONE: {frames_written} frames")
    return frames_written