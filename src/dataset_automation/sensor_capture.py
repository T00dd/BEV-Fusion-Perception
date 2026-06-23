import queue
import numpy as np
import carla

from coordinate_frames import lidar_points_to_rh


class SyncSensorRig:

    # initializes a sensor rig with a LiDAR and multiple cameras, all in synchronous mode
    def __init__(self, world, vehicle, cfg):
        self.world = world
        self.vehicle = vehicle
        self.cfg = cfg
        self.sensors = {}          # name -> actor
        self.queues = {}           # name -> queue.Queue
        self.cam_intrinsics = {}   # name -> 3x3 K
        self.cam_extrinsics = {}   # name -> 4x4 (cam<-ego) at spawn
        self.cam_exposure = {}     # name -> applied photometric attrs
        self.lidar_mount = None    # carla.Transform relative to ego
        self.front_crop_halfdeg = None  # set by _spawn_lidar from config

        bp_lib = world.get_blueprint_library()
        self._spawn_lidar(bp_lib)
        for cam in cfg["cameras"]:
            self._spawn_camera(bp_lib, cam)

    # spawn sensors in synchronous mode, with queues to hold the latest frame for each sensor
    def _spawn_lidar(self, bp_lib):
        lc = self.cfg["lidar"]
        bp = bp_lib.find("sensor.lidar.ray_cast")
        bp.set_attribute("channels", str(lc["channels"]))
        bp.set_attribute("range", str(lc["range"]))
        bp.set_attribute("points_per_second", str(lc["points_per_second"]))
        bp.set_attribute("upper_fov", str(lc["upper_fov"]))
        bp.set_attribute("lower_fov", str(lc["lower_fov"]))
        bp.set_attribute("rotation_frequency", str(lc["rotation_frequency"]))

        noise = float(lc.get("noise_stddev", 0.0))
        if noise > 0.0:
            bp.set_attribute("noise_stddev", str(noise))

        dropoff = float(lc.get("dropoff_general_rate", 0.0))
        if noise <= 0.0 and dropoff <= 0.0:
            bp.set_attribute("dropoff_general_rate", "0.0")
            bp.set_attribute("dropoff_intensity_limit", "1.0")
            bp.set_attribute("dropoff_zero_intensity", "0.0")
        else:
            bp.set_attribute("dropoff_general_rate", str(dropoff))

        fc = lc.get("front_crop", None)
        if fc and fc.get("enabled", False):
            self.front_crop_halfdeg = float(fc.get("half_angle_deg", 60.0))
        else:
            self.front_crop_halfdeg = None

        mount = carla.Transform(carla.Location(
            x=lc["mount"]["x"], y=lc["mount"]["y"], z=lc["mount"]["z"]))
        self.lidar_mount = mount
        sensor = self.world.spawn_actor(bp, mount, attach_to=self.vehicle,
                                        attachment_type=carla.AttachmentType.Rigid)
        q = queue.Queue()
        sensor.listen(q.put)
        self.sensors["lidar"] = sensor
        self.queues["lidar"] = q

    def _spawn_camera(self, bp_lib, cam):
        bp = bp_lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(cam["width"]))
        bp.set_attribute("image_size_y", str(cam["height"]))
        bp.set_attribute("fov", str(cam["fov"]))

        bp.set_attribute("motion_blur_intensity", "0.0")
        bp.set_attribute("motion_blur_max_distortion", "0.0")
        bp.set_attribute("blur_amount", "0.0")

        exp = cam.get("exposure", {}) or {}
        applied = {}
        _photo_map = [
            ("mode", "exposure_mode"),
            ("compensation", "exposure_compensation"),
            ("shutter_speed", "shutter_speed"),
            ("iso", "iso"),
            ("fstop", "fstop"),
            ("gamma", "gamma"),
            # Depth-of-field / lens controls:
            ("focal_distance", "focal_distance"),
            ("min_fstop", "min_fstop"),
            ("blade_count", "blade_count"),
            ("enable_postprocess_effects", "enable_postprocess_effects"),
        ]
        for cfg_key, attr in _photo_map:
            if cfg_key in exp and bp.has_attribute(attr):
                bp.set_attribute(attr, str(exp[cfg_key]))
                applied[attr] = exp[cfg_key]
        self.cam_exposure[cam["name"]] = applied

        mount = carla.Transform(
            carla.Location(x=cam["mount"]["x"], y=cam["mount"]["y"], z=cam["mount"]["z"]),
            carla.Rotation(yaw=cam["mount"].get("yaw", 0.0),
                           pitch=cam["mount"].get("pitch", 0.0),
                           roll=cam["mount"].get("roll", 0.0)),
        )
        sensor = self.world.spawn_actor(bp, mount, attach_to=self.vehicle,
                                        attachment_type=carla.AttachmentType.Rigid)
        q = queue.Queue()
        sensor.listen(q.put)
        name = cam["name"]
        self.sensors[name] = sensor
        self.queues[name] = q

        # Intrinsics K from width/height/fov (pinhole model).
        w, h, fov = cam["width"], cam["height"], cam["fov"]
        f = w / (2.0 * np.tan(np.radians(fov) / 2.0))
        K = np.array([[f, 0, w / 2.0],
                      [0, f, h / 2.0],
                      [0, 0, 1.0]], dtype=np.float64)
        self.cam_intrinsics[name] = K
        # Extrinsic: camera mount relative to ego (left-handed CARLA frame)
        self.cam_extrinsics[name] = np.array(mount.get_matrix(), dtype=np.float64)

    def _drain_to_frame(self, q, frame_id, timeout=2.0):
        # Drain the queue until we find the frame_id we want, or a later frame
        # Return the item for that frame
        while True:
            item = q.get(timeout=timeout)
            if item.frame == frame_id:
                return item
            if item.frame > frame_id:
                # If we missed the frame we wanted, return the next one
                return item

    def grab(self, frame_id):
        out = {"cameras": {}}

        # LiDAR
        raw = self._drain_to_frame(self.queues["lidar"], frame_id)
        pts = np.frombuffer(raw.raw_data, dtype=np.float32).reshape(-1, 4).copy()
        xyz = lidar_points_to_rh(pts[:, :3])
        points_rh = np.concatenate([xyz, pts[:, 3:4]], axis=1).astype(np.float32)

        # Optional front-sector crop (emulate forward-facing solid-state).
        # In the right-handed LiDAR frame x is forward, y is left, so azimuth
        # measured from the +x axis is atan2(y, x). Keep |azimuth| <= half.
        if self.front_crop_halfdeg is not None:
            az = np.degrees(np.arctan2(points_rh[:, 1], points_rh[:, 0]))
            keep = np.abs(az) <= self.front_crop_halfdeg
            points_rh = points_rh[keep]

        out["lidar"] = {"points_rh": points_rh}
        out["lidar_world_transform"] = self.sensors["lidar"].get_transform()

        # Cameras
        for cam in self.cfg["cameras"]:
            name = cam["name"]
            img = self._drain_to_frame(self.queues[name], frame_id)
            arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(
                img.height, img.width, 4)
            out["cameras"][name] = {"image": arr[:, :, :3].copy()}  # BGRA->BGR
        return out

    def calib_dict(self):
        lm = self.lidar_mount
        calib = {
            "lidar_mount_xyz": [lm.location.x, lm.location.y, lm.location.z],
            "cameras": {},
        }
        for cam in self.cfg["cameras"]:
            name = cam["name"]
            calib["cameras"][name] = {
                "intrinsic_K": self.cam_intrinsics[name].tolist(),
                "extrinsic_cam_from_ego_carla": self.cam_extrinsics[name].tolist(),
                "width": cam["width"],
                "height": cam["height"],
                "fov": cam["fov"],
                "exposure": self.cam_exposure.get(name, {}),
            }
        return calib

    def destroy(self):
        for s in self.sensors.values():
            try:
                s.stop()
            except Exception:
                pass
            try:
                s.destroy()
            except Exception:
                pass
        self.sensors.clear()
        self.queues.clear()