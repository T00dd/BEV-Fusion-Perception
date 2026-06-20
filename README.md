# Dataset Automation (CARLA → BEV cone dataset)

Generates a synthetic Formula-Student cone dataset from CARLA: synchronized
LiDAR + stereo-camera frames with ground-truth cone **3D bounding boxes** in the
right-handed LiDAR frame, ready to feed an OpenPCDet detector (SECOND /
CenterPoint).

## Where this lives

```
src/dataset_automation/
  dataset_builder.py      # entry point (orchestrator: manifest, resume, logs)
  scene_recorder.py       # records one scene end-to-end
  sensor_capture.py       # synchronous stereo-cam + LiDAR capture (dataset-grade)
  gt_extraction.py        # cone GT -> RH LiDAR frame, color, 3D box, point count
  coordinate_frames.py    # SINGLE source of truth for frame conventions
  dataset_config.yaml     # the only file you edit
```

Reuses existing modules unchanged: `track_spawner.py`, `centerline_pipeline.py`,
`pursuit_controller.py`, plus the two submodules under `lib/` and the compiled
`build/track_to_centerline`.

## Coordinate convention (read coordinate_frames.py)

CARLA is left-handed (x fwd, **y right**, z up). The dataset is **right-handed**
(x fwd, **y left**, z up) to match KITTI/nuScenes/OpenPCDet. The only change is
`y → -y`, applied once at save time to both point clouds and cone positions.
The convention is also stamped into every scene's `calib.yaml`.

## Ground truth: cones as 3D boxes

The network (SECOND/CenterPoint) is trained on **3D oriented bounding boxes**,
not points — the detection loss assigns LiDAR points to boxes and the IoU
metric compares box volumes, so a box is required even for tiny cones. Each cone
point is therefore promoted to a box `[x, y, z, dx, dy, dz, heading]`:

- `dx, dy, dz` come from per-class dimensions in `dataset_config.yaml`
  (`cones.dimensions`) — cone geometry lives in exactly one place.
- `heading = 0` (cones are rotationally symmetric).
- `z_is_base: true` means the stored cone position is the **base**, so the box
  center is lifted by `dz/2`. If you ever confirm positions are already
  centered, set it `false` — no code change needed. (If generated boxes look
  half-buried, this flag is the culprit.)

Each box also carries `num_lidar_points` (how many points fall inside it) and
`distance`. The point count is what lets you later report recall vs cone point
density, and to drop cones the LiDAR can't actually see.

## Sensors

**LiDAR** is configured to emulate a forward-facing solid-state rather than a
sparse 360° spinner: higher density (`channels: 128`, `points_per_second:
1.5M`), a vertical FOV narrowed to the band where cones live, and an azimuthal
**front-sector crop** (`front_crop.half_angle_deg`) applied in post so only the
front ±N° of points are kept. This is an *approximation* of a real solid-state's
density pattern via `ray_cast` parameters, not a faithful MEMS scan — state it
as such in the thesis.
> If your CARLA build rejects 128 channels, drop to 64 with the same
> `points_per_second`; density stays good.

**Cameras** are a stereo pair (`left` + `right`); baseline is the difference in
mount `y`. The capture rig handles any number of cameras — add/remove entries in
`sensors.cameras`. Intrinsics and per-camera extrinsics are written to
`calib.yaml` for the camera branch.

## Measurement noise (orthogonal to weather)

LiDAR range jitter is a **per-condition** knob, kept separate from weather so
analysis can tell "weather hurt the camera" apart from "range jitter hurt the
LiDAR". Set it in each condition via `lidar_noise`:

```yaml
lidar_noise: 0.01                                   # std-dev in metres, or
lidar_noise: {noise_stddev: 0.015, dropoff_general_rate: 0.10}
```

Keep it realistic (≈1–2 cm for a good sensor). With few points per cone, large
noise makes cones nearly unusable and matches no real hardware.

## Run

Test end-to-end first (config ships small: `scenes.count: 1`,
`max_frames_per_scene: 25`):

```bash
cd src/dataset_automation
python dataset_builder.py --config dataset_config.yaml
```

Then scale up by editing `dataset_config.yaml` (`scenes.count`,
`max_frames_per_scene`, add conditions) and re-running.

## Stop / resume

Ctrl-C any time. Each scene writes a `_COMPLETE` marker only on full success;
re-running skips completed scenes and continues. To check status without
generating:

```bash
python dataset_builder.py --config dataset_config.yaml --verify
```

This also (re)builds `splits/{train,val,test}.txt` from completed scenes.

## Output layout

```
dataset/
  manifest.json                 # the frozen plan (scene -> condition -> split)
  logs/build_YYYYMMDD_HHMMSS.log
  scenes/scene_0000/
    calib.yaml                  # intrinsics, extrinsics, frame convention
    condition.yaml              # weather + lidar_noise used (traceability)
    ego_poses.csv               # per-frame ego world pose
    lidar/frame_000000.bin      # float32 (N,4) x,y,z,intensity, RH lidar frame
    images/frame_000000_cam_left.png
    images/frame_000000_cam_right.png
    labels/frame_000000.json    # see below
    _COMPLETE                   # written only after the scene fully succeeds
  splits/{train,val,test}.txt   # split BY SCENE (never by frame)
```

Each `labels/frame_NNNNNN.json`:

```json
{
  "frame": 0,
  "cones": [
    {
      "instance_id": 312,
      "class": "blue",
      "position": [3.20, 1.50, 0.00],
      "box": [3.20, 1.50, 0.16, 0.23, 0.23, 0.32, 0.0],
      "num_lidar_points": 12,
      "distance": 3.54
    }
  ]
}
```

`position` is the cone base (kept for debug); `box` is the center-based 3D box
consumed by OpenPCDet.

## Notes

- Splits are per-scene: a whole CARLA scenario lands in exactly one split, so
  near-identical consecutive frames can't leak between train and val.
- The BEV heatmap is **not** saved. With a CenterPoint head, OpenPCDet renders
  the target heatmap from the boxes at train time (its own sigma/resolution), so
  pre-computing it on disk would only lock those choices in.
- Conditions cycle across scenes; each carries its own weather **and**
  `lidar_noise`. Add entries to diversify lighting/weather/noise.

## Next step (not yet in this folder)

The OpenPCDet bridge lives outside the cloned framework, in
`src/dataset_automation/conedataset/`: a `ConeDataset(DatasetTemplate)` that
reads this format and yields `points` + `gt_boxes`, plus the dataset/model YAMLs
for SECOND and CenterPoint, and a `create_gt_database` step for GT-paste
augmentation. Keeping it here (not under `lib/OpenPCDet/`) lets OpenPCDet be
updated from upstream without conflicts.