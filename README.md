# Dataset Automation (CARLA to BEV cone dataset)

Generates a synthetic Formula-Student cone dataset from CARLA: synchronized
LiDAR + camera frames with ground-truth cone positions in the ego/LiDAR frame.

## Where this lives

```
src/dataset_automation/
  dataset_builder.py      # entry point (orchestrator: manifest, resume, logs)
  scene_recorder.py       # records one scene end-to-end
  sensor_capture.py       # synchronous lidar+camera capture (dataset-grade)
  gt_extraction.py        # cone GT -> ego/lidar frame, color, instance id
  coordinate_frames.py    # SINGLE source of truth for frame conventions
  dataset_config.yaml     # the only file you edit
```

Reuses the existing modules unchanged: `track_spawner.py` (the updated one),
`centerline_pipeline.py`, `pursuit_controller.py`, plus the two submodules under
`lib/` and the compiled `build/track_to_centerline`.

## Coordinate convention (read coordinate_frames.py)

CARLA is left-handed (x fwd, **y right**, z up). The dataset is **right-handed**
(x fwd, **y left**, z up) to match KITTI/nuScenes/OpenPCDet. The only change is
`y -> -y`, applied once at save time to both point clouds and cone positions.
The convention is also stamped into every scene's `calib.yaml`.

## Run

Test end-to-end first (config ships with `scenes.count: 1`,
`max_frames_per_scene: 100`):

```bash
cd src/dataset_automation
python dataset_builder.py --config dataset_config.yaml
```

Then scale up by editing `dataset_config.yaml` (`scenes.count: 100`,
`max_frames_per_scene: 500`, add cameras/conditions) and re-running.

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
    condition.yaml              # weather/time used (traceability)
    ego_poses.csv               # per-frame ego world pose
    lidar/frame_000000.bin      # float32 (N,4) x,y,z,intensity, RH lidar frame
    images/frame_000000_cam_front.png
    labels/frame_000000.json    # cones: instance_id, class, position (RH lidar)
    _COMPLETE                   # written only after the scene fully succeeds
  splits/{train,val,test}.txt   # split BY SCENE (never by frame)
```

## Notes

- Splits are per-scene: a whole CARLA scenario lands in exactly one split, so
  near-identical consecutive frames can't leak between train and val.
- The BEV heatmap is **not** saved; generate it in the PyTorch `Dataset` so you
  can tune sigma/resolution without re-rasterizing everything.
- Conditions cycle across scenes. Add weather entries in the config to diversify
  (the schedule calls for varied lighting/weather/curvature/speed).