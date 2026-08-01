<<<<<<< HEAD
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
=======
# BEV-Fusion-Perception
## 1. LiDAR branch
Before diving deep in this branch content we introduce some key findings and theory to better understand the modalities offered:
### 1.1 Class Agnosticity
The dataset annotates three chromatic classes (`blue`, `yellow`, `orange_small`). An
initial multi-class experiment showed the network localizes cones correctly but
classifies colour at chance (blue/yellow confusion ≈ 50/50; class-agnostic AP ≈ 0.82
vs per-class mAP ≈ 0.12). The cause of this poor performance was discovered lying in the data, since the CARLA ray-cast LiDAR `intensity` is a purely geometric attenuation term and carries no material/colour information. By plotting the intensity distributions of the three classes, indeed, it can be noticed that the classes are statistically identical.
Consequently the LiDAR head is formulated as a single-class ("cone") detector: the
heatmap collapses from 3 channels to 1, the feature extractor is unchanged, and chromatic
classification is delegated to the camera modality in the fusion stage. The colour
annotations are retained on disk still; `ConeDataset` merges them to a single class at load time.

### 1.2 Capacity vs. Coverage
A cone is detectable only if the LiDAR returns enough points from it. Evaluation must
therefore separate two distinct numbers:
- **Network capacity**: recall/precision on the *detectable* population (cones with `>= MIN_POINTS_FOR_GT` points, inside the sensor range). This is essentially what the network can do, achieving stunning results later documented.
- **Sensor coverage**: the *fraction* of annotated cones that are detectable, broken down by distance. This is an intrinsic property of the sensor and its mounting (think about a forward LiDAR that has a near blind zone and a far range thinning), not of the network itself, and is exactly what the camera should compensate in the fusion step.

Counting *undetectable* cones (0-2 points) as "missed" deflates the raw recall (≈ 0.68) and
hides the true capacity (≈ 0.99). `evaluate.py` reports both using an ignore region, meaning that
a prediction that matches a sub-threshold cone is neither a true nor a false positive.

### 1.3 The detectability threshold
Training with `MIN_POINTS_FOR_GT = 3` ignores 1-2 point cones. The band corresponding to the far range looks weak purely because far cones are sparse. Lowering it to 2 makes 2-point cones training targets and recovers them at essentially no cost to dense cones or precision. `2` is the adopted value. `1` is still not used since a single return is indistinguishable from clutter.

### 1.4 Sim to real gap
The clean CARLA geometry makes the model critically fragile to measurement noise. A robustness probe showed that ~1-3 cm of per-point position jitter, a noise level approximating that of a real LiDAR, collapses precision (from 0.997 to 0.54 at 3cm) and explodes false positives. Point-level domain randomization during
training (jitter + dropout + unlabelled cone-like clutter) reduces this gap: after hardening, precision at 3cm jitter is 0.99 and clean performance is unchanged.

### 1.5 Confidence metrics (AUROC and ECE)
Every detection carries a confidence *score* (the heatmap peak value). Two independent
questions can be asked about it, and `evaluate.py` reports one metric for each.

* **AUROC** (Area Under the ROC Curve): a measure of *separation*. It is the probability that
  a random true positive scores higher than a random false positive. `0.5` means the
  score is useless (true and false detections are indistinguishable); `1.0` means perfect
  separation (every true positive scores above every false one). High AUROC says a single
  threshold can cleanly split correct from spurious detections, i.e. the score is a
  reliable confidence signal for thresholding and for weighting the branch in fusion. It
  depends only on the *ordering* of scores, not on their absolute values.
* **ECE** (Expected Calibration Error): a measure of *calibration*, whether the score
  behaves like an actual probability. Predictions are binned by score. In each bin the
  mean score (confidence) is compared to the empirical fraction that are true positives
  (accuracy), and ECE is the population-weighted average of `|confidence − accuracy|`.
  `0` means a score of, say, `0.8` corresponds to being correct 80 % of the time. A large
  ECE means the model is mis-calibrated, over-confident (scores above the diagonal of the
  reliability diagram) or under-confident (below). This matters only if the raw score is
  used as a probability/weight, e.g. in the fusion node.

The two are complementary: a model can separate perfectly (AUROC ≈ 1) yet be mis-calibrated
(scores systematically too high/low, ECE large), and vice-versa. This branch achieves
AUROC ≈ 1.0 (excellent separation) with ECE ≈ 0.05 (well calibrated, mildly
under-confident), so the score is usable both as a threshold and as a fusion weight.

## 2. Repository structure

```
lib/OpenPCDet/
    ...

src/lidar_detection/
  datasets/
    __init__.py
    dataset_adapter.py
  configs/
    base_3class/
      cone_dataset.yaml
      second_centerpoint_cones.yaml
    agnostic/
      cone_dataset_agnostic.yaml
      second_centerpoint_agnostic.yaml
    gt_sampling/
      cone_dataset_agnostic_gtsampling.yaml
      second_centerpoint_agnostic_gts.yaml
    noise/
      cone_dataset_agnostic_noise.yaml
      second_centerpoint_agnostic_noise.yaml
    test/
      cone_dataset_agnostic_test.yaml
      second_centerpoint_agnostic_test.yaml

  tools/
    generate_db_info.py
    repack_db_infos_agnostic.py
    train_launcher.py
    check_sanity_overfit.py
    run_inference.py
    evaluate.py
    robustness_probe.py
    wandb_log_pcl.py
    test_intensity_distribution.py
    dataset_stats.py
    visualize_samples.py
    train_example.py
    train_example_subset.py
```
**datasets/** — `__init__.py` registers `ConeDataset` in the OpenPCDet dataset registry.
`dataset_adapter.py` is `ConeDataset`, the disk to OpenPCDet adapter: it reads scenes and
JSON labels, collapses the three colour classes to a single `cone` class
(`MERGE_CLASSES_TO`), filters ground-truth cones by point count (`MIN_POINTS_FOR_GT`), and
applies the training-only point-noise domain randomization (`NOISE_AUG`: per-point jitter,
dropout, and unlabelled cone-like clutter).

**configs/** — each folder is a dataset+model pair for one experiment, and each model
config references its dataset config through `DATA_CONFIG._BASE_CONFIG_`. `base_3class/`
holds the original three-class configs, used only by `generate_db_info.py` as the source
of the infos and the gt_database. `agnostic/` is the **adopted** class-agnostic model
(`MIN_POINTS_FOR_GT = 2`); the `MIN_POINTS_FOR_GT = 3` ablation is the same config with that
single value changed. `gt_sampling/` is the discarded gt_sampling ablation. `noise/` is
the final, sim-to-real-hardened model that enables `NOISE_AUG`. `test/` holds the held-out
test-split variants (`DATA_SPLIT test: test`), used once for the final evaluation.

**tools/** — `generate_db_info.py` builds `cone_infos_{train,val,test}.pkl` and the
gt_database (the LiDAR points cropped per cone) from the raw scenes.
`repack_db_infos_agnostic.py` merges the colour-keyed gt_database into a single `cone`
key, with an optional point-count filter (`--min-points/--max-points`) to build a sparse
database for targeted gt_sampling. `train_launcher.py` launches OpenPCDet training with
wandb: system metrics, weight and gradient histograms (`wandb.watch`). `check_sanity_overfit.py`
overfits a single batch as a wiring sanity check (the loss must crash toward zero).
`run_inference.py` runs the model on a split and dumps the full 7D predicted and
ground-truth boxes, scores and per-cone point counts to a `.pkl`. `evaluate.py` produces
the geometry-focused evaluation from a dump: capacity vs coverage, breakdowns by distance,
point density and azimuth, localization error (radial/lateral/z), false-positive analysis,
score confidence and calibration (AUROC, ECE), the thesis figures, and a cumulative
results JSON plus CSV keyed by run name and point threshold. `robustness_probe.py`
measures sim-to-real fragility: it perturbs the point clouds (dropout, jitter, cone-like
clutter) at increasing severity, re-runs inference at a fixed operating point, and plots
the degradation curves. `wandb_log_pcl.py` logs interactive 3D scenes (points,
ground truth in green, predictions in red) to wandb from a checkpoint.
`test_intensity_distribution.py` runs the LiDAR intensity separability analysis (from the
gt_database crops) that justifies the class-agnostic decision. `dataset_stats.py`
characterises the dataset: per-class ground-truth counts, cones-per-frame and
frames-per-scene, the BEV spatial and radial cone distributions, the mean LiDAR point
density, and the per-class intensity distributions with a Bhattacharyya overlap
coefficient. `visualize_samples.py` renders a static BEV view of a labelled scene.
`train_example.py` and `train_example_subset.py` are thin launchers and a subset-split
helper.

## 3. Requirements and `PYTHONPATH`
Python 3, PyTorch + CUDA, `spconv`, a working **OpenPCDet** install (`lib/OpenPCDet`,
installed with `python setup.py develop`), plus `wandb`, `numpy`, `matplotlib`, and `pandas`
(used by `dataset_stats.py`).

`ConeDataset` lives in the `lidar_detection` package under `src/`. Any script that
**imports `lidar_detection`** (i.e. builds the dataset/model or runs the network) must be
run with `src` on the path. From the repo root:
```bash
PYTHONPATH=src python3 src/lidar_detection/tools/<script>.py ...
```

Scripts that **need** `PYTHONPATH=src`: `generate_db_info.py`, `run_inference.py`,
`robustness_probe.py`, `wandb_log_pcl.py`, `check_sanity_overfit.py`
(and `train_launcher.py`, which also inserts `src` itself, so the prefix is harmless).

Scripts that **do not** need it (pure post-processing on files): `evaluate.py`,
`test_intensity_distribution.py`, `dataset_stats.py`, `visualize_samples.py`.

`--cfg` paths are relative to the repo root. Model configs reference their dataset config
via `DATA_CONFIG._BASE_CONFIG_`, so keep those paths in sync with the folder layout above.

## 4. Data preparation (run once)
Generate the frame infos (train/val/test) and the gt_database from the raw scenes. Use a
**3-class** model config (its `CLASS_NAMES` and `gt_sampling.PREPARE` block are read here). The
infos keep the original colours and are collapsed to `cone` only at training load time.

```bash
PYTHONPATH=src python3 src/lidar_detection/tools/generate_db_info.py \
    --cfg src/lidar_detection/configs/base_3class/second_centerpoint_cones.yaml
```

`splits/{train,val,test}.txt` must exist and be **scene-disjoint** (partition is by scene,
not by frame, to avoid leakage between near-identical consecutive frames).

Note: `ConeDataset` reads scenes and JSON labels directly, so a plain training/eval run
does not strictly require the `*_infos_*.pkl`; they are needed by the diagnostics and by
gt_sampling (via the gt_database).

## 5. Training
```bash
# 0) sanity: loss must crash toward ~0 on a single batch
PYTHONPATH=src python3 src/lidar_detection/tools/check_sanity_overfit.py \
    --cfg src/lidar_detection/configs/agnostic/second_centerpoint_agnostic.yaml --iters 300

# 1) full training (wandb run name via env)
WANDB_RUN_NAME=CenterPoint_ClassAgnostic \
PYTHONPATH=src python3 train_launcher.py \
    --cfg_file src/lidar_detection/configs/agnostic/second_centerpoint_agnostic.yaml \
    --batch_size 24 --epochs 80
```

`train_launcher.py` mirrors OpenPCDet's TensorBoard logs into wandb (System tab: GPU/VRAM
/IO), attaches `wandb.watch` (weight + gradient histograms; disable with `WANDB_WATCH=0`),
and logs a class-agnostic AP at the end.

Which config to train:
* `agnostic/…`         adopted class-agnostic model (`MIN_POINTS_FOR_GT = 2`; set to `3` for
                       the threshold ablation).
* `noise/…`            final, sim-to-real-hardened model (requires the NOISE_AUG-aware
                       `dataset_adapter.py`).
* `gt_sampling/…`      Needs a single-class db first:
  `PYTHONPATH=src python3 src/lidar_detection/tools/repack_db_infos_agnostic.py \
     --in <dataset>/cone_dbinfos_train.pkl --out <dataset>/cone_dbinfos_train_sparse.pkl \
     --min-points 2 --max-points 6`

## 6. Inference and evaluation
```bash
# dump predictions (low score threshold to keep the full PR/score range)
PYTHONPATH=src python3 src/lidar_detection/tools/run_inference.py \
    --cfg src/lidar_detection/configs/test/second_centerpoint_agnostic_test.yaml \
    --ckpt lib/OpenPCDet/output/.../ckpt/checkpoint_epoch_80.pth \
    --out preds_TEST.pkl --score_thresh 0.01

# geometry report + figures + cumulative metrics (no PYTHONPATH needed)
python3 src/lidar_detection/tools/evaluate.py \
    --dump preds_TEST.pkl --run_name agnostic_minpts2_TEST --min_gt_points 2
```

`evaluate.py` writes per-run figures + `metrics.json` under `eval_report/<run>_minpts<N>/`,
and appends a row to `results/eval_results.json` + `results/eval_summary.csv` keyed by
`<run_name>@minpts<N>` (so all configurations accumulate into one comparison table).
Use matched `--min_gt_points` for training and evaluation (2 for the adopted model).

## 7. Diagnostics and analysis
```bash
# dataset characterisation (counts, spatial/radial distribution, density, intensity)
python3 src/lidar_detection/tools/dataset_stats.py \
    --root <data> --out dataset_report

# colour separability from LiDAR intensity (justifies class-agnostic)
python3 src/lidar_detection/tools/test_intensity_distribution.py \
    --dbinfos <dataset>/cone_dbinfos_train.pkl --root <data> --out diag_0a

# interactive 3D scenes to wandb (qualitative)
PYTHONPATH=src python3 src/lidar_detection/tools/wandb_log_pcl.py \
    --cfg <test cfg> --ckpt <ckpt> --num_frames 12 --score_thresh 0.3
```

### Robustness probing (sim-to-real)
```bash
PYTHONPATH=src python3 src/lidar_detection/tools/robustness_probe.py \
    --cfg src/lidar_detection/configs/test/second_centerpoint_agnostic_test.yaml \
    --ckpt lib/OpenPCDet/output/.../ckpt/checkpoint_epoch_80.pth --out_dir robustness --max_frames 800
```
Perturbs the point clouds. **dropout** (missing returns, stresses recall), **jitter**
(measurement noise, stresses localization/precision), **clutter** (cone-like debris,
stresses precision), at increasing severity, re-runs inference at a fixed operating
threshold, and writes `robustness_{dropout,jitter,clutter}.png` + `robustness.json`.
Run it on both the plain and the noise-hardened checkpoint for a before/after comparison.

## 8. Experiments and results
All figures below are class-agnostic. Capacity numbers are on the detectable population
(`>= 2` points, in range) unless noted. Tolerance 0.5 m.

| # | Experiment | Split | Recall | Precision | FP/frame | AP | Loc (cm) | Coverage | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | agnostic, MIN_POINTS=3 | val | 0.997 | 0.999 | 0.013 | 0.999 | 0.81 | 66.3% | perfect on ≥3-pt cones |
| 2 | agnostic, MIN_POINTS=2 | val | 0.996 | 0.998 | 0.030 | 0.999 | 0.88 | 75.1% | dominates #1 (recovers 2-pt cones) |
| 3 | agnostic, MIN_POINTS=2 | **test** | 0.997 | 0.998 | 0.029 | 0.999 | 0.89 | 74.7% | test≈val, no overfit/leakage |
| 4 | + gt_sampling (sparse) | test | 0.994 | 0.994 | 0.080 | 0.999 | 0.90 | 74.7% | **discarded**: loss of precision, no gain |
| 5 | + noise (domain rand.) | test | 0.995 | 0.998 | 0.029 | 0.998 | 0.92 | 74.7% | **final**: robust, clean perf. kept |

Rows 1–2 share the `agnostic/` config; #1 is reproduced by setting `MIN_POINTS_FOR_GT: 3`.

Confidence (run #3): score AUROC ≈ 1.0 (near-perfect TP/FP separation), ECE ≈ 0.05
(well-calibrated, mildly under-confident). Score degrades gracefully with distance and
sparsity (usable as a fusion weight).

Gradient-flow analysis (run #2, from the `wandb.watch` histograms logged by
`train_launcher.py`): the 3D and 2D backbones receive gradients of the same order of
magnitude as the heatmap head throughout training, implying that the feature extractor is
genuinely trained (not a frozen backbone with the head memorizing). The `dim`/`rot` heads
do **not** vanish in the gradient plot; this is an L1-loss artifact (constant sub-gradient
near the optimum). Their *output* is constant and correct (see `dim` error ≈ 0 in the
eval), which is the real evidence the model ignores those degrees of freedom.

## Robustness (before vs after noise hardening)
| Perturbation (real-regime point) | Metric | Before | After |
|---|---|---|---|
| jitter 3 cm | precision | 0.543 | **0.991** |
| jitter 3 cm | FP/frame | 11.6 | **0.13** |
| jitter 5 cm | recall | 0.404 | **0.986** |
| clutter 100/frame | precision | 0.481 | **0.993** |
| dropout 60% | recall | 0.694 | **0.871** |
| clean (severity 0) | recall / precision | 0.995 / 0.997 | 0.995 / 0.998 |

The dominant sim-to-real risk (position noise) is closed in the realistic 1–3 cm regime at
no cost to clean accuracy. (Beyond ~10 cm — outside both the real regime and the training
range — the model still collapses.) Robustness to *simulated*
shift is not a guarantee of real-world transfer, which only real data would confirm.

*Note*: Training and inference performed on an RTX 4090 GPU (24 GB VRAM), with `batch_size` = 24 and 80 epochs.

## 9. Regeneration
Everything not versioned is reproducible from the code above, in order: `generate_db_info` --> (optional `repack_db_infos_agnostic` for gt_sampling) --> `train_launcher` --> `run_inference` --> `evaluate`
--> `robustness_probe`. Figures and metric tables are written by `evaluate.py` and
`robustness_probe.py`.
>>>>>>> feat/lidar-pipeline
