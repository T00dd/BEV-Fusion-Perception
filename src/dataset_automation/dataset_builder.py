import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import carla
import yaml

from scene_recorder import record_scene, scene_is_complete


def setup_logging(out_dir):
    log_dir = Path(out_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / time.strftime("build_%Y%m%d_%H%M%S.log")

    logger = logging.getLogger("dataset_builder")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Logging to {log_path}")
    return logger


def build_manifest(cfg):

    # cycle conditions across scenes then shuffle to assign splits
    # ensure that each split has a good mix of conditions, rather than all of one condition in train, etc.
    conditions = cfg["conditions"]
    n_scenes = cfg["scenes"]["count"]

    # Per-scene randomisation (zone + track geometry), seeded for reproducibility.
    # Everything that varies per scene is frozen HERE in the manifest, so a run
    # is fully determined by split_seed and can be resumed/reproduced exactly.
    scn = cfg["scenes"]
    geo_seed = scn.get("geometry_seed", scn.get("split_seed", 0))
    grng = random.Random(geo_seed)

    zones = scn.get("zones", [1, 2])              # alternate across these
    lobes_min = scn.get("lobes_min", 2)           # keep a low floor of curves
    lobes_max_lo = scn.get("lobes_max_min", 4)    # randomise the CEILING in
    lobes_max_hi = scn.get("lobes_max_max", 7)    # [lobes_max_lo, lobes_max_hi]

    scenes = []
    for i in range(n_scenes):
        cond = conditions[i % len(conditions)]
        zone = grng.choice(zones)
        lobes_max = grng.randint(lobes_max_lo, lobes_max_hi)
        scenes.append({
            "scene_id": f"scene_{i:04d}",
            "condition": cond["name"],
            "zone": zone,
            "track_seed": grng.randint(0, 2**31 - 1),  # distinct track per scene
            "lobes_min": lobes_min,
            "lobes_max": lobes_max,
        })

    rng = random.Random(cfg["scenes"].get("split_seed", 0))
    order = list(range(n_scenes))
    rng.shuffle(order)
    r = cfg["scenes"]["split_ratios"]
    n_train = int(n_scenes * r["train"])
    n_val = int(n_scenes * r["val"])
    split_of = {}
    for rank, idx in enumerate(order):
        if rank < n_train:
            split_of[idx] = "train"
        elif rank < n_train + n_val:
            split_of[idx] = "val"
        else:
            split_of[idx] = "test"
    for i, s in enumerate(scenes):
        s["split"] = split_of[i]
    return scenes


def load_or_create_manifest(cfg, out_dir, logger):
    mpath = Path(out_dir) / "manifest.json"
    if mpath.exists():
        manifest = json.loads(mpath.read_text())
        logger.info(f"Loaded existing manifest with {len(manifest)} scenes")
        return manifest
    manifest = build_manifest(cfg)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2))
    logger.info(f"Created manifest with {len(manifest)} scenes")
    return manifest


def frames_in_scene(scene_dir):
    # Read frames_written from a completed scene's _COMPLETE marker (0 if absent).
    marker = Path(scene_dir) / "_COMPLETE"
    if not marker.exists():
        return 0
    try:
        return int(json.loads(marker.read_text()).get("frames_written", 0))
    except Exception:
        return 0


def condition_by_name(cfg, name):
    for c in cfg["conditions"]:
        if c["name"] == name:
            return c
    raise KeyError(f"condition '{name}' not found in config")


def write_splits(out_dir, manifest):
    
    split_dir = Path(out_dir) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    buckets = {"train": [], "val": [], "test": []}
    for s in manifest:
        scene_dir = Path(out_dir) / "scenes" / s["scene_id"]
        if scene_is_complete(scene_dir):
            buckets[s["split"]].append(s["scene_id"])
    for split, ids in buckets.items():
        (split_dir / f"{split}.txt").write_text("\n".join(ids) + ("\n" if ids else ""))
    return {k: len(v) for k, v in buckets.items()}


def verify(out_dir, manifest, logger):
    # report completion status and rebuild splits, then exit
    done, missing = [], []
    for s in manifest:
        scene_dir = Path(out_dir) / "scenes" / s["scene_id"]
        (done if scene_is_complete(scene_dir) else missing).append(s["scene_id"])
    logger.info(f"VERIFY: {len(done)} complete, {len(missing)} missing")
    if missing:
        logger.info("Missing: " + ", ".join(missing))
    counts = write_splits(out_dir, manifest)
    logger.info(f"Split counts (completed scenes): {counts}")
    return len(missing) == 0


def wipe_stale_cones(world, client, logger):
    # At startup (and after a resumed stop) the world may still hold cones from
    # an interrupted scene. Remove them so a new scene's cones don't overlap
    # leftovers. Reuses track_spawner's cone detection.
    from track_spawner import clean_previous_track
    try:
        clean_previous_track(world, client)
        logger.info("Startup: cleared any stale cones from a previous run.")
    except Exception as e:
        logger.warning(f"Startup cone cleanup failed (continuing): {e}")


def connect(cfg, logger):
    client = carla.Client(cfg["carla"]["host"], cfg["carla"]["port"])
    client.set_timeout(cfg["carla"].get("timeout", 20.0))
    world = client.get_world()

    # Force synchronous mode for deterministic capture.
    settings = world.get_settings()
    original = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = cfg["carla"]["fixed_delta_seconds"]
    world.apply_settings(settings)

    tm = client.get_trafficmanager(cfg["carla"].get("tm_port", 8000))
    tm.set_synchronous_mode(True)
    logger.info(f"Connected to CARLA, sync dt={cfg['carla']['fixed_delta_seconds']}")
    return client, world, original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dataset_config.yaml")
    ap.add_argument("--verify", action="store_true",
                    help="report completion status and rebuild splits, then exit")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(cfg["output"]["dataset_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(out_dir)
    logger.info(f"Config: {args.config}")

    manifest = load_or_create_manifest(cfg, out_dir, logger)

    if args.verify:
        verify(out_dir, manifest, logger)
        return

    client, world, original_settings = connect(cfg, logger)
    wipe_stale_cones(world, client, logger)

    total_done = 0
    total_frames = 0
    n_scenes = len(manifest)
    try:
        for s in manifest:
            scene_id = s["scene_id"]
            scene_dir = out_dir / "scenes" / scene_id

            if scene_is_complete(scene_dir):
                f = frames_in_scene(scene_dir)
                total_frames += f
                total_done += 1
                logger.info(f"[{scene_id}] already complete ({f} frames) -> "
                            f"skip (resume)")
                continue

            cond = condition_by_name(cfg, s["condition"])
            logger.info(f"[{scene_id}] zone={s.get('zone')} "
                        f"lobes=[{s.get('lobes_min')},{s.get('lobes_max')}] "
                        f"track_seed={s.get('track_seed')} "
                        f"condition={s['condition']} split={s.get('split')}")
            try:
                f = record_scene(client, world, scene_id, scene_dir, cond, cfg,
                                 logger, scene_meta=s)
                total_frames += int(f or 0)
                total_done += 1
            except KeyboardInterrupt:
                logger.warning("Interrupted by user; current scene left incomplete "
                               "(no marker). Re-run to resume.")
                raise
            except Exception as e:
                logger.error(f"[{scene_id}] FAILED: {e}. Leaving incomplete, moving on.")
                # No _COMPLETE marker -> it will be retried on next run.
                continue

            # Global progress: frames so far + a projection from the running
            # average frames-per-scene (refines as more scenes complete).
            avg = total_frames / max(total_done, 1)
            projected = int(round(avg * n_scenes))
            logger.info(f"PROGRESS: {total_frames} frames across "
                        f"{total_done}/{n_scenes} scenes "
                        f"(~{projected} projected at {avg:.0f} frames/scene)")

            # Refresh splits as we go so partial datasets are usable.
            write_splits(out_dir, manifest)

    finally:
        world.apply_settings(original_settings)
        counts = write_splits(out_dir, manifest)
        logger.info(f"Run finished. Completed {total_done}/{n_scenes} scenes, "
                    f"{total_frames} frames total. Split counts: {counts}")


if __name__ == "__main__":
    main()