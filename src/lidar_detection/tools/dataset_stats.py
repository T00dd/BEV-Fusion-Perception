#!/usr/bin/env python3
import argparse
import os
import glob
import pickle
import sys
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")            
import matplotlib.pyplot as plt


CLASS_COLORS = {
    "blue":   "#1f77b4",
    "yellow": "#e6b800",
    "orange": "#e8730c",
}

def canon_class(name):
    n = str(name).strip().lower()
    if "blue" in n:   return "blue"
    if "yellow" in n: return "yellow"
    if "orange" in n: return "orange"
    return n

def load_infos(root):
    
    candidates = {
        "train": "cone_infos_train.pkl",
        "val":   "cone_infos_val.pkl",
        "test":  "cone_infos_test.pkl",
    }
    infos, split_of = [], []
    for split, fname in candidates.items():
        path = os.path.join(root, fname)
        if not os.path.isfile(path):
            print(f"  [info] {fname} non trovato, salto.")
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, list):

            data = data.get("infos", data) if isinstance(data, dict) else list(data)
        print(f"  [info] {fname}: {len(data)} frame")
        infos.extend(data)
        split_of.extend([split] * len(data))
    return infos, split_of


def extract_gt_names(frame):
    
    annos = frame.get("annos") if isinstance(frame, dict) else None
    if isinstance(annos, dict):
        for k in ("gt_names", "name", "names"):
            if k in annos and annos[k] is not None:
                return list(np.atleast_1d(annos[k]))

    for k in ("gt_names", "name", "names"):
        if isinstance(frame, dict) and k in frame and frame[k] is not None:
            return list(np.atleast_1d(frame[k]))
    return []


def extract_gt_boxes(frame):

    annos = frame.get("annos") if isinstance(frame, dict) else None
    if isinstance(annos, dict):
        for k in ("gt_boxes_lidar", "boxes_lidar", "gt_boxes", "location"):
            if k in annos and annos[k] is not None:
                return np.atleast_2d(np.asarray(annos[k], dtype=np.float32))
    for k in ("gt_boxes_lidar", "gt_boxes", "boxes_lidar"):
        if isinstance(frame, dict) and k in frame and frame[k] is not None:
            return np.atleast_2d(np.asarray(frame[k], dtype=np.float32))
    return None


def frame_lidar_path(frame, root):
    
    for k in ("lidar_path", "point_cloud", "velodyne_path", "path", "frame_id"):
        if isinstance(frame, dict) and k in frame:
            v = frame[k]
            if isinstance(v, dict): 
                continue
            p = os.path.join(root, str(v)) if not os.path.isabs(str(v)) else str(v)
            if os.path.isfile(p):
                return p
    return None

def scan_scenes(root):
    
    scenes = {}
    scenes_dir = os.path.join(root, "scenes")
    if not os.path.isdir(scenes_dir):
        print("  [scenes] cartella 'scenes/' non trovata.")
        return scenes
    for sc in sorted(glob.glob(os.path.join(scenes_dir, "*"))):
        if not os.path.isdir(sc):
            continue
        bins = sorted(glob.glob(os.path.join(sc, "lidar", "*.bin")))
        if bins:
            scenes[os.path.basename(sc)] = bins
    return scenes


def read_bin(path, n_feat=4):
    """Legge un .bin LiDAR come (N, n_feat) float32."""
    pts = np.fromfile(path, dtype=np.float32)
    if pts.size % n_feat != 0:

        for nf in (5, 3, 4):
            if pts.size % nf == 0:
                n_feat = nf
                break
    return pts.reshape(-1, n_feat)

def points_in_box(points_xy, box):
    
    cx, cy = box[0], box[1]
    dx = box[3] if len(box) > 3 else 0.3
    dy = box[4] if len(box) > 4 else 0.3
    hx, hy = dx / 2.0 + 0.05, dy / 2.0 + 0.05   # piccolo margine
    m = (np.abs(points_xy[:, 0] - cx) <= hx) & (np.abs(points_xy[:, 1] - cy) <= hy)
    return m


def collect_intensity_by_class(infos, root, max_frames=None):
    
    inten = defaultdict(list) 
    inten_raw = defaultdict(list)      
    used = 0
    for i, fr in enumerate(infos):
        if max_frames and used >= max_frames:
            break
        boxes = extract_gt_boxes(fr)
        names = extract_gt_names(fr)
        binp = frame_lidar_path(fr, root)
        if boxes is None or not names or binp is None:
            continue
        try:
            pc = read_bin(binp)
        except Exception:
            continue
        if pc.shape[1] < 4:
            continue
        xy = pc[:, :2]
        inten_col = pc[:, 3]
        for b, nm in zip(boxes, names):
            cls = canon_class(nm)
            m = points_in_box(xy, b)
            if m.sum() == 0:
                continue
            vals = inten_col[m]
            inten[cls].append(float(vals.mean()))
            inten_raw[cls].extend(vals.tolist())
        used += 1
    print(f"  [intensity] frame usati: {used}")
    return inten, inten_raw

def fig_class_counts(class_counter, outdir):
    labels = [c for c in ["blue", "yellow", "orange"] if c in class_counter]
    labels += [c for c in class_counter if c not in labels]
    vals = [class_counter[c] for c in labels]
    colors = [CLASS_COLORS.get(c, "#888") for c in labels]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:,}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("numero istanze (GT)")
    ax.set_title("Istanze ground-truth per classe di colore")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "gt_class_counts.png"), dpi=160); plt.close(fig)


def fig_cones_per_frame(per_frame_counts, outdir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(per_frame_counts, bins=range(0, max(per_frame_counts)+2),
            color="#4c72b0", edgecolor="black", linewidth=0.5, align="left")
    ax.set_xlabel("coni per frame"); ax.set_ylabel("numero frame")
    ax.set_title("Distribuzione del numero di coni per frame")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "cones_per_frame.png"), dpi=160); plt.close(fig)


def fig_frames_per_scene(frames_per_scene, outdir):
    scenes = list(frames_per_scene.keys())
    vals = [frames_per_scene[s] for s in scenes]
    fig, ax = plt.subplots(figsize=(max(7, len(scenes)*0.25), 4.5))
    ax.bar(range(len(scenes)), vals, color="#55a868", edgecolor="none")
    ax.axhline(np.mean(vals), color="k", ls="--", lw=1, label=f"media={np.mean(vals):.1f}")
    ax.set_xlabel("scena (indice)"); ax.set_ylabel("frame per scena")
    ax.set_title("Frame per scena"); ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "frames_per_scene.png"), dpi=160); plt.close(fig)


def fig_gt_spatial_heatmap(all_xy_by_class, pc_range, outdir):
    
    xmin, ymin, _, xmax, ymax, _ = pc_range
    bins = [np.linspace(xmin, xmax, 120), np.linspace(ymin, ymax, 120)]

    allxy = np.vstack([v for v in all_xy_by_class.values() if len(v)]) if all_xy_by_class else np.empty((0,2))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if len(allxy):
        H, xe, ye = np.histogram2d(allxy[:,0], allxy[:,1], bins=bins)
        im = ax.imshow(H.T, origin="lower", extent=[xmin,xmax,ymin,ymax],
                       aspect="auto", cmap="gist_heat_r")
        fig.colorbar(im, ax=ax, label="conteggio centri GT")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Distribuzione spaziale BEV dei coni (tutte le classi)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "gt_spatial_heatmap.png"), dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cls, xy in all_xy_by_class.items():
        if len(xy):
            xy = np.asarray(xy)
            ax.scatter(xy[:,0], xy[:,1], s=3, alpha=0.35,
                       c=CLASS_COLORS.get(cls, "#888"), label=cls)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Posizione dei coni per classe (BEV)")
    ax.legend(markerscale=3, loc="upper right")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "gt_spatial_by_class.png"), dpi=160); plt.close(fig)


def fig_radial_distance(all_xy_by_class, outdir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cls, xy in all_xy_by_class.items():
        if len(xy):
            xy = np.asarray(xy); r = np.sqrt((xy**2).sum(1))
            ax.hist(r, bins=40, histtype="step", linewidth=1.8,
                    color=CLASS_COLORS.get(cls, "#888"), label=cls, density=True)
    ax.set_xlabel("distanza radiale dal sensore [m]"); ax.set_ylabel("densita'")
    ax.set_title("Distribuzione della distanza dei coni dal LiDAR")
    ax.legend(); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "gt_radial_distance.png"), dpi=160); plt.close(fig)


def fig_lidar_density_heatmap(scenes, pc_range, outdir, max_frames=200):
    """Heatmap BEV della densita' MEDIA dei punti LiDAR (contesto sensore)."""
    xmin, ymin, _, xmax, ymax, _ = pc_range
    bins = [np.linspace(xmin, xmax, 150), np.linspace(ymin, ymax, 150)]
    H = np.zeros((149, 149)); nf = 0
    for sc, bins_list in scenes.items():
        for bp in bins_list:
            if nf >= max_frames: break
            try:
                pc = read_bin(bp)
            except Exception:
                continue
            h, _, _ = np.histogram2d(pc[:,0], pc[:,1], bins=bins)
            H += h; nf += 1
        if nf >= max_frames: break
    if nf == 0: return
    H /= nf
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(np.log1p(H).T, origin="lower", extent=[xmin,xmax,ymin,ymax],
                   aspect="auto", cmap="magma")
    fig.colorbar(im, ax=ax, label="log(1 + punti medi / cella)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"Densita' media punti LiDAR (BEV), {nf} frame")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "lidar_density_heatmap.png"), dpi=160); plt.close(fig)


def fig_intensity_by_class(inten_mean, inten_raw, outdir):
    
    have = [c for c in ["blue","yellow","orange"] if inten_mean.get(c)]
    if not have:
        print("  [intensity] nessun dato di intensita' raccolto; salto la figura.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for cls in have:
        v = np.asarray(inten_mean[cls])
        axes[0].hist(v, bins=40, histtype="stepfilled", alpha=0.4, density=True,
                     color=CLASS_COLORS.get(cls, "#888"), label=f"{cls} (n={len(v)})")
        axes[0].hist(v, bins=40, histtype="step", density=True,
                     color=CLASS_COLORS.get(cls, "#888"), linewidth=1.6)
    axes[0].set_xlabel("intensita' media per cono"); axes[0].set_ylabel("densita'")
    axes[0].set_title("(a) Intensita' media per cono, per classe")
    axes[0].legend(); axes[0].spines[["top","right"]].set_visible(False)

    for cls in have:
        v = np.asarray(inten_raw[cls])
        if len(v) > 200000:  
            v = np.random.choice(v, 200000, replace=False)
        axes[1].hist(v, bins=60, histtype="step", density=True, linewidth=1.6,
                     color=CLASS_COLORS.get(cls, "#888"), label=cls)
    axes[1].set_xlabel("intensita' punto LiDAR"); axes[1].set_ylabel("densita'")
    axes[1].set_title("(b) Intensita' punto-per-punto, per classe")
    axes[1].legend(); axes[1].spines[["top","right"]].set_visible(False)

    fig.suptitle("Sovrapposizione delle distribuzioni di intensita': "
                 "il colore non e' separabile dal solo LiDAR", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "intensity_by_class.png"), dpi=160); plt.close(fig)

    def bhattacharyya(a, b, bins=50):
        lo = min(a.min(), b.min()); hi = max(a.max(), b.max())
        ha, e = np.histogram(a, bins=bins, range=(lo,hi), density=True)
        hb, _ = np.histogram(b, bins=bins, range=(lo,hi), density=True)
        ha = ha/ha.sum(); hb = hb/hb.sum()
        return float(np.sum(np.sqrt(ha*hb))) 
    print("  [intensity] coefficiente di sovrapposizione (Bhattacharyya, 1=identico):")
    for i in range(len(have)):
        for j in range(i+1, len(have)):
            a = np.asarray(inten_mean[have[i]]); b = np.asarray(inten_mean[have[j]])
            print(f"      {have[i]:7s} vs {have[j]:7s}: {bhattacharyya(a,b):.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="cartella carla_dataset_three_classses")
    ap.add_argument("--out", default="./dataset_report", help="cartella output figure")
    ap.add_argument("--pc-range", default="0,-25,-3,50,25,1",
                    help="xmin,ymin,zmin,xmax,ymax,zmax (dal config)")
    ap.add_argument("--max-intensity-frames", type=int, default=1500,
                    help="limite frame per l'analisi intensita' (velocita')")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    outdir = os.path.abspath(args.out)
    os.makedirs(outdir, exist_ok=True)
    pc_range = [float(x) for x in args.pc_range.split(",")]

    print("="*70)
    print("ROOT:", root)
    print("OUT :", outdir)
    print("="*70)

    print("\n[1] Caricamento info .pkl")
    infos, split_of = load_infos(root)
    if not infos:
        print("!! Nessun info .pkl caricato. I conteggi GT non saranno disponibili.")

    class_counter = Counter()
    per_frame_counts = []
    all_xy_by_class = defaultdict(list)
    for fr in infos:
        names = extract_gt_names(fr)
        boxes = extract_gt_boxes(fr)
        per_frame_counts.append(len(names))
        for idx, nm in enumerate(names):
            cls = canon_class(nm)
            class_counter[cls] += 1
            if boxes is not None and idx < len(boxes):
                all_xy_by_class[cls].append(boxes[idx][:2])

    print("\n[2] Scansione scenes/")
    scenes = scan_scenes(root)
    frames_per_scene = {s: len(b) for s, b in scenes.items()}
    tot_frames_fs = sum(frames_per_scene.values())

    print("\n[3] Raccolta intensita' per classe")
    inten_mean, inten_raw = collect_intensity_by_class(
        infos, root, max_frames=args.max_intensity_frames)

    print("\n" + "="*70)
    print("RISULTATI")
    print("="*70)
    print(f"Scene totali (filesystem)      : {len(scenes)}")
    print(f"Frame totali (filesystem .bin) : {tot_frames_fs}")
    print(f"Frame totali (info .pkl)       : {len(infos)}")
    if frames_per_scene:
        vmax = max(frames_per_scene.values()); vmin = min(frames_per_scene.values())
        smax = max(frames_per_scene, key=frames_per_scene.get)
        smin = min(frames_per_scene, key=frames_per_scene.get)
        print(f"Frame per scena  MAX           : {vmax}  ({smax})")
        print(f"Frame per scena  MIN           : {vmin}  ({smin})")
        print(f"Frame per scena  media         : {np.mean(list(frames_per_scene.values())):.2f}")
    print("-"*70)
    print("Istanze GT per classe:")
    tot_inst = sum(class_counter.values())
    for c in ["blue","yellow","orange"]:
        if c in class_counter:
            print(f"   {c:8s}: {class_counter[c]:7d}  ({100*class_counter[c]/max(tot_inst,1):.1f}%)")
    for c in class_counter:
        if c not in ("blue","yellow","orange"):
            print(f"   {c:8s}: {class_counter[c]:7d}  [classe non attesa]")
    print(f"   {'TOTALE':8s}: {tot_inst:7d}")
    if per_frame_counts:
        print("-"*70)
        print(f"Coni per frame  media/min/max  : "
              f"{np.mean(per_frame_counts):.2f} / {min(per_frame_counts)} / {max(per_frame_counts)}")

    with open(os.path.join(outdir, "report.txt"), "w") as f:
        f.write(f"scene_totali={len(scenes)}\n")
        f.write(f"frame_totali_fs={tot_frames_fs}\n")
        f.write(f"frame_totali_pkl={len(infos)}\n")
        if frames_per_scene:
            f.write(f"frame_per_scena_max={max(frames_per_scene.values())}\n")
            f.write(f"frame_per_scena_min={min(frames_per_scene.values())}\n")
        for c in class_counter:
            f.write(f"istanze_{c}={class_counter[c]}\n")

    print("\n[4] Generazione figure ->", outdir)
    if class_counter:      fig_class_counts(class_counter, outdir)
    if per_frame_counts:   fig_cones_per_frame(per_frame_counts, outdir)
    if frames_per_scene:   fig_frames_per_scene(frames_per_scene, outdir)
    if all_xy_by_class:
        fig_gt_spatial_heatmap(all_xy_by_class, pc_range, outdir)
        fig_radial_distance(all_xy_by_class, outdir)
    if scenes:             fig_lidar_density_heatmap(scenes, pc_range, outdir)
    fig_intensity_by_class(inten_mean, inten_raw, outdir)

    print("\nFatto. Figure e report in:", outdir)


if __name__ == "__main__":
    main()