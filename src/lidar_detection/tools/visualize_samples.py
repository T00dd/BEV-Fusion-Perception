import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')          
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np

CLASS_MAP = {
    'blue': 'blue', 'yellow': 'yellow',
    'orange_small': 'orange_small', 'orange_large': 'orange_large',
    'orange-small': 'orange_small', 'orange-large': 'orange_large',
}

CLASS_COLOR = {
    'blue': 'tab:blue', 'yellow': 'gold',
    'orange_small': 'orange', 'orange_large': 'darkorange',
}


def read_point_cloud(path, num_features=4):
    pts = np.fromfile(str(path), dtype=np.float32)
    assert pts.size % num_features == 0, (
        f'{path}: {pts.size} valori non divisibili per {num_features}.'
    )
    return pts.reshape(-1, num_features)


def read_cone_labels(path):
    cones = json.load(open(path))['cones']
    boxes, names, npts = [], [], []
    for c in cones:
        boxes.append(c['box'])                       # [x,y,z,dx,dy,dz,heading]
        names.append(CLASS_MAP[c['class']])
        npts.append(c['num_lidar_points'])
    boxes = np.array(boxes, np.float32).reshape(-1, 7)
    return boxes, np.array(names), np.array(npts, np.int32)


def box_corners_bev(box):
    """4 angoli della box in pianta (gestisce anche heading != 0, per robustezza)."""
    x, y, _, dx, dy, _, yaw = box
    c, s = np.cos(yaw), np.sin(yaw)
    
    local = np.array([[ dx/2,  dy/2], [ dx/2, -dy/2],
                      [-dx/2, -dy/2], [-dx/2,  dy/2]])
    rot = np.array([[c, -s], [s, c]])
    return (local @ rot.T) + np.array([x, y])


def render_bev(points, boxes, names, npts, pc_range, out_path, title,
               min_points=1, max_plot_points=60000):
   
    if len(points) > max_plot_points:
        sel = np.random.choice(len(points), max_plot_points, replace=False)
        points = points[sel]

    x_min, y_min, _, x_max, y_max, _ = pc_range
    fig, ax = plt.subplots(figsize=(10, 8))

    col = points[:, 3] if points.shape[1] >= 4 else points[:, 2]
    ax.scatter(points[:, 0], points[:, 1], s=1, c=col, cmap='viridis',
               alpha=0.35, linewidths=0)

    ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                           fill=False, ec='gray', ls='--', lw=1.0))

    for box, name, n in zip(boxes, names, npts):
        corners = box_corners_bev(box)
        kept = n >= min_points
        ax.add_patch(Polygon(corners, closed=True, fill=False,
                             ec=CLASS_COLOR.get(name, 'red'),
                             lw=1.8, ls='-' if kept else ':'))
        # annota il numero di punti LiDAR sul cono
        ax.text(box[0], box[1], str(int(n)), fontsize=6,
                ha='center', va='center', color='black')

    ax.plot(0, 0, marker=(3, 0, -90), markersize=14, color='red', zorder=5)

    ax.set_xlabel('x  avanti  [m]'); ax.set_ylabel('y  sinistra  [m]')
    ax.set_aspect('equal'); ax.grid(alpha=0.2)
    ax.set_xlim(x_min - 1, x_max + 1); ax.set_ylim(y_min - 1, y_max + 1)
    ax.set_title(title, fontsize=10)

    handles = [plt.Line2D([0], [0], color=c, lw=2, label=k)
               for k, c in CLASS_COLOR.items()]
    ax.legend(handles=handles, loc='upper right', fontsize=8)

    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def list_frames(data_path, split):
    scenes = [ln.strip() for ln in open(data_path / 'splits' / f'{split}.txt') if ln.strip()]
    frames = []
    for sc in scenes:
        ld = data_path / 'scenes' / sc / 'lidar'
        if ld.exists():
            frames += [(sc, p.stem) for p in sorted(ld.glob('*.bin'))]
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', type=Path, required=True, help='radice dataset/')
    ap.add_argument('--split', default='train')
    ap.add_argument('--num_samples', type=int, default=20)
    ap.add_argument('--out_dir', type=Path, default=Path('./viz_check'))
    ap.add_argument('--num_features', type=int, default=4)
    ap.add_argument('--min_points', type=int, default=1)
    ap.add_argument('--pc_range', type=float, nargs=6,
                    default=[0, -12.8, -3, 35.2, 12.8, 1],
                    help='[x_min y_min z_min x_max y_max z_max], stesso del config')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = list_frames(args.data_path, args.split)
    assert frames, f'nessun frame trovato per split {args.split}'
    picks = random.sample(frames, min(args.num_samples, len(frames)))

    for sc, fr in picks:
        pts = read_point_cloud(args.data_path / 'scenes' / sc / 'lidar' / f'{fr}.bin',
                               args.num_features)
        boxes, names, npts = read_cone_labels(
            args.data_path / 'scenes' / sc / 'labels' / f'{fr}.json')
        kept = int((npts >= args.min_points).sum())
        title = f'{sc}/{fr}   punti={len(pts)}   coni={len(boxes)} (tenuti={kept})'
        out = args.out_dir / f'{sc}_{fr}.png'
        render_bev(pts, boxes, names, npts, args.pc_range, out, title,
                   min_points=args.min_points)
        print('salvato', out)


if __name__ == '__main__':
    main()