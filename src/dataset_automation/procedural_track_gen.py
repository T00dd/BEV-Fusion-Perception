import numpy as np
from shapely.geometry import Polygon, Point, LineString


def _resample_by_arclength(xy, ds):
    seg = np.diff(xy, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    s = np.concatenate([[0], np.cumsum(seglen)])
    total = s[-1]
    if total < 1e-6:
        return xy
    n = max(2, int(total / ds))
    s_new = np.linspace(0, total, n)
    x = np.interp(s_new, s, xy[:, 0])
    y = np.interp(s_new, s, xy[:, 1])
    return np.column_stack([x, y])


def _principal_axes(poly):
    
    pts = np.array(poly.exterior.coords[:-1])  # drop repeated last point
    c = pts.mean(axis=0)
    u, s, vt = np.linalg.svd(pts - c)
    long_axis = vt[0] / np.linalg.norm(vt[0])
    short_axis = vt[1] / np.linalg.norm(vt[1])
    # Extent of the polygon along each axis (half-widths)
    proj_long = (pts - c) @ long_axis
    proj_short = (pts - c) @ short_axis
    half_long = (proj_long.max() - proj_long.min()) / 2.0
    half_short = (proj_short.max() - proj_short.min()) / 2.0
    return c, long_axis, short_axis, half_long, half_short


def generate_serpentine_in_polygon(
    corners, track_width=3.5, cone_spacing=4.0,
    edge_margin=2.5, seed=None,
    lobes_range=(3, 6), amp_fill_range=(0.6, 0.9),
    length_fill=0.9, start_side=None, missing_cone_ratio=0.2,
):
    
    rng = np.random.default_rng(seed)
    poly = Polygon([(float(x), float(y)) for (x, y) in corners])
    safe = poly.buffer(-edge_margin)        # erode: hard containment region
    if safe.is_empty:
        raise RuntimeError("Polygon too small for the requested edge_margin.")

    c, long_axis, short_axis, half_long, half_short = _principal_axes(poly)

    # --- Build the centreline in (long, short) local coordinates -------------
    n_lobes = int(rng.integers(lobes_range[0], lobes_range[1] + 1))
    amp_fill = rng.uniform(*amp_fill_range)
    phase = rng.uniform(0, 2 * np.pi)
    env_peak = 1.30
    # amplitude along the short axis, capped so even the envelope peak fits
    amp = (half_short - edge_margin - track_width / 2.0) * amp_fill / env_peak
    amp = max(0.0, amp)

    span = (half_long - edge_margin - track_width / 2.0) * length_fill
    n = 600
    t = np.linspace(-span, span, n)         # coordinate along the long axis
    env = 1.0 + 0.3 * np.sin(phase + rng.uniform(2.0, 4.0) * np.pi *
                             (t + span) / (2 * span + 1e-9))
    osc = amp * env * np.sin(n_lobes * np.pi * (t + span) / (2 * span + 1e-9)
                             + phase)

    if start_side is None:
        start_side = rng.choice(["low", "high"])
    if start_side == "high":
        t = t[::-1].copy()
        osc = osc[::-1].copy()

    # Map local (long=t, short=osc) back to world coordinates
    center = c + np.outer(t, long_axis) + np.outer(osc, short_axis)

    # --- Offset into left/right cones ----------------------------------------
    center = _resample_by_arclength(center, cone_spacing)
    d = np.gradient(center, axis=0)
    tang = d / (np.hypot(d[:, 0], d[:, 1])[:, None] + 1e-9)
    normal = np.column_stack([-tang[:, 1], tang[:, 0]])
    half = track_width / 2.0
    left = center + normal * half
    right = center - normal * half

    # --- HARD CLIP: keep only cone PAIRS where BOTH sides are inside ----------
    cones_left, cones_right = [], []
    for L, R in zip(left, right):
        if safe.contains(Point(L)) and safe.contains(Point(R)):
            cones_left.append(L)
            cones_right.append(R)
    cones_left = np.array(cones_left)
    cones_right = np.array(cones_right)

    if len(cones_left) < 4:
        raise RuntimeError(
            "Track collapsed after clipping (too few cones inside). "
            "Reduce edge_margin/track_width or amp_fill."
        )

    # --- Missing cones: drop a random fraction independently per side --------
    if missing_cone_ratio > 0:
        cones_left = _drop_random(cones_left, missing_cone_ratio, rng)
        cones_right = _drop_random(cones_right, missing_cone_ratio, rng)

    return {
        "cones_left": cones_left,
        "cones_right": cones_right,
        "start_side": str(start_side),
        "n_lobes": n_lobes,
        "track_width": track_width,
    }


def _drop_random(cones, ratio, rng):
    """Randomly remove `ratio` fraction of cones (simulate missing/occluded)."""
    if len(cones) == 0:
        return cones
    keep_mask = rng.random(len(cones)) >= ratio
    if keep_mask.sum() < 2:           # never delete everything
        return cones
    return cones[keep_mask]

if __name__ == "__main__":
    CORNERS = [(-65.3, 111.2), (-30.8, 119.1), (5.0, -24.1), (-30.8, -32.0)]
    poly = Polygon(CORNERS)

    print(f"Arena polygon area = {poly.area:.0f} m2")
    print("Testing 12 seeds: every cone MUST be inside the polygon.\n")
    print(f"{'seed':>4} {'side':>5} {'lobes':>5} {'Lcones':>6} {'Rcones':>6} "
          f"{'wmin':>5} {'wmax':>5} {'allInside':>9}")

    all_ok = True
    for seed in range(12):
        t = generate_serpentine_in_polygon(
            CORNERS, track_width=3.5, seed=seed,
            edge_margin=2.5, missing_cone_ratio=0.3,
        )
        L, R = t["cones_left"], t["cones_right"]
        # verify containment against the RAW polygon (not eroded)
        inside = all(poly.contains(Point(p)) for p in L) and \
                 all(poly.contains(Point(p)) for p in R)
        all_ok = all_ok and inside
        # lane width between paired cones
        m = min(len(L), len(R))
        w = np.hypot(L[:m, 0] - R[:m, 0], L[:m, 1] - R[:m, 1])
        print(f"{seed:>4} {t['start_side']:>5} {t['n_lobes']:>5} "
              f"{len(L):>6} {len(R):>6} {w.min():>5.2f} {w.max():>5.2f} "
              f"{str(inside):>9}")

    print(f"\nAll cones inside the polygon for all seeds: {all_ok}")