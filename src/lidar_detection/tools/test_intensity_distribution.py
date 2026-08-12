import argparse
import os
import pickle
import sys

import numpy as np

CLASSES = ["blue", "yellow", "orange_small"]


# --------------------------------------------------------------------------- #
# util
# --------------------------------------------------------------------------- #
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve_path(p, root):
    if os.path.isabs(p):
        return p
    return os.path.join(root, p)


def inspect(dbinfos):
    print("=== struttura cone_dbinfos ===")
    if not isinstance(dbinfos, dict):
        print(f"ATTENZIONE: mi aspettavo un dict, trovato {type(dbinfos)}")
        return
    for cls, lst in dbinfos.items():
        print(f"\nclasse '{cls}': {len(lst)} oggetti")
        if len(lst):
            ex = lst[0]
            print("  chiavi entry:", list(ex.keys()))
            for k, v in ex.items():
                if isinstance(v, np.ndarray):
                    print(f"    {k}: ndarray shape={v.shape} dtype={v.dtype}")
                else:
                    print(f"    {k}: {v!r}")


# --------------------------------------------------------------------------- #
# metriche di separabilita' (1D)
# --------------------------------------------------------------------------- #
def ks_statistic(a, b):
    """Statistica KS a due campioni (max distanza tra le CDF empiriche)."""
    a = np.sort(a)
    b = np.sort(b)
    grid = np.concatenate([a, b])
    grid.sort()
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def hist_overlap(a, b, bins=60):
    """Coefficiente di sovrapposizione (histogram intersection) in [0,1].
    1 = identiche, 0 = disgiunte."""
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if hi <= lo:
        return 1.0
    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(a, bins=edges, density=False)
    hb, _ = np.histogram(b, bins=edges, density=False)
    ha = ha / ha.sum() if ha.sum() else ha
    hb = hb / hb.sum() if hb.sum() else hb
    return float(np.minimum(ha, hb).sum())


def best_threshold_bal_acc(a, b):
    """Miglior accuratezza bilanciata di un classificatore a soglia singola che
    separa 'a' (classe 0) da 'b' (classe 1). ~0.5 => non separabile."""
    vals = np.unique(np.concatenate([a, b]))
    if vals.size < 2:
        return 0.5
    mids = (vals[:-1] + vals[1:]) / 2.0
    best = 0.5
    for t in mids:
        # direzione 1: a<t, b>=t
        tpr_a = np.mean(a < t)
        tpr_b = np.mean(b >= t)
        best = max(best, 0.5 * (tpr_a + tpr_b))
        # direzione 2: a>=t, b<t
        tpr_a = np.mean(a >= t)
        tpr_b = np.mean(b < t)
        best = max(best, 0.5 * (tpr_a + tpr_b))
    return float(best)


def summarize(name, x):
    q = np.percentile(x, [5, 25, 50, 75, 95])
    print(
        f"  {name:14s} n={x.size:>9d}  mean={x.mean():8.4f}  std={x.std():8.4f}  "
        f"p5={q[0]:.4f} p25={q[1]:.4f} p50={q[2]:.4f} p75={q[3]:.4f} p95={q[4]:.4f}"
    )


# --------------------------------------------------------------------------- #
# estrazione intensity
# --------------------------------------------------------------------------- #
def collect_intensity(dbinfos, root, num_features, intensity_col, max_per_class):
    """Ritorna due dict: intensita' a livello di punto e media per-cono, per classe."""
    point_level = {c: [] for c in CLASSES}
    obj_mean = {c: [] for c in CLASSES}
    missing = {c: 0 for c in CLASSES}

    for cls in CLASSES:
        if cls not in dbinfos:
            print(f"ATTENZIONE: classe '{cls}' assente dal dbinfos")
            continue
        entries = dbinfos[cls]
        if max_per_class and len(entries) > max_per_class:
            idx = np.random.RandomState(0).choice(len(entries), max_per_class, replace=False)
            entries = [entries[i] for i in idx]
        for e in entries:
            binp = resolve_path(e["path"], root)
            if not os.path.exists(binp):
                missing[cls] += 1
                continue
            raw = np.fromfile(binp, dtype=np.float32)
            if raw.size % num_features != 0:
                missing[cls] += 1
                continue
            pts = raw.reshape(-1, num_features)
            if pts.shape[0] == 0 or intensity_col >= pts.shape[1]:
                missing[cls] += 1
                continue
            inten = pts[:, intensity_col]
            point_level[cls].append(inten)
            obj_mean[cls].append(float(inten.mean()))

    point_level = {c: (np.concatenate(v) if v else np.array([])) for c, v in point_level.items()}
    obj_mean = {c: np.array(v) for c, v in obj_mean.items()}
    return point_level, obj_mean, missing


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def plot_hists(data, title, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as ex:  # pragma: no cover
        print(f"(matplotlib non disponibile: {ex}; salto i grafici)")
        return
    colors = {"blue": "tab:blue", "yellow": "gold", "orange_small": "tab:orange"}
    plt.figure(figsize=(8, 5))
    allv = np.concatenate([v for v in data.values() if v.size])
    lo, hi = np.percentile(allv, [0.5, 99.5])
    edges = np.linspace(lo, hi, 60)
    for cls, v in data.items():
        if v.size:
            plt.hist(v, bins=edges, density=True, alpha=0.5,
                     label=f"{cls} (n={v.size})", color=colors.get(cls))
    plt.xlabel("intensity")
    plt.ylabel("densita'")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close()
    print(f"  salvato: {out_png}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dbinfos", required=True, help="cone_dbinfos_train.pkl")
    ap.add_argument("--root", default=None, help="root per risolvere i path .bin (default: cartella del dbinfos)")
    ap.add_argument("--num-features", type=int, default=4)
    ap.add_argument("--intensity-col", type=int, default=3)
    ap.add_argument("--max-per-class", type=int, default=0, help="sottocampiona N coni per classe (0=tutti)")
    ap.add_argument("--out", default="./diag_out_0a")
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()

    dbinfos = load_pickle(args.dbinfos)
    if args.inspect:
        inspect(dbinfos)
        return

    root = args.root or os.path.dirname(os.path.abspath(args.dbinfos))
    os.makedirs(args.out, exist_ok=True)

    point_level, obj_mean, missing = collect_intensity(
        dbinfos, root, args.num_features, args.intensity_col, args.max_per_class
    )

    print("\n=== conteggi ===")
    for c in CLASSES:
        n_obj = obj_mean[c].size
        n_pts = point_level[c].size
        print(f"  {c:14s} coni={n_obj:>7d}  punti={n_pts:>10d}  file mancanti/skip={missing[c]}")

    print("\n=== intensity per PUNTO ===")
    for c in CLASSES:
        if point_level[c].size:
            summarize(c, point_level[c])

    print("\n=== intensity MEDIA per cono (segnale piu' pulito) ===")
    for c in CLASSES:
        if obj_mean[c].size:
            summarize(c, obj_mean[c])

    # separabilita' (per-cono, che e' cio' che conta per classificare un cono)
    def report_pair(a_name, b_name):
        a, b = obj_mean[a_name], obj_mean[b_name]
        if a.size == 0 or b.size == 0:
            print(f"  {a_name} vs {b_name}: dati insufficienti")
            return
        ks = ks_statistic(a, b)
        ov = hist_overlap(a, b)
        acc = best_threshold_bal_acc(a, b)
        verdict = "SEPARABILE" if acc >= 0.65 else ("dubbio" if acc >= 0.57 else "NON separabile (~coin flip)")
        print(f"  {a_name:12s} vs {b_name:12s}  KS={ks:.3f}  overlap={ov:.3f}  "
              f"best-1D-bal-acc={acc:.3f}  -> {verdict}")

    print("\n=== SEPARABILITA' (media per cono) ===")
    report_pair("blue", "yellow")          # <-- la domanda chiave
    report_pair("blue", "orange_small")
    report_pair("yellow", "orange_small")

    print("\n=== grafici ===")
    plot_hists(point_level, "Intensity per punto", os.path.join(args.out, "intensity_point_level.png"))
    plot_hists(obj_mean, "Intensity media per cono", os.path.join(args.out, "intensity_obj_mean.png"))

    print("\nInterpretazione: se blue-vs-yellow ha best-1D-bal-acc ~0.5 e overlap alto,")
    print("il colore non e' nel LiDAR -> branch single-class per il feature extractor.")


if __name__ == "__main__":
    sys.exit(main())