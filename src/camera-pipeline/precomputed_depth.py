#script per precalcolare offline le depth con SGBM per tutte le scene e frame del dataset, salvandole in file .npy
# vengono salvate nella cartella ../../data/precomputed_depth


import argparse
import json
from pathlib import Path
 
import numpy as np
from PIL import Image
 
from bev_config import BEVConfig
from bev_dataset import load_calib
from stereo_depth import compare_depth, compute_depth_from_stereo


def aggregate(stats_list):
    #media delle statistiche su tutti i frame (nan ignorati)
    if not stats_list:
        return {}
    out = {}
    for k in stats_list[0]:
        vals = [s[k] for s in stats_list if isinstance(s[k], (int, float))
                and np.isfinite(s[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def print_report(stats):
    print("\n" + "=" * 62)
    print("CONFRONTO SGBM vs DEPTH GT CARLA")
    print("=" * 62)
    print(f"  coverage totale : {stats.get('coverage', float('nan'))*100:6.2f} %"  "   <- frazione di pixel con depth valida")
    print(f"  MAE (di quanti metri in media sbaglia): {stats.get('mae_m', float('nan')):6.3f} m")
    print(f"  RMSE (eleva l'errore al quadrato prima della media): {stats.get('rmse_m', float('nan')):6.3f} m")
    print(f"  bias (MAE ma senza valore assoluto): {stats.get('bias_m', float('nan')):+6.3f} m" "   <- positivo = sovrastima la distanza")
    print(f"  entro 0.5 m (percentuale di precisione entro il mezzo metro): {stats.get('pct_within_0.5m', float('nan'))*100:6.2f} %")
    print("\n  per fascia di distanza (l'errore stereo cresce con z^2):")
    print(f"  {'fascia':>10} | {'MAE (m)':>9} | {'coverage':>9}")
    print("  " + "-" * 34)
    for lo, hi in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 50)]:
        mae = stats.get(f"mae_{lo}-{hi}m", float("nan"))
        cov = stats.get(f"coverage_{lo}-{hi}m", float("nan"))
        print(f"  {f'{lo}-{hi}m':>10} | {mae:9.3f} | {cov*100:8.2f}%")
    print("=" * 62)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--splits", type=str, nargs="*",
                        default=["splits/train.txt", "splits/val.txt", "splits/test.txt"],)
    parser.add_argument("--compare", action="store_true",
                        help="Confronta con la depth GT di CARLA e stampa un report")
    parser.add_argument("--dry_run", action="store_true",
                        help="Non salva nulla (utile per tarare i parametri SGBM)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Processa al massimo N frame")
    parser.add_argument("--overwrite", action="store_true",
                        help="Ricalcola anche i frame gia' presenti")
    args = parser.parse_args()

    cfg = BEVConfig()
    root = Path(args.dataset_root) if args.dataset_root else Path(cfg.dataset_root)


    #raccogliamo i frame da tutti gli split
    frames = []
    for split in args.splits:
        split_path = root / split
        if not split_path.is_file():
            print(f"[warn] split non trovato, salto: {split_path}")
            continue
        with open(split_path) as f:
            for scene_id in [l.strip() for l in f if l.strip()]:
                img_dir = root / "scenes" / scene_id / "images"
                if not img_dir.is_dir():
                    continue
                for p in sorted(img_dir.glob("*_cam_left.png")):
                    frames.append((scene_id, p.name.replace("_cam_left.png", "")))


    frames = sorted(set(frames))
    if args.limit:
        frames = frames[:args.limit]
    print(f"Frame da processare: {len(frames)}")
    if not frames:
        return


    all_stats = []
    calib_cache = {}
    skipped = 0


    for i, (scene_id, frame_stem) in enumerate(frames):
        scene_dir = root / "scenes" / scene_id

        out_dir = cfg.depth_dir
        out_path = out_dir / f"{frame_stem}.npy"
        if out_path.is_file() and not args.overwrite and not args.compare:
            skipped += 1
            continue
 
        if scene_id not in calib_cache:
            calib_cache[scene_id] = load_calib(scene_dir / "calib.yaml")
        calib = calib_cache[scene_id]
 
        left_path = scene_dir / "images" / f"{frame_stem}_cam_left.png"
        right_path = scene_dir / "images" / f"{frame_stem}_cam_right.png"
        if not right_path.is_file():
            raise FileNotFoundError(
                f"Immagine destra mancante: {right_path}. "
                f"Senza coppia stereo non si puo' calcolare la depth"
            )
 
        left = np.asarray(Image.open(left_path).convert("RGB"))
        right = np.asarray(Image.open(right_path).convert("RGB"))
 

        fx_native = float(calib["K"][0]) * (left.shape[1] / calib["calib_size"][1])
 
        depth = compute_depth_from_stereo(
            left, right, fx_native, calib["baseline"],
            sgbm_params=cfg.sgbm_params(),
            min_depth_m=cfg.min_depth_m, max_depth_m=cfg.max_depth_m,
        )
 
        if args.compare:
            gt_path = scene_dir / cfg.depth_gt_dir / f"{frame_stem}.npy"
            if gt_path.is_file():
                gt = np.load(gt_path).astype(np.float32)
                if gt.shape == depth.shape:
                    all_stats.append(compare_depth(depth, gt, max_depth_m=cfg.x_max))
                else:
                    print(f"[warn] shape depth GT {gt.shape} != SGBM {depth.shape}, salto confronto")
 
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_path, depth.astype(np.float32))
 
        if (i + 1) % 50 == 0 or i == len(frames) - 1:
            cov = np.mean(depth > 0) * 100
            print(f"  [{i+1}/{len(frames)}] {scene_id}/{frame_stem}  coverage {cov:.1f}%")
 
    if skipped:
        print(f"Saltati {skipped} frame gia' presenti (--overwrite per rifarli)")
 
    if args.compare and all_stats:
        stats = aggregate(all_stats)
        print_report(stats)
        if not args.dry_run:
            report_path = root / "depth_sgbm_report.json"
            with open(report_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"\nReport salvato: {report_path}")
 
    print("\n[Done]")
 
 
if __name__ == "__main__":
    main()



 
