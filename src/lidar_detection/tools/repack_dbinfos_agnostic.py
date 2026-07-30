"""
repack_dbinfos_single_class.py -- ricompatta il gt_database in UNA sola classe 'cone',
con filtro opzionale sul NUMERO DI PUNTI per costruire un db "sparse" (coni radi).

Perche' il filtro
-----------------
Il gt_sampling pesca uniformemente dal db: se il db e' quello naturale, incolla
soprattutto coni PIENI (dominano). Per usare il gt_sampling a favore dei coni RADI
(l'unica fascia con margine: 1-2 punti), si costruisce un db di soli coni radi e si
incolla quello. I .bin del gt_database NON vengono toccati (restano referenziati).

USO
---
  # db completo a classe singola (come prima):
  python repack_dbinfos_single_class.py --in .../cone_dbinfos_train.pkl \
      --out .../cone_dbinfos_train_cone.pkl

  # db SPARSE (solo coni con 2-6 punti) per il gt_sampling mirato:
  python repack_dbinfos_single_class.py --in .../cone_dbinfos_train.pkl \
      --out .../cone_dbinfos_train_sparse.pkl --min-points 2 --max-points 6
"""
import argparse
import pickle

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--class-name', default='cone')
    ap.add_argument('--min-points', type=int, default=0, help='tieni solo coni con >= N punti')
    ap.add_argument('--max-points', type=int, default=10 ** 9, help='tieni solo coni con <= N punti')
    args = ap.parse_args()

    db = pickle.load(open(args.inp, 'rb'))
    if not isinstance(db, dict):
        raise SystemExit(f'formato inatteso: {type(db)} (mi aspetto un dict classe->lista)')

    merged, per_class, kept_pts = [], {}, []
    for cls, entries in db.items():
        per_class[cls] = len(entries)
        for e in entries:
            npts = int(e.get('num_points_in_gt', 0))
            if npts < args.min_points or npts > args.max_points:
                continue
            e = dict(e)                    # copia difensiva
            e['name'] = args.class_name    # coerenza col gruppo del sampler
            merged.append(e)
            kept_pts.append(npts)

    with open(args.out, 'wb') as f:
        pickle.dump({args.class_name: merged}, f)

    kp = np.array(kept_pts) if kept_pts else np.zeros(0)
    print('classi in ingresso:', per_class)
    print(f'filtro punti: [{args.min_points}, {args.max_points}]')
    line = f"-> '{args.class_name}': {len(merged)} coni tenuti"
    if len(kp):
        line += f' | punti tenuti: min {kp.min()}  mediana {int(np.median(kp))}  max {kp.max()}'
    print(line)
    print('scritto:', args.out)
    print(f"\nNel dataset yaml: DB_INFO_PATH -> [{args.out.split('/')[-1]}], SAMPLE_GROUPS -> ['{args.class_name}:N']")


if __name__ == '__main__':
    main()