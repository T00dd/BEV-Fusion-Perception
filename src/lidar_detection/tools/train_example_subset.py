# subset of the train set to test the overall pipeline and the scheudlers
# a separated set is created to avoid contaminating the train set with test data

import argparse
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits_dir', type=Path, required=True,
                    help='cartella che contiene train.txt')
    ap.add_argument('--source', default='train', help='split di partenza')
    ap.add_argument('--out', default='train_subset', help='nome split in uscita')
    ap.add_argument('--num_scenes', type=int, default=60)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    src = args.splits_dir / f'{args.source}.txt'
    scenes = [ln.strip() for ln in open(src) if ln.strip()]
    random.seed(args.seed)
    pick = sorted(random.sample(scenes, min(args.num_scenes, len(scenes))))

    out = args.splits_dir / f'{args.out}.txt'
    out.write_text('\n'.join(pick) + '\n')
    print(f'{len(pick)} scene campionate da {src.name} -> {out}')
    print('(con ~33 frame/scena sono circa', len(pick) * 33, 'frame)')


if __name__ == '__main__':
    main()