import argparse
import time

import torch

import lidar_detection.datasets

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--measure', type=int, default=30)
    args = ap.parse_args()

    cfg_from_yaml_file(args.cfg, cfg)
    logger = common_utils.create_logger()

    dataset, loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size, dist=False, workers=args.workers,
        logger=logger, training=True,
    )

    it = iter(loader)
    for _ in range(args.warmup):   
        next(it)
    t0 = time.time()
    for _ in range(args.measure):
        next(it)
    load_ms = (time.time() - t0) / args.measure * 1000
    print(f'\n[A] solo dataloader : {load_ms:7.1f} ms/batch '
          f'({1000/load_ms:5.1f} batch/s, workers={args.workers})')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.cuda(); model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    import copy
    raw = next(iter(loader))

    def gpu_step():
        batch = copy.deepcopy(raw)
        load_data_to_gpu(batch)
        optimizer.zero_grad()
        ret, _, _ = model(batch)
        ret['loss'].backward()
        optimizer.step()

    for _ in range(args.warmup):     
        gpu_step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.measure):
        gpu_step()
    torch.cuda.synchronize()             
    gpu_ms = (time.time() - t0) / args.measure * 1000
    print(f'[B] solo step GPU   : {gpu_ms:7.1f} ms/iter '
          f'({1000/gpu_ms:5.1f} iter/s)')

    it = iter(loader)
    for _ in range(args.warmup):
        b = next(it); load_data_to_gpu(b)
        optimizer.zero_grad(); r, _, _ = model(b); r['loss'].backward(); optimizer.step()
    torch.cuda.synchronize()
    t0 = time.time()
    n = 0
    for _ in range(args.measure):
        b = next(it); load_data_to_gpu(b)
        optimizer.zero_grad(); r, _, _ = model(b); r['loss'].backward(); optimizer.step()
        n += 1
    torch.cuda.synchronize()
    real_ms = (time.time() - t0) / n * 1000
    print(f'[C] loop reale      : {real_ms:7.1f} ms/iter '
          f'({1000/real_ms:5.1f} iter/s)')


    ratio = gpu_ms / load_ms
    print(f'\n  rapporto throughput loader/GPU = {ratio:.1f}x')
    if load_ms < gpu_ms / 5:
        print('  OTTIMO: il loader e\' >5x piu\' veloce della GPU, il prefetch lo nasconde.')
    elif load_ms < gpu_ms:
        print('  OK: il loader sta sotto il tempo GPU, ma con poco margine.')
    else:
        print('  COLLO DI BOTTIGLIA: il loader e\' piu\' lento della GPU.')
        print('  -> alza --workers, oppure riduci il costo di augment/voxelizzazione,')
        print('     oppure verifica che il disco non sia la fonte (cache fredda).')
    overhead = (real_ms - gpu_ms) / gpu_ms * 100
    print(f'  overhead del loop reale vs solo-GPU: {overhead:+.0f}%  '
          f'(vicino a 0 = caricamento ben nascosto)')


if __name__ == '__main__':
    main()