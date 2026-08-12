import argparse
import copy

import torch

import lidar_detection.datasets   #

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--iters', type=int, default=300)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()

    cfg_from_yaml_file(args.cfg, cfg)
    logger = common_utils.create_logger()

    dataset, loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=0,            
        logger=logger,
        training=True,
    )
    model = build_network(model_cfg=cfg.MODEL,
                          num_class=len(cfg.CLASS_NAMES),
                          dataset=dataset)
    model.cuda()
    model.train()

    raw_batch = next(iter(loader))

    batch = copy.deepcopy(raw_batch)
    load_data_to_gpu(batch)
    ret_dict, tb_dict, _ = model(batch)
    print('\n[FORWARD OK] le shape combaciano. loss iniziale = '
          f'{ret_dict["loss"].item():.4f}')
    print('[componenti]', {k: round(float(v), 4) for k, v in tb_dict.items()
                           if isinstance(v, (int, float)) or hasattr(v, "item")})

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print('\n[OVERFIT] la loss deve crollare verso ~0:')
    for it in range(args.iters):
        batch = copy.deepcopy(raw_batch)
        load_data_to_gpu(batch)
        optimizer.zero_grad()
        ret_dict, tb_dict, _ = model(batch)
        loss = ret_dict['loss']
        loss.backward()
        optimizer.step()
        if it % 25 == 0 or it == args.iters - 1:
            print(f'  iter {it:4d}   loss {loss.item():.4f}')

if __name__ == '__main__':
    main()