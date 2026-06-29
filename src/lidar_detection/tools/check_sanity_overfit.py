"""
sanity_overfit.py  --  Verifica end-to-end: la rete e' montata bene E sa imparare.

Posizione consigliata:  src/cone_detection/tools/sanity_overfit.py

DUE CONTROLLI IN UNO
--------------------
1) FORWARD PASS: costruisce dataset e modello dal config, passa UN batch nella
   rete e calcola la loss. Se la loss si calcola senza errori, le shape combaciano
   end-to-end (output testa <-> target). E' la verifica di "montaggio".
2) OVERFIT DI UN BATCH: ripete forward+backward sullo STESSO batch ~300 volte.
   Una rete sana memorizza pochi esempi: la loss DEVE crollare verso ~0. Se resta
   piatta in alto, c'e' un bug strutturale (stride, assegnazione target, gradienti).
   Costa due minuti e ti risparmia un run di giorni su una rete che non imparerebbe.

PRE-REQUISITI
-------------
- OpenPCDet installato (import pcdet funziona).
- 'lidar_detection' importabile: lancia con  PYTHONPATH=src  dalla radice repo.
- Adapter + config pronti. gt_sampling DISABILITATO (serve il DB, non ancora creato):
  metti 'gt_sampling' in DISABLE_AUG_LIST nel cone_dataset.yaml per questo test.

NOTA SU _BASE_CONFIG_
---------------------
OpenPCDet risolve _BASE_CONFIG_ rispetto alla CWD. Il modo piu' semplice e' lanciare
questo script dalla cartella dei config, oppure passare un --cfg con path assoluto e
mettere in second_centerpoint_cones.yaml un _BASE_CONFIG_ con path assoluto/risolvibile.

USO
---
  PYTHONPATH=src python src/lidar_detection/tools/sanity_overfit.py \
      --cfg src/lidar_detection/configs/second_centerpoint_cones.yaml \
      --iters 300
"""

import argparse
import copy

import torch

# IMPORT FONDAMENTALE: esegue l'iniezione di ConeDataset nel registry di pcdet.
import lidar_detection.datasets   # noqa: F401

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

    # --- Costruzione dataset + modello dal config ---
    dataset, loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=0,                 # sincrono: piu' semplice da debuggare
        logger=logger,
        training=True,
    )
    model = build_network(model_cfg=cfg.MODEL,
                          num_class=len(cfg.CLASS_NAMES),
                          dataset=dataset)
    model.cuda()
    model.train()

    # --- Prendo UN batch e lo CONGELO (stesso input a ogni iterazione) ---
    # Lo tengo su CPU e ne faccio una copia fresca per iterazione: cosi' i dati
    # sono identici ogni volta (niente nuova augmentation) e non ci sono effetti
    # collaterali da chiavi aggiunte/rimosse dal forward.
    raw_batch = next(iter(loader))

    # --- 1) FORWARD PASS: verifica di montaggio ---
    batch = copy.deepcopy(raw_batch)
    load_data_to_gpu(batch)
    ret_dict, tb_dict, _ = model(batch)
    print('\n[FORWARD OK] le shape combaciano. loss iniziale = '
          f'{ret_dict["loss"].item():.4f}')
    print('[componenti]', {k: round(float(v), 4) for k, v in tb_dict.items()
                           if isinstance(v, (int, float)) or hasattr(v, "item")})

    # --- 2) OVERFIT DI UN BATCH ---
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

    print('\nEsito: se la loss e\' scesa nettamente, la rete IMPARA (montaggio ok).')
    print('Se e\' rimasta piatta in alto, c\'e\' un bug strutturale da stanare PRIMA del training.')


if __name__ == '__main__':
    main()