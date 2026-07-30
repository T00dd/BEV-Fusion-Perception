"""
train_launcher.py -- lancia il training OpenPCDet con logging wandb avanzato.

Uso normale (identico a prima):
  python train_launcher.py --cfg_file src/lidar_detection/configs/cone_centerpoint_agnostic.yaml \
      --batch_size 24 --epochs 80

In piu' rispetto alla versione base:
  * Metriche di sistema (GPU%, VRAM, CPU, I/O, temperature): AUTOMATICHE con wandb.init,
    zero codice. Le trovi nella tab "System" della dashboard.
  * wandb.watch: istogrammi di PESI e GRADIENTI di ogni layer (diagnosi vanishing/
    exploding gradient, layer "morti"). Attivo di default; disattiva con WANDB_WATCH=0.
  * Sweep: se lanciato da un agent, gli iperparametri di wandb.config vengono iniettati
    nel cfg via --set (vedi SWEEP_MAP).
  * Final eval: sotto sweep (o con WANDB_FINAL_EVAL=1) logga val/ap_agnostic e affini,
    cosi' lo sweep puo' ottimizzare la metrica giusta.
"""
import os
import sys
import glob

import wandb

OPENPCDET_ROOT = "/workspace/BEV-fusion-sw/lib/OpenPCDet"
OPENPCDET_TOOLS = OPENPCDET_ROOT + "/tools"
sys.path.insert(0, OPENPCDET_TOOLS)
sys.path.insert(0, OPENPCDET_ROOT)
sys.path.insert(0, "/workspace/BEV-fusion-sw/src")   # per lidar_detection e i tools di eval

# --------------------------------------------------------------------------- #
# 1. INIT WANDB (standalone o dentro un agent di sweep)
# --------------------------------------------------------------------------- #
run = wandb.init(
    entity="andrewboa-universit-degli-studi-di-trento",
    project="thesis",
    name=os.environ.get("WANDB_RUN_NAME", "CP_agnostic_basenoise_nogts_minpts2"),
    sync_tensorboard=True,                 # ruba i log TensorBoard di OpenPCDet
    config={
        "architecture": "CenterPoint",
        "dataset": "CARLA Cones (class-agnostic / single 'cone')",
        "classes": 1,
        "gt_sampling": True,
        "min_pts_for_gt": 2,
        "batch_size": 24,
        "epochs": 80,
        "jitter_std_max": 0.03,
        "dropout_max": 0.3,
        "clutter_max": 12,
    },
)

# --------------------------------------------------------------------------- #
# 2. SWEEP: inietta gli iperparametri di wandb.config come override cfg (--set)
#    NB: --set usa nargs=REMAINDER in OpenPCDet -> deve stare IN FONDO a sys.argv.
# --------------------------------------------------------------------------- #
SWEEP_MAP = {
    "lr":             "OPTIMIZATION.LR",
    "loc_weight":     "MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.loc_weight",
    "min_radius":     "MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.MIN_RADIUS",
    "grad_norm_clip": "OPTIMIZATION.GRAD_NORM_CLIP",
    "weight_decay":   "OPTIMIZATION.WEIGHT_DECAY",
}
overrides = []
for k, cfg_key in SWEEP_MAP.items():
    if k in wandb.config.keys():
        overrides += [cfg_key, str(wandb.config[k])]
if overrides:
    if "--set" in sys.argv:                # rispetta un --set gia' passato a mano
        sys.argv += overrides
    else:
        sys.argv += ["--set"] + overrides
    print("[sweep] override cfg:", overrides)

# --------------------------------------------------------------------------- #
# 3. wandb.watch: pesi + gradienti. Patchiamo build_network cosi' NON serve
#    entrare nel loop di OpenPCDet.
# --------------------------------------------------------------------------- #
if os.environ.get("WANDB_WATCH", "1") == "1":
    import pcdet.models as _pm
    _orig_build = _pm.build_network

    def _build_watched(*a, **kw):
        model = _orig_build(*a, **kw)
        try:
            # log_freq alto per non appesantire: istogrammi ogni 200 step
            wandb.watch(model, log="all", log_freq=200, log_graph=False)
            print("[wandb] watch attivo: pesi+gradienti, log_freq=200")
        except Exception as e:
            print("[wandb] watch non attivata:", e)
        return model

    _pm.build_network = _build_watched   # train.py fara' `from pcdet.models import build_network`

import lidar_detection.datasets   # registra ConeDataset
from train import main


# --------------------------------------------------------------------------- #
# 4. FINAL EVAL: sotto sweep, logga la metrica class-agnostic da ottimizzare.
#    Riusa il cfg globale (gia' popolato da train.main) e le funzioni di evaluate.py.
# --------------------------------------------------------------------------- #
def final_eval_and_log():
    import numpy as np
    import torch
    from pcdet.config import cfg
    from pcdet.datasets import build_dataloader
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils
    import evaluate as EV

    # checkpoint piu' recente prodotto dal training
    cands = glob.glob(os.path.join("output", "**", "*.pth"), recursive=True)
    if not cands:
        print("[final_eval] nessun checkpoint trovato, salto")
        return
    ckpt = max(cands, key=os.path.getmtime)
    print("[final_eval] uso checkpoint:", ckpt)

    logger = common_utils.create_logger()
    dataset, loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=8, dist=False, workers=4, logger=logger, training=False)
    merge = getattr(dataset, "merge_to", None)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=False)
    model.cuda(); model.eval()

    frames = []
    with torch.no_grad():
        for batch in loader:
            load_data_to_gpu(batch)
            preds, _ = model(batch)
            for i, pd in enumerate(preds):
                pb = pd["pred_boxes"].cpu().numpy()
                ps = pd["pred_scores"].cpu().numpy()
                scene, fr = batch["frame_id"][i].rsplit("_", 2)[0], "_".join(batch["frame_id"][i].rsplit("_", 2)[1:])
                gb, gn, gnp = dataset.get_label(scene, fr)
                frames.append({
                    "frame_id": batch["frame_id"][i],
                    "pred_boxes": pb.reshape(-1, 7), "pred_scores": ps,
                    "gt_boxes": gb.reshape(-1, 7), "gt_npts": gnp.astype(int),
                    "pred_xy": pb.reshape(-1, 7)[:, :2], "gt_xy": gb.reshape(-1, 7)[:, :2],
                })

    ap = EV.ap_band(frames, 0.5)
    ap_far = EV.ap_band(frames, 0.5, (30, 50))
    s, rec, prec, f1, ngt = EV.global_pr_curve(frames, 0.5)
    op = float(s[int(np.argmax(f1))]) if len(f1) else 0.1
    R = EV.collect_operating(frames, 0.5, op)
    recall = len(R["tp"]) / max(R["n_gt"], 1)
    fp_pf = len(R["fp"]) / max(len(frames), 1)
    wandb.log({
        "val/ap_agnostic": ap,
        "val/ap_agnostic_30_50m": ap_far,
        "val/recall_at_op": recall,
        "val/fp_per_frame": fp_pf,
        "val/op_score_thresh": op,
    })
    print(f"[final_eval] ap_agnostic={ap:.3f} ap_far={ap_far:.3f} recall={recall:.3f} fp/frame={fp_pf:.2f}")


if __name__ == '__main__':
    try:
        main()
        do_final = (run.sweep_id is not None) or os.environ.get("WANDB_FINAL_EVAL") == "1"
        if do_final:
            try:
                final_eval_and_log()
            except Exception as e:
                print("[final_eval] fallita (non blocca il training):", e)
    finally:
        wandb.finish()