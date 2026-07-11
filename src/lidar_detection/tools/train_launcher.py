import sys
from pathlib import Path
import wandb

openpcdet_root = "/workspace/BEV-fusion-sw/lib/OpenPCDet"
openpcdet_tools = "/workspace/BEV-fusion-sw/lib/OpenPCDet/tools"

sys.path.insert(0, openpcdet_tools)
sys.path.insert(0, openpcdet_root)

# 1. INIZIALIZZAZIONE WANDB (Come da doc ufficiale, ma con il trucco TensorBoard)
run = wandb.init(
    entity="andrewboa-universit-degli-studi-di-trento",  # Il tuo team/utente
    project="thesis",            # Il nome generale del progetto
    name="CenterPoint_3Classes_Batch24",                 # Il nome di QUESTA specifica run
    sync_tensorboard=True,                               # FONDAMENTALE: ruba i log a OpenPCDet
    config={
        "architecture": "CenterPoint",
        "dataset": "CARLA Cones (3 Classes)",
        "batch_size": 24,
        "epochs": 80
    }
)

import lidar_detection.datasets
from train import main

if __name__ == '__main__':
    try:
        # Avvia l'addestramento massivo di OpenPCDet
        main()
    finally:
        # 2. CHIUSURA WANDB (Come da doc ufficiale)
        # Il blocco 'finally' assicura che i dati vengano inviati 
        # anche se stoppi l'addestramento a metà con Ctrl+C
        wandb.finish()