import csv
import time
from pathlib import Path
from typing import Dict, Optional

#scrive su stdout e su due file csv per poter plottare successivamente

class TrainingLogger:

    def __init__(self, output_dir: Path, log_every_n_steps: int = 50):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_every_n_steps = log_every_n_steps

        self.step_log_path = self.output_dir / "step_log.csv"
        self.epoch_log_path = self.output_dir / "epoch_log.csv"

        self.step_writer = None
        self.epoch_writer = None
        self.step_file = None
        self.epoch_file = None
        
        self.step_keys = None
        self.epoch_keys = None

        #conserviamo tutte le righe scritte cosi' da poter riscrivere il csv se compaiono chiavi nuove (es. epoche con validation vs senza)
        self.step_rows = []
        self.epoch_rows = []

        self.start_time = time.time()


    def ensure_step_writer(self, log_dict: Dict):
        if self.step_writer is None:
            self.step_keys = ["epoch", "global_step", "lr_backbone", "lr_head"] + list(log_dict.keys())
            self.step_file = open(self.step_log_path, "w", newline="")
            self.step_writer = csv.DictWriter(self.step_file, fieldnames=self.step_keys)
            self.step_writer.writeheader()
    
    def ensure_epoch_writer(self, log_dict: Dict):
        if self.epoch_writer is None:
            self.epoch_keys = ["epoch", "elapsed_time_s"] + list(log_dict.keys())
            self.epoch_file = open(self.epoch_log_path, "w", newline="")
            self.epoch_writer = csv.DictWriter(self.epoch_file, fieldnames=self.epoch_keys)
            self.epoch_writer.writeheader()

    def _rewrite_step_csv(self):
        #se compaiono chiavi nuove riscriviamo l'intero csv con tutte le colonne
        self.step_file.close()
        self.step_file = open(self.step_log_path, "w", newline="")
        self.step_writer = csv.DictWriter(self.step_file, fieldnames=self.step_keys)
        self.step_writer.writeheader()
        for r in self.step_rows:
            self.step_writer.writerow(r)
        self.step_file.flush()

    def _rewrite_epoch_csv(self):
        #se compaiono chiavi nuove riscriviamo l'intero csv con tutte le colonne
        self.epoch_file.close()
        self.epoch_file = open(self.epoch_log_path, "w", newline="")
        self.epoch_writer = csv.DictWriter(self.epoch_file, fieldnames=self.epoch_keys)
        self.epoch_writer.writeheader()
        for r in self.epoch_rows:
            self.epoch_writer.writerow(r)
        self.epoch_file.flush()

    def log_step(
        self,
        epoch: int,
        global_step: int,
        log_dict: Dict[str, float],
        lr_backbone: float,
        lr_head: float,
    ):
        self.ensure_step_writer(log_dict)
        
        row = {"epoch": epoch, "global_step": global_step, "lr_backbone": lr_backbone, "lr_head": lr_head}
        row.update(log_dict)

        #se ci sono chiavi nuove le aggiungiamo alle colonne e riscriviamo il csv
        new_keys = [k for k in row.keys() if k not in self.step_keys]
        if new_keys:
            self.step_keys = self.step_keys + new_keys
            self.step_rows.append(row)
            self._rewrite_step_csv()
        else:
            self.step_rows.append(row)
            self.step_writer.writerow(row)
            self.step_file.flush()

        if global_step % self.log_every_n_steps == 0:
            metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in log_dict.items())
            print(f"[Step {global_step:6d} | Ep {epoch:3d}] {metrics_str} | lr_bb={lr_backbone:.2e} lr_hd={lr_head:.2e}")



    def log_epoch(self, epoch: int, log_dict: Dict[str, float]):
        self.ensure_epoch_writer(log_dict)
        elapsed = time.time() - self.start_time
        row = {"epoch": epoch, "elapsed_time_s": elapsed}
        row.update(log_dict)

        #se ci sono chiavi nuove le aggiungiamo alle colonne e riscriviamo il csv
        new_keys = [k for k in row.keys() if k not in self.epoch_keys]
        if new_keys:
            self.epoch_keys = self.epoch_keys + new_keys
            self.epoch_rows.append(row)
            self._rewrite_epoch_csv()
        else:
            self.epoch_rows.append(row)
            self.epoch_writer.writerow(row)
            self.epoch_file.flush()

        print(f"\n==== Epoch {epoch} Summary =======")
        for k, v in log_dict.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print(f"  elapsed_total: {elapsed:.0f}s")
        print("==============================\n")
    
    def close(self):
        if self.step_file is not None:
            self.step_file.close()
        if self.epoch_file is not None:
            self.epoch_file.close()
