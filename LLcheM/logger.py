from torch.utils.tensorboard import SummaryWriter as TBSummaryWriter
from collections import defaultdict
import json
import wandb

class Logger:
    def __init__(self, tensorbard_dir=None, log_wandb=False) -> None:
        if tensorbard_dir:
            self.tensorbard_writer = TBSummaryWriter(log_dir=tensorbard_dir)
        else:
            self.tensorbard_writer = None
        
        self.log_wandb = log_wandb
        self.history = defaultdict(list)

    def log_scalars(self, data: dict, global_step: int, tag: str="train"):
        self.history[tag].append({global_step: data})
        
        if self.tensorbard_writer is not None:
            for k, v in data.items():
                self.tensorbard_writer.add_scalar(f"{k}/{tag}", v, global_step)
        
        if self.log_wandb:
            wandb.log({tag: data}, step=global_step)

    def save_history(self, out_file):
        with open(out_file, "w") as f:
            json.dump(self.history, f)