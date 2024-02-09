import optuna
from dataloader import get_dataloaders
from models import BindingX
import torch
from torch import nn
from tqdm.auto import tqdm
from train import train_model, evaluate_model
import pickle
from pathlib import Path
import json


def objective(trial, train_loader, val_loader, device):
    torch.manual_seed(42)
    
    # n_layers = trial.suggest_int("n_layers", 5, 25, step=5)
    # input_attn_heads = trial.suggest_int("input_attn_heads", 4, 20, step=2)
    # self_attn_heads = trial.suggest_int("self_attn_heads", 4, 20, step=2)
    # dim_feedforward_scale = trial.suggest_int("suggest_categorical", 1, 4)
    # dropout = trial.suggest_float("dropout", 0, 0.3, step=0.1)

    n_layers = trial.suggest_int("n_layers", 1, 3)
    input_attn_heads = trial.suggest_int("input_attn_heads", 4, 20, step=2)
    self_attn_heads = trial.suggest_int("self_attn_heads", 4, 20, step=2)
    dim_feedforward_scale = trial.suggest_int("suggest_categorical", 1, 2)
    dropout = trial.suggest_float("dropout", 0, 0.3, step=0.1)


    model = BindingX(1280, n_layers, input_attn_heads, self_attn_heads, 512, dim_feedforward_scale, dropout)

    loss = train_model(model, train_loader, val_loader, device, epochs=3, autocast=False) # change epochjs
    return loss


def collate_fn(batch):
    proteins = [p[0] for p in batch]
    ligands = [p[1] for p in batch]
    target = [p[2] for p in batch]

    def col(seqs):
        max_len = max(len(x) for x in seqs)
        padding = torch.zeros(len(seqs), max_len)
        for i, p in enumerate(seqs):
            pad_len = max_len - len(p)
            seqs[i] = torch.nn.functional.pad(p, (0, 0, 0, pad_len))
            if pad_len:
                padding[i, -pad_len:] = 1
        return torch.stack(seqs, dim=0), padding

    return col(proteins), col(ligands), torch.FloatTensor(target)



def main():
    (train_loader, val_loader, test_loader), normalisation = get_dataloaders(csv_path="/DATA/binding_data.tsv", target_col=target_col, proteins_dir="/DATA/binding/targets/esm2_t33_650M_UR50D/per_tok/", 
                                       ligands_file="/DATA/binding/ligands/train.lmdb/",
                                         concat=False,
                                                train=True, val=True, test=True, 
                                               train_loader_args=dict(batch_size=8, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn),
                                               val_loader_args=dict(batch_size=8, collate_fn=collate_fn),
                                               test_loader_args=dict(batch_size=8, collate_fn=collate_fn))


    device = "cuda"
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(multivariate=True, seed=42))
    study.optimize(lambda trial: objective(trial, train_loader, val_loader, device), n_trials=5) # change

    
    model = BindingX(1280, **study.best_params, kdim=512)
    out_dir = Path("/DATA/binding")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_model(model, train_loader, val_loader, "cuda", weights_path=out_dir / "binding_x.pt")
    mae, r2 = evaluate_model(model, test_loader, normalisation, device)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"r2": r2, "mae": mae}, f)
    with open(out_dir / "normalisation.pkl", "wb") as f:
        pickle.dump(normalisation, f)
    with open(out_dir / "study.pkl", "wb") as f:
        pickle.dump(study, f)


if __name__ == "__main__":
    target_col = "IC50 (nM)"
    main()
