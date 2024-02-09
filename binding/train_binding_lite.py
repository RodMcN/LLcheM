import optuna
from dataloader import get_dataloaders
from models import binding_lite
import torch
from train import train_model, evaluate_model
import pickle
from pathlib import Path
import json

def objective(trial, train_loader, val_loader, device):

    torch.manual_seed(42)
    
    hidden_dim_scale = trial.suggest_int("hidden_dim_scale", 0, 5)
    hidden_dim = 64*(2**hidden_dim_scale)
    num_hidden_layers = trial.suggest_int('hidden_layers', 1, 10)
    batch_norm = trial.suggest_categorical("batch_norm", [True, False])
    dropout = trial.suggest_float("dropout", 0, 0.6, step=0.1)
    activation = trial.suggest_categorical("activation", ["Tanh", "ReLU", "PReLU", "Mish", "CELU", "Identity"])
    
    model = binding_lite(1792, hidden_dim, num_hidden_layers, batch_norm, activation, dropout)

    loss = train_model(model, train_loader, val_loader, device)
    return loss


def train():
    (train_loader, val_loader, test_loader), normalisation = get_dataloaders("/DATA/binding_data.tsv", target_col,
                                                "/DATA/binding/targets/esm2_t33_650M_UR50D/mean_repr/",
                                                "/DATA/binding/ligands/bindingdb_embeddings.pt", concat=True,
                                                train=True, val=True, test=True, 
                                               train_loader_args=dict(batch_size=256, shuffle=True, num_workers=2, pin_memory=True),
                                               val_loader_args=dict(batch_size=256),
                                               test_loader_args=dict(batch_size=256))


    device = "cuda"
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(multivariate=True, seed=42))
    study.optimize(lambda trial: objective(trial, train_loader, val_loader, device), n_trials=100)


    # retrain model with best params
    params = study.best_params
    params['hidden_dim'] = 64*(2**params["hidden_dim_scale"])
    params['num_hidden_layers'] = params['hidden_layers']
    del params['hidden_dim_scale'], params['hidden_layers']
    model = binding_lite(1792, **params)

    out_dir = Path("/DATA/binding")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_model(model, train_loader, val_loader, "cuda", weights_path=out_dir / "binding_lite.pt")
    mae, r2 = evaluate_model(model, test_loader, normalisation, device)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"r2": r2, "mae": mae}, f)
    with open(out_dir / "normalisation.pkl", "wb") as f:
        pickle.dump(normalisation, f)
    with open(out_dir / "study.pkl", "wb") as f:
        pickle.dump(study, f)


if __name__ == "__main__":
    target_col = "IC50 (nM)"
    train()
