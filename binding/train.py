import optuna
from dataloader import get_dataloaders
from models import binding_lite, BindingX
import torch
from torch import nn
from torch.cuda.amp import GradScaler
from tqdm.auto import tqdm
from sklearn import metrics

def train_model(model, train_loader, val_loader, device, epochs=100, loss_fn=None, weights_path=None, autocast=False):
    model = model.to(device)
    
    opt = torch.optim.AdamW(model.parameters())
    if loss_fn is None:
        loss_fn = nn.SmoothL1Loss()
    best_loss = 9999
    epochs_since_improvement = 0

    if autocast:
        scaler = GradScaler()
    
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, leave=False, desc=f"Train. Epoch {epoch+1}. Best loss: {best_loss:.3f}"):
            opt.zero_grad()
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=autocast):
                if isinstance(model, BindingX):
                    (proteins, protein_padding), (ligands, ligand_padding), target = batch
                    preds = model(proteins.to(device, non_blocking=True), 
                                ligands.to(device, non_blocking=True),
                                protein_padding.to(device, non_blocking=True),
                                ligand_padding.to(device, non_blocking=True))
                else:
                    x, target = batch
                    preds = model(x.to(device))
                    

                loss = loss_fn(preds, target.to(device))
            
            if autocast:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
        opt.zero_grad(set_to_none=True)
        
        total_loss = 0
        model.eval()
        with torch.no_grad():
            for x, y in tqdm(val_loader, leave=False, desc=f"Val. Epoch {epoch+1}. Best loss: {best_loss:.3f}"):
                preds = model(x.to(device))
                loss = loss_fn(preds, y.to(device))
                total_loss += loss.item()
    
        loss = total_loss / len(val_loader)
        if loss < best_loss:
            best_loss = loss
            epochs_since_improvement = 0
            if weights_path is not None:
                torch.save(model.state_dict(), weights_path)
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement > 10:
                break
    
    return best_loss


def evaluate_model(model, data_loader, normalisation, device):
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(data_loader):
            preds = model(x.to(device)).cpu()
            y_true.extend(y.numpy())
            y_pred.extend(preds.numpy())
    
    y_true = normalisation.inverse_transform(y_pred)
    y_pred = normalisation.inverse_transform(y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    return mae, r2
