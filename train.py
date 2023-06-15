import config
import torch
from inspect import getmembers
import math as maths
from model import Model, SinusoidalPosEmbedding, TrainablePosEmbedding
from data import get_dataloaders
from tqdm.auto import tqdm
import torch.nn.functional as F
from data import download_zinc20
import os

def get_optimiser(name: str, model: torch.nn.Module, learning_rate: float, weight_decay: float, **kwargs):
    opt = None
    for n, obj in getmembers(torch.optim):
        if name == n:
            opt = obj
            break
    if opt is None:
        raise AttributeError(f"Optimiser {name} doesn't exist")
    
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {'params': [p for p in params if p.dim() >= 2], 'weight_decay': weight_decay},
        {'params': [p for p in params if p.dim() < 2], 'weight_decay': 0.0}
    ]

    if "Adam" in name and "betas" not in kwargs:
        kwargs['betas'] = (0.9, 0.95)
    opt = opt(groups, lr=learning_rate, **kwargs)

    return opt


class LinearWarmupCosineDecay:
    def __init__(self, optimiser, warmup_steps, decay_steps, lr_gamma=0.9, decay_gamma=0.5):
        self.optimiser = optimiser
        
        self.current_step = 1
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.lr_gamma = lr_gamma
        self.decay_gamma = decay_gamma

        self.max_lrs = [pg['lr'] for pg in optimiser.param_groups]

    def get_lr_scale(self):
        if self.current_step < self.warmup_steps:
            return 1 / (self.warmup_steps / self.current_step)
        else:
            return (1 + maths.cos(maths.pi * ((self.current_step - self.warmup_steps) * (1 / self.decay_steps)))) / 2

    def step(self):
        self.current_step += 1
        if self.current_step > (self.warmup_steps + self.decay_steps):
            self.current_step = 1
            if self.lr_gamma:
                self.max_lrs = [lr * self.lr_gamma for lr in self.max_lrs]
            if self.decay_gamma:
                self.decay_steps *= 1 + (1 - self.decay_gamma)

        lr_scale = self.get_lr_scale()
        self.scale_lrs(lr_scale)
    
    def scale_lrs(self, lr_scale):
        for param_group, max_lr in zip(self.optimiser.param_groups, self.max_lrs):
            param_group['lr'] = max_lr * lr_scale

    def set_lrs(self, lr):
        if isinstance(lr, list):
            for param_group, val in zip(self.optimiser.param_groups, lr):
                param_group['lr'] = val
        else:
            for param_group in self.optimiser.param_groups:
                param_group['lr'] = lr


def train():
    set_seed()
    device = "cuda" if torch.cuda.is_available() and config.n_gpu > 0 else "cpu"
    autocast = config.mixed_precision
    accumulation_steps = config.accumulation_steps
    grad_clip = config.grad_clip
    batch_size =config.batch_size
    
    ### Setup

    train_loader, val_loader, vocab, tokens = get_dataloaders(config.dataset_path, 
                                                      train_args=dict(num_workers=config.num_workers, batch_size=batch_size, pin_memory=config.n_gpu > 0))

    # compile?
    model = Model(vocab, 
                  embed_dim=config.embed_dim,
                  num_embeddings=len(vocab),
                  embed_padding_idx=vocab["<pad>"],
                  num_outputs=len(vocab) - 2, # -2 because not predicting pad and mask tokens
                  num_attn_heads=config.num_attn_heads,
                  num_encoder_layers=config.num_encoder_layers,
                  pos_encoding_layer=None,#TrainablePosEmbedding(config.embed_dim, 1000),
                  rotary_embeddings=True).to(device)
    
    print(model)
    print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    model = torch.compile(model, dynamic=True)

    if autocast:
        print("Using mixed precision")
        grad_scaler = torch.cuda.amp.GradScaler()

    opt = config.optimiser
    opt_args = {}
    if "cuda" in device and "Adam" in opt:
        opt_args["fused"] = True
    opt = get_optimiser(opt, model, config.learning_rate, config.weight_decay, **opt_args)
    
    lr_warmup_steps = config.lr_warmup_steps // (batch_size * accumulation_steps)
    scheduler = LinearWarmupCosineDecay(opt, lr_warmup_steps, config.lr_decay_step, lr_gamma=0.8, decay_gamma=0.1)

    if config.use_weighted_loss:
        total = sum(tokens.values())
        weights = torch.FloatTensor([1 - (v / total) for v in tokens.values()] + [1]).to(device)
        print("Using weighted loss")
    else:
        weights = None
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)


    ### Training
    
    n_epochs = config.n_epochs
    best_loss = 1e6
    rounds_since_improvement = 0

    for epoch in range(config.n_epochs):
        ### Train ###
        model.train()
        running_loss = 0

        progress = tqdm(enumerate(train_loader, 1), total=len(train_loader))
        for batch_idx, batch in progress:
            with torch.autocast(device, enabled=autocast):
                loss = model.step(*batch, loss_fn, device) / accumulation_steps
                running_loss += loss.item()
                if autocast:
                    grad_scaler.scale(loss).backward()
                else:
                    loss.backward()

            if batch_idx % accumulation_steps == 0:
                if grad_clip:
                    if autocast: grad_scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if autocast:
                    grad_scaler.step(opt)
                    grad_scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                scheduler.step()
                progress.set_description(f"Epoch {epoch}/{n_epochs} | Train | Loss={running_loss / batch_idx:.4f}")
        opt.zero_grad(set_to_none=True)
        progress.close()

        ### EVAL ###
        model.eval()
        running_loss = 0
        with torch.no_grad():
            progress = tqdm(enumerate(val_loader, 1), total=len(val_loader))
            for batch_idx, batch in progress:
                with torch.autocast(device, enabled=autocast):
                    loss = model.step(*batch, loss_fn, device) / accumulation_steps
                    running_loss += loss.item()
                    progress.set_description(f"Epoch {epoch}/{n_epochs} | Val | Loss={running_loss / batch_idx:.4f}")
            progress.close()
        
        val_loss = running_loss / batch_idx
        if val_loss < best_loss:
            best_loss = val_loss
            rounds_since_improvement = 0
            print(f"\033[92mNew best loss: {val_loss:.4f}\033[0m")
            if config.model_path:
                save_model(model)
        else:
            rounds_since_improvement += 1
            print(f"Loss not improved for {rounds_since_improvement} epochs")
            if config.early_stopping_rounds and rounds_since_improvement > config.early_stopping_rounds:
                print("Stopping")
                break
        print()


def save_model(model):
    model_path = config.model_path
    # if not model_path.endswith(".pt") or model_path.endswith(".pth"):
    #     model_path = os.path.join(model_path, "model.pt")
    if config.save_weights_only:
        print("Saving weights to", model_path)
        torch.save(model.state_dict(), model_path)
    else:
        print("Saving model to", model_path)
        torch.save(model, model_path)

def set_seed(seed=42):
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # call with offset for ddp
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    # download_zinc20(config.dataset_path, config.dataset_path, 1)
    train()
