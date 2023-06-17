import config
import torch
from model import Model, SinusoidalPosEmbedding, TrainablePosEmbedding
from data import get_dataloaders
from tqdm.auto import tqdm
from data import download_zinc20
import os
from train_utils import get_optimiser, LinearWarmupCosineDecay

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from contextlib import nullcontext


def step(model, x, y, mask, padding, loss_fn, device):
    x = x.to(device, non_blocking=True)
    y = y.to(device)
    preds = model(x, padding_mask=padding.to(device), prediction_mask=mask.to(device))
    loss = loss_fn(preds, y)
    return loss


def train():
    ### Setup
    if config.n_gpu > 1:
        assert torch.cuda.is_available()
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        print(f"Starting DDP on rank {rank}.")
    else:
        rank = 0
    set_seed(42 + rank)

    device = rank if torch.cuda.is_available() and config.n_gpu > 0 else "cpu"
    is_cuda = device != "cpu"
    device_type = "cuda" if is_cuda else "cpu"
    autocast = config.mixed_precision
    accumulation_steps = config.accumulation_steps
    grad_clip = config.grad_clip
    batch_size = int(config.effective_batch_size / config.n_gpu / config.accumulation_steps)
    print(f"{batch_size=}, {config.effective_batch_size=}, {config.n_gpu=}, {config.accumulation_steps=}")
    
    train_loader, val_loader, vocab, tokens = get_dataloaders(config.dataset_path, 
                                                      train_args=dict(num_workers=config.num_workers, batch_size=batch_size, pin_memory=config.n_gpu > 0),
                                                      test_args=dict(num_workers=8, batch_size=config.eval_batch_size))

    model = Model(vocab, 
                  embed_dim=config.embed_dim,
                  num_embeddings=len(vocab),
                  embed_padding_idx=vocab["<pad>"],
                  num_outputs=len(vocab) - 2, # -2 because not predicting pad and mask tokens
                  num_attn_heads=config.num_attn_heads,
                  num_encoder_layers=config.num_encoder_layers,
                  pos_encoding_layer=None,#TrainablePosEmbedding(config.embed_dim, 1000),
                  rotary_embeddings=True).to(device)
    
    # print(model)
    print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    if config.n_gpu < 2:
        # TODO compile not working with DDP, maybe related to dynamic batch sizes
        model = torch.compile(model, dynamic=True)
    if config.n_gpu > 1:
        model = DDP(model, device_ids=[rank])
        ddp_no_sync = model.no_sync
    else:
        ddp_no_sync = nullcontext

    if autocast:
        print("Using mixed precision")
        grad_scaler = torch.cuda.amp.GradScaler()

    opt = config.optimiser
    opt_args = {}
    if is_cuda and "Adam" in opt:
        opt_args["fused"] = True
    opt = get_optimiser(opt, model, config.learning_rate, config.weight_decay, **opt_args)
    
    lr_warmup_steps = config.lr_warmup_steps# // (batch_size * accumulation_steps)
    scheduler = LinearWarmupCosineDecay(opt, lr_warmup_steps, config.lr_decay_step, lr_gamma=0.8, decay_gamma=0.1)

    if config.use_weighted_loss:
        total = sum(tokens.values())
        weights = torch.FloatTensor([1 - (v / total) for v in tokens.values()] + [1]).to(device)
        print("Using weighted loss")
    else:
        weights = None
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    ### Training
    best_loss = 1e6
    rounds_since_improvement = 0
    train_iterator = iter(train_loader)
    total_iters = 0
    while True:
        
        n_microbatches = (config.eval_every_n_updates * accumulation_steps) // config.n_gpu
        if total_iters + config.eval_every_n_updates > config.max_train_updates:
            n_microbatches = (config.max_train_updates - total_iters) * accumulation_steps

        steps = range(1, n_microbatches + 1)
        if rank == 0:
            print(f"Completed {total_iters}/{config.max_train_updates} Training steps")
            steps = tqdm(steps)
        
        ### TRAIN ###
        model.train()
        running_loss = 0
        iters_since_eval = 0
        for batch_idx in steps:
            batch = next(train_iterator)
            with torch.autocast(device_type=device_type, enabled=autocast):
                loss = step(model, *batch, loss_fn, device) / accumulation_steps
                running_loss += loss.item()
                if autocast:
                    grad_scaler.scale(loss).backward()
                else:
                    loss.backward()
    
            if batch_idx % accumulation_steps == 0:
                # TODO if doing gradient accumulation and DDP, only need to sync models/GPUs here
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

                if rank == 0:
                    steps.set_description(f"Train batch {batch_idx}/{n_microbatches} | Loss={running_loss / batch_idx:.4f}")
                iters_since_eval += 1
                total_iters += 1
                
        opt.zero_grad(set_to_none=True)
        if rank == 0:
            steps.close()
        
        ### EVAL ###

        model.eval()
        # if rank == 0:
        running_loss = 0
        with torch.no_grad(), ddp_no_sync():
            loader = enumerate(val_loader, 1)
            if rank == 0:
                loader = tqdm(loader, total=len(val_loader), desc="Evaluating")
            for batch_idx, batch in loader:
                # no need to sync
                with torch.autocast(device_type=device_type, enabled=autocast):
                    loss = step(model, *batch, loss_fn, device) / accumulation_steps
                    running_loss += loss.item()

        # if last batch is smaller, val_loss will be incorrect, but difference is insignificant        
        val_loss = running_loss / batch_idx
        if config.n_gpu > 1:
            # Average val loss over GPUs
            print(f"Val loss before reduce = {val_loss}. Rank = {rank}")
            val_loss = torch.tensor(val_loss).to(rank)
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            val_loss = val_loss.item() / config.n_gpu
            print(f"Val loss after reduce = {val_loss}. Rank = {rank}")

        if val_loss < best_loss:
            best_loss = val_loss
            rounds_since_improvement = 0
            if rank == 0:
                print(f"\033[92mNew best loss: {val_loss:.4f}\033[0m")
                if config.model_path and rank == 0:
                    save_model(model)
        else:
            rounds_since_improvement += 1
            print(f"Loss not improved for {rounds_since_improvement} epochs")
            if config.early_stopping_rounds and rounds_since_improvement > config.early_stopping_rounds:
                print(f"Stopping ({rank=})")
                break
        if total_iters >= config.max_train_updates:
            print(f"Training finished. Stopping ({rank=})")
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

#  torchrun --standalone --nproc_per_node=2 train.py