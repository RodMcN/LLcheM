from config import Config
import torch
from model import Model, SinusoidalPosEmbedding, TrainablePosEmbedding
from data import get_dataloaders
from tqdm.auto import tqdm
from data import download_zinc20
import os
from train_utils import get_optimiser, LinearWarmupCosineDecay
from logger import Logger
import torch.nn.functional as F
import wandb


from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from contextlib import nullcontext


# temporarily here because model.step doesn't work with DDP
# could use model.module.step instead
def step(model, x, y, mask, padding, loss_fn, device, perplexity=False):
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    padding = padding.to(device, non_blocking=True)

    # when prediction_mask is provided to model.forward it only returns predictions for masked elements
    # y contains only labels for masked elements
    preds = model(x, padding_mask=padding, prediction_mask=mask)
    loss = loss_fn(preds, y)
    if perplexity:
        # not true perplexity, but estimate which is averaged over entire test set
        # TODO if loss_fn is cross entrpoy then computation is wasted here by duplicating steps
        # loss_fn may be weighted
        perplexity = torch.exp(F.cross_entropy(preds, y)).item()
        return loss, perplexity
    return loss


def train():
    ### Setup ###
    if Config.n_gpu > 1:
        assert torch.cuda.is_available()
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        print(f"Starting DDP on rank {rank}.")
    else:
        rank = 0
    set_seed(42 + rank)

    if rank == 0:
        logger = Logger(tensorbard_dir=Config.tensorboard_dir_, log_wandb=Config.log_wandb_)

    device = rank if torch.cuda.is_available() and Config.n_gpu > 0 else "cpu"
    is_cuda = device != "cpu"
    device_type = "cuda" if is_cuda else "cpu"
    autocast = Config.mixed_precision
    accumulation_steps = Config.accumulation_steps
    grad_clip = Config.grad_clip
    batch_size = int(Config.effective_batch_size / Config.n_gpu / Config.accumulation_steps)
    print(f"{batch_size=}, {Config.effective_batch_size=}, {Config.n_gpu=}, {Config.accumulation_steps=}")
    
    train_loader, val_loader, vocab, tokens = get_dataloaders(Config.dataset_path_, 
                                                      train_args=dict(num_workers=Config.num_workers_, batch_size=batch_size, pin_memory=Config.n_gpu > 0),
                                                      test_args=dict(num_workers=2, batch_size=Config.eval_batch_size), distributed=Config.n_gpu > 1)

    model = Model(vocab, 
                  embed_dim=Config.embed_dim,
                  num_embeddings=len(vocab),
                  embed_padding_idx=vocab["<pad>"],
                  num_outputs=len(vocab) - 2, # -2 because not predicting pad and mask tokens
                  num_attn_heads=Config.num_attn_heads,
                  num_encoder_layers=Config.num_encoder_layers,
                  pos_encoding_layer=TrainablePosEmbedding(Config.embed_dim, 1000),
                  rotary_embeddings=False).to(device)
    
    # print(model)
    print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    # if Config.n_gpu < 2:
        # TODO compile not working with DDP, maybe related to dynamic batch sizes
        # Also not working in Docker
        # model = torch.compile(model, dynamic=True)
    
    if Config.n_gpu > 1:
        model = DDP(model, device_ids=[rank])
        ddp_no_sync = model.no_sync
    else:
        # if not using DDP, replace model.no_sync with nullcontext
        # makes optional use of no_sync context manager easier
        ddp_no_sync = nullcontext

    if autocast:
        print("Using mixed precision")
        grad_scaler = torch.cuda.amp.GradScaler()

    opt = Config.optimiser
    opt_args = {}
    # Use the fused implementation for Adam and Adam variants
    if is_cuda and "Adam" in opt:
        opt_args["fused"] = True
    opt = get_optimiser(opt, model, Config.learning_rate, Config.weight_decay, **opt_args)
    
    lr_warmup_steps = Config.lr_warmup_steps# // (batch_size * accumulation_steps)
    scheduler = LinearWarmupCosineDecay(opt, lr_warmup_steps, Config.lr_decay_step, lr_gamma=0.8, decay_gamma=0.1)

    if Config.use_weighted_loss:
        total = sum(tokens.values())
        weights = torch.FloatTensor([1 - (v / total) for v in tokens.values()] + [1]).to(device)
        print("Using weighted loss")
    else:
        weights = None
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    ### Training
    best_loss = 1e9
    rounds_since_improvement = 0
    train_iterator = iter(train_loader)
    total_iters = 0

    # Set up profiler
    if Config.profile_ and (rank == 0):
        profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(Config.tensorboard_dir_),
                record_shapes=True,
                with_stack=True)
        profiler.start()
    else:
        profiler = None

    if Config.wandb_watch_ and Config.n_gpu < 2:
        # not tested with DDP
        wandb.watch(model)

    while True:
        # how many minibatches to run before evaluating
        n_minibatches = Config.eval_every_n_updates * accumulation_steps
        if total_iters + Config.eval_every_n_updates > Config.max_train_updates:
            n_minibatches = (Config.max_train_updates - total_iters) * accumulation_steps

        steps = range(1, n_minibatches + 1)
        if rank == 0:
            print(f"Completed {total_iters}/{Config.max_train_updates} Training steps")
            steps = tqdm(steps)
        
        ### TRAIN ###
        model.train()
        running_loss = 0
        for batch_idx in steps:
            batch = next(train_iterator)

            # when accumulating gradients, ony need to sync models/GPUs during model updates
            accumulated = batch_idx % accumulation_steps == 0
            maybe_no_sync = ddp_no_sync if not accumulated else nullcontext
            
            with torch.autocast(device_type=device_type, enabled=autocast), maybe_no_sync():
                loss = step(model, *batch, loss_fn, device) / accumulation_steps
                running_loss += loss.item()
                if autocast:
                    grad_scaler.scale(loss).backward()
                else:
                    loss.backward()
    
            if accumulated:
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
                if profiler is not None:
                    profiler.step()

                if rank == 0:
                    steps.set_description(f"Train batch {batch_idx}/{n_minibatches} | Loss={running_loss / batch_idx:.4f}")
                total_iters += 1
                
        opt.zero_grad(set_to_none=True)
        if rank == 0:
            steps.close()
            logger.log_scalars({
                "loss": running_loss / batch_idx,
                "LR": opt.param_groups[0]['lr']}, total_iters, tag="train")
        
        if profiler is not None:
            profiler.stop()
            profiler = None


        ### EVAL ###
        model.eval()
        running_loss = 0
        running_perplexity = 0
        # no need to sync models during evaluation
        with torch.no_grad(), ddp_no_sync():
            loader = enumerate(val_loader, 1)
            if rank == 0:
                loader = tqdm(loader, total=len(val_loader), desc="Evaluating")
            for batch_idx, batch in loader:
                with torch.autocast(device_type=device_type, enabled=autocast):
                    loss, perplexity = step(model, *batch, loss_fn, device, perplexity=True)
                    loss /= accumulation_steps
                    running_loss += loss.item()

                    running_perplexity += perplexity

        # if last batch is smaller, val_loss will be incorrect, but difference is insignificant        
        val_loss = running_loss / batch_idx
        perplexity = running_perplexity / batch_idx
        if Config.n_gpu > 1:
            # Average val loss over GPUs
            print(f"Val loss before reduce = {val_loss}. Rank = {rank}")
            val_loss = torch.tensor(val_loss).to(rank)
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            val_loss = val_loss.item() / Config.n_gpu
            print(f"Val loss after reduce = {val_loss}. Rank = {rank}")

            perplexity = torch.tensor(perplexity).to(rank)
            dist.all_reduce(perplexity, op=dist.ReduceOp.SUM)
            perplexity = perplexity.item() / Config.n_gpu

        # Log metrics
        if rank == 0:
            logger.log_scalars({"loss": val_loss, "perplexity": perplexity}, total_iters, "val")
            if Config.history_save_path_:
                logger.save_history(Config.history_save_path_)

        # Check if val loss has improved
        if val_loss < best_loss:
            best_loss = val_loss
            rounds_since_improvement = 0
            if rank == 0:
                # Save model if validation loss has improved
                print(f"\033[92mNew best loss: {val_loss:.4f}\033[0m")
                print(f"Perplexity = {perplexity:.3f}")
                if Config.model_path_ and rank == 0:
                    save_model(model)
        else:
            # Early stopping
            rounds_since_improvement += 1
            print(f"Loss not improved for {rounds_since_improvement} epochs")
            if Config.early_stopping_rounds and rounds_since_improvement > Config.early_stopping_rounds:
                print(f"Stopping ({rank=})")
                break
        if total_iters >= Config.max_train_updates:
            print(f"Training finished. Stopping ({rank=})")
            break
        print()
    
    if Config.n_gpu > 1:
        dist.destroy_process_group()


def save_model(model):
    if Config.n_gpu > 1:
        model = model.module
    model_path = Config.model_path_
    # if not model_path.endswith(".pt") or model_path.endswith(".pth"):
    #     model_path = os.path.join(model_path, "model.pt")
    if Config.save_weights_only_:
        print("Saving weights to", model_path)
        model.save_model(model_path)
    else:
        print("Saving model to", model_path)
        torch.save(model, model_path)

def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    train()