import os
import local_data
from dataclasses import dataclass

class Config:
    seed = 42

    expt_name = local_data.expt_name

    # Model
    embed_dim=512
    num_attn_heads=8
    num_encoder_layers=10
    pos_layer = "rotary"

    # Training
    n_gpu = 2
    effective_batch_size = 2048 # actual batch size (per gpu) = effective_batch_size / num_gpus / accumulation_steps
    accumulation_steps = 1 # how many times to call .backward before .step
    eval_batch_size = 256
    mixed_precision = True
    grad_clip_norm = 1.0
    optimiser = "AdamW"
    learning_rate = 1e-3
    weight_decay = 0.01
    lr_warmup_steps = 100_000
    lr_decay_step = 1_000_000
    use_weighted_loss = True
    early_stopping_rounds = 10

    max_train_updates = 1_000_000 # max total number of optimiser steps to do in training
    eval_every_n_updates = 10_000 # how frequently to run eval

    num_workers_ = 4
    save_weights_only_ = True
    profile_ = False
    lmdb_ = True

    # local_data contains config paths and keys that are not commited to git
    dataset_path_ = local_data.dataset_path # path to folder containing selfies txt files, may include subfolders
    model_path_ = local_data.model_path # where to save the model
    history_save_path_ = local_data.history_save_path # where to save training history
    tensorboard_dir_ = local_data.tensorboard_dir
    tensorboard_dir_ = os.path.join(tensorboard_dir_, expt_name)
    log_wandb_ = True
    wandb_watch_ = False

    @staticmethod
    def export_dict():
        return {k: v for k, v in Config.__dict__.items() if not k.endswith("_")}