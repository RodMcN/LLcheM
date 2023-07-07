import os
import local_data
from dataclasses import dataclass

class Config:
    seed = 42

    expt_name = "test"

    # Model
    embed_dim=256
    num_attn_heads=4
    num_encoder_layers=6

    # Training
    n_gpu = 1
    effective_batch_size = 64 # actual batch size (per gpu) = effective_batch_size / num_gpus / accumulation_steps
    accumulation_steps = 1
    eval_batch_size = 64
    mixed_precision = True
    grad_clip = 1.0
    optimiser = "AdamW"
    learning_rate = 1.5e-3
    weight_decay = 0.01
    lr_warmup_steps = 100_000
    lr_decay_step = 10_000_000
    n_epochs = 10
    use_weighted_loss = True
    early_stopping_rounds = 10

    max_train_updates = 1_000
    eval_every_n_updates = 200

    num_workers_ = 2
    save_weights_only_ = True
    profile_ = False

    # local_data contains config paths and keys that are not commited to git
    dataset_path_ = local_data.dataset_path
    model_path_ = local_data.model_path
    history_save_path_ = local_data.history_save_path
    tensorboard_dir_ = local_data.tensorboard_dir
    tensorboard_dir_ = os.path.join(tensorboard_dir_, expt_name)
    log_wandb_ = True
    wandb_watch_ = False

    @staticmethod
    def export_dict():
        return {k: v for k, v in Config.__dict__.items() if not k.endswith("_")}