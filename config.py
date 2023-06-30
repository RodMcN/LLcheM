seed = 42

expt_name = "test"

# Model
embed_dim=256
num_attn_heads=4
num_encoder_layers=3

# Training
n_gpu = 2
effective_batch_size = 1024 # actual batch size (per gpu) = effective_batch_size / num_gpus / accumulation_steps
accumulation_steps = 1 
eval_batch_size = 1024
mixed_precision = True
grad_clip = 1.0
optimiser = "AdamW"
learning_rate = 1.5e-3
weight_decay = 0.01
lr_warmup_steps = 100_000
lr_decay_step = 10_000_000
n_epochs = 10
use_weighted_loss = True
num_workers = 32
early_stopping_rounds = 10

max_train_updates = 100_000
eval_every_n_updates = 10_000

save_weights_only = True

# local_data contains config paths and keys that are not commited to git
import local_data
dataset_path = local_data.dataset_path
model_path = local_data.model_path
tensorboard_dir = local_data.tensorboard_dir
log_wandb = False