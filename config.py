seed = 42

# Model
embed_dim=512
num_attn_heads=8
num_encoder_layers=8

n_gpu = 1
batch_size = 128
accumulation_steps = 8
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

import local_data
dataset_path = local_data.dataset_path
model_path = local_data.model_path
save_weights_only = True
