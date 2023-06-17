seed = 42

# Model
embed_dim=256
num_attn_heads=4
num_encoder_layers=3

n_gpu = 2
effective_batch_size = 128 # actual batch size (per gpu) = effective_batch_size / num_gpus / accumulation_steps
accumulation_steps = 1 
eval_batch_size = 128 # eval is currently done on single GPU
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

max_train_updates = 2234
eval_every_n_updates = 1000

import local_data
dataset_path = local_data.dataset_path
model_path = local_data.model_path
save_weights_only = True
