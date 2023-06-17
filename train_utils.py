import torch
from inspect import getmembers
import math as maths
import torch.nn.functional as F


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
