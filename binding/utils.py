import torch
from inspect import getmembers, isclass

def get_module(name, *args, **kwargs):
    """
    returns a class from the torch.nn.modules.activation module by name
    eg get_activation("LeakyReLU") will return torch.nn.modules.activation.LeakyReLU
    """
    nn_classes = getmembers(torch.nn.modules, isclass)
    for c in nn_classes:
        if c[0].lower() == name.lower():
            return c[1](*args, **kwargs)
    raise ValueError(f"module '{name}' not found")
