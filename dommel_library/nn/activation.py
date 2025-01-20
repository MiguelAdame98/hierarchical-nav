import torch.nn as nn
import torch.nn.functional as F

from dommel_library.nn import module_factory


def get_activation(activation="Activation", **kwargs):
    if not activation:
        # no activation
        activation = "Identity"
    if isinstance(activation, str):
        # Search registered backends for the activation and create the module
        return module_factory.get_module_from_dict(
            {"type": activation, "args": kwargs}
        )
    else:
        # uniform way to also handle passing the objects directly
        return activation


class Activation(nn.Module):

    def __init__(self, **kwargs):
        nn.Module.__init__(self)

    def forward(self, x):
        return F.leaky_relu(x, negative_slope=0.02)

    def __repr__(self):
        return "LeakyReLU(negative_slope=0.02)"
