""" Module to aggregate our own neural modules """
import torch
from torch import nn

from dommel_library.nn.activation import get_activation


class View(nn.Module):
    """nn.Module for the PyTorch view method"""

    def __init__(self, shape, **kwargs):
        """
        :param shape: New shape to reshape the data into, should be a tuple
        """
        nn.Module.__init__(self)
        self._shape = shape

    def forward(self, x):
        return x.view(x.data.size(0), *self._shape)


class Reshape(nn.Module):
    """nn.Module for the PyTorch reshape method"""

    def __init__(self, shape, **kwargs):
        """
        :param shape: New shape to reshape the data into, should be a tuple
        Unlike view, this one uses reshape. Might be required in cases the
        input is not contiguous, for example after broadcasting.
        """
        nn.Module.__init__(self)
        self._shape = shape

    def forward(self, x):
        x = x.reshape(x.data.size(0), *self._shape)
        return x


class MLP(nn.Module):
    """Multi Layer Perceptron Module"""

    def __init__(
        self,
        num_inputs,
        num_outputs,
        hidden_layers=None,
        activation="Activation",
        **kwargs
    ):
        """Initializes a multilayer perceptron. Note that the final layer
        DOES NOT have an activation, if you have subsequent computations,
        you should add these yourself!
        :param num_inputs: Single int representing the amount of inputs.
        :param num_outputs: Single int representing the amount of outputs.
        :param hidden_layers: Optional list indicating the amount of
        intermediate neurons
        :param activation: The activation used in the pipeline
        :param kwargs: Optional key word args for the activation function
        """
        nn.Module.__init__(self)

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_layers = hidden_layers
       
        self.layers = nn.ModuleList()
        if hidden_layers:
            self.layers.append(nn.Linear(self.num_inputs, hidden_layers[0]))
            self.layers.append(get_activation(activation, **kwargs))
        else:
            self.layers.append(nn.Linear(self.num_inputs, self.num_outputs))
        if hidden_layers:
            for i in range(0, len(hidden_layers) - 1):
                self.layers.append(
                    nn.Linear(hidden_layers[i], hidden_layers[i + 1])
                )
                self.layers.append(get_activation(activation, **kwargs))
            self.layers.append(nn.Linear(hidden_layers[-1], num_outputs))

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x



class Sample(nn.Module):
    def __init__(self, **kwargs):
        """ Initialize a Sample block, which will just execute the wrapped
        distribution objects sample method. This allows the inclusion of the
        sample step in our yaml builder.
        """
        nn.Module.__init__(self)

    def forward(self, dist):
        return dist.sample()


class Cat(nn.Module):
    def __init__(self, dim=1):
        """Wrapper around torch.cat
        :param dim: The dimension on which to cat. 0 is batch dim, 1 is first
        data dim.
        """
        nn.Module.__init__(self)
        self.dim = dim

    def forward(self, *x):
        return torch.cat(x, dim=self.dim)


class Sum(nn.Module):
    def __init__(self, dim=1):
        """Wrapper around torch.sum
        :param dim: The dimension on which to cat. 0 is batch dim, 1 is first
        data dim.
        """
        nn.Module.__init__(self)
        self.dim = dim

    def forward(self, *x):
        return torch.stack(x, dim=self.dim).sum(dim=self.dim)


class Broadcast(nn.Module):
    def __init__(self, shape, position_encode=False):
        """Expand a vector into spatial dimensions
           :param shape: Spatial dimensions in H x W
           :param position_encode: Add extra 2 channels encoding
           spatial position
        """
        nn.Module.__init__(self)
        self.shape = shape
        self.position_encode = position_encode

    def forward(self, x):
        s = x.shape
        expanded = x.unsqueeze(-1).unsqueeze(-1).expand(*s, *self.shape)

        if not self.position_encode:
            return expanded
        else:
            hs = torch.linspace(-1, 1, steps=self.shape[0])
            ws = torch.linspace(-1, 1, steps=self.shape[1])
            h, w = torch.meshgrid(hs, ws)
            positions = torch.stack([h, w])
            for i in reversed(range(len(s) - 1)):
                positions.unsqueeze(0)
                positions = positions.expand(s[i], *positions.shape)
            positions = positions.to(expanded.device)

            return torch.cat([expanded, positions], dim=-3)
