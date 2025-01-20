import torch
from torch import nn
from torch.nn import functional as F

from dommel_library.distributions import MultivariateNormal
from dommel_library.datastructs import TensorDict
from dommel_library.nn import MLP


class VariationalMLP(MLP):
    """ Variational Multi Layer Perceptron
    A fully connected neural network outputting mu and sigma for each output
    """

    def __init__(
        self,
        num_inputs,
        num_outputs,
        hidden_layers=None,
        sigma_scale=1.0,
        sigma_offset=0.0,
        activation="Activation",
        **kwargs
    ):
        """Creates a MLP with mu and sigma outputs
        :param num_inputs: Amount of inputs, single int.
        :param num_outputs: The size of the output _space_, single int.
        :param hidden_layers: Amount of hidden neurons, list. Optional.
        :param sigma_scale: How much the sigma is to be rescaled.
        :param sigma_offset: How much the sigma is to be shifted.
        :param activation: The activation function for this module.
        :param kwargs: Optional arguments for MLP.
        """
        MLP.__init__(
            self,
            num_inputs,
            2 * num_outputs,
            hidden_layers,
            activation,
            **kwargs
        )
        self.sigma_scale = sigma_scale
        self.sigma_offset = sigma_offset

    def forward(self, x):
        x = MLP.forward(self, x)
        mu = x[:, : int(self.num_outputs / 2)]
        sigma = x[:, int(self.num_outputs / 2):]
        sigma = (
            self.sigma_scale * (F.softplus(sigma) + 1e-6) + self.sigma_offset
        )
        return MultivariateNormal(mu, sigma)


class VariationalLayer(nn.Module):
    """Module that takes a vector and outputs 2 vectors: mu and sigma"""

    def __init__(
        self,
        num_inputs,
        num_outputs,
        sigma_scale=1.0,
        sigma_offset=0.0,
        **kwargs
    ):
        """
        Creates a single linear layer with variational output
        :param num_inputs: The amount of inputs. Single int.
        :param num_outputs: The amount of outputs. Single int.
        :param sigma_scale: How much sigma must be rescaled.
        :param sigma_offset: How much simga must be shifted after scaling.
        """
        nn.Module.__init__(self)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.sigma_scale = sigma_scale
        self.sigma_offset = sigma_offset
        self.linear_mu = nn.Linear(num_inputs, num_outputs)
        self.linear_sigma = nn.Linear(num_inputs, num_outputs)

    def forward(self, y):
        mu = self.linear_mu(y)
        sigma = self.linear_sigma(y)
        sigma = (
            self.sigma_scale * (F.softplus(sigma) + 1e-6) + self.sigma_offset
        )
        return MultivariateNormal(mu, sigma)


class VariationalGRU(nn.Module):
    """Variational wrapper around a GRU block"""

    def __init__(self, num_inputs, num_outputs, hidden_layers=None, **kwargs):
        """Initialize the module
        :param num_inputs: Length of the input vector.
        :param num_outputs: Desired output length.
        :param hidden_layers: Optional list of hidden neurons for the cells.
        """
        nn.Module.__init__(self)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_layers = hidden_layers

        self.cells = nn.ModuleList()
        if not hidden_layers:
            hidden_layers = []
            self.output = VariationalLayer(num_inputs, num_outputs)
        else:
            self.output = VariationalLayer(hidden_layers[-1], num_outputs)
            self.cells.append(nn.GRUCell(self.num_inputs, hidden_layers[0]))

        for i in range(0, len(hidden_layers) - 1):
            self.cells.append(
                nn.GRUCell(hidden_layers[i], hidden_layers[i + 1])
            )

    def forward(self, x, hidden):
        next_hidden = {}
        for i in range(len(self.cells)):
            gru = self.cells[i]
            # TODO allow to namespace this to avoid collisions?!
            key = "gru" + str(i)
            if hidden is None:
                h = torch.zeros(x.shape[0], self.hidden_layers[i]).to(x.device)
            else:
                h = hidden[key]
            x = gru(x, h)
            next_hidden[key] = x

        dist = self.output(x)
        return dist, next_hidden


class VariationalLSTM(nn.Module):
    """Variational version of a LSTM block"""

    def __init__(self, num_inputs, num_outputs, hidden_layers=None, **kwargs):
        """Initializes the Module
        :param num_inputs: Input length.
        :param num_outputs: Desired output length.
        :param hidden_layers: Optional list of hidden layer neurons.
        """
        nn.Module.__init__(self)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_layers = hidden_layers

        self.cells = nn.ModuleList()
        if not hidden_layers:
            hidden_layers = []
            self.output = VariationalLayer(num_inputs, num_outputs)
        else:
            self.output = VariationalLayer(hidden_layers[-1], num_outputs)
            self.cells.append(nn.LSTMCell(self.num_inputs, hidden_layers[0]))

        for i in range(0, len(hidden_layers) - 1):
            self.cells.append(
                nn.LSTMCell(hidden_layers[i], hidden_layers[i + 1])
            )

    def forward(self, x, hidden):
        next_hidden = TensorDict({})
        for i in range(len(self.cells)):
            lstm = self.cells[i]
            # TODO allow to namespace this to avoid collisions?!
            hkey = "hidden" + str(i)
            ckey = "cell" + str(i)
            if hidden is None:
                h = torch.zeros(x.shape[0], self.hidden_layers[i]).to(x.device)
                c = torch.zeros(x.shape[0], self.hidden_layers[i]).to(x.device)
            else:
                h = hidden[hkey]
                c = hidden[ckey]

            x, c = lstm(x, (h, c))

            next_hidden[hkey] = x
            next_hidden[ckey] = c

        dist = self.output(x)
        return dist, next_hidden
